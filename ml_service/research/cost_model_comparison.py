#!/usr/bin/env python3
"""
Cost Model & Baseline Comparison
=================================
Applies a realistic 3-component transaction cost model to:
  1. V6 dual-ensemble (40/60 base/sector blend) — from walk-forward predictions
  2. 12-1 momentum baseline (top decile, monthly rebalance)
  3. Sector-neutral 12-1 momentum (top stock per GICS sector, monthly)
  4. Low-vol momentum (top quintile momentum within bottom-half volatility)

Cost model:
  - Half-spread: 2 bps for S&P 500 stocks, 5 bps for ETFs
  - Market impact: k * (order_size / ADV_20d)^0.5 * intraday_vol
  - Per-trade commission: $0 (IBKR Lite placeholder)

Reports gross & net CAGR/Sharpe/DD, turnover, avg holding period.

Run:
    cd ml_service && python3 research/cost_model_comparison.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WF_DIR = DATA_DIR / "walkforward"

INITIAL_CASH = 100_000.0
TOP_N = 5
HOLD_DAYS = 10
POSITION_PCT = 1.0 / TOP_N  # equal-weight top-5

# ── Cost model parameters ────────────────────────────────────────────────────
HALF_SPREAD_STOCK_BPS = 2.0   # S&P 500 stocks
HALF_SPREAD_ETF_BPS = 5.0     # ETFs
MARKET_IMPACT_K = 0.1         # sqrt impact coefficient
COMMISSION_PER_TRADE = 0.0    # $0 for IBKR Lite

ETF_SYMBOLS = {
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY",
    "XLB", "XLU", "XLRE", "XLC", "EWZ", "EWJ", "FXI", "INDA", "EFA", "EEM",
    "VGK", "VWO", "IEFA", "GLD", "SLV", "USO", "DBC", "CPER", "TLT", "IEF",
    "SHY", "HYG", "LQD", "VIXY", "UUP",
}

# GICS sector mapping via sector ETFs (approximate, for sector-neutral baseline)
SECTOR_ETF_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care",
    "XLE": "Energy", "XLI": "Industrials", "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Communication Services",
}


def log(msg: str):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Cost Model
# ══════════════════════════════════════════════════════════════════════════════

def compute_trade_cost(symbol: str, trade_value: float, adv_20d: float,
                       intraday_vol: float) -> float:
    """
    3-component cost for a single trade (one-way: buy or sell).

    Returns dollar cost.
    """
    # 1. Half-spread
    spread_bps = HALF_SPREAD_ETF_BPS if symbol in ETF_SYMBOLS else HALF_SPREAD_STOCK_BPS
    spread_cost = trade_value * spread_bps / 10_000

    # 2. Market impact: k * sqrt(order_size / ADV) * intraday_vol * trade_value
    if adv_20d > 0 and intraday_vol > 0:
        participation = trade_value / adv_20d
        impact_cost = MARKET_IMPACT_K * np.sqrt(participation) * intraday_vol * trade_value
    else:
        impact_cost = 0.0

    # 3. Commission
    commission = COMMISSION_PER_TRADE

    return spread_cost + impact_cost + commission


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_price_data(symbols: list, start: str, end: str) -> pd.DataFrame:
    """Load close prices via Massive with disk cache."""
    from massive_data_provider import MassiveDataProvider

    cache_file = DATA_DIR / "cache_prices_costmodel.parquet"
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        cached.index = pd.to_datetime(cached.index)
        missing = set(symbols) - set(cached.columns)
        if not missing and cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5):
            return cached[list(set(symbols) & set(cached.columns))]

    log(f"  Downloading prices for {len(symbols)} symbols from Massive ...")
    provider = MassiveDataProvider(validate_vs_yfinance=False)
    warmup = (pd.Timestamp(end) - pd.Timestamp(start)).days + 30
    bars = provider.fetch_bars_batch(list(set(symbols)), warmup_days=warmup)

    close_frames = {}
    for sym, df in bars.items():
        if len(df) > 0:
            close_frames[sym] = df["close"]

    if not close_frames:
        raise RuntimeError("Failed to download any price data from Massive")

    prices = pd.DataFrame(close_frames)
    prices.index = pd.to_datetime(prices.index)
    prices.to_parquet(cache_file)
    return prices


def load_volume_data(symbols: list, start: str, end: str) -> pd.DataFrame:
    """Load volume data via Massive with disk cache."""
    from massive_data_provider import MassiveDataProvider

    cache_file = DATA_DIR / "cache_volume_costmodel.parquet"
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        cached.index = pd.to_datetime(cached.index)
        missing = set(symbols) - set(cached.columns)
        if not missing and cached.index.min() <= pd.Timestamp(start):
            return cached[list(set(symbols) & set(cached.columns))]

    log(f"  Downloading volume for {len(symbols)} symbols from Massive ...")
    provider = MassiveDataProvider(validate_vs_yfinance=False)
    warmup = (pd.Timestamp(end) - pd.Timestamp(start)).days + 30
    bars = provider.fetch_bars_batch(list(set(symbols)), warmup_days=warmup)

    vol_frames = {}
    for sym, df in bars.items():
        if len(df) > 0:
            vol_frames[sym] = df["volume"]

    if not vol_frames:
        raise RuntimeError("Failed to download volume data from Massive")

    volume = pd.DataFrame(vol_frames)
    volume.index = pd.to_datetime(volume.index)
    volume.to_parquet(cache_file)
    return volume


def get_sector_map(symbols: list) -> dict:
    """Map symbols to GICS sectors via Massive (Polygon). Cache to disk."""
    import json
    from massive_data_provider import MassiveDataProvider

    provider = MassiveDataProvider(validate_vs_yfinance=False)
    return provider.build_sector_map(symbols)


# ══════════════════════════════════════════════════════════════════════════════
#  Position-Level Simulator (shared across all strategies)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_strategy(trade_schedule: list, prices: pd.DataFrame,
                      adv_20d: pd.DataFrame, intraday_vol: pd.DataFrame,
                      label: str) -> dict:
    """
    Run a position-level backtest with the cost model.

    Args:
        trade_schedule: list of (date, [symbols]) tuples — on each date, the
            portfolio is rebalanced to hold exactly those symbols equal-weight.
        prices: DataFrame of close prices (index=dates, columns=symbols)
        adv_20d: DataFrame of 20-day average dollar volume
        intraday_vol: DataFrame of 20-day realized daily vol (std of returns)
        label: strategy name for reporting

    Returns:
        dict with gross/net metrics, turnover, holding period stats
    """
    trading_dates = sorted(prices.index)
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    # Build trade schedule lookup
    schedule = {}
    for date, symbols in trade_schedule:
        if date in date_to_idx:
            schedule[date] = symbols

    cash_gross = INITIAL_CASH
    cash_net = INITIAL_CASH
    holdings_gross = {}  # symbol -> {"shares": float, "entry_price": float, "entry_date": date}
    holdings_net = {}
    gross_values = []
    net_values = []
    total_cost_dollars = 0.0
    total_turnover_dollars = 0.0
    n_trades = 0
    hold_durations = []

    for date in trading_dates:
        target_symbols = schedule.get(date, None)
        day_prices = prices.loc[date] if date in prices.index else pd.Series(dtype=float)

        # Mark-to-market current holdings
        def portfolio_value(cash, holdings):
            val = cash
            for sym, h in holdings.items():
                px = day_prices.get(sym, h["entry_price"])
                if pd.isna(px):
                    px = h["entry_price"]
                val += h["shares"] * px
            return val

        # Rebalance if we have a target for today
        if target_symbols is not None:
            target_set = set(target_symbols)
            port_val_gross = portfolio_value(cash_gross, holdings_gross)
            port_val_net = portfolio_value(cash_net, holdings_net)

            # Sell positions not in target
            for sym in list(holdings_gross.keys()):
                if sym not in target_set:
                    px = day_prices.get(sym)
                    if px is None or pd.isna(px):
                        continue
                    h = holdings_gross.pop(sym)
                    proceeds = h["shares"] * px
                    cash_gross += proceeds
                    # Track holding duration
                    if h["entry_date"] in date_to_idx:
                        dur = (date - h["entry_date"]).days
                        hold_durations.append(dur)

            for sym in list(holdings_net.keys()):
                if sym not in target_set:
                    px = day_prices.get(sym)
                    if px is None or pd.isna(px):
                        continue
                    h = holdings_net.pop(sym)
                    proceeds = h["shares"] * px
                    # Sell cost
                    adv_val = adv_20d.loc[date].get(sym, 0) if date in adv_20d.index else 0
                    iv = intraday_vol.loc[date].get(sym, 0.02) if date in intraday_vol.index else 0.02
                    if pd.isna(adv_val): adv_val = 0
                    if pd.isna(iv): iv = 0.02
                    sell_cost = compute_trade_cost(sym, proceeds, adv_val, iv)
                    cash_net += proceeds - sell_cost
                    total_cost_dollars += sell_cost
                    total_turnover_dollars += proceeds
                    n_trades += 1

            # Buy new positions
            port_val_gross = portfolio_value(cash_gross, holdings_gross)
            port_val_net = portfolio_value(cash_net, holdings_net)

            new_buys = [s for s in target_symbols if s not in holdings_gross]
            if new_buys:
                target_weight = 1.0 / max(len(target_symbols), 1)

                for sym in new_buys:
                    px = day_prices.get(sym)
                    if px is None or pd.isna(px) or px <= 0:
                        continue

                    # Gross portfolio (no costs)
                    alloc_gross = port_val_gross * target_weight
                    shares_gross = alloc_gross / px
                    if cash_gross >= alloc_gross:
                        cash_gross -= alloc_gross
                        holdings_gross[sym] = {
                            "shares": shares_gross, "entry_price": px,
                            "entry_date": date,
                        }

                    # Net portfolio (with costs deducted from allocation)
                    alloc_net = port_val_net * target_weight
                    adv_val = adv_20d.loc[date].get(sym, 0) if date in adv_20d.index else 0
                    iv = intraday_vol.loc[date].get(sym, 0.02) if date in intraday_vol.index else 0.02
                    if pd.isna(adv_val): adv_val = 0
                    if pd.isna(iv): iv = 0.02
                    buy_cost = compute_trade_cost(sym, alloc_net, adv_val, iv)
                    invested_net = alloc_net - buy_cost  # costs come out of allocation
                    if cash_net >= alloc_net and invested_net > 0:
                        shares_net = invested_net / px
                        cash_net -= alloc_net
                        holdings_net[sym] = {
                            "shares": shares_net, "entry_price": px,
                            "entry_date": date,
                        }
                        total_cost_dollars += buy_cost
                        total_turnover_dollars += alloc_net
                        n_trades += 1

        # Record daily portfolio value
        gross_values.append((date, portfolio_value(cash_gross, holdings_gross)))
        net_values.append((date, portfolio_value(cash_net, holdings_net)))

    # Build return series
    gross_series = pd.Series(
        [v for _, v in gross_values],
        index=pd.DatetimeIndex([d for d, _ in gross_values]),
    )
    net_series = pd.Series(
        [v for _, v in net_values],
        index=pd.DatetimeIndex([d for d, _ in net_values]),
    )

    years = (gross_series.index[-1] - gross_series.index[0]).days / 365.25
    if years <= 0:
        years = 1.0

    def _metrics(vals, tag):
        daily_ret = vals.pct_change().dropna()
        cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        down = daily_ret[daily_ret < 0]
        sortino = daily_ret.mean() / down.std() * np.sqrt(252) if len(down) > 0 else np.nan
        peak = vals.cummax()
        dd = ((vals - peak) / peak).min()
        return {
            f"cagr_{tag}": cagr, f"sharpe_{tag}": sharpe,
            f"sortino_{tag}": sortino, f"max_dd_{tag}": dd,
        }

    metrics = {**_metrics(gross_series, "gross"), **_metrics(net_series, "net")}

    # Turnover: annualized (relative to average portfolio value, not initial cash)
    avg_portfolio = np.mean([v for _, v in net_values])
    ann_turnover = total_turnover_dollars / (avg_portfolio * years) if (years > 0 and avg_portfolio > 0) else 0
    avg_hold = np.mean(hold_durations) if hold_durations else 0

    metrics.update({
        "label": label,
        "n_trades": n_trades,
        "total_cost_dollars": total_cost_dollars,
        "cost_per_trade_bps": (total_cost_dollars / total_turnover_dollars * 10000) if total_turnover_dollars > 0 else 0,
        "annualized_turnover": ann_turnover,
        "avg_hold_days": avg_hold,
        "years": years,
        "gross_final": gross_series.iloc[-1],
        "net_final": net_series.iloc[-1],
    })

    return metrics, gross_series, net_series


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: V6 (from walk-forward predictions)
# ══════════════════════════════════════════════════════════════════════════════

def build_v6_schedule(preds: pd.DataFrame) -> list:
    """
    Build trade schedule from V6 walk-forward predictions.
    Every HOLD_DAYS trading days, select top-5 SP500 stocks by prob_ensemble.
    """
    preds = preds[preds["in_sp500"] == True].copy()
    dates = sorted(preds["date"].unique())

    schedule = []
    for i, date in enumerate(dates):
        if i % HOLD_DAYS != 0:
            continue
        day_preds = preds[preds["date"] == date].nlargest(TOP_N, "prob_ensemble")
        symbols = day_preds["symbol"].tolist()
        if symbols:
            schedule.append((date, symbols))

    return schedule


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: 12-1 Momentum
# ══════════════════════════════════════════════════════════════════════════════

def build_momentum_12_1_schedule(prices: pd.DataFrame,
                                 sp500_symbols: set,
                                 top_frac: float = 0.10) -> list:
    """
    Monthly rebalance. Rank by 11-month return skipping the most recent month.
    Hold equal-weight top decile.
    """
    # Monthly dates (last trading day of each month)
    monthly_dates = prices.resample("ME").last().index

    # Returns
    ret_12m = prices.pct_change(252)   # ~12 months
    ret_1m = prices.pct_change(21)     # ~1 month
    ret_12_1 = ret_12m - ret_1m        # skip recent month (approximate)

    schedule = []
    for date in monthly_dates:
        if date not in ret_12_1.index:
            continue
        day_ret = ret_12_1.loc[date].dropna()
        # Filter to SP500 only
        valid = day_ret[day_ret.index.isin(sp500_symbols)]
        if len(valid) < 20:
            continue
        n_top = max(1, int(len(valid) * top_frac))
        top = valid.nlargest(n_top).index.tolist()
        schedule.append((date, top))

    return schedule


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: Sector-Neutral 12-1 Momentum
# ══════════════════════════════════════════════════════════════════════════════

def build_sector_neutral_momentum_schedule(prices: pd.DataFrame,
                                            sp500_symbols: set,
                                            sector_map: dict) -> list:
    """
    Monthly rebalance. Top stock per GICS sector by 12-1 momentum, equal-weight.
    """
    monthly_dates = prices.resample("ME").last().index
    ret_12m = prices.pct_change(252)
    ret_1m = prices.pct_change(21)
    ret_12_1 = ret_12m - ret_1m

    schedule = []
    for date in monthly_dates:
        if date not in ret_12_1.index:
            continue
        day_ret = ret_12_1.loc[date].dropna()
        valid = day_ret[day_ret.index.isin(sp500_symbols)]

        # Group by sector, pick top per sector
        picks = []
        sector_groups = {}
        for sym in valid.index:
            sec = sector_map.get(sym, "Unknown")
            if sec == "Unknown":
                continue
            if sec not in sector_groups:
                sector_groups[sec] = []
            sector_groups[sec].append((sym, valid[sym]))

        for sec, entries in sector_groups.items():
            best = max(entries, key=lambda x: x[1])
            picks.append(best[0])

        if picks:
            schedule.append((date, picks))

    return schedule


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: Low-Vol Momentum
# ══════════════════════════════════════════════════════════════════════════════

def build_lowvol_momentum_schedule(prices: pd.DataFrame,
                                    sp500_symbols: set) -> list:
    """
    Monthly rebalance. Bottom half by 60-day vol, then top quintile by momentum.
    """
    monthly_dates = prices.resample("ME").last().index
    ret_12m = prices.pct_change(252)
    ret_1m = prices.pct_change(21)
    ret_12_1 = ret_12m - ret_1m
    vol_60d = prices.pct_change().rolling(60).std()

    schedule = []
    for date in monthly_dates:
        if date not in ret_12_1.index or date not in vol_60d.index:
            continue
        day_ret = ret_12_1.loc[date].dropna()
        day_vol = vol_60d.loc[date].dropna()

        valid_syms = set(day_ret.index) & set(day_vol.index) & sp500_symbols
        if len(valid_syms) < 40:
            continue

        valid_ret = day_ret[list(valid_syms)]
        valid_vol = day_vol[list(valid_syms)]

        # Bottom half by vol
        vol_median = valid_vol.median()
        low_vol = valid_vol[valid_vol <= vol_median].index

        # Top quintile by momentum within low-vol
        low_vol_ret = valid_ret[low_vol]
        n_top = max(1, int(len(low_vol_ret) * 0.20))
        top = low_vol_ret.nlargest(n_top).index.tolist()

        if top:
            schedule.append((date, top))

    return schedule


# ══════════════════════════════════════════════════════════════════════════════
#  SPY Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def build_spy_schedule(prices: pd.DataFrame) -> list:
    """Buy and hold SPY."""
    first_date = prices.index[0]
    return [(first_date, ["SPY"])]


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()

    log("=" * 75)
    log("  COST MODEL & BASELINE COMPARISON")
    log("  V6 vs 12-1 Momentum vs Sector-Neutral Momentum vs Low-Vol Momentum")
    log("=" * 75)

    # ── Load walk-forward predictions ──
    log("\n1. Loading walk-forward predictions ...")
    wf_file = WF_DIR / "predictions_walkforward_all.parquet"
    if not wf_file.exists():
        sys.exit(f"ERROR: {wf_file} not found — run walk_forward_validation.py first")
    preds = pd.read_parquet(wf_file)
    preds["date"] = pd.to_datetime(preds["date"])
    log(f"   {len(preds):,} rows, {preds['date'].min().date()} -> {preds['date'].max().date()}")

    # ── Get universe ──
    all_symbols = sorted(preds["symbol"].unique().tolist())
    sp500_set = set(all_symbols)
    log(f"   {len(sp500_set)} SP500 symbols")

    # ── Load price & volume data ──
    log("\n2. Loading price and volume data ...")
    start_date = "2013-06-01"  # need lookback for 12-month momentum
    end_date = "2026-01-15"
    extra_syms = ["SPY"]  # ensure SPY is included
    all_download = list(set(all_symbols + extra_syms))

    prices = load_price_data(all_download, start_date, end_date)
    volume = load_volume_data(all_download, start_date, end_date)
    log(f"   Prices: {prices.shape}, Volume: {volume.shape}")

    # ── Compute ADV and intraday vol ──
    log("\n3. Computing ADV-20d and intraday volatility ...")
    # ADV in dollar terms
    adv_20d = (prices * volume).rolling(20, min_periods=10).mean()
    # Intraday vol approximated by 20-day std of daily returns
    intraday_vol = prices.pct_change().rolling(20, min_periods=10).std()

    # ── Get sector mapping ──
    log("\n4. Fetching sector mapping for sector-neutral baseline ...")
    sector_map = get_sector_map(all_symbols)
    sectors_found = set(sector_map.values()) - {"Unknown"}
    log(f"   {len(sectors_found)} sectors mapped")

    # ── Build trade schedules ──
    log("\n5. Building trade schedules ...")

    v6_schedule = build_v6_schedule(preds)
    log(f"   V6: {len(v6_schedule)} rebalance dates")

    mom_12_1_schedule = build_momentum_12_1_schedule(prices, sp500_set)
    log(f"   12-1 Momentum: {len(mom_12_1_schedule)} rebalance dates")

    sec_neutral_schedule = build_sector_neutral_momentum_schedule(prices, sp500_set, sector_map)
    log(f"   Sector-neutral 12-1: {len(sec_neutral_schedule)} rebalance dates")

    lowvol_mom_schedule = build_lowvol_momentum_schedule(prices, sp500_set)
    log(f"   Low-vol momentum: {len(lowvol_mom_schedule)} rebalance dates")

    # ── Trim price data to walk-forward period ──
    wf_start = preds["date"].min()
    wf_end = preds["date"].max()
    prices_wf = prices[(prices.index >= wf_start) & (prices.index <= wf_end)]
    adv_wf = adv_20d[(adv_20d.index >= wf_start) & (adv_20d.index <= wf_end)]
    ivol_wf = intraday_vol[(intraday_vol.index >= wf_start) & (intraday_vol.index <= wf_end)]

    # Also filter schedules to WF period
    def _filter_schedule(sched):
        return [(d, s) for d, s in sched if wf_start <= d <= wf_end]

    mom_12_1_schedule = _filter_schedule(mom_12_1_schedule)
    sec_neutral_schedule = _filter_schedule(sec_neutral_schedule)
    lowvol_mom_schedule = _filter_schedule(lowvol_mom_schedule)
    spy_schedule = build_spy_schedule(prices_wf)

    # ── Run simulations ──
    log("\n6. Running simulations with cost model ...")

    strategies = [
        ("V6 Dual-Ensemble (40/60)", v6_schedule),
        ("12-1 Momentum (top decile)", mom_12_1_schedule),
        ("Sector-Neutral 12-1 Momentum", sec_neutral_schedule),
        ("Low-Vol Momentum", lowvol_mom_schedule),
        ("SPY Buy & Hold", spy_schedule),
    ]

    results = []
    for name, schedule in strategies:
        log(f"\n   Running: {name} ...")
        metrics, gross_s, net_s = simulate_strategy(
            schedule, prices_wf, adv_wf, ivol_wf, name)
        results.append(metrics)
        log(f"     Gross: CAGR={metrics['cagr_gross']:+.1%}  Sharpe={metrics['sharpe_gross']:.2f}  DD={metrics['max_dd_gross']:.1%}")
        log(f"     Net:   CAGR={metrics['cagr_net']:+.1%}  Sharpe={metrics['sharpe_net']:.2f}  DD={metrics['max_dd_net']:.1%}")
        log(f"     Trades={metrics['n_trades']}  Avg hold={metrics['avg_hold_days']:.1f}d  "
            f"Turnover={metrics['annualized_turnover']:.1f}x  "
            f"Cost/trade={metrics['cost_per_trade_bps']:.1f}bps  "
            f"Total costs=${metrics['total_cost_dollars']:,.0f}")

    # ── Summary Table ──
    log(f"\n{'='*75}")
    log("  RESULTS SUMMARY")
    log(f"{'='*75}")

    header = (f"  {'Strategy':<35} {'CAGR(G)':>8} {'CAGR(N)':>8} "
              f"{'Shp(G)':>7} {'Shp(N)':>7} {'DD(G)':>7} {'DD(N)':>7} "
              f"{'Turn':>5} {'Hold':>5} {'Cost':>6}")
    log(header)
    log(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*6}")

    for m in results:
        log(f"  {m['label']:<35} "
            f"{m['cagr_gross']:>+7.1%} {m['cagr_net']:>+7.1%} "
            f"{m['sharpe_gross']:>7.2f} {m['sharpe_net']:>7.2f} "
            f"{m['max_dd_gross']:>6.1%} {m['max_dd_net']:>6.1%} "
            f"{m['annualized_turnover']:>5.1f} {m['avg_hold_days']:>5.1f} "
            f"{m['cost_per_trade_bps']:>5.1f}bp")

    # ── Edge Analysis ──
    v6 = results[0]
    log(f"\n{'='*75}")
    log("  EDGE ANALYSIS")
    log(f"{'='*75}")

    for baseline in results[1:]:
        net_edge = v6["sharpe_net"] - baseline["sharpe_net"]
        gross_edge = v6["sharpe_gross"] - baseline["sharpe_gross"]
        log(f"  V6 vs {baseline['label']}:")
        log(f"    Gross Sharpe edge: {gross_edge:+.2f}")
        log(f"    Net Sharpe edge:   {net_edge:+.2f}")
        log(f"    V6 net CAGR advantage: {v6['cagr_net'] - baseline['cagr_net']:+.1%}")

    # ── Cost Impact ──
    log(f"\n{'='*75}")
    log("  COST IMPACT")
    log(f"{'='*75}")
    for m in results:
        sharpe_drag = m["sharpe_gross"] - m["sharpe_net"]
        cagr_drag = m["cagr_gross"] - m["cagr_net"]
        log(f"  {m['label']:<35}  Sharpe drag: {sharpe_drag:.2f}  CAGR drag: {cagr_drag:.1%}  "
            f"Total costs: ${m['total_cost_dollars']:,.0f}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
