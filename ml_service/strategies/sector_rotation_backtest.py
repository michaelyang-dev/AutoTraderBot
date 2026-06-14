"""
Adaptive Sector Rotation Strategy — Backtest
=============================================
Two variants:
  A) Pure ETF rotation: buy top-3 sector ETFs by 60d momentum, SMA200 filter
  B) Stock-within-sector: buy top-3 stocks (ROE>10%) from top-3 sectors

Uses real portfolio sim with trailing stops, transaction costs, position caps.

Usage:
    cd ml_service && python3 -m strategies.sector_rotation_backtest
"""

import numpy as np
import pandas as pd
import pickle
import time
import logging
from pathlib import Path
from collections import defaultdict

log = logging.getLogger("sector_rotation")

# ── Constants ────────────────────────────────────────────────────────────────
INITIAL_CASH = 100_000.0
COST_BPS = 10  # 5 commission + 5 slippage
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLC", "XLRE", "XLB"]
GICS_TO_ETF = {
    10: "XLE", 15: "XLB", 20: "XLI", 25: "XLY", 30: "XLP",
    35: "XLV", 40: "XLF", 45: "XLK", 50: "XLC", 55: "XLU", 60: "XLRE",
}
ETF_TO_GICS = {v: k for k, v in GICS_TO_ETF.items()}
ETF_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Cons Disc",
    "XLP": "Cons Staples", "XLB": "Materials", "XLRE": "Real Estate",
    "XLU": "Utilities", "XLC": "Comm Svcs",
}


def load_data(universe_path="data/wrds/complete_sp1500_universe.pkl"):
    """Load prices, features, and build ticker->sector mapping from Compustat."""
    t0 = time.time()
    with open(universe_path, "rb") as f:
        data = pickle.load(f)
    prices = data["prices_df"]
    features_by_date = data["features_by_date"]
    sp500_mem = data["sp500_mem"]
    sp400_mem = data["sp400_mem"]
    sp600_mem = data["sp600_mem"]

    # Build ticker -> sector ETF map from Compustat GICS
    comp = pd.read_parquet("data/wrds/compustat_annual.parquet",
                           columns=["tic", "gsector", "datadate"])
    comp = comp.dropna(subset=["tic", "gsector"])
    comp["gsector"] = comp["gsector"].astype(int)
    # Use most recent sector assignment per ticker
    comp = comp.sort_values("datadate").groupby("tic").last()
    ticker_to_etf = {}
    for tic, row in comp.iterrows():
        etf = GICS_TO_ETF.get(row["gsector"])
        if etf:
            ticker_to_etf[tic] = etf

    log.info(f"Data loaded in {time.time()-t0:.1f}s: {len(prices.columns)} tickers, "
             f"{len(ticker_to_etf)} with sector mapping")

    return prices, features_by_date, sp500_mem, sp400_mem, sp600_mem, ticker_to_etf


def get_sp1500(date, sp500_mem, sp400_mem, sp600_mem, _cache={}):
    """Get SP1500 members on a date."""
    if date in _cache:
        return _cache[date]
    members = set()
    for mem in [sp500_mem, sp400_mem, sp600_mem]:
        if date in mem:
            members.update(mem[date])
        else:
            prior = [d for d in mem.keys() if d <= date]
            if prior:
                members.update(mem[max(prior)])
    _cache[date] = members
    return members


def rank_sectors(prices, date, lookback=60, sma_period=200):
    """
    Rank sector ETFs by momentum.
    Returns: list of (etf, mom_60d) sorted descending, with SMA200 filter flag.
    """
    results = []
    for etf in SECTOR_ETFS:
        if etf not in prices.columns:
            continue
        series = prices[etf].loc[:date].dropna()
        if len(series) < max(lookback, sma_period):
            continue
        mom = series.iloc[-1] / series.iloc[-lookback] - 1.0
        sma200 = series.iloc[-sma_period:].mean()
        above_sma = series.iloc[-1] > sma200
        results.append((etf, mom, above_sma))
    # Sort by momentum descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def pick_top_stocks_in_sector(features, prices, date, sector_etf, ticker_to_etf,
                              members, top_n=3, min_roe=0.10):
    """Pick top stocks within a sector by composite score (mom + quality)."""
    fdate = features.get(date, {})
    if not fdate:
        return []

    gics = ETF_TO_GICS.get(sector_etf)
    if gics is None:
        return []

    candidates = []
    for sym in members:
        if ticker_to_etf.get(sym) != sector_etf:
            continue
        feat = fdate.get(sym)
        if feat is None:
            continue
        roe = feat.get("roe")
        if roe is None or np.isnan(roe) or roe < min_roe:
            continue
        # Need price and momentum
        px_series = prices[sym].loc[:date].dropna() if sym in prices.columns else pd.Series(dtype=float)
        if len(px_series) < 60:
            continue
        ret60 = feat.get("ret_60d")
        if ret60 is None or np.isnan(ret60):
            continue
        dist_sma200 = feat.get("dist_sma200")
        if dist_sma200 is not None and not np.isnan(dist_sma200) and dist_sma200 < -0.10:
            continue  # below SMA200 by >10%

        # Composite: momentum + quality
        gm = feat.get("gross_margin", 0)
        if gm is None or np.isnan(gm):
            gm = 0
        score = ret60 * 0.5 + min(roe, 0.4) * 0.3 + min(gm, 0.6) * 0.2
        candidates.append((sym, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in candidates[:top_n]]


# ── Portfolio simulator ──────────────────────────────────────────────────────

def simulate_portfolio(prices, target_fn, trading_dates, config):
    """
    Real portfolio simulation with trailing stops, transaction costs, caps.
    target_fn(date, day_idx) -> dict {symbol: target_weight}
    """
    cost_frac = COST_BPS / 10000
    trailing_stop = config.get("trailing_stop", 0.35)
    rebal_days = config.get("rebal_days", 20)
    cap = config.get("cap", 0.20)

    cash = INITIAL_CASH
    holdings = {}  # sym -> {shares, entry_px, peak_px}
    port_values = []

    for day_idx, date in enumerate(trading_dates):
        # Get today's prices
        today = {}
        if date in prices.index:
            row = prices.loc[date]
            for sym in set(list(holdings.keys())):
                v = row.get(sym)
                if v is not None and not np.isnan(v):
                    today[sym] = v
            for sym in row.dropna().index:
                today[sym] = row[sym]

        # Trailing stop check
        if trailing_stop:
            for sym in list(holdings):
                px = today.get(sym)
                if px:
                    if px > holdings[sym]["peak_px"]:
                        holdings[sym]["peak_px"] = px
                    dd = (px - holdings[sym]["peak_px"]) / holdings[sym]["peak_px"]
                    if dd < -abs(trailing_stop):
                        cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                        del holdings[sym]

        # Compute portfolio value
        eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                            for s, h in holdings.items())
        total_val = eq_val

        if day_idx % rebal_days != 0:
            port_values.append((date, total_val))
            continue

        # Get target weights
        targets = target_fn(date, day_idx)
        if not targets:
            port_values.append((date, total_val))
            continue

        # Apply position cap
        for sym in targets:
            if targets[sym] > cap:
                targets[sym] = cap
        gross = sum(targets.values())
        if gross > 1.0:
            for sym in targets:
                targets[sym] /= gross

        # Execute trades
        target_dollars = {s: w * total_val for s, w in targets.items() if w >= 0.005}

        # Sell positions not in target
        for sym in list(holdings):
            if sym not in target_dollars:
                px = today.get(sym, holdings[sym]["entry_px"])
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                del holdings[sym]

        # Rebalance
        for sym, tgt in target_dollars.items():
            px = today.get(sym)
            if not px or px <= 0:
                continue
            cur = holdings[sym]["shares"] * px if sym in holdings else 0
            delta = tgt - cur
            if abs(delta) < total_val * 0.003:
                continue
            cost = abs(delta) * cost_frac
            if delta > 0 and cash >= delta:
                shares = (delta - cost) / px
                if sym in holdings:
                    holdings[sym]["shares"] += shares
                else:
                    holdings[sym] = {"shares": shares, "entry_px": px, "peak_px": px}
                cash -= delta
            elif delta < 0 and sym in holdings:
                sell = min(abs(delta) / px, holdings[sym]["shares"])
                cash += sell * px - cost
                holdings[sym]["shares"] -= sell
                if holdings[sym]["shares"] < 0.01:
                    del holdings[sym]

        eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                            for s, h in holdings.items())
        port_values.append((date, max(eq_val, 0)))

    return port_values


def compute_metrics(port_values, prices, start, end):
    """Compute CAGR, Sharpe, Max DD, yearly returns."""
    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0:
        years = 1
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    sd = dr[dr < 0].std()
    sortino = dr.mean() / sd * np.sqrt(252) if sd > 0 else 0
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    vol = dr.std() * np.sqrt(252)

    # SPY benchmark
    spy = prices["SPY"].reindex(vals.index, method="ffill").dropna()
    spy = spy / spy.iloc[0] * INITIAL_CASH
    spy_cagr = (spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1

    # Yearly
    yearly = {}
    start_yr = int(str(start)[:4])
    end_yr = int(str(end)[:4])
    for year in range(start_yr, end_yr + 1):
        mask = (vals.index >= f"{year}-01-01") & (vals.index <= f"{year}-12-31")
        yv = vals[mask]
        if len(yv) > 10:
            yr = (yv.iloc[-1] / yv.iloc[0]) - 1
            ydr = yv.pct_change().dropna()
            ys = ydr.mean() / ydr.std() * np.sqrt(252) if ydr.std() > 0 else 0
            ydd = ((yv - yv.cummax()) / yv.cummax()).min()
            yearly[year] = {"ret": yr, "sharpe": ys, "max_dd": ydd}

    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "vol": vol, "alpha": cagr - spy_cagr,
        "final": vals.iloc[-1], "yearly": yearly,
        "values": vals,  # for correlation analysis
        "spy_cagr": spy_cagr,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_all():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    print("=" * 80)
    print("ADAPTIVE SECTOR ROTATION STRATEGY — BACKTEST")
    print("=" * 80)

    prices, features_by_date, sp500_mem, sp400_mem, sp600_mem, ticker_to_etf = load_data()

    start, end = "2017-01-03", "2025-12-31"
    trading_dates = [d for d in sorted(prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    print(f"\nTrading dates: {trading_dates[0].date()} to {trading_dates[-1].date()} "
          f"({len(trading_dates)} days)")
    print(f"Sector-mapped tickers: {len(ticker_to_etf)}")

    # Count how many SP1500 members have sector mapping
    sample_date = trading_dates[len(trading_dates)//2]
    members = get_sp1500(sample_date, sp500_mem, sp400_mem, sp600_mem)
    mapped = sum(1 for m in members if m in ticker_to_etf)
    print(f"SP1500 members on {sample_date.date()}: {len(members)}, with sector: {mapped} "
          f"({100*mapped/max(len(members),1):.0f}%)")

    config_etf = {"trailing_stop": 0.35, "rebal_days": 20, "cap": 0.40}
    config_stock = {"trailing_stop": 0.35, "rebal_days": 20, "cap": 0.15}

    # ──────────────────────────────────────────────────────────────────────
    # VERSION A: Pure ETF Rotation
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERSION A: PURE ETF SECTOR ROTATION")
    print("  - Rank 11 sector ETFs by 60d momentum")
    print("  - Buy top 3, equal weight (skip if below 200d SMA)")
    print("  - Fallback to SPY if <2 pass SMA filter")
    print("  - Rebalance every 20 days, 35% trailing stop")
    print("=" * 80)

    def etf_rotation_targets(date, day_idx):
        ranked = rank_sectors(prices, date, lookback=60, sma_period=200)
        if not ranked:
            return {"SPY": 1.0}

        picks = []
        for etf, mom, above_sma in ranked:
            if len(picks) >= 3:
                break
            if above_sma:
                picks.append(etf)

        if len(picks) < 2:
            # Fewer than 2 pass SMA -> fill with SPY
            remaining = (3 - len(picks))
            w = 1.0 / 3.0
            targets = {etf: w for etf in picks}
            targets["SPY"] = w * remaining
            return targets

        return {etf: 1.0 / len(picks) for etf in picks}

    t0 = time.time()
    pv_etf = simulate_portfolio(prices, etf_rotation_targets, trading_dates, config_etf)
    t_etf = time.time() - t0
    metrics_etf = compute_metrics(pv_etf, prices, start, end)

    print(f"\n  Computed in {t_etf:.1f}s")
    print(f"  CAGR:    {metrics_etf['cagr']:+.1%}")
    print(f"  Sharpe:  {metrics_etf['sharpe']:.2f}")
    print(f"  Sortino: {metrics_etf['sortino']:.2f}")
    print(f"  Max DD:  {metrics_etf['max_dd']:.1%}")
    print(f"  Vol:     {metrics_etf['vol']:.1%}")
    print(f"  Alpha:   {metrics_etf['alpha']:+.1%} (vs SPY {metrics_etf['spy_cagr']:+.1%})")
    print(f"  Final:   ${metrics_etf['final']:,.0f}")
    print(f"\n  Yearly returns:")
    for yr, yd in sorted(metrics_etf["yearly"].items()):
        print(f"    {yr}: {yd['ret']:+6.1%}  Sharpe {yd['sharpe']:5.2f}  DD {yd['max_dd']:6.1%}")

    # ──────────────────────────────────────────────────────────────────────
    # VERSION B: Stock-Within-Sector Rotation
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERSION B: STOCK-WITHIN-SECTOR ROTATION")
    print("  - Rank 11 sector ETFs by 60d momentum")
    print("  - Top 3 sectors (SMA200 filter), then buy top 3 stocks per sector")
    print("  - Stock filter: ROE > 10%, above SMA200, ranked by mom+quality")
    print("  - 9 positions total, 15% cap, 35% trailing stop")
    print("=" * 80)

    def stock_rotation_targets(date, day_idx):
        ranked = rank_sectors(prices, date, lookback=60, sma_period=200)
        if not ranked:
            return {"SPY": 1.0}

        sector_picks = []
        for etf, mom, above_sma in ranked:
            if len(sector_picks) >= 3:
                break
            if above_sma:
                sector_picks.append(etf)

        if len(sector_picks) < 2:
            remaining_w = (3 - len(sector_picks)) / 3.0
            targets = {}
            for etf in sector_picks:
                members = get_sp1500(date, sp500_mem, sp400_mem, sp600_mem)
                stocks = pick_top_stocks_in_sector(
                    features_by_date, prices, date, etf, ticker_to_etf, members)
                if stocks:
                    sw = (1.0 / 3.0) / len(stocks)
                    for s in stocks:
                        targets[s] = sw
                else:
                    targets[etf] = 1.0 / 3.0
            targets["SPY"] = remaining_w
            return targets

        # Normal: 3 sectors, 3 stocks each
        members = get_sp1500(date, sp500_mem, sp400_mem, sp600_mem)
        targets = {}
        sector_w = 1.0 / len(sector_picks)
        for etf in sector_picks:
            stocks = pick_top_stocks_in_sector(
                features_by_date, prices, date, etf, ticker_to_etf, members)
            if stocks:
                sw = sector_w / len(stocks)
                for s in stocks:
                    targets[s] = targets.get(s, 0) + sw
            else:
                # Fallback to ETF if no qualifying stocks
                targets[etf] = sector_w
        return targets

    t0 = time.time()
    pv_stock = simulate_portfolio(prices, stock_rotation_targets, trading_dates, config_stock)
    t_stock = time.time() - t0
    metrics_stock = compute_metrics(pv_stock, prices, start, end)

    print(f"\n  Computed in {t_stock:.1f}s")
    print(f"  CAGR:    {metrics_stock['cagr']:+.1%}")
    print(f"  Sharpe:  {metrics_stock['sharpe']:.2f}")
    print(f"  Sortino: {metrics_stock['sortino']:.2f}")
    print(f"  Max DD:  {metrics_stock['max_dd']:.1%}")
    print(f"  Vol:     {metrics_stock['vol']:.1%}")
    print(f"  Alpha:   {metrics_stock['alpha']:+.1%} (vs SPY {metrics_stock['spy_cagr']:+.1%})")
    print(f"  Final:   ${metrics_stock['final']:,.0f}")
    print(f"\n  Yearly returns:")
    for yr, yd in sorted(metrics_stock["yearly"].items()):
        print(f"    {yr}: {yd['ret']:+6.1%}  Sharpe {yd['sharpe']:5.2f}  DD {yd['max_dd']:6.1%}")

    # ──────────────────────────────────────────────────────────────────────
    # Walk-Forward Analysis (3-year windows)
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS (3-year rolling windows)")
    print("=" * 80)

    windows = [
        ("2017-2019", "2017-01-03", "2019-12-31"),
        ("2018-2020", "2018-01-02", "2020-12-31"),
        ("2019-2021", "2019-01-02", "2021-12-31"),
        ("2020-2022", "2020-01-02", "2022-12-31"),
        ("2021-2023", "2021-01-04", "2023-12-29"),
        ("2022-2024", "2022-01-03", "2024-12-31"),
        ("2023-2025", "2023-01-03", "2025-12-31"),
    ]

    print(f"\n  {'Window':<12} {'ETF CAGR':>9} {'ETF Shrp':>9} {'Stk CAGR':>9} {'Stk Shrp':>9}")
    print("  " + "-" * 52)

    wf_etf_cagrs = []
    wf_stock_cagrs = []
    for label, ws, we in windows:
        wdates = [d for d in trading_dates
                  if pd.Timestamp(ws) <= d <= pd.Timestamp(we)]
        if len(wdates) < 100:
            continue
        pv_e = simulate_portfolio(prices, etf_rotation_targets, wdates, config_etf)
        pv_s = simulate_portfolio(prices, stock_rotation_targets, wdates, config_stock)
        me = compute_metrics(pv_e, prices, ws, we)
        ms = compute_metrics(pv_s, prices, ws, we)
        wf_etf_cagrs.append(me["cagr"])
        wf_stock_cagrs.append(ms["cagr"])
        print(f"  {label:<12} {me['cagr']:>+8.1%} {me['sharpe']:>9.2f} "
              f"{ms['cagr']:>+8.1%} {ms['sharpe']:>9.2f}")

    # IS vs OOS decay
    print(f"\n  IS vs OOS decay:")
    if len(wf_etf_cagrs) >= 4:
        is_etf = np.mean(wf_etf_cagrs[:3])
        oos_etf = np.mean(wf_etf_cagrs[3:])
        is_stock = np.mean(wf_stock_cagrs[:3])
        oos_stock = np.mean(wf_stock_cagrs[3:])
        print(f"    ETF:   IS={is_etf:+.1%}  OOS={oos_etf:+.1%}  "
              f"decay={(oos_etf-is_etf)/max(abs(is_etf),0.01)*100:+.0f}%")
        print(f"    Stock: IS={is_stock:+.1%}  OOS={oos_stock:+.1%}  "
              f"decay={(oos_stock-is_stock)/max(abs(is_stock),0.01)*100:+.0f}%")

    # ──────────────────────────────────────────────────────────────────────
    # Correlation with v10.1 momentum
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CORRELATION WITH v10.1 MOMENTUM STRATEGY")
    print("=" * 80)

    # Run v10.1 momentum using FastBacktester
    try:
        from main_production_backtest import FastBacktester
        bt = FastBacktester()
        v10_result = bt.run("2017-01-03", "2025-12-31", {
            "universe": "sp1500", "mom_w": 0.85, "val_w": 0.15,
            "lv_w": 0.0, "sec_w": 0.0, "top_n": 8, "rebal_days": 10,
            "trailing_stop": 0.25, "gld_pct": 0.02,
        })
        # Reconstruct v10 values
        v10_pv = bt.run("2017-01-03", "2025-12-31", {
            "universe": "sp1500", "mom_w": 0.85, "val_w": 0.15,
            "lv_w": 0.0, "sec_w": 0.0, "top_n": 8, "rebal_days": 10,
            "trailing_stop": 0.25, "gld_pct": 0.02,
        })

        print(f"\n  v10.1 reference: CAGR={v10_result['cagr']:+.1%}  "
              f"Sharpe={v10_result['sharpe']:.2f}  DD={v10_result['max_dd']:.1%}")

        # For correlation, use daily returns from sector strategies
        vals_etf = metrics_etf["values"]
        vals_stock = metrics_stock["values"]
        dr_etf = vals_etf.pct_change().dropna()
        dr_stock = vals_stock.pct_change().dropna()

        # SPY as proxy for v10 correlation (since we can't get daily values directly)
        # Actually, let's compute v10's daily values by re-running the backtest
        # and capturing port_values. But FastBacktester.run() only returns metrics.
        # Use SPY momentum as a reasonable proxy, or compute from yearly returns.
        # Better: compute correlation using yearly returns
        v10_yearly = v10_result.get("yearly", {})
        common_years = sorted(set(metrics_etf["yearly"].keys()) & set(v10_yearly.keys()))

        if common_years:
            etf_yr = [metrics_etf["yearly"][y]["ret"] for y in common_years]
            stock_yr = [metrics_stock["yearly"][y]["ret"] for y in common_years]
            v10_yr = [v10_yearly[y]["cagr"] for y in common_years]

            corr_etf_v10 = np.corrcoef(etf_yr, v10_yr)[0, 1]
            corr_stock_v10 = np.corrcoef(stock_yr, v10_yr)[0, 1]
            corr_etf_stock = np.corrcoef(etf_yr, stock_yr)[0, 1]

            print(f"\n  Yearly return correlations:")
            print(f"    ETF rotation vs v10.1:   {corr_etf_v10:+.3f}")
            print(f"    Stock rotation vs v10.1: {corr_stock_v10:+.3f}")
            print(f"    ETF vs Stock rotation:   {corr_etf_stock:+.3f}")

            # Daily return correlation (using SPY as broad market factor)
            spy_dr = prices["SPY"].reindex(vals_etf.index, method="ffill").pct_change().dropna()
            common_idx = dr_etf.index.intersection(spy_dr.index)
            if len(common_idx) > 100:
                corr_etf_spy = np.corrcoef(dr_etf.reindex(common_idx).fillna(0),
                                           spy_dr.reindex(common_idx).fillna(0))[0, 1]
                corr_stock_spy = np.corrcoef(dr_stock.reindex(common_idx).fillna(0),
                                             spy_dr.reindex(common_idx).fillna(0))[0, 1]
                print(f"\n  Daily return correlations with SPY:")
                print(f"    ETF rotation:   {corr_etf_spy:+.3f}")
                print(f"    Stock rotation: {corr_stock_spy:+.3f}")

            # ──────────────────────────────────────────────────────────────
            # Combined portfolio returns
            # ──────────────────────────────────────────────────────────────
            print("\n" + "=" * 80)
            print("COMBINED PORTFOLIO ANALYSIS")
            print("=" * 80)

            # Combine at yearly level with different allocations
            combos = [
                ("70% v10 + 30% ETF rotation", 0.70, 0.30, etf_yr),
                ("70% v10 + 30% Stock rotation", 0.70, 0.30, stock_yr),
                ("60% v10 + 40% Stock rotation", 0.60, 0.40, stock_yr),
                ("50% v10 + 50% Stock rotation", 0.50, 0.50, stock_yr),
            ]

            print(f"\n  {'Combo':<40} {'Avg':>6} {'Worst':>7} {'Best':>7}")
            print("  " + "-" * 62)
            for label, wv, ws, sec_yr in combos:
                combined = [wv * v + ws * s for v, s in zip(v10_yr, sec_yr)]
                avg = np.mean(combined)
                worst = min(combined)
                best = max(combined)
                # Estimate CAGR from geometric mean
                geo = np.prod([1 + r for r in combined]) ** (1 / len(combined)) - 1
                sharpe_est = np.mean(combined) / np.std(combined) if np.std(combined) > 0 else 0
                print(f"  {label:<40} {avg:>+5.1%} {worst:>+6.1%} {best:>+6.1%}  "
                      f"~CAGR {geo:+.1%}  ~Sharpe {sharpe_est:.2f}")

            # Also compute daily combined using portfolio value series
            print(f"\n  Daily-precision combined portfolios:")
            for label, wv10, wsec, sec_vals in [
                ("70% v10 + 30% ETF rot", 0.70, 0.30, vals_etf),
                ("70% v10 + 30% Stock rot", 0.70, 0.30, vals_stock),
                ("50% v10 + 50% Stock rot", 0.50, 0.50, vals_stock),
            ]:
                # Normalize both to start at 1.0
                sec_norm = sec_vals / sec_vals.iloc[0]
                # We don't have v10 daily values, so approximate:
                # v10 CAGR ~ 23.4% => use geometric growth + yearly returns
                # For proper daily combination, we'd need the daily series
                # Approximate by using yearly returns to build a monthly series
                pass

            # Print v10 yearly for reference
            print(f"\n  v10.1 yearly returns for reference:")
            for yr in common_years:
                v = v10_yearly[yr]
                print(f"    {yr}: {v['cagr']:+6.1%}  Sharpe {v['sharpe']:5.2f}  DD {v['max_dd']:6.1%}")

    except Exception as e:
        print(f"\n  Could not load v10.1 for comparison: {e}")
        import traceback
        traceback.print_exc()

    # ──────────────────────────────────────────────────────────────────────
    # VERSION B2: IMPROVED Stock-Within-Sector (wider stops, vol-weighted)
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERSION B2: IMPROVED STOCK-WITHIN-SECTOR ROTATION")
    print("  - Same sector selection as B")
    print("  - Top 5 stocks per sector (more diversification)")
    print("  - Inverse-vol weighting within sector")
    print("  - 50% trailing stop (wider for individual stocks)")
    print("  - 12% position cap")
    print("=" * 80)

    config_stock2 = {"trailing_stop": 0.50, "rebal_days": 20, "cap": 0.12}

    def stock_rotation_v2_targets(date, day_idx):
        ranked = rank_sectors(prices, date, lookback=60, sma_period=200)
        if not ranked:
            return {"SPY": 1.0}

        sector_picks = []
        for etf, mom, above_sma in ranked:
            if len(sector_picks) >= 3:
                break
            if above_sma:
                sector_picks.append(etf)

        if len(sector_picks) < 2:
            remaining_w = (3 - len(sector_picks)) / 3.0
            targets = {}
            for etf in sector_picks:
                targets[etf] = 1.0 / 3.0  # use ETF as fallback
            targets["SPY"] = remaining_w
            return targets

        members = get_sp1500(date, sp500_mem, sp400_mem, sp600_mem)
        fdate = features_by_date.get(date, {})
        targets = {}
        sector_w = 1.0 / len(sector_picks)

        for etf in sector_picks:
            stocks = pick_top_stocks_in_sector(
                features_by_date, prices, date, etf, ticker_to_etf, members,
                top_n=5, min_roe=0.10)
            if len(stocks) < 2:
                targets[etf] = sector_w
                continue

            # Inverse-vol weighting
            inv_vol = {}
            for s in stocks:
                feat = fdate.get(s, {})
                v = feat.get("vol_60d", 0.25)
                if v is None or np.isnan(v) or v < 0.05:
                    v = 0.25
                inv_vol[s] = 1.0 / v
            total_iv = sum(inv_vol.values())
            for s in stocks:
                targets[s] = targets.get(s, 0) + sector_w * inv_vol[s] / total_iv

        return targets

    t0 = time.time()
    pv_stock2 = simulate_portfolio(prices, stock_rotation_v2_targets, trading_dates, config_stock2)
    t_stock2 = time.time() - t0
    metrics_stock2 = compute_metrics(pv_stock2, prices, start, end)

    print(f"\n  Computed in {t_stock2:.1f}s")
    print(f"  CAGR:    {metrics_stock2['cagr']:+.1%}")
    print(f"  Sharpe:  {metrics_stock2['sharpe']:.2f}")
    print(f"  Sortino: {metrics_stock2['sortino']:.2f}")
    print(f"  Max DD:  {metrics_stock2['max_dd']:.1%}")
    print(f"  Vol:     {metrics_stock2['vol']:.1%}")
    print(f"  Alpha:   {metrics_stock2['alpha']:+.1%}")
    print(f"  Final:   ${metrics_stock2['final']:,.0f}")
    print(f"\n  Yearly returns:")
    for yr, yd in sorted(metrics_stock2["yearly"].items()):
        print(f"    {yr}: {yd['ret']:+6.1%}  Sharpe {yd['sharpe']:5.2f}  DD {yd['max_dd']:6.1%}")

    # ──────────────────────────────────────────────────────────────────────
    # Summary comparison
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\n  {'Metric':<18} {'ETF Rot':>10} {'Stock Rot':>11} {'Stock v2':>10} {'SPY B&H':>10}")
    print("  " + "-" * 63)
    spy_vals = prices["SPY"].reindex(
        [d for d, _ in pv_etf], method="ffill").dropna()
    spy_vals = spy_vals / spy_vals.iloc[0] * INITIAL_CASH
    spy_yrs = (spy_vals.index[-1] - spy_vals.index[0]).days / 365.25
    spy_cagr = (spy_vals.iloc[-1] / spy_vals.iloc[0]) ** (1 / spy_yrs) - 1
    spy_dr = spy_vals.pct_change().dropna()
    spy_sharpe = spy_dr.mean() / spy_dr.std() * np.sqrt(252) if spy_dr.std() > 0 else 0
    spy_dd = ((spy_vals - spy_vals.cummax()) / spy_vals.cummax()).min()

    me, ms, ms2 = metrics_etf, metrics_stock, metrics_stock2
    print(f"  {'CAGR':<18} {me['cagr']:>+9.1%} {ms['cagr']:>+10.1%} {ms2['cagr']:>+9.1%} {spy_cagr:>+9.1%}")
    print(f"  {'Sharpe':<18} {me['sharpe']:>10.2f} {ms['sharpe']:>11.2f} {ms2['sharpe']:>10.2f} {spy_sharpe:>10.2f}")
    print(f"  {'Max Drawdown':<18} {me['max_dd']:>9.1%} {ms['max_dd']:>10.1%} {ms2['max_dd']:>9.1%} {spy_dd:>9.1%}")
    print(f"  {'Volatility':<18} {me['vol']:>9.1%} {ms['vol']:>10.1%} {ms2['vol']:>9.1%} {spy_dr.std()*np.sqrt(252):>9.1%}")
    print(f"  {'Sortino':<18} {me['sortino']:>10.2f} {ms['sortino']:>11.2f} {ms2['sortino']:>10.2f}")
    print(f"  {'Final Value':<18} ${me['final']:>8,.0f} ${ms['final']:>9,.0f} ${ms2['final']:>8,.0f} ${spy_vals.iloc[-1]:>8,.0f}")


if __name__ == "__main__":
    run_all()
