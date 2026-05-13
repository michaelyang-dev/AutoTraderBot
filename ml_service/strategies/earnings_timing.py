"""
Earnings Announcement Timing Strategy
======================================
Buy stocks 2-3 days BEFORE their scheduled earnings announcement,
sell the day after.  Well-documented pre-earnings drift anomaly.

Quality filters: ROE > 10%, price > $10, SP1500 membership.
Portfolio: top-10 positions, equal-weight, rotate as new earnings come.

Usage:
    cd ml_service && python3 -m strategies.earnings_timing
"""

import os, sys, warnings, json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def log(msg):
    print(msg, flush=True)


def load_data():
    """Load earnings dates, prices, and features."""
    log("Loading data...")

    # Earnings dates
    earn = pd.read_parquet(DATA_DIR / "fundamentals_earnings.parquet")
    earn["date"] = pd.to_datetime(earn["date"])
    # Keep only rows with actual earnings data (not future estimates)
    earn = earn.dropna(subset=["eps_actual"])
    log(f"  Earnings events: {len(earn):,} (with eps_actual)")

    # SP1500 universe
    with open(DATA_DIR / "wrds" / "complete_sp1500_universe.pkl", "rb") as f:
        uni = pickle.load(f)

    prices_df = uni["prices_df"]
    features_by_date = uni["features_by_date"]
    log(f"  Prices: {prices_df.shape[0]} days x {prices_df.shape[1]} symbols")
    log(f"  Features: {len(features_by_date)} dates")

    return earn, prices_df, features_by_date


def build_earnings_calendar(earn, prices_df):
    """Build a calendar of earnings events aligned to trading dates."""
    trading_dates = prices_df.index.sort_values()
    td_set = set(trading_dates)

    # Map each calendar date to the nearest trading date (on or after)
    def to_trading_date(d):
        """Find the trading date on or just before d."""
        idx = trading_dates.searchsorted(d, side="right") - 1
        if idx < 0:
            return None
        return trading_dates[idx]

    def trading_date_offset(d, offset):
        """Get trading date `offset` days from d."""
        idx = trading_dates.searchsorted(d, side="right") - 1
        if idx < 0:
            return None
        target = idx + offset
        if target < 0 or target >= len(trading_dates):
            return None
        return trading_dates[target]

    records = []
    symbols_in_prices = set(prices_df.columns)

    for _, row in earn.iterrows():
        sym = row["symbol"]
        if sym not in symbols_in_prices:
            continue

        earn_date = row["date"]
        # Find the trading date on/before earnings
        earn_td = to_trading_date(earn_date)
        if earn_td is None:
            continue

        # Buy date: 3 trading days before earnings
        buy_date = trading_date_offset(earn_td, -3)
        # Sell date: 1 trading day after earnings
        sell_date = trading_date_offset(earn_td, +1)

        if buy_date is None or sell_date is None:
            continue

        # Get prices
        buy_px = prices_df.loc[buy_date, sym] if buy_date in prices_df.index else np.nan
        sell_px = prices_df.loc[sell_date, sym] if sell_date in prices_df.index else np.nan

        if np.isnan(buy_px) or np.isnan(sell_px) or buy_px <= 0:
            continue

        ret = sell_px / buy_px - 1.0

        records.append({
            "symbol": sym,
            "earn_date": earn_td,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "buy_px": buy_px,
            "sell_px": sell_px,
            "return": ret,
            "eps_actual": row["eps_actual"],
            "eps_estimated": row.get("eps_estimated", np.nan),
        })

    cal = pd.DataFrame(records)
    log(f"  Earnings calendar: {len(cal):,} events with valid prices")
    return cal


def filter_quality(cal, features_by_date, prices_df, min_roe=0.10, min_price=10.0):
    """Filter to quality stocks: ROE > 10%, price > $10."""
    keep = []
    fbd_dates = sorted(features_by_date.keys())

    for _, row in cal.iterrows():
        buy_date = row["buy_date"]
        sym = row["symbol"]

        # Price filter
        if row["buy_px"] < min_price:
            continue

        # Find nearest feature date on or before buy_date
        idx = np.searchsorted(fbd_dates, buy_date, side="right") - 1
        if idx < 0:
            continue
        feat_date = fbd_dates[idx]
        feats = features_by_date[feat_date]

        if sym not in feats:
            continue

        sym_feats = feats[sym]
        roe = sym_feats.get("roe", 0)
        if roe is None or roe < min_roe:
            continue

        keep.append(row)

    filtered = pd.DataFrame(keep)
    log(f"  After quality filter: {len(filtered):,} events (ROE>{min_roe:.0%}, price>${min_price})")
    return filtered


def rank_candidates(events, features_by_date, fbd_dates, buy_date):
    """Rank earnings candidates by quality signals for better selection."""
    idx = np.searchsorted(fbd_dates, buy_date, side="right") - 1
    if idx < 0:
        return events

    feat_date = fbd_dates[idx]
    feats = features_by_date[feat_date]

    scores = []
    for _, ev in events.iterrows():
        sym = ev["symbol"]
        if sym not in feats:
            scores.append(0)
            continue
        f = feats[sym]
        # Prefer: high ROE, positive momentum, lower vol, positive SUE
        roe = f.get("roe", 0) or 0
        mom = f.get("ret_252d", 0) or 0
        vol = f.get("vol_60d", 0.3) or 0.3
        sue = f.get("recent_sue", 0) or 0
        # Composite score
        score = roe * 2 + mom * 1.5 - vol * 1.0 + sue * 0.5
        scores.append(score)

    events = events.copy()
    events["_score"] = scores
    return events.sort_values("_score", ascending=False)


def backtest_portfolio(cal, prices_df, features_by_date, start="2017-01-01", end="2025-12-31",
                       max_positions=10, cost_bps=10):
    """
    Backtest: hold up to max_positions pre-earnings trades, equal-weight.
    Each position: buy at close T-3, sell at close T+1.

    Key improvement: fully invest capital across available positions,
    rank candidates by quality, and handle overlapping earnings windows.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    trading_dates = prices_df.index[(prices_df.index >= start_dt) & (prices_df.index <= end_dt)]
    cost_frac = cost_bps / 10000

    # Group events by buy_date
    cal = cal[(cal["buy_date"] >= start_dt) & (cal["sell_date"] <= end_dt)].copy()
    cal = cal.sort_values("buy_date")

    fbd_dates = sorted(features_by_date.keys())

    initial_cash = 100_000
    cash = initial_cash
    active_trades = []
    portfolio_values = []
    trade_log = []

    # Pre-compute: for each buy_date, which events are available
    buy_date_events = cal.groupby("buy_date")
    # Track symbols already held to avoid duplicates
    held_symbols = set()

    for date in trading_dates:
        # Close expired trades (sell on sell_date)
        new_active = []
        for t in active_trades:
            if date >= t["sell_date"]:
                sell_px = prices_df.loc[t["sell_date"], t["symbol"]]
                if np.isnan(sell_px):
                    sell_px = t["buy_px"]
                proceeds = t["shares"] * sell_px * (1 - cost_frac)
                cash += proceeds
                held_symbols.discard(t["symbol"])
                trade_log.append({
                    "symbol": t["symbol"],
                    "buy_date": t["buy_date"],
                    "sell_date": t["sell_date"],
                    "buy_px": t["buy_px"],
                    "sell_px": sell_px,
                    "return": sell_px / t["buy_px"] - 1,
                    "pnl": proceeds - t["shares"] * t["buy_px"] * (1 + cost_frac),
                })
            else:
                new_active.append(t)
        active_trades = new_active

        # Open new trades on buy_date
        if date in buy_date_events.groups:
            events = buy_date_events.get_group(date)
            # Filter out already-held symbols
            events = events[~events["symbol"].isin(held_symbols)]
            slots = max_positions - len(active_trades)
            if slots > 0 and len(events) > 0:
                # Rank candidates by quality
                events = rank_candidates(events, features_by_date, fbd_dates, date)
                candidates = events.head(slots)

                # Equal-weight across ALL positions (active + new)
                total_equity = cash
                for t in active_trades:
                    px = prices_df.loc[date, t["symbol"]] if date in prices_df.index else t["buy_px"]
                    if np.isnan(px): px = t["buy_px"]
                    total_equity += t["shares"] * px

                n_new = min(len(candidates), slots)
                position_size = total_equity / max(len(active_trades) + n_new, 1) * 0.98

                for _, ev in candidates.iterrows():
                    buy_px = ev["buy_px"]
                    alloc = min(position_size, cash * 0.98)
                    shares = int(alloc / buy_px)
                    if shares <= 0:
                        continue
                    cost = shares * buy_px * (1 + cost_frac)
                    if cost > cash:
                        shares = int(cash * 0.95 / buy_px / (1 + cost_frac))
                        if shares <= 0:
                            continue
                        cost = shares * buy_px * (1 + cost_frac)
                    cash -= cost
                    active_trades.append({
                        "symbol": ev["symbol"],
                        "buy_date": ev["buy_date"],
                        "sell_date": ev["sell_date"],
                        "buy_px": buy_px,
                        "shares": shares,
                    })
                    held_symbols.add(ev["symbol"])

        # Mark to market
        equity = cash
        for t in active_trades:
            px = prices_df.loc[date, t["symbol"]] if date in prices_df.index else t["buy_px"]
            if np.isnan(px):
                px = t["buy_px"]
            equity += t["shares"] * px

        portfolio_values.append((date, equity))

    pv = pd.DataFrame(portfolio_values, columns=["date", "equity"]).set_index("date")
    trades = pd.DataFrame(trade_log)
    return pv, trades


def compute_metrics(pv):
    """Compute CAGR, Sharpe, max drawdown, yearly returns."""
    rets = pv["equity"].pct_change().dropna()
    years = (pv.index[-1] - pv.index[0]).days / 365.25
    total_ret = pv["equity"].iloc[-1] / pv["equity"].iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1

    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    # Max drawdown
    cummax = pv["equity"].cummax()
    dd = (pv["equity"] - cummax) / cummax
    max_dd = dd.min()

    # Yearly returns
    yearly = pv["equity"].resample("YE").last().pct_change().dropna()

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_ret,
        "years": years,
        "yearly": yearly,
        "daily_returns": rets,
    }


def run_momentum_backtest(prices_df, features_by_date, start="2017-01-01", end="2025-12-31"):
    """Simple momentum strategy backtest for correlation analysis.

    Replicates v10.1 momentum: buy top-8 by 12M-1M momentum, rebal every 10 days.
    """
    log("\n--- Running momentum comparison backtest ---")
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    trading_dates = prices_df.index[(prices_df.index >= start_dt) & (prices_df.index <= end_dt)]

    fbd_dates = sorted(features_by_date.keys())
    cost_frac = 10 / 10000
    cash = 100_000
    holdings = {}
    port_values = []
    rebal_counter = 0

    for date in trading_dates:
        rebal_counter += 1

        if rebal_counter % 10 == 1:
            # Find features
            idx = np.searchsorted(fbd_dates, date, side="right") - 1
            if idx < 0:
                equity = cash
                for sym, h in holdings.items():
                    px = prices_df.loc[date, sym] if sym in prices_df.columns else h["px"]
                    if np.isnan(px): px = h["px"]
                    equity += h["shares"] * px
                port_values.append((date, equity))
                continue

            feat_date = fbd_dates[idx]
            feats = features_by_date[feat_date]

            # Score: 12M momentum (ret_252d) minus 1M (ret_20d)
            scores = {}
            for sym, f in feats.items():
                if sym not in prices_df.columns:
                    continue
                px = prices_df.loc[date, sym] if date in prices_df.index else np.nan
                if np.isnan(px) or px < 10:
                    continue
                mom = f.get("ret_252d", 0) or 0
                rev = f.get("ret_20d", 0) or 0
                vol = f.get("vol_60d", 999) or 999
                roe = f.get("roe", 0) or 0
                if vol > 0.6 or roe < 0.05:
                    continue
                scores[sym] = mom - rev

            top = sorted(scores, key=scores.get, reverse=True)[:8]

            # Calculate equity
            equity = cash
            for sym, h in holdings.items():
                px = prices_df.loc[date, sym] if sym in prices_df.columns else h["px"]
                if np.isnan(px): px = h["px"]
                equity += h["shares"] * px

            # Sell all
            for sym, h in holdings.items():
                px = prices_df.loc[date, sym] if sym in prices_df.columns else h["px"]
                if np.isnan(px): px = h["px"]
                cash += h["shares"] * px * (1 - cost_frac)
            holdings = {}

            # Buy top 8
            if top:
                per_stock = cash * 0.95 / len(top)
                for sym in top:
                    px = prices_df.loc[date, sym]
                    if np.isnan(px) or px <= 0:
                        continue
                    shares = int(per_stock / px)
                    if shares > 0:
                        cash -= shares * px * (1 + cost_frac)
                        holdings[sym] = {"shares": shares, "px": px}

        # Mark to market
        equity = cash
        for sym, h in holdings.items():
            px = prices_df.loc[date, sym] if sym in prices_df.columns else h["px"]
            if np.isnan(px): px = h["px"]
            equity += h["shares"] * px

        port_values.append((date, equity))

    pv = pd.DataFrame(port_values, columns=["date", "equity"]).set_index("date")
    return pv


def main():
    earn, prices_df, features_by_date = load_data()

    log("\nStep 1: Build earnings calendar...")
    cal = build_earnings_calendar(earn, prices_df)

    log("\nStep 2: Apply quality filters...")
    cal_filtered = filter_quality(cal, features_by_date, prices_df)

    # Stats on pre-earnings drift
    log("\n=== PRE-EARNINGS DRIFT STATISTICS ===")
    log(f"  Mean return (T-3 to T+1): {cal_filtered['return'].mean():.4f} ({cal_filtered['return'].mean()*100:.2f}%)")
    log(f"  Median return: {cal_filtered['return'].median():.4f}")
    log(f"  Win rate: {(cal_filtered['return'] > 0).mean():.1%}")
    log(f"  Std dev: {cal_filtered['return'].std():.4f}")

    # By year
    cal_filtered = cal_filtered.copy()
    cal_filtered["year"] = cal_filtered["buy_date"].dt.year
    yearly_stats = cal_filtered.groupby("year")["return"].agg(["mean", "count", "std"])
    log("\n  Per-year drift:")
    for yr, row in yearly_stats.iterrows():
        if 2017 <= yr <= 2025:
            log(f"    {yr}: mean={row['mean']:.4f} ({row['mean']*100:.2f}%), n={int(row['count'])}, std={row['std']:.4f}")

    log("\nStep 3: Backtest portfolio (2017-2025)...")
    pv, trades = backtest_portfolio(cal_filtered, prices_df, features_by_date)

    m = compute_metrics(pv)
    log("\n" + "="*60)
    log("  EARNINGS TIMING STRATEGY RESULTS (2017-2025)")
    log("="*60)
    log(f"  CAGR:          {m['cagr']:.1%}")
    log(f"  Sharpe Ratio:  {m['sharpe']:.2f}")
    log(f"  Max Drawdown:  {m['max_dd']:.1%}")
    log(f"  Total Return:  {m['total_return']:.1%}")
    log(f"  Total Trades:  {len(trades):,}")
    if len(trades) > 0:
        log(f"  Win Rate:      {(trades['return'] > 0).mean():.1%}")
        log(f"  Avg Trade Ret: {trades['return'].mean():.2%}")
        log(f"  Avg Trade PnL: ${trades['pnl'].mean():.0f}")

    log("\n  Yearly Returns:")
    for yr, ret in m["yearly"].items():
        log(f"    {yr.year}: {ret:.1%}")

    # Step 4: Momentum comparison and correlation
    pv_mom = run_momentum_backtest(prices_df, features_by_date)
    m_mom = compute_metrics(pv_mom)

    log("\n" + "="*60)
    log("  MOMENTUM STRATEGY RESULTS (comparison)")
    log("="*60)
    log(f"  CAGR:          {m_mom['cagr']:.1%}")
    log(f"  Sharpe Ratio:  {m_mom['sharpe']:.2f}")
    log(f"  Max Drawdown:  {m_mom['max_dd']:.1%}")

    log("\n  Yearly Returns:")
    for yr, ret in m_mom["yearly"].items():
        log(f"    {yr.year}: {ret:.1%}")

    # Correlation
    earn_daily = m["daily_returns"].rename("earnings")
    mom_daily = m_mom["daily_returns"].rename("momentum")
    combined = pd.concat([earn_daily, mom_daily], axis=1).dropna()
    corr = combined.corr().loc["earnings", "momentum"]
    log(f"\n  Daily return correlation (earnings vs momentum): {corr:.3f}")

    # Combined portfolio: 70% momentum, 30% earnings
    log("\n" + "="*60)
    log("  COMBINED PORTFOLIO (70% Momentum / 30% Earnings)")
    log("="*60)

    # Align and combine
    pv_e = pv["equity"].reindex(pv_mom.index).ffill().bfill()
    pv_m = pv_mom["equity"]
    pv_e_ret = pv_e.pct_change().fillna(0)
    pv_m_ret = pv_m.pct_change().fillna(0)
    combined_ret = 0.70 * pv_m_ret + 0.30 * pv_e_ret
    combined_equity = (1 + combined_ret).cumprod() * 100_000
    combined_pv = pd.DataFrame({"equity": combined_equity})

    m_comb = compute_metrics(combined_pv)
    log(f"  CAGR:          {m_comb['cagr']:.1%}")
    log(f"  Sharpe Ratio:  {m_comb['sharpe']:.2f}")
    log(f"  Max Drawdown:  {m_comb['max_dd']:.1%}")

    log("\n  Yearly Returns:")
    for yr, ret in m_comb["yearly"].items():
        log(f"    {yr.year}: {ret:.1%}")

    # Capital efficiency: how often is capital deployed?
    log(f"\n  Capital utilization:")
    invested_pct = []
    for date in pv.index:
        row_px = prices_df.loc[date] if date in prices_df.index else pd.Series()
        # Rough: if equity barely moves day-to-day, cash heavy
    # Count days with active positions from trade log
    if len(trades) > 0:
        active_days = set()
        for _, t in trades.iterrows():
            d_range = prices_df.index[(prices_df.index >= t["buy_date"]) & (prices_df.index < t["sell_date"])]
            active_days.update(d_range)
        total_days = len(pv)
        log(f"    Days with active positions: {len(active_days)}/{total_days} ({len(active_days)/total_days:.1%})")
        log(f"    Avg trades per event day: {len(trades) / max(len(active_days), 1):.1f}")

    # Also test 80/20 and 60/40
    for w_mom, w_earn in [(0.80, 0.20), (0.60, 0.40), (0.50, 0.50)]:
        cr = w_mom * pv_m_ret + w_earn * pv_e_ret
        ce = (1 + cr).cumprod() * 100_000
        cpv = pd.DataFrame({"equity": ce})
        mc = compute_metrics(cpv)
        log(f"\n  {w_mom:.0%}/{w_earn:.0%} Mom/Earn — CAGR: {mc['cagr']:.1%}, Sharpe: {mc['sharpe']:.2f}, DD: {mc['max_dd']:.1%}")

    log("\nDone.")


if __name__ == "__main__":
    main()
