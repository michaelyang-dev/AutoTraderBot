"""
WRDS Walk-Forward Validation
=============================
Runs the SAME v9.6 strategy on WRDS institutional-grade data to get the
TRUE out-of-sample performance with:
  - Survivorship-bias-free SP500 membership (CRSP ground truth)
  - Point-in-time Compustat fundamentals (using rdq filing dates)
  - Proper delisting returns (no silent stock disappearances)

Compare results to the FMP/Massive-based backtest to measure the "honesty gap."

Usage:
    cd ml_service && python3 -m strategies.wrds_walkforward
"""

import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal,
    strategy3_sector_rotation,
    strategy4_index_inclusion,
    strategy5_lowvol_quality,
    STRATEGY_CONFIG_BULL,
    STRATEGY_CONFIG_BEAR,
    INITIAL_CASH,
    COST_BPS,
)
from wrds_universe import build_wrds_universe


def log(msg):
    print(msg, flush=True)


def run_backtest(uni, start, end, open_prices=None,
                 s1_top_n=8, s1_rebal_days=10):
    """Run the v9.6 strategy on a WRDSUniverse.

    Identical logic to true_walkforward.run_backtest() — same rebalancing,
    same constraints, same next-day-open execution. Only data source differs.
    """
    trading_dates = [d for d in sorted(uni.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

    # Next-day-open execution setup
    _next_date = {}
    for idx in range(len(trading_dates) - 1):
        _next_date[idx] = trading_dates[idx + 1]
    _open_lookup = {}
    if open_prices is not None:
        dates_in_open = open_prices.index.intersection(trading_dates)
        if len(dates_in_open) > 0:
            _open_lookup = open_prices.loc[dates_in_open].to_dict(orient="index")

    cost_frac = COST_BPS / 10000
    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    last_targets = {}
    s4_active = {}
    trade_count = 0

    for day_idx, date in enumerate(trading_dates):
        today_prices = {}
        if date in uni.prices.index:
            row = uni.prices.loc[date]
            for sym in list(holdings.keys()):
                v = row.get(sym)
                if v is not None and not np.isnan(v):
                    today_prices[sym] = v
            for sym in row.dropna().index:
                today_prices[sym] = row[sym]

        regime = uni.get_regime(date)
        vix = regime.get("vix", 20)

        t1 = strategy1_momentum_reversal(date, uni, day_idx,
                                         top_n=s1_top_n, rebal_days=s1_rebal_days)
        t3 = strategy3_sector_rotation(date, uni, day_idx)
        t4 = strategy4_index_inclusion(date, uni, day_idx, s4_active)
        t5 = strategy5_lowvol_quality(date, uni, day_idx)

        if t1 is not None:
            last_targets["s1_momentum"] = t1
        if t3 is not None:
            last_targets["s3_sector"] = t3
        last_targets["s4_inclusion"] = t4 if t4 else {}
        if t5 is not None:
            last_targets["s5_lowvol"] = t5
        last_targets["s6_short"] = {}

        major_rebal = any(x is not None for x in [t1, t3, t5])
        if not major_rebal:
            equity = cash
            for sym, h in holdings.items():
                equity += h["shares"] * today_prices.get(sym, h["entry_px"])
            port_values.append((date, equity))
            continue

        # Breadth blend
        dist_sma50 = uni.get_feature_map(date, "dist_sma50")
        breadth = (sum(1 for v in dist_sma50.values() if v > 0) /
                   max(len(dist_sma50), 1)) if dist_sma50 else 0.5
        blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))

        blended = {}
        for name, bull_pct in STRATEGY_CONFIG_BULL:
            bear_pct = dict(STRATEGY_CONFIG_BEAR).get(name, 0)
            blended[name] = bull_pct * blend + bear_pct * (1 - blend)

        paused = set()
        if vix > 40:
            paused = {"s1_momentum", "s4_inclusion"}

        combined = {}
        for name, cap_pct in blended.items():
            if name in paused:
                continue
            for sym, w in last_targets.get(name, {}).items():
                if w > 0:
                    combined[sym] = combined.get(sym, 0) + w * cap_pct

        # Constraints: 15% single-name cap, 35% sector cap
        longs = {s: w for s, w in combined.items() if w > 0}
        for sym in list(longs):
            if longs[sym] > 0.15:
                longs[sym] = 0.15
        sec_tot = {}
        for sym, w in longs.items():
            sec = uni.sector_map.get(sym, "X")
            sec_tot[sec] = sec_tot.get(sec, 0) + w
        for sec, tot in sec_tot.items():
            if tot > 0.35:
                scale = 0.35 / tot
                for sym in list(longs):
                    if uni.sector_map.get(sym, "X") == sec:
                        longs[sym] *= scale
        gross = sum(longs.values())
        if gross > 1.0:
            for sym in longs:
                longs[sym] /= gross
        combined = {s: w for s, w in longs.items() if w >= 0.005}

        # Equity
        equity = cash
        for sym, h in holdings.items():
            equity += h["shares"] * today_prices.get(sym, h["entry_px"])

        # Rebalance
        target_d = {s: w * equity for s, w in combined.items()}
        for sym in list(holdings):
            if sym not in target_d:
                px = today_prices.get(sym, holdings[sym]["entry_px"])
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                trade_count += 1
                del holdings[sym]

        next_d = _next_date.get(day_idx)
        next_open = _open_lookup.get(next_d, {}) if next_d else {}

        for sym, tgt in target_d.items():
            px = today_prices.get(sym)
            if not px or px <= 0:
                continue
            cur = (holdings[sym]["shares"] * today_prices.get(sym, holdings[sym]["entry_px"])
                   if sym in holdings else 0)
            delta = tgt - cur
            if abs(delta) < equity * 0.005:
                continue
            cost = abs(delta) * cost_frac
            trade_count += 1
            if delta > 0 and cash >= delta:
                if sym not in holdings and next_open:
                    buy_px = next_open.get(sym, px)
                    if buy_px is None or np.isnan(buy_px) or buy_px <= 0:
                        buy_px = px
                else:
                    buy_px = px
                shares = (delta - cost) / buy_px
                if sym in holdings:
                    holdings[sym]["shares"] += shares
                else:
                    holdings[sym] = {"shares": shares, "entry_px": buy_px}
                cash -= delta
            elif delta < 0 and sym in holdings:
                sell = min(abs(delta) / px, holdings[sym]["shares"])
                cash += sell * px - cost
                holdings[sym]["shares"] -= sell
                if holdings[sym]["shares"] < 0.01:
                    del holdings[sym]

        equity = cash
        for sym, h in holdings.items():
            equity += h["shares"] * today_prices.get(sym, h["entry_px"])
        port_values.append((date, max(equity, 0)))

    # Compute metrics
    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0:
        years = 1
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd_s = dr[dr < 0].std()
    sortino = dr.mean() / dd_s * np.sqrt(252) if dd_s > 0 else np.nan
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    vol = dr.std() * np.sqrt(252)

    spy = uni.prices["SPY"].reindex(vals.index, method="ffill").dropna()
    spy = spy / spy.iloc[0] * INITIAL_CASH
    spy_cagr = (spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1

    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "vol": vol, "trades": trade_count,
        "final": vals.iloc[-1], "years": years,
        "spy_cagr": spy_cagr, "alpha": cagr - spy_cagr,
    }


def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  WRDS WALK-FORWARD VALIDATION (Survivorship-Bias-Free)")
    log("  Data: CRSP prices + Compustat fundamentals + CRSP SP500 membership")
    log("  Design: 2018-2021 | Test: 2022-2025 (no parameter changes)")
    log("=" * 80)

    # Build WRDS universe (loads CRSP prices, Compustat, SP500 membership)
    log("\n1. Building WRDSUniverse...")
    uni = build_wrds_universe(start="2018-01-01", end="2025-12-31", warmup_days=550)

    # Use open prices from the same data load as close prices (consistent tickers)
    open_prices = uni.open_prices
    log(f"  Open prices: {len(open_prices)} dates × {len(open_prices.columns)} tickers")

    # ── IN-SAMPLE (2018-2021) ────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("  IN-SAMPLE: 2018-2021 (WRDS data)")
    log("=" * 80)

    is_full = run_backtest(uni, "2018-01-01", "2021-12-31", open_prices=open_prices)
    if is_full:
        log(f"\n  CAGR:   {is_full['cagr']:+.1%}")
        log(f"  Sharpe: {is_full['sharpe']:.2f}")
        log(f"  Max DD: {is_full['max_dd']:.1%}")
        log(f"  Alpha:  {is_full['alpha']:+.1%}")

    log("\n  Per year:")
    for year in range(2018, 2022):
        m = run_backtest(uni, f"{year}-01-01", f"{year}-12-31", open_prices=open_prices)
        if m:
            log(f"    {year}: CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:.2f}  "
                f"DD={m['max_dd']:.1%}  Alpha={m['alpha']:+.1%}")

    # ── OUT-OF-SAMPLE (2022-2025) ────────────────────────────────────────
    log("\n" + "=" * 80)
    log("  OUT-OF-SAMPLE: 2022-2025 (WRDS data, NO parameter changes)")
    log("=" * 80)

    oos_full = run_backtest(uni, "2022-01-01", "2025-12-31", open_prices=open_prices)
    if oos_full:
        log(f"\n  CAGR:    {oos_full['cagr']:+.1%}")
        log(f"  Sharpe:  {oos_full['sharpe']:.2f}")
        log(f"  Sortino: {oos_full['sortino']:.2f}")
        log(f"  Max DD:  {oos_full['max_dd']:.1%}")
        log(f"  Vol:     {oos_full['vol']:.1%}")
        log(f"  Alpha:   {oos_full['alpha']:+.1%}")
        log(f"  Trades:  {oos_full['trades']}")
        log(f"  $100K -> ${oos_full['final']:,.0f}")

    log("\n  Per year:")
    oos_yearly = []
    for year in range(2022, 2026):
        m = run_backtest(uni, f"{year}-01-01", f"{year}-12-31", open_prices=open_prices)
        if m:
            m["year"] = year
            oos_yearly.append(m)
            log(f"    {year}: CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:.2f}  "
                f"DD={m['max_dd']:.1%}  Alpha={m['alpha']:+.1%}")

    if oos_yearly:
        med_cagr = np.median([m["cagr"] for m in oos_yearly])
        med_sharpe = np.median([m["sharpe"] for m in oos_yearly])
        pos = sum(1 for m in oos_yearly if m["cagr"] > 0)
        log(f"\n  Median CAGR: {med_cagr:+.1%}  Median Sharpe: {med_sharpe:.2f}  "
            f"Positive: {pos}/{len(oos_yearly)}")

    # ── COMPARISON ───────────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("  HONESTY GAP: WRDS vs FMP/Massive")
    log("  Compare these results to the FMP-based backtest to measure bias.")
    log("=" * 80)

    # Previous FMP-based results (from the last run of true_walkforward.py)
    fmp_oos = {
        "cagr": 0.258, "sharpe": 1.37, "max_dd": -0.214,
        "alpha": 0.258 - 0.15,  # approximate SPY CAGR
    }

    if oos_full:
        log(f"\n  {'Metric':<12} {'FMP/Massive':>14} {'WRDS (honest)':>14} {'Gap':>10}")
        log(f"  {'-'*52}")
        log(f"  {'CAGR':<12} {fmp_oos['cagr']:>+13.1%} {oos_full['cagr']:>+13.1%} "
            f"{oos_full['cagr']-fmp_oos['cagr']:>+9.1%}")
        log(f"  {'Sharpe':<12} {fmp_oos['sharpe']:>14.2f} {oos_full['sharpe']:>14.2f} "
            f"{oos_full['sharpe']-fmp_oos['sharpe']:>+9.2f}")
        log(f"  {'Max DD':<12} {fmp_oos['max_dd']:>13.1%} {oos_full['max_dd']:>13.1%} "
            f"{oos_full['max_dd']-fmp_oos['max_dd']:>+9.1%}")

    if is_full and oos_full:
        log(f"\n  IS vs OOS (within WRDS):")
        log(f"  IS CAGR: {is_full['cagr']:+.1%}  OOS CAGR: {oos_full['cagr']:+.1%}")
        log(f"  IS Sharpe: {is_full['sharpe']:.2f}  OOS Sharpe: {oos_full['sharpe']:.2f}")
        if is_full["sharpe"] > 0:
            retention = oos_full["sharpe"] / is_full["sharpe"]
            log(f"  Sharpe retention: {retention:.2f} "
                f"({'GOOD (>0.5)' if retention > 0.5 else 'OVERFITTING CONCERN'})")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
