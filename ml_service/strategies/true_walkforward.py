"""
True Walk-Forward Validation
============================
1. Design phase: tune strategy on 2018-2021 data ONLY
2. Lock phase: freeze all parameters
3. Test phase: run 2022-2025 without any changes

This gives the honest out-of-sample number.

Usage:
    cd ml_service && python3 -m strategies.true_walkforward
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from massive_data_provider import MassiveDataProvider
from sp500_history import get_sp500_on_date, load_sp500_changes
from strategies.multi_strategy_engine import (
    FastUniverse, strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, strategy4_index_inclusion, strategy6_bear_short,
    STRATEGY_CONFIG_BULL, STRATEGY_CONFIG_BEAR,
    INITIAL_CASH, COST_BPS, SECTOR_ETFS,
)
from strategies.enhanced_features import build_enhanced_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def log(msg):
    print(msg, flush=True)


def run_backtest(uni, start, end):
    """Run the strategy exactly as implemented in multi_strategy_engine."""
    trading_dates = [d for d in sorted(uni.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

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
            for sym in row.dropna().index[:500]:
                today_prices[sym] = row[sym]

        regime = uni.get_regime(date)
        vix = regime.get("vix", 20)

        t1 = strategy1_momentum_reversal(date, uni, day_idx)
        t3 = strategy3_sector_rotation(date, uni, day_idx)
        t4 = strategy4_index_inclusion(date, uni, day_idx, s4_active)
        t5 = strategy5_lowvol_quality(date, uni, day_idx)

        if t1 is not None: last_targets["s1_momentum"] = t1
        if t3 is not None: last_targets["s3_sector"] = t3
        last_targets["s4_inclusion"] = t4 if t4 else {}
        if t5 is not None: last_targets["s5_lowvol"] = t5
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
        breadth = sum(1 for v in dist_sma50.values() if v > 0) / max(len(dist_sma50), 1) if dist_sma50 else 0.5
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
            if name in paused: continue
            for sym, w in last_targets.get(name, {}).items():
                if w > 0:
                    combined[sym] = combined.get(sym, 0) + w * cap_pct

        # Constraints
        longs = {s: w for s, w in combined.items() if w > 0}
        for sym in list(longs):
            if longs[sym] > 0.15: longs[sym] = 0.15
        sec_tot = {}
        for sym, w in longs.items():
            sec = uni.sector_map.get(sym, "X")
            sec_tot[sec] = sec_tot.get(sec, 0) + w
        for sec, tot in sec_tot.items():
            if tot > 0.35:
                scale = 0.35 / tot
                for sym in list(longs):
                    if uni.sector_map.get(sym, "X") == sec: longs[sym] *= scale
        gross = sum(longs.values())
        if gross > 1.0:
            for sym in longs: longs[sym] /= gross
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

        for sym, tgt in target_d.items():
            px = today_prices.get(sym)
            if not px or px <= 0: continue
            cur = holdings[sym]["shares"] * today_prices.get(sym, holdings[sym]["entry_px"]) if sym in holdings else 0
            delta = tgt - cur
            if abs(delta) < equity * 0.005: continue
            cost = abs(delta) * cost_frac
            trade_count += 1
            if delta > 0 and cash >= delta:
                shares = (delta - cost) / px
                if sym in holdings: holdings[sym]["shares"] += shares
                else: holdings[sym] = {"shares": shares, "entry_px": px}
                cash -= delta
            elif delta < 0 and sym in holdings:
                sell = min(abs(delta) / px, holdings[sym]["shares"])
                cash += sell * px - cost
                holdings[sym]["shares"] -= sell
                if holdings[sym]["shares"] < 0.01: del holdings[sym]

        equity = cash
        for sym, h in holdings.items():
            equity += h["shares"] * today_prices.get(sym, h["entry_px"])
        port_values.append((date, max(equity, 0)))

    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0: years = 1
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
    log("  TRUE WALK-FORWARD VALIDATION")
    log("  Design: 2018-2021 | Test: 2022-2025 (no parameter changes)")
    log("=" * 80)

    # Load all data
    log("\n1. Loading data ...")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])

    provider = MassiveDataProvider(validate_vs_yfinance=False)
    all_syms = sorted(features["symbol"].unique().tolist())
    bars = provider.fetch_bars_batch(list(set(all_syms + ["SPY"])), warmup_days=3800)
    close_frames = {sym: df["close"] for sym, df in bars.items() if len(df) > 0}
    prices = pd.DataFrame(close_frames)
    prices.index = pd.to_datetime(prices.index)

    with open(DATA_DIR / "cache_sectors.json") as f:
        sector_map = json.load(f)
    with open(DATA_DIR / "sp500_changes.json") as f:
        sp500_changes = json.load(f)

    # Load cached VIX data for deterministic results
    vix_cache = DATA_DIR / "enhanced_data" / "vix_cache.parquet"
    vix_data = None
    if vix_cache.exists():
        vix_data = pd.read_parquet(vix_cache)
        vix_data.index = pd.to_datetime(vix_data.index)
        log(f"  VIX data: {len(vix_data)} rows (cached)")
    else:
        try:
            vix_raw = yf.download(["^VIX", "^VIX3M"], start="2016-01-01",
                                   end="2027-01-01", progress=False, auto_adjust=True)
            vix_data = vix_raw["Close"]
            vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)
        except Exception:
            pass

    # Load enhanced data
    enhanced_data = {}
    enhanced_dir = DATA_DIR / "enhanced_data"
    if enhanced_dir.exists():
        for fname, key in [("price_targets.parquet", "price_targets"),
                            ("dcf_values.parquet", "dcf"),
                            ("financial_growth.parquet", "financial_growth"),
                            ("enterprise_values.parquet", "enterprise_values"),
                            ("company_profiles.parquet", "profiles"),
                            ("crypto_forex_extended.parquet", "crypto_forex")]:
            fpath = enhanced_dir / fname
            if fpath.exists():
                enhanced_data[key] = pd.read_parquet(fpath)
                if "date" in enhanced_data[key].columns:
                    enhanced_data[key]["date"] = pd.to_datetime(enhanced_data[key]["date"])
        if enhanced_data:
            log(f"  Enhanced data: {', '.join(enhanced_data.keys())}")

    uni = FastUniverse(features, prices, sector_map, sp500_changes, vix_data,
                       enhanced_data=enhanced_data)

    # ── PHASE 1: In-sample design period (2018-2021) ──
    log("\n" + "=" * 80)
    log("  PHASE 1: IN-SAMPLE DESIGN (2018-2021)")
    log("  Strategy parameters were tuned during this period.")
    log("=" * 80)

    is_full = run_backtest(uni, "2018-01-01", "2021-12-31")
    if is_full:
        log(f"\n  In-sample full period:")
        log(f"    CAGR:   {is_full['cagr']:+.1%}")
        log(f"    Sharpe: {is_full['sharpe']:.2f}")
        log(f"    Max DD: {is_full['max_dd']:.1%}")
        log(f"    Alpha:  {is_full['alpha']:+.1%}")

    log("\n  In-sample per year:")
    for year in range(2018, 2022):
        m = run_backtest(uni, f"{year}-01-01", f"{year}-12-31")
        if m:
            log(f"    {year}: CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:.2f}  "
                f"DD={m['max_dd']:.1%}  Alpha={m['alpha']:+.1%}")

    # ── PHASE 2: Out-of-sample test (2022-2025) ──
    log("\n" + "=" * 80)
    log("  PHASE 2: OUT-OF-SAMPLE TEST (2022-2025)")
    log("  NO parameter changes allowed. Exact same code as in-sample.")
    log("=" * 80)

    oos_full = run_backtest(uni, "2022-01-01", "2025-12-31")
    if oos_full:
        log(f"\n  Out-of-sample full period:")
        log(f"    CAGR:    {oos_full['cagr']:+.1%}")
        log(f"    Sharpe:  {oos_full['sharpe']:.2f}")
        log(f"    Sortino: {oos_full['sortino']:.2f}")
        log(f"    Max DD:  {oos_full['max_dd']:.1%}")
        log(f"    Vol:     {oos_full['vol']:.1%}")
        log(f"    Alpha:   {oos_full['alpha']:+.1%}")
        log(f"    Trades:  {oos_full['trades']}")
        log(f"    $100K -> ${oos_full['final']:,.0f}")

    log("\n  Out-of-sample per year:")
    oos_yearly = []
    for year in range(2022, 2026):
        m = run_backtest(uni, f"{year}-01-01", f"{year}-12-31")
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

    # ── PHASE 3: Comparison ──
    log("\n" + "=" * 80)
    log("  COMPARISON: IN-SAMPLE vs OUT-OF-SAMPLE")
    log("=" * 80)

    if is_full and oos_full:
        log(f"\n  {'Metric':<15} {'In-Sample':>12} {'Out-of-Sample':>15} {'Degradation':>12}")
        log(f"  {'─'*15} {'─'*12} {'─'*15} {'─'*12}")
        for metric, fmt in [("cagr", "+.1%"), ("sharpe", ".2f"), ("max_dd", ".1%"), ("alpha", "+.1%")]:
            is_val = is_full[metric]
            oos_val = oos_full[metric]
            if metric in ("cagr", "alpha"):
                degrad = oos_val - is_val
                log(f"  {metric:<15} {is_val:>11{fmt}} {oos_val:>14{fmt}} {degrad:>11{fmt}}")
            else:
                log(f"  {metric:<15} {is_val:>11{fmt}} {oos_val:>14{fmt}}")

        # Overfitting ratio
        if is_full["cagr"] > 0 and oos_full["cagr"] > 0:
            overfit_ratio = oos_full["sharpe"] / is_full["sharpe"] if is_full["sharpe"] > 0 else 0
            log(f"\n  Sharpe retention ratio: {overfit_ratio:.2f} "
                f"({'GOOD (>0.5)' if overfit_ratio > 0.5 else 'OVERFITTING CONCERN (<0.5)'})")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
