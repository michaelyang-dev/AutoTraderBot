"""
Research: Alternative momentum constructions
=============================================
Tests different ways to score/select momentum stocks.
Uses the backtester directly but with custom momentum scoring.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
import copy
from fast_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy3_sector_rotation, strategy5_lowvol_quality,
    INITIAL_CASH, COST_BPS,
)


def custom_momentum(date, uni, day_idx, top_n=8, rebal_days=15, variant="baseline"):
    """Momentum strategy with different scoring variants."""
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)
    if len(members) < 50:
        return {}

    ret_20 = uni.get_feature_map(date, "ret_20d", members)
    ret_252 = uni.get_feature_map(date, "ret_252d", members)
    ret_126 = uni.get_feature_map(date, "ret_126d", members)
    ret_60 = uni.get_feature_map(date, "ret_60d", members)
    vol_60 = uni.get_feature_map(date, "vol_60d", members)
    dist_sma200 = uni.get_feature_map(date, "dist_sma200", members)
    eps_surp = uni.get_feature_map(date, "eps_surprise_last", members)
    roe_map = uni.get_feature_map(date, "roe")

    composite = {}
    for sym in members:
        r252 = ret_252.get(sym)
        r20 = ret_20.get(sym)
        r126 = ret_126.get(sym)
        r60 = ret_60.get(sym)
        v60 = vol_60.get(sym)

        if r252 is None or r20 is None or np.isnan(r252) or np.isnan(r20):
            continue

        # ── Momentum score variant ─────────────
        if variant == "baseline":
            # Standard 12-1 momentum
            score = r252 - r20

        elif variant == "6month":
            # 6-month momentum (skip last month)
            if r126 is None or np.isnan(r126):
                continue
            score = r126 - r20

        elif variant == "risk_adjusted":
            # Risk-adjusted momentum: (12-1 ret) / volatility
            if v60 is None or np.isnan(v60) or v60 < 0.01:
                continue
            score = (r252 - r20) / v60

        elif variant == "dual_momentum":
            # Dual momentum: average of 6m and 12-1m
            if r126 is None or np.isnan(r126):
                continue
            score = 0.5 * (r252 - r20) + 0.5 * (r126 - r20)

        elif variant == "acceleration":
            # Momentum acceleration: recent momentum > long-term momentum
            if r126 is None or r60 is None or np.isnan(r126) or np.isnan(r60):
                continue
            score = r252 - r20
            # Boost stocks where recent momentum is accelerating
            long_rate = (r252 - r20) / 12  # monthly rate
            short_rate = r60 / 3  # monthly rate
            if short_rate > long_rate * 1.2:  # accelerating
                score *= 1.20

        elif variant == "composite_rank":
            # Rank-based composite: average ranks of 3-month, 6-month, 12-1 month
            # (simplified: use raw scores as proxies, rank at the end)
            if r126 is None or r60 is None or np.isnan(r126) or np.isnan(r60):
                continue
            score = (r252 - r20) + r126 + r60  # sum of multiple lookbacks

        elif variant == "momentum_quality":
            # Momentum + quality gate: only high-quality momentum
            score = r252 - r20
            roe_val = roe_map.get(sym)
            if roe_val is None or np.isnan(roe_val) or roe_val < 0.10:
                continue  # skip low quality entirely
            # Weight by quality
            score *= (1 + min(roe_val, 0.40))

        elif variant == "vol_scaled":
            # Volatility-scaled position sizing built into scoring
            if v60 is None or np.isnan(v60) or v60 < 0.01:
                continue
            raw_mom = r252 - r20
            # Target: higher allocation to lower-vol momentum stocks
            score = raw_mom * (0.20 / v60)  # scale by inverse vol

        else:
            score = r252 - r20

        # ── Quality boosts (same as production) ─────
        es = eps_surp.get(sym)
        if es is not None and not np.isnan(es) and es > 0:
            score *= 1.15

        roe_val = roe_map.get(sym)
        if roe_val is not None and not np.isnan(roe_val) and roe_val > 0.15:
            score *= 1.05

        # Trend filter: above 200d SMA
        d200 = dist_sma200.get(sym, 0)
        if d200 is not None and d200 > 0:
            composite[sym] = score

    if not composite:
        return {}
    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:top_n]
    scores = [max(composite[s], 0.001) for s in sorted_syms]
    total = sum(scores)
    if total > 0:
        weights = {s: min(sc / total, 2.0 / len(sorted_syms)) for s, sc in zip(sorted_syms, scores)}
        wt = sum(weights.values())
        if wt > 0:
            weights = {s: w / wt for s, w in weights.items()}
        return weights
    return {s: 1.0 / len(sorted_syms) for s in sorted_syms}


def run_momentum_variant(bt, start, end, variant, mom_w=0.35, val_w=0.25, lv_w=0.40,
                          top_n=8, rebal_days=15, trailing_stop=0.35, cap=0.25):
    """Run backtest with a specific momentum variant."""
    trading_dates = [d for d in sorted(bt.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

    cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    last_targets = {}

    for day_idx, date in enumerate(trading_dates):
        today = {}
        if date in bt.prices.index:
            row = bt.prices.loc[date]
            for sym in list(holdings.keys()):
                v = row.get(sym)
                if v is not None and not np.isnan(v):
                    today[sym] = v
            for sym in row.dropna().index:
                today[sym] = row[sym]

        # Trailing stop
        if trailing_stop:
            for sym in list(holdings):
                px = today.get(sym)
                if px:
                    if "peak_px" not in holdings[sym]:
                        holdings[sym]["peak_px"] = px
                    if px > holdings[sym]["peak_px"]:
                        holdings[sym]["peak_px"] = px
                    dd = (px - holdings[sym]["peak_px"]) / holdings[sym]["peak_px"]
                    if dd < -abs(trailing_stop):
                        cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                        del holdings[sym]

        total_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                               for s, h in holdings.items())

        if day_idx % rebal_days != 0:
            port_values.append((date, total_val))
            continue

        # Strategy signals
        t1 = custom_momentum(date, bt.uni, day_idx, top_n=top_n,
                              rebal_days=rebal_days, variant=variant)
        if t1 is None:
            t1 = last_targets.get("mom", {})

        members = bt.uni.get_sp500(date)
        t_val = bt._strategy_value(date, members, top_n=10)
        t5 = strategy5_lowvol_quality(date, bt.uni, day_idx)
        if t5 is None:
            t5 = last_targets.get("s5", {})
        last_targets.update({"mom": t1, "val": t_val, "s5": t5})

        # UMD regime
        nu = bt.umd_20d.loc[:date]
        in_crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < -0.05
        if in_crash:
            ew = {"mom": 0.15, "val": 0.45, "s5": 0.30}
        else:
            ew = {"mom": mom_w, "val": val_w, "s5": lv_w}

        # Breadth blending
        fdate = bt.features_by_date.get(date, {})
        above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
        total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
        breadth = above / max(total_f, 1)
        blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))
        bear = {"mom": 0.10, "val": 0.20, "s5": 0.60}
        blended = {n: ew[n] * blend + bear.get(n, 0) * (1 - blend) for n in ew}

        combined = {}
        for name, cap_pct in blended.items():
            tgt = last_targets.get(name, {})
            for sym, w in tgt.items():
                if w > 0:
                    combined[sym] = combined.get(sym, 0) + w * cap_pct

        longs = {s: w for s, w in combined.items() if w > 0}
        for sym in list(longs):
            if longs[sym] > cap:
                longs[sym] = cap
        gross = sum(longs.values())
        if gross > 1.0:
            for sym in longs:
                longs[sym] /= gross
        combined = {s: w for s, w in longs.items() if w >= 0.005}

        # Trend filter
        eq_pct = 1.0
        if "SPY" in bt.prices.columns:
            spy_px = today.get("SPY", 0)
            spy_hist = bt.prices["SPY"].loc[:date].dropna()
            if len(spy_hist) >= 200:
                spy_sma200 = spy_hist.tail(200).mean()
                spy_sma50 = spy_hist.tail(50).mean()
                if spy_px < spy_sma200:
                    eq_pct *= 0.50
                elif spy_px < spy_sma50:
                    eq_pct *= 0.75

        target_d = {s: w * total_val * eq_pct for s, w in combined.items()}

        for sym in list(holdings):
            if sym not in target_d:
                px = today.get(sym, holdings[sym]["entry_px"])
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                del holdings[sym]

        for sym, tgt in target_d.items():
            px = today.get(sym)
            if not px or px <= 0:
                continue
            cur_val = holdings[sym]["shares"] * px if sym in holdings else 0
            diff = tgt - cur_val
            if abs(diff) < total_val * 0.01:
                continue
            shares_delta = diff / px
            if sym in holdings:
                holdings[sym]["shares"] += shares_delta
                cost = abs(shares_delta * px) * cost_frac
                cash -= shares_delta * px + cost
            else:
                if shares_delta > 0:
                    cost = shares_delta * px * cost_frac
                    holdings[sym] = {"shares": shares_delta, "entry_px": px}
                    cash -= shares_delta * px + cost

        port_values.append((date, total_val))

    if len(port_values) < 2:
        return None

    dates = [v[0] for v in port_values]
    vals = [v[1] for v in port_values]
    rets = pd.Series(vals).pct_change().dropna()
    n_years = (dates[-1] - dates[0]).days / 365.25
    cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    peak = pd.Series(vals).cummax()
    dd = (pd.Series(vals) - peak) / peak
    max_dd = dd.min()

    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}


def main():
    print("Loading universe...")
    t0 = time.time()
    # Use smaller universe for speed (2016-2025 only)
    universe_path = "data/wrds/complete_sp1500_universe.pkl"
    bt = FastBacktester(universe_path)
    print(f"Loaded in {time.time()-t0:.1f}s")

    variants = [
        "baseline",
        "6month",
        "risk_adjusted",
        "dual_momentum",
        "acceleration",
        "composite_rank",
        "momentum_quality",
        "vol_scaled",
    ]

    # ═══════════════════════════════════════════════════════════
    # Test on 2018-2025
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("Momentum Variants: 2018-2025 (v11 config: 35/25/40)")
    print("="*70)

    print(f"\n{'Variant':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*55)

    results = {}
    for v in variants:
        r = run_momentum_variant(bt, "2018-01-01", "2025-12-31", v,
                                  mom_w=0.35, val_w=0.25, lv_w=0.40,
                                  top_n=8, rebal_days=15, trailing_stop=0.35)
        if r:
            results[v] = r
            print(f"{v:<25} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Test on 25-year
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("Momentum Variants: 2001-2025 (v11 config)")
    print("="*70)

    print(f"\n{'Variant':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*55)

    for v in variants:
        r = run_momentum_variant(bt, "2001-01-01", "2025-12-31", v,
                                  mom_w=0.35, val_w=0.25, lv_w=0.40,
                                  top_n=8, rebal_days=15, trailing_stop=0.35)
        if r:
            print(f"{v:<25} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Test best variant with higher momentum weight
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("Best Variant with 50/35/15 weights")
    print("="*70)

    best_variant = max(results, key=lambda v: results[v]["sharpe"]) if results else "baseline"
    print(f"Best variant from 2018: {best_variant}")

    for v in [best_variant, "baseline"]:
        r = run_momentum_variant(bt, "2018-01-01", "2025-12-31", v,
                                  mom_w=0.50, val_w=0.35, lv_w=0.15,
                                  top_n=5, rebal_days=20, trailing_stop=None)
        if r:
            print(f"  {v} (50/35/15 n5 rd20): {r['cagr']*100:.1f}% CAGR, {r['sharpe']:.2f} Sharpe, {r['max_dd']*100:.1f}% MaxDD")

    # ═══════════════════════════════════════════════════════════
    # Start-day averaged for top 2 variants (2018-2025)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("Start-day averaged (top 2 + baseline)")
    print("="*70)

    sorted_v = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    top_variants = [v for v, _ in sorted_v[:2]]
    if "baseline" not in top_variants:
        top_variants.append("baseline")

    for v in top_variants:
        cagrs = []
        for offset in range(5):
            start_dt = pd.Timestamp("2018-01-01") + pd.Timedelta(days=offset)
            r = run_momentum_variant(bt, str(start_dt.date()), "2025-12-31", v,
                                      mom_w=0.35, val_w=0.25, lv_w=0.40,
                                      top_n=8, rebal_days=15, trailing_stop=0.35)
            if r and r["cagr"]:
                cagrs.append(r["cagr"])
        if cagrs:
            avg_cagr = np.mean(cagrs)
            std_cagr = np.std(cagrs)
            print(f"  {v}: avg CAGR = {avg_cagr*100:.1f}% +/- {std_cagr*100:.1f}%")

    print("\n" + "="*70)
    print("MOMENTUM RESEARCH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
