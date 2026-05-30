"""
Research: Sleeve-level improvements
====================================
Tests modifications to the value sleeve and portfolio construction
that could improve the overall strategy without look-ahead bias.

All tests use 25-year data (2001-2025) and start-day averaging for honest numbers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
from fast_backtest import FastBacktester

def run_avg(bt, start, end, config, n_offsets=5):
    """Run start-day averaged backtest for honest numbers.
    Shifts start date by 1-4 trading days to average out start-day luck."""
    cagrs, sharpes, dds = [], [], []
    start_dt = pd.Timestamp(start)
    for offset in range(n_offsets):
        shifted_start = start_dt + pd.Timedelta(days=offset)
        r = bt.run(str(shifted_start.date()), end, config)
        if r and r.get("cagr"):
            cagrs.append(r["cagr"])
            sharpes.append(r["sharpe"])
            dds.append(r["max_dd"])
    if not cagrs:
        return None
    return {
        "cagr": np.mean(cagrs),
        "sharpe": np.mean(sharpes),
        "max_dd": np.mean(dds),
        "cagr_std": np.std(cagrs),
        "n": len(cagrs),
    }

def run_single(bt, start, end, config):
    """Run single backtest (faster, for screening)."""
    r = bt.run(start, end, config)
    if r:
        return {"cagr": r.get("cagr", 0), "sharpe": r.get("sharpe", 0), "max_dd": r.get("max_dd", 0)}
    return None

def main():
    print("Loading universe (this takes ~30s)...")
    t0 = time.time()

    # Try 2000 universe first, fall back to complete
    universe_path = "data/wrds/sp1500_universe_2000.pkl"
    if not os.path.exists(universe_path):
        universe_path = "data/wrds/complete_sp1500_universe.pkl"

    bt = FastBacktester(universe_path)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # BASELINE: Current v11 config
    # ═══════════════════════════════════════════════════════════
    baseline = {
        "mom_w": 0.35, "val_w": 0.25, "lv_w": 0.40, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 15, "trailing_stop": 0.35,
        "cap": 0.25, "use_rp": False,
        "trend_scale": {"bear": 0.50, "caution": 0.75},
    }

    # Best candidate from prior research
    best_candidate = {
        "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": None,
        "cap": 0.25, "use_rp": False,
        "trend_scale": {"bear": 0.50, "caution": 0.75},
    }

    # ═══════════════════════════════════════════════════════════
    # TEST CONFIGS
    # ═══════════════════════════════════════════════════════════
    tests = {}

    # --- Value sleeve weight experiments ---
    # Test: pure momentum + trend filter (no value/lowvol at all)
    tests["pure_mom_trend"] = {
        "mom_w": 1.0, "val_w": 0.0, "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 15, "trailing_stop": None,
        "cap": 0.25, "use_rp": False,
        "trend_scale": {"bear": 0.50, "caution": 0.75},
    }

    # Test: 70/30 momentum/value (drop lowvol entirely)
    tests["mom70_val30"] = {
        "mom_w": 0.70, "val_w": 0.30, "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 15, "trailing_stop": None,
        "cap": 0.25, "use_rp": False,
        "trend_scale": {"bear": 0.50, "caution": 0.75},
    }

    # Test: 60/40 mom/value
    tests["mom60_val40"] = {
        "mom_w": 0.60, "val_w": 0.40, "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 15, "trailing_stop": None,
        "cap": 0.25, "use_rp": False,
        "trend_scale": {"bear": 0.50, "caution": 0.75},
    }

    # Test: 80/20 mom/lowvol (no value at all)
    tests["mom80_lv20"] = {
        "mom_w": 0.80, "val_w": 0.0, "lv_w": 0.20, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 15, "trailing_stop": None,
        "cap": 0.25, "use_rp": False,
        "trend_scale": {"bear": 0.50, "caution": 0.75},
    }

    # --- Risk parity experiments ---
    # Test: risk parity weighting ON
    tests["baseline_rp"] = {
        **baseline, "use_rp": True, "rp_power": 1.0,
    }

    # Test: best candidate + risk parity
    tests["best_rp"] = {
        **best_candidate, "use_rp": True, "rp_power": 1.0,
    }

    # --- Position count experiments ---
    # Test: 10 positions (more diversified)
    tests["n10_rd15"] = {
        **baseline, "top_n": 10,
    }

    # Test: 12 positions
    tests["n12_rd15"] = {
        **baseline, "top_n": 12,
    }

    # Test: 15 positions
    tests["n15_rd20"] = {
        **baseline, "top_n": 15, "rebal_days": 20,
    }

    # --- Cap experiments ---
    # Test: tighter cap (15%)
    tests["cap15"] = {
        **baseline, "cap": 0.15,
    }

    # Test: no cap
    tests["nocap"] = {
        **baseline, "cap": 1.0,
    }

    # --- Trailing stop experiments ---
    # Test: 25% stop (tighter)
    tests["stop25"] = {
        **baseline, "trailing_stop": 0.25,
    }

    # Test: no stop
    tests["nostop"] = {
        **baseline, "trailing_stop": None,
    }

    # --- Trend filter variants ---
    # Test: more aggressive bear reduction (30% instead of 50%)
    tests["bear30"] = {
        **baseline, "trend_scale": {"bear": 0.30, "caution": 0.60},
    }

    # Test: no trend filter at all (always 100%)
    tests["no_trend"] = {
        **baseline, "trend_scale": None,
    }

    # --- Combined best ideas ---
    # Aggressive concentrated momentum
    tests["aggressive_mom"] = {
        "mom_w": 0.70, "val_w": 0.20, "lv_w": 0.10, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 15, "trailing_stop": None,
        "cap": 0.30, "use_rp": False,
        "trend_scale": {"bear": 0.50, "caution": 0.75},
    }

    # Moderate diversified
    tests["moderate_diverse"] = {
        "mom_w": 0.45, "val_w": 0.30, "lv_w": 0.25, "sec_w": 0.0,
        "top_n": 10, "rebal_days": 15, "trailing_stop": 0.35,
        "cap": 0.20, "use_rp": True, "rp_power": 1.0,
        "trend_scale": {"bear": 0.50, "caution": 0.75},
    }

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: Quick screen on 2018-2025 (single run)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 1: Quick screen (2018-2025, single run)")
    print("="*70)

    results_2018 = {}

    print(f"\n{'Config':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*55)

    # Baseline first
    r = run_single(bt, "2018-01-01", "2025-12-31", baseline)
    if r:
        results_2018["BASELINE_v11"] = r
        print(f"{'BASELINE_v11':<25} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    r = run_single(bt, "2018-01-01", "2025-12-31", best_candidate)
    if r:
        results_2018["BEST_PRIOR"] = r
        print(f"{'BEST_PRIOR':<25} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    for name, cfg in tests.items():
        r = run_single(bt, "2018-01-01", "2025-12-31", cfg)
        if r:
            results_2018[name] = r
            print(f"{name:<25} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: 25-year screen for top candidates
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 2: 25-year screen (2001-2025, single run)")
    print("="*70)

    # Pick top 8 configs by Sharpe on 2018-2025
    sorted_configs = sorted(results_2018.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    top_names = [name for name, _ in sorted_configs[:8]]

    all_configs = {"BASELINE_v11": baseline, "BEST_PRIOR": best_candidate, **tests}

    print(f"\n{'Config':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*55)

    results_25yr = {}
    for name in top_names:
        cfg = all_configs[name]
        r = run_single(bt, "2001-01-01", "2025-12-31", cfg)
        if r:
            results_25yr[name] = r
            print(f"{name:<25} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Start-day averaged for top 3
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 3: Start-day averaged (top 3 by 25yr Sharpe)")
    print("="*70)

    sorted_25yr = sorted(results_25yr.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    top3 = [name for name, _ in sorted_25yr[:3]]
    # Always include baseline
    if "BASELINE_v11" not in top3:
        top3.append("BASELINE_v11")

    print(f"\n{'Config':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'CAGR_std':>10}")
    print("-"*65)

    for name in top3:
        cfg = all_configs[name]

        # 2018-2025 averaged
        r18 = run_avg(bt, "2018-01-01", "2025-12-31", cfg, n_offsets=5)
        if r18:
            print(f"{name+' (2018)':<25} {r18['cagr']*100:>7.1f}% {r18['sharpe']:>8.2f} {r18['max_dd']*100:>7.1f}% {r18['cagr_std']*100:>9.1f}%")

        # 25-year averaged
        r25 = run_avg(bt, "2001-01-01", "2025-12-31", cfg, n_offsets=5)
        if r25:
            print(f"{name+' (25yr)':<25} {r25['cagr']*100:>7.1f}% {r25['sharpe']:>8.2f} {r25['max_dd']*100:>7.1f}% {r25['cagr_std']*100:>9.1f}%")

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: Sub-period analysis for the best
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PHASE 4: Sub-period robustness")
    print("="*70)

    best_name = sorted_25yr[0][0] if sorted_25yr else "BASELINE_v11"
    best_cfg = all_configs[best_name]

    periods = [
        ("2001-2007", "2001-01-01", "2007-12-31"),
        ("2008-2009", "2008-01-01", "2009-12-31"),
        ("2010-2015", "2010-01-01", "2015-12-31"),
        ("2016-2019", "2016-01-01", "2019-12-31"),
        ("2020-2021", "2020-01-01", "2021-12-31"),
        ("2022-2025", "2022-01-01", "2025-12-31"),
    ]

    print(f"\nBest config: {best_name}")
    print(f"{'Period':<15} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*45)

    for label, s, e in periods:
        r = run_single(bt, s, e, best_cfg)
        if r:
            print(f"{label:<15} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # Same for baseline
    print(f"\nBaseline (v11):")
    print(f"{'Period':<15} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*45)

    for label, s, e in periods:
        r = run_single(bt, s, e, baseline)
        if r:
            print(f"{label:<15} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    print("\n" + "="*70)
    print("RESEARCH COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
