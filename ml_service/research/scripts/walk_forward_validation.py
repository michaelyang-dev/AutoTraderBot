"""
Walk-Forward Validation — Honest OOS Performance Assessment
=============================================================
Tests whether the skip-month momentum improvement is genuine by running
multiple non-overlapping IS/OOS windows. No parameter optimization within
windows — same parameters throughout.

Walk-forward protocol:
  - 3-year IS → 1-year OOS, rolling by 1 year
  - Parameters FIXED across all windows (no re-optimization)
  - Report each OOS year independently
  - Compute aggregate OOS statistics

This avoids the "lucky OOS period" problem by testing across ALL years.
"""

import numpy as np
import pandas as pd
import time
import logging
from functools import partial
from fast_backtest import FastBacktester
from scoring_variants import strategy1_skip_month
from strategies.multi_strategy_engine import strategy1_momentum_reversal

logging.basicConfig(level=logging.WARNING)


def monkey_patch_run(bt, mom_func, start, end, config):
    """Run with swapped momentum function."""
    import strategies.multi_strategy_engine as mse
    import fast_backtest as fb
    orig_mse = mse.strategy1_momentum_reversal
    orig_fb = fb.strategy1_momentum_reversal
    mse.strategy1_momentum_reversal = mom_func
    fb.strategy1_momentum_reversal = mom_func
    try:
        result = bt.run(start, end, config)
    finally:
        mse.strategy1_momentum_reversal = orig_mse
        fb.strategy1_momentum_reversal = orig_fb
    return result


if __name__ == "__main__":
    print("Loading data...", flush=True)
    bt = FastBacktester()
    print("Data loaded.\n", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # CONFIGS TO VALIDATE
    # ═══════════════════════════════════════════════════════════════
    # Config A: Best OOS from integrated sweep (with regime blending)
    config_A = {
        "universe": "sp1500", "mom_w": 0.85, "val_w": 0.15,
        "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25, "cap": 0.15,
    }
    func_A = partial(strategy1_skip_month, trend_filter="sma200", quality_boosts=True)

    # Config B: Pure momentum, no regime blending (highest raw OOS)
    config_B = {
        "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0,
        "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 10, "rebal_days": 10, "trailing_stop": 0.25, "cap": 0.15,
    }
    func_B = partial(strategy1_skip_month, trend_filter="sma200", quality_boosts=False)

    # Config C: Pure momentum, no regime, no trailing stop (max aggression)
    config_C = {
        "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0,
        "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 10, "rebal_days": 10, "trailing_stop": None, "cap": 0.15,
    }
    func_C = partial(strategy1_skip_month, trend_filter="sma200", quality_boosts=False)

    # Config D: Pure momentum, SMA200, t8, no regime
    config_D = {
        "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0,
        "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25, "cap": 0.15,
    }
    func_D = partial(strategy1_skip_month, trend_filter="sma200", quality_boosts=True)

    # Config E: Original production scoring (baseline)
    config_E = {
        "universe": "sp1500", "mom_w": 0.85, "val_w": 0.15,
        "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25, "cap": 0.15,
        "gld_pct": 0.02, "vixm_pct": 0.02,
    }
    func_E = strategy1_momentum_reversal  # Uses the NEW skip-month code

    configs = [
        ("A: skip12-1 sma200 85/15 t8 r10 qual", func_A, config_A),
        ("B: skip12-1 sma200 100/0 t10 r10 noQ", func_B, config_B),
        ("C: skip12-1 sma200 100/0 t10 r10 noQ noTrail", func_C, config_C),
        ("D: skip12-1 sma200 100/0 t8 r10 qual", func_D, config_D),
        ("E: PRODUCTION (new skip-month) 85/15 G+V", func_E, config_E),
    ]

    # ═══════════════════════════════════════════════════════════════
    # WALK-FORWARD: 1-year OOS windows
    # ═══════════════════════════════════════════════════════════════
    print("=" * 120, flush=True)
    print("WALK-FORWARD VALIDATION: Each year tested independently (NO parameter changes between windows)", flush=True)
    print("=" * 120, flush=True)

    oos_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    for name, func, config in configs:
        print(f"\n  {name}", flush=True)
        print(f"  {'Year':<6} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7}", flush=True)
        print(f"  {'-'*30}", flush=True)

        yearly_returns = []
        yearly_sharpes = []

        for year in oos_years:
            start = f"{year}-01-01"
            end = f"{year}-12-31" if year < 2025 else "2025-04-30"

            result = monkey_patch_run(bt, func, start, end, config)
            if result and result.get("yearly", {}).get(year):
                yr = result["yearly"][year]
                cagr = yr["cagr"]
                sharpe = yr["sharpe"]
                dd = yr["max_dd"]
                yearly_returns.append(cagr)
                yearly_sharpes.append(sharpe)
                print(f"  {year:<6} {cagr:>+6.1%} {sharpe:>7.2f} {dd:>6.0%}", flush=True)
            else:
                print(f"  {year:<6} {'N/A':>7}", flush=True)

        if yearly_returns:
            # Compute aggregate statistics
            avg_return = np.mean(yearly_returns)
            med_return = np.median(yearly_returns)
            geo_mean = np.exp(np.mean(np.log([1 + r for r in yearly_returns]))) - 1
            hit_rate = sum(1 for r in yearly_returns if r > 0) / len(yearly_returns)
            worst = min(yearly_returns)
            best = max(yearly_returns)
            avg_sharpe = np.mean(yearly_sharpes)

            print(f"  {'─'*30}", flush=True)
            print(f"  Geometric mean annual return: {geo_mean:+.1%}", flush=True)
            print(f"  Arithmetic mean:              {avg_return:+.1%}", flush=True)
            print(f"  Median:                       {med_return:+.1%}", flush=True)
            print(f"  Best/Worst year:              {best:+.0%} / {worst:+.0%}", flush=True)
            print(f"  Hit rate (positive years):    {hit_rate:.0%} ({sum(1 for r in yearly_returns if r > 0)}/{len(yearly_returns)})", flush=True)
            print(f"  Avg annual Sharpe:            {avg_sharpe:.2f}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # FULL SAMPLE RUNS (for context)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 120}", flush=True)
    print("FULL SAMPLE RESULTS (2018-01 to 2025-04)", flush=True)
    print("=" * 120, flush=True)

    for name, func, config in configs:
        t0 = time.time()
        full = monkey_patch_run(bt, func, "2018-01-01", "2025-04-30", config)
        oos = monkey_patch_run(bt, func, "2022-01-01", "2025-04-30", config)
        elapsed = time.time() - t0

        if full and oos:
            print(f"  {name}", flush=True)
            print(f"    Full: CAGR={full['cagr']:+.1%} Sharpe={full['sharpe']:.2f} DD={full['max_dd']:.0%}", flush=True)
            print(f"    OOS:  CAGR={oos['cagr']:+.1%} Sharpe={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}", flush=True)
            yrs = " ".join(f"{y}:{full['yearly'][y]['cagr']:+.0%}" for y in sorted(full['yearly'].keys()))
            print(f"    Yearly: {yrs}", flush=True)
            print(f"    [{elapsed:.0f}s]", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # ROLLING 3-YEAR IS → 1-YEAR OOS WALK-FORWARD
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 120}", flush=True)
    print("ROLLING WALK-FORWARD: 3yr IS → 1yr OOS (verifies no look-ahead)", flush=True)
    print("=" * 120, flush=True)

    # For the best config, check IS→OOS Sharpe decay
    for name, func, config in configs[:4]:
        print(f"\n  {name}", flush=True)
        print(f"  {'Window':<20} {'IS CAGR':>8} {'IS Shrp':>8} {'OOS CAGR':>9} {'OOS Shrp':>9} {'Decay':>6}", flush=True)
        print(f"  {'-'*65}", flush=True)

        decays = []
        for oos_year in range(2021, 2026):
            is_start = f"{oos_year-3}-01-01"
            is_end = f"{oos_year-1}-12-31"
            oos_start = f"{oos_year}-01-01"
            oos_end = f"{oos_year}-12-31" if oos_year < 2025 else "2025-04-30"

            is_result = monkey_patch_run(bt, func, is_start, is_end, config)
            oos_result = monkey_patch_run(bt, func, oos_start, oos_end, config)

            if is_result and oos_result:
                is_sharpe = is_result["sharpe"]
                oos_sharpe = oos_result["sharpe"]
                is_cagr = is_result["cagr"]
                oos_cagr = oos_result["cagr"]
                decay = (oos_sharpe - is_sharpe) / abs(is_sharpe) if is_sharpe != 0 else 0
                decays.append(decay)

                window = f"{oos_year-3}-{oos_year-1}→{oos_year}"
                print(f"  {window:<20} {is_cagr:>+7.1%} {is_sharpe:>8.2f} {oos_cagr:>+8.1%} {oos_sharpe:>9.2f} {decay:>+5.0%}", flush=True)

        if decays:
            avg_decay = np.mean(decays)
            print(f"  {'─'*65}", flush=True)
            print(f"  Average IS→OOS Sharpe decay: {avg_decay:+.0%}", flush=True)

    print(f"\n\nDone.", flush=True)
