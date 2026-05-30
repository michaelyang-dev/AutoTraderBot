"""
Research: Sleeve Attribution
============================
Tests each sleeve in ISOLATION to understand which sleeves
actually contribute to returns.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
from fast_backtest import FastBacktester

def main():
    print("Loading universe...")
    t0 = time.time()
    # Use smaller universe for speed
    universe_path = "data/wrds/complete_sp1500_universe.pkl"
    bt = FastBacktester(universe_path)
    print(f"Loaded in {time.time()-t0:.1f}s")

    configs = {
        # Baseline
        "BASELINE (35/25/40)": {"mom_w": 0.35, "val_w": 0.25, "lv_w": 0.40, "sec_w": 0.0,
                                 "top_n": 8, "rebal_days": 15, "trailing_stop": 0.35,
                                 "cap": 0.25, "use_rp": False,
                                 "trend_scale": {"bear": 0.50, "caution": 0.75}},

        # Pure sleeves
        "MOM ONLY (100/0/0)": {"mom_w": 1.0, "val_w": 0.0, "lv_w": 0.0, "sec_w": 0.0,
                                "top_n": 8, "rebal_days": 15, "trailing_stop": 0.35,
                                "cap": 0.25, "use_rp": False,
                                "trend_scale": {"bear": 0.50, "caution": 0.75}},

        "VAL ONLY (0/100/0)": {"mom_w": 0.0, "val_w": 1.0, "lv_w": 0.0, "sec_w": 0.0,
                                "top_n": 8, "rebal_days": 15, "trailing_stop": 0.35,
                                "cap": 0.25, "use_rp": False,
                                "trend_scale": {"bear": 0.50, "caution": 0.75}},

        "LV ONLY (0/0/100)":  {"mom_w": 0.0, "val_w": 0.0, "lv_w": 1.0, "sec_w": 0.0,
                                "top_n": 8, "rebal_days": 15, "trailing_stop": 0.35,
                                "cap": 0.25, "use_rp": False,
                                "trend_scale": {"bear": 0.50, "caution": 0.75}},

        # Two-sleeve combos
        "MOM+VAL (60/40/0)":  {"mom_w": 0.60, "val_w": 0.40, "lv_w": 0.0, "sec_w": 0.0,
                                "top_n": 8, "rebal_days": 15, "trailing_stop": 0.35,
                                "cap": 0.25, "use_rp": False,
                                "trend_scale": {"bear": 0.50, "caution": 0.75}},

        "MOM+LV (60/0/40)":  {"mom_w": 0.60, "val_w": 0.0, "lv_w": 0.40, "sec_w": 0.0,
                                "top_n": 8, "rebal_days": 15, "trailing_stop": 0.35,
                                "cap": 0.25, "use_rp": False,
                                "trend_scale": {"bear": 0.50, "caution": 0.75}},

        "VAL+LV (0/50/50)":  {"mom_w": 0.0, "val_w": 0.50, "lv_w": 0.50, "sec_w": 0.0,
                                "top_n": 8, "rebal_days": 15, "trailing_stop": 0.35,
                                "cap": 0.25, "use_rp": False,
                                "trend_scale": {"bear": 0.50, "caution": 0.75}},

        # Interesting combos from Phase 1 results
        "MOM+VAL 70/30":     {"mom_w": 0.70, "val_w": 0.30, "lv_w": 0.0, "sec_w": 0.0,
                               "top_n": 8, "rebal_days": 15, "trailing_stop": None,
                               "cap": 0.25, "use_rp": False,
                               "trend_scale": {"bear": 0.50, "caution": 0.75}},

        # More positions with no stop
        "n15 MOM+VAL 60/40": {"mom_w": 0.60, "val_w": 0.40, "lv_w": 0.0, "sec_w": 0.0,
                               "top_n": 15, "rebal_days": 20, "trailing_stop": None,
                               "cap": 0.20, "use_rp": False,
                               "trend_scale": {"bear": 0.50, "caution": 0.75}},

        "n15 35/25/40 nostop": {"mom_w": 0.35, "val_w": 0.25, "lv_w": 0.40, "sec_w": 0.0,
                                 "top_n": 15, "rebal_days": 20, "trailing_stop": None,
                                 "cap": 0.20, "use_rp": False,
                                 "trend_scale": {"bear": 0.50, "caution": 0.75}},
    }

    # ═══════════════════════════════════════════════════════════
    # 2018-2025
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SLEEVE ATTRIBUTION: 2018-2025")
    print("="*70)

    print(f"\n{'Config':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*60)

    for name, cfg in configs.items():
        r = bt.run("2018-01-01", "2025-12-31", cfg)
        if r:
            print(f"{name:<30} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 2001-2025
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SLEEVE ATTRIBUTION: 2001-2025")
    print("="*70)

    print(f"\n{'Config':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*60)

    for name, cfg in configs.items():
        r = bt.run("2001-01-01", "2025-12-31", cfg)
        if r:
            print(f"{name:<30} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Yearly breakdown for baseline vs best
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("YEARLY BREAKDOWN: Baseline vs MOM+VAL 60/40")
    print("="*70)

    for label, cfg in [("BASELINE", configs["BASELINE (35/25/40)"]),
                        ("MOM+VAL 60/40", configs["MOM+VAL (60/40/0)"])]:
        print(f"\n{label}:")
        print(f"{'Year':<8} {'Return':>8}")
        print("-"*20)

        for year in range(2001, 2026):
            r = bt.run(f"{year}-01-01", f"{year}-12-31", cfg)
            if r and r.get("cagr") is not None:
                print(f"{year:<8} {r['cagr']*100:>7.1f}%")

    print("\n" + "="*70)
    print("ATTRIBUTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
