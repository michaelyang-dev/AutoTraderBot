"""
Train and test the institutional-style alpha pipeline.
=====================================================
1. Feature Engine: 60+ features from existing data
2. ML Cross-Sectional Ranker: LightGBM walk-forward
3. Portfolio Optimizer: correlation-aware construction
4. Honest validation: OOS, walk-forward, start-day averaging
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
log = logging.getLogger("train_alpha")

from strategies.alpha_engine import AlphaBacktester

def main():
    print("="*70)
    print("INSTITUTIONAL ALPHA PIPELINE — TRAINING & VALIDATION")
    print("="*70)

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Load data and train ML model
    # ═══════════════════════════════════════════════════════════
    print("\n[1/5] Loading universe and building feature engine...")
    t0 = time.time()
    bt = AlphaBacktester("data/wrds/complete_sp1500_universe.pkl")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Feature preview: show what features we generate
    print("\n[1b] Feature preview (sample date)...")
    sample_dates = sorted(bt.prices.index)
    sample_date = sample_dates[len(sample_dates)//2]
    members = bt._get_sp1500(sample_date)
    feat_df = bt.feature_engine.generate_features(sample_date, members)
    print(f"  Date: {sample_date.date()}")
    print(f"  Stocks: {len(feat_df)}")
    print(f"  Features: {len(feat_df.columns)}")
    print(f"  Feature list: {sorted(feat_df.columns.tolist())}")

    # Show NaN rates
    nan_rates = feat_df.isna().mean().sort_values(ascending=False)
    print(f"\n  Features with >30% NaN:")
    for col, rate in nan_rates.items():
        if rate > 0.30:
            print(f"    {col}: {rate*100:.0f}% NaN")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Train ML model (walk-forward)
    # ═══════════════════════════════════════════════════════════
    print("\n[2/5] Training ML cross-sectional ranker (walk-forward)...")
    print("  Training period: 2016-07 to 2022-12")
    print("  This predicts 10-day forward cross-sectional return rank")
    t0 = time.time()

    result = bt.train_model("2016-07-01", "2022-12-31")

    if result:
        print(f"\n  Training complete in {time.time()-t0:.1f}s")
        print(f"  Walk-forward splits: {result['n_splits']}")
        print(f"  OOS Information Coefficients: {[f'{ic:.4f}' for ic in result['oos_ics']]}")
        print(f"  Mean OOS IC: {result['mean_ic']:.4f}")

        # IC interpretation
        ic = result['mean_ic']
        if ic > 0.05:
            print(f"  ✓ IC > 0.05: STRONG signal (institutional grade)")
        elif ic > 0.03:
            print(f"  ✓ IC > 0.03: MODERATE signal (usable)")
        elif ic > 0.01:
            print(f"  ~ IC > 0.01: WEAK signal (marginal)")
        else:
            print(f"  ✗ IC ≤ 0.01: NO signal (model doesn't work)")

        # Feature importance
        print(f"\n  Top 15 features by importance:")
        importance = sorted(result['feature_importance'].items(),
                            key=lambda x: x[1], reverse=True)
        for name, imp in importance[:15]:
            print(f"    {name:<25} {imp:.0f}")
    else:
        print("  ✗ Training FAILED")
        return

    # ═══════════════════════════════════════════════════════════
    # STEP 3: OOS backtest (2023-2025)
    # ═══════════════════════════════════════════════════════════
    print("\n[3/5] Out-of-sample backtest...")

    configs = {
        "ML_pure": {
            "rebal_days": 15, "max_positions": 15,
            "trailing_stop": None, "use_trend_filter": True,
            "factor_blend": 0.0,  # pure ML
        },
        "ML_70_factor_30": {
            "rebal_days": 15, "max_positions": 15,
            "trailing_stop": None, "use_trend_filter": True,
            "factor_blend": 0.3,  # 70% ML + 30% factor
        },
        "ML_50_factor_50": {
            "rebal_days": 15, "max_positions": 15,
            "trailing_stop": None, "use_trend_filter": True,
            "factor_blend": 0.5,  # 50/50
        },
        "Factor_pure": {
            "rebal_days": 15, "max_positions": 15,
            "trailing_stop": None, "use_trend_filter": True,
            "factor_blend": 1.0,  # pure factor (no ML)
        },
    }

    # Test on pure OOS period (2023-2025)
    print(f"\n  {'Config':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("  " + "-"*55)

    for name, cfg in configs.items():
        r = bt.run("2023-01-01", "2025-12-31", cfg)
        if r:
            print(f"  {name:<25} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Full period backtest (2018-2025)
    # ═══════════════════════════════════════════════════════════
    print("\n[4/5] Full period backtest (2018-2025)...")

    print(f"\n  {'Config':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("  " + "-"*55)

    best_config = None
    best_sharpe = -999

    for name, cfg in configs.items():
        r = bt.run("2018-01-01", "2025-12-31", cfg)
        if r:
            print(f"  {name:<25} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")
            if r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_config = name

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Start-day averaged for best config
    # ═══════════════════════════════════════════════════════════
    print(f"\n[5/5] Start-day averaged (best config: {best_config})...")

    if best_config:
        cfg = configs[best_config]
        cagrs = []
        sharpes = []
        for offset in range(5):
            start_dt = pd.Timestamp("2018-01-01") + pd.Timedelta(days=offset)
            r = bt.run(str(start_dt.date()), "2025-12-31", cfg)
            if r and r['cagr']:
                cagrs.append(r['cagr'])
                sharpes.append(r['sharpe'])

        if cagrs:
            print(f"  {best_config} (start-day avg):")
            print(f"    CAGR: {np.mean(cagrs)*100:.1f}% +/- {np.std(cagrs)*100:.1f}%")
            print(f"    Sharpe: {np.mean(sharpes):.2f} +/- {np.std(sharpes):.2f}")

        # Also average the pure factor for comparison
        cfg_factor = configs["Factor_pure"]
        cagrs_f = []
        for offset in range(5):
            start_dt = pd.Timestamp("2018-01-01") + pd.Timedelta(days=offset)
            r = bt.run(str(start_dt.date()), "2025-12-31", cfg_factor)
            if r and r['cagr']:
                cagrs_f.append(r['cagr'])

        if cagrs_f:
            print(f"  Factor_pure (start-day avg):")
            print(f"    CAGR: {np.mean(cagrs_f)*100:.1f}% +/- {np.std(cagrs_f)*100:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PIPELINE VALIDATION COMPLETE")
    print("="*70)

    if result:
        ic = result['mean_ic']
        print(f"\nML Model:")
        print(f"  OOS Information Coefficient: {ic:.4f}")
        print(f"  Signal quality: {'STRONG' if ic > 0.05 else 'MODERATE' if ic > 0.03 else 'WEAK' if ic > 0.01 else 'NONE'}")
        print(f"  Walk-forward splits: {result['n_splits']}, all positive: {all(x > 0 for x in result['oos_ics'])}")

    print(f"\nBest config: {best_config}")
    print(f"\nNOTE: These numbers use the alpha_engine backtest (different from")
    print(f"fast_backtest.py). Compare RELATIVE performance between configs,")
    print(f"not absolute CAGR vs the fast_backtest baseline.")


if __name__ == "__main__":
    main()
