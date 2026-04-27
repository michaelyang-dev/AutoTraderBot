"""
Train Factor Ranker — Walk-Forward Validation
==============================================
Trains LGBMRanker on 12 factor scores with expanding window walk-forward.
Evaluates NDCG@8 and compares top-8 forward returns vs factor-only picks.

Usage:
    cd ml_service && python3 -m strategies.train_factor_ranker
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.ml_factor_ranker import MLFactorRanker, FACTOR_FEATURES, PURGE_DAYS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  ML FACTOR RANKER — WALK-FORWARD TRAINING")
    log("=" * 80)

    # ── Load data ──
    log("\n1. Loading data ...")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    log(f"   Features: {len(features)} rows, {features['symbol'].nunique()} symbols")
    log(f"   Date range: {features['date'].min().date()} to {features['date'].max().date()}")

    # Financial growth (for rev_growth, eps_growth)
    fin_growth = None
    fg_path = DATA_DIR / "enhanced_data" / "financial_growth.parquet"
    if fg_path.exists():
        fin_growth = pd.read_parquet(fg_path)
        fin_growth["date"] = pd.to_datetime(fin_growth["date"])
        log(f"   Financial growth: {len(fin_growth)} rows, {fin_growth['symbol'].nunique()} symbols")
    else:
        log("   WARNING: No financial_growth.parquet — rev_growth/eps_growth will be NaN")

    # ── Build training data ──
    log("\n2. Building training data ...")
    ranker = MLFactorRanker()
    full_data = ranker.build_training_data(features, fin_growth,
                                           start_date="2017-01-01", end_date="2025-12-31")
    log(f"   Training data: {len(full_data)} rows, {full_data['date'].nunique()} dates")
    log(f"   Date range: {full_data['date'].min().date()} to {full_data['date'].max().date()}")

    # Check feature coverage
    for col in FACTOR_FEATURES:
        pct = full_data[col].notna().mean() * 100
        log(f"   {col}: {pct:.1f}% coverage")

    # ── Walk-forward training ──
    log("\n3. Walk-forward training ...")
    test_years = [2022, 2023, 2024, 2025]
    all_oos_predictions = []
    models = {}

    for test_year in test_years:
        log(f"\n   ── Test year: {test_year} ──")

        # Train on all data before test year (with purge gap)
        train_end = pd.Timestamp(f"{test_year}-01-01") - pd.Timedelta(days=PURGE_DAYS)
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")

        train_df = full_data[full_data["date"] <= train_end].copy()
        test_df = full_data[(full_data["date"] >= test_start) & (full_data["date"] <= test_end)].copy()

        log(f"   Train: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
        log(f"   Test:  {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

        if len(train_df) < 10000 or len(test_df) < 1000:
            log(f"   SKIP: insufficient data")
            continue

        # Train
        t1 = time.perf_counter()

        # Use last 20% of train as validation for early stopping
        val_cutoff = train_df["date"].quantile(0.8)
        val_df = train_df[train_df["date"] > val_cutoff]
        train_only = train_df[train_df["date"] <= val_cutoff]

        fold_ranker = MLFactorRanker()
        fold_ranker.train(train_only, val_df)
        elapsed = time.perf_counter() - t1
        log(f"   Trained in {elapsed:.1f}s ({fold_ranker.model.num_trees()} trees)")

        # Predict on test
        test_df["ml_score"] = fold_ranker.predict_from_df(test_df)

        # Blend score: 0.6 * factor + 0.4 * ML (both normalized per date)
        def _blend_scores(grp):
            f = grp["momentum_score"]
            m = grp["ml_score"]
            f_norm = (f - f.min()) / (f.max() - f.min()) if f.max() > f.min() else 0.5
            m_norm = (m - m.min()) / (m.max() - m.min()) if m.max() > m.min() else 0.5
            grp["blend_score"] = 0.6 * f_norm + 0.4 * m_norm
            return grp

        test_df = test_df.groupby("date", group_keys=False).apply(_blend_scores)

        # Evaluate
        ndcg_ml = MLFactorRanker.evaluate_ndcg(test_df, "ml_score", k=8)
        ndcg_factor = MLFactorRanker.evaluate_ndcg(test_df, "momentum_score", k=8)
        ndcg_blend = MLFactorRanker.evaluate_ndcg(test_df, "blend_score", k=8)

        log(f"   NDCG@8: Factor={ndcg_factor:.4f}  ML={ndcg_ml:.4f}  Blend={ndcg_blend:.4f}")

        # Compare top-8 forward returns
        comparison = MLFactorRanker.compare_top_k_returns(test_df, k=8)
        for method, stats in comparison.items():
            log(f"   Top-8 avg 10d ret ({method}): {stats['avg_10d_ret']*100:.3f}%  "
                f"hit={stats['hit_rate']*100:.0f}%  n={stats['n_periods']}")

        all_oos_predictions.append(test_df)
        models[test_year] = fold_ranker

    # ── Overall OOS results ──
    if all_oos_predictions:
        all_oos = pd.concat(all_oos_predictions, ignore_index=True)
        log("\n" + "=" * 80)
        log("  OVERALL OOS RESULTS (2022-2025)")
        log("=" * 80)

        ndcg_ml = MLFactorRanker.evaluate_ndcg(all_oos, "ml_score", k=8)
        ndcg_factor = MLFactorRanker.evaluate_ndcg(all_oos, "momentum_score", k=8)
        ndcg_blend = MLFactorRanker.evaluate_ndcg(all_oos, "blend_score", k=8)

        log(f"\n  NDCG@8: Factor={ndcg_factor:.4f}  ML={ndcg_ml:.4f}  Blend={ndcg_blend:.4f}")

        comparison = MLFactorRanker.compare_top_k_returns(all_oos, k=8)
        log("\n  Top-8 Average Forward 10-Day Returns:")
        for method, stats in comparison.items():
            annualized = (1 + stats["avg_10d_ret"]) ** (252 / 10) - 1
            log(f"    {method:8s}: {stats['avg_10d_ret']*100:.3f}% per 10d  "
                f"(~{annualized*100:.1f}% ann.)  "
                f"hit={stats['hit_rate']*100:.0f}%  n={stats['n_periods']}")

        # Feature importance
        if models:
            last_model = list(models.values())[-1]
            importances = last_model.model.feature_importance(importance_type="gain")
            total = importances.sum()
            log("\n  Feature Importance (gain):")
            for feat, imp in sorted(zip(FACTOR_FEATURES, importances),
                                     key=lambda x: x[1], reverse=True):
                log(f"    {feat:25s}: {imp/total*100:5.1f}%")

            # Check: no single feature > 40%
            max_pct = max(importances) / total * 100
            if max_pct > 40:
                log(f"\n  WARNING: Feature '{FACTOR_FEATURES[np.argmax(importances)]}' "
                    f"dominates at {max_pct:.1f}% — overfitting risk")
            else:
                log(f"\n  Feature importance OK: max {max_pct:.1f}% (threshold: <40%)")

        # ── Decision gate ──
        log("\n" + "=" * 80)
        log("  DECISION GATE")
        log("=" * 80)

        blend_better = comparison.get("Blend", {}).get("avg_10d_ret", 0) > comparison.get("Factor", {}).get("avg_10d_ret", 0)
        ndcg_ok = ndcg_blend > 0.55

        if blend_better and ndcg_ok:
            log("\n  PASS: ML blend improves top-8 forward returns and NDCG@8 > 0.55")
            log("  Proceed to backtest integration.")

            # Save the model trained on ALL data up to 2025
            log("\n  Training final model on full 2017-2024 data ...")
            final_train = full_data[full_data["date"] < "2025-01-01"]
            final_ranker = MLFactorRanker()
            final_ranker.train(final_train)
            model_path = DATA_DIR / "ml_factor_model.pkl"
            final_ranker.save(model_path)
            log(f"  Model saved to {model_path}")
        else:
            log("\n  FAIL: ML blend does not improve over factor-only.")
            if not blend_better:
                log("  Reason: Blend top-8 avg return <= Factor top-8 avg return")
            if not ndcg_ok:
                log(f"  Reason: NDCG@8 = {ndcg_blend:.4f} < 0.55 threshold")
            log("  Do NOT proceed to backtest integration.")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
