#!/usr/bin/env python3
"""
Train Validation Model v5c (LGBM + RF Ensemble)
=================================================
Validation model -- trains with held-out 2024+ for honest backtest evaluation.
NEVER deploy this model live.

Uses the same architecture as train_production_model.py but with a 3-way
date-based split:
  - Train:   dates <= 2022-12-31
  - Calib:   2023-01-01 to 2023-12-31
  - Holdout: 2024-01-01+ (NEVER used during training or calibration)

This produces predictions_validation.parquet which can be used by
backtest.py --validation to get honest OOS performance estimates.

Pipeline:
  1. Load features.parquet (79 features, no days_until_earnings)
  2. Add 4 cross-sectional rank features (vol_rank_20d, momentum_rank_60d, rsi_rank, dist_sma50_rank)
  3. Split by date: train <= 2022, calibrate = 2023, holdout = 2024+
  4. Train LGBMClassifier + RandomForestClassifier
  5. Calibrate both with IsotonicRegression
  6. Generate ensemble predictions: 0.5 * LGBM + 0.5 * RF
  7. Report AUC on train, calib, and holdout separately
  8. Verify: load models, re-predict 20 random rows, compare

Run with:
    python3 train_validation_model.py
"""

import sys
import time
import os
import warnings
from pathlib import Path

# Prevent OpenMP thread deadlock on macOS ARM64
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).resolve().parent / "data"
INPUT_FILE    = DATA_DIR / "features.parquet"
MODEL_FILE    = DATA_DIR / "model_validation.lgb"
RF_MODEL_FILE = DATA_DIR / "model_rf_validation.pkl"
PRED_FILE     = DATA_DIR / "predictions_validation.parquet"
IMPUTER_FILE  = DATA_DIR / "imputer_validation.pkl"

TRAIN_END    = "2022-12-31"   # train on dates <= this
CALIB_END    = "2023-12-31"   # calibrate on 2023; holdout = 2024+
PROB_THRESH  = 0.55

N_TREES = 500
EARLY_STOP_ROUNDS = 100

LGB_PARAMS = dict(
    n_estimators      = N_TREES,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 31,
    min_child_samples = 50,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    objective         = "binary",
    metric            = "auc",
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

# Fundamental feature columns (LightGBM handles NaN natively for ETFs)
FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

# Cross-sectional rank features computed at runtime (added to 79 parquet features -> 83 total)
RANK_FEATURES = [
    ("vol_20d", "vol_rank_20d"),
    ("ret_60d", "momentum_rank_60d"),
    ("rsi_14", "rsi_rank"),
    ("dist_sma50", "dist_sma50_rank"),
]

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=50,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target", "target_v5", "in_sp500", "pct_rank"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


# ── STEP 1: Load data and train ──────────────────────────────────────────────

def train_validation_model():
    """Train LGBM + RF ensemble with 3-way date split: train/calib/holdout."""
    log(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found -- run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Verify days_until_earnings removed
    assert "days_until_earnings" not in df.columns, "days_until_earnings still present -- re-run data_pipeline.py"

    # Add cross-sectional rank features (computed per date, like signal_server does at runtime)
    for base_col, rank_col in RANK_FEATURES:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    # Cross-sectional target: top 20% of S&P 500 by forward 10-day return
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_v5"] = np.nan
    valid_df = df[sp500_mask & has_fwd].copy()
    valid_df["pct_rank"] = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_v5"] = (valid_df["pct_rank"] >= 0.80).astype(int)
    df.loc[valid_df.index, "target_v5"] = valid_df["target_v5"]

    # Get feature columns (79 parquet + 4 rank = 83)
    feature_cols = get_feature_cols(df)
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    before = len(df)
    df = df.dropna(subset=non_fund_cols + ["target_v5"])
    log(f"Loaded {before:,} -> {len(df):,} rows after dropping NaN")

    # Survivorship bias filter
    train_df = df[df["in_sp500"] == True].copy()
    log(f"  Survivorship filter: {len(df):,} -> {len(train_df):,} rows")
    log(f"  {df['symbol'].nunique()} symbols  |  "
        f"{df['date'].min().date()} -> {df['date'].max().date()}")
    log(f"  Features: {len(feature_cols)} columns")

    # Forward returns for predictions
    df = df.sort_values(["symbol", "date"])
    df["fwd_ret"] = df["fwd_10d_ret"]

    X_all = df[feature_cols].values
    X_filtered = train_df[feature_cols].values
    y_filtered = train_df["target_v5"].values

    # 3-way date split: train <= 2022, calib = 2023, holdout = 2024+
    train_end_ts = pd.Timestamp(TRAIN_END)
    calib_end_ts = pd.Timestamp(CALIB_END)

    train_mask = train_df["date"] <= train_end_ts
    calib_mask = (train_df["date"] > train_end_ts) & (train_df["date"] <= calib_end_ts)
    holdout_mask = train_df["date"] > calib_end_ts

    # Assert no overlap between holdout and train/calib
    assert (train_mask & holdout_mask).sum() == 0, "Holdout leaks into train!"
    assert (calib_mask & holdout_mask).sum() == 0, "Holdout leaks into calib!"
    assert (train_mask & calib_mask).sum() == 0, "Train leaks into calib!"
    assert holdout_mask.sum() > 0, "No holdout data -- check date boundaries"

    X_train, y_train = X_filtered[train_mask], y_filtered[train_mask]
    X_calib, y_calib = X_filtered[calib_mask], y_filtered[calib_mask]
    X_holdout, y_holdout = X_filtered[holdout_mask], y_filtered[holdout_mask]

    log(f"\n  3-way date split (survivorship-filtered):")
    log(f"    Train:   {train_mask.sum():>7,} rows  ({train_df[train_mask]['date'].min().date()} -> {train_df[train_mask]['date'].max().date()})")
    log(f"    Calib:   {calib_mask.sum():>7,} rows  ({train_df[calib_mask]['date'].min().date()} -> {train_df[calib_mask]['date'].max().date()})")
    log(f"    Holdout: {holdout_mask.sum():>7,} rows  ({train_df[holdout_mask]['date'].min().date()} -> {train_df[holdout_mask]['date'].max().date()})")

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    log(f"    Train pos/neg: {int(y_train.sum()):,} / {int(len(y_train) - y_train.sum()):,}  "
        f"(scale_pos_weight={scale:.2f})")

    # Imputer for RF (can't handle NaN)
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_calib_imp = imp.transform(X_calib)
    X_holdout_imp = imp.transform(X_holdout)
    X_all_imp = imp.transform(X_all)

    # ── Train LightGBM ──
    log(f"\n  Training LGBMClassifier ({N_TREES} trees, early_stop={EARLY_STOP_ROUNDS}) ...")
    model_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_calib, y_calib)],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    n_trees = model_lgb.booster_.num_trees()
    log(f"  LGBM: {n_trees} trees (early stopped from {N_TREES})")

    calib_lgbm = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
    calib_lgbm.fit(X_calib, y_calib)
    lgbm_train_probs = calib_lgbm.predict_proba(X_train)[:, 1]
    lgbm_calib_probs = calib_lgbm.predict_proba(X_calib)[:, 1]
    lgbm_holdout_probs = calib_lgbm.predict_proba(X_holdout)[:, 1]
    lgbm_train_auc = roc_auc_score(y_train, lgbm_train_probs)
    lgbm_auc = roc_auc_score(y_calib, lgbm_calib_probs)
    lgbm_holdout_auc = roc_auc_score(y_holdout, lgbm_holdout_probs)
    log(f"  LGBM AUC -- Train: {lgbm_train_auc:.4f}  Calib: {lgbm_auc:.4f}  Holdout: {lgbm_holdout_auc:.4f}")

    # ── Train Random Forest ──
    log(f"\n  Training RandomForestClassifier ({RF_PARAMS['n_estimators']} trees) ...")
    model_rf = RandomForestClassifier(**RF_PARAMS)
    model_rf.fit(X_train_imp, y_train)

    calib_rf = CalibratedClassifierCV(model_rf, method="isotonic", cv="prefit")
    calib_rf.fit(X_calib_imp, y_calib)
    rf_train_probs = calib_rf.predict_proba(X_train_imp)[:, 1]
    rf_calib_probs = calib_rf.predict_proba(X_calib_imp)[:, 1]
    rf_holdout_probs = calib_rf.predict_proba(X_holdout_imp)[:, 1]
    rf_train_auc = roc_auc_score(y_train, rf_train_probs)
    rf_auc = roc_auc_score(y_calib, rf_calib_probs)
    rf_holdout_auc = roc_auc_score(y_holdout, rf_holdout_probs)
    log(f"  RF AUC -- Train: {rf_train_auc:.4f}  Calib: {rf_auc:.4f}  Holdout: {rf_holdout_auc:.4f}")

    # ── Ensemble predictions ──
    log(f"\n  Generating ensemble predictions for ALL {len(df):,} rows ...")
    lgbm_all = calib_lgbm.predict_proba(X_all)[:, 1]
    rf_all = calib_rf.predict_proba(X_all_imp)[:, 1]
    ensemble_probs = 0.5 * lgbm_all + 0.5 * rf_all

    df["prob_lgbm"] = lgbm_all
    df["prob_rf"] = rf_all
    df["prob_ensemble"] = ensemble_probs

    ens_train_auc = roc_auc_score(y_train, 0.5 * lgbm_train_probs + 0.5 * rf_train_probs)
    ens_auc = roc_auc_score(y_calib, 0.5 * lgbm_calib_probs + 0.5 * rf_calib_probs)
    ens_holdout_auc = roc_auc_score(y_holdout, 0.5 * lgbm_holdout_probs + 0.5 * rf_holdout_probs)
    log(f"  Ensemble AUC -- Train: {ens_train_auc:.4f}  Calib: {ens_auc:.4f}  Holdout: {ens_holdout_auc:.4f}")
    log(f"  Ensemble range: [{ensemble_probs.min():.4f}, {ensemble_probs.max():.4f}]")
    log(f"  Mean: {ensemble_probs.mean():.4f}  Median: {np.median(ensemble_probs):.4f}")

    # Save models
    log(f"\n  Saving LGBM -> {MODEL_FILE.name}")
    joblib.dump(calib_lgbm, str(MODEL_FILE))
    log(f"  LGBM size: {MODEL_FILE.stat().st_size / 1024:.1f} KB")

    log(f"  Saving RF -> {RF_MODEL_FILE.name}")
    joblib.dump(calib_rf, str(RF_MODEL_FILE))
    log(f"  RF size: {RF_MODEL_FILE.stat().st_size / 1024:.1f} KB")

    # Save imputer (needed at inference for RF)
    joblib.dump(imp, str(IMPUTER_FILE))
    log(f"  Imputer -> {IMPUTER_FILE.name}")

    # Save predictions
    save_cols = ["date", "symbol", "target_v5", "prob_lgbm", "prob_rf",
                 "prob_ensemble", "fwd_ret", "in_sp500"]
    df[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow", compression="snappy")
    log(f"  Predictions -> {PRED_FILE.name} ({len(df):,} rows)")

    return calib_lgbm, calib_rf, imp, df, feature_cols


# ── STEP 2: Verify match ────────────────────────────────────────────────────

def verify_predictions(feature_cols):
    """Load models and predictions independently, re-predict 20 random rows, verify match."""
    log(f"\n{'='*70}")
    log("STEP 2: VERIFICATION -- models vs predictions_validation.parquet (20 rows)")
    log(f"{'='*70}")

    # Load independently
    loaded_lgbm = joblib.load(str(MODEL_FILE))
    loaded_rf = joblib.load(str(RF_MODEL_FILE))
    loaded_imp = joblib.load(str(IMPUTER_FILE))
    preds_df = pd.read_parquet(PRED_FILE)
    features_df = pd.read_parquet(INPUT_FILE)
    features_df["date"] = pd.to_datetime(features_df["date"])
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    # Add cross-sectional rank features to features_df
    for base_col, rank_col in RANK_FEATURES:
        if base_col in features_df.columns:
            features_df[rank_col] = features_df.groupby("date")[base_col].rank(pct=True)

    # Pick 20 random rows
    np.random.seed(42)
    unique_dates = preds_df["date"].unique()
    sample_dates = np.random.choice(unique_dates, size=min(20, len(unique_dates)), replace=False)

    mismatches = 0
    log(f"\n  {'Date':<12} {'Symbol':<8} {'Stored':>10} {'Re-pred':>10} {'Delta':>8} {'Match':>6}")
    log(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")

    for date in sorted(sample_dates):
        date_rows = preds_df[preds_df["date"] == date]
        if len(date_rows) == 0:
            continue
        row = date_rows.sample(1, random_state=int(pd.Timestamp(date).timestamp()) % 10000).iloc[0]

        feat_row = features_df[
            (features_df["date"] == row["date"]) &
            (features_df["symbol"] == row["symbol"])
        ]
        if len(feat_row) == 0:
            continue

        X_single = feat_row[feature_cols].values
        X_single_imp = loaded_imp.transform(X_single)

        re_lgbm = loaded_lgbm.predict_proba(X_single)[:, 1][0]
        re_rf = loaded_rf.predict_proba(X_single_imp)[:, 1][0]
        re_ensemble = 0.5 * re_lgbm + 0.5 * re_rf
        stored = row["prob_ensemble"]
        delta = abs(re_ensemble - stored)
        match = delta < 0.001

        status = "OK" if match else "FAIL"
        log(f"  {str(pd.Timestamp(date).date()):<12} {row['symbol']:<8} "
            f"{stored:>10.6f} {re_ensemble:>10.6f} {delta:>8.6f} {status:>6}")

        if not match:
            mismatches += 1

    log(f"\n  Results: {20 - mismatches}/20 matched (tolerance < 0.001)")

    if mismatches > 0:
        log(f"\n  *** VERIFICATION FAILED: {mismatches} mismatches ***")
        sys.exit(1)
    else:
        log(f"  *** VERIFICATION PASSED: all 20 rows match ***")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  \u26a0\ufe0f  VALIDATION MODEL -- DO NOT DEPLOY LIVE")
    log("  LGBM + RF ensemble, 83 features, 3-way split (holdout = 2024+)")
    log("=" * 70)

    # STEP 1: Train
    log(f"\n{'='*70}")
    log("STEP 1: TRAIN LGBM + RF ENSEMBLE (VALIDATION)")
    log(f"{'='*70}")
    calib_lgbm, calib_rf, imp, df, feature_cols = train_validation_model()

    # STEP 2: Verify (20 random rows)
    verify_predictions(feature_cols)

    n_lgbm = calib_lgbm.calibrated_classifiers_[0].estimator.n_features_in_
    n_rf = calib_rf.calibrated_classifiers_[0].estimator.n_features_in_
    log(f"\n  LGBM features: {n_lgbm}  |  RF features: {n_rf}")
    log(f"  Feature columns: {len(feature_cols)}")
    assert n_lgbm == len(feature_cols), f"LGBM feature count mismatch: LGBM={n_lgbm}, cols={len(feature_cols)}"
    assert n_rf <= len(feature_cols), f"RF has more features than expected: RF={n_rf}, cols={len(feature_cols)}"
    if n_rf < len(feature_cols):
        log(f"  Note: RF reports {n_rf} features (imputer may have reduced dimensionality) -- OK")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"\nFiles created:")
    log(f"  {MODEL_FILE}")
    log(f"  {RF_MODEL_FILE}")
    log(f"  {PRED_FILE}")
    log(f"  {IMPUTER_FILE}")
    log(f"\n  \u26a0\ufe0f  These are VALIDATION artifacts -- do NOT use for live trading.")


if __name__ == "__main__":
    main()
