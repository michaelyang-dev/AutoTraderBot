#!/usr/bin/env python3
"""
Train Alt-Data Model v5 (LGBM + RF Ensemble with Insider Features)
===================================================================
Same architecture as train_production_model.py but trained on
features_v5.parquet (83 base + 5 insider = 88 features).

Outputs separate model artifacts for A/B comparison:
  - data/model_v5.lgb          (calibrated LightGBM)
  - data/model_v5_rf.pkl       (calibrated Random Forest)
  - data/imputer_v5.pkl        (median imputer for RF)
  - data/predictions_v5.parquet (full predictions)

Usage:
    python3 train_altdata_model.py
"""

import sys
import time
import warnings
from pathlib import Path

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
DATA_DIR     = Path(__file__).resolve().parent / "data"
INPUT_FILE   = DATA_DIR / "features_v5.parquet"
MODEL_FILE   = DATA_DIR / "model_v5.lgb"
RF_MODEL_FILE = DATA_DIR / "model_v5_rf.pkl"
IMPUTER_FILE = DATA_DIR / "imputer_v5.pkl"
PRED_FILE    = DATA_DIR / "predictions_v5.parquet"

CALIB_FRAC   = 0.20   # last 20% of dates for calibration
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

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=50,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

# Cross-sectional rank features computed at runtime
RANK_FEATURES = [
    ("vol_20d", "vol_rank_20d"),
    ("ret_60d", "momentum_rank_60d"),
    ("rsi_14", "rsi_rank"),
    ("dist_sma50", "dist_sma50_rank"),
]


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target", "target_v5", "in_sp500", "pct_rank"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


def train():
    """Train LGBM + RF ensemble on features_v5.parquet."""
    log(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run compute_insider_features.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Verify insider features present
    insider_cols = [
        "insider_net_buy_count_30d", "insider_cluster_score_10d",
        "insider_buy_dollar_30d", "insider_ceo_buy_90d", "insider_director_buy_90d",
    ]
    missing = [c for c in insider_cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing insider features: {missing}")
    log(f"  Insider features confirmed: {len(insider_cols)} columns")

    # Add cross-sectional rank features
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

    # Get feature columns (should be 88 = 84 parquet + 4 rank)
    feature_cols = get_feature_cols(df)
    log(f"  Feature columns: {len(feature_cols)}")

    # Show which are insider features
    insider_in_features = [c for c in feature_cols if c.startswith("insider_")]
    log(f"  Insider features in model: {insider_in_features}")

    # Drop rows with NaN in non-fundamental columns
    FUNDAMENTAL_FEATURE_COLS = [
        "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
        "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
        "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
        "debt_to_equity", "current_ratio", "roe", "roa",
        "days_since_earnings", "eps_surprise_last",
        "eps_revision_30d", "revenue_revision_30d",
        "insider_buy_ratio_90d", "insider_net_shares_90d",
    ]
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    before = len(df)
    df = df.dropna(subset=non_fund_cols + ["target_v5"])
    log(f"  Rows: {before:,} → {len(df):,} after dropping NaN")

    # Survivorship bias filter
    train_df = df[df["in_sp500"] == True].copy()
    log(f"  Survivorship filter: {len(df):,} → {len(train_df):,} rows")
    log(f"  {df['symbol'].nunique()} symbols  |  "
        f"{df['date'].min().date()} → {df['date'].max().date()}")

    # Forward returns for predictions
    df = df.sort_values(["symbol", "date"])
    df["fwd_ret"] = df["fwd_10d_ret"]

    X_all = df[feature_cols].values
    X_filtered = train_df[feature_cols].values
    y_filtered = train_df["target_v5"].values

    # Date-based split
    all_dates = np.sort(train_df["date"].unique())
    split_idx = int(len(all_dates) * (1 - CALIB_FRAC))
    calib_start = pd.Timestamp(all_dates[split_idx])

    train_mask = train_df["date"] < calib_start
    calib_mask = train_df["date"] >= calib_start

    X_train, y_train = X_filtered[train_mask], y_filtered[train_mask]
    X_calib, y_calib = X_filtered[calib_mask], y_filtered[calib_mask]

    log(f"\n  Split by date:")
    log(f"    Train: {train_mask.sum():,} rows  ({train_df[train_mask]['date'].min().date()} → {train_df[train_mask]['date'].max().date()})")
    log(f"    Calib: {calib_mask.sum():,} rows  ({train_df[calib_mask]['date'].min().date()} → {train_df[calib_mask]['date'].max().date()})")

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    log(f"    Train pos/neg: {int(y_train.sum()):,} / {int(len(y_train) - y_train.sum()):,}  "
        f"(scale_pos_weight={scale:.2f})")

    # Imputer for RF
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_calib_imp = imp.transform(X_calib)
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
    lgbm_calib_probs = calib_lgbm.predict_proba(X_calib)[:, 1]
    lgbm_auc = roc_auc_score(y_calib, lgbm_calib_probs)
    log(f"  LGBM Calibration AUC: {lgbm_auc:.4f}")

    # ── Train Random Forest ──
    log(f"\n  Training RandomForestClassifier ({RF_PARAMS['n_estimators']} trees) ...")
    model_rf = RandomForestClassifier(**RF_PARAMS)
    model_rf.fit(X_train_imp, y_train)

    calib_rf = CalibratedClassifierCV(model_rf, method="isotonic", cv="prefit")
    calib_rf.fit(X_calib_imp, y_calib)
    rf_calib_probs = calib_rf.predict_proba(X_calib_imp)[:, 1]
    rf_auc = roc_auc_score(y_calib, rf_calib_probs)
    log(f"  RF Calibration AUC: {rf_auc:.4f}")

    # ── Ensemble predictions ──
    log(f"\n  Generating ensemble predictions for ALL {len(df):,} rows ...")
    lgbm_all = calib_lgbm.predict_proba(X_all)[:, 1]
    rf_all = calib_rf.predict_proba(X_all_imp)[:, 1]
    ensemble_probs = 0.5 * lgbm_all + 0.5 * rf_all

    df["prob_lgbm"] = lgbm_all
    df["prob_rf"] = rf_all
    df["prob_ensemble"] = ensemble_probs

    ens_auc = roc_auc_score(y_calib, 0.5 * lgbm_calib_probs + 0.5 * rf_calib_probs)
    log(f"  Ensemble Calibration AUC: {ens_auc:.4f}")
    log(f"  Ensemble range: [{ensemble_probs.min():.4f}, {ensemble_probs.max():.4f}]")
    log(f"  Mean: {ensemble_probs.mean():.4f}  Median: {np.median(ensemble_probs):.4f}")

    # ── Feature importance (top 20) ──
    log(f"\n  Top 20 features by LightGBM importance:")
    importances = model_lgb.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    for i, (name, imp_val) in enumerate(feat_imp[:20], 1):
        marker = " ← INSIDER" if name.startswith("insider_") else ""
        log(f"    {i:2d}. {name:<35s} {imp_val:>6d}{marker}")

    # Show insider feature ranks
    insider_ranks = [(i+1, name, imp_val) for i, (name, imp_val) in enumerate(feat_imp)
                     if name.startswith("insider_")]
    log(f"\n  Insider feature ranks:")
    for rank, name, imp_val in insider_ranks:
        log(f"    #{rank}: {name} (importance={imp_val})")

    # Save models
    log(f"\n  Saving LGBM → {MODEL_FILE.name}")
    joblib.dump(calib_lgbm, str(MODEL_FILE))
    log(f"  LGBM size: {MODEL_FILE.stat().st_size / 1024:.1f} KB")

    log(f"  Saving RF → {RF_MODEL_FILE.name}")
    joblib.dump(calib_rf, str(RF_MODEL_FILE))
    log(f"  RF size: {RF_MODEL_FILE.stat().st_size / 1024:.1f} KB")

    log(f"  Saving Imputer → {IMPUTER_FILE.name}")
    joblib.dump(imp, str(IMPUTER_FILE))

    # Save predictions
    save_cols = ["date", "symbol", "target_v5", "prob_lgbm", "prob_rf",
                 "prob_ensemble", "fwd_ret", "in_sp500"]
    df[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow", compression="snappy")
    log(f"  Predictions → {PRED_FILE.name} ({len(df):,} rows)")

    return feature_cols, lgbm_auc, rf_auc, ens_auc


def verify(feature_cols):
    """Load models independently and verify 20 random rows match."""
    log(f"\n{'='*70}")
    log("VERIFICATION — models vs predictions_v5.parquet (20 rows)")
    log(f"{'='*70}")

    loaded_lgbm = joblib.load(str(MODEL_FILE))
    loaded_rf = joblib.load(str(RF_MODEL_FILE))
    loaded_imp = joblib.load(str(IMPUTER_FILE))
    preds_df = pd.read_parquet(PRED_FILE)
    features_df = pd.read_parquet(INPUT_FILE)
    features_df["date"] = pd.to_datetime(features_df["date"])
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    for base_col, rank_col in RANK_FEATURES:
        if base_col in features_df.columns:
            features_df[rank_col] = features_df.groupby("date")[base_col].rank(pct=True)

    np.random.seed(42)
    unique_dates = preds_df["date"].unique()
    sample_dates = np.random.choice(unique_dates, size=min(20, len(unique_dates)), replace=False)

    mismatches = 0
    log(f"\n  {'Date':<12} {'Symbol':<8} {'Stored':>10} {'Re-pred':>10} {'Delta':>8} {'Match':>6}")
    log(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*6}")

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

    if mismatches > 0:
        log(f"\n  *** VERIFICATION FAILED: {mismatches} mismatches ***")
        sys.exit(1)
    else:
        log(f"\n  *** VERIFICATION PASSED: all rows match ***")


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  ALT-DATA MODEL v5 TRAINING")
    log("  LGBM + RF ensemble, 88 features (83 base + 5 insider)")
    log("=" * 70)

    feature_cols, lgbm_auc, rf_auc, ens_auc = train()
    verify(feature_cols)

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"\nFiles created:")
    log(f"  {MODEL_FILE}")
    log(f"  {RF_MODEL_FILE}")
    log(f"  {IMPUTER_FILE}")
    log(f"  {PRED_FILE}")


if __name__ == "__main__":
    main()
