#!/usr/bin/env python3
"""
Train Production Model v7 — LambdaRank Dual-Ensemble 40/60 Blend
=================================================================
Trains TWO LGBMRanker+XGBRanker ensembles and blends them 40/60:
  - Base ensemble:   83 features (no sector-relative)
  - Sector ensemble: 87 features (with sector-relative)
  - Final:           0.4 * base + 0.6 * sector

Key change from v6: ranking objective (LambdaMART / NDCG) instead of
binary classification. The model directly optimizes NDCG@5, which is
what we actually care about (picking the best 5 stocks per day).

Labels: cross-sectional decile (0-9) of forward 10-day return, where
9 = top decile. This replaces the binary "top 20% = 1" target.

Pipeline:
  1. Load features.parquet (83 base + 4 sector-relative = 87 columns)
  2. Add 4 cross-sectional rank features
  3. Compute decile relevance labels (0-9) per trading day
  4. Split by date with purge gap
  5. Train base ensemble:   LGBMRanker + XGBRanker on 83 features
  6. Train sector ensemble: LGBMRanker + XGBRanker on 87 features
  7. Wrap rankers in RankerWrapper (predict_proba compatibility)
  8. Generate blended predictions: 0.4 * base + 0.6 * sector
  9. Verify: load models, re-predict 20 random rows, compare

Run with:
    python3 train_production_model.py
"""

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).resolve().parent / "data"
INPUT_FILE   = DATA_DIR / "features.parquet"
PRED_FILE    = DATA_DIR / "predictions.parquet"

# Model files — base ensemble (no sector features)
MODEL_BASE_LGBM = DATA_DIR / "model_base_lgbm.pkl"
MODEL_BASE_XGB  = DATA_DIR / "model_base_xgb.pkl"
IMPUTER_BASE    = DATA_DIR / "imputer_base.pkl"

# Model files — sector ensemble (with sector features)
MODEL_SECT_LGBM = DATA_DIR / "model_sector_lgbm.pkl"
MODEL_SECT_XGB  = DATA_DIR / "model_sector_xgb.pkl"
IMPUTER_SECT    = DATA_DIR / "imputer_sector.pkl"

# Legacy model paths (signal_server also checks these)
MODEL_FILE     = DATA_DIR / "model.lgb"
XGB_MODEL_FILE = DATA_DIR / "model_xgb.pkl"
IMPUTER_FILE   = DATA_DIR / "imputer.pkl"

# Blend weight: 0.4 * base + 0.6 * sector
BLEND_WEIGHT_BASE   = 0.4
BLEND_WEIGHT_SECTOR = 0.6

CALIB_FRAC   = 0.20   # last 20% of dates for calibration
PURGE_TRADING_DAYS = 10  # must match label horizon (10-day forward return)

N_TREES = 500
EARLY_STOP_ROUNDS = 100
N_DECILES = 10  # graded relevance labels 0..(N_DECILES-1)

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
    objective         = "lambdarank",
    metric            = "ndcg",
    eval_at           = [5],
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=1,
    random_state=42,
    objective="rank:ndcg",
    eval_metric="ndcg@5",
)

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

SECTOR_FEATURE_COLS = [
    "ret_10d_vs_sector", "ret_20d_vs_sector",
    "rsi_14_vs_sector", "vol_20d_vs_sector",
]

RANK_FEATURES = [
    ("vol_20d", "vol_rank_20d"),
    ("ret_60d", "momentum_rank_60d"),
    ("rsi_14", "rsi_rank"),
    ("dist_sma50", "dist_sma50_rank"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  RankerWrapper — backward-compatible predict_proba() for signal_server
# ══════════════════════════════════════════════════════════════════════════════

class RankerWrapper:
    """
    Wraps LGBMRanker/XGBRanker to provide predict_proba() interface.

    signal_server.py calls model.predict_proba(X)[:, 1]. This wrapper
    normalizes raw ranker scores to [0,1] via sigmoid so the signal
    server works unchanged. Ranking order is preserved.
    """

    def __init__(self, ranker):
        self.ranker = ranker

    def predict_proba(self, X):
        scores = self.ranker.predict(X)
        probs = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return self.ranker.predict(X)


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame, exclude_cols: set = None) -> list:
    exclude = {"date", "symbol", "target", "target_v5", "target_rank",
               "in_sp500", "pct_rank"}
    if exclude_cols:
        exclude.update(exclude_cols)
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


def compute_groups(dates_series: pd.Series) -> np.ndarray:
    """Compute group sizes for LGBMRanker from a date-sorted Series."""
    return dates_series.groupby(dates_series).size().values


def compute_qids(dates_series: pd.Series) -> np.ndarray:
    """Compute query IDs for XGBRanker from a date-sorted Series."""
    unique_dates = dates_series.unique()
    date_to_qid = {d: i for i, d in enumerate(unique_dates)}
    return dates_series.map(date_to_qid).values


def train_ensemble(X_train, y_train, X_calib, y_calib,
                   train_groups, calib_groups,
                   train_qids, calib_qids, label):
    """Train LGBMRanker + XGBRanker ensemble, return wrapped models."""
    # ── LGBMRanker ──
    model_lgb = lgb.LGBMRanker(**LGB_PARAMS)
    model_lgb.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_calib, y_calib)],
        eval_group=[calib_groups],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    n_trees = model_lgb.booster_.num_trees()
    wrapped_lgbm = RankerWrapper(model_lgb)
    lgbm_scores = wrapped_lgbm.predict_proba(X_calib)[:, 1]
    log(f"    {label} LGBM: {n_trees} trees, score range=[{lgbm_scores.min():.3f}, {lgbm_scores.max():.3f}]")

    # ── XGBRanker ──
    model_xgb = xgb.XGBRanker(**XGB_PARAMS)
    model_xgb.fit(
        X_train, y_train,
        qid=train_qids,
        eval_set=[(X_calib, y_calib)],
        eval_qid=[calib_qids],
        verbose=False,
    )
    wrapped_xgb = RankerWrapper(model_xgb)
    xgb_scores = wrapped_xgb.predict_proba(X_calib)[:, 1]
    log(f"    {label} XGB:  {XGB_PARAMS['n_estimators']} trees, score range=[{xgb_scores.min():.3f}, {xgb_scores.max():.3f}]")

    ens_scores = 0.5 * lgbm_scores + 0.5 * xgb_scores
    log(f"    {label} Ensemble score range=[{ens_scores.min():.3f}, {ens_scores.max():.3f}]")

    return wrapped_lgbm, wrapped_xgb


def train_production_model():
    """Train dual LGBMRanker+XGBRanker ensemble (base + sector) on all data."""
    log(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    assert "days_until_earnings" not in df.columns, "days_until_earnings still present — re-run data_pipeline.py"

    # Add cross-sectional rank features
    for base_col, rank_col in RANK_FEATURES:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    # Cross-sectional decile target (0-9) for ranking
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_rank"] = np.nan
    valid_df = df[sp500_mask & has_fwd].copy()
    pct = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_rank"] = np.clip((pct * N_DECILES).astype(int), 0, N_DECILES - 1)
    df.loc[valid_df.index, "target_rank"] = valid_df["target_rank"]

    # Also keep binary target for compatibility (predictions output)
    valid_df["pct_rank"] = pct
    valid_df["target_v5"] = (valid_df["pct_rank"] >= 0.80).astype(int)
    df["target_v5"] = np.nan
    df.loc[valid_df.index, "target_v5"] = valid_df["target_v5"]

    # Feature columns for both ensembles
    all_feature_cols = get_feature_cols(df)
    base_feature_cols = get_feature_cols(df, exclude_cols=set(SECTOR_FEATURE_COLS))

    non_fund_cols = [c for c in all_feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    before = len(df)
    df = df.dropna(subset=non_fund_cols + ["target_rank"])
    log(f"Loaded {before:,} → {len(df):,} rows after dropping NaN")

    # Survivorship bias filter
    train_df = df[df["in_sp500"] == True].copy()
    log(f"  Survivorship filter: {len(df):,} → {len(train_df):,} rows")
    log(f"  {df['symbol'].nunique()} symbols  |  "
        f"{df['date'].min().date()} → {df['date'].max().date()}")
    log(f"  Base features:   {len(base_feature_cols)} columns")
    log(f"  Sector features: {len(all_feature_cols)} columns")

    # Forward returns for predictions
    df = df.sort_values(["symbol", "date"])
    df["fwd_ret"] = df["fwd_10d_ret"]

    # ── Sort train_df by date for proper ranking groups ──
    train_df = train_df.sort_values("date").reset_index(drop=True)

    # Date-based split
    all_dates = np.sort(train_df["date"].unique())
    split_idx = int(len(all_dates) * (1 - CALIB_FRAC))
    calib_start = pd.Timestamp(all_dates[split_idx])

    # Purge gap
    purge_train_end = pd.Timestamp(all_dates[max(0, split_idx - PURGE_TRADING_DAYS)])
    purge_calib_start = pd.Timestamp(all_dates[min(len(all_dates) - 1, split_idx + PURGE_TRADING_DAYS)])

    train_mask = train_df["date"] < purge_train_end
    calib_mask = train_df["date"] >= purge_calib_start

    y_train = train_df.loc[train_mask, "target_rank"].values.astype(int)
    y_calib = train_df.loc[calib_mask, "target_rank"].values.astype(int)

    # Compute group/qid arrays for rankers
    train_dates = train_df.loc[train_mask, "date"]
    calib_dates = train_df.loc[calib_mask, "date"]
    train_groups = compute_groups(train_dates)
    calib_groups = compute_groups(calib_dates)
    train_qids = compute_qids(train_dates)
    calib_qids = compute_qids(calib_dates)

    purged_rows = (~train_mask & ~calib_mask).sum()
    log(f"\n  Split by date with {PURGE_TRADING_DAYS}-day purge gap:")
    log(f"    Train: {train_mask.sum():,} rows  ({train_dates.min().date()} → {train_dates.max().date()})  [{len(train_groups)} groups]")
    log(f"    Purge: {purged_rows:,} rows dropped")
    log(f"    Calib: {calib_mask.sum():,} rows  ({calib_dates.min().date()} → {calib_dates.max().date()})  [{len(calib_groups)} groups]")
    log(f"    Target: decile labels 0-{N_DECILES-1}")

    # ── Train base ensemble (no sector features) ──
    log(f"\n  Training BASE ensemble ({len(base_feature_cols)} features) ...")
    X_train_base = train_df.loc[train_mask, base_feature_cols].values
    X_calib_base = train_df.loc[calib_mask, base_feature_cols].values

    imp_base = SimpleImputer(strategy="median")
    X_train_base_imp = imp_base.fit_transform(X_train_base)
    X_calib_base_imp = imp_base.transform(X_calib_base)

    wrapped_base_lgbm, wrapped_base_xgb = train_ensemble(
        X_train_base_imp, y_train, X_calib_base_imp, y_calib,
        train_groups, calib_groups, train_qids, calib_qids, "Base")

    # ── Train sector ensemble (all features) ──
    log(f"\n  Training SECTOR ensemble ({len(all_feature_cols)} features) ...")
    X_train_sect = train_df.loc[train_mask, all_feature_cols].values
    X_calib_sect = train_df.loc[calib_mask, all_feature_cols].values

    imp_sect = SimpleImputer(strategy="median")
    X_train_sect_imp = imp_sect.fit_transform(X_train_sect)
    X_calib_sect_imp = imp_sect.transform(X_calib_sect)

    wrapped_sect_lgbm, wrapped_sect_xgb = train_ensemble(
        X_train_sect_imp, y_train, X_calib_sect_imp, y_calib,
        train_groups, calib_groups, train_qids, calib_qids, "Sector")

    # ── Generate blended predictions ──
    log(f"\n  Generating 40/60 blended predictions for ALL {len(df):,} rows ...")
    X_all_base = imp_base.transform(df[base_feature_cols].values)
    X_all_sect = imp_sect.transform(df[all_feature_cols].values)

    base_probs = 0.5 * wrapped_base_lgbm.predict_proba(X_all_base)[:, 1] + \
                 0.5 * wrapped_base_xgb.predict_proba(X_all_base)[:, 1]
    sect_probs = 0.5 * wrapped_sect_lgbm.predict_proba(X_all_sect)[:, 1] + \
                 0.5 * wrapped_sect_xgb.predict_proba(X_all_sect)[:, 1]
    blended = BLEND_WEIGHT_BASE * base_probs + BLEND_WEIGHT_SECTOR * sect_probs

    df["prob_base"] = base_probs
    df["prob_sector"] = sect_probs
    df["prob_ensemble"] = blended

    log(f"  Blend range: [{blended.min():.4f}, {blended.max():.4f}]")
    log(f"  Mean: {blended.mean():.4f}  Median: {np.median(blended):.4f}")

    # ── Save models ──
    log(f"\n  Saving models ...")
    for path, obj, label in [
        (MODEL_BASE_LGBM, wrapped_base_lgbm, "Base LGBM"),
        (MODEL_BASE_XGB,  wrapped_base_xgb,  "Base XGB"),
        (IMPUTER_BASE,    imp_base,           "Base Imputer"),
        (MODEL_SECT_LGBM, wrapped_sect_lgbm, "Sector LGBM"),
        (MODEL_SECT_XGB,  wrapped_sect_xgb,  "Sector XGB"),
        (IMPUTER_SECT,    imp_sect,           "Sector Imputer"),
    ]:
        joblib.dump(obj, str(path))
        log(f"    {label} → {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    # Legacy compatibility
    joblib.dump(wrapped_sect_lgbm, str(MODEL_FILE))
    joblib.dump(wrapped_sect_xgb, str(XGB_MODEL_FILE))
    joblib.dump(imp_sect, str(IMPUTER_FILE))
    log(f"    Legacy copies: model.lgb, model_xgb.pkl, imputer.pkl")

    # Save predictions
    save_cols = ["date", "symbol", "target_v5", "prob_base", "prob_sector",
                 "prob_ensemble", "fwd_ret", "in_sp500"]
    df[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow", compression="snappy")
    log(f"  Predictions → {PRED_FILE.name} ({len(df):,} rows)")

    return {
        "base_lgbm": wrapped_base_lgbm, "base_xgb": wrapped_base_xgb, "imp_base": imp_base,
        "sect_lgbm": wrapped_sect_lgbm, "sect_xgb": wrapped_sect_xgb, "imp_sect": imp_sect,
        "base_feature_cols": base_feature_cols, "all_feature_cols": all_feature_cols,
        "df": df,
    }


def verify_predictions(result):
    """Load models and predictions independently, re-predict 20 random rows, verify match."""
    log(f"\n{'='*70}")
    log("VERIFICATION — models vs predictions.parquet (20 rows)")
    log(f"{'='*70}")

    base_lgbm = joblib.load(str(MODEL_BASE_LGBM))
    base_xgb  = joblib.load(str(MODEL_BASE_XGB))
    imp_base  = joblib.load(str(IMPUTER_BASE))
    sect_lgbm = joblib.load(str(MODEL_SECT_LGBM))
    sect_xgb  = joblib.load(str(MODEL_SECT_XGB))
    imp_sect  = joblib.load(str(IMPUTER_SECT))

    preds_df = pd.read_parquet(PRED_FILE)
    features_df = pd.read_parquet(INPUT_FILE)
    features_df["date"] = pd.to_datetime(features_df["date"])
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    for base_col, rank_col in RANK_FEATURES:
        if base_col in features_df.columns:
            features_df[rank_col] = features_df.groupby("date")[base_col].rank(pct=True)

    base_feature_cols = result["base_feature_cols"]
    all_feature_cols = result["all_feature_cols"]

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

        X_base = imp_base.transform(feat_row[base_feature_cols].values)
        X_sect = imp_sect.transform(feat_row[all_feature_cols].values)

        re_base = 0.5 * base_lgbm.predict_proba(X_base)[:, 1][0] + \
                  0.5 * base_xgb.predict_proba(X_base)[:, 1][0]
        re_sect = 0.5 * sect_lgbm.predict_proba(X_sect)[:, 1][0] + \
                  0.5 * sect_xgb.predict_proba(X_sect)[:, 1][0]
        re_blend = BLEND_WEIGHT_BASE * re_base + BLEND_WEIGHT_SECTOR * re_sect

        stored = row["prob_ensemble"]
        delta = abs(re_blend - stored)
        match = delta < 0.001

        status = "OK" if match else "FAIL"
        log(f"  {str(pd.Timestamp(date).date()):<12} {row['symbol']:<8} "
            f"{stored:>10.6f} {re_blend:>10.6f} {delta:>8.6f} {status:>6}")

        if not match:
            mismatches += 1

    log(f"\n  Results: {20 - mismatches}/20 matched (tolerance < 0.001)")

    if mismatches > 0:
        log(f"\n  *** VERIFICATION FAILED: {mismatches} mismatches ***")
        sys.exit(1)
    else:
        log(f"  *** VERIFICATION PASSED: all 20 rows match ***")


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  PRODUCTION MODEL v7 TRAINING — LambdaRank")
    log("  Dual-ensemble 40/60 blend (base + sector)")
    log("  Objective: lambdarank (NDCG@5) on decile labels")
    log("=" * 70)

    result = train_production_model()
    verify_predictions(result)

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"\nModel files:")
    log(f"  Base:   {MODEL_BASE_LGBM.name}, {MODEL_BASE_XGB.name}, {IMPUTER_BASE.name}")
    log(f"  Sector: {MODEL_SECT_LGBM.name}, {MODEL_SECT_XGB.name}, {IMPUTER_SECT.name}")
    log(f"  Legacy: model.lgb, model_xgb.pkl, imputer.pkl")
    log(f"  Predictions: {PRED_FILE.name}")


if __name__ == "__main__":
    main()
