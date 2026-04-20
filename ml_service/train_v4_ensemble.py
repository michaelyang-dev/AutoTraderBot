#!/usr/bin/env python3
"""
A3 Ensemble: LightGBM + XGBoost + Random Forest
=================================================
Trains XGBoost and Random Forest models on the same V4 cross-sectional
ranking task, creates 3-way ensemble predictions, and backtests.

Tasks:
  1. Train XGBoost (calibrated, isotonic)
  2. Train Random Forest (calibrated, isotonic)
  3. Create ensemble predictions (simple average)
  4. Verify ensemble math
  5. Measure model diversity (correlations, AUC)
  6. Backtest ensemble with Path B + vol targeting
  7. Compare all variants
  8. Memory check

Run: python3 train_v4_ensemble.py
"""

import sys
import time
import warnings
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.isotonic import IsotonicRegression


class ManualCalibratedModel:
    """Wrapper that applies isotonic calibration to any model with predict_proba."""
    def __init__(self, base_model, X_calib, y_calib):
        self.base_model = base_model
        raw_probs = base_model.predict_proba(X_calib)[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.iso.fit(raw_probs, y_calib)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.iso.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).resolve().parent / "data"
INPUT_FILE   = DATA_DIR / "features.parquet"

# Existing V4 LightGBM artifacts
LGBM_MODEL   = DATA_DIR / "model_v4.lgb"
LGBM_PREDS   = DATA_DIR / "predictions_v4.parquet"

# New model artifacts
XGB_MODEL    = DATA_DIR / "model_v4_xgb.pkl"
XGB_PREDS    = DATA_DIR / "predictions_v4_xgb.parquet"
RF_MODEL     = DATA_DIR / "model_v4_rf.pkl"
RF_PREDS     = DATA_DIR / "predictions_v4_rf.parquet"
ENS_PREDS    = DATA_DIR / "predictions_v4_ensemble.parquet"

CALIB_FRAC     = 0.20
TOP_PERCENTILE = 0.20
TOP_N_PICKS    = 5

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "days_until_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

PRIOR_RESULTS = {
    "V2 (60-stock)":              {"cagr": 0.2559, "sharpe": 1.65, "max_dd": -0.21},
    "V4 LightGBM (current prod)": {"cagr": 0.3460, "sharpe": 1.77, "max_dd": -0.243},
}


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target", "target_v4", "in_sp500"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


def compute_rank_target(df: pd.DataFrame) -> pd.DataFrame:
    """Identical to train_v4_ranking.py — cross-sectional top-20% target."""
    log("Computing cross-sectional rank target ...")
    df = df.sort_values(["symbol", "date"]).copy()
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)

    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    valid_mask = sp500_mask & has_fwd
    df["target_v4"] = np.nan

    valid_df = df[valid_mask].copy()
    valid_df["pct_rank"] = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_v4"] = (valid_df["pct_rank"] >= (1.0 - TOP_PERCENTILE)).astype(int)
    df.loc[valid_df.index, "target_v4"] = valid_df["target_v4"]

    n_valid = valid_df["target_v4"].notna().sum()
    n_pos = (valid_df["target_v4"] == 1).sum()
    log(f"  Valid: {n_valid:,}  |  Pos: {n_pos:,} ({n_pos/n_valid*100:.1f}%)")
    return df


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same as V4 — adds 4 rank features."""
    for col, src in [("vol_rank_20d", "vol_20d"), ("momentum_rank_60d", "ret_60d"),
                     ("rsi_rank", "rsi_14"), ("dist_sma50_rank", "dist_sma50")]:
        if src in df.columns:
            df[col] = df.groupby("date")[src].rank(pct=True)
    return df


def prepare_data():
    """Load features, compute target, prepare train/calib splits."""
    log(f"\n{'='*70}")
    log("DATA PREPARATION")
    log(f"{'='*70}")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    log(f"  Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")

    df = compute_rank_target(df)
    df = add_cross_sectional_features(df)

    feature_cols = get_feature_cols(df)
    non_fund = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    log(f"  Features: {len(feature_cols)} cols ({len(non_fund)} non-fundamental)")

    # Drop NaN in non-fundamental + target
    before = len(df)
    df = df.dropna(subset=non_fund + ["target_v4"])
    log(f"  After NaN drop: {before:,} → {len(df):,}")

    # Survivorship filter
    train_df = df[df["in_sp500"] == True].copy()
    log(f"  S&P 500 filter: {len(df):,} → {len(train_df):,}")

    # Date split
    all_dates = np.sort(train_df["date"].unique())
    split_idx = int(len(all_dates) * (1 - CALIB_FRAC))
    calib_start = pd.Timestamp(all_dates[split_idx])

    train_mask = train_df["date"] < calib_start
    calib_mask = train_df["date"] >= calib_start

    X_train = train_df.loc[train_mask, feature_cols].values
    y_train = train_df.loc[train_mask, "target_v4"].values
    X_calib = train_df.loc[calib_mask, feature_cols].values
    y_calib = train_df.loc[calib_mask, "target_v4"].values
    X_all = df[feature_cols].values

    log(f"  Train: {len(X_train):,} rows  |  Calib: {len(X_calib):,} rows  |  All: {len(X_all):,}")
    log(f"  Train dates: {train_df[train_mask]['date'].min().date()} → {train_df[train_mask]['date'].max().date()}")
    log(f"  Calib dates: {train_df[calib_mask]['date'].min().date()} → {train_df[calib_mask]['date'].max().date()}")

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    log(f"  Class balance: pos={int(y_train.sum()):,}, neg={int(len(y_train)-y_train.sum()):,}, scale={scale:.2f}")

    return df, feature_cols, X_train, y_train, X_calib, y_calib, X_all, scale


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1: Train XGBoost
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_calib, y_calib, X_all, df, feature_cols, scale):
    log(f"\n{'='*70}")
    log("TASK 1: TRAIN XGBoost")
    log(f"{'='*70}")

    try:
        from xgboost import XGBClassifier
    except ImportError:
        log("  ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    t0 = time.perf_counter()

    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric="auc",
        use_label_encoder=False,
    )

    # Replace NaN with large negative for XGBoost (doesn't handle NaN natively in v1.x)
    X_train_xgb = np.nan_to_num(X_train, nan=-999.0)
    X_calib_xgb = np.nan_to_num(X_calib, nan=-999.0)
    X_all_xgb = np.nan_to_num(X_all, nan=-999.0)

    log(f"  Training XGBClassifier (500 trees, early_stop=50, max_depth=7) ...")
    xgb_model.fit(
        X_train_xgb, y_train,
        eval_set=[(X_calib_xgb, y_calib)],
        early_stopping_rounds=50,
        verbose=False,
    )
    n_trees = xgb_model.best_iteration + 1 if hasattr(xgb_model, 'best_iteration') and xgb_model.best_iteration is not None else 500
    elapsed = time.perf_counter() - t0
    log(f"  Trained: {n_trees} trees in {elapsed:.1f}s")

    # Calibrate with manual isotonic (avoids sklearn/xgboost version conflicts)
    log(f"  Calibrating with isotonic regression ...")
    calib_xgb = ManualCalibratedModel(xgb_model, X_calib_xgb, y_calib)

    # AUC (use raw pre-calibration probs for proper AUC)
    raw_probs = xgb_model.predict_proba(X_calib_xgb)[:, 1]
    auc = roc_auc_score(y_calib, raw_probs)
    log(f"  Holdout AUC: {auc:.4f}")

    # Predictions for ALL rows
    log(f"  Generating predictions for {len(X_all_xgb):,} rows ...")
    all_probs = calib_xgb.predict_proba(X_all_xgb)[:, 1]
    log(f"  Range: [{all_probs.min():.4f}, {all_probs.max():.4f}]  Mean: {all_probs.mean():.4f}")

    # Save
    joblib.dump(calib_xgb, str(XGB_MODEL))
    log(f"  Model saved: {XGB_MODEL.name} ({XGB_MODEL.stat().st_size/1024:.1f} KB)")

    save_df = df[["date", "symbol", "target_v4"]].copy()
    save_df["prob_xgb"] = all_probs
    save_df["fwd_ret"] = df["fwd_10d_ret"]
    save_df["in_sp500"] = df["in_sp500"]
    save_df.to_parquet(XGB_PREDS, index=False)
    log(f"  Predictions saved: {XGB_PREDS.name} ({len(save_df):,} rows)")

    # Feature importance (top 10)
    imp = xgb_model.feature_importances_
    top_feats = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)[:10]
    log(f"  Top 10 features:")
    for f, v in top_feats:
        log(f"    {f:<30s} {v:.4f}")

    return calib_xgb, auc


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2: Train Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train, X_calib, y_calib, X_all, df, feature_cols):
    log(f"\n{'='*70}")
    log("TASK 2: TRAIN Random Forest")
    log(f"{'='*70}")

    t0 = time.perf_counter()

    # RF doesn't handle NaN — fill with median from training set
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train_rf = imputer.fit_transform(X_train)
    X_calib_rf = imputer.transform(X_calib)
    X_all_rf = imputer.transform(X_all)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=100,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    log(f"  Training RandomForestClassifier (300 trees, max_depth=10, min_leaf=100) ...")
    rf_model.fit(X_train_rf, y_train)
    elapsed = time.perf_counter() - t0
    log(f"  Trained in {elapsed:.1f}s")

    # Calibrate with manual isotonic
    log(f"  Calibrating with isotonic regression ...")
    calib_rf = ManualCalibratedModel(rf_model, X_calib_rf, y_calib)

    # AUC
    raw_probs = rf_model.predict_proba(X_calib_rf)[:, 1]
    auc = roc_auc_score(y_calib, raw_probs)
    log(f"  Holdout AUC: {auc:.4f}")

    # Predictions
    log(f"  Generating predictions for {len(X_all_rf):,} rows ...")
    all_probs = calib_rf.predict_proba(X_all_rf)[:, 1]
    log(f"  Range: [{all_probs.min():.4f}, {all_probs.max():.4f}]  Mean: {all_probs.mean():.4f}")

    # Save (store imputer with model for deployment)
    joblib.dump({"model": calib_rf, "imputer": imputer}, str(RF_MODEL))
    log(f"  Model saved: {RF_MODEL.name} ({RF_MODEL.stat().st_size/1024:.1f} KB)")

    save_df = df[["date", "symbol", "target_v4"]].copy()
    save_df["prob_rf"] = all_probs
    save_df["fwd_ret"] = df["fwd_10d_ret"]
    save_df["in_sp500"] = df["in_sp500"]
    save_df.to_parquet(RF_PREDS, index=False)
    log(f"  Predictions saved: {RF_PREDS.name} ({len(save_df):,} rows)")

    # Feature importance (top 10)
    imp = rf_model.feature_importances_
    top_feats = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)[:10]
    log(f"  Top 10 features:")
    for f, v in top_feats:
        log(f"    {f:<30s} {v:.4f}")

    return calib_rf, auc, imputer


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3: Create Ensemble Predictions
# ══════════════════════════════════════════════════════════════════════════════

def create_ensemble():
    log(f"\n{'='*70}")
    log("TASK 3: CREATE ENSEMBLE PREDICTIONS")
    log(f"{'='*70}")

    lgbm = pd.read_parquet(LGBM_PREDS)
    xgb = pd.read_parquet(XGB_PREDS)
    rf = pd.read_parquet(RF_PREDS)

    lgbm["date"] = pd.to_datetime(lgbm["date"])
    xgb["date"] = pd.to_datetime(xgb["date"])
    rf["date"] = pd.to_datetime(rf["date"])

    log(f"  LightGBM preds: {len(lgbm):,} rows")
    log(f"  XGBoost preds:  {len(xgb):,} rows")
    log(f"  RF preds:       {len(rf):,} rows")

    # Merge on (date, symbol)
    ens = lgbm[["date", "symbol", "prob_v4", "fwd_ret", "in_sp500", "target_v4"]].merge(
        xgb[["date", "symbol", "prob_xgb"]], on=["date", "symbol"], how="inner"
    ).merge(
        rf[["date", "symbol", "prob_rf"]], on=["date", "symbol"], how="inner"
    )
    log(f"  Merged: {len(ens):,} rows (inner join)")

    # Simple average
    ens["prob_ensemble"] = (ens["prob_v4"] + ens["prob_xgb"] + ens["prob_rf"]) / 3.0

    log(f"  Ensemble range: [{ens['prob_ensemble'].min():.4f}, {ens['prob_ensemble'].max():.4f}]")
    log(f"  Mean: {ens['prob_ensemble'].mean():.4f}")

    ens.to_parquet(ENS_PREDS, index=False)
    log(f"  Saved: {ENS_PREDS.name} ({len(ens):,} rows)")

    return ens


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 4: Verify Ensemble Math
# ══════════════════════════════════════════════════════════════════════════════

def verify_ensemble(ens):
    log(f"\n{'='*70}")
    log("TASK 4: VERIFY ENSEMBLE MATH")
    log(f"{'='*70}")

    np.random.seed(42)
    sample = ens.sample(n=20)
    mismatches = 0

    log(f"  {'Date':<12} {'Symbol':<8} {'LGBM':>8} {'XGB':>8} {'RF':>8} {'Avg':>8} {'Ens':>8} {'Delta':>10}")
    log(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    for _, row in sample.iterrows():
        manual = (row["prob_v4"] + row["prob_xgb"] + row["prob_rf"]) / 3.0
        delta = abs(manual - row["prob_ensemble"])
        ok = delta < 0.0001
        if not ok:
            mismatches += 1
        log(f"  {str(row['date'].date()):<12} {row['symbol']:<8} "
            f"{row['prob_v4']:>8.4f} {row['prob_xgb']:>8.4f} {row['prob_rf']:>8.4f} "
            f"{manual:>8.4f} {row['prob_ensemble']:>8.4f} {delta:>10.6f} {'✓' if ok else '✗'}")

    if mismatches > 0:
        log(f"\n  *** VERIFICATION FAILED: {mismatches} mismatches ***")
        sys.exit(1)
    else:
        log(f"\n  *** VERIFICATION PASSED: 20/20 rows match (tolerance < 0.0001) ***")


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 5: Measure Model Diversity
# ══════════════════════════════════════════════════════════════════════════════

def measure_diversity(ens, lgbm_auc, xgb_auc, rf_auc):
    log(f"\n{'='*70}")
    log("TASK 5: MODEL DIVERSITY ANALYSIS")
    log(f"{'='*70}")

    # Correlations
    corr_lgbm_xgb = ens["prob_v4"].corr(ens["prob_xgb"])
    corr_lgbm_rf  = ens["prob_v4"].corr(ens["prob_rf"])
    corr_xgb_rf   = ens["prob_xgb"].corr(ens["prob_rf"])

    log(f"\n  Prediction Correlations:")
    log(f"    corr(LightGBM, XGBoost): {corr_lgbm_xgb:.4f}")
    log(f"    corr(LightGBM, RF):      {corr_lgbm_rf:.4f}")
    log(f"    corr(XGBoost, RF):       {corr_xgb_rf:.4f}")

    max_corr = max(corr_lgbm_xgb, corr_lgbm_rf, corr_xgb_rf)
    if max_corr > 0.95:
        log(f"  ⚠ WARNING: Max correlation {max_corr:.4f} > 0.95 — limited diversity")
    elif max_corr > 0.90:
        log(f"  ⚠ NOTE: Max correlation {max_corr:.4f} > 0.90 — moderate diversity")
    else:
        log(f"  ✓ Good diversity: all correlations < 0.90")

    # Individual AUCs
    log(f"\n  Holdout AUC:")
    log(f"    LightGBM: {lgbm_auc:.4f}")
    log(f"    XGBoost:  {xgb_auc:.4f}")
    log(f"    RF:       {rf_auc:.4f}")

    # Ensemble AUC (on rows with valid target)
    valid = ens.dropna(subset=["target_v4"])
    if len(valid) > 0 and valid["target_v4"].nunique() > 1:
        ens_auc = roc_auc_score(valid["target_v4"], valid["prob_ensemble"])
        lgbm_full_auc = roc_auc_score(valid["target_v4"], valid["prob_v4"])
        xgb_full_auc = roc_auc_score(valid["target_v4"], valid["prob_xgb"])
        rf_full_auc = roc_auc_score(valid["target_v4"], valid["prob_rf"])
        log(f"\n  Full-dataset AUC (all rows with target):")
        log(f"    LightGBM:  {lgbm_full_auc:.4f}")
        log(f"    XGBoost:   {xgb_full_auc:.4f}")
        log(f"    RF:        {rf_full_auc:.4f}")
        log(f"    Ensemble:  {ens_auc:.4f}")
        auc_lift = ens_auc - lgbm_full_auc
        log(f"    Ensemble AUC lift vs LightGBM: {auc_lift:+.4f}")

    # Rank disagreement: how often do models disagree on top-5?
    log(f"\n  Rank Disagreement (top-5 overlap per date):")
    sp500 = ens[ens["in_sp500"] == True]
    overlaps = []
    for date, grp in sp500.groupby("date"):
        top_lgbm = set(grp.nlargest(5, "prob_v4")["symbol"])
        top_xgb  = set(grp.nlargest(5, "prob_xgb")["symbol"])
        top_rf   = set(grp.nlargest(5, "prob_rf")["symbol"])
        top_ens  = set(grp.nlargest(5, "prob_ensemble")["symbol"])
        overlaps.append({
            "lgbm_xgb": len(top_lgbm & top_xgb),
            "lgbm_rf": len(top_lgbm & top_rf),
            "xgb_rf": len(top_xgb & top_rf),
            "lgbm_ens": len(top_lgbm & top_ens),
        })
    ov = pd.DataFrame(overlaps)
    log(f"    Avg top-5 overlap (LGBM vs XGB): {ov['lgbm_xgb'].mean():.1f}/5")
    log(f"    Avg top-5 overlap (LGBM vs RF):  {ov['lgbm_rf'].mean():.1f}/5")
    log(f"    Avg top-5 overlap (XGB vs RF):   {ov['xgb_rf'].mean():.1f}/5")
    log(f"    Avg top-5 overlap (LGBM vs Ens): {ov['lgbm_ens'].mean():.1f}/5")

    return {
        "corr_lgbm_xgb": corr_lgbm_xgb,
        "corr_lgbm_rf": corr_lgbm_rf,
        "corr_xgb_rf": corr_xgb_rf,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 6: Backtest Ensemble (and individual XGB/RF for comparison)
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(pred_file, prob_col, label):
    """Run Path B backtest with vol targeting for a given prediction column."""
    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS, SLIPPAGE, POSITION_PCT,
        SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
        COOLDOWN_DAYS,
        Signal, Position, SlotConfig,
        Strategy,
        MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from backtest_ml import calc_metrics, calc_alpha_beta

    preds_df = pd.read_parquet(pred_file)
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df = preds_df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    class TopNStrategy(Strategy):
        def __init__(self, predictions_df, prob_column, top_n=TOP_N_PICKS, position_pct=POSITION_PCT):
            self._top_n = top_n
            self._position_pct = position_pct
            self._signals_by_date = {}
            for date, grp in predictions_df.groupby("date"):
                sp500_grp = grp[grp["in_sp500"] == True]
                if sp500_grp.empty:
                    continue
                top = sp500_grp.nlargest(self._top_n, prob_column)
                self._signals_by_date[date] = [
                    (row.symbol, getattr(row, prob_column), row.fwd_ret)
                    for row in top.itertuples(index=False)
                ]

        @property
        def name(self):
            return "ml_medium"

        def generate_signals(self, date, universe_data):
            raw = self._signals_by_date.get(date, [])
            return [Signal(symbol=s, confidence=p, strategy_name=self.name, fwd_ret=f)
                    for s, p, f in raw]

        def check_exit(self, position, current_data):
            if current_data["idx"] >= position.exit_idx:
                return True, "hold_complete"
            return False, ""

        def get_position_size(self, signal, portfolio_value):
            ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
            return portfolio_value * self._position_pct * ml_mult

    # Fetch OHLCV
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

    log(f"    Fetching OHLCV ({len(all_syms)} symbols) ...")
    raw = yf.download(all_syms, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume = None
    if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
        volume = raw["Volume"]
        volume.index = pd.to_datetime(volume.index).tz_localize(None)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close_aligned["SPY"].to_dict()

    ml_strat = TopNStrategy(preds_df, prob_col, top_n=TOP_N_PICKS)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # Run with vol targeting
    from backtest_v4_voltarget import instrumented_run_voltarget
    vals, trades, diag = instrumented_run_voltarget(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label=label)

    m = calc_metrics(vals, trades, years, label)
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    m["alpha"] = alpha
    m["beta"] = beta

    # Year-by-year
    yearly = vals.resample("YE").last()
    prev = vals.iloc[0]
    yr_rets = {}
    for yr_end in yearly.index:
        yr_ret = (yearly[yr_end] / prev) - 1.0
        yr_rets[yr_end.year] = yr_ret
        prev = yearly[yr_end]

    return m, yr_rets


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 8: Memory Check
# ══════════════════════════════════════════════════════════════════════════════

def check_memory():
    log(f"\n{'='*70}")
    log("TASK 8: MEMORY CHECK")
    log(f"{'='*70}")

    import os
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
    except ImportError:
        mem_before = 0
        log("  (psutil not installed — using file sizes as proxy)")

    # Load all 3 models
    lgbm = joblib.load(str(LGBM_MODEL))
    xgb = joblib.load(str(XGB_MODEL))
    rf_bundle = joblib.load(str(RF_MODEL))

    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_models = mem_after - mem_before
        log(f"  Memory after loading 3 models: {mem_after:.0f} MB (delta: {mem_models:.0f} MB)")
    except ImportError:
        mem_models = 0

    # File sizes
    lgbm_size = LGBM_MODEL.stat().st_size / 1024 / 1024
    xgb_size = XGB_MODEL.stat().st_size / 1024 / 1024
    rf_size = RF_MODEL.stat().st_size / 1024 / 1024
    total_size = lgbm_size + xgb_size + rf_size

    log(f"  File sizes:")
    log(f"    LightGBM: {lgbm_size:.1f} MB")
    log(f"    XGBoost:  {xgb_size:.1f} MB")
    log(f"    RF:       {rf_size:.1f} MB")
    log(f"    Total:    {total_size:.1f} MB")

    fits_in_2gb = total_size < 500 and (mem_models < 2000 if mem_models > 0 else True)
    log(f"\n  Fits in 2GB RAM: {'YES' if fits_in_2gb else 'NO'}")

    return {"file_mb": total_size, "ram_mb": mem_models, "fits_2gb": fits_in_2gb}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()

    # ── Data prep ──
    df, feature_cols, X_train, y_train, X_calib, y_calib, X_all, scale = prepare_data()

    # ── Get LightGBM holdout AUC ──
    lgbm_model = joblib.load(str(LGBM_MODEL))
    non_fund = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    lgbm_calib_probs = lgbm_model.predict_proba(X_calib)[:, 1]
    lgbm_auc = roc_auc_score(y_calib, lgbm_calib_probs)
    log(f"\n  LightGBM holdout AUC: {lgbm_auc:.4f}")

    # ── Task 1: XGBoost ──
    xgb_model, xgb_auc = train_xgboost(X_train, y_train, X_calib, y_calib, X_all, df, feature_cols, scale)

    # ── Task 2: Random Forest ──
    rf_model, rf_auc, rf_imputer = train_random_forest(X_train, y_train, X_calib, y_calib, X_all, df, feature_cols)

    # ── Task 3: Ensemble ──
    ens = create_ensemble()

    # ── Task 4: Verify ──
    verify_ensemble(ens)

    # ── Task 5: Diversity ──
    diversity = measure_diversity(ens, lgbm_auc, xgb_auc, rf_auc)

    # ── Task 6: Backtest all variants ──
    log(f"\n{'='*70}")
    log("TASK 6: BACKTEST ALL VARIANTS")
    log(f"{'='*70}")

    log(f"\n  --- XGBoost only ---")
    m_xgb, yr_xgb = run_backtest(XGB_PREDS, "prob_xgb", "V4 XGBoost")

    log(f"\n  --- Random Forest only ---")
    m_rf, yr_rf = run_backtest(RF_PREDS, "prob_rf", "V4 RF")

    log(f"\n  --- 3-Way Ensemble ---")
    m_ens, yr_ens = run_backtest(ENS_PREDS, "prob_ensemble", "V4 Ensemble")

    # ── Task 7: Comparison ──
    log(f"\n{'='*70}")
    log("TASK 7: FULL COMPARISON")
    log(f"{'='*70}")

    all_results = {
        **PRIOR_RESULTS,
        "V4 XGBoost":     {"cagr": m_xgb["cagr"], "sharpe": m_xgb["sharpe"], "max_dd": m_xgb["max_dd"]},
        "V4 RF":          {"cagr": m_rf["cagr"], "sharpe": m_rf["sharpe"], "max_dd": m_rf["max_dd"]},
        "V4 Ensemble (3)":{"cagr": m_ens["cagr"], "sharpe": m_ens["sharpe"], "max_dd": m_ens["max_dd"]},
    }

    log(f"\n  {'Variant':<30s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Sortino':>8s} {'Win%':>6s} {'Trades':>7s}")
    log(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*7}")

    for name, r in PRIOR_RESULTS.items():
        log(f"  {name:<30s} {r['cagr']*100:>7.2f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    for name, m in [("V4 XGBoost", m_xgb), ("V4 RF", m_rf), ("V4 Ensemble (3-way)", m_ens)]:
        log(f"  {name:<30s} {m['cagr']*100:>7.2f}% {m['sharpe']:>8.2f} {m['max_dd']*100:>7.1f}% "
            f"{m['sortino']:>8.3f} {m['win_rate']*100:>5.1f}% {m['n_trades']:>7,}")

    # Year-by-year for ensemble
    log(f"\n  Year-by-Year Returns (Ensemble vs LightGBM):")
    log(f"  {'Year':<6s} {'Ensemble':>10s} {'XGBoost':>10s} {'RF':>10s}")
    log(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
    all_years = sorted(set(list(yr_ens.keys()) + list(yr_xgb.keys()) + list(yr_rf.keys())))
    for yr in all_years:
        e = yr_ens.get(yr, 0)
        x = yr_xgb.get(yr, 0)
        r = yr_rf.get(yr, 0)
        log(f"  {yr:<6d} {e*100:>+9.2f}% {x*100:>+9.2f}% {r*100:>+9.2f}%")

    # Ensemble additional metrics
    log(f"\n  Ensemble Additional Metrics:")
    log(f"    Alpha vs SPY:  {m_ens['alpha']*100:+.2f}%")
    log(f"    Beta:          {m_ens['beta']:.3f}")
    log(f"    Final Value:   ${m_ens['final_value']:,.0f}")
    log(f"    Profit Factor: {m_ens['profit_factor']:.3f}" if np.isfinite(m_ens['profit_factor']) else f"    Profit Factor: inf")

    # ── Task 8: Memory ──
    mem = check_memory()

    # ── Pass/Fail ──
    log(f"\n{'='*70}")
    log("DEPLOYMENT CRITERIA CHECK")
    log(f"{'='*70}")

    checks = [
        ("Sharpe >= 1.82",       m_ens["sharpe"] >= 1.82,    f"{m_ens['sharpe']:.3f}"),
        ("CAGR >= 33%",          m_ens["cagr"] >= 0.33,      f"{m_ens['cagr']*100:.2f}%"),
        ("Max DD <= -26%",       m_ens["max_dd"] >= -0.26,    f"{m_ens['max_dd']*100:.1f}%"),
        ("Correlations < 0.95",  max(diversity.values()) < 0.95, f"max={max(diversity.values()):.4f}"),
        ("Models fit 2GB RAM",   mem["fits_2gb"],             f"{mem['file_mb']:.1f} MB files"),
    ]

    all_pass = True
    for desc, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        log(f"  [{status}] {desc:<25s} — {val}")

    log(f"\n  Overall: {'ALL PASS — ready for deployment review' if all_pass else 'SOME FAILED — do NOT deploy'}")

    elapsed = time.perf_counter() - t_start
    log(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
