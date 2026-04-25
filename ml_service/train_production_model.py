#!/usr/bin/env python3
"""
Train Production Model v5d (LGBM + XGB Ensemble)
=================================================
Trains two models (LightGBM + XGBoost) with bug-fixed features,
generates 50/50 ensemble predictions, and verifies consistency.

v5d changes from v5c:
  - Swapped RandomForest for XGBoost (walk-forward validated)
  - Added 4 sector-relative features (87 total)

Pipeline:
  1. Load features.parquet (83 features, no days_until_earnings)
  2. Add 4 cross-sectional rank features (vol_rank_20d, momentum_rank_60d, rsi_rank, dist_sma50_rank)
  3. Split by date: first 80% → train, last 20% → calibration
  4. Train LGBMClassifier + XGBClassifier
  5. Calibrate both with IsotonicRegression
  6. Generate ensemble predictions: 0.5 * LGBM + 0.5 * XGB
  7. Verify: load models, re-predict 20 random rows, compare

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).resolve().parent / "data"
INPUT_FILE   = DATA_DIR / "features.parquet"
MODEL_FILE   = DATA_DIR / "model.lgb"
XGB_MODEL_FILE = DATA_DIR / "model_xgb.pkl"
PRED_FILE    = DATA_DIR / "predictions.parquet"

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

# Cross-sectional rank features computed at runtime (added to 79 parquet features → 83 total)
RANK_FEATURES = [
    ("vol_20d", "vol_rank_20d"),
    ("ret_60d", "momentum_rank_60d"),
    ("rsi_14", "rsi_rank"),
    ("dist_sma50", "dist_sma50_rank"),
]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=1,
    random_state=42,
    eval_metric="auc",
)

# V1 baseline (from prior backtest)
BASELINE_CAGR   = 0.2408
BASELINE_SHARPE = 1.655


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target", "target_v5", "in_sp500", "pct_rank"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


# ── STEP 1: Load data and train ──────────────────────────────────────────────

def train_production_model():
    """Train LGBM + XGB ensemble on all data with date-based calibration split."""
    log(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Verify days_until_earnings removed
    assert "days_until_earnings" not in df.columns, "days_until_earnings still present — re-run data_pipeline.py"

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
    log(f"Loaded {before:,} → {len(df):,} rows after dropping NaN")

    # Survivorship bias filter
    train_df = df[df["in_sp500"] == True].copy()
    log(f"  Survivorship filter: {len(df):,} → {len(train_df):,} rows")
    log(f"  {df['symbol'].nunique()} symbols  |  "
        f"{df['date'].min().date()} → {df['date'].max().date()}")
    log(f"  Features: {len(feature_cols)} columns")

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

    log(f"\n  Split by date (survivorship-filtered):")
    log(f"    Train: {train_mask.sum():,} rows  ({train_df[train_mask]['date'].min().date()} → {train_df[train_mask]['date'].max().date()})")
    log(f"    Calib: {calib_mask.sum():,} rows  ({train_df[calib_mask]['date'].min().date()} → {train_df[calib_mask]['date'].max().date()})")

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    log(f"    Train pos/neg: {int(y_train.sum()):,} / {int(len(y_train) - y_train.sum()):,}  "
        f"(scale_pos_weight={scale:.2f})")

    # Imputer for XGBoost
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

    # ── Train XGBoost ──
    log(f"\n  Training XGBClassifier ({XGB_PARAMS['n_estimators']} trees) ...")
    model_xgb = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale)
    model_xgb.fit(X_train_imp, y_train)

    calib_xgb = CalibratedClassifierCV(model_xgb, method="isotonic", cv="prefit")
    calib_xgb.fit(X_calib_imp, y_calib)
    xgb_calib_probs = calib_xgb.predict_proba(X_calib_imp)[:, 1]
    xgb_auc = roc_auc_score(y_calib, xgb_calib_probs)
    log(f"  XGB Calibration AUC: {xgb_auc:.4f}")

    # ── Ensemble predictions ──
    log(f"\n  Generating ensemble predictions for ALL {len(df):,} rows ...")
    lgbm_all = calib_lgbm.predict_proba(X_all)[:, 1]
    xgb_all = calib_xgb.predict_proba(X_all_imp)[:, 1]
    ensemble_probs = 0.5 * lgbm_all + 0.5 * xgb_all

    df["prob_lgbm"] = lgbm_all
    df["prob_xgb"] = xgb_all
    df["prob_ensemble"] = ensemble_probs

    ens_auc = roc_auc_score(y_calib, 0.5 * lgbm_calib_probs + 0.5 * xgb_calib_probs)
    log(f"  Ensemble Calibration AUC: {ens_auc:.4f}")
    log(f"  Ensemble range: [{ensemble_probs.min():.4f}, {ensemble_probs.max():.4f}]")
    log(f"  Mean: {ensemble_probs.mean():.4f}  Median: {np.median(ensemble_probs):.4f}")

    # Save models
    log(f"\n  Saving LGBM → {MODEL_FILE.name}")
    joblib.dump(calib_lgbm, str(MODEL_FILE))
    log(f"  LGBM size: {MODEL_FILE.stat().st_size / 1024:.1f} KB")

    log(f"  Saving XGB → {XGB_MODEL_FILE.name}")
    joblib.dump(calib_xgb, str(XGB_MODEL_FILE))
    log(f"  XGB size: {XGB_MODEL_FILE.stat().st_size / 1024:.1f} KB")

    # Save imputer (needed at inference for XGB)
    imputer_file = DATA_DIR / "imputer.pkl"
    joblib.dump(imp, str(imputer_file))
    log(f"  Imputer → {imputer_file.name}")

    # Save predictions
    save_cols = ["date", "symbol", "target_v5", "prob_lgbm", "prob_xgb",
                 "prob_ensemble", "fwd_ret", "in_sp500"]
    df[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow", compression="snappy")
    log(f"  Predictions → {PRED_FILE.name} ({len(df):,} rows)")

    return calib_lgbm, calib_xgb, imp, df, feature_cols


# ── STEP 2: Verify match ────────────────────────────────────────────────────

def verify_predictions(feature_cols):
    """Load models and predictions independently, re-predict 20 random rows, verify match."""
    log(f"\n{'='*70}")
    log("STEP 2: VERIFICATION — models vs predictions.parquet (20 rows)")
    log(f"{'='*70}")

    # Load independently
    loaded_lgbm = joblib.load(str(MODEL_FILE))
    loaded_xgb = joblib.load(str(XGB_MODEL_FILE))
    loaded_imp = joblib.load(str(DATA_DIR / "imputer.pkl"))
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
        re_xgb = loaded_xgb.predict_proba(X_single_imp)[:, 1][0]
        re_ensemble = 0.5 * re_lgbm + 0.5 * re_xgb
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


# ── STEP 3: Path B backtest ─────────────────────────────────────────────────

def run_backtest():
    """Run Path B backtest with the new predictions."""
    log(f"\n{'='*70}")
    log("STEP 3: PATH B BACKTEST")
    log(f"{'='*70}")

    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS,
        MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from backtest_utils import calc_metrics, calc_alpha_beta
    from diagnose_combined import instrumented_run

    preds_df = pd.read_parquet(PRED_FILE)
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df = preds_df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    log(f"  Predictions: {len(preds_df):,} rows  |  {len(universe_syms)} symbols  |  {years:.1f} years")
    log(f"  Date range: {pd.Timestamp(all_dates[0]).date()} → {pd.Timestamp(all_dates[-1]).date()}")

    # Fetch OHLCV
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

    log(f"  Fetching OHLCV for {len(all_syms)} symbols ...")
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

    # Build strategies
    ml_strat = MLMediumStrategy(preds_df, threshold=PROB_THRESH)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # Run backtest
    log(f"  Running Path B (ML Medium + Momentum + Mean Reversion) ...")
    vals, trades, diag = instrumented_run(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B")

    m = calc_metrics(vals, trades, years, "Path B")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    m["alpha"] = alpha
    m["avg_pos"] = float(np.mean(diag["daily_pos_count"]))

    return m


# ── STEP 4: Decision logic ─────────────────────────────────────────────────

def print_recommendation(metrics: dict):
    """Print deployment recommendation based on backtest results."""
    log(f"\n{'='*70}")
    log("STEP 4: DEPLOYMENT RECOMMENDATION")
    log(f"{'='*70}")

    cagr   = metrics.get("cagr", 0)
    sharpe = metrics.get("sharpe", 0)
    sortino = metrics.get("sortino", 0)
    max_dd = metrics.get("max_dd", 0)
    n_trades = metrics.get("n_trades", 0)
    win_rate = metrics.get("win_rate", 0)
    alpha  = metrics.get("alpha", 0)

    log(f"\n  {'Metric':<20} {'New Model':>12} {'V1 Baseline':>12} {'Delta':>10}")
    log(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10}")
    log(f"  {'CAGR':<20} {cagr*100:>11.2f}% {BASELINE_CAGR*100:>11.2f}% {(cagr-BASELINE_CAGR)*100:>+9.2f}%")
    log(f"  {'Sharpe':<20} {sharpe:>12.3f} {BASELINE_SHARPE:>12.3f} {sharpe-BASELINE_SHARPE:>+10.3f}")
    log(f"  {'Sortino':<20} {sortino:>12.3f}")
    log(f"  {'Max Drawdown':<20} {max_dd*100:>11.2f}%")
    log(f"  {'Win Rate':<20} {win_rate*100:>11.2f}%")
    log(f"  {'Trades':<20} {n_trades:>12,}")
    log(f"  {'Alpha':<20} {alpha*100:>11.2f}%")

    # Decision
    cagr_delta = cagr - BASELINE_CAGR
    sharpe_delta = sharpe - BASELINE_SHARPE

    if cagr >= BASELINE_CAGR and sharpe >= BASELINE_SHARPE:
        verdict = "DEPLOY"
        reason = "Both CAGR and Sharpe meet or exceed v1 baseline"
    elif cagr_delta >= -0.01 and sharpe_delta >= -0.05:
        verdict = "CONSIDER DEPLOY"
        reason = f"CAGR within 1% ({cagr_delta*100:+.2f}%) and Sharpe within 0.05 ({sharpe_delta:+.3f})"
    else:
        verdict = "DO NOT DEPLOY, keep v1"
        reasons = []
        if cagr_delta < -0.01:
            reasons.append(f"CAGR degrades {cagr_delta*100:+.2f}% (>1%)")
        if sharpe_delta < -0.05:
            reasons.append(f"Sharpe degrades {sharpe_delta:+.3f} (>0.05)")
        if not reasons:
            reasons.append(f"Mixed results: CAGR {cagr_delta*100:+.2f}%, Sharpe {sharpe_delta:+.3f}")
        reason = "; ".join(reasons)

    log(f"\n  {'='*50}")
    log(f"  VERDICT: {verdict}")
    log(f"  Reason:  {reason}")
    log(f"  {'='*50}")
    log(f"\n  *** Do NOT deploy automatically — human review required ***")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  PRODUCTION MODEL v5d TRAINING")
    log("  LGBM + XGB ensemble, bug-fixed")
    log("=" * 70)

    # STEP 1: Train
    log(f"\n{'='*70}")
    log("STEP 1: TRAIN LGBM + XGB ENSEMBLE")
    log(f"{'='*70}")
    calib_lgbm, calib_xgb, imp, df, feature_cols = train_production_model()

    # STEP 2: Verify (20 random rows)
    verify_predictions(feature_cols)

    n_lgbm = calib_lgbm.calibrated_classifiers_[0].estimator.n_features_in_
    n_xgb = calib_xgb.calibrated_classifiers_[0].estimator.n_features_in_
    log(f"\n  LGBM features: {n_lgbm}  |  XGB features: {n_xgb}")
    log(f"  Feature columns: {len(feature_cols)}")
    assert n_lgbm == len(feature_cols), f"LGBM feature count mismatch: LGBM={n_lgbm}, cols={len(feature_cols)}"
    assert n_xgb <= len(feature_cols), f"XGB has more features than expected: XGB={n_xgb}, cols={len(feature_cols)}"

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"\nFiles created:")
    log(f"  {MODEL_FILE}")
    log(f"  {XGB_MODEL_FILE}")
    log(f"  {PRED_FILE}")
    log(f"  {DATA_DIR / 'imputer.pkl'}")


if __name__ == "__main__":
    main()
