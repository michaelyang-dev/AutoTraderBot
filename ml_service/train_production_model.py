#!/usr/bin/env python3
"""
Train Production Model (Single Model for Live Inference)
=========================================================
Fixes the critical mismatch between model.lgb and predictions.parquet.

Problem: train_model.py creates 20+ walk-forward models and aggregates their
predictions into predictions.parquet, but only saves the LAST window's model
as model.lgb. Live inference uses this single model, producing different
predictions than what the backtest validated.

Solution: Train ONE model on the full dataset (with calibration holdout),
then generate ALL predictions from that single model. This guarantees that
model.lgb and predictions.parquet are perfectly consistent.

Pipeline:
  1. Load features.parquet (80 features including fundamentals)
  2. Split by date: first 80% → LGB training, last 20% → calibration
  3. Train LGBMClassifier on the 80% split
  4. Calibrate with CalibratedClassifierCV (isotonic) on the 20% holdout
  5. Generate predictions for ALL rows using this single model
  6. Verify predictions match: load model, re-predict random rows, compare
  7. Run Path B backtest, report metrics, recommend deploy/reject

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
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).resolve().parent / "data"
INPUT_FILE   = DATA_DIR / "features.parquet"
MODEL_FILE   = DATA_DIR / "model.lgb"
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
    "days_since_earnings", "days_until_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

# V1 baseline (from prior backtest)
BASELINE_CAGR   = 0.2408
BASELINE_SHARPE = 1.655


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target", "in_sp500"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


# ── STEP 1: Load data and train ──────────────────────────────────────────────

def train_production_model():
    """Train one calibrated model on all data with date-based calibration split."""
    log(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Drop NaN only for non-fundamental columns
    feature_cols = get_feature_cols(df)
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    before = len(df)
    df = df.dropna(subset=non_fund_cols + ["target"])
    log(f"Loaded {before:,} → {len(df):,} rows after dropping NaN")

    # ── Survivorship bias filter ─────────────────────────────────────────
    # Only train on stocks that were in the S&P 500 on each date.
    # ETFs are always included. Predictions are generated for ALL rows.
    if "in_sp500" in df.columns:
        n_before_filter = len(df)
        train_df = df[df["in_sp500"] == True].copy()
        log(f"  Survivorship filter: {n_before_filter:,} → {len(train_df):,} rows "
            f"(removed {n_before_filter - len(train_df):,} non-S&P500 rows from training)")
    else:
        log(f"  WARNING: in_sp500 column not found — training on all rows (no survivorship filter)")
        train_df = df.copy()
    log(f"  {df['symbol'].nunique()} symbols  |  "
        f"{df['date'].min().date()} → {df['date'].max().date()}")
    log(f"  Features: {len(feature_cols)} columns")

    # Forward returns for backtest (on full df)
    df = df.sort_values(["symbol", "date"])
    df["fwd_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)

    # Use survivorship-filtered train_df for training/calibration
    train_df = train_df.sort_values(["symbol", "date"])
    train_df["fwd_ret"] = train_df.groupby("symbol")["ret_10d"].shift(-10)

    X_all = df[feature_cols].values
    y_all = df["target"].values

    X_filtered = train_df[feature_cols].values
    y_filtered = train_df["target"].values

    # Date-based split: first 80% of unique dates → train, last 20% → calibration
    # Split based on filtered data dates
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

    # Class balance
    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    log(f"    Train pos/neg: {int(y_train.sum()):,} / {int(len(y_train) - y_train.sum()):,}  "
        f"(scale_pos_weight={scale:.2f})")

    # Train LightGBM
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
    log(f"  Trained with {n_trees} trees (early stopped from {N_TREES})")

    # Calibrate
    log(f"  Calibrating with isotonic regression on {calib_mask.sum():,} rows ...")
    calib_model = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
    calib_model.fit(X_calib, y_calib)

    # Calibration AUC
    calib_probs = calib_model.predict_proba(X_calib)[:, 1]
    calib_auc = roc_auc_score(y_calib, calib_probs)
    log(f"  Calibration AUC: {calib_auc:.4f}")

    # Generate predictions for ALL rows (including non-S&P500) using this single model
    log(f"\n  Generating predictions for ALL {len(df):,} rows ...")
    all_probs = calib_model.predict_proba(X_all)[:, 1]
    all_preds = (all_probs >= 0.5).astype(int)

    df["prob"] = all_probs
    df["pred"] = all_preds

    # Summary stats
    log(f"  Prediction range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    log(f"  Mean: {all_probs.mean():.4f}  Median: {np.median(all_probs):.4f}")
    n_above = int((all_probs >= PROB_THRESH).sum())
    log(f"  Predictions above {PROB_THRESH}: {n_above:,} / {len(all_probs):,} "
        f"({n_above/len(all_probs)*100:.1f}%)")

    # Save model
    log(f"\n  Saving model → {MODEL_FILE.name}")
    joblib.dump(calib_model, str(MODEL_FILE))
    log(f"  Model size: {MODEL_FILE.stat().st_size / 1024:.1f} KB")

    # Save predictions
    save_cols = ["date", "symbol", "target", "prob", "pred", "fwd_ret"]
    df[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow", compression="snappy")
    log(f"  Predictions → {PRED_FILE.name} ({len(df):,} rows)")

    return calib_model, df, feature_cols


# ── STEP 2: Verify match ────────────────────────────────────────────────────

def verify_predictions(model, feature_cols):
    """Load model and predictions independently, re-predict random rows, verify match."""
    log(f"\n{'='*70}")
    log("STEP 2: VERIFICATION — model.lgb vs predictions.parquet")
    log(f"{'='*70}")

    # Load independently
    loaded_model = joblib.load(str(MODEL_FILE))
    preds_df = pd.read_parquet(PRED_FILE)
    features_df = pd.read_parquet(INPUT_FILE)
    features_df["date"] = pd.to_datetime(features_df["date"])
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    # Pick 10 random rows across different dates and symbols
    np.random.seed(42)
    unique_dates = preds_df["date"].unique()
    sample_dates = np.random.choice(unique_dates, size=min(10, len(unique_dates)), replace=False)

    mismatches = 0
    log(f"\n  {'Date':<12} {'Symbol':<8} {'Pred.parquet':>13} {'Re-predicted':>13} {'Delta':>10} {'Match':>6}")
    log(f"  {'─'*12} {'─'*8} {'─'*13} {'─'*13} {'─'*10} {'─'*6}")

    for date in sorted(sample_dates):
        date_rows = preds_df[preds_df["date"] == date]
        if len(date_rows) == 0:
            continue
        row = date_rows.sample(1, random_state=int(pd.Timestamp(date).timestamp()) % 10000).iloc[0]

        # Get features for this date/symbol
        feat_row = features_df[
            (features_df["date"] == row["date"]) &
            (features_df["symbol"] == row["symbol"])
        ]
        if len(feat_row) == 0:
            continue

        X_single = feat_row[feature_cols].values
        re_prob = loaded_model.predict_proba(X_single)[:, 1][0]
        stored_prob = row["prob"]
        delta = abs(re_prob - stored_prob)
        match = delta < 0.001

        status = "OK" if match else "FAIL"
        log(f"  {str(pd.Timestamp(date).date()):<12} {row['symbol']:<8} "
            f"{stored_prob:>13.6f} {re_prob:>13.6f} {delta:>10.6f} {status:>6}")

        if not match:
            mismatches += 1

    log(f"\n  Results: {10 - mismatches}/10 matched (tolerance < 0.001)")

    if mismatches > 0:
        log(f"\n  *** VERIFICATION FAILED: {mismatches} mismatches detected ***")
        log(f"  *** model.lgb and predictions.parquet are NOT consistent ***")
        sys.exit(1)
    else:
        log(f"  *** VERIFICATION PASSED: model.lgb and predictions.parquet are consistent ***")


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
    from backtest_ml import calc_metrics, calc_alpha_beta
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
    log("  PRODUCTION MODEL TRAINING")
    log("  Single model for live inference — fixes prediction mismatch")
    log("=" * 70)

    # STEP 1: Train
    log(f"\n{'='*70}")
    log("STEP 1: TRAIN PRODUCTION MODEL")
    log(f"{'='*70}")
    model, df, feature_cols = train_production_model()

    # STEP 2: Verify
    verify_predictions(model, feature_cols)

    # STEP 3: Backtest
    metrics = run_backtest()

    # STEP 4: Recommendation
    print_recommendation(metrics)

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
