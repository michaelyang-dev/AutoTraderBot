"""
Train ML Medium v2 with Fundamental Features
=============================================
Retrains the calibrated ML Medium model using all 80 features
(57 original + 22 fundamentals + vol_126d).

Same approach as production:
  - CalibratedClassifierCV + isotonic calibration
  - Walk-forward validation (3yr train, 6mo test, 10d purge)
  - Same LightGBM hyperparameters

Saves model as model_v2_with_fundamentals.lgb (joblib).
Generates predictions_v2_fundamentals.parquet.
Does NOT overwrite original model.lgb.

Run with:
    python3 train_model_v2_fundamentals.py
"""

import logging
import sys
import warnings
import time
from pathlib import Path
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "features.parquet"
MODEL_FILE       = DATA_DIR / "model_v2_with_fundamentals.lgb"
PRED_FILE        = DATA_DIR / "predictions_v2_fundamentals.parquet"

TRAIN_YEARS  = 3
TEST_MONTHS  = 6
PURGE_DAYS   = 10
PROB_THRESH  = 0.55
CALIB_SPLIT  = 0.80

N_TREES = 500
EARLY_STOP_PATIENCE = 50
EARLY_STOP_ROUNDS = 100
FINAL_MODEL_TREES = 100

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

# Fundamental feature columns (LightGBM handles NaN natively)
FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "days_until_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger(__name__)


def load_data():
    log.info(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    before = len(df)
    # Drop NaN only for non-fundamental columns (fundamentals can be NaN for ETFs)
    exclude = {"date", "symbol", "target"}
    forward_keywords = {"fwd", "forward", "future"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)
    ]
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund_cols + ["target"])
    log.info(f"Loaded {before:,} rows → {len(df):,} after dropping NaN  |  "
             f"{df['symbol'].nunique()} symbols  |  "
             f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df, feature_cols


def walk_forward_windows(dates: pd.Series):
    all_dates  = np.sort(dates.unique())
    start_date = pd.Timestamp(all_dates[0])
    end_date   = pd.Timestamp(all_dates[-1])
    train_end = start_date + relativedelta(years=TRAIN_YEARS)

    window = 0
    while True:
        purge_idx = np.searchsorted(all_dates, np.datetime64(train_end, "ns"))
        purge_idx = min(purge_idx + PURGE_DAYS, len(all_dates) - 1)
        test_start = pd.Timestamp(all_dates[purge_idx])
        test_end   = test_start + relativedelta(months=TEST_MONTHS)

        if test_start >= end_date:
            break

        test_end = min(test_end, end_date)
        window  += 1

        train_mask = (dates >= start_date) & (dates < train_end)
        test_mask  = (dates >= test_start) & (dates <= test_end)

        label = (f"W{window:02d}  "
                 f"train {start_date.date()}→{train_end.date()}  "
                 f"test  {test_start.date()}→{test_end.date()}")
        yield train_mask, test_mask, label

        train_end = train_end + relativedelta(months=TEST_MONTHS)


def main():
    t0 = time.perf_counter()
    log.info("=" * 68)
    log.info("ML Medium v2 Training — With Fundamental Features")
    log.info("=" * 68)

    df, feature_cols = load_data()

    log.info(f"\nFeature columns ({len(feature_cols)}):")
    fund_in = [c for c in feature_cols if c in FUNDAMENTAL_FEATURE_COLS]
    non_fund = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    log.info(f"  Original features: {len(non_fund)}")
    log.info(f"  Fundamental features: {len(fund_in)}")
    for i in range(0, len(feature_cols), 6):
        log.info("  " + "  ".join(f"{c:<22}" for c in feature_cols[i:i+6]))

    # Reconstruct forward return proxy
    df = df.sort_values(["symbol", "date"])
    df["fwd_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)

    X = df[feature_cols].values
    y = df["target"].values
    dates_arr = df["date"]

    windows = list(walk_forward_windows(dates_arr))
    n_windows = len(windows)

    if n_windows == 0:
        sys.exit("ERROR: no walk-forward windows.")

    log.info(f"\nWalk-forward schedule: {TRAIN_YEARS}yr train | "
             f"{PURGE_DAYS}d purge | {TEST_MONTHS}mo test | "
             f"{n_windows} window(s)")
    log.info("-" * 68)

    all_preds_calib = []
    window_aucs = []
    last_model_calib = None

    for train_mask, test_mask, label in windows:
        log.info(f"\n{label}")

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        n_total = len(X_train)
        n_pos = int(y_train.sum())
        n_neg = n_total - n_pos
        log.info(f"  Train: {n_total:,} (pos={n_pos}, neg={n_neg})  |  Test: {len(X_test):,}")

        if len(X_test) == 0:
            log.info("  No test rows — skipping")
            continue

        # Split: 80% LGB train, 20% calibration
        split_idx = int(n_total * CALIB_SPLIT)
        X_lgb,   y_lgb   = X_train[:split_idx], y_train[:split_idx]
        X_calib, y_calib  = X_train[split_idx:], y_train[split_idx:]

        scale_lgb = (len(y_lgb) - y_lgb.sum()) / max(y_lgb.sum(), 1)

        model_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale_lgb)
        model_lgb.fit(
            X_lgb, y_lgb,
            eval_set=[(X_calib, y_calib)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        calib_model = CalibratedClassifierCV(
            model_lgb, method="isotonic", cv="prefit",
        )
        calib_model.fit(X_calib, y_calib)

        probs_calib = calib_model.predict_proba(X_test)[:, 1]
        preds_calib = (probs_calib >= 0.5).astype(int)
        acc_c = accuracy_score(y_test, preds_calib)
        auc_c = roc_auc_score(y_test, probs_calib)
        n_trees = model_lgb.booster_.num_trees()
        window_aucs.append(auc_c)
        log.info(f"  Calibrated: acc={acc_c:.4f}  AUC={auc_c:.4f}  trees={n_trees}")
        log.info(f"  Pred range: [{probs_calib.min():.4f}, {probs_calib.max():.4f}]  "
                 f"picks(>0.55)={int((probs_calib >= 0.55).sum())}")

        test_df = df[test_mask].copy()
        test_df["prob"] = probs_calib
        test_df["pred"] = preds_calib
        all_preds_calib.append(test_df)
        last_model_calib = calib_model

    if not all_preds_calib:
        sys.exit("ERROR: no predictions produced.")

    # ── Combined OOS results ─────────────────────────────────────────────────
    combined = pd.concat(all_preds_calib, ignore_index=True)
    yt = combined["target"].values
    yp = combined["prob"].values
    yh = combined["pred"].values

    acc  = accuracy_score(yt, yh)
    prec = precision_score(yt, yh, zero_division=0)
    rec  = recall_score(yt, yh, zero_division=0)
    f1   = f1_score(yt, yh, zero_division=0)
    auc  = roc_auc_score(yt, yp)
    cm   = confusion_matrix(yt, yh)

    log.info(f"\n{'='*68}")
    log.info("COMBINED OOS RESULTS — CALIBRATED (ISOTONIC)")
    log.info(f"{'='*68}")
    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  F1 Score  : {f1:.4f}")
    log.info(f"  AUC-ROC   : {auc:.4f}")
    log.info(f"\n  Confusion Matrix:")
    log.info(f"              Pred 0   Pred 1")
    log.info(f"  Actual 0    {cm[0,0]:6d}   {cm[0,1]:6d}")
    log.info(f"  Actual 1    {cm[1,0]:6d}   {cm[1,1]:6d}")

    # Per-window AUC
    log.info(f"\n  Per-window AUC: {' | '.join(f'{a:.4f}' for a in window_aucs)}")
    log.info(f"  Mean AUC: {np.mean(window_aucs):.4f}")

    # Trading signal quality
    valid = combined.dropna(subset=["fwd_ret"])
    if len(valid) > 0:
        avg_all   = valid["fwd_ret"].mean()
        picks     = valid[valid["prob"] >= PROB_THRESH]
        n_picks   = len(picks)
        avg_picks = picks["fwd_ret"].mean() if n_picks > 0 else np.nan
        lift      = avg_picks - avg_all if n_picks > 0 else np.nan
        hit_rate  = (picks["target"] == 1).mean() if n_picks > 0 else np.nan

        log.info(f"\n  Trading Signal Quality (threshold = {PROB_THRESH:.0%}):")
        log.info(f"  Avg 10d return — all: {avg_all*100:+.2f}%  |  "
                 f"picks: {avg_picks*100:+.2f}% (n={n_picks:,})")
        log.info(f"  Lift: {lift*100:+.2f}%  |  Hit rate: {hit_rate*100:.1f}%")

    # Probability distribution
    log.info(f"\n  Predicted Probability Distribution:")
    for lo, hi in [(0,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.55),(0.55,0.6),(0.6,0.7),(0.7,1.0)]:
        mask = (yp >= lo) & (yp < hi)
        n = mask.sum()
        pct = n / len(yp) * 100
        wr = yt[mask].mean() * 100 if n > 0 else 0
        log.info(f"    [{lo:.2f}, {hi:.2f})  n={n:>6,}  ({pct:>5.1f}%)  win_rate={wr:.1f}%")

    # ── Feature importance ───────────────────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info("TOP 25 FEATURES BY GAIN")
    log.info(f"{'─'*68}")

    # Extract LGB model from the calibrated wrapper
    lgb_model = last_model_calib.calibrated_classifiers_[0].estimator
    importance = pd.Series(
        lgb_model.booster_.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)

    total_gain = importance.sum()
    fund_gain = sum(importance.get(c, 0) for c in FUNDAMENTAL_FEATURE_COLS)
    price_gain = total_gain - fund_gain

    for rank, (feat, gain) in enumerate(importance.head(25).items(), 1):
        is_fund = "FUND" if feat in FUNDAMENTAL_FEATURE_COLS else "    "
        pct = gain / total_gain * 100
        bar = "█" * int(pct * 2)
        log.info(f"  {rank:2d}. [{is_fund}] {feat:<28} {pct:5.1f}%  {bar}")

    n_fund_top25 = sum(1 for f in importance.head(25).index if f in FUNDAMENTAL_FEATURE_COLS)
    log.info(f"\n  Fundamental features in top 25: {n_fund_top25}/25")
    log.info(f"  Fundamental gain share: {fund_gain/total_gain*100:.1f}% "
             f"({fund_gain:.0f} / {total_gain:.0f})")
    log.info(f"  Price/tech gain share:  {price_gain/total_gain*100:.1f}%")

    # ── Retrain final model on ALL data ──────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info(f"Retraining final model on ALL data ({FINAL_MODEL_TREES} fixed trees) ...")

    n_pos_all = int(y.sum())
    n_neg_all = len(y) - n_pos_all
    scale_all = n_neg_all / max(n_pos_all, 1)

    # Split all data 80/20 for LGB/calibration
    split_all = int(len(X) * CALIB_SPLIT)
    X_lgb_all, y_lgb_all = X[:split_all], y[:split_all]
    X_cal_all, y_cal_all = X[split_all:], y[split_all:]

    scale_lgb_all = (len(y_lgb_all) - y_lgb_all.sum()) / max(y_lgb_all.sum(), 1)

    final_params = dict(LGB_PARAMS)
    final_params["n_estimators"] = FINAL_MODEL_TREES
    final_lgb = lgb.LGBMClassifier(**final_params, scale_pos_weight=scale_lgb_all)
    final_lgb.fit(X_lgb_all, y_lgb_all)

    final_calib = CalibratedClassifierCV(
        final_lgb, method="isotonic", cv="prefit",
    )
    final_calib.fit(X_cal_all, y_cal_all)

    final_trees = final_lgb.booster_.num_trees()
    log.info(f"  Final calibrated model: {final_trees} trees")

    # Verify on recent data
    recent_mask = dates_arr >= (dates_arr.max() - pd.Timedelta(days=180))
    if recent_mask.sum() > 0:
        recent_probs = final_calib.predict_proba(X[recent_mask])[:, 1]
        log.info(f"  Recent pred range: [{recent_probs.min():.4f}, "
                 f"{recent_probs.max():.4f}]  "
                 f"picks(>0.55)={int((recent_probs >= 0.55).sum())}"
                 f"/{recent_mask.sum()}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info("Saving outputs ...")

    joblib.dump(final_calib, str(MODEL_FILE))
    log.info(f"  Model → {MODEL_FILE}")

    save_cols = ["date", "symbol", "target", "prob", "pred", "fwd_ret"] + feature_cols
    save_cols = [c for c in save_cols if c in combined.columns]
    combined[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow",
                                   compression="snappy")
    log.info(f"  Predictions → {PRED_FILE} ({len(combined):,} rows)")

    elapsed = time.perf_counter() - t0
    log.info(f"\n{'='*68}")
    log.info(f"Training complete. Runtime: {elapsed:.0f}s")
    log.info(f"{'='*68}")


if __name__ == "__main__":
    main()
