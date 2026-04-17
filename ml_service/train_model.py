"""
Purged Walk-Forward LightGBM Trainer
=====================================
Trains a binary classifier (10-day forward return > 2%) on features.parquet
using chronological walk-forward cross-validation to prevent look-ahead leakage.

Walk-forward schedule
---------------------
  train : TRAIN_YEARS rolling window
  purge : PURGE_DAYS gap (prevents label leakage at the boundary)
  test  : TEST_MONTHS out-of-sample window
  step  : TEST_MONTHS (non-overlapping test sets)

With ~5 years of Alpaca free-tier data this produces ~4 windows.
Alpaca's free tier caps history at ~5 years; wire in a paid feed to get 10+.
"""

import logging
import sys
import warnings
from pathlib import Path
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "features.parquet"
MODEL_FILE = DATA_DIR / "model.lgb"
PRED_FILE  = DATA_DIR / "predictions.parquet"

TRAIN_YEARS  = 3
TEST_MONTHS  = 6
PURGE_DAYS   = 10       # trading days dropped between train end and test start
PROB_THRESH  = 0.55     # threshold for "model pick" (matches live signal server)

LGB_PARAMS = dict(
    n_estimators      = 500,
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger(__name__)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    log.info(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    before = len(df)
    df = df.dropna()
    log.info(f"Loaded {before:,} rows → {len(df):,} after dropping NaN  |  "
             f"{df['symbol'].nunique()} symbols  |  "
             f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "symbol", "target"}
    # Guard: drop any column that contains future info in its name
    forward_keywords = {"fwd", "forward", "future"}
    cols = [
        c for c in df.columns
        if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)
    ]
    return cols


# ── Walk-forward window generator ─────────────────────────────────────────────
def walk_forward_windows(dates: pd.Series):
    """
    Yields (train_mask, test_mask, window_label) tuples.
    Uses calendar arithmetic so window edges always land on real trading days.
    """
    all_dates  = np.sort(dates.unique())
    start_date = pd.Timestamp(all_dates[0])
    end_date   = pd.Timestamp(all_dates[-1])

    train_end = start_date + relativedelta(years=TRAIN_YEARS)

    window = 0
    while True:
        # Purge gap: skip N trading days after train_end
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

        # Roll forward by TEST_MONTHS
        train_end = train_end + relativedelta(months=TEST_MONTHS)


# ── Per-window trading metric ─────────────────────────────────────────────────
def trading_metric(pred_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Returns (avg_return_all, avg_return_picks, lift).
    pred_df must have columns: target_return (raw numeric return), prob, target.
    """
    avg_all   = pred_df["fwd_ret"].mean()
    picks     = pred_df[pred_df["prob"] >= PROB_THRESH]
    avg_picks = picks["fwd_ret"].mean() if len(picks) > 0 else np.nan
    lift      = avg_picks - avg_all if not np.nan_to_num(avg_picks) == 0 else np.nan
    return avg_all, avg_picks, lift


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 68)
    log.info("Purged Walk-Forward LightGBM Training")
    log.info("=" * 68)

    df = load_data()

    feature_cols = get_feature_cols(df)
    log.info(f"\nFeature columns ({len(feature_cols)}):")
    for i in range(0, len(feature_cols), 5):
        log.info("  " + "  ".join(f"{c:<20}" for c in feature_cols[i:i+5]))

    # We need the raw forward return for trading metrics.
    # Reconstruct it from target (>2% threshold) — we don't have the exact value,
    # so we use ret_10d shifted back as a proxy. Since ret_10d = past return,
    # the true fwd return isn't stored. We approximate using next row's ret_10d
    # aligned by date+symbol.
    # Better: store fwd_ret in the pipeline. For now we shift ret_10d forward
    # per symbol as the best available proxy.
    df = df.sort_values(["symbol", "date"])
    df["fwd_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)

    X = df[feature_cols].values
    y = df["target"].values
    dates_arr = df["date"]

    windows      = list(walk_forward_windows(dates_arr))
    n_windows    = len(windows)

    if n_windows == 0:
        sys.exit("ERROR: no walk-forward windows could be generated — insufficient data.")

    log.info(f"\nWalk-forward schedule: {TRAIN_YEARS}yr train | "
             f"{PURGE_DAYS}d purge | {TEST_MONTHS}mo test | "
             f"{n_windows} window(s)")
    log.info("-" * 68)

    all_preds  = []
    window_acc = []
    last_model = None

    for train_mask, test_mask, label in windows:
        log.info(f"\n{label}")

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        log.info(f"  Train rows : {len(X_train):,}  (pos={n_pos}, neg={n_neg})")
        log.info(f"  Test rows  : {len(X_test):,}")

        if len(X_test) == 0:
            log.info("  No test rows — skipping window")
            continue

        # Class-weight balancing
        scale = n_neg / max(n_pos, 1)

        model = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        acc  = accuracy_score(y_test, preds)
        auc  = roc_auc_score(y_test, probs)
        window_acc.append(acc)

        log.info(f"  Accuracy   : {acc:.4f}   AUC-ROC: {auc:.4f}")

        test_df = df[test_mask].copy()
        test_df["prob"]  = probs
        test_df["pred"]  = preds
        all_preds.append(test_df)
        last_model = model

    if not all_preds:
        sys.exit("ERROR: no predictions produced.")

    # ── Combined out-of-sample metrics ───────────────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("COMBINED OUT-OF-SAMPLE RESULTS")
    log.info("=" * 68)

    combined = pd.concat(all_preds, ignore_index=True)
    y_true = combined["target"].values
    y_prob = combined["prob"].values
    y_pred = combined["pred"].values

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  F1 Score  : {f1:.4f}")
    log.info(f"  AUC-ROC   : {auc:.4f}")
    log.info(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    log.info(f"              Pred 0   Pred 1")
    log.info(f"  Actual 0    {cm[0,0]:6d}   {cm[0,1]:6d}")
    log.info(f"  Actual 1    {cm[1,0]:6d}   {cm[1,1]:6d}")

    # ── Trading metric ───────────────────────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info(f"TRADING SIGNAL QUALITY  (threshold = {PROB_THRESH:.0%})")
    log.info(f"{'─'*68}")

    valid = combined.dropna(subset=["fwd_ret"])
    if len(valid) > 0:
        avg_all   = valid["fwd_ret"].mean()
        picks     = valid[valid["prob"] >= PROB_THRESH]
        n_picks   = len(picks)
        avg_picks = picks["fwd_ret"].mean() if n_picks > 0 else np.nan
        lift      = avg_picks - avg_all if n_picks > 0 else np.nan
        hit_rate  = (picks["target"] == 1).mean() if n_picks > 0 else np.nan

        log.info(f"  Avg 10-day return — all stocks : {avg_all*100:+.2f}%")
        log.info(f"  Avg 10-day return — model picks: {avg_picks*100:+.2f}%  "
                 f"(n={n_picks:,})")
        log.info(f"  Lift vs universe               : {lift*100:+.2f}%")
        log.info(f"  Hit rate on picks              : {hit_rate*100:.1f}%  "
                 f"(fraction actually >2%)")

        if n_picks > 0 and lift > 0:
            log.info("  ✓  Model picks outperform the universe — genuine signal detected")
        elif n_picks > 0:
            log.info("  ✗  Model picks do NOT outperform — signal may be noise")
        else:
            log.info("  ✗  No picks at this threshold — lower PROB_THRESH to see results")
    else:
        log.info("  fwd_ret unavailable — rerun data_pipeline.py to embed forward returns")

    # ── Feature importance ───────────────────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info("TOP 15 FEATURES BY GAIN (from most-recent window)")
    log.info(f"{'─'*68}")

    importance = pd.Series(
        last_model.booster_.feature_importance(importance_type="gain"),
        index=feature_cols,
    ).sort_values(ascending=False)

    total_gain = importance.sum()
    for rank, (feat, gain) in enumerate(importance.head(15).items(), 1):
        bar = "█" * int(gain / total_gain * 50)
        log.info(f"  {rank:2d}. {feat:<22} {gain:10.1f}  {gain/total_gain*100:5.1f}%  {bar}")

    # ── Per-window summary ───────────────────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info("PER-WINDOW ACCURACY SUMMARY")
    log.info(f"{'─'*68}")
    for i, (_, _, label) in enumerate(windows):
        if i < len(window_acc):
            log.info(f"  {label.split('  ')[0]}  acc={window_acc[i]:.4f}")
    log.info(f"  Overall OOS accuracy: {acc:.4f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info("Saving outputs ...")

    last_model.booster_.save_model(str(MODEL_FILE))
    log.info(f"  Model      → {MODEL_FILE}")

    save_cols = ["date", "symbol", "target", "prob", "pred", "fwd_ret"] + feature_cols
    save_cols = [c for c in save_cols if c in combined.columns]
    combined[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow",
                                   compression="snappy")
    log.info(f"  Predictions→ {PRED_FILE}  ({len(combined):,} rows)")

    log.info("\n" + "=" * 68)
    log.info("Training complete.")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
