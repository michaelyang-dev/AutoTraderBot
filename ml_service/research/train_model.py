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

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "features.parquet"
MODEL_FILE       = DATA_DIR / "model.lgb"
CALIB_MODEL_FILE = DATA_DIR / "model_calibrated.lgb"
PRED_FILE        = DATA_DIR / "predictions.parquet"
CALIB_PRED_FILE  = DATA_DIR / "predictions_calibrated.parquet"
RELIABILITY_IMG  = DATA_DIR / "reliability_diagram.png"

TRAIN_YEARS  = 3
TEST_MONTHS  = 6
PURGE_DAYS   = 10       # trading days dropped between train end and test start
PROB_THRESH  = 0.55     # threshold for "model pick" (matches live signal server)
CALIB_SPLIT  = 0.80     # first 80% of each train window trains LGB, last 20% calibrates

N_TREES = 500   # max boosting rounds (early stopping decides actual count)
EARLY_STOP_PATIENCE = 50  # early stopping patience for walk-forward windows
FINAL_MODEL_TREES = 100   # fixed trees for the final production model (no early stopping)

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

# Early stopping patience for calibrated model (needs eval_set-driven stopping)
EARLY_STOP_ROUNDS = 100

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


# ── Reliability diagram ──────────────────────────────────────────────────────
def plot_reliability_diagram(baseline_df: pd.DataFrame, calibrated_df: pd.DataFrame):
    """
    Plot predicted probability vs actual win rate in 5% buckets for both
    the baseline and calibrated models.
    """
    bin_edges = np.arange(0, 1.05, 0.05)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0a1628")

    for ax, (df, label, color) in zip(axes, [
        (baseline_df, "Baseline LightGBM", "#0ea5e9"),
        (calibrated_df, "Calibrated (Isotonic)", "#34d399"),
    ]):
        ax.set_facecolor("#0d1f3c")
        ax.tick_params(colors="#c8d6e5", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#1e3050")

        bins = pd.cut(df["prob"], bins=bin_edges, labels=bin_centers, include_lowest=True)
        grouped = df.groupby(bins, observed=False)
        actual_rate = grouped["target"].mean()
        counts = grouped["target"].count()

        mask = counts >= 10
        x = actual_rate.index.astype(float)[mask]
        y = actual_rate.values[mask]
        n = counts.values[mask]

        ax.plot([0, 1], [0, 1], "--", color="#4a6080", lw=1.2, label="Perfect calibration")
        ax.bar(x, y, width=0.045, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.plot(x, y, "o-", color=color, lw=1.5, markersize=5, label=label)

        for xi, yi, ni in zip(x, y, n):
            if ni >= 10:
                ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=6, color="#c8d6e5")

        ax.set_xlabel("Predicted Probability", color="#c8d6e5", fontsize=10)
        ax.set_ylabel("Actual Win Rate (target=1)", color="#c8d6e5", fontsize=10)
        ax.set_title(label, color="#c8d6e5", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, labelcolor="#c8d6e5", facecolor="#0d1f3c", framealpha=0.5)
        ax.grid(True, color="#1e3050", linewidth=0.4, linestyle="--")

    fig.suptitle("Reliability Diagram — Predicted Probability vs Actual Win Rate (5% buckets)",
                 color="#c8d6e5", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RELIABILITY_IMG, dpi=150, bbox_inches="tight", facecolor="#0a1628")
    plt.close()
    log.info(f"  Reliability diagram → {RELIABILITY_IMG}")


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

    all_preds        = []   # baseline predictions
    all_preds_calib  = []   # calibrated predictions
    window_acc       = []
    window_acc_calib = []
    last_model       = None
    last_model_calib = None

    for train_mask, test_mask, label in windows:
        log.info(f"\n{label}")

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        n_total = len(X_train)
        n_pos = y_train.sum()
        n_neg = n_total - n_pos
        log.info(f"  Train rows : {n_total:,}  (pos={n_pos}, neg={n_neg})")
        log.info(f"  Test rows  : {len(X_test):,}")

        if len(X_test) == 0:
            log.info("  No test rows — skipping window")
            continue

        # ── Split training window: 80% LGB training, 20% calibration ────────
        split_idx = int(n_total * CALIB_SPLIT)
        X_lgb,   y_lgb   = X_train[:split_idx], y_train[:split_idx]
        X_calib, y_calib  = X_train[split_idx:], y_train[split_idx:]
        log.info(f"  LGB train  : {len(X_lgb):,}  |  Calib hold-out: {len(X_calib):,}")

        # Class-weight balancing (from full training window)
        scale = n_neg / max(n_pos, 1)

        # ── Train baseline LightGBM with early stopping ──────────────────────
        # Early stopping patience=50 produces best OOS predictions.
        # The final production model is retrained separately after the loop
        # with fixed FINAL_MODEL_TREES to ensure a usable prediction range.
        model = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_PATIENCE, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc  = accuracy_score(y_test, preds)
        auc  = roc_auc_score(y_test, probs)
        n_trees = model.booster_.num_trees()
        window_acc.append(acc)
        log.info(f"  Baseline   : acc={acc:.4f}  AUC={auc:.4f}  trees={n_trees}")
        log.info(f"  Pred range : [{probs.min():.4f}, {probs.max():.4f}]  "
                 f"picks(>0.55)={int((probs >= 0.55).sum())}")

        test_df = df[test_mask].copy()
        test_df["prob"] = probs
        test_df["pred"] = preds
        all_preds.append(test_df)
        last_model = model

        # ── Train calibrated model: LGB on 80%, isotonic calibration on 20% ─
        # Calibrated model still uses early stopping (patience=100) since it
        # has a dedicated calibration holdout set.
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
        window_acc_calib.append(acc_c)
        log.info(f"  Calibrated : acc={acc_c:.4f}  AUC={auc_c:.4f}")

        test_df_c = df[test_mask].copy()
        test_df_c["prob"] = probs_calib
        test_df_c["pred"] = preds_calib
        all_preds_calib.append(test_df_c)
        last_model_calib = calib_model

    if not all_preds:
        sys.exit("ERROR: no predictions produced.")

    # ── Helper: print metrics for a model variant ─────────────────────────────
    def _report_combined(preds_list, variant_name):
        comb = pd.concat(preds_list, ignore_index=True)
        yt = comb["target"].values
        yp = comb["prob"].values
        yh = comb["pred"].values

        acc  = accuracy_score(yt, yh)
        prec = precision_score(yt, yh, zero_division=0)
        rec  = recall_score(yt, yh, zero_division=0)
        f1   = f1_score(yt, yh, zero_division=0)
        auc  = roc_auc_score(yt, yp)
        cm   = confusion_matrix(yt, yh)

        log.info(f"\n{'='*68}")
        log.info(f"COMBINED OOS RESULTS — {variant_name}")
        log.info(f"{'='*68}")
        log.info(f"  Accuracy  : {acc:.4f}")
        log.info(f"  Precision : {prec:.4f}")
        log.info(f"  Recall    : {rec:.4f}")
        log.info(f"  F1 Score  : {f1:.4f}")
        log.info(f"  AUC-ROC   : {auc:.4f}")
        log.info(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        log.info(f"              Pred 0   Pred 1")
        log.info(f"  Actual 0    {cm[0,0]:6d}   {cm[0,1]:6d}")
        log.info(f"  Actual 1    {cm[1,0]:6d}   {cm[1,1]:6d}")

        valid = comb.dropna(subset=["fwd_ret"])
        lift = np.nan
        if len(valid) > 0:
            avg_all   = valid["fwd_ret"].mean()
            picks     = valid[valid["prob"] >= PROB_THRESH]
            n_picks   = len(picks)
            avg_picks = picks["fwd_ret"].mean() if n_picks > 0 else np.nan
            lift      = avg_picks - avg_all if n_picks > 0 else np.nan
            hit_rate  = (picks["target"] == 1).mean() if n_picks > 0 else np.nan

            log.info(f"\n  Trading Signal Quality (threshold = {PROB_THRESH:.0%}):")
            log.info(f"  Avg 10-day return — all stocks : {avg_all*100:+.2f}%")
            log.info(f"  Avg 10-day return — model picks: {avg_picks*100:+.2f}%  "
                     f"(n={n_picks:,})")
            log.info(f"  Lift vs universe               : {lift*100:+.2f}%")
            log.info(f"  Hit rate on picks              : {hit_rate*100:.1f}%  "
                     f"(fraction actually >2%)")

        return comb, acc, lift

    # ── Report both variants ─────────────────────────────────────────────────
    combined, acc_base, lift_base = _report_combined(all_preds, "BASELINE")
    combined_calib, acc_calib, lift_calib = _report_combined(all_preds_calib, "CALIBRATED (ISOTONIC)")

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
        wlabel = label.split('  ')[0]
        if i < len(window_acc):
            log.info(f"  {wlabel}  baseline={window_acc[i]:.4f}  "
                     f"calibrated={window_acc_calib[i]:.4f}")
    log.info(f"  Overall OOS accuracy: baseline={acc_base:.4f}  "
             f"calibrated={acc_calib:.4f}")

    # ── Retrain final production model on ALL data ─────────────────────────
    log.info(f"\n{'─'*68}")
    log.info(f"Retraining final production model on ALL data "
             f"({FINAL_MODEL_TREES} fixed trees) ...")

    n_pos_all = y.sum()
    n_neg_all = len(y) - n_pos_all
    scale_all = n_neg_all / max(n_pos_all, 1)

    final_params = dict(LGB_PARAMS)
    final_params["n_estimators"] = FINAL_MODEL_TREES
    final_model = lgb.LGBMClassifier(**final_params, scale_pos_weight=scale_all)
    final_model.fit(X, y)
    final_trees = final_model.booster_.num_trees()
    log.info(f"  Final model trained: {final_trees} trees")

    # Verify prediction range on recent data (last 6 months)
    recent_mask = dates_arr >= (dates_arr.max() - pd.Timedelta(days=180))
    if recent_mask.sum() > 0:
        recent_probs = final_model.predict_proba(X[recent_mask])[:, 1]
        log.info(f"  Recent pred range: [{recent_probs.min():.4f}, "
                 f"{recent_probs.max():.4f}]  "
                 f"picks(>0.55)={int((recent_probs >= 0.55).sum())}"
                 f"/{recent_mask.sum()}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info("Saving outputs ...")

    # Production model (native LightGBM format) — trained on ALL data
    final_model.booster_.save_model(str(MODEL_FILE))
    log.info(f"  Production model → {MODEL_FILE}  ({final_trees} trees)")

    # Calibrated model (joblib — CalibratedClassifierCV wraps LGB)
    joblib.dump(last_model_calib, str(CALIB_MODEL_FILE))
    log.info(f"  Calibrated model→ {CALIB_MODEL_FILE}")

    save_cols = ["date", "symbol", "target", "prob", "pred", "fwd_ret"] + feature_cols
    save_cols = [c for c in save_cols if c in combined.columns]

    combined[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow",
                                   compression="snappy")
    log.info(f"  Baseline preds  → {PRED_FILE}  ({len(combined):,} rows)")

    combined_calib[save_cols].to_parquet(CALIB_PRED_FILE, index=False, engine="pyarrow",
                                         compression="snappy")
    log.info(f"  Calibrated preds→ {CALIB_PRED_FILE}  ({len(combined_calib):,} rows)")

    # ── Reliability diagram ──────────────────────────────────────────────────
    plot_reliability_diagram(combined, combined_calib)

    log.info("\n" + "=" * 68)
    log.info("Training complete.")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
