"""
Purged Walk-Forward LightGBM Trainer — ML Slow (30-day predictions)
====================================================================
Trains a calibrated binary classifier:
    target = 1 if forward 30-day return > 5%, else 0

Uses the same 34 features, same 60-symbol universe, and same walk-forward
validation as the ML Medium model (train_model.py), but with a longer
prediction horizon.

Walk-forward schedule
---------------------
  train : 3 years rolling window
  purge : 10 trading days gap
  test  : 6 months out-of-sample
  step  : 6 months (non-overlapping test sets)
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
import yfinance as yf
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
MODEL_FILE = DATA_DIR / "model_slow.lgb"
PRED_FILE  = DATA_DIR / "predictions_slow.parquet"

# Target definition: 30-day forward return > 5%
FWD_DAYS      = 30
FWD_THRESHOLD = 0.05

TRAIN_YEARS  = 3
TEST_MONTHS  = 6
PURGE_DAYS   = 10       # trading days dropped between train end and test start
PROB_THRESH  = 0.55     # threshold for "model pick"
CALIB_SPLIT  = 0.80     # first 80% of each train window trains LGB, last 20% calibrates

N_TREES = 500
EARLY_STOP_PATIENCE = 50
EARLY_STOP_ROUNDS = 100  # for calibrated model

LGB_PARAMS = dict(
    n_estimators      = N_TREES,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 20,
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

# Universe (same 60 symbols as ML Medium)
UNIVERSE = [
    "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
    "CRM","ORCL","ADBE","CSCO","QCOM","COST","WMT","HD","LOW",
    "LLY","JNJ","ABBV","BAC","GS","MS","CVX","XOM","CAT","DE","BA",
    "XLE","XLF","XLV","XLI","XLK","XLY","XLP","XLU","XLRE","XLB","XLC",
    "EWZ","EWJ","FXI","INDA","EFA","EEM","VGK","VWO","IEFA",
    "GLD","SLV","USO","DBC",
    "TLT","HYG","LQD",
    "VIXY",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger(__name__)


# ── Download close prices for target computation ─────────────────────────────
def download_close_prices():
    """Download close prices for all universe symbols via yfinance."""
    log.info("Downloading close prices for 30-day forward target computation ...")
    all_closes = {}
    batch_size = 10
    symbols = list(UNIVERSE)

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        ticker_str = " ".join(batch)
        try:
            data = yf.download(ticker_str, start="2010-01-01", end="2027-01-01",
                               progress=False, auto_adjust=True)
            if "Close" in data.columns.get_level_values(0):
                close = data["Close"]
            else:
                close = data
            for sym in batch:
                if sym in close.columns:
                    series = close[sym].dropna()
                    if len(series) > 0:
                        all_closes[sym] = series
        except Exception as e:
            log.warning(f"  Failed batch {batch}: {e}")

    log.info(f"  Downloaded close prices for {len(all_closes)}/{len(symbols)} symbols")
    return all_closes


# ── Load data and compute new target ─────────────────────────────────────────
def load_data():
    log.info(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Download close prices to compute 30-day forward return target
    close_prices = download_close_prices()

    # Compute 30-day forward return for each symbol
    log.info(f"Computing {FWD_DAYS}-day forward return target (threshold > {FWD_THRESHOLD:.0%}) ...")
    fwd_ret_list = []

    for sym in df["symbol"].unique():
        sym_mask = df["symbol"] == sym
        sym_dates = df.loc[sym_mask, "date"]

        if sym not in close_prices:
            # No price data — mark as NaN
            fwd_ret_list.append(pd.Series(np.nan, index=sym_dates.index))
            continue

        prices = close_prices[sym]
        prices.index = pd.to_datetime(prices.index)

        # For each date in the features, find the close price FWD_DAYS later
        fwd_rets = []
        for date in sym_dates:
            # Find current price
            if date not in prices.index:
                fwd_rets.append(np.nan)
                continue
            cur_price = prices[date]

            # Find price FWD_DAYS trading days later
            future_prices = prices[prices.index > date]
            if len(future_prices) < FWD_DAYS:
                fwd_rets.append(np.nan)
                continue
            future_price = future_prices.iloc[FWD_DAYS - 1]
            fwd_ret = (future_price / cur_price) - 1.0
            fwd_rets.append(fwd_ret)

        fwd_ret_list.append(pd.Series(fwd_rets, index=sym_dates.index))

    df["fwd_ret_30d"] = pd.concat(fwd_ret_list).sort_index()
    df["target_slow"] = (df["fwd_ret_30d"] > FWD_THRESHOLD).astype("Int8")

    # Drop rows with no target (last 30 days per symbol, or missing prices)
    before = len(df)
    df = df[df["target_slow"].notna()].copy()
    df = df.dropna(subset=["target_slow"])
    # Also drop rows with NaN features
    feature_cols = get_feature_cols(df)
    df = df.dropna(subset=feature_cols)

    n_pos = (df["target_slow"] == 1).sum()
    n_neg = (df["target_slow"] == 0).sum()
    log.info(f"Loaded {before:,} rows → {len(df):,} after computing 30d target  |  "
             f"{df['symbol'].nunique()} symbols  |  "
             f"{df['date'].min().date()} → {df['date'].max().date()}")
    log.info(f"Target balance: {n_pos:,} positive ({n_pos/(n_pos+n_neg)*100:.1f}%) / "
             f"{n_neg:,} negative ({n_neg/(n_pos+n_neg)*100:.1f}%)")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "symbol", "target", "target_slow", "fwd_ret_30d", "fwd_ret"}
    forward_keywords = {"fwd", "forward", "future"}
    cols = [
        c for c in df.columns
        if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)
    ]
    return cols


# ── Walk-forward window generator ─────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 68)
    log.info("ML Slow Trainer — 30-day forward predictions (calibrated)")
    log.info("=" * 68)

    df = load_data()

    feature_cols = get_feature_cols(df)
    log.info(f"\nFeature columns ({len(feature_cols)}):")
    for i in range(0, len(feature_cols), 5):
        log.info("  " + "  ".join(f"{c:<20}" for c in feature_cols[i:i+5]))

    X = df[feature_cols].values
    y = df["target_slow"].values.astype(int)
    dates_arr = df["date"]

    windows   = list(walk_forward_windows(dates_arr))
    n_windows = len(windows)

    if n_windows == 0:
        sys.exit("ERROR: no walk-forward windows could be generated — insufficient data.")

    log.info(f"\nWalk-forward schedule: {TRAIN_YEARS}yr train | "
             f"{PURGE_DAYS}d purge | {TEST_MONTHS}mo test | "
             f"{n_windows} window(s)")
    log.info("-" * 68)

    all_preds_calib  = []
    window_auc       = []
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

        # Split: 80% LGB training, 20% calibration
        split_idx = int(n_total * CALIB_SPLIT)
        X_lgb,   y_lgb   = X_train[:split_idx], y_train[:split_idx]
        X_calib, y_calib  = X_train[split_idx:], y_train[split_idx:]
        log.info(f"  LGB train  : {len(X_lgb):,}  |  Calib hold-out: {len(X_calib):,}")

        scale_lgb = (len(y_lgb) - y_lgb.sum()) / max(y_lgb.sum(), 1)

        # Train LightGBM with early stopping on calibration holdout
        model_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale_lgb)
        model_lgb.fit(
            X_lgb, y_lgb,
            eval_set=[(X_calib, y_calib)],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # Calibrate with isotonic regression
        calib_model = CalibratedClassifierCV(
            model_lgb, method="isotonic", cv="prefit",
        )
        calib_model.fit(X_calib, y_calib)

        probs_calib = calib_model.predict_proba(X_test)[:, 1]
        preds_calib = (probs_calib >= 0.5).astype(int)
        acc_c = accuracy_score(y_test, preds_calib)
        auc_c = roc_auc_score(y_test, probs_calib)
        window_auc.append(auc_c)

        n_picks = int((probs_calib >= PROB_THRESH).sum())
        log.info(f"  Calibrated : acc={acc_c:.4f}  AUC={auc_c:.4f}")
        log.info(f"  Pred range : [{probs_calib.min():.4f}, {probs_calib.max():.4f}]  "
                 f"picks(>0.55)={n_picks}/{len(probs_calib)}")

        test_df_c = df[test_mask].copy()
        test_df_c["prob"] = probs_calib
        test_df_c["pred"] = preds_calib
        all_preds_calib.append(test_df_c)
        last_model_calib = calib_model

    if not all_preds_calib:
        sys.exit("ERROR: no predictions produced.")

    # ── Combined OOS results ─────────────────────────────────────────────────
    combined = pd.concat(all_preds_calib, ignore_index=True)
    yt = combined["target_slow"].values.astype(int)
    yp = combined["prob"].values
    yh = combined["pred"].values

    acc  = accuracy_score(yt, yh)
    prec = precision_score(yt, yh, zero_division=0)
    rec  = recall_score(yt, yh, zero_division=0)
    f1   = f1_score(yt, yh, zero_division=0)
    auc  = roc_auc_score(yt, yp)
    cm   = confusion_matrix(yt, yh)

    log.info(f"\n{'='*68}")
    log.info(f"COMBINED OOS RESULTS — CALIBRATED ML SLOW (30-day)")
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

    # Trading signal quality
    valid = combined.dropna(subset=["fwd_ret_30d"])
    if len(valid) > 0:
        avg_all   = valid["fwd_ret_30d"].mean()
        picks     = valid[valid["prob"] >= PROB_THRESH]
        n_picks   = len(picks)
        avg_picks = picks["fwd_ret_30d"].mean() if n_picks > 0 else np.nan
        lift      = avg_picks - avg_all if n_picks > 0 else np.nan
        hit_rate  = (picks["target_slow"] == 1).mean() if n_picks > 0 else np.nan

        log.info(f"\n  Trading Signal Quality (threshold = {PROB_THRESH:.0%}):")
        log.info(f"  Avg 30-day return — all stocks : {avg_all*100:+.2f}%")
        log.info(f"  Avg 30-day return — model picks: {avg_picks*100:+.2f}%  "
                 f"(n={n_picks:,})")
        log.info(f"  Lift vs universe               : {lift*100:+.2f}%")
        log.info(f"  Hit rate on picks              : {hit_rate*100:.1f}%  "
                 f"(fraction actually >5%)")

    # Per-window AUC summary
    log.info(f"\n{'─'*68}")
    log.info("PER-WINDOW AUC SUMMARY")
    log.info(f"{'─'*68}")
    for i, (_, _, label) in enumerate(windows):
        wlabel = label.split('  ')[0]
        if i < len(window_auc):
            log.info(f"  {wlabel}  AUC={window_auc[i]:.4f}")
    log.info(f"  Overall OOS AUC: {auc:.4f}")

    # Feature importance
    log.info(f"\n{'─'*68}")
    log.info("TOP 15 FEATURES BY GAIN (from most-recent window)")
    log.info(f"{'─'*68}")
    # Get the underlying LGBMClassifier from the calibrated model
    base_lgb = last_model_calib.estimators_[0].estimator if hasattr(last_model_calib, 'estimators_') else None
    if base_lgb and hasattr(base_lgb, 'booster_'):
        importance = pd.Series(
            base_lgb.booster_.feature_importance(importance_type="gain"),
            index=feature_cols,
        ).sort_values(ascending=False)
        total_gain = importance.sum()
        for rank, (feat, gain) in enumerate(importance.head(15).items(), 1):
            bar = "█" * int(gain / total_gain * 50)
            log.info(f"  {rank:2d}. {feat:<22} {gain:10.1f}  {gain/total_gain*100:5.1f}%  {bar}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    log.info(f"\n{'─'*68}")
    log.info("Saving outputs ...")

    # Save calibrated model (joblib)
    joblib.dump(last_model_calib, str(MODEL_FILE))
    log.info(f"  Calibrated model → {MODEL_FILE}")

    # Save predictions
    save_cols = ["date", "symbol", "target_slow", "prob", "pred", "fwd_ret_30d"] + feature_cols
    save_cols = [c for c in save_cols if c in combined.columns]
    combined[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow",
                                   compression="snappy")
    log.info(f"  Predictions      → {PRED_FILE}  ({len(combined):,} rows)")

    log.info("\n" + "=" * 68)
    log.info("ML Slow training complete.")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
