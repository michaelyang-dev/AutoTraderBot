"""
ML Slow v3 Trainer — Relative Outperformance Target
=====================================================
Target: 1 if (stock 30-day return - SPY 30-day return) > 3%

Previous versions used absolute return > 5%, which produced zero picks
during low-volatility regimes (2023-2024). Relative targets capture
signal when stocks outperform the index even in flat markets.

Same 58 features as v2. Same walk-forward CV and calibration approach.

Run with:
    python3 train_model_slow_v3.py
"""

import logging
import sys
import warnings
from pathlib import Path
from dateutil.relativedelta import relativedelta

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "features.parquet"
MODEL_FILE = DATA_DIR / "model_slow_v3.lgb"
PRED_FILE  = DATA_DIR / "predictions_slow_v3.parquet"

# Target: stock 30-day return minus SPY 30-day return > 3%
FWD_DAYS         = 30
EXCESS_THRESHOLD = 0.03

TRAIN_YEARS  = 3
TEST_MONTHS  = 6
PURGE_DAYS   = 10
PROB_THRESH  = 0.55
CALIB_SPLIT  = 0.80

N_TREES = 500
EARLY_STOP_ROUNDS = 100

# Same LGB params as ML Medium
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

# Universe — same 60 symbols as ML Medium (exclude feature-only symbols)
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


def download_close_prices():
    """Download close prices for universe + SPY."""
    log.info("Downloading close prices for target computation ...")
    all_closes = {}
    symbols = list(set(UNIVERSE + ["SPY"]))
    batch_size = 10

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        ticker_str = " ".join(batch)
        try:
            data = yf.download(ticker_str, start="2010-01-01", end="2027-01-01",
                               progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data
            for sym in batch:
                if sym in close.columns:
                    series = close[sym].dropna()
                    if len(series) > 0:
                        series.index = pd.to_datetime(series.index).tz_localize(None)
                        all_closes[sym] = series
        except Exception as e:
            log.warning(f"  Failed batch {batch}: {e}")

    log.info(f"  Downloaded close prices for {len(all_closes)}/{len(symbols)} symbols")
    return all_closes


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"date", "symbol", "target", "target_slow", "target_v3",
               "fwd_ret_30d", "fwd_ret", "fwd_excess_ret_30d"}
    forward_keywords = {"fwd", "forward", "future"}
    cols = [
        c for c in df.columns
        if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)
    ]
    return cols


def compute_fwd_return(prices, date, fwd_days):
    """Compute forward return for a symbol on a given date."""
    if date not in prices.index:
        return np.nan
    cur_price = prices[date]
    future_prices = prices[prices.index > date]
    if len(future_prices) < fwd_days:
        return np.nan
    future_price = future_prices.iloc[fwd_days - 1]
    return (future_price / cur_price) - 1.0


def load_data():
    log.info(f"Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Filter to UNIVERSE only
    df = df[df["symbol"].isin(UNIVERSE)].copy()

    close_prices = download_close_prices()

    if "SPY" not in close_prices:
        sys.exit("ERROR: SPY close prices not available")

    spy_prices = close_prices["SPY"]

    # ── Compute relative target ──────────────────────────────────────────
    log.info(f"Computing relative target: (stock_30d_ret - SPY_30d_ret) > {EXCESS_THRESHOLD:.0%} ...")

    # Pre-compute SPY forward returns for all dates
    spy_fwd_cache = {}
    unique_dates = df["date"].unique()
    for date in unique_dates:
        spy_fwd_cache[date] = compute_fwd_return(spy_prices, date, FWD_DAYS)

    # Compute per-symbol forward returns and excess returns
    fwd_stock_list = []
    fwd_excess_list = []

    for sym in df["symbol"].unique():
        sym_mask = df["symbol"] == sym
        sym_dates = df.loc[sym_mask, "date"]

        if sym not in close_prices:
            fwd_stock_list.append(pd.Series(np.nan, index=sym_dates.index))
            fwd_excess_list.append(pd.Series(np.nan, index=sym_dates.index))
            continue

        prices = close_prices[sym]
        stock_rets = []
        excess_rets = []

        for date in sym_dates:
            stock_ret = compute_fwd_return(prices, date, FWD_DAYS)
            spy_ret = spy_fwd_cache.get(date, np.nan)
            stock_rets.append(stock_ret)
            if not np.isnan(stock_ret) and not np.isnan(spy_ret):
                excess_rets.append(stock_ret - spy_ret)
            else:
                excess_rets.append(np.nan)

        fwd_stock_list.append(pd.Series(stock_rets, index=sym_dates.index))
        fwd_excess_list.append(pd.Series(excess_rets, index=sym_dates.index))

    df["fwd_ret_30d"] = pd.concat(fwd_stock_list).sort_index()
    df["fwd_excess_ret_30d"] = pd.concat(fwd_excess_list).sort_index()
    df["target_v3"] = (df["fwd_excess_ret_30d"] > EXCESS_THRESHOLD).astype("Int8")

    # Drop rows with no target
    before = len(df)
    df = df[df["target_v3"].notna()].copy()
    df = df.dropna(subset=["target_v3"])
    feature_cols = get_feature_cols(df)
    df = df.dropna(subset=feature_cols)

    n_pos = (df["target_v3"] == 1).sum()
    n_neg = (df["target_v3"] == 0).sum()
    log.info(f"Loaded {before:,} rows -> {len(df):,} after target  |  "
             f"{df['symbol'].nunique()} symbols  |  "
             f"{df['date'].min().date()} -> {df['date'].max().date()}")
    log.info(f"Target balance: {n_pos:,} positive ({n_pos/(n_pos+n_neg)*100:.1f}%) / "
             f"{n_neg:,} negative ({n_neg/(n_pos+n_neg)*100:.1f}%)")

    # Compare with v2 absolute target
    df["target_v2"] = (df["fwd_ret_30d"] > 0.05).astype("Int8")
    n_v2_pos = (df["target_v2"] == 1).sum()
    log.info(f"  v2 absolute target (>5%): {n_v2_pos:,} positive ({n_v2_pos/len(df)*100:.1f}%)")
    log.info(f"  v3 relative target (>3% excess): {n_pos:,} positive ({n_pos/len(df)*100:.1f}%)")

    # Show target rate by year
    df["_year"] = df["date"].dt.year
    log.info(f"\n  Target rate by year:")
    log.info(f"  {'Year':>6} {'v2 (>5% abs)':>14} {'v3 (>3% excess)':>16}")
    log.info(f"  {'─'*6} {'─'*14} {'─'*16}")
    for yr in sorted(df["_year"].unique()):
        yr_mask = df["_year"] == yr
        v2_rate = df.loc[yr_mask, "target_v2"].mean() * 100
        v3_rate = df.loc[yr_mask, "target_v3"].mean() * 100
        print(f"  {yr:>6} {v2_rate:>13.1f}% {v3_rate:>15.1f}%")
    df.drop(columns=["_year", "target_v2"], inplace=True)

    return df


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
                 f"train {start_date.date()}->{train_end.date()}  "
                 f"test  {test_start.date()}->{test_end.date()}")
        yield train_mask, test_mask, label

        train_end = train_end + relativedelta(months=TEST_MONTHS)


def main():
    log.info("=" * 72)
    log.info("ML Slow v3 — Relative Outperformance Target")
    log.info(f"Target: (stock_30d_ret - SPY_30d_ret) > {EXCESS_THRESHOLD:.0%}")
    log.info("=" * 72)

    df = load_data()

    feature_cols = get_feature_cols(df)
    log.info(f"\nFeature columns: {len(feature_cols)}")

    X = df[feature_cols].values
    y = df["target_v3"].values.astype(int)
    dates_arr = df["date"]

    windows   = list(walk_forward_windows(dates_arr))
    n_windows = len(windows)

    if n_windows == 0:
        sys.exit("ERROR: no walk-forward windows.")

    log.info(f"Walk-forward: {TRAIN_YEARS}yr train | {PURGE_DAYS}d purge | "
             f"{TEST_MONTHS}mo test | {n_windows} windows")
    log.info("-" * 72)

    all_preds  = []
    window_auc = []
    last_model = None

    for train_mask, test_mask, label in windows:
        log.info(f"\n{label}")

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        n_total = len(X_train)
        n_pos = y_train.sum()
        log.info(f"  Train: {n_total:,} (pos={n_pos}, neg={n_total-n_pos})  |  "
                 f"Test: {len(X_test):,}")

        if len(X_test) == 0:
            log.info("  No test rows — skipping")
            continue

        # 80/20 split for calibration
        split_idx = int(n_total * CALIB_SPLIT)
        X_lgb,   y_lgb   = X_train[:split_idx], y_train[:split_idx]
        X_calib, y_calib  = X_train[split_idx:], y_train[split_idx:]

        scale = (len(y_lgb) - y_lgb.sum()) / max(y_lgb.sum(), 1)

        model_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
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
            FrozenEstimator(model_lgb), method="isotonic", cv=2,
        )
        calib_model.fit(X_calib, y_calib)

        probs = calib_model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        window_auc.append(auc)

        n_picks_55 = int((probs >= 0.55).sum())
        n_picks_50 = int((probs >= 0.50).sum())
        log.info(f"  AUC={auc:.4f}  |  range=[{probs.min():.4f}, {probs.max():.4f}]  |  "
                 f"picks @0.55={n_picks_55}  @0.50={n_picks_50}")

        test_df = df[test_mask].copy()
        test_df["prob"] = probs
        test_df["pred"] = preds
        all_preds.append(test_df)
        last_model = calib_model

    if not all_preds:
        sys.exit("ERROR: no predictions produced.")

    # ── Combined OOS results ─────────────────────────────────────────────
    combined = pd.concat(all_preds, ignore_index=True)
    yt = combined["target_v3"].values.astype(int)
    yp = combined["prob"].values
    yh = combined["pred"].values

    acc  = accuracy_score(yt, yh)
    prec = precision_score(yt, yh, zero_division=0)
    rec  = recall_score(yt, yh, zero_division=0)
    f1   = f1_score(yt, yh, zero_division=0)
    auc  = roc_auc_score(yt, yp)
    cm   = confusion_matrix(yt, yh)

    log.info(f"\n{'='*72}")
    log.info(f"COMBINED OOS RESULTS — ML SLOW v3 (relative outperformance)")
    log.info(f"{'='*72}")
    log.info(f"  Accuracy  : {acc:.4f}")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  F1 Score  : {f1:.4f}")
    log.info(f"  AUC-ROC   : {auc:.4f}")
    log.info(f"\n  Confusion Matrix:")
    log.info(f"              Pred 0   Pred 1")
    log.info(f"  Actual 0    {cm[0,0]:6d}   {cm[0,1]:6d}")
    log.info(f"  Actual 1    {cm[1,0]:6d}   {cm[1,1]:6d}")

    # Signal quality
    valid = combined.dropna(subset=["fwd_excess_ret_30d"])
    if len(valid) > 0:
        avg_all   = valid["fwd_excess_ret_30d"].mean()
        for thresh in [0.55, 0.50]:
            picks     = valid[valid["prob"] >= thresh]
            n_picks   = len(picks)
            avg_picks = picks["fwd_excess_ret_30d"].mean() if n_picks > 0 else np.nan
            lift      = avg_picks - avg_all if n_picks > 0 else np.nan
            hit_rate  = (picks["target_v3"] == 1).mean() if n_picks > 0 else np.nan
            log.info(f"\n  Signal Quality (threshold = {thresh:.2f}):")
            log.info(f"    Avg excess return — all stocks : {avg_all*100:+.2f}%")
            log.info(f"    Avg excess return — model picks: {avg_picks*100:+.2f}%  (n={n_picks:,})")
            log.info(f"    Lift vs universe                : {lift*100:+.2f}%")
            log.info(f"    Hit rate (actually >3% excess)  : {hit_rate*100:.1f}%")

    # ── Pick rate by year ────────────────────────────────────────────────
    log.info(f"\n{'─'*72}")
    log.info("PICK RATE BY YEAR")
    log.info(f"{'─'*72}")
    combined["year"] = combined["date"].dt.year
    all_years = sorted(combined["year"].unique())

    log.info(f"  {'Year':>6}  {'@0.55':>8}  {'@0.50':>8}  {'Total':>8}  {'%@0.55':>8}  {'%@0.50':>8}")
    log.info(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    sparse_55 = []
    sparse_50 = []
    for yr in all_years:
        yr_data = combined[combined["year"] == yr]
        n_total = len(yr_data)
        n_55 = len(yr_data[yr_data["prob"] >= 0.55])
        n_50 = len(yr_data[yr_data["prob"] >= 0.50])
        r_55 = n_55 / n_total * 100 if n_total > 0 else 0
        r_50 = n_50 / n_total * 100 if n_total > 0 else 0
        flag_55 = " !" if n_55 < 3 else ""
        flag_50 = " !" if n_50 < 3 else ""
        log.info(f"  {yr:>6}  {n_55:>8}{flag_55}  {n_50:>8}{flag_50}  {n_total:>8}  {r_55:>7.1f}%  {r_50:>7.1f}%")
        if n_55 < 3:
            sparse_55.append(yr)
        if n_50 < 3:
            sparse_50.append(yr)

    if sparse_55:
        log.info(f"\n  Sparse @0.55: {', '.join(str(y) for y in sparse_55)}")
    if sparse_50:
        log.info(f"  Sparse @0.50: {', '.join(str(y) for y in sparse_50)}")

    # ── Per-window AUC summary ───────────────────────────────────────────
    log.info(f"\n{'─'*72}")
    log.info("PER-WINDOW AUC")
    log.info(f"{'─'*72}")
    for i, (_, _, label) in enumerate(windows):
        wlabel = label.split('  ')[0]
        if i < len(window_auc):
            log.info(f"  {wlabel}  AUC={window_auc[i]:.4f}")
    log.info(f"  Overall OOS AUC: {auc:.4f}")

    # ── Feature importance ───────────────────────────────────────────────
    log.info(f"\n{'─'*72}")
    log.info("TOP 15 FEATURES BY GAIN")
    log.info(f"{'─'*72}")

    new_feats = [
        "ret_126d","ret_252d","dist_52w_high","dist_52w_low","sma200_slope",
        "max_dd_6m","consec_up_months","consec_down_months",
        "yield_curve_10y2y","yield_curve_30d_change","hy_spread","hy_spread_30d_change",
        "dxy_level","dxy_30d_change","hyg_lqd_ratio","hyg_lqd_30d_change",
        "copper_gold_ratio","copper_gold_30d_change",
        "return_rank_3m","return_rank_6m","return_rank_12m",
        "vol_rank_3m","vol_126d","vol_rank_6m",
    ]

    try:
        cc = last_model.calibrated_classifiers_[0]
        lgb_model = cc.estimator.estimator
        if hasattr(lgb_model, 'booster_'):
            importance = pd.Series(
                lgb_model.booster_.feature_importance(importance_type="gain"),
                index=feature_cols,
            ).sort_values(ascending=False)
            total = importance.sum()
            for rank, (feat, gain) in enumerate(importance.head(15).items(), 1):
                is_new = "NEW" if feat in new_feats else "   "
                pct = gain / total * 100
                bar = "█" * int(pct * 2)
                log.info(f"  {rank:2d}. [{is_new}] {feat:<24} {pct:5.1f}%  {bar}")

            n_new_top15 = sum(1 for f in importance.head(15).index if f in new_feats)
            log.info(f"\n  New features in top 15: {n_new_top15}/15")
    except Exception as e:
        log.info(f"  Could not extract: {e}")

    # ── Save outputs ─────────────────────────────────────────────────────
    log.info(f"\n{'─'*72}")
    log.info("Saving outputs ...")

    joblib.dump(last_model, str(MODEL_FILE))
    log.info(f"  Model       -> {MODEL_FILE}")

    save_cols = ["date", "symbol", "target_v3", "prob", "pred",
                 "fwd_ret_30d", "fwd_excess_ret_30d"] + feature_cols
    save_cols = [c for c in save_cols if c in combined.columns]
    combined[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow",
                                   compression="snappy")
    log.info(f"  Predictions -> {PRED_FILE}  ({len(combined):,} rows)")

    log.info("\n" + "=" * 72)
    log.info("ML Slow v3 training complete.")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
