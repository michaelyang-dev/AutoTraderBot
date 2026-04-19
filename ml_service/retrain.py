#!/usr/bin/env python3
"""
Monthly Model Retraining (Calibrated Production Model)
=======================================================
Automates the full retrain pipeline for the calibrated 3-strategy system:

  1. Run data_pipeline.py  → rebuild features.parquet with latest prices + VIXY fix
  2. Train calibrated model (CalibratedClassifierCV + isotonic, walk-forward CV)
     → save candidate model + predictions
  3. Run Path B backtest (ML Medium + Momentum + Mean Reversion)
     with the candidate predictions via unified_backtester
  4. Deploy gate: candidate Path B must have CAGR >= baseline AND Sharpe >= baseline
  5. If PASS: backup current model, promote candidate, update metrics, email
  6. If FAIL: discard candidate, keep current model, email

Cron example (1st of each month at 6 AM):
    0 6 1 * * cd /path/to/auto-trader/ml_service && /path/to/python3 retrain.py >> /path/to/retrain.log 2>&1

Run manually:
    python3 retrain.py
"""

import json
import os
import shutil
import smtplib
import subprocess
import sys
import time
import warnings
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from dateutil.relativedelta import relativedelta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
FEATURES_FILE = DATA_DIR / "features.parquet"
MODEL_FILE   = DATA_DIR / "model.lgb"
PRED_FILE    = DATA_DIR / "predictions.parquet"
METRICS_FILE = DATA_DIR / "model_metrics.json"

# Candidate files (promoted to production on deploy)
CANDIDATE_MODEL = DATA_DIR / "model_retrain_candidate.lgb"
CANDIDATE_PREDS = DATA_DIR / "predictions_retrain_candidate.parquet"

LOG_FILE = BASE_DIR / "retrain.log"

# ── Env ──────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(BASE_DIR.parent / ".env")

EMAIL_USER         = os.getenv("EMAIL_USER", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    ET = pytz.timezone("America/New_York")

# ── Training config (same as train_model.py) ─────────────────────────────────
TRAIN_YEARS  = 3
TEST_MONTHS  = 6
PURGE_DAYS   = 10
CALIB_SPLIT  = 0.80
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    line = f"[{ts}]  {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def format_pct(val, decimals=2):
    if val is None:
        return "N/A"
    return f"{val*100:+.{decimals}f}%"


def send_email(subject: str, body: str):
    if not EMAIL_USER or not EMAIL_APP_PASSWORD:
        log("Email not configured — skipping notification")
        return
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_USER
    msg["To"]      = EMAIL_USER
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(EMAIL_USER, EMAIL_APP_PASSWORD)
            server.send_message(msg)
        log(f"Email sent: {subject}")
    except Exception as exc:
        log(f"Email failed: {exc}")


def run_step(script: str, label: str) -> subprocess.CompletedProcess:
    """Run a Python script as a subprocess, streaming output to stdout."""
    log(f"START — {label}")
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=str(BASE_DIR),
        capture_output=True, text=True, timeout=1800,
    )
    elapsed = time.perf_counter() - t0
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="")
        log(f"FAILED — {label} (exit code {result.returncode}, {elapsed:.0f}s)")
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    log(f"DONE  — {label} ({elapsed:.0f}s)")
    return result


def load_previous_metrics():
    if not METRICS_FILE.exists():
        return None
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_metrics(metrics: dict):
    metrics["timestamp"] = datetime.now(ET).isoformat()
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Metrics saved to {METRICS_FILE}")


# ── Walk-forward window generator ────────────────────────────────────────────

def walk_forward_windows(dates: pd.Series):
    all_dates  = np.sort(dates.unique())
    start_date = pd.Timestamp(all_dates[0])
    end_date   = pd.Timestamp(all_dates[-1])
    train_end  = start_date + relativedelta(years=TRAIN_YEARS)
    window = 0
    while True:
        purge_idx = np.searchsorted(all_dates, np.datetime64(train_end, "ns"))
        purge_idx = min(purge_idx + PURGE_DAYS, len(all_dates) - 1)
        test_start = pd.Timestamp(all_dates[purge_idx])
        test_end   = test_start + relativedelta(months=TEST_MONTHS)
        if test_start >= end_date:
            break
        test_end = min(test_end, end_date)
        window += 1
        train_mask = (dates >= start_date) & (dates < train_end)
        test_mask  = (dates >= test_start) & (dates <= test_end)
        label = (f"W{window:02d}  train {start_date.date()}→{train_end.date()}  "
                 f"test {test_start.date()}→{test_end.date()}")
        yield train_mask, test_mask, label
        train_end = train_end + relativedelta(months=TEST_MONTHS)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


# ── Train calibrated model ───────────────────────────────────────────────────

def train_calibrated_model():
    """
    Train a calibrated LightGBM model using walk-forward CV.
    Returns (candidate_model, candidate_predictions_df, feature_cols, oos_auc).
    """
    log("Loading features.parquet ...")
    if not FEATURES_FILE.exists():
        raise RuntimeError(f"{FEATURES_FILE} not found — data pipeline may have failed")

    df = pd.read_parquet(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    before = len(df)
    df = df.dropna()
    log(f"Loaded {before:,} → {len(df):,} rows after dropping NaN  |  "
        f"{df['symbol'].nunique()} symbols  |  "
        f"{df['date'].min().date()} → {df['date'].max().date()}")

    feature_cols = get_feature_cols(df)
    log(f"Features: {len(feature_cols)} columns")

    # Reconstruct forward returns for backtest compatibility
    df = df.sort_values(["symbol", "date"])
    df["fwd_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)

    X = df[feature_cols].values
    y = df["target"].values
    dates_arr = df["date"]

    windows = list(walk_forward_windows(dates_arr))
    if not windows:
        raise RuntimeError("No walk-forward windows — insufficient data")

    log(f"Walk-forward: {len(windows)} windows  |  "
        f"{TRAIN_YEARS}yr train, {PURGE_DAYS}d purge, {TEST_MONTHS}mo test")

    all_preds_calib = []
    last_model_calib = None

    for train_mask, test_mask, label in windows:
        log(f"  {label}")

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        n_total = len(X_train)
        if len(X_test) == 0:
            log("    No test rows — skipping")
            continue

        # Split: 80% LGB training, 20% calibration
        split_idx = int(n_total * CALIB_SPLIT)
        X_lgb,   y_lgb   = X_train[:split_idx], y_train[:split_idx]
        X_calib, y_calib  = X_train[split_idx:], y_train[split_idx:]

        scale_lgb = (len(y_lgb) - y_lgb.sum()) / max(y_lgb.sum(), 1)

        # Train LightGBM
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
        calib_model = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
        calib_model.fit(X_calib, y_calib)

        probs = calib_model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        n_picks = int((probs >= PROB_THRESH).sum())

        log(f"    acc={acc:.4f}  AUC={auc:.4f}  picks(>0.55)={n_picks}/{len(X_test)}")

        test_df = df[test_mask].copy()
        test_df["prob"] = probs
        test_df["pred"] = preds
        all_preds_calib.append(test_df)
        last_model_calib = calib_model

    if not all_preds_calib:
        raise RuntimeError("No predictions produced — training failed")

    combined = pd.concat(all_preds_calib, ignore_index=True)
    yt = combined["target"].values
    yp = combined["prob"].values
    oos_auc = roc_auc_score(yt, yp)

    log(f"Combined OOS AUC: {oos_auc:.4f}  |  {len(combined):,} predictions")

    return last_model_calib, combined, feature_cols, oos_auc


# ── Run Path B backtest ──────────────────────────────────────────────────────

def run_pathb_backtest(predictions_df):
    """
    Run the 3-strategy Path B backtest (ML Medium + Momentum + Mean Reversion)
    using the given predictions DataFrame.
    Returns metrics dict with cagr, sharpe, sortino, max_dd, n_trades, win_rate, alpha.
    """
    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS,
        MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from backtest_ml import calc_metrics, calc_alpha_beta
    from diagnose_combined import instrumented_run

    # Date range from predictions
    all_dates = sorted(predictions_df["date"].unique().tolist())
    universe_syms = sorted(predictions_df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    log(f"Backtest: {len(all_dates)} days  |  {years:.1f} years  |  {len(universe_syms)} symbols")

    # Fetch OHLCV data
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

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
    ml_strat = MLMediumStrategy(predictions_df, threshold=PROB_THRESH)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # Run backtest
    vals, trades, diag = instrumented_run(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B")

    m = calc_metrics(vals, trades, years, "Path B")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    m["alpha"] = alpha
    m["avg_pos"] = float(np.mean(diag["daily_pos_count"]))

    return m


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    log("=" * 65)
    log("Monthly Model Retraining (Calibrated Production Model)")
    log("=" * 65)

    # ── 1. Run data pipeline (fetch latest prices, rebuild features) ─────
    try:
        run_step("data_pipeline.py", "Data Pipeline (fetch prices & rebuild features)")
    except RuntimeError as exc:
        log(f"Data pipeline failed: {exc}")
        send_email(
            "AutoTrader: Model Retrain FAILED — Data Pipeline Error",
            f"Model retraining failed at data pipeline stage.\n\n"
            f"Error: {exc}\n\n"
            f"The current production model is unchanged.\n"
            f"Timestamp: {datetime.now(ET).isoformat()}",
        )
        return 1

    # ── 2. Train calibrated model ────────────────────────────────────────
    log("")
    log("─" * 65)
    log("Training calibrated model (CalibratedClassifierCV + isotonic)")
    log("─" * 65)

    try:
        candidate_model, candidate_preds, feature_cols, oos_auc = train_calibrated_model()
    except RuntimeError as exc:
        log(f"Training failed: {exc}")
        send_email(
            "AutoTrader: Model Retrain FAILED — Training Error",
            f"Model retraining failed at training stage.\n\n"
            f"Error: {exc}\n\n"
            f"The current production model is unchanged.\n"
            f"Timestamp: {datetime.now(ET).isoformat()}",
        )
        return 1

    # ── 3. Save candidate files ──────────────────────────────────────────
    log("")
    log("Saving candidate model and predictions ...")
    joblib.dump(candidate_model, str(CANDIDATE_MODEL))
    log(f"  Candidate model → {CANDIDATE_MODEL.name}")

    save_cols = ["date", "symbol", "target", "prob", "pred", "fwd_ret"] + feature_cols
    save_cols = [c for c in save_cols if c in candidate_preds.columns]
    candidate_preds[save_cols].to_parquet(
        CANDIDATE_PREDS, index=False, engine="pyarrow", compression="snappy")
    log(f"  Candidate preds → {CANDIDATE_PREDS.name}  ({len(candidate_preds):,} rows)")

    # ── 4. Run Path B backtest with candidate predictions ────────────────
    log("")
    log("─" * 65)
    log("Running Path B backtest with candidate model ...")
    log("─" * 65)

    try:
        new_metrics = run_pathb_backtest(candidate_preds)
    except Exception as exc:
        log(f"Backtest failed: {exc}")
        # Clean up candidate files
        CANDIDATE_MODEL.unlink(missing_ok=True)
        CANDIDATE_PREDS.unlink(missing_ok=True)
        send_email(
            "AutoTrader: Model Retrain FAILED — Backtest Error",
            f"Model retraining failed at backtest stage.\n\n"
            f"Error: {exc}\n\n"
            f"Candidate files have been cleaned up.\n"
            f"The current production model is unchanged.\n"
            f"Timestamp: {datetime.now(ET).isoformat()}",
        )
        return 1

    new_metrics["oos_auc"] = oos_auc

    new_cagr   = new_metrics.get("cagr", 0)
    new_sharpe = new_metrics.get("sharpe", 0)
    new_dd     = new_metrics.get("max_dd", 0)
    new_trades = new_metrics.get("n_trades", 0)
    new_wr     = new_metrics.get("win_rate", 0)
    new_alpha  = new_metrics.get("alpha", 0)

    log(f"Candidate results:  CAGR={format_pct(new_cagr)}  Sharpe={new_sharpe:.3f}  "
        f"MaxDD={format_pct(new_dd)}  Trades={new_trades}  WR={format_pct(new_wr)}  "
        f"Alpha={format_pct(new_alpha)}  AUC={oos_auc:.4f}")

    # ── 5. Compare to baseline ───────────────────────────────────────────
    log("")
    log("─" * 65)
    log("Deploy gate check")
    log("─" * 65)

    prev_metrics = load_previous_metrics()

    if prev_metrics is None:
        log("No previous metrics found — deploying as first production model")
        status = "DEPLOYED"
        reason = "First production model (no baseline to compare)"
        prev_cagr = None
        prev_sharpe = None
    else:
        prev_cagr   = prev_metrics.get("cagr", 0)
        prev_sharpe = prev_metrics.get("sharpe", 0)

        cagr_ok   = new_cagr >= prev_cagr
        sharpe_ok = new_sharpe >= prev_sharpe

        log(f"Baseline:   CAGR={format_pct(prev_cagr)}  Sharpe={prev_sharpe:.3f}")
        log(f"Candidate:  CAGR={format_pct(new_cagr)}  Sharpe={new_sharpe:.3f}")
        log(f"CAGR gate:   {'PASS' if cagr_ok else 'FAIL'}  "
            f"({format_pct(new_cagr)} vs {format_pct(prev_cagr)})")
        log(f"Sharpe gate: {'PASS' if sharpe_ok else 'FAIL'}  "
            f"({new_sharpe:.3f} vs {prev_sharpe:.3f})")

        if cagr_ok and sharpe_ok:
            status = "DEPLOYED"
            reason = "Candidate meets or exceeds both CAGR and Sharpe baselines"
        else:
            status = "REJECTED"
            failures = []
            if not cagr_ok:
                failures.append(f"CAGR {format_pct(new_cagr)} < baseline {format_pct(prev_cagr)}")
            if not sharpe_ok:
                failures.append(f"Sharpe {new_sharpe:.3f} < baseline {prev_sharpe:.3f}")
            reason = "; ".join(failures)

    # ── 6. Deploy or reject ──────────────────────────────────────────────
    log("")
    if status == "DEPLOYED":
        log(f"DEPLOYING candidate model — {reason}")

        # Backup current production model with date stamp
        if MODEL_FILE.exists():
            date_str = datetime.now(ET).strftime("%Y%m%d")
            backup_path = DATA_DIR / f"model_backup_{date_str}.lgb"
            shutil.copy2(MODEL_FILE, backup_path)
            log(f"  Backed up current model → {backup_path.name}")

        # Promote candidate to production
        shutil.copy2(CANDIDATE_MODEL, MODEL_FILE)
        log(f"  Promoted {CANDIDATE_MODEL.name} → {MODEL_FILE.name}")

        shutil.copy2(CANDIDATE_PREDS, PRED_FILE)
        log(f"  Promoted {CANDIDATE_PREDS.name} → {PRED_FILE.name}")

        # Update metrics with Path B baseline values
        save_metrics(new_metrics)

        # Clean up candidate files
        CANDIDATE_MODEL.unlink(missing_ok=True)
        CANDIDATE_PREDS.unlink(missing_ok=True)

        send_email(
            f"AutoTrader: Model Retrain DEPLOYED — "
            f"New CAGR: {format_pct(new_cagr)}, Sharpe: {new_sharpe:.3f}",
            f"Model Retrain DEPLOYED\n"
            f"{'=' * 55}\n\n"
            f"Timestamp: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}\n\n"
            f"{'─' * 55}\n"
            f"{'Metric':<20} {'Previous':<15} {'New':<15}\n"
            f"{'─' * 55}\n"
            f"{'CAGR':<20} {format_pct(prev_cagr) if prev_cagr is not None else 'N/A':<15} {format_pct(new_cagr):<15}\n"
            f"{'Sharpe':<20} {f'{prev_sharpe:.3f}' if prev_sharpe is not None else 'N/A':<15} {new_sharpe:.3f}\n"
            f"{'Max Drawdown':<20} {format_pct(prev_metrics.get('max_dd')) if prev_metrics else 'N/A':<15} {format_pct(new_dd):<15}\n"
            f"{'Win Rate':<20} {format_pct(prev_metrics.get('win_rate')) if prev_metrics else 'N/A':<15} {format_pct(new_wr):<15}\n"
            f"{'Alpha':<20} {format_pct(prev_metrics.get('alpha')) if prev_metrics else 'N/A':<15} {format_pct(new_alpha):<15}\n"
            f"{'OOS AUC':<20} {prev_metrics.get('oos_auc', 'N/A') if prev_metrics else 'N/A':<15} {oos_auc:.4f}\n"
            f"{'Trades':<20} {prev_metrics.get('n_trades', 'N/A') if prev_metrics else 'N/A':<15} {new_trades}\n"
            f"{'─' * 55}\n\n"
            f"Backtest: Path B (ML Medium + Momentum + Mean Reversion)\n"
            f"Model type: CalibratedClassifierCV (isotonic)\n"
            f"Production model updated successfully.\n",
        )

    else:
        log(f"REJECTING candidate model — {reason}")

        # Clean up candidate files
        CANDIDATE_MODEL.unlink(missing_ok=True)
        CANDIDATE_PREDS.unlink(missing_ok=True)
        log("  Deleted candidate files")

        send_email(
            f"AutoTrader: Model Retrain REJECTED — "
            f"CAGR: {format_pct(new_cagr)}, Sharpe: {new_sharpe:.3f}",
            f"Model Retrain REJECTED\n"
            f"{'=' * 55}\n\n"
            f"Timestamp: {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}\n"
            f"Reason: {reason}\n\n"
            f"{'─' * 55}\n"
            f"{'Metric':<20} {'Baseline':<15} {'Candidate':<15}\n"
            f"{'─' * 55}\n"
            f"{'CAGR':<20} {format_pct(prev_cagr):<15} {format_pct(new_cagr):<15}\n"
            f"{'Sharpe':<20} {f'{prev_sharpe:.3f}':<15} {new_sharpe:.3f}\n"
            f"{'Max Drawdown':<20} {format_pct(prev_metrics.get('max_dd')):<15} {format_pct(new_dd):<15}\n"
            f"{'Win Rate':<20} {format_pct(prev_metrics.get('win_rate')):<15} {format_pct(new_wr):<15}\n"
            f"{'Alpha':<20} {format_pct(prev_metrics.get('alpha')):<15} {format_pct(new_alpha):<15}\n"
            f"{'─' * 55}\n\n"
            f"Keeping current production model unchanged.\n"
            f"Candidate files have been deleted.\n",
        )

    # ── 7. Summary ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    log("")
    log("=" * 65)
    log(f"Retrain complete — Status: {status}")
    log(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
