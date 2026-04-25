#!/usr/bin/env python3
"""
Monthly Retrain Pipeline — V5d LGBM+XGB Ensemble
=================================================
Automated monthly retraining with deploy gate validation,
model backup, atomic swap, and Telegram notifications.

Steps:
  1. Update market data (yfinance via data_pipeline, FMP, FRED)
  2. Regenerate features.parquet
  3. Train LGBM + XGB (same hyperparameters as train_production_model.py)
  4. Generate 50/50 ensemble predictions
  5. Run deploy gate (90-day backtest validation)
  6. Backup old models → timestamped directory
  7. Atomic swap: write to tmp, verify, rename to production
  8. Reload signal server via pm2, verify /health
  9. Notify via Telegram

Usage:
    python3 retrain.py                 # full retrain + deploy
    python3 retrain.py --dry-run       # everything except atomic swap
    python3 retrain.py --test          # small data subset for quick verification

Cron (1st of month, 2 AM ET):
    0 2 1 * * cd /home/ubuntu/AutoTraderBot && .venv/bin/python ml_service/retrain.py \
        >> logs/retrain_cron.log 2>&1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import requests

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
TMP_DIR     = DATA_DIR / "retrain_tmp"
LOG_DIR     = BASE_DIR.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────

_log_file = None


def _init_log():
    global _log_file
    log_path = LOG_DIR / f"retrain_{datetime.now().strftime('%Y%m%d')}.log"
    _log_file = open(log_path, "a")
    log(f"\n{'='*70}")
    log(f"  RETRAIN PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    log(f"{'='*70}")
    return log_path


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def _close_log():
    if _log_file:
        _log_file.close()


# ── Telegram notifications ──────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv(BASE_DIR.parent / ".env")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED   = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def notify_telegram(message: str):
    """Send a Telegram notification. Silent fail on error."""
    if not TELEGRAM_ENABLED:
        log(f"  Telegram disabled — would send: {message[:100]}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }

    for attempt in range(1, 3):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.ok:
                log("  Telegram notification sent")
                return
            log(f"  Telegram API error (attempt {attempt}/2): {resp.status_code} — {resp.text[:200]}")
            # If HTML parse failed, retry without parse_mode
            if resp.status_code == 400 and "can't parse" in resp.text:
                payload.pop("parse_mode", None)
                retry = requests.post(url, json=payload, timeout=10)
                if retry.ok:
                    log("  Telegram notification sent (plain text fallback)")
                    return
        except Exception as exc:
            log(f"  Telegram send failed (attempt {attempt}/2): {exc}")

        if attempt == 1:
            time.sleep(5)


# ── Step 1: Update data ─────────────────────────────────────────────────────

def step1_update_data(test_mode: bool = False):
    """Run FMP fundamentals pipeline, FRED macro pipeline, then data pipeline."""
    log(f"\n{'='*70}")
    log("STEP 1: UPDATE MARKET DATA")
    log(f"{'='*70}")

    # 1a. FMP fundamentals
    fmp_script = BASE_DIR / "fmp_fundamentals_pipeline.py"
    if fmp_script.exists():
        log("  Running FMP fundamentals pipeline ...")
        result = subprocess.run(
            [sys.executable, str(fmp_script)],
            cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            log(f"  WARNING: FMP pipeline failed: {result.stderr[-500:]}")
        else:
            log("  FMP fundamentals updated")
    else:
        log("  WARNING: fmp_fundamentals_pipeline.py not found — skipping")

    # 1b. FRED macro data
    fred_script = BASE_DIR / "fred_data_pipeline.py"
    if fred_script.exists():
        log("  Running FRED macro pipeline ...")
        result = subprocess.run(
            [sys.executable, str(fred_script)],
            cwd=str(BASE_DIR),
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log(f"  WARNING: FRED pipeline failed: {result.stderr[-500:]}")
        else:
            log("  FRED macro data updated")

    # 1c. Main data pipeline (yfinance + features)
    log("  Running main data pipeline (yfinance + feature computation) ...")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "data_pipeline.py")],
        cwd=str(BASE_DIR),
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Data pipeline failed:\n{result.stderr[-1000:]}")

    features_file = DATA_DIR / "features.parquet"
    if not features_file.exists():
        raise RuntimeError("features.parquet not generated")

    df = pd.read_parquet(features_file)
    log(f"  features.parquet: {len(df):,} rows, {len(df.columns)} columns")
    log(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    return True


# ── Step 2: Train models ────────────────────────────────────────────────────

def step2_train_models(test_mode: bool = False, rolling_years: int = 12):
    """
    Train LGBM + XGB ensemble with same hyperparameters as train_production_model.py.
    Uses rolling 12-year training window.
    Writes models to TMP_DIR for atomic swap.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    import joblib
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score

    from train_production_model import (
        LGB_PARAMS, XGB_PARAMS, RANK_FEATURES, FUNDAMENTAL_FEATURE_COLS,
        CALIB_FRAC, N_TREES, EARLY_STOP_ROUNDS,
        get_feature_cols,
    )

    log(f"\n{'='*70}")
    log("STEP 2: TRAIN LGBM + XGB ENSEMBLE")
    log(f"{'='*70}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load features
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Drop days_until_earnings if present (retroactive date leakage)
    if "days_until_earnings" in df.columns:
        df = df.drop(columns=["days_until_earnings"])
        log("  Dropped days_until_earnings column")

    # Add cross-sectional rank features
    for base_col, rank_col in RANK_FEATURES:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    # Cross-sectional target
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_v5"] = np.nan
    valid_df = df[sp500_mask & has_fwd].copy()
    valid_df["pct_rank"] = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_v5"] = (valid_df["pct_rank"] >= 0.80).astype(int)
    df.loc[valid_df.index, "target_v5"] = valid_df["target_v5"]

    feature_cols = get_feature_cols(df)
    log(f"  Feature columns: {len(feature_cols)}")
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund_cols + ["target_v5"])

    # Survivorship filter
    train_df = df[df["in_sp500"] == True].copy()
    log(f"  Total rows: {len(df):,}  |  SP500 filtered: {len(train_df):,}")
    log(f"  Features: {len(feature_cols)}")

    # Rolling window: only use last N years for training
    max_date = train_df["date"].max()
    min_train_date = max_date - pd.Timedelta(days=rolling_years * 365)
    train_df = train_df[train_df["date"] >= min_train_date]
    log(f"  Rolling {rolling_years}-year window: {train_df['date'].min().date()} → {max_date.date()}")
    log(f"  Rows after windowing: {len(train_df):,}")

    if test_mode:
        # Subsample for quick test
        sample_dates = sorted(train_df["date"].unique())[-60:]
        train_df = train_df[train_df["date"].isin(sample_dates)]
        df = df[df["date"].isin(sample_dates)]
        log(f"  TEST MODE: subsampled to {len(train_df):,} rows (last 60 dates)")

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

    log(f"  Train: {train_mask.sum():,} rows | Calib: {calib_mask.sum():,} rows")

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    # Imputer
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_calib_imp = imp.transform(X_calib)
    X_all_imp = imp.transform(X_all)

    # Train LGBM
    log(f"  Training LGBMClassifier ({N_TREES} trees) ...")
    t0 = time.perf_counter()
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
    log(f"  LGBM: {n_trees} trees in {time.perf_counter() - t0:.0f}s")

    calib_lgbm = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
    calib_lgbm.fit(X_calib, y_calib)
    lgbm_auc = roc_auc_score(y_calib, calib_lgbm.predict_proba(X_calib)[:, 1])
    log(f"  LGBM AUC: {lgbm_auc:.4f}")

    # Train XGBoost
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    log(f"  Training XGBClassifier ({XGB_PARAMS['n_estimators']} trees) ...")
    t0 = time.perf_counter()
    model_xgb = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale)
    model_xgb.fit(X_train_imp, y_train)
    log(f"  XGB trained in {time.perf_counter() - t0:.0f}s")

    calib_xgb = CalibratedClassifierCV(model_xgb, method="isotonic", cv="prefit")
    calib_xgb.fit(X_calib_imp, y_calib)
    xgb_auc = roc_auc_score(y_calib, calib_xgb.predict_proba(X_calib_imp)[:, 1])
    log(f"  XGB AUC: {xgb_auc:.4f}")

    # Ensemble predictions
    lgbm_all = calib_lgbm.predict_proba(X_all)[:, 1]
    xgb_all = calib_xgb.predict_proba(X_all_imp)[:, 1]
    ensemble_probs = 0.5 * lgbm_all + 0.5 * xgb_all

    df["prob_lgbm"] = lgbm_all
    df["prob_xgb"] = xgb_all
    df["prob_ensemble"] = ensemble_probs

    ens_auc = roc_auc_score(
        y_calib,
        0.5 * calib_lgbm.predict_proba(X_calib)[:, 1] +
        0.5 * calib_xgb.predict_proba(X_calib_imp)[:, 1],
    )
    log(f"  Ensemble AUC: {ens_auc:.4f}")
    log(f"  Prob range: [{ensemble_probs.min():.4f}, {ensemble_probs.max():.4f}]")

    # Save to tmp
    joblib.dump(calib_lgbm, str(TMP_DIR / "model.lgb"))
    joblib.dump(calib_xgb, str(TMP_DIR / "model_xgb.pkl"))
    joblib.dump(imp, str(TMP_DIR / "imputer.pkl"))

    save_cols = ["date", "symbol", "target_v5", "prob_lgbm", "prob_xgb",
                 "prob_ensemble", "fwd_ret", "in_sp500"]
    df[save_cols].to_parquet(TMP_DIR / "predictions.parquet", index=False,
                             engine="pyarrow", compression="snappy")

    log(f"  Models saved to {TMP_DIR}")
    log(f"  LGBM: {(TMP_DIR / 'model.lgb').stat().st_size / 1024:.0f}KB")
    log(f"  XGB: {(TMP_DIR / 'model_xgb.pkl').stat().st_size / 1024:.0f}KB")

    return {
        "lgbm_auc": round(lgbm_auc, 4),
        "xgb_auc": round(xgb_auc, 4),
        "ensemble_auc": round(ens_auc, 4),
        "n_trees_lgbm": n_trees,
        "n_features": len(feature_cols),
        "n_rows": len(df),
    }


# ── Step 3: Deploy gate ─────────────────────────────────────────────────────

def step3_deploy_gate(train_metrics: dict):
    """Run deploy gate validation on new predictions."""
    from deploy_gate import run_deploy_gate

    log(f"\n{'='*70}")
    log("STEP 3: DEPLOY GATE VALIDATION")
    log(f"{'='*70}")

    passed, metrics, reasons = run_deploy_gate(
        TMP_DIR / "predictions.parquet",
        ensemble_auc=train_metrics.get("ensemble_auc"),
    )

    log(f"  Sharpe: {metrics.get('sharpe', 'N/A')}")
    log(f"  CAGR:   {metrics.get('cagr', 'N/A')}")
    log(f"  Max DD: {metrics.get('max_dd', 'N/A')}")
    log(f"  Trades: {metrics.get('n_trades', 'N/A')}")

    if passed:
        log("  DEPLOY GATE: PASSED")
    else:
        log("  DEPLOY GATE: FAILED")
        for r in reasons:
            log(f"    - {r}")

    return passed, metrics, reasons


# ── Step 4: Backup + atomic swap ─────────────────────────────────────────────

def step4_backup_and_swap(dry_run: bool = False):
    """Backup old models, then atomic rename new models to production."""
    from backup_models import backup_current, cleanup_old_backups

    log(f"\n{'='*70}")
    log("STEP 4: BACKUP & ATOMIC SWAP")
    log(f"{'='*70}")

    # Backup current
    backup_path = backup_current(log_fn=log)

    # Clean up old backups
    cleanup_old_backups(log_fn=log)

    if dry_run:
        log("  DRY RUN: Skipping atomic swap — models NOT deployed")
        return backup_path

    # Verify new files exist and are non-empty
    model_files = ["model.lgb", "model_xgb.pkl", "imputer.pkl", "predictions.parquet"]
    for fname in model_files:
        tmp_file = TMP_DIR / fname
        if not tmp_file.exists() or tmp_file.stat().st_size == 0:
            raise RuntimeError(f"New {fname} missing or empty in {TMP_DIR}")

    # Atomic swap: rename from tmp to production
    for fname in model_files:
        src = TMP_DIR / fname
        dst = DATA_DIR / fname
        # On same filesystem, os.replace is atomic
        os.replace(str(src), str(dst))
        log(f"  Swapped: {fname}")

    log("  Atomic swap complete")
    return backup_path


# ── Step 5: Reload & verify ─────────────────────────────────────────────────

def step5_reload_and_verify(dry_run: bool = False):
    """Restart signal server via pm2 and verify /health endpoint."""
    log(f"\n{'='*70}")
    log("STEP 5: RELOAD & VERIFY")
    log(f"{'='*70}")

    if dry_run:
        log("  DRY RUN: Skipping pm2 restart and verification")
        return True

    # Restart signal server
    log("  Restarting ml-server via pm2 ...")
    result = subprocess.run(
        ["pm2", "restart", "ml-server"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log(f"  WARNING: pm2 restart returned code {result.returncode}: {result.stderr}")

    # Wait and verify health endpoint
    log("  Waiting for signal server to come up ...")
    for attempt in range(24):  # up to 2 minutes (24 * 5s)
        time.sleep(5)
        try:
            req = urllib.request.Request("http://localhost:5001/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            if data.get("status") == "ok" and data.get("cached_signals", 0) > 0:
                log(f"  Health check passed (attempt {attempt + 1}):")
                log(f"    version: {data.get('model_version')}")
                log(f"    features: {data.get('feature_count')}")
                log(f"    signals: {data.get('cached_signals')}")
                return True
            else:
                log(f"  Attempt {attempt + 1}: status={data.get('status')}, "
                    f"signals={data.get('cached_signals', 0)}")
        except Exception as exc:
            log(f"  Attempt {attempt + 1}: connection failed ({exc})")

    log("  ERROR: Health check failed after 2 minutes")
    return False


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monthly retrain pipeline for V5d LGBM+XGB ensemble")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all steps except atomic swap and pm2 restart")
    parser.add_argument("--test", action="store_true",
                        help="Use small data subset for quick verification")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data update (use existing features.parquet)")
    parser.add_argument("--rolling-years", type=int, default=12,
                        help="Training window size in years (default: 12)")
    args = parser.parse_args()

    log_path = _init_log()
    log(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}"
        f"{'  (TEST)' if args.test else ''}")
    log(f"  Log: {log_path}")

    step_num = 0
    train_metrics = {}
    gate_metrics = {}
    backup_path = None

    try:
        # Step 1: Update data
        step_num = 1
        if not args.skip_data:
            step1_update_data(test_mode=args.test)
        else:
            log("\n  Skipping data update (--skip-data)")

        # Step 2: Train models
        step_num = 2
        train_metrics = step2_train_models(
            test_mode=args.test,
            rolling_years=args.rolling_years,
        )

        # Step 3: Deploy gate
        step_num = 3
        passed, gate_metrics, reasons = step3_deploy_gate(train_metrics)

        if not passed:
            log("\n  DEPLOY GATE FAILED — keeping old models")
            notify_telegram(
                f"<b>AutoTrader Retrain: DEPLOY GATE FAILED</b>\n\n"
                f"Old model retained.\n"
                f"Sharpe: {gate_metrics.get('sharpe')}\n"
                f"CAGR: {gate_metrics.get('cagr')}\n"
                f"Max DD: {gate_metrics.get('max_dd')}\n"
                f"Reasons: {'; '.join(reasons)}"
            )
            # Clean up tmp
            if TMP_DIR.exists():
                shutil.rmtree(str(TMP_DIR))
            _close_log()
            sys.exit(1)

        # Step 4: Backup and swap
        step_num = 4
        backup_path = step4_backup_and_swap(dry_run=args.dry_run)

        # Step 5: Reload and verify
        step_num = 5
        verified = step5_reload_and_verify(dry_run=args.dry_run)

        if not verified and not args.dry_run:
            log("\n  VERIFICATION FAILED — rolling back to backup")
            from backup_models import restore_from
            if backup_path:
                restore_from(backup_path, log_fn=log)
                # Restart again with old models
                subprocess.run(["pm2", "restart", "ml-server"],
                               capture_output=True, timeout=30)
            notify_telegram(
                f"<b>AutoTrader Retrain: VERIFICATION FAILED</b>\n\n"
                f"Rolled back to backup: {backup_path}\n"
                f"Error: Health check failed within 2 minutes"
            )
            _close_log()
            sys.exit(1)

        # Update baseline for next month
        if not args.dry_run:
            from deploy_gate import save_baseline
            save_baseline(gate_metrics)
            log("  Updated deploy baseline")

        # Success
        log(f"\n{'='*70}")
        log("  RETRAIN COMPLETE — SUCCESS")
        log(f"{'='*70}")
        log(f"  Ensemble AUC: {train_metrics.get('ensemble_auc', 'N/A')}")
        log(f"  Sharpe: {gate_metrics.get('sharpe', 'N/A')}")
        log(f"  CAGR: {gate_metrics.get('cagr', 'N/A')}")
        log(f"  Max DD: {gate_metrics.get('max_dd', 'N/A')}")

        if not args.dry_run:
            notify_telegram(
                f"<b>AutoTrader Retrain: SUCCESS</b>\n\n"
                f"New model active.\n"
                f"Ensemble AUC: {train_metrics.get('ensemble_auc')}\n"
                f"Sharpe: {gate_metrics.get('sharpe')}\n"
                f"CAGR: {gate_metrics.get('cagr')}\n"
                f"Max DD: {gate_metrics.get('max_dd')}"
            )

        # Clean up tmp
        if TMP_DIR.exists():
            shutil.rmtree(str(TMP_DIR))

    except Exception as exc:
        log(f"\n  FATAL ERROR at step {step_num}: {exc}")
        log(traceback.format_exc())

        # Rollback if we got past the swap step
        if step_num > 4 and backup_path and not args.dry_run:
            log("  Attempting rollback ...")
            from backup_models import restore_from
            restore_from(backup_path, log_fn=log)
            subprocess.run(["pm2", "restart", "ml-server"],
                           capture_output=True, timeout=30)

        notify_telegram(
            f"<b>AutoTrader Retrain: FAILED at step {step_num}</b>\n\n"
            f"Old model still active.\n"
            f"Error: {exc}"
        )

        # Clean up tmp
        if TMP_DIR.exists():
            shutil.rmtree(str(TMP_DIR))

        _close_log()
        sys.exit(1)

    _close_log()


if __name__ == "__main__":
    main()
