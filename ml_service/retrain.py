#!/usr/bin/env python3
"""
Monthly Model Retraining
========================
Automates the full retrain pipeline:

  1. Run data_pipeline.py  → rebuild features.parquet with latest prices
  2. Run train_model.py    → retrain LightGBM on updated data
  3. Run backtest_ml.py    → capture CAGR, Sharpe, win rate, alpha
  4. Compare to previous model metrics in data/model_metrics.json
  5. Deploy only if CAGR AND Sharpe are both >= previous model
  6. Save metrics to model_metrics.json on successful deployment
  7. Email notification with results
  8. Cron-compatible entry point (schedule for 1st of each month)

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
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
MODEL_FILE   = DATA_DIR / "model.lgb"
METRICS_FILE = DATA_DIR / "model_metrics.json"
PRED_FILE    = DATA_DIR / "predictions.parquet"

# ── Env ──────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(BASE_DIR.parent / ".env")

EMAIL_USER         = os.getenv("EMAIL_USER", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")

ET = ZoneInfo("America/New_York")


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    print(f"[{ts}]  {msg}", flush=True)


def run_step(script: str, label: str) -> subprocess.CompletedProcess:
    """Run a Python script as a subprocess, streaming output to stdout."""
    log(f"START — {label}")
    t0 = time.perf_counter()

    result = subprocess.run(
        [sys.executable, str(BASE_DIR / script)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=1800,  # 30-minute timeout per step
    )

    elapsed = time.perf_counter() - t0
    print(result.stdout, end="")

    if result.returncode != 0:
        print(result.stderr, end="")
        log(f"FAILED — {label} (exit code {result.returncode}, {elapsed:.0f}s)")
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    log(f"DONE  — {label} ({elapsed:.0f}s)")
    return result


def extract_backtest_metrics(output: str) -> dict:
    """
    Parse backtest_ml.py stdout to extract the ML+SPY >0.55 threshold metrics
    and alpha. Falls back to the first ML threshold if 0.55 is not found.
    """
    import re

    metrics = {}

    # Extract from the ML + SPY idle sweep lines:
    #   ► threshold >0.55 (+ SPY idle) ...  CAGR +13.13%  Sharpe 1.697  Trades 852  WR 58.2%
    pattern = (
        r"threshold >0\.55 \(\+ SPY idle\).*?"
        r"CAGR ([+\-]?[\d.]+)%\s+"
        r"Sharpe ([\d.]+)\s+"
        r"Trades ([\d,]+)\s+"
        r"WR ([\d.]+)%"
    )
    match = re.search(pattern, output)
    if match:
        metrics["cagr"]     = float(match.group(1)) / 100
        metrics["sharpe"]   = float(match.group(2))
        metrics["n_trades"] = int(match.group(3).replace(",", ""))
        metrics["win_rate"] = float(match.group(4)) / 100

    # If ML+SPY 0.55 not found, try pure ML 0.55
    if not metrics:
        pattern_pure = (
            r"threshold >0\.55 \.\.\.\s+"
            r"CAGR ([+\-]?[\d.]+)%\s+"
            r"Sharpe ([\d.]+)\s+"
            r"Trades ([\d,]+)\s+"
            r"WR ([\d.]+)%"
        )
        match = re.search(pattern_pure, output)
        if match:
            metrics["cagr"]     = float(match.group(1)) / 100
            metrics["sharpe"]   = float(match.group(2))
            metrics["n_trades"] = int(match.group(3).replace(",", ""))
            metrics["win_rate"] = float(match.group(4)) / 100

    # Extract alpha from the sweep table (ML + SPY idle, threshold 0.55)
    #   >0.55         +13.13%   1.697    -22.7%        852       58.2%     ...   +9.33%
    alpha_pattern = r">0\.55\s+[+\-]?[\d.]+%\s+[\d.]+\s+[+\-]?[\d.]+%\s+[\d,]+\s+[\d.]+%\s+[\S]+\s+([+\-]?[\d.]+)%"
    alpha_match = re.search(alpha_pattern, output)
    if alpha_match:
        metrics["alpha"] = float(alpha_match.group(1)) / 100

    if not metrics:
        raise RuntimeError("Could not parse backtest metrics from output")

    return metrics


def load_previous_metrics():
    """Load the previous model's metrics from model_metrics.json."""
    if not METRICS_FILE.exists():
        return None
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_metrics(metrics: dict):
    """Save the current model's metrics to model_metrics.json."""
    metrics["timestamp"] = datetime.now(ET).isoformat()
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Metrics saved to {METRICS_FILE}")


def send_email(subject: str, body: str):
    """Send email notification via Gmail SMTP. Silently skips if not configured."""
    if not EMAIL_USER or not EMAIL_APP_PASSWORD:
        log("Email not configured (EMAIL_USER / EMAIL_APP_PASSWORD not set) — skipping")
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


def format_pct(val, decimals=2):
    """Format a float as a percentage string."""
    if val is None:
        return "N/A"
    return f"{val*100:+.{decimals}f}%"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    log("=" * 65)
    log("Monthly Model Retraining")
    log("=" * 65)

    # ── 1. Back up current production model ──────────────────────────────────
    backup_file = None
    if MODEL_FILE.exists():
        backup_file = DATA_DIR / "model_backup.lgb"
        shutil.copy2(MODEL_FILE, backup_file)
        log(f"Backed up current model to {backup_file.name}")

    try:
        # ── 2. Run data pipeline ─────────────────────────────────────────────
        run_step("data_pipeline.py", "Data Pipeline (fetch prices & rebuild features)")

        # ── 3. Run training ──────────────────────────────────────────────────
        run_step("train_model.py", "Model Training (walk-forward LightGBM)")

        # ── 4. Run backtest and capture metrics ──────────────────────────────
        backtest_result = run_step("backtest_ml.py", "Backtest (threshold sweep)")
        new_metrics = extract_backtest_metrics(backtest_result.stdout)
        log(f"New model metrics: CAGR={format_pct(new_metrics.get('cagr'))}  "
            f"Sharpe={new_metrics.get('sharpe', 'N/A'):.3f}  "
            f"WR={format_pct(new_metrics.get('win_rate'))}  "
            f"Alpha={format_pct(new_metrics.get('alpha'))}")

    except RuntimeError as exc:
        # Pipeline step failed — restore backup and notify
        log(f"PIPELINE FAILED: {exc}")
        if backup_file and backup_file.exists():
            shutil.copy2(backup_file, MODEL_FILE)
            log("Restored model from backup")

        send_email(
            "AutoTrader: Model Retrain FAILED",
            f"Model retraining failed at: {datetime.now(ET).isoformat()}\n\n"
            f"Error: {exc}\n\n"
            f"The previous production model has been restored.\n"
            f"Check logs for details.",
        )
        sys.exit(1)

    # ── 5. Compare to previous model ─────────────────────────────────────────
    prev_metrics = load_previous_metrics()

    new_cagr   = new_metrics.get("cagr", 0)
    new_sharpe = new_metrics.get("sharpe", 0)

    if prev_metrics is None:
        # First run — no previous model to compare against
        log("No previous metrics found — deploying as first production model")
        status = "DEPLOYED"
        reason = "First production model (no baseline to compare)"
    else:
        prev_cagr   = prev_metrics.get("cagr", 0)
        prev_sharpe = prev_metrics.get("sharpe", 0)

        cagr_ok   = new_cagr >= prev_cagr
        sharpe_ok = new_sharpe >= prev_sharpe

        log(f"Previous model:  CAGR={format_pct(prev_cagr)}  Sharpe={prev_sharpe:.3f}")
        log(f"New model:       CAGR={format_pct(new_cagr)}  Sharpe={new_sharpe:.3f}")
        log(f"CAGR check:   {'PASS' if cagr_ok else 'FAIL'}  "
            f"({format_pct(new_cagr)} vs {format_pct(prev_cagr)})")
        log(f"Sharpe check: {'PASS' if sharpe_ok else 'FAIL'}  "
            f"({new_sharpe:.3f} vs {prev_sharpe:.3f})")

        if cagr_ok and sharpe_ok:
            status = "DEPLOYED"
            reason = "New model meets or exceeds both CAGR and Sharpe thresholds"
        else:
            status = "REJECTED"
            failures = []
            if not cagr_ok:
                failures.append(f"CAGR declined ({format_pct(new_cagr)} < {format_pct(prev_cagr)})")
            if not sharpe_ok:
                failures.append(f"Sharpe declined ({new_sharpe:.3f} < {prev_sharpe:.3f})")
            reason = "; ".join(failures)

    # ── 6. Deploy or rollback ────────────────────────────────────────────────
    if status == "DEPLOYED":
        log(f"DEPLOYING new model — {reason}")
        save_metrics(new_metrics)
    else:
        log(f"REJECTING new model — {reason}")
        if backup_file and backup_file.exists():
            shutil.copy2(backup_file, MODEL_FILE)
            log("Restored previous production model from backup")

    # Clean up backup
    if backup_file and backup_file.exists():
        backup_file.unlink()

    # ── 7. Email notification ────────────────────────────────────────────────
    prev_cagr_str  = format_pct(prev_metrics.get("cagr")) if prev_metrics else "N/A"
    prev_sharpe_str = f"{prev_metrics.get('sharpe', 0):.3f}" if prev_metrics else "N/A"

    email_subject = (
        f"AutoTrader: Model Retrain Results — "
        f"New CAGR: {format_pct(new_cagr)}, "
        f"Old CAGR: {prev_cagr_str}, "
        f"Status: {status}"
    )

    email_body = (
        f"Model Retrain Results — {datetime.now(ET).strftime('%Y-%m-%d %I:%M %p ET')}\n"
        f"{'=' * 55}\n\n"
        f"Status: {status}\n"
        f"Reason: {reason}\n\n"
        f"{'─' * 55}\n"
        f"{'Metric':<20} {'Previous':<15} {'New':<15}\n"
        f"{'─' * 55}\n"
        f"{'CAGR':<20} {prev_cagr_str:<15} {format_pct(new_cagr):<15}\n"
        f"{'Sharpe':<20} {prev_sharpe_str:<15} {new_sharpe:.3f}\n"
        f"{'Win Rate':<20} "
        f"{format_pct(prev_metrics.get('win_rate')) if prev_metrics else 'N/A':<15} "
        f"{format_pct(new_metrics.get('win_rate')):<15}\n"
        f"{'Alpha':<20} "
        f"{format_pct(prev_metrics.get('alpha')) if prev_metrics else 'N/A':<15} "
        f"{format_pct(new_metrics.get('alpha')):<15}\n"
        f"{'─' * 55}\n\n"
        f"Production model: {'Updated to new model' if status == 'DEPLOYED' else 'Kept previous model'}\n"
    )

    send_email(email_subject, email_body)

    # ── 8. Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    log("")
    log("=" * 65)
    log(f"Retrain complete — Status: {status}")
    log(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log("=" * 65)

    # Exit 0 for deployed, 0 for rejected (both are valid outcomes)
    # Exit 1 only on pipeline failure (handled above)
    return 0


if __name__ == "__main__":
    sys.exit(main())
