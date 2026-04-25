#!/usr/bin/env python3
"""
Deploy Gate — validates a newly trained model before production swap.

Runs a 90-day backtest on the new predictions and checks absolute thresholds:
  1. Sharpe >= 1.0
  2. CAGR > 0% (annualized, not compared to multi-year baseline)
  3. Max DD <= 25%
  4. Ensemble AUC >= 0.55 (if calibration data available)

Usage:
    from deploy_gate import run_deploy_gate
    passed, metrics, reasons = run_deploy_gate(predictions_path)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Absolute thresholds (no baseline comparison) ─────────────────────────────

GATE_SHARPE_MIN  = 1.0     # 90-day windows are noisy; 1.0 is reasonable
GATE_CAGR_MIN    = 0.0     # must be positive (not losing money)
GATE_MAX_DD      = -0.25   # max drawdown <= 25%
GATE_AUC_MIN     = 0.55    # ensemble AUC sanity check
LOOKBACK_DAYS    = 90       # backtest window

BASELINE_FILE    = Path(__file__).resolve().parent / "data" / "deploy_baseline.json"


def save_baseline(metrics: dict, path: Path = BASELINE_FILE):
    """Save current metrics as reference (informational, not used for gating)."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def _backtest_90d(preds_df: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS) -> dict:
    """
    Simple top-5 long-only backtest over the last `lookback_days` trading days.
    Returns dict with sharpe, cagr, max_dd, n_trades.

    Strategy: each day, go equal-weight long the top 5 stocks by prob_ensemble.
    Hold for 10 days (matching the model's forward-return target).
    """
    preds_df = preds_df.copy()
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    all_dates = sorted(preds_df["date"].unique())
    if len(all_dates) < lookback_days:
        lookback_days = len(all_dates)

    test_dates = all_dates[-lookback_days:]
    test_df = preds_df[preds_df["date"].isin(test_dates)].copy()

    if "fwd_ret" not in test_df.columns or test_df["fwd_ret"].isna().all():
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "n_trades": 0,
                "error": "No forward returns in test window"}

    # Daily portfolio returns: equal-weight top 5 each day
    daily_returns = []
    n_trades = 0

    for date in test_dates:
        day = test_df[test_df["date"] == date].dropna(subset=["fwd_ret", "prob_ensemble"])
        if len(day) == 0:
            daily_returns.append(0.0)
            continue

        top5 = day.nlargest(5, "prob_ensemble")
        # Use per-day return: fwd_ret is 10-day, so divide by 10 for daily approx
        avg_ret = top5["fwd_ret"].mean() / 10.0
        daily_returns.append(avg_ret)
        n_trades += len(top5)

    daily_returns = np.array(daily_returns)

    # Metrics
    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "n_trades": n_trades}

    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    cum = np.cumprod(1 + daily_returns)
    total_ret = cum[-1] / cum[0] - 1
    years = len(daily_returns) / 252
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))

    return {
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(max_dd), 4),
        "n_trades": n_trades,
    }


def run_deploy_gate(
    predictions_path: Path,
    lookback_days: int = LOOKBACK_DAYS,
    ensemble_auc: float = None,
) -> tuple[bool, dict, list[str]]:
    """
    Run deploy gate validation using absolute thresholds only.

    Args:
        predictions_path: Path to predictions.parquet
        lookback_days: Number of trading days for backtest window
        ensemble_auc: Optional AUC from training (checked if provided)

    Returns:
        (passed, metrics, reasons)
        passed: True if all checks pass
        metrics: dict of computed metrics
        reasons: list of failure reasons (empty if passed)
    """
    preds = pd.read_parquet(predictions_path)
    metrics = _backtest_90d(preds, lookback_days)

    if "error" in metrics:
        return False, metrics, [f"Backtest error: {metrics['error']}"]

    if ensemble_auc is not None:
        metrics["ensemble_auc"] = round(ensemble_auc, 4)

    reasons = []

    # Check 1: Sharpe >= 1.0
    if metrics["sharpe"] < GATE_SHARPE_MIN:
        reasons.append(
            f"Sharpe {metrics['sharpe']:.3f} < {GATE_SHARPE_MIN} minimum"
        )

    # Check 2: CAGR > 0% (model must not lose money over 90 days)
    if metrics["cagr"] <= GATE_CAGR_MIN:
        reasons.append(
            f"CAGR {metrics['cagr']:.4f} <= {GATE_CAGR_MIN} (model is losing money)"
        )

    # Check 3: Max DD within limit
    if metrics["max_dd"] < GATE_MAX_DD:
        reasons.append(
            f"Max DD {metrics['max_dd']:.4f} worse than {GATE_MAX_DD} limit"
        )

    # Check 4: AUC sanity (only if provided)
    if ensemble_auc is not None and ensemble_auc < GATE_AUC_MIN:
        reasons.append(
            f"Ensemble AUC {ensemble_auc:.4f} < {GATE_AUC_MIN} minimum"
        )

    passed = len(reasons) == 0
    return passed, metrics, reasons


if __name__ == "__main__":
    import sys
    pred_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent / "data" / "predictions.parquet"
    )
    passed, metrics, reasons = run_deploy_gate(pred_path)
    print(f"\nDeploy Gate: {'PASSED' if passed else 'FAILED'}")
    print(f"  Sharpe:  {metrics.get('sharpe', 'N/A')}  (min: {GATE_SHARPE_MIN})")
    print(f"  CAGR:    {metrics.get('cagr', 'N/A')}  (min: > {GATE_CAGR_MIN})")
    print(f"  Max DD:  {metrics.get('max_dd', 'N/A')}  (limit: {GATE_MAX_DD})")
    print(f"  Trades:  {metrics.get('n_trades', 'N/A')}")
    if reasons:
        print(f"\nFailure reasons:")
        for r in reasons:
            print(f"  - {r}")
    sys.exit(0 if passed else 1)
