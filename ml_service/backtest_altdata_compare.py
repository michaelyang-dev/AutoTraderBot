#!/usr/bin/env python3
"""
A/B Backtest: Baseline (83 features) vs Alt-Data (88 features)
==============================================================
Runs identical combined_live backtest with two different prediction files:
  A) predictions.parquet     — baseline 83-feature model
  B) predictions_v5.parquet  — 88-feature model with 5 insider features

Decision rule:
  Alpha ≥ +17% AND Sharpe ≥ 1.40  →  clear win

Usage:
    python3 backtest_altdata_compare.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    MegaCapStrategy, PortfolioManager,
    SLOT_LIVE, INITIAL_CASH,
    load_bars_cached, load_predictions_cached,
)
from backtest_utils import calc_metrics, calc_alpha_beta

DATA_DIR = Path(__file__).resolve().parent / "data"


def run_one(label: str, pred_file: Path, close, spy_bh, spy_dict, all_dates, volume_data=None):
    """Run combined_live backtest with given predictions and return metrics."""
    print(f"\n{'─'*60}")
    print(f"  Running: {label}")
    print(f"{'─'*60}")

    preds = load_predictions_cached(pred_file=pred_file, no_cache=True)
    # Filter to dates in our sim window
    date_set = set(pd.Timestamp(d) for d in all_dates)
    preds = preds[preds["date"].isin(date_set)]

    years = (all_dates[-1] - all_dates[0]).days / 365.25
    print(f"  Predictions: {len(preds):,} rows | {preds['symbol'].nunique()} symbols | {years:.1f} years")

    ml_strat = MLMediumStrategy(preds, threshold=0.55, top_n=5, selection_mode="top_n")
    mom_strat = MomentumStrategy(close, volume_data=volume_data)
    mr_strat = MeanReversionStrategy(close, volume_data=volume_data)
    mcap_strat = MegaCapStrategy(close)

    strategies = [ml_strat, mom_strat, mr_strat, mcap_strat]
    pm = PortfolioManager(strategies=strategies, slot_config=SLOT_LIVE)
    vals, trades = pm.run(all_dates, spy_prices=spy_dict, price_data=close)

    m = calc_metrics(vals, trades, years, label)
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    m["alpha"] = float(alpha) if not np.isnan(alpha) else 0.0
    m["beta"] = float(beta) if not np.isnan(beta) else 0.0

    print(f"  CAGR: {m['cagr']:+.2%}  Sharpe: {m['sharpe']:.3f}  "
          f"Alpha: {m['alpha']:+.2%}  MaxDD: {m['max_dd']:.1%}  "
          f"Trades: {m['n_trades']:,}  WinRate: {m['win_rate']:.1%}")

    return m


def main():
    t0 = time.perf_counter()

    print("=" * 60)
    print("  A/B BACKTEST: Baseline vs Alt-Data (Insider Features)")
    print("=" * 60)

    # Load baseline predictions to get date range
    baseline_pred = DATA_DIR / "predictions.parquet"
    altdata_pred = DATA_DIR / "predictions_v5.parquet"

    if not baseline_pred.exists():
        sys.exit(f"ERROR: {baseline_pred} not found")
    if not altdata_pred.exists():
        sys.exit(f"ERROR: {altdata_pred} not found")

    # Use baseline dates as reference
    preds_base = pd.read_parquet(baseline_pred)
    preds_base["date"] = pd.to_datetime(preds_base["date"])
    preds_base = preds_base.dropna(subset=["fwd_ret"])

    # Use alt-data dates
    preds_alt = pd.read_parquet(altdata_pred)
    preds_alt["date"] = pd.to_datetime(preds_alt["date"])
    preds_alt = preds_alt.dropna(subset=["fwd_ret"])

    # Common date range
    common_dates = sorted(set(preds_base["date"].unique()) & set(preds_alt["date"].unique()))
    all_syms = sorted(set(preds_base["symbol"].unique()) | set(preds_alt["symbol"].unique()))

    print(f"  Common dates: {len(common_dates)}")
    print(f"  Date range: {pd.Timestamp(common_dates[0]).date()} → {pd.Timestamp(common_dates[-1]).date()}")
    print(f"  Symbols: {len(all_syms)}")

    # Fetch price bars
    start_str = (pd.Timestamp(common_dates[0]) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(common_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    print(f"\n  Fetching price bars ...")
    close = load_bars_cached(all_syms, start_str, end_str)
    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in common_dates])
    close = close.reindex(sim_index, method="ffill")

    spy_px = close["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close["SPY"].to_dict()

    # Run A/B
    m_base = run_one("A: Baseline (83 features)", baseline_pred,
                     close, spy_bh, spy_dict, common_dates)
    m_alt = run_one("B: Alt-Data (88 features, +5 insider)", altdata_pred,
                    close, spy_bh, spy_dict, common_dates)

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print(f"  A/B COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<20} {'A: Baseline':>14} {'B: Alt-Data':>14} {'Delta':>10}")
    print(f"  {'─'*20} {'─'*14} {'─'*14} {'─'*10}")

    for key, fmt, pct in [
        ("cagr", ".2%", True),
        ("sharpe", ".3f", False),
        ("sortino", ".3f", False),
        ("max_dd", ".1%", True),
        ("alpha", ".2%", True),
        ("win_rate", ".1%", True),
        ("n_trades", ",d", False),
    ]:
        va = m_base.get(key, 0)
        vb = m_alt.get(key, 0)
        delta = vb - va

        if pct:
            print(f"  {key:<20} {va:>13{fmt}} {vb:>13{fmt}} {delta:>+9{fmt}}")
        elif fmt == ",d":
            print(f"  {key:<20} {int(va):>14,} {int(vb):>14,} {int(delta):>+10,}")
        else:
            print(f"  {key:<20} {va:>14{fmt}} {vb:>14{fmt}} {delta:>+10{fmt}}")

    # ── Decision ──
    print(f"\n{'='*70}")
    print(f"  DECISION")
    print(f"{'='*70}")

    alpha_b = m_alt.get("alpha", 0) * 100
    sharpe_b = m_alt.get("sharpe", 0)

    alpha_pass = alpha_b >= 17.0
    sharpe_pass = sharpe_b >= 1.40

    print(f"\n  Alpha ≥ +17%?   {alpha_b:>+.2f}%  {'PASS' if alpha_pass else 'FAIL'}")
    print(f"  Sharpe ≥ 1.40?  {sharpe_b:>.3f}   {'PASS' if sharpe_pass else 'FAIL'}")

    if alpha_pass and sharpe_pass:
        verdict = "CLEAR WIN — deploy alt-data model"
    elif alpha_pass or sharpe_pass:
        verdict = "MIXED — review manually before deploying"
    else:
        verdict = "NO IMPROVEMENT — keep baseline model"

    # Also check relative improvement
    cagr_delta = (m_alt["cagr"] - m_base["cagr"]) * 100
    sharpe_delta = m_alt["sharpe"] - m_base["sharpe"]

    print(f"\n  Relative improvement:")
    print(f"    CAGR delta:   {cagr_delta:>+.2f}%")
    print(f"    Sharpe delta: {sharpe_delta:>+.3f}")

    print(f"\n  {'='*50}")
    print(f"  VERDICT: {verdict}")
    print(f"  {'='*50}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
