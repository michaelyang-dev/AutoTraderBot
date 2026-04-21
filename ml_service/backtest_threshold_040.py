#!/usr/bin/env python3
"""
Backtest V3 model at threshold 0.40 (instead of 0.55).
Uses existing predictions.parquet — no retraining.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR   = Path(__file__).resolve().parent / "data"
PRED_FILE  = DATA_DIR / "predictions.parquet"
THRESHOLD  = 0.40

# Baselines
V1_CAGR, V1_SHARPE = 0.2408, 1.655
V3_055_CAGR, V3_055_SHARPE, V3_055_DD = 0.3280, 1.520, -0.3405


def log(msg: str):
    print(msg, flush=True)


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log(f"  V3 BACKTEST — Threshold {THRESHOLD}")
    log("=" * 70)

    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS,
        MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from backtest_utils import calc_metrics, calc_alpha_beta
    from diagnose_combined import instrumented_run

    # Load predictions
    preds_df = pd.read_parquet(PRED_FILE)
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df = preds_df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    log(f"\n  Predictions: {len(preds_df):,} rows  |  {len(universe_syms)} symbols  |  {years:.1f} years")
    log(f"  Date range: {pd.Timestamp(all_dates[0]).date()} → {pd.Timestamp(all_dates[-1]).date()}")

    # Threshold analysis
    n_above = (preds_df["prob"] >= THRESHOLD).sum()
    log(f"  Predictions >= {THRESHOLD}: {n_above:,} / {len(preds_df):,} ({n_above/len(preds_df)*100:.1f}%)")

    # Fetch OHLCV
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

    log(f"  Fetching OHLCV for {len(all_syms)} symbols ...")
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

    # Build strategies — ML Medium at 0.40 threshold
    ml_strat = MLMediumStrategy(preds_df, threshold=THRESHOLD)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # Run backtest
    log(f"\n  Running Path B (ML Medium @ {THRESHOLD} + Momentum + Mean Reversion) ...")
    vals, trades, diag = instrumented_run(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B")

    m = calc_metrics(vals, trades, years, "Path B")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    m["alpha"] = alpha
    m["beta"] = beta
    m["avg_pos"] = float(np.mean(diag["daily_pos_count"]))

    # ── Year-by-year returns ─────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("YEAR-BY-YEAR RETURNS")
    log(f"{'='*70}")

    vals_series = vals.copy()
    vals_series.index = pd.to_datetime(vals_series.index)
    spy_aligned = spy_bh.reindex(vals_series.index, method="ffill")

    log(f"\n  {'Year':<6} {'Strategy':>10} {'SPY':>10} {'Alpha':>10}")
    log(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10}")

    yearly_returns = {}
    unique_years = sorted(set(vals_series.index.year))
    for yr in unique_years:
        yr_mask = vals_series.index.year == yr
        yr_vals = vals_series[yr_mask]
        yr_spy = spy_aligned[yr_mask]

        if len(yr_vals) < 2:
            continue

        strat_ret = yr_vals.iloc[-1] / yr_vals.iloc[0] - 1
        spy_ret = yr_spy.iloc[-1] / yr_spy.iloc[0] - 1 if len(yr_spy) >= 2 and yr_spy.iloc[0] > 0 else np.nan
        yr_alpha = strat_ret - spy_ret if not np.isnan(spy_ret) else np.nan

        yearly_returns[yr] = {"strategy": strat_ret, "spy": spy_ret, "alpha": yr_alpha}
        log(f"  {yr:<6} {strat_ret*100:>9.2f}% {spy_ret*100:>9.2f}% {yr_alpha*100:>+9.2f}%")

    # Count years with positive alpha
    pos_alpha_years = sum(1 for v in yearly_returns.values() if v["alpha"] > 0)
    log(f"\n  Positive alpha years: {pos_alpha_years}/{len(yearly_returns)}")

    # ── Comparison table ─────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("COMPARISON")
    log(f"{'='*70}")

    cagr = m["cagr"]
    sharpe = m["sharpe"]
    sortino = m["sortino"]
    max_dd = m["max_dd"]
    win_rate = m["win_rate"]
    n_trades = m["n_trades"]

    log(f"\n  {'Metric':<20} {'V3 @0.40':>12} {'V3 @0.55':>12} {'V1 (60sym)':>12}")
    log(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")
    log(f"  {'CAGR':<20} {cagr*100:>11.2f}% {V3_055_CAGR*100:>11.2f}% {V1_CAGR*100:>11.2f}%")
    log(f"  {'Sharpe':<20} {sharpe:>12.3f} {V3_055_SHARPE:>12.3f} {V1_SHARPE:>12.3f}")
    log(f"  {'Sortino':<20} {sortino:>12.3f}")
    log(f"  {'Max Drawdown':<20} {max_dd*100:>11.2f}% {V3_055_DD*100:>11.2f}%")
    log(f"  {'Win Rate':<20} {win_rate*100:>11.2f}%")
    log(f"  {'Trades':<20} {n_trades:>12,}")
    log(f"  {'Alpha':<20} {alpha*100:>11.2f}%")
    log(f"  {'Beta':<20} {beta:>12.3f}")
    log(f"  {'Avg Positions':<20} {m['avg_pos']:>12.1f}")

    # ── Decision ─────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("DEPLOYMENT DECISION")
    log(f"{'='*70}")

    if cagr >= 0.28 and sharpe >= 1.55 and max_dd >= -0.28:
        verdict = "STRONG BUY — DEPLOY"
        reason = f"CAGR {cagr*100:.1f}% >= 28%, Sharpe {sharpe:.3f} >= 1.55, DD {max_dd*100:.1f}% >= -28%"
    elif cagr >= 0.25 and sharpe >= 1.60:
        verdict = "DEPLOY"
        reason = f"CAGR {cagr*100:.1f}% >= 25%, Sharpe {sharpe:.3f} >= 1.60"
    else:
        verdict = "DO NOT DEPLOY"
        reasons = []
        if cagr < 0.25:
            reasons.append(f"CAGR {cagr*100:.1f}% < 25%")
        if sharpe < 1.55:
            reasons.append(f"Sharpe {sharpe:.3f} < 1.55")
        if max_dd < -0.28:
            reasons.append(f"DD {max_dd*100:.1f}% < -28%")
        reason = "; ".join(reasons) if reasons else "Mixed results"

    log(f"\n  {'='*50}")
    log(f"  VERDICT: {verdict}")
    log(f"  Reason:  {reason}")
    log(f"  {'='*50}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
