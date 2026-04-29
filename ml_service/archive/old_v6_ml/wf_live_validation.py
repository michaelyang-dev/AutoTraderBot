#!/usr/bin/env python3
"""
Walk-Forward Validation — Exact Live Production Config
========================================================
Runs a proper walk-forward backtest matching the EXACT live production settings:

  Strategies:    ML (CautiousMLStrategy) + Momentum (regime filter ON)
  Slots:         ML=5, Momentum=3, MR=0, MegaCap=0, flex=0, max=8
  SPY parking:   OFF
  CAUTIOUS:      ON (top 2 picks during CAUTIOUS regime)
  ML top_n:      5
  Hold days:     10

Each year Y uses predictions from a model trained on data <= Y-1 (OOS).
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    CautiousMLStrategy, compute_regime_live,
    MomentumStrategy,
    PortfolioManager, SlotConfig,
    load_bars_cached,
    INITIAL_CASH, SLIPPAGE, HOLD_DAYS,
)
from backtest_utils import calc_metrics, calc_alpha_beta

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"

YEARS = list(range(2015, 2026))

# Exact live production slot config
SLOT_LIVE_PRODUCTION = SlotConfig(
    strategy_slots={"ml_medium": 5, "momentum": 3, "mean_reversion": 0, "mega_cap": 0},
    flex_slots=0,
    max_positions=8,
)


def log(msg: str):
    print(msg, flush=True)


def _fix_prob_col(df):
    if "prob_ensemble" in df.columns and "prob" not in df.columns:
        df = df.rename(columns={"prob_ensemble": "prob"})
    return df


def run_year(preds_df, year, close, spy_full, regime_dict):
    """Run one walk-forward year with exact live production config."""
    preds_df = _fix_prob_col(preds_df)

    all_dates = sorted(preds_df["date"].unique().tolist())
    if len(all_dates) < 10:
        return None

    years_span = (all_dates[-1] - all_dates[0]).days / 365.25
    if years_span <= 0:
        years_span = len(all_dates) / 252.0

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        return None

    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    # Exact live strategies:
    # 1. CautiousMLStrategy (CAUTIOUS regime filter ON, top_n=5)
    # 2. MomentumStrategy (regime_filter=True, matching ENABLE_MOMENTUM_REGIME_FILTER)
    ml_strat = CautiousMLStrategy(
        preds_df, regime_dict=regime_dict,
        threshold=0.55, top_n=5,
        selection_mode="top_n"
    )
    mom_strat = MomentumStrategy(close, volume_data=None, regime_filter=True)

    strategies = [ml_strat, mom_strat]

    # SPY parking OFF (spy_prices=None)
    pm = PortfolioManager(strategies=strategies, slot_config=SLOT_LIVE_PRODUCTION)
    result = pm.run(all_dates, spy_prices=None, price_data=close, detail_log=True)
    vals, trades, _, trade_details = result

    metrics = calc_metrics(vals, trades, years_span, f"live_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year

    if not trade_details.empty and "strategy" in trade_details.columns:
        sells = trade_details[trade_details["action"] == "sell"]
        metrics["n_ml_trades"] = int((sells["strategy"] == "ml_medium").sum())
        metrics["n_mom_trades"] = int((sells["strategy"] == "momentum").sum())
    else:
        metrics["n_ml_trades"] = 0
        metrics["n_mom_trades"] = 0
    return metrics


def main():
    t0 = time.perf_counter()

    # ── Load walk-forward predictions ──
    all_preds = {}
    for year in YEARS:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            log(f"  {year}: SKIP (no predictions file)")
            continue
        df = pd.read_parquet(pred_file)
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = df
        log(f"  Loaded predictions_{year}.parquet ({len(df):,} rows)")

    if not all_preds:
        log("ERROR: No walk-forward prediction files found")
        sys.exit(1)

    # Collect all symbols for price download
    all_syms = set()
    for df in all_preds.values():
        all_syms.update(df["symbol"].unique())
    all_syms = sorted(all_syms)

    # ── Load price data ──
    log(f"\nLoading price bars for {len(all_syms)} symbols ...")
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    if "SPY" not in close.columns:
        log("ERROR: SPY not in price data")
        sys.exit(1)

    spy_full = close["SPY"].dropna()
    regime_series = compute_regime_live(spy_full)
    regime_dict = regime_series.to_dict()
    log(f"Price data: {close.shape[0]} days × {close.shape[1]} symbols")

    # ── Run walk-forward per year ──
    all_results = []
    for year in YEARS:
        if year not in all_preds:
            continue

        preds_df = _fix_prob_col(all_preds[year])

        # Filter to this year only
        preds_year = preds_df[preds_df["date"].dt.year == year]
        if preds_year.empty:
            log(f"  {year}: SKIP (no predictions for this year)")
            continue

        # Reindex close data to include all prediction dates
        all_dates = sorted(preds_year["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        # Regime stats for this year
        yr_regime = regime_series[(regime_series.index >= pd.Timestamp(f"{year}-01-01"))
                                  & (regime_series.index <= pd.Timestamp(f"{year}-12-31"))]
        vc = yr_regime.value_counts()
        bull, caut, bear = int(vc.get("BULLISH", 0)), int(vc.get("CAUTIOUS", 0)), int(vc.get("BEARISH", 0))

        metrics = run_year(preds_year, year, close_year, spy_full, regime_dict)
        if metrics is None:
            log(f"  {year}: SKIP (insufficient data)")
            continue

        all_results.append(metrics)
        log(f"  {year}: CAGR={metrics['cagr']:+.1%}  Sharpe={metrics['sharpe']:.2f}  "
            f"MaxDD={metrics['max_dd']:.1%}  Alpha={metrics.get('alpha', 0):+.1%}  "
            f"Trades={metrics['n_trades']}  Win={metrics['win_rate']:.0%}  "
            f"(ML={metrics['n_ml_trades']}, Mom={metrics['n_mom_trades']})  "
            f"[BULL={bull} CAUT={caut} BEAR={bear}]")

    if not all_results:
        log("No results — check walk-forward prediction files.")
        sys.exit(1)

    # ── Summary ──
    n = len(all_results)
    cagrs = [m["cagr"] for m in all_results]
    sharpes = [m["sharpe"] for m in all_results]
    dds = [m["max_dd"] for m in all_results]
    alphas = [m["alpha"] for m in all_results if m.get("alpha") is not None]
    betas = [m["beta"] for m in all_results if m.get("beta") is not None]
    win_rates = [m["win_rate"] for m in all_results]
    trades = [m["n_trades"] for m in all_results]

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION — EXACT LIVE PRODUCTION CONFIG")
    print("=" * 70)
    print(f"\nConfig: ML(cautious, top5) + Momentum(regime_filter) | Slots: ML=5, Mom=3 | SPY parking: OFF")
    print(f"Years: {YEARS[0]}-{YEARS[-1]} ({n} years with data)")
    print(f"Method: Year Y uses model trained on data <= Y-1 (out-of-sample)")

    print(f"\n{'─' * 50}")
    print(f"  {'Metric':<25s} {'Median':>10s} {'Mean':>10s} {'Best':>10s} {'Worst':>10s}")
    print(f"  {'─' * 25}  {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(f"  {'CAGR':<25s} {np.median(cagrs):>+10.1%} {np.mean(cagrs):>+10.1%} {max(cagrs):>+10.1%} {min(cagrs):>+10.1%}")
    print(f"  {'Sharpe':<25s} {np.median(sharpes):>10.2f} {np.mean(sharpes):>10.2f} {max(sharpes):>10.2f} {min(sharpes):>10.2f}")
    print(f"  {'Max Drawdown':<25s} {np.median(dds):>10.1%} {np.mean(dds):>10.1%} {max(dds):>10.1%} {min(dds):>10.1%}")
    if alphas:
        print(f"  {'Alpha vs SPY':<25s} {np.median(alphas):>+10.1%} {np.mean(alphas):>+10.1%} {max(alphas):>+10.1%} {min(alphas):>+10.1%}")
    if betas:
        print(f"  {'Beta':<25s} {np.median(betas):>10.2f} {np.mean(betas):>10.2f} {max(betas):>10.2f} {min(betas):>10.2f}")
    print(f"  {'Win Rate':<25s} {np.median(win_rates):>10.0%} {np.mean(win_rates):>10.0%} {max(win_rates):>10.0%} {min(win_rates):>10.0%}")
    print(f"  {'Trades / year':<25s} {np.median(trades):>10.0f} {np.mean(trades):>10.0f} {max(trades):>10.0f} {min(trades):>10.0f}")

    pos_cagr = sum(1 for c in cagrs if c > 0)
    pos_alpha = sum(1 for a in alphas if a > 0)
    print(f"\n  Positive CAGR years:   {pos_cagr}/{n}")
    print(f"  Alpha-positive years:  {pos_alpha}/{n}")

    # Per-year table
    print(f"\n{'─' * 50}")
    print(f"  {'Year':>6s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s} {'Beta':>6s} {'Win%':>6s} {'Trades':>7s} {'ML':>4s} {'Mom':>4s}")
    for m in all_results:
        a = f"{m['alpha']:+.1%}" if m.get("alpha") is not None else "N/A"
        b = f"{m['beta']:.2f}" if m.get("beta") is not None else "N/A"
        print(f"  {m['year']:>6d} {m['cagr']:>+8.1%} {m['sharpe']:>8.2f} {m['max_dd']:>8.1%} {a:>8s} {b:>6s} {m['win_rate']:>6.0%} {m['n_trades']:>7d} {m['n_ml_trades']:>4d} {m['n_mom_trades']:>4d}")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save results
    out_path = WF_DIR / "live_production_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
