#!/usr/bin/env python3
"""
Exit Reason Analysis
====================
Runs the same walk-forward logic as wf_live_validation.py but collects
trade_details across all years and prints a comprehensive breakdown of
exit reasons: distribution, avg return, win rate, avg hold days, and
per-year breakdown.
"""

import os
import sys
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

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"

YEARS = list(range(2015, 2026))

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


def run_year_details(preds_df, year, close, regime_dict):
    """Run one walk-forward year and return trade_details DataFrame."""
    preds_df = _fix_prob_col(preds_df)
    all_dates = sorted(preds_df["date"].unique().tolist())
    if len(all_dates) < 10:
        return None

    ml_strat = CautiousMLStrategy(
        preds_df, regime_dict=regime_dict,
        threshold=0.55, top_n=5,
        selection_mode="top_n"
    )
    mom_strat = MomentumStrategy(close, volume_data=None, regime_filter=True)
    strategies = [ml_strat, mom_strat]

    pm = PortfolioManager(strategies=strategies, slot_config=SLOT_LIVE_PRODUCTION)
    result = pm.run(all_dates, spy_prices=None, price_data=close, detail_log=True)
    _, _, _, trade_details = result

    if isinstance(trade_details, list):
        if len(trade_details) == 0:
            return None
        trade_details = pd.DataFrame(trade_details)

    if trade_details.empty:
        return None

    trade_details["year"] = year
    return trade_details


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


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

    # Collect all symbols
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
    log(f"Price data: {close.shape[0]} days x {close.shape[1]} symbols")

    # ── Run walk-forward per year, collect trade details ──
    all_details = []
    for year in YEARS:
        if year not in all_preds:
            continue
        preds_df = _fix_prob_col(all_preds[year])
        preds_year = preds_df[preds_df["date"].dt.year == year]
        if preds_year.empty:
            log(f"  {year}: SKIP (no predictions for this year)")
            continue

        all_dates = sorted(preds_year["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        td = run_year_details(preds_year, year, close_year, regime_dict)
        if td is not None:
            all_details.append(td)
            sells = td[td["action"] == "sell"]
            log(f"  {year}: {len(sells)} sell trades collected")

    if not all_details:
        log("No trade details collected.")
        sys.exit(1)

    df_all = pd.concat(all_details, ignore_index=True)
    sells = df_all[df_all["action"] == "sell"].copy()
    log(f"\nTotal sell trades collected: {len(sells)}")

    # ── 1. Exit Reason Distribution ──
    print_section("1. EXIT REASON DISTRIBUTION")
    dist = sells["exit_reason"].value_counts()
    total = len(sells)
    print(f"\n  {'Exit Reason':<25s} {'Count':>8s} {'Pct':>8s}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8}")
    for reason, count in dist.items():
        print(f"  {reason:<25s} {count:>8d} {count/total:>8.1%}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8}")
    print(f"  {'TOTAL':<25s} {total:>8d} {'100.0%':>8s}")

    # ── 2. Average Return by Exit Reason ──
    print_section("2. AVERAGE RETURN BY EXIT REASON")
    grp = sells.groupby("exit_reason")["ret"]
    stats = grp.agg(["mean", "median", "std", "min", "max", "count"])
    stats = stats.sort_values("mean", ascending=False)
    print(f"\n  {'Exit Reason':<25s} {'Mean':>8s} {'Median':>8s} {'StdDev':>8s} {'Min':>8s} {'Max':>8s} {'N':>6s}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 6}")
    for reason, row in stats.iterrows():
        print(f"  {reason:<25s} {row['mean']:>+8.2%} {row['median']:>+8.2%} {row['std']:>8.2%} {row['min']:>+8.2%} {row['max']:>+8.2%} {int(row['count']):>6d}")

    # ── 3. Win Rate by Exit Reason ──
    print_section("3. WIN RATE BY EXIT REASON")
    print(f"\n  {'Exit Reason':<25s} {'Wins':>8s} {'Losses':>8s} {'Total':>8s} {'Win Rate':>10s} {'Avg Win':>10s} {'Avg Loss':>10s}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 10}")
    for reason in dist.index:
        subset = sells[sells["exit_reason"] == reason]
        wins = subset[subset["ret"] > 0]
        losses = subset[subset["ret"] <= 0]
        n = len(subset)
        wr = len(wins) / n if n > 0 else 0
        avg_win = wins["ret"].mean() if len(wins) > 0 else 0
        avg_loss = losses["ret"].mean() if len(losses) > 0 else 0
        print(f"  {reason:<25s} {len(wins):>8d} {len(losses):>8d} {n:>8d} {wr:>10.1%} {avg_win:>+10.2%} {avg_loss:>+10.2%}")

    # ── 4. Average Hold Days by Exit Reason ──
    print_section("4. AVERAGE HOLD DAYS BY EXIT REASON")
    hd = sells.groupby("exit_reason")["hold_days"]
    hd_stats = hd.agg(["mean", "median", "min", "max"])
    print(f"\n  {'Exit Reason':<25s} {'Mean':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for reason in dist.index:
        if reason in hd_stats.index:
            row = hd_stats.loc[reason]
            print(f"  {reason:<25s} {row['mean']:>8.1f} {row['median']:>8.0f} {int(row['min']):>8d} {int(row['max']):>8d}")

    # ── 5. Per-Year Breakdown of Exit Reasons ──
    print_section("5. PER-YEAR EXIT REASON BREAKDOWN")

    # Build pivot table
    year_reason = sells.groupby(["year", "exit_reason"]).size().unstack(fill_value=0)
    all_reasons = sorted(sells["exit_reason"].unique())

    # Print header
    header = f"  {'Year':>6s}"
    for r in all_reasons:
        short = r[:12]
        header += f" {short:>12s}"
    header += f" {'Total':>8s}"
    print(f"\n{header}")
    print(f"  {'─' * 6}" + " ─" * len(all_reasons) * 6 + "")

    for year in sorted(year_reason.index):
        row_str = f"  {year:>6d}"
        row_total = 0
        for r in all_reasons:
            val = year_reason.loc[year, r] if r in year_reason.columns else 0
            row_str += f" {val:>12d}"
            row_total += val
        row_str += f" {row_total:>8d}"
        print(row_str)

    # Also show per-year percentages
    print(f"\n  Per-Year Exit Reason Percentages:")
    header2 = f"  {'Year':>6s}"
    for r in all_reasons:
        short = r[:12]
        header2 += f" {short:>12s}"
    print(f"\n{header2}")

    for year in sorted(year_reason.index):
        row_str = f"  {year:>6d}"
        row_total = year_reason.loc[year].sum()
        for r in all_reasons:
            val = year_reason.loc[year, r] if r in year_reason.columns else 0
            pct = val / row_total if row_total > 0 else 0
            row_str += f" {pct:>11.1%} "
            # row_str += f" {pct:>12.1%}"
        print(row_str)

    # ── 5b. Per-Year Average Return by Exit Reason ──
    print(f"\n  Per-Year Average Return by Exit Reason:")
    header3 = f"  {'Year':>6s}"
    for r in all_reasons:
        short = r[:12]
        header3 += f" {short:>12s}"
    print(f"\n{header3}")

    yr_ret = sells.groupby(["year", "exit_reason"])["ret"].mean().unstack(fill_value=np.nan)
    for year in sorted(yr_ret.index):
        row_str = f"  {year:>6d}"
        for r in all_reasons:
            if r in yr_ret.columns and not np.isnan(yr_ret.loc[year, r]):
                row_str += f" {yr_ret.loc[year, r]:>+11.2%} "
            else:
                row_str += f" {'N/A':>12s}"
        print(row_str)

    # ── 6. Strategy x Exit Reason Crosstab ──
    if "strategy" in sells.columns:
        print_section("6. STRATEGY x EXIT REASON CROSSTAB")
        ct = pd.crosstab(sells["strategy"], sells["exit_reason"])
        print(f"\n{ct.to_string()}")

        print(f"\n  Average Return by Strategy x Exit Reason:")
        sr_ret = sells.groupby(["strategy", "exit_reason"])["ret"].mean().unstack(fill_value=np.nan)
        for strat in sr_ret.index:
            row_str = f"  {strat:<15s}"
            for r in sr_ret.columns:
                val = sr_ret.loc[strat, r]
                if not np.isnan(val):
                    row_str += f"  {r}={val:+.2%}"
            print(row_str)

    elapsed = time.perf_counter() - t0
    print(f"\n\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
