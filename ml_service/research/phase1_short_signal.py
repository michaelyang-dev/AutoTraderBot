#!/usr/bin/env python3
"""
Phase 1: Short Signal Validation
=================================
Does the ML model predict UNDERPERFORMERS (bottom-5)?

For each walk-forward year (2015-2025):
  - Each day, rank stocks by prob_ensemble
  - Compute forward 10-day return for top-5, bottom-5, middle, random-5
  - Test if bottom-5 reliably underperforms

Decision criteria:
  - bottom-5 avg < -0.3%: STRONG signal → proceed
  - bottom-5 avg -0.3% to 0%: WEAK → ask user
  - bottom-5 avg > 0%: NO signal → STOP

Output: ml_service/data/longshort/PHASE1_SHORT_SIGNAL.md
"""

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"
OUT_DIR = DATA_DIR / "longshort"
YEARS = list(range(2015, 2026))


def log(msg: str):
    print(msg, flush=True)


def analyze_year(year):
    """Analyze top-5 vs bottom-5 vs random signal quality for one year."""
    pred_file = WF_DIR / f"predictions_{year}.parquet"
    if not pred_file.exists():
        return None

    df = pd.read_parquet(pred_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret", "prob_ensemble"])

    dates = sorted(df["date"].unique())
    np.random.seed(42 + year)

    top5_rets = []
    bot5_rets = []
    mid_rets = []
    rand5_rets = []
    top5_minus_bot5 = []

    # Per-day spreads for t-test
    daily_spreads = []

    for date in dates:
        day = df[df["date"] == date].copy()
        if len(day) < 20:
            continue

        day = day.sort_values("prob_ensemble", ascending=False)

        top5 = day.head(5)
        bot5 = day.tail(5)
        mid = day.iloc[len(day) // 3 : 2 * len(day) // 3]
        rand5 = day.sample(min(5, len(day)))

        t5_ret = top5["fwd_ret"].mean()
        b5_ret = bot5["fwd_ret"].mean()
        m_ret = mid["fwd_ret"].mean()
        r5_ret = rand5["fwd_ret"].mean()

        top5_rets.append(t5_ret)
        bot5_rets.append(b5_ret)
        mid_rets.append(m_ret)
        rand5_rets.append(r5_ret)
        top5_minus_bot5.append(t5_ret - b5_ret)
        daily_spreads.append(t5_ret - b5_ret)

    if not top5_rets:
        return None

    # T-test: is the top-bottom spread significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(daily_spreads, 0)

    # Also test if bottom-5 is significantly below 0
    t_bot, p_bot = stats.ttest_1samp(bot5_rets, 0)

    return {
        "year": year,
        "n_days": len(top5_rets),
        "top5_mean": np.mean(top5_rets),
        "bot5_mean": np.mean(bot5_rets),
        "mid_mean": np.mean(mid_rets),
        "rand5_mean": np.mean(rand5_rets),
        "spread_mean": np.mean(daily_spreads),
        "spread_median": np.median(daily_spreads),
        "spread_std": np.std(daily_spreads),
        "spread_t": t_stat,
        "spread_p": p_value,
        "bot5_t": t_bot,
        "bot5_p": p_bot,
        "top5_median": np.median(top5_rets),
        "bot5_median": np.median(bot5_rets),
        # Quintile analysis
        "top5_rets": top5_rets,
        "bot5_rets": bot5_rets,
        "daily_spreads": daily_spreads,
    }


def generate_report(results):
    lines = []
    lines.append("# Phase 1: Short Signal Validation")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("\n## Question")
    lines.append("\nDoes the ML model predict **underperformers** (bottom-5 by prob_ensemble)?")
    lines.append("If bottom-5 picks reliably have negative forward returns, we can short them.")
    lines.append("\n## Methodology")
    lines.append("\nFor each trading day in each walk-forward year:")
    lines.append("1. Rank all S&P 500 stocks by `prob_ensemble` (ML prediction)")
    lines.append("2. Select Top-5 (highest prob), Bottom-5 (lowest prob), Middle third, Random-5")
    lines.append("3. Compute mean forward 10-day return for each group")
    lines.append("4. Test statistical significance of Top-Bottom spread")

    # Per-year table
    lines.append("\n## Results Per Year\n")
    lines.append("| Year | Days | Top-5 Fwd | Bot-5 Fwd | Middle Fwd | Random | T-B Spread | Spread p-val | Bot<0 p-val |")
    lines.append("|------|------|-----------|-----------|------------|--------|------------|--------------|-------------|")

    for r in results:
        sig = "***" if r["spread_p"] < 0.01 else "**" if r["spread_p"] < 0.05 else "*" if r["spread_p"] < 0.10 else ""
        lines.append(
            f"| {r['year']} "
            f"| {r['n_days']} "
            f"| {r['top5_mean']:+.3%} "
            f"| {r['bot5_mean']:+.3%} "
            f"| {r['mid_mean']:+.3%} "
            f"| {r['rand5_mean']:+.3%} "
            f"| {r['spread_mean']:+.3%}{sig} "
            f"| {r['spread_p']:.4f} "
            f"| {r['bot5_p']:.4f} |"
        )

    # Aggregate stats
    all_top5 = np.mean([r["top5_mean"] for r in results])
    all_bot5 = np.mean([r["bot5_mean"] for r in results])
    all_mid = np.mean([r["mid_mean"] for r in results])
    all_rand = np.mean([r["rand5_mean"] for r in results])
    all_spread = np.mean([r["spread_mean"] for r in results])

    # Pool all daily spreads for aggregate t-test
    all_daily_spreads = []
    all_bot5_daily = []
    for r in results:
        all_daily_spreads.extend(r["daily_spreads"])
        all_bot5_daily.extend(r["bot5_rets"])

    agg_t, agg_p = stats.ttest_1samp(all_daily_spreads, 0)
    bot_t, bot_p = stats.ttest_1samp(all_bot5_daily, 0)

    lines.append("\n## Aggregate Summary (all 11 years pooled)\n")
    lines.append(f"- **Top-5 avg forward 10d return**: {all_top5:+.3%}")
    lines.append(f"- **Bottom-5 avg forward 10d return**: {all_bot5:+.3%}")
    lines.append(f"- **Middle avg forward 10d return**: {all_mid:+.3%}")
    lines.append(f"- **Random avg forward 10d return**: {all_rand:+.3%}")
    lines.append(f"- **Top-Bottom spread**: {all_spread:+.3%}")
    lines.append(f"- **Spread t-statistic**: {agg_t:.2f}")
    lines.append(f"- **Spread p-value**: {agg_p:.6f}")
    lines.append(f"- **Bottom-5 < 0 t-stat**: {bot_t:.2f}, p={bot_p:.6f}")
    lines.append(f"- **Positive spread years**: {sum(1 for r in results if r['spread_mean'] > 0)}/{len(results)}")

    # Monotonicity check: is there a gradient from top to bottom?
    lines.append("\n## Monotonicity Check\n")
    lines.append("Is there a smooth gradient from top to bottom of the probability ranking?\n")
    lines.append("| Quintile | Avg Fwd 10d Ret | vs Random |")
    lines.append("|----------|-----------------|-----------|")

    # Recompute quintile returns across all years
    all_quintile_rets = {q: [] for q in range(5)}
    for year in YEARS:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            continue
        df = pd.read_parquet(pred_file)
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["fwd_ret", "prob_ensemble"])
        for date in df["date"].unique():
            day = df[df["date"] == date].sort_values("prob_ensemble", ascending=False)
            if len(day) < 20:
                continue
            n = len(day)
            q_size = n // 5
            for q in range(5):
                start = q * q_size
                end = start + q_size if q < 4 else n
                q_ret = day.iloc[start:end]["fwd_ret"].mean()
                all_quintile_rets[q].append(q_ret)

    quintile_labels = ["Q1 (highest prob)", "Q2", "Q3 (middle)", "Q4", "Q5 (lowest prob)"]
    for q in range(5):
        avg = np.mean(all_quintile_rets[q])
        vs_rand = avg - all_rand
        lines.append(f"| {quintile_labels[q]} | {avg:+.3%} | {vs_rand:+.3%} |")

    q1_avg = np.mean(all_quintile_rets[0])
    q5_avg = np.mean(all_quintile_rets[4])
    monotonic = all(
        np.mean(all_quintile_rets[i]) >= np.mean(all_quintile_rets[i + 1])
        for i in range(4)
    )
    lines.append(f"\n- **Q1-Q5 spread**: {q1_avg - q5_avg:+.3%}")
    lines.append(f"- **Monotonic decreasing**: {'YES' if monotonic else 'NO'}")

    # Decision
    lines.append("\n## Decision\n")

    if all_bot5 < -0.003:
        decision = "STRONG"
        lines.append(f"**STRONG SHORT SIGNAL** (bottom-5 avg = {all_bot5:+.3%} < -0.3%)")
        lines.append(f"\nThe ML model reliably identifies underperformers. "
                     f"Bottom-5 picks have negative forward returns on average. "
                     f"Proceed to Phase 2.")
    elif all_bot5 < 0:
        decision = "WEAK"
        lines.append(f"**WEAK SHORT SIGNAL** (bottom-5 avg = {all_bot5:+.3%}, between -0.3% and 0%)")
        lines.append(f"\nThe bottom-5 picks underperform slightly but the signal is weak. "
                     f"A long/short strategy may work but with thin margins. "
                     f"**User decision required** before proceeding to Phase 2.")
    else:
        decision = "NONE"
        lines.append(f"**NO SHORT SIGNAL** (bottom-5 avg = {all_bot5:+.3%} >= 0%)")
        lines.append(f"\nThe ML model does NOT predict underperformers. "
                     f"Bottom-5 picks still have positive forward returns. "
                     f"B4 long/short strategy **will not work** with current model. "
                     f"**STOP — do not proceed to Phase 2.**")

    # Additional context regardless of decision
    lines.append("\n### Key context\n")
    if all_spread > 0:
        lines.append(f"- The Top-Bottom spread IS positive ({all_spread:+.3%}/trade), meaning the model "
                     f"does rank stocks correctly (top > bottom)")
    else:
        lines.append(f"- The Top-Bottom spread is negative ({all_spread:+.3%}/trade) — the model "
                     f"ranking is not even directionally correct")

    if bot_p < 0.05:
        lines.append(f"- Bottom-5 returns are statistically significantly different from zero (p={bot_p:.4f})")
    else:
        lines.append(f"- Bottom-5 returns are NOT statistically distinguishable from zero (p={bot_p:.4f})")

    # What the spread means for long/short
    if all_spread > 0:
        annual_spread = all_spread * 252 / 10  # 10-day hold, ~25 non-overlapping periods/yr
        lines.append(f"\n### Long/short P&L estimate")
        lines.append(f"- Per-trade spread: {all_spread:+.3%}")
        lines.append(f"- ~25 non-overlapping 10-day periods per year")
        lines.append(f"- Estimated annual long-short alpha: ~{annual_spread:+.1%} "
                     f"(before slippage, borrow costs, and execution)")

    return "\n".join(lines), decision


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  PHASE 1: SHORT SIGNAL VALIDATION")
    log("=" * 70)

    results = []
    for year in YEARS:
        log(f"  Analyzing {year} ...")
        r = analyze_year(year)
        if r:
            results.append(r)
            log(f"    Top-5: {r['top5_mean']:+.3%}  Bot-5: {r['bot5_mean']:+.3%}  "
                f"Spread: {r['spread_mean']:+.3%} (p={r['spread_p']:.4f})")

    if not results:
        sys.exit("No walk-forward predictions found.")

    report, decision = generate_report(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUT_DIR / "PHASE1_SHORT_SIGNAL.md"
    report_file.write_text(report)

    elapsed = time.perf_counter() - t0
    log(f"\n  Report: {report_file}")
    log(f"  Decision: {decision}")
    log(f"  Runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
