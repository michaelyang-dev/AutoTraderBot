#!/usr/bin/env python3
"""
Walk-Forward Phase 2 — Portfolio Construction Fix Comparison
=============================================================
Re-runs walk-forward backtests with old vs new slot config:

  Old: SLOT_LIVE   (ml:2, mom:3, mcap:2, flex:1, max:8, top_n=5)
  New: SLOT_LIVE_V2 (ml:5, mom:3, mcap:0, flex:0, max:8, top_n=8)

Also compares with/without momentum regime filter on the new config.

Uses existing per-year prediction files from ml_service/data/walkforward/.

Produces: ml_service/data/walkforward/PHASE2_RESULTS.md
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
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    MegaCapStrategy, PortfolioManager,
    SLOT_LIVE, SLOT_LIVE_V2,
    load_bars_cached, load_predictions_cached,
    INITIAL_CASH, DATA_DIR,
)
from backtest_utils import calc_metrics, calc_alpha_beta

WF_DIR = DATA_DIR / "walkforward"
YEARS = list(range(2015, 2026))


def log(msg: str):
    print(msg, flush=True)


def run_backtest(year, slot_config, top_n, strat_names, momentum_filter=False):
    """Run backtest for a single year with given config."""
    pred_file = WF_DIR / f"predictions_{year}.parquet"
    if not pred_file.exists():
        return None

    preds = load_predictions_cached(pred_file=pred_file, no_cache=True)
    all_dates = sorted(preds["date"].unique().tolist())
    universe_syms = sorted(preds["symbol"].unique().tolist())

    if len(all_dates) < 10:
        return None

    years_span = max((all_dates[-1] - all_dates[0]).days / 365.25, len(all_dates) / 252.0)

    start_str = (pd.Timestamp(all_dates[0]) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close = load_bars_cached(universe_syms, start_str, end_str)
    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close = close.reindex(sim_index, method="ffill")

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        close = load_bars_cached(universe_syms, start_str, end_str, no_cache=True)
        close = close.reindex(sim_index, method="ffill")
        spy_px = close["SPY"].dropna()
        if spy_px.empty:
            return None

    spy_dict = close["SPY"].to_dict()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    if "prob_ensemble" in preds.columns and "prob" not in preds.columns:
        preds = preds.rename(columns={"prob_ensemble": "prob"})

    strategies = []
    for name in strat_names:
        if name == "ml":
            strategies.append(MLMediumStrategy(preds, threshold=0.55, top_n=top_n, selection_mode="top_n"))
        elif name == "momentum":
            strategies.append(MomentumStrategy(close, volume_data=None, regime_filter=momentum_filter))
        elif name == "mean_reversion":
            strategies.append(MeanReversionStrategy(close, volume_data=None))
        elif name == "mega_cap":
            strategies.append(MegaCapStrategy(close))

    pm = PortfolioManager(strategies=strategies, slot_config=slot_config)
    vals, trades = pm.run(all_dates, spy_prices=spy_dict, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"wf_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    metrics["spy_ret"] = float((spy_px.iloc[-1] / spy_px.iloc[0]) - 1.0)

    return metrics


def run_all_years(label, slot_config, top_n, strat_names, momentum_filter=False):
    """Run all 11 years and return list of metrics dicts."""
    results = []
    for year in YEARS:
        t0 = time.perf_counter()
        m = run_backtest(year, slot_config, top_n, strat_names, momentum_filter)
        if m:
            results.append(m)
            elapsed = time.perf_counter() - t0
            log(f"  {label} {year}: CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:.2f}  "
                f"Beta={m['beta']:.2f}  Alpha={m['alpha']:+.1%}  ({elapsed:.0f}s)")
    return results


def generate_report(old_results, new_results, new_filter_results):
    lines = []
    lines.append("# Portfolio Construction Fix — Walk-Forward Results")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Config Changes\n")
    lines.append("| Parameter | Old (SLOT_LIVE) | New (SLOT_LIVE_V2) |")
    lines.append("|-----------|-----------------|---------------------|")
    lines.append("| ml_medium slots | 2 | **5** |")
    lines.append("| momentum slots | 3 | 3 |")
    lines.append("| mega_cap slots | 2 | **0** (removed) |")
    lines.append("| flex slots | 1 | **0** (removed) |")
    lines.append("| max_positions | 8 | 8 |")
    lines.append("| ML top_n | 5 | **8** |")

    # Old vs New comparison
    lines.append("\n## Old vs New Comparison (no momentum filter)\n")
    lines.append("| Year | Old CAGR | New CAGR | Delta | Old Alpha | New Alpha | Old Beta | New Beta | Old Sharpe | New Sharpe | SPY |")
    lines.append("|------|----------|----------|-------|-----------|-----------|----------|----------|------------|------------|-----|")

    old_by_year = {m["year"]: m for m in old_results}
    new_by_year = {m["year"]: m for m in new_results}

    for year in YEARS:
        o = old_by_year.get(year)
        n = new_by_year.get(year)
        if not o or not n:
            continue

        delta_cagr = n["cagr"] - o["cagr"]
        o_alpha = o.get("alpha") or 0
        n_alpha = n.get("alpha") or 0
        o_beta = o.get("beta") or 0
        n_beta = n.get("beta") or 0

        lines.append(
            f"| {year} "
            f"| {o['cagr']:+.1%} "
            f"| {n['cagr']:+.1%} "
            f"| {delta_cagr:+.1%} "
            f"| {o_alpha:+.1%} "
            f"| {n_alpha:+.1%} "
            f"| {o_beta:.2f} "
            f"| {n_beta:.2f} "
            f"| {o['sharpe']:.2f} "
            f"| {n['sharpe']:.2f} "
            f"| {o['spy_ret']:+.1%} |"
        )

    # Summary stats
    def summarize(results, label):
        cagrs = [m["cagr"] for m in results]
        sharpes = [m["sharpe"] for m in results]
        alphas = [m["alpha"] for m in results if m.get("alpha") is not None]
        betas = [m["beta"] for m in results if m.get("beta") is not None]
        max_dds = [m["max_dd"] for m in results]
        trades = [m["n_trades"] for m in results]

        s = []
        s.append(f"- Median CAGR: **{np.median(cagrs):+.1%}**")
        s.append(f"- Mean CAGR: {np.mean(cagrs):+.1%}")
        s.append(f"- Positive CAGR years: **{sum(1 for c in cagrs if c > 0)}/{len(cagrs)}**")
        s.append(f"- Median Sharpe: **{np.median(sharpes):.2f}**")
        s.append(f"- Mean Sharpe: {np.mean(sharpes):.2f}")
        s.append(f"- Sharpe > 1.0: {sum(1 for s in sharpes if s > 1.0)}/{len(sharpes)} years")
        if alphas:
            s.append(f"- Median alpha: **{np.median(alphas):+.1%}**")
            s.append(f"- Alpha-positive years: **{sum(1 for a in alphas if a > 0)}/{len(alphas)}**")
        if betas:
            s.append(f"- Mean beta: **{np.mean(betas):.2f}**")
        s.append(f"- Worst max DD: {min(max_dds):.1%}")
        s.append(f"- Mean trades/year: {np.mean(trades):.0f}")
        return s

    lines.append("\n### Summary: Old Config (SLOT_LIVE)\n")
    lines.extend(summarize(old_results, "old"))

    lines.append("\n### Summary: New Config (SLOT_LIVE_V2)\n")
    lines.extend(summarize(new_results, "new"))

    # Improvement summary
    old_med_cagr = np.median([m["cagr"] for m in old_results])
    new_med_cagr = np.median([m["cagr"] for m in new_results])
    old_pos = sum(1 for m in old_results if m["cagr"] > 0)
    new_pos = sum(1 for m in new_results if m["cagr"] > 0)
    old_med_alpha = np.median([m["alpha"] for m in old_results if m.get("alpha") is not None])
    new_med_alpha = np.median([m["alpha"] for m in new_results if m.get("alpha") is not None])
    old_beta_mean = np.mean([m["beta"] for m in old_results if m.get("beta") is not None])
    new_beta_mean = np.mean([m["beta"] for m in new_results if m.get("beta") is not None])

    lines.append("\n### Improvement Summary\n")
    lines.append("| Metric | Old | New | Change |")
    lines.append("|--------|-----|-----|--------|")
    lines.append(f"| Median CAGR | {old_med_cagr:+.1%} | {new_med_cagr:+.1%} | {new_med_cagr - old_med_cagr:+.1%} |")
    lines.append(f"| Positive years | {old_pos}/11 | {new_pos}/11 | {new_pos - old_pos:+d} |")
    lines.append(f"| Median alpha | {old_med_alpha:+.1%} | {new_med_alpha:+.1%} | {new_med_alpha - old_med_alpha:+.1%} |")
    lines.append(f"| Mean beta | {old_beta_mean:.2f} | {new_beta_mean:.2f} | {new_beta_mean - old_beta_mean:+.2f} |")

    # With momentum filter
    if new_filter_results:
        lines.append("\n## New Config: With vs Without Momentum Filter\n")
        lines.append("| Year | No Filter CAGR | Filter CAGR | No Filter Sharpe | Filter Sharpe | No Filter DD | Filter DD |")
        lines.append("|------|----------------|-------------|------------------|---------------|--------------|-----------|")

        nf_by_year = {m["year"]: m for m in new_filter_results}
        for year in YEARS:
            n = new_by_year.get(year)
            f = nf_by_year.get(year)
            if not n or not f:
                continue
            lines.append(
                f"| {year} "
                f"| {n['cagr']:+.1%} "
                f"| {f['cagr']:+.1%} "
                f"| {n['sharpe']:.2f} "
                f"| {f['sharpe']:.2f} "
                f"| {n['max_dd']:.1%} "
                f"| {f['max_dd']:.1%} |"
            )

        new_sharpes = [m["sharpe"] for m in new_results]
        filt_sharpes = [m["sharpe"] for m in new_filter_results]
        improved = sum(1 for a, b in zip(new_sharpes, filt_sharpes) if b > a)

        lines.append("")
        lines.append(f"### Filter Impact on New Config\n")
        lines.append(f"- Median CAGR: {np.median([m['cagr'] for m in new_results]):+.1%} -> "
                     f"{np.median([m['cagr'] for m in new_filter_results]):+.1%}")
        lines.append(f"- Median Sharpe: {np.median(new_sharpes):.2f} -> {np.median(filt_sharpes):.2f}")
        lines.append(f"- Filter improves Sharpe: {improved}/{len(new_sharpes)} years")

    # Verdict
    lines.append("\n## Verdict\n")

    criteria_met = []
    criteria_failed = []

    if new_pos >= 7:
        criteria_met.append(f"Positive years: {new_pos}/11 >= 7/11")
    else:
        criteria_failed.append(f"Positive years: {new_pos}/11 < 7/11")

    if new_med_alpha > 0:
        criteria_met.append(f"Median alpha: {new_med_alpha:+.1%} > 0")
    else:
        criteria_failed.append(f"Median alpha: {new_med_alpha:+.1%} <= 0")

    if criteria_failed:
        lines.append("**Deploy decision: NOT READY for live deployment.**\n")
        lines.append("Criteria met:")
        for c in criteria_met:
            lines.append(f"- :white_check_mark: {c}")
        lines.append("\nCriteria failed:")
        for c in criteria_failed:
            lines.append(f"- :x: {c}")
        lines.append("\n**Recommendation**: Further work needed before deploying to live.")
    else:
        lines.append("**Deploy decision: READY for live deployment.**\n")
        for c in criteria_met:
            lines.append(f"- :white_check_mark: {c}")
        lines.append("\n**Recommendation**: Deploy SLOT_LIVE_V2 config to tradingEngine.js.")

    # What improved, what didn't
    lines.append("\n### What changed\n")
    lines.append(f"- **Beta**: {old_beta_mean:.2f} -> {new_beta_mean:.2f} "
                 f"(capital deployment {'improved' if new_beta_mean > old_beta_mean else 'unchanged'})")
    lines.append(f"- **Positive years**: {old_pos}/11 -> {new_pos}/11")
    lines.append(f"- **Median CAGR**: {old_med_cagr:+.1%} -> {new_med_cagr:+.1%}")

    return "\n".join(lines)


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  PHASE 2: PORTFOLIO CONSTRUCTION FIX COMPARISON")
    log("=" * 70)

    # Check predictions exist
    missing = [y for y in YEARS if not (WF_DIR / f"predictions_{y}.parquet").exists()]
    if missing:
        sys.exit(f"Missing predictions for years: {missing}. Run walk_forward_validation.py first.")

    # Config A: Old (SLOT_LIVE, top_n=5, all strategies)
    log("\n── Old config (SLOT_LIVE, top_n=5) ──")
    old_strats = ["ml", "momentum", "mean_reversion", "mega_cap"]
    old_results = run_all_years("OLD", SLOT_LIVE, top_n=5, strat_names=old_strats)

    # Config B: New (SLOT_LIVE_V2, top_n=8, no mega_cap)
    log("\n── New config (SLOT_LIVE_V2, top_n=8, no mega_cap) ──")
    new_strats = ["ml", "momentum"]  # mr:0, mcap:0 — don't instantiate dead strategies
    new_results = run_all_years("NEW", SLOT_LIVE_V2, top_n=8, strat_names=new_strats)

    # Config C: New + momentum filter
    log("\n── New config + momentum regime filter ──")
    new_filter_results = run_all_years("NEW+F", SLOT_LIVE_V2, top_n=8,
                                        strat_names=new_strats, momentum_filter=True)

    # Generate report
    report = generate_report(old_results, new_results, new_filter_results)
    report_file = WF_DIR / "PHASE2_RESULTS.md"
    report_file.write_text(report)

    elapsed = time.perf_counter() - t0
    log(f"\nReport: {report_file}")
    log(f"Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
