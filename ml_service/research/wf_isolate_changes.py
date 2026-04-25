#!/usr/bin/env python3
"""
Isolate Impact of Three Changes vs Live Production
====================================================
5 configs to isolate each change between live production and backtester default:

  1. Baseline (=live):  top_n=5, CAUTIOUS filter ON, SPY parking ON
  2. +top8:             top_n=8, CAUTIOUS filter ON, SPY parking ON
  3. -CAUTIOUS:         top_n=5, CAUTIOUS filter OFF, SPY parking ON
  4. -parking:          top_n=5, CAUTIOUS filter ON, SPY parking OFF
  5. All three (=B):    top_n=8, CAUTIOUS filter OFF, SPY parking OFF

Uses existing walk-forward predictions (trained OOS per year).
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
    MLMediumStrategy, CautiousMLStrategy, compute_regime_live,
    MomentumStrategy, MeanReversionStrategy,
    MegaCapStrategy, PortfolioManager, Signal,
    SLOT_LIVE_V2, load_bars_cached,
    INITIAL_CASH, SLIPPAGE, HOLD_DAYS,
    POSITION_PCT,
)
from backtest_utils import calc_metrics, calc_alpha_beta

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"

YEARS = list(range(2015, 2026))


def log(msg: str):
    print(msg, flush=True)


# ── Generic backtest runner ─────────────────────────────────────────────────

def _fix_prob_col(df):
    if "prob_ensemble" in df.columns and "prob" not in df.columns:
        df = df.rename(columns={"prob_ensemble": "prob"})
    return df


def run_config(preds_df, year, close, spy_full, regime_dict,
               top_n, cautious_filter, spy_parking, label):
    """Run a single config with the specified settings."""
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

    spy_dict = close["SPY"].to_dict() if spy_parking else None
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    # ML strategy: with or without CAUTIOUS filter
    if cautious_filter:
        ml_strat = CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                                      threshold=0.55, top_n=top_n,
                                      selection_mode="top_n")
    else:
        ml_strat = MLMediumStrategy(preds_df, threshold=0.55, top_n=top_n,
                                    selection_mode="top_n")

    strategies = [
        ml_strat,
        MomentumStrategy(close, volume_data=None, regime_filter=True),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]

    pm = PortfolioManager(strategies=strategies, slot_config=SLOT_LIVE_V2)
    vals, trades = pm.run(all_dates, spy_prices=spy_dict, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"{label}_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    return metrics


# ── Config definitions ──────────────────────────────────────────────────────

CONFIGS = [
    {
        "name": "baseline",
        "label": "Baseline (Live)",
        "short": "Baseline",
        "top_n": 5,
        "cautious_filter": True,
        "spy_parking": True,
    },
    {
        "name": "+top8",
        "label": "+top8 only",
        "short": "+top8",
        "top_n": 8,
        "cautious_filter": True,
        "spy_parking": True,
    },
    {
        "name": "-cautious",
        "label": "-CAUTIOUS only",
        "short": "-CAUTIOUS",
        "top_n": 5,
        "cautious_filter": False,
        "spy_parking": True,
    },
    {
        "name": "-parking",
        "label": "-parking only",
        "short": "-parking",
        "top_n": 5,
        "cautious_filter": True,
        "spy_parking": False,
    },
    {
        "name": "all_three",
        "label": "All three (=B)",
        "short": "All three",
        "top_n": 8,
        "cautious_filter": False,
        "spy_parking": False,
    },
]


# ── Report generation ──────────────────────────────────────────────────────

def generate_report(all_results, regime_stats):
    n = len(YEARS)
    lines = []
    lines.append("# Isolated Impact of Three Changes vs Live Production")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Methodology")
    lines.append("Proper walk-forward: year Y uses model trained only on data <= Y-1.")
    lines.append("Same universe, features, model. Only difference is strategy/portfolio construction.")

    # Config table
    lines.append("\n## Configs Tested")
    lines.append("")
    lines.append("| # | Config | top_n | CAUTIOUS filter | SPY parking | Change from baseline |")
    lines.append("|---|--------|-------|-----------------|-------------|---------------------|")
    changes = ["(reference)", "top_n 5->8", "CAUTIOUS filter OFF", "SPY parking OFF", "all three changes"]
    for i, cfg in enumerate(CONFIGS):
        caut = "ON (top 2)" if cfg["cautious_filter"] else "OFF"
        park = "ON" if cfg["spy_parking"] else "OFF"
        lines.append(f"| {i+1} | {cfg['label']} | {cfg['top_n']} | {caut} | {park} | {changes[i]} |")

    # Per-config per-year results
    lines.append("\n## Per-Year Results by Config")
    for ci, cfg in enumerate(CONFIGS):
        lines.append(f"\n### {cfg['label']}")
        lines.append("")
        lines.append("| Year | CAGR | Sharpe | Max DD | Alpha | Beta | Trades | Win% |")
        lines.append("|------|------|--------|--------|-------|------|--------|------|")
        for m in all_results[ci]:
            a_str = f"{m['alpha']:+.1%}" if m.get("alpha") is not None else "N/A"
            b_str = f"{m['beta']:.2f}" if m.get("beta") is not None else "N/A"
            lines.append(
                f"| {m['year']} | {m['cagr']:+.1%} | {m['sharpe']:.2f} "
                f"| {m['max_dd']:.1%} | {a_str} | {b_str} "
                f"| {m['n_trades']} | {m['win_rate']:.1%} |"
            )

    # Head-to-head CAGR
    lines.append("\n## Head-to-Head: CAGR by Year")
    lines.append("")
    hdr = "| Year | " + " | ".join(c["short"] for c in CONFIGS) + " | Best |"
    sep = "|------|" + "|".join("-" * (len(c["short"]) + 2) for c in CONFIGS) + "|------|"
    lines.append(hdr)
    lines.append(sep)
    for yi in range(n):
        yr = all_results[0][yi]["year"]
        cagrs = [all_results[ci][yi]["cagr"] for ci in range(5)]
        best_idx = int(np.argmax(cagrs))
        best_lbl = CONFIGS[best_idx]["short"]
        row = f"| {yr} | " + " | ".join(f"{c:+.1%}" for c in cagrs) + f" | **{best_lbl}** |"
        lines.append(row)

    # Head-to-head Sharpe
    lines.append("\n## Head-to-Head: Sharpe by Year")
    lines.append("")
    lines.append(hdr.replace("CAGR", "Sharpe"))
    lines.append(sep)
    for yi in range(n):
        yr = all_results[0][yi]["year"]
        sharpes = [all_results[ci][yi]["sharpe"] for ci in range(5)]
        best_idx = int(np.argmax(sharpes))
        best_lbl = CONFIGS[best_idx]["short"]
        row = f"| {yr} | " + " | ".join(f"{s:.2f}" for s in sharpes) + f" | **{best_lbl}** |"
        lines.append(row)

    # Summary metrics
    lines.append("\n## Summary Metrics")
    lines.append("")
    hdr2 = "| Metric | " + " | ".join(c["short"] for c in CONFIGS) + " |"
    sep2 = "|--------|" + "|".join("-" * (len(c["short"]) + 2) for c in CONFIGS) + "|"
    lines.append(hdr2)
    lines.append(sep2)

    def extract(field):
        return [[m[field] for m in all_results[ci]] for ci in range(5)]

    def extract_nn(field):
        return [[m[field] for m in all_results[ci] if m.get(field) is not None] for ci in range(5)]

    all_cagrs = extract("cagr")
    all_sharpes = extract("sharpe")
    all_dds = extract("max_dd")
    all_alphas = extract_nn("alpha")
    all_betas = extract_nn("beta")
    all_trades = extract("n_trades")
    all_wr = extract("win_rate")

    def row_fmt(label, vals, fmt):
        return f"| {label} | " + " | ".join(fmt.format(v) for v in vals) + " |"

    lines.append(row_fmt("Median CAGR", [np.median(c) for c in all_cagrs], "{:+.1%}"))
    lines.append(row_fmt("Mean CAGR", [np.mean(c) for c in all_cagrs], "{:+.1%}"))
    lines.append(row_fmt("Median Sharpe", [np.median(c) for c in all_sharpes], "{:.2f}"))
    lines.append(row_fmt("Mean Sharpe", [np.mean(c) for c in all_sharpes], "{:.2f}"))
    lines.append(row_fmt("Worst Max DD", [min(c) for c in all_dds], "{:.1%}"))
    lines.append(row_fmt("Mean Max DD", [np.mean(c) for c in all_dds], "{:.1%}"))
    lines.append(row_fmt("Median Alpha", [np.median(c) for c in all_alphas], "{:+.1%}"))
    lines.append(row_fmt("Mean Beta", [np.mean(c) for c in all_betas], "{:.2f}"))

    pos_cagr = [sum(1 for c in cs if c > 0) for cs in all_cagrs]
    pos_alpha = [sum(1 for a in al if a > 0) for al in all_alphas]
    lines.append(f"| Positive CAGR years | " + " | ".join(f"{p}/{n}" for p in pos_cagr) + " |")
    lines.append(f"| Alpha-positive years | " + " | ".join(f"{p}/{n}" for p in pos_alpha) + " |")
    lines.append(row_fmt("Mean trades/year", [np.mean(c) for c in all_trades], "{:.0f}"))
    lines.append(row_fmt("Mean win rate", [np.mean(c) for c in all_wr], "{:.1%}"))

    # Per-change impact vs baseline
    lines.append("\n## Per-Change Impact vs Baseline")
    lines.append("")
    lines.append("| Change | \u0394 Median CAGR | \u0394 Median Sharpe | \u0394 Worst DD | Years Sharpe helped | Years Sharpe hurt |")
    lines.append("|--------|--------------|----------------|-----------|--------------------|--------------------|")

    base_med_cagr = np.median(all_cagrs[0])
    base_med_sharpe = np.median(all_sharpes[0])
    base_worst_dd = min(all_dds[0])

    change_deltas = []  # (name, d_cagr, d_sharpe, d_dd, helped, hurt)
    for ci in range(1, 5):
        med_cagr = np.median(all_cagrs[ci])
        med_sharpe = np.median(all_sharpes[ci])
        worst_dd = min(all_dds[ci])
        d_cagr = med_cagr - base_med_cagr
        d_sharpe = med_sharpe - base_med_sharpe
        d_dd = worst_dd - base_worst_dd
        helped = sum(1 for a, b in zip(all_sharpes[0], all_sharpes[ci]) if b > a)
        hurt = sum(1 for a, b in zip(all_sharpes[0], all_sharpes[ci]) if b < a)
        change_deltas.append((CONFIGS[ci]["short"], d_cagr, d_sharpe, d_dd, helped, hurt))
        lines.append(
            f"| {CONFIGS[ci]['short']} | {d_cagr:+.1%} | {d_sharpe:+.2f} "
            f"| {d_dd:+.1%} | {helped}/{n} | {hurt}/{n} |"
        )

    # Stacking check
    lines.append("\n## Stacking Check")
    lines.append("Does the sum of individual changes approximate the combined change?")
    lines.append("")
    sum_d_cagr = sum(d[1] for d in change_deltas[:3])
    sum_d_sharpe = sum(d[2] for d in change_deltas[:3])
    combined_d_cagr = change_deltas[3][1]
    combined_d_sharpe = change_deltas[3][2]
    gap_cagr = combined_d_cagr - sum_d_cagr
    gap_sharpe = combined_d_sharpe - sum_d_sharpe

    lines.append("| | \u0394 Median CAGR | \u0394 Median Sharpe |")
    lines.append("|---|--------------|----------------|")
    lines.append(f"| Sum of individual changes | {sum_d_cagr:+.1%} | {sum_d_sharpe:+.2f} |")
    lines.append(f"| Combined (all three) | {combined_d_cagr:+.1%} | {combined_d_sharpe:+.02f} |")
    lines.append(f"| Gap (interaction effects) | {gap_cagr:+.1%} | {gap_sharpe:+.02f} |")

    if abs(gap_sharpe) < 0.10:
        lines.append("\nChanges are approximately additive -- minimal interaction effects.")
    elif gap_sharpe > 0:
        lines.append("\nPositive interaction: changes are synergistic (combined > sum of parts).")
    else:
        lines.append("\nNegative interaction: some changes partially overlap in effect.")

    # Sharpe win counts
    lines.append("\n## Sharpe Win Counts")
    lines.append("")
    lines.append("| Config | Wins (best Sharpe that year) |")
    lines.append("|--------|-----------------------------|")
    wins = [0] * 5
    for yi in range(n):
        sharpes = [all_results[ci][yi]["sharpe"] for ci in range(5)]
        wins[int(np.argmax(sharpes))] += 1
    for ci in range(5):
        lines.append(f"| {CONFIGS[ci]['short']} | {wins[ci]}/{n} |")

    # Bear market focus
    lines.append("\n## Bear Market Focus (2022)")
    lines.append("")
    lines.append("| Config | CAGR | Sharpe | Max DD |")
    lines.append("|--------|------|--------|--------|")
    for ci in range(5):
        m = next((m for m in all_results[ci] if m["year"] == 2022), None)
        if m:
            lines.append(f"| {CONFIGS[ci]['short']} | {m['cagr']:+.1%} | {m['sharpe']:.2f} | {m['max_dd']:.1%} |")

    # Recommendation
    lines.append("\n## Recommendation")
    lines.append("")

    # Rank by Sharpe impact
    ranked = sorted(change_deltas[:3], key=lambda x: x[2], reverse=True)
    lines.append("### Ranked by Sharpe impact (individual changes only)")
    lines.append("")
    for i, (name, d_cagr, d_sharpe, d_dd, helped, hurt) in enumerate(ranked, 1):
        risk = "low" if name == "-parking" else ("low" if name == "-CAUTIOUS" else "medium")
        lines.append(
            f"{i}. **{name}**: \u0394Sharpe={d_sharpe:+.02f}, "
            f"\u0394CAGR={d_cagr:+.1%}, \u0394DD={d_dd:+.1%}, "
            f"helped {helped}/{n} years, hurt {hurt}/{n} years "
            f"(implementation risk: {risk})"
        )

    lines.append("")

    # Check if one change dominates
    total_sharpe_gain = combined_d_sharpe
    if total_sharpe_gain > 0:
        biggest = ranked[0]
        biggest_pct = biggest[2] / total_sharpe_gain * 100 if total_sharpe_gain > 0 else 0
        if biggest_pct >= 80:
            lines.append(
                f"**{biggest[0]}** provides {biggest_pct:.0f}% of the total Sharpe improvement. "
                f"Deploy this change first for maximum impact with minimal disruption."
            )
        elif biggest_pct >= 50:
            lines.append(
                f"**{biggest[0]}** provides {biggest_pct:.0f}% of the Sharpe improvement. "
                f"Consider deploying it first, then adding the others incrementally."
            )
        else:
            lines.append(
                "No single change dominates. All three contribute meaningfully. "
                "Consider deploying all three together."
            )

    # Check for bear-regime concerns
    lines.append("")
    for name, d_cagr, d_sharpe, d_dd, helped, hurt in change_deltas[:3]:
        m_base_2022 = next((m for m in all_results[0] if m["year"] == 2022), None)
        ci = [c["short"] for c in CONFIGS].index(name)
        m_cfg_2022 = next((m for m in all_results[ci] if m["year"] == 2022), None)
        if m_base_2022 and m_cfg_2022:
            dd_delta = m_cfg_2022["max_dd"] - m_base_2022["max_dd"]
            if dd_delta < -0.02:
                lines.append(
                    f"**WARNING**: {name} worsens 2022 drawdown by {dd_delta:.1%}. "
                    f"Review bear-market risk carefully."
                )

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  ISOLATE IMPACT OF THREE CHANGES VS LIVE PRODUCTION")
    log("  1:Baseline  2:+top8  3:-CAUTIOUS  4:-parking  5:All three")
    log("=" * 70)

    # Load predictions
    all_preds = {}
    for year in YEARS:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            log(f"ERROR: {pred_file} not found.")
            sys.exit(1)
        df = pd.read_parquet(pred_file)
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = df
        log(f"  Loaded predictions_{year}.parquet ({len(df):,} rows)")

    all_syms = set()
    for df in all_preds.values():
        all_syms.update(df["symbol"].unique())
    all_syms = sorted(all_syms)

    log(f"\nLoading price bars for {len(all_syms)} symbols ...")
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    log(f"  Price data: {close.shape}")

    spy_full = close["SPY"].dropna()
    regime_series = compute_regime_live(spy_full)
    regime_dict = regime_series.to_dict()

    regime_stats = {}
    for year in YEARS:
        ys = pd.Timestamp(f"{year}-01-01")
        ye = pd.Timestamp(f"{year}-12-31")
        yr = regime_series[(regime_series.index >= ys) & (regime_series.index <= ye)]
        vc = yr.value_counts()
        regime_stats[year] = {
            "BULLISH": int(vc.get("BULLISH", 0)),
            "CAUTIOUS": int(vc.get("CAUTIOUS", 0)),
            "BEARISH": int(vc.get("BEARISH", 0)),
        }

    # Run all configs
    all_results = [[] for _ in CONFIGS]

    for year in YEARS:
        log(f"\n{'─'*70}")
        s = regime_stats[year]
        log(f"  YEAR {year}  BULL={s['BULLISH']}d  CAUT={s['CAUTIOUS']}d  BEAR={s['BEARISH']}d")
        log(f"{'─'*70}")

        preds_df = all_preds[year]
        all_dates = sorted(preds_df["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        for ci, cfg in enumerate(CONFIGS):
            t1 = time.perf_counter()
            m = run_config(
                preds_df.copy(), year, close_year, spy_full, regime_dict,
                top_n=cfg["top_n"],
                cautious_filter=cfg["cautious_filter"],
                spy_parking=cfg["spy_parking"],
                label=cfg["name"],
            )
            if m:
                all_results[ci].append(m)
                elapsed = time.perf_counter() - t1
                log(f"  {cfg['short']:15s} CAGR={m['cagr']:+.1%}  "
                    f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']:.1%}  "
                    f"Trades={m['n_trades']}  ({elapsed:.1f}s)")

    # Generate report
    log(f"\n{'='*70}")
    log("  GENERATING REPORT")
    log(f"{'='*70}")

    report = generate_report(all_results, regime_stats)
    report_file = WF_DIR / "ISOLATED_CHANGES.md"
    report_file.write_text(report)
    log(f"\nReport: {report_file}")

    # Save JSON
    for ci, cfg in enumerate(CONFIGS):
        json_file = WF_DIR / f"isolated_{cfg['name']}_results.json"
        json_file.write_text(json.dumps(all_results[ci], indent=2, default=str))

    total = time.perf_counter() - t0
    log(f"\nTotal: {total:.0f}s ({total/60:.1f}min)")
    log("Done.")


if __name__ == "__main__":
    main()
