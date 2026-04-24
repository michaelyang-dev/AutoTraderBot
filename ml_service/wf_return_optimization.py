#!/usr/bin/env python3
"""
Walk-Forward Test: Return Optimization Variants
=================================================
Tests multiple parameter changes to improve returns.

Variants:
  1. Baseline        — current live (top5, hold10, ML5+Mom3)
  2. top3            — more concentrated ML picks
  3. top7            — more diversified ML picks
  4. hold5           — shorter hold period
  5. hold15          — longer hold period
  6. hold20          — even longer hold
  7. ml_only         — drop momentum entirely, ML gets all 8 slots
  8. ml7_mom1        — shift slots: ML=7, Mom=1
  9. equal_weight    — equal position sizing (no confidence scaling)
  10. top3_hold15    — concentrated + longer hold (best of both?)
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
    MomentumStrategy,
    PortfolioManager, SlotConfig, Strategy, Signal,
    load_bars_cached,
    INITIAL_CASH, SLIPPAGE, HOLD_DAYS, POSITION_PCT,
)
from backtest_utils import calc_metrics, calc_alpha_beta

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"
YEARS = list(range(2015, 2026))


def log(msg: str):
    print(msg, flush=True)


def _fix_prob_col(df):
    if "prob_ensemble" in df.columns and "prob" not in df.columns:
        df = df.rename(columns={"prob_ensemble": "prob"})
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Equal-weight ML Strategy (no confidence scaling)
# ══════════════════════════════════════════════════════════════════════════════

class EqualWeightMLStrategy(CautiousMLStrategy):
    """Same as CautiousMLStrategy but equal position sizing."""

    def get_position_size(self, signal, portfolio_value, date=None):
        return portfolio_value * self._position_pct


# ══════════════════════════════════════════════════════════════════════════════
#  Config runner
# ══════════════════════════════════════════════════════════════════════════════

def run_config(preds_df, year, close, regime_dict, config):
    """Run one year with a specific config."""
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

    top_n = config.get("top_n", 5)
    hold_days = config.get("hold_days", 10)
    ml_slots = config.get("ml_slots", 5)
    mom_slots = config.get("mom_slots", 3)
    max_pos = config.get("max_pos", 8)
    use_equal_weight = config.get("equal_weight", False)
    include_mom = config.get("include_mom", True)

    # Build ML strategy
    if use_equal_weight:
        ml = EqualWeightMLStrategy(preds_df, regime_dict=regime_dict,
                                   threshold=0.55, top_n=top_n,
                                   selection_mode="top_n")
    else:
        ml = CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                                threshold=0.55, top_n=top_n,
                                selection_mode="top_n")

    strategies = [ml]
    slots = {"ml_medium": ml_slots}

    if include_mom:
        mom = MomentumStrategy(close, volume_data=None, regime_filter=True)
        strategies.append(mom)
        slots["momentum"] = mom_slots

    slot_config = SlotConfig(
        strategy_slots=slots,
        flex_slots=0,
        max_positions=max_pos,
    )

    pm = PortfolioManager(strategies=strategies, slot_config=slot_config,
                          hold_days=hold_days)
    vals, trades = pm.run(all_dates, spy_prices=None, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"{config['name']}_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Configs
# ══════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    {"name": "baseline",     "label": "Baseline (live)",        "short": "Base",
     "top_n": 5, "hold_days": 10, "ml_slots": 5, "mom_slots": 3, "max_pos": 8},

    {"name": "top3",         "label": "Top 3 (concentrated)",   "short": "Top3",
     "top_n": 3, "hold_days": 10, "ml_slots": 5, "mom_slots": 3, "max_pos": 8},

    {"name": "top7",         "label": "Top 7 (diversified)",    "short": "Top7",
     "top_n": 7, "hold_days": 10, "ml_slots": 5, "mom_slots": 3, "max_pos": 8},

    {"name": "hold5",        "label": "Hold 5 days",            "short": "Hld5",
     "top_n": 5, "hold_days": 5, "ml_slots": 5, "mom_slots": 3, "max_pos": 8},

    {"name": "hold15",       "label": "Hold 15 days",           "short": "Hld15",
     "top_n": 5, "hold_days": 15, "ml_slots": 5, "mom_slots": 3, "max_pos": 8},

    {"name": "hold20",       "label": "Hold 20 days",           "short": "Hld20",
     "top_n": 5, "hold_days": 20, "ml_slots": 5, "mom_slots": 3, "max_pos": 8},

    {"name": "ml_only",      "label": "ML only (no momentum)",  "short": "MLonly",
     "top_n": 5, "hold_days": 10, "ml_slots": 8, "mom_slots": 0, "max_pos": 8,
     "include_mom": False},

    {"name": "ml7_mom1",     "label": "ML=7 Mom=1",             "short": "ML7M1",
     "top_n": 5, "hold_days": 10, "ml_slots": 7, "mom_slots": 1, "max_pos": 8},

    {"name": "equal_weight", "label": "Equal weight sizing",    "short": "EqWt",
     "top_n": 5, "hold_days": 10, "ml_slots": 5, "mom_slots": 3, "max_pos": 8,
     "equal_weight": True},

    {"name": "top3_hold15",  "label": "Top3 + Hold15",          "short": "T3H15",
     "top_n": 3, "hold_days": 15, "ml_slots": 5, "mom_slots": 3, "max_pos": 8},

    {"name": "top3_ml_only", "label": "Top3 + ML only",         "short": "T3ML",
     "top_n": 3, "hold_days": 10, "ml_slots": 8, "mom_slots": 0, "max_pos": 8,
     "include_mom": False},
]


def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  RETURN OPTIMIZATION — WALK-FORWARD COMPARISON")
    log("=" * 70)

    # Load predictions
    all_preds = {}
    for year in YEARS:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            continue
        df = pd.read_parquet(pred_file)
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = _fix_prob_col(df)
        log(f"  Loaded predictions_{year}.parquet ({len(df):,} rows)")

    all_syms = sorted(set().union(*(df["symbol"].unique() for df in all_preds.values())))

    log(f"\nLoading price bars for {len(all_syms)} symbols ...")
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    log(f"  Price data: {close.shape}")

    spy_full = close["SPY"].dropna()
    regime_series = compute_regime_live(spy_full)
    regime_dict = regime_series.to_dict()

    # Run all configs × all years
    all_results = {cfg["name"]: [] for cfg in CONFIGS}

    for year in YEARS:
        if year not in all_preds:
            continue
        preds_df = all_preds[year]
        preds_year = preds_df[preds_df["date"].dt.year == year]
        if preds_year.empty:
            continue

        all_dates = sorted(preds_year["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        yr = regime_series[(regime_series.index >= pd.Timestamp(f"{year}-01-01"))
                           & (regime_series.index <= pd.Timestamp(f"{year}-12-31"))]
        vc = yr.value_counts()
        log(f"\n  {year}  BULL={vc.get('BULLISH', 0)}  CAUT={vc.get('CAUTIOUS', 0)}  BEAR={vc.get('BEARISH', 0)}")

        for cfg in CONFIGS:
            metrics = run_config(preds_year, year, close_year, regime_dict, cfg)
            if metrics:
                all_results[cfg["name"]].append(metrics)
                log(f"    {cfg['short']:8s}  CAGR={metrics['cagr']:+.1%}  "
                    f"Sharpe={metrics['sharpe']:.2f}  MaxDD={metrics['max_dd']:.1%}  "
                    f"Alpha={metrics.get('alpha', 0):+.1%}")

    # ══════════════════════════════════════════════════════════════════════
    #  Report
    # ══════════════════════════════════════════════════════════════════════
    n = len(all_results["baseline"])
    if n == 0:
        log("No results.")
        return

    print("\n" + "=" * 130)
    print("RETURN OPTIMIZATION — WALK-FORWARD RESULTS")
    print("=" * 130)

    # Summary table
    print(f"\n{'Metric':<22s}", end="")
    for cfg in CONFIGS:
        print(f" {cfg['short']:>7s}", end="")
    print()
    print("─" * (22 + 8 * len(CONFIGS)))

    def row(label, values, fmt):
        print(f"  {label:<20s}", end="")
        for v in values:
            print(f" {fmt.format(v):>7s}", end="")
        print()

    def get_vals(field):
        return [
            [m[field] for m in all_results[cfg["name"]]]
            for cfg in CONFIGS
        ]

    cagrs = get_vals("cagr")
    sharpes = get_vals("sharpe")
    dds = get_vals("max_dd")
    alphas_raw = get_vals("alpha")
    alphas = [[a for a in al if a is not None] for al in alphas_raw]
    win_rates = get_vals("win_rate")
    n_trades = get_vals("n_trades")

    row("Median CAGR", [np.median(c) for c in cagrs], "{:+.1%}")
    row("Mean CAGR", [np.mean(c) for c in cagrs], "{:+.1%}")
    row("Median Sharpe", [np.median(c) for c in sharpes], "{:.2f}")
    row("Mean Sharpe", [np.mean(c) for c in sharpes], "{:.2f}")
    row("Worst Max DD", [min(c) for c in dds], "{:.1%}")
    row("Mean Max DD", [np.mean(c) for c in dds], "{:.1%}")
    if all(len(a) > 0 for a in alphas):
        row("Median Alpha", [np.median(a) for a in alphas], "{:+.1%}")
    row("Mean Win Rate", [np.mean(w) for w in win_rates], "{:.0%}")
    row("Mean Trades/yr", [np.mean(t) for t in n_trades], "{:.0f}")

    pos_cagr = [sum(1 for c in cs if c > 0) for cs in cagrs]
    print(f"  {'Positive CAGR yrs':<20s}", end="")
    for p in pos_cagr:
        print(f" {p:>4d}/{n:<2d}", end="")
    print()

    # Per-year CAGR
    print(f"\n{'Year':<6s}", end="")
    for cfg in CONFIGS:
        print(f" {cfg['short']:>7s}", end="")
    print(f" {'Best':>7s}")
    print("─" * (6 + 8 * len(CONFIGS) + 8))

    for yi in range(n):
        yr = all_results["baseline"][yi]["year"]
        year_cagrs = [all_results[cfg["name"]][yi]["cagr"] for cfg in CONFIGS]
        best_idx = int(np.argmax(year_cagrs))
        print(f"{yr:<6d}", end="")
        for c in year_cagrs:
            print(f" {c:>+7.1%}", end="")
        print(f" {CONFIGS[best_idx]['short']:>7s}")

    # Per-year Sharpe
    print(f"\n{'Year':<6s}", end="")
    for cfg in CONFIGS:
        print(f" {cfg['short']:>7s}", end="")
    print(f" {'Best':>7s}")
    print("─" * (6 + 8 * len(CONFIGS) + 8))

    for yi in range(n):
        yr = all_results["baseline"][yi]["year"]
        year_sharpes = [all_results[cfg["name"]][yi]["sharpe"] for cfg in CONFIGS]
        best_idx = int(np.argmax(year_sharpes))
        print(f"{yr:<6d}", end="")
        for s in year_sharpes:
            print(f" {s:>7.2f}", end="")
        print(f" {CONFIGS[best_idx]['short']:>7s}")

    # Delta vs baseline
    print(f"\n{'─' * 90}")
    print("DELTA vs BASELINE (sorted by Median Sharpe improvement)")
    print(f"{'Variant':<25s} {'ΔCAGR':>8s} {'ΔSharpe':>8s} {'ΔWorstDD':>9s} {'Sharpe↑':>8s} {'Sharpe↓':>8s}")
    print("─" * 90)

    base_sharpes = sharpes[0]
    deltas = []
    for ci in range(1, len(CONFIGS)):
        cfg = CONFIGS[ci]
        d_cagr = np.median(cagrs[ci]) - np.median(cagrs[0])
        d_sharpe = np.median(sharpes[ci]) - np.median(sharpes[0])
        d_dd = min(dds[ci]) - min(dds[0])
        helped = sum(1 for a, b in zip(base_sharpes, sharpes[ci]) if b > a)
        hurt = sum(1 for a, b in zip(base_sharpes, sharpes[ci]) if b < a)
        deltas.append((ci, cfg, d_cagr, d_sharpe, d_dd, helped, hurt))

    deltas.sort(key=lambda x: x[3], reverse=True)
    for ci, cfg, d_cagr, d_sharpe, d_dd, helped, hurt in deltas:
        print(f"  {cfg['label']:<23s} {d_cagr:>+8.1%} {d_sharpe:>+8.2f} {d_dd:>+9.1%} {helped:>5d}/{n:<2d} {hurt:>5d}/{n:<2d}")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    out = {cfg["name"]: all_results[cfg["name"]] for cfg in CONFIGS}
    out_path = WF_DIR / "return_optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
