#!/usr/bin/env python3
"""
Walk-Forward — Regime Sizing Fine-Tuning
==========================================
Find regime multipliers that reduce DD without hurting returns too much.
Base config: TP25 + Trail12 (already confirmed as best).

Tests mild → aggressive regime sizing to find the sweet spot.
"""

import os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    CautiousMLStrategy, compute_regime_live, _consensus_score,
    MomentumStrategy, PortfolioManager, SlotConfig, Signal,
    load_bars_cached, INITIAL_CASH, SLIPPAGE, HOLD_DAYS, POSITION_PCT,
)
from backtest_utils import calc_metrics, calc_alpha_beta

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"
YEARS = list(range(2015, 2026))


def log(msg):
    print(msg, flush=True)


def _fix_prob_col(df):
    if "prob_ensemble" in df.columns and "prob" not in df.columns:
        df = df.rename(columns={"prob_ensemble": "prob"})
    return df


class RegimeTunableML(CautiousMLStrategy):
    """ML with configurable regime-based position sizing."""

    def __init__(self, predictions_df, regime_dict, bull_mult=1.0,
                 caut_mult=1.0, bear_mult=1.0, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._sizing_regime = regime_dict
        self._bull_m = bull_mult
        self._caut_m = caut_mult
        self._bear_m = bear_mult

    def get_position_size(self, signal, portfolio_value, date=None):
        base = portfolio_value * self._position_pct
        regime = self._sizing_regime.get(date, "CAUTIOUS")
        if regime == "BULLISH":
            return base * self._bull_m
        elif regime == "CAUTIOUS":
            return base * self._caut_m
        else:
            return base * self._bear_m


SLOT_CONFIG = SlotConfig(
    strategy_slots={"ml_medium": 5, "momentum": 3},
    flex_slots=0, max_positions=8,
)


def run_variant(preds_df, year, close, regime_dict, variant):
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

    b, c, br = variant["bull"], variant["caut"], variant["bear"]

    ml = RegimeTunableML(
        preds_df, regime_dict=regime_dict,
        bull_mult=b, caut_mult=c, bear_mult=br,
        threshold=0.55, top_n=5, selection_mode="top_n"
    )
    mom = MomentumStrategy(close, volume_data=None, regime_filter=True)

    pm = PortfolioManager(strategies=[ml, mom], slot_config=SLOT_CONFIG)
    vals, trades = pm.run(all_dates, spy_prices=None, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"{variant['name']}_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    return metrics


def main():
    t0 = time.perf_counter()

    log("=" * 80)
    log("  REGIME SIZING FINE-TUNING (base: TP25 + Trail12)")
    log("=" * 80)

    all_preds = {}
    for year in YEARS:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            continue
        df = pd.read_parquet(pred_file)
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = _fix_prob_col(df)

    all_syms = sorted(set().union(*(df["symbol"].unique() for df in all_preds.values())))
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    spy_full = close["SPY"].dropna()
    regime_series = compute_regime_live(spy_full)
    regime_dict = regime_series.to_dict()

    # Variant definitions: (bull, cautious, bear) multipliers
    variants = [
        {"name": "no_regime",    "label": "No regime (baseline)",      "short": "None",  "bull": 1.0, "caut": 1.0, "bear": 1.0},

        # Bear-only reduction (keep cautious at full)
        {"name": "bear_only_70", "label": "Bear only 70%",             "short": "B70",   "bull": 1.0, "caut": 1.0, "bear": 0.7},
        {"name": "bear_only_50", "label": "Bear only 50%",             "short": "B50",   "bull": 1.0, "caut": 1.0, "bear": 0.5},
        {"name": "bear_only_30", "label": "Bear only 30%",             "short": "B30",   "bull": 1.0, "caut": 1.0, "bear": 0.3},
        {"name": "bear_only_0",  "label": "No ML in bear",             "short": "B0",    "bull": 1.0, "caut": 1.0, "bear": 0.0},

        # Mild cautious reduction + bear reduction
        {"name": "mild_a",       "label": "1.0 / 0.9 / 0.5",          "short": "C9B5",  "bull": 1.0, "caut": 0.9, "bear": 0.5},
        {"name": "mild_b",       "label": "1.0 / 0.8 / 0.5",          "short": "C8B5",  "bull": 1.0, "caut": 0.8, "bear": 0.5},
        {"name": "mild_c",       "label": "1.0 / 0.9 / 0.3",          "short": "C9B3",  "bull": 1.0, "caut": 0.9, "bear": 0.3},
        {"name": "mild_d",       "label": "1.0 / 0.8 / 0.3",          "short": "C8B3",  "bull": 1.0, "caut": 0.8, "bear": 0.3},

        # Original for comparison
        {"name": "original",     "label": "1.0 / 0.6 / 0.3 (R1 best)","short": "C6B3",  "bull": 1.0, "caut": 0.6, "bear": 0.3},

        # Bull boost + bear cut
        {"name": "boost_a",      "label": "1.1 / 1.0 / 0.4",          "short": "B11B4", "bull": 1.1, "caut": 1.0, "bear": 0.4},
        {"name": "boost_b",      "label": "1.2 / 1.0 / 0.3",          "short": "B12B3", "bull": 1.2, "caut": 1.0, "bear": 0.3},
        {"name": "boost_c",      "label": "1.1 / 0.9 / 0.4",          "short": "B1C9",  "bull": 1.1, "caut": 0.9, "bear": 0.4},
    ]

    all_results = {v["name"]: [] for v in variants}

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

        for variant in variants:
            metrics = run_variant(preds_year, year, close_year, regime_dict, variant)
            if metrics:
                all_results[variant["name"]].append(metrics)
                log(f"    {variant['short']:6s}  CAGR={metrics['cagr']:+.1%}  "
                    f"Sharpe={metrics['sharpe']:.2f}  MaxDD={metrics['max_dd']:.1%}")

    # ── Report ──
    n = len(all_results["no_regime"])
    if n == 0:
        return

    print("\n" + "=" * 150)
    print("REGIME SIZING FINE-TUNING — RESULTS (base: TP25 + Trail12)")
    print("=" * 150)

    base_cagrs = [m["cagr"] for m in all_results["no_regime"]]
    base_sharpes = [m["sharpe"] for m in all_results["no_regime"]]
    base_dds = [m["max_dd"] for m in all_results["no_regime"]]

    print(f"\n{'Config':<30s} {'B':>4s} {'C':>4s} {'Br':>4s}  "
          f"{'MedCAGR':>8s} {'MnCAGR':>8s} {'MedShp':>7s} {'WrstDD':>8s} {'MnDD':>8s} "
          f"{'dCAGR':>8s} {'dDD':>8s} {'2020c':>7s} {'2020d':>7s} {'2022c':>7s} {'2022d':>7s}")
    print("─" * 150)

    for variant in variants:
        results = all_results[variant["name"]]
        if not results:
            continue

        cagrs = [m["cagr"] for m in results]
        sharpes = [m["sharpe"] for m in results]
        dds = [m["max_dd"] for m in results]

        med_cagr = np.median(cagrs)
        mn_cagr = np.mean(cagrs)
        med_shp = np.median(sharpes)
        worst_dd = min(dds)
        mn_dd = np.mean(dds)

        d_cagr = med_cagr - np.median(base_cagrs)
        d_dd = worst_dd - min(base_dds)

        # Find 2020 and 2022 results
        y2020 = next((m for m in results if m["year"] == 2020), None)
        y2022 = next((m for m in results if m["year"] == 2022), None)
        c20 = y2020["cagr"] if y2020 else 0
        d20 = y2020["max_dd"] if y2020 else 0
        c22 = y2022["cagr"] if y2022 else 0
        d22 = y2022["max_dd"] if y2022 else 0

        b, c, br = variant["bull"], variant["caut"], variant["bear"]
        print(f"  {variant['label']:<28s} {b:>4.1f} {c:>4.1f} {br:>4.1f}  "
              f"{med_cagr:>+7.1%} {mn_cagr:>+7.1%} {med_shp:>7.2f} "
              f"{worst_dd:>8.1%} {mn_dd:>8.1%} "
              f"{d_cagr:>+7.1%} {d_dd:>+7.1%} "
              f"{c20:>+6.1%} {d20:>7.1%} {c22:>+6.1%} {d22:>7.1%}")

    # Per-year CAGR
    print(f"\n{'Year':>6s}", end="")
    for v in variants:
        print(f" {v['short']:>7s}", end="")
    print()
    print("─" * (6 + 8 * len(variants)))

    for yi in range(n):
        yr = all_results["no_regime"][yi]["year"]
        print(f"{yr:>6d}", end="")
        for v in variants:
            c = all_results[v["name"]][yi]["cagr"]
            print(f" {c:>+7.1%}", end="")
        print()

    # Per-year DD
    print(f"\n{'Year':>6s}", end="")
    for v in variants:
        print(f" {v['short']:>7s}", end="")
    print("   (Max DD)")
    print("─" * (6 + 8 * len(variants) + 12))

    for yi in range(n):
        yr = all_results["no_regime"][yi]["year"]
        print(f"{yr:>6d}", end="")
        for v in variants:
            dd = all_results[v["name"]][yi]["max_dd"]
            print(f" {dd:>7.1%}", end="")
        print()

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
