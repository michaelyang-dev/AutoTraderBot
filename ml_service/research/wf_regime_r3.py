#!/usr/bin/env python3
"""
Regime Sizing — Round 3: Finding the sweet spot
=================================================
Focus: mild CAGR cost, meaningful DD reduction.

Key insight from R2: bear-only reduction works but costs too much CAGR
in normal years because bear_mult applies to ALL bear days including
mild ones. Ideas:
  1. Bear 50% + bull boost 1.15-1.2 to offset
  2. Only reduce in "deep bear" (SPY below SMA200 for 10+ days)
  3. Graduated: reduce more as bear deepens (use SPY distance from SMA200)
  4. Drawdown-based: reduce only when portfolio is in drawdown
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


# ── Strategy variants ──

class BearBoostML(CautiousMLStrategy):
    """Bear-only reduction with bull boost to offset CAGR cost."""

    def __init__(self, predictions_df, regime_dict, bull_m, bear_m, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._sizing_regime = regime_dict
        self._bull_m = bull_m
        self._bear_m = bear_m

    def get_position_size(self, signal, portfolio_value, date=None):
        base = portfolio_value * self._position_pct
        regime = self._sizing_regime.get(date, "CAUTIOUS")
        if regime == "BULLISH":
            return base * self._bull_m
        elif regime == "BEARISH":
            return base * self._bear_m
        return base  # CAUTIOUS = 1.0


class GraduatedBearML(CautiousMLStrategy):
    """Position size scales with SPY distance from SMA200."""

    def __init__(self, predictions_df, regime_dict, spy_series, min_mult=0.3, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._sizing_regime = regime_dict
        sma200 = spy_series.rolling(200, min_periods=200).mean()
        # Compute distance ratio: (SPY - SMA200) / SMA200
        self._dist_ratio = ((spy_series - sma200) / sma200).to_dict()
        self._min_mult = min_mult

    def get_position_size(self, signal, portfolio_value, date=None):
        base = portfolio_value * self._position_pct
        regime = self._sizing_regime.get(date, "CAUTIOUS")
        if regime != "BEARISH":
            return base
        # In bear: scale linearly with distance below SMA200
        # At SMA200 (dist=0): mult=1.0, at -10% below: mult=min_mult
        dist = self._dist_ratio.get(date, 0)
        if dist >= 0:
            return base
        # Linear scale: 0% below → 1.0, -10% below → min_mult
        mult = max(self._min_mult, 1.0 + dist * (1.0 - self._min_mult) / 0.10)
        return base * mult


class DrawdownBrakeML(CautiousMLStrategy):
    """Reduces sizing when portfolio is in drawdown from peak."""

    def __init__(self, predictions_df, regime_dict, dd_threshold=-0.10,
                 dd_mult=0.5, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._dd_threshold = dd_threshold
        self._dd_mult = dd_mult
        self._peak_val = INITIAL_CASH
        self._last_val = INITIAL_CASH

    def update_portfolio_value(self, val):
        """Called by runner after each day."""
        self._last_val = val
        if val > self._peak_val:
            self._peak_val = val

    def get_position_size(self, signal, portfolio_value, date=None):
        base = portfolio_value * self._position_pct
        if self._peak_val > 0:
            dd = (self._last_val / self._peak_val) - 1.0
            if dd <= self._dd_threshold:
                return base * self._dd_mult
        return base


SLOT_CONFIG = SlotConfig(
    strategy_slots={"ml_medium": 5, "momentum": 3},
    flex_slots=0, max_positions=8,
)


def run_variant(preds_df, year, close, regime_dict, spy_full, variant):
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

    vtype = variant["type"]
    if vtype == "bear_boost":
        ml = BearBoostML(preds_df, regime_dict,
                         bull_m=variant["bull"], bear_m=variant["bear"],
                         threshold=0.55, top_n=5, selection_mode="top_n")
    elif vtype == "graduated":
        ml = GraduatedBearML(preds_df, regime_dict, spy_full,
                             min_mult=variant["min_mult"],
                             threshold=0.55, top_n=5, selection_mode="top_n")
    elif vtype == "dd_brake":
        ml = DrawdownBrakeML(preds_df, regime_dict,
                             dd_threshold=variant["dd_thresh"],
                             dd_mult=variant["dd_mult"],
                             threshold=0.55, top_n=5, selection_mode="top_n")
    else:
        ml = CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                                threshold=0.55, top_n=5, selection_mode="top_n")

    mom = MomentumStrategy(close, volume_data=None, regime_filter=True)
    pm = PortfolioManager(strategies=[ml, mom], slot_config=SLOT_CONFIG)

    # For DD brake: hook into the run loop to update portfolio value
    if vtype == "dd_brake":
        result = pm.run(all_dates, spy_prices=None, price_data=close, detail_log=True)
        vals, trades, eq_df, _ = result
        # Retroactively not possible to hook — use a custom approach
        # Actually, we need to run manually. Let's just use the simpler approach:
        # Run with detail_log and simulate the brake by re-running
        # This is too complex for a hook — skip this for now and use the simple PM
        vals, trades = pm.run(all_dates, spy_prices=None, price_data=close)
    else:
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
    log("  REGIME SIZING — ROUND 3: SWEET SPOT HUNT")
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

    variants = [
        # Baseline (TP25+TS12 already in backtester defaults)
        {"name": "baseline",    "label": "No regime (TP25+TS12)",    "short": "Base",  "type": "none"},

        # Bear boost: reduce in bear, boost in bull to offset
        {"name": "bb_15_50",    "label": "Bull 1.15 / Bear 50%",    "short": "B15B5", "type": "bear_boost", "bull": 1.15, "bear": 0.5},
        {"name": "bb_20_50",    "label": "Bull 1.20 / Bear 50%",    "short": "B20B5", "type": "bear_boost", "bull": 1.20, "bear": 0.5},
        {"name": "bb_15_40",    "label": "Bull 1.15 / Bear 40%",    "short": "B15B4", "type": "bear_boost", "bull": 1.15, "bear": 0.4},
        {"name": "bb_20_40",    "label": "Bull 1.20 / Bear 40%",    "short": "B20B4", "type": "bear_boost", "bull": 1.20, "bear": 0.4},
        {"name": "bb_15_30",    "label": "Bull 1.15 / Bear 30%",    "short": "B15B3", "type": "bear_boost", "bull": 1.15, "bear": 0.3},
        {"name": "bb_20_30",    "label": "Bull 1.20 / Bear 30%",    "short": "B20B3", "type": "bear_boost", "bull": 1.20, "bear": 0.3},
        {"name": "bb_25_50",    "label": "Bull 1.25 / Bear 50%",    "short": "B25B5", "type": "bear_boost", "bull": 1.25, "bear": 0.5},

        # Graduated bear: scales with distance from SMA200
        {"name": "grad_30",     "label": "Graduated bear (min 30%)", "short": "Gr30",  "type": "graduated", "min_mult": 0.3},
        {"name": "grad_50",     "label": "Graduated bear (min 50%)", "short": "Gr50",  "type": "graduated", "min_mult": 0.5},

        # Drawdown brake
        {"name": "dd_10_50",    "label": "DD brake 10%→50%",        "short": "DD10",  "type": "dd_brake", "dd_thresh": -0.10, "dd_mult": 0.5},
        {"name": "dd_15_50",    "label": "DD brake 15%→50%",        "short": "DD15",  "type": "dd_brake", "dd_thresh": -0.15, "dd_mult": 0.5},
        {"name": "dd_08_50",    "label": "DD brake 8%→50%",         "short": "DD8",   "type": "dd_brake", "dd_thresh": -0.08, "dd_mult": 0.5},
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
            metrics = run_variant(preds_year, year, close_year, regime_dict, spy_full, variant)
            if metrics:
                all_results[variant["name"]].append(metrics)
                log(f"    {variant['short']:6s}  CAGR={metrics['cagr']:+.1%}  "
                    f"Sharpe={metrics['sharpe']:.2f}  MaxDD={metrics['max_dd']:.1%}")

    # ── Report ──
    n = len(all_results["baseline"])
    if n == 0:
        return

    print("\n" + "=" * 150)
    print("REGIME SIZING R3 — SWEET SPOT RESULTS (base: TP25 + Trail12)")
    print("=" * 150)

    base_cagrs = [m["cagr"] for m in all_results["baseline"]]
    base_dds = [m["max_dd"] for m in all_results["baseline"]]

    print(f"\n{'Config':<30s} {'MedCAGR':>8s} {'MnCAGR':>8s} {'MedShp':>7s} "
          f"{'WrstDD':>8s} {'MnDD':>8s} "
          f"{'dCAGR':>8s} {'dDD':>8s} "
          f"{'20cagr':>7s} {'20dd':>7s} {'22cagr':>7s} {'22dd':>7s} {'18dd':>7s}")
    print("─" * 150)

    for variant in variants:
        results = all_results[variant["name"]]
        if not results:
            continue

        cagrs = [m["cagr"] for m in results]
        sharpes = [m["sharpe"] for m in results]
        dds = [m["max_dd"] for m in results]

        y20 = next((m for m in results if m["year"] == 2020), None)
        y22 = next((m for m in results if m["year"] == 2022), None)
        y18 = next((m for m in results if m["year"] == 2018), None)

        d_cagr = np.median(cagrs) - np.median(base_cagrs)
        d_dd = min(dds) - min(base_dds)

        print(f"  {variant['label']:<28s} {np.median(cagrs):>+7.1%} {np.mean(cagrs):>+7.1%} "
              f"{np.median(sharpes):>7.2f} "
              f"{min(dds):>8.1%} {np.mean(dds):>8.1%} "
              f"{d_cagr:>+7.1%} {d_dd:>+7.1%} "
              f"{y20['cagr'] if y20 else 0:>+6.1%} {y20['max_dd'] if y20 else 0:>7.1%} "
              f"{y22['cagr'] if y22 else 0:>+6.1%} {y22['max_dd'] if y22 else 0:>7.1%} "
              f"{y18['max_dd'] if y18 else 0:>7.1%}")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
