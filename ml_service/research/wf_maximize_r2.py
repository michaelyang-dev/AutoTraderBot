#!/usr/bin/env python3
"""
Walk-Forward Optimization — Round 2
=====================================
Combines Round 1 winners into refined variants.

Round 1 winners:
  - Take-profit 25% (best CAGR boost)
  - No TP cap + trailing stop (best Sharpe)
  - Regime sizing 1.0/0.6/0.3 (best DD reduction: -23% vs -42.8%)
  - Wider trailing stop 12% (consistently better than 8%)
  - Trailing stop removal also good

Round 2 tests combinations plus refined regime sizing parameters.
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
    CautiousMLStrategy, compute_regime_live, _consensus_score,
    MomentumStrategy,
    PortfolioManager, SlotConfig, Signal,
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
#  Strategy variants
# ══════════════════════════════════════════════════════════════════════════════

class TunableMLStrategy(CautiousMLStrategy):
    """ML strategy with configurable exit parameters and optional regime sizing."""

    def __init__(self, predictions_df, regime_dict, config, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._cfg = config
        self._sizing_regime_dict = regime_dict

    def check_exit(self, position, current_data):
        cfg = self._cfg
        sl = cfg.get("stop_loss", -0.08)
        tp = cfg.get("take_profit", 0.15)
        ts = cfg.get("trailing_stop", -0.08)
        min_hold = cfg.get("min_hold", 5)
        use_trailing = cfg.get("use_trailing", True)
        consensus_exit = cfg.get("consensus_exit", True)

        if position.entry_price > 0:
            px = current_data["prices"].get(position.symbol)
            if px is not None and not np.isnan(px):
                ret = (px / position.entry_price) - 1.0
                if ret <= sl:
                    return True, "stop_loss"
                if ret >= tp:
                    return True, "take_profit"
                if use_trailing and position.peak_price > 0:
                    drop = (px / position.peak_price) - 1.0
                    if drop <= ts:
                        return True, "trailing_stop"

        days_held = current_data["idx"] - position.entry_idx
        if consensus_exit and days_held >= min_hold and hasattr(self, "_price_data"):
            score = _consensus_score(self._price_data, position.symbol,
                                     current_data["date"])
            if score is not None and score <= -1.0:
                return True, "consensus_sell"

        if current_data["idx"] >= position.exit_idx:
            return True, "hold_complete"
        return False, ""

    def get_position_size(self, signal, portfolio_value, date=None):
        base = portfolio_value * self._position_pct
        # Regime sizing if configured
        bull_mult = self._cfg.get("bull_mult")
        if bull_mult is not None:
            regime = self._sizing_regime_dict.get(date, "CAUTIOUS")
            if regime == "BULLISH":
                return base * bull_mult
            elif regime == "CAUTIOUS":
                return base * self._cfg.get("caut_mult", 0.6)
            else:
                return base * self._cfg.get("bear_mult", 0.3)
        return base


# ══════════════════════════════════════════════════════════════════════════════

SLOT_CONFIG = SlotConfig(
    strategy_slots={"ml_medium": 5, "momentum": 3},
    flex_slots=0,
    max_positions=8,
)


def run_variant(preds_df, year, close, regime_dict, variant):
    """Run one year with a specific variant config."""
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

    cfg = variant.get("exit_cfg", {})
    hold_days = variant.get("hold_days", 10)

    ml = TunableMLStrategy(
        preds_df, regime_dict=regime_dict, config=cfg,
        threshold=0.55, top_n=5, selection_mode="top_n"
    )
    mom = MomentumStrategy(close, volume_data=None, regime_filter=True)

    pm = PortfolioManager(strategies=[ml, mom], slot_config=SLOT_CONFIG,
                          hold_days=hold_days)
    result = pm.run(all_dates, spy_prices=None, price_data=close, detail_log=True)
    vals, trades, _, trade_details = result

    metrics = calc_metrics(vals, trades, years_span, f"{variant['name']}_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    metrics["n_trades"] = len(trades)
    return metrics


def build_variants():
    """Round 2 variants — combinations of Round 1 winners."""
    v = []

    # ── BASELINE ──
    v.append({
        "name": "baseline",
        "label": "Current Live (baseline)",
        "short": "Base",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # ── ROUND 1 BEST SINGLES ──
    v.append({
        "name": "tp25",
        "label": "TP 25%",
        "short": "TP25",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    v.append({
        "name": "regime",
        "label": "Regime sizing (1.0/0.6/0.3)",
        "short": "Rgm",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # ── ROUND 2: COMBINATIONS ──

    # C1: TP25 + wider trailing 12%
    v.append({
        "name": "tp25_ts12",
        "label": "TP25 + Trail 12%",
        "short": "T25T12",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # C2: TP25 + no trailing
    v.append({
        "name": "tp25_nots",
        "label": "TP25 + No trailing",
        "short": "T25NT",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "use_trailing": False,
                     "consensus_exit": True, "min_hold": 5},
    })

    # C3: TP25 + regime sizing
    v.append({
        "name": "tp25_regime",
        "label": "TP25 + Regime sizing",
        "short": "T25Rg",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # C4: TP25 + regime sizing + wider trailing
    v.append({
        "name": "tp25_regime_ts12",
        "label": "TP25 + Regime + Trail12",
        "short": "T25RT",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # C5: TP25 + regime sizing + no trailing
    v.append({
        "name": "tp25_regime_nots",
        "label": "TP25 + Regime + No trail",
        "short": "T25RN",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "use_trailing": False,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # C6: No TP cap + regime sizing + wider trailing
    v.append({
        "name": "notp_regime_ts12",
        "label": "NoTP + Regime + Trail12",
        "short": "NTRT",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 1.0,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # ── REFINED REGIME PARAMETERS ──

    # R1: Aggressive regime (1.2/0.5/0.2)
    v.append({
        "name": "regime_agg",
        "label": "Regime (1.2/0.5/0.2)",
        "short": "RgAgg",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.2, "caut_mult": 0.5, "bear_mult": 0.2},
    })

    # R2: Moderate regime (1.0/0.7/0.4)
    v.append({
        "name": "regime_mod",
        "label": "Regime (1.0/0.7/0.4)",
        "short": "RgMod",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.7, "bear_mult": 0.4},
    })

    # R3: Zero bear (1.0/0.5/0.0) — no ML trades in bear markets
    v.append({
        "name": "regime_zero_bear",
        "label": "Regime (1.0/0.5/0.0)",
        "short": "RgZB",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.5, "bear_mult": 0.0},
    })

    # ── TP REFINEMENTS ──

    # T1: TP 20%
    v.append({
        "name": "tp20_ts12_regime",
        "label": "TP20 + Trail12 + Regime",
        "short": "T20RT",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.20,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # T2: TP 30%
    v.append({
        "name": "tp30_ts12_regime",
        "label": "TP30 + Trail12 + Regime",
        "short": "T30RT",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.30,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # ── SL REFINEMENTS ──

    # S1: SL 10% (slightly wider) + TP25 + regime
    v.append({
        "name": "sl10_tp25_regime",
        "label": "SL10 + TP25 + Regime",
        "short": "S10TR",
        "exit_cfg": {"stop_loss": -0.10, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # S2: SL 6% + TP25 + regime
    v.append({
        "name": "sl6_tp25_regime",
        "label": "SL6 + TP25 + Regime",
        "short": "S6TR",
        "exit_cfg": {"stop_loss": -0.06, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # ── HOLD PERIOD + COMBOS ──

    # H1: Hold 12 + TP25 + regime + trail12
    v.append({
        "name": "h12_tp25_regime",
        "label": "Hold12 + TP25 + Regime",
        "short": "H12TR",
        "hold_days": 12,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    return v


def main():
    t0 = time.perf_counter()

    log("=" * 80)
    log("  ROUND 2 — COMBINING WINNERS")
    log("=" * 80)

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

    variants = build_variants()
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
                log(f"    {variant['short']:8s}  CAGR={metrics['cagr']:+.1%}  "
                    f"Sharpe={metrics['sharpe']:.2f}  MaxDD={metrics['max_dd']:.1%}")

    # ── Report ──
    n = len(all_results["baseline"])
    if n == 0:
        log("No results.")
        return

    print("\n" + "=" * 150)
    print("ROUND 2 — COMBINING WINNERS")
    print("=" * 150)

    def get_agg(field, results):
        return [m[field] for m in results]

    base_cagrs = get_agg("cagr", all_results["baseline"])
    base_sharpes = get_agg("sharpe", all_results["baseline"])
    base_dds = get_agg("max_dd", all_results["baseline"])

    print(f"\n{'Variant':<30s} {'MedCAGR':>8s} {'MnCAGR':>8s} {'MedShp':>7s} {'MnShp':>7s} "
          f"{'WrstDD':>8s} {'MnDD':>8s} {'MedAlp':>8s} {'WinR':>5s} {'+CAGR':>5s} "
          f"{'dCAGR':>8s} {'dShp':>7s} {'dDD':>8s}")
    print("─" * 150)

    rows = []
    for variant in variants:
        results = all_results[variant["name"]]
        if not results:
            continue

        cagrs = get_agg("cagr", results)
        sharpes = get_agg("sharpe", results)
        dds = get_agg("max_dd", results)
        alphas = [a for a in get_agg("alpha", results) if a is not None]
        win_rates = get_agg("win_rate", results)

        med_cagr = np.median(cagrs)
        mn_cagr = np.mean(cagrs)
        med_shp = np.median(sharpes)
        mn_shp = np.mean(sharpes)
        worst_dd = min(dds)
        mn_dd = np.mean(dds)
        med_alpha = np.median(alphas) if alphas else 0
        mn_wr = np.mean(win_rates)
        pos_cagr = sum(1 for c in cagrs if c > 0)

        d_cagr = med_cagr - np.median(base_cagrs)
        d_shp = med_shp - np.median(base_sharpes)
        d_dd = worst_dd - min(base_dds)

        rows.append((variant, med_cagr, mn_cagr, med_shp, mn_shp,
                      worst_dd, mn_dd, med_alpha, mn_wr,
                      pos_cagr, d_cagr, d_shp, d_dd, cagrs, sharpes, dds))

        print(f"  {variant['label']:<28s} {med_cagr:>+7.1%} {mn_cagr:>+7.1%} "
              f"{med_shp:>7.2f} {mn_shp:>7.2f} "
              f"{worst_dd:>8.1%} {mn_dd:>8.1%} {med_alpha:>+7.1%} "
              f"{mn_wr:>5.0%} {pos_cagr:>3d}/{n:<1d} "
              f"{d_cagr:>+7.1%} {d_shp:>+6.02f} {d_dd:>+7.1%}")

    # Per-year CAGR table
    print(f"\n{'─' * 120}")
    print("PER-YEAR CAGR")
    print(f"{'Year':>6s}", end="")
    for row in rows:
        print(f" {row[0]['short']:>7s}", end="")
    print()
    print("─" * (6 + 8 * len(rows)))

    for yi in range(n):
        yr = all_results["baseline"][yi]["year"]
        print(f"{yr:>6d}", end="")
        for row in rows:
            c = row[13][yi]
            print(f" {c:>+7.1%}", end="")
        print()

    # Per-year MaxDD
    print(f"\n{'─' * 120}")
    print("PER-YEAR MAX DRAWDOWN")
    print(f"{'Year':>6s}", end="")
    for row in rows:
        print(f" {row[0]['short']:>7s}", end="")
    print()
    print("─" * (6 + 8 * len(rows)))

    for yi in range(n):
        yr = all_results["baseline"][yi]["year"]
        print(f"{yr:>6d}", end="")
        for row in rows:
            dd = row[15][yi]
            print(f" {dd:>7.1%}", end="")
        print()

    # Ranking
    rows_sorted = sorted(rows, key=lambda r: r[10] + r[11] * 0.1 + r[12] * 0.5, reverse=True)

    print(f"\n{'─' * 100}")
    print("RANKING (composite = dCAGR + 0.1*dSharpe + 0.5*dDD)")
    print(f"{'─' * 100}")
    print(f"  {'Rank':>4s}  {'Variant':<30s} {'dCAGR':>8s} {'dSharpe':>8s} {'dWorstDD':>9s} {'Score':>7s}")
    print("─" * 80)

    for i, row in enumerate(rows_sorted):
        v = row[0]
        d_cagr, d_shp, d_dd = row[10], row[11], row[12]
        score = d_cagr + d_shp * 0.1 + d_dd * 0.5
        print(f"  {i+1:>4d}  {v['label']:<30s} {d_cagr:>+7.1%} {d_shp:>+8.2f} {d_dd:>+9.1%} {score:>+7.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
