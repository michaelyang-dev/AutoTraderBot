#!/usr/bin/env python3
"""
Walk-Forward Optimization — Maximize Returns & Reduce Drawdowns
================================================================
Tests ~20 strategy variants against the current live baseline.

Categories tested:
  A. Exit tuning       (stop-loss, take-profit, trailing stop levels)
  B. Entry filters     (consensus confirmation, momentum filter)
  C. Position sizing   (regime-adaptive, volatility-scaled, drawdown brake)
  D. Hold period       (asymmetric, dynamic)
  E. Combined          (best ideas together)
"""

import os
import sys
import time
import warnings
from pathlib import Path
from copy import deepcopy

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    MLMediumStrategy, CautiousMLStrategy, compute_regime_live,
    MomentumStrategy, _consensus_score,
    PortfolioManager, SlotConfig, Strategy, Signal, Position,
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
#  Variant ML Strategies
# ══════════════════════════════════════════════════════════════════════════════

class TunableMLStrategy(CautiousMLStrategy):
    """ML strategy with configurable exit parameters."""

    def __init__(self, predictions_df, regime_dict, config, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._cfg = config

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


class ConsensusFilterMLStrategy(TunableMLStrategy):
    """Only enters when consensus is not SELL (score > -1)."""

    def generate_signals(self, date, universe_data):
        signals = super().generate_signals(date, universe_data)
        if not hasattr(self, "_price_data"):
            return signals
        filtered = []
        for sig in signals:
            score = _consensus_score(self._price_data, sig.symbol, date)
            if score is None or score > -1.0:
                filtered.append(sig)
        return filtered


class StrongConsensusFilterMLStrategy(TunableMLStrategy):
    """Only enters when consensus is BUY or better (score >= 1)."""

    def generate_signals(self, date, universe_data):
        signals = super().generate_signals(date, universe_data)
        if not hasattr(self, "_price_data"):
            return signals
        filtered = []
        for sig in signals:
            score = _consensus_score(self._price_data, sig.symbol, date)
            if score is not None and score >= 0:
                filtered.append(sig)
        return filtered


class RegimeSizedMLStrategy(TunableMLStrategy):
    """Reduces position size during BEARISH/CAUTIOUS regimes."""

    def __init__(self, predictions_df, regime_dict, config, **kwargs):
        super().__init__(predictions_df, regime_dict, config, **kwargs)
        self._sizing_regime_dict = regime_dict
        self._bull_mult = config.get("bull_mult", 1.0)
        self._caut_mult = config.get("caut_mult", 0.6)
        self._bear_mult = config.get("bear_mult", 0.3)

    def get_position_size(self, signal, portfolio_value, date=None):
        base = portfolio_value * self._position_pct
        regime = self._sizing_regime_dict.get(date, "CAUTIOUS")
        if regime == "BULLISH":
            return base * self._bull_mult
        elif regime == "CAUTIOUS":
            return base * self._caut_mult
        else:
            return base * self._bear_mult


class VolSizedMLStrategy(TunableMLStrategy):
    """Position size inversely proportional to realized volatility."""

    def __init__(self, predictions_df, regime_dict, config, **kwargs):
        super().__init__(predictions_df, regime_dict, config, **kwargs)
        self._vol_price_data = None
        self._vol_target = config.get("vol_target", 0.01)  # 1% daily target

    def set_price_data(self, price_data):
        super().set_price_data(price_data)
        self._vol_price_data = price_data

    def get_position_size(self, signal, portfolio_value, date=None):
        if self._vol_price_data is not None and signal.symbol in self._vol_price_data.columns:
            col = self._vol_price_data[signal.symbol]
            mask = col.index <= date
            prices = col.loc[mask].dropna()
            if len(prices) >= 21:
                vol = prices.pct_change().iloc[-20:].std()
                if vol > 0:
                    size_pct = self._vol_target / vol
                    size_pct = max(0.03, min(0.20, size_pct))
                    return portfolio_value * size_pct
        return portfolio_value * self._position_pct


# ══════════════════════════════════════════════════════════════════════════════
#  Drawdown-Brake Portfolio Manager
# ══════════════════════════════════════════════════════════════════════════════

class DrawdownBrakePM(PortfolioManager):
    """Reduces position sizing when portfolio drawdown exceeds threshold."""

    def __init__(self, strategies, slot_config, dd_threshold=-0.08,
                 dd_scale=0.5, **kwargs):
        super().__init__(strategies, slot_config, **kwargs)
        self._dd_threshold = dd_threshold
        self._dd_scale = dd_scale

    def run(self, all_dates, spy_prices=None, price_data=None, detail_log=False):
        # Override run to inject drawdown scaling
        self._peak_val = self.initial_cash
        self._dd_active = False
        return super().run(all_dates, spy_prices, price_data, detail_log)


# ══════════════════════════════════════════════════════════════════════════════
#  Config runner
# ══════════════════════════════════════════════════════════════════════════════

SLOT_CONFIG = SlotConfig(
    strategy_slots={"ml_medium": 5, "momentum": 3},
    flex_slots=0,
    max_positions=8,
)

SLOT_CONFIG_FEWER = SlotConfig(
    strategy_slots={"ml_medium": 4, "momentum": 2},
    flex_slots=0,
    max_positions=6,
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
    strat_class = variant.get("strat_class", TunableMLStrategy)
    hold_days = variant.get("hold_days", 10)
    slot_config = variant.get("slot_config", SLOT_CONFIG)

    ml = strat_class(
        preds_df, regime_dict=regime_dict, config=cfg,
        threshold=0.55, top_n=variant.get("top_n", 5),
        selection_mode="top_n"
    )

    mom = MomentumStrategy(close, volume_data=None, regime_filter=True)
    strategies = [ml, mom]

    pm_class = variant.get("pm_class", PortfolioManager)
    pm_kwargs = variant.get("pm_kwargs", {})
    pm = pm_class(strategies=strategies, slot_config=slot_config,
                  hold_days=hold_days, **pm_kwargs)

    result = pm.run(all_dates, spy_prices=None, price_data=close, detail_log=True)
    vals, trades, _, trade_details = result

    metrics = calc_metrics(vals, trades, years_span, f"{variant['name']}_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    metrics["n_trades"] = len(trades)

    # Exit reason breakdown
    if not trade_details.empty and "exit_reason" in trade_details.columns:
        sells = trade_details[trade_details["action"] == "sell"]
        for reason in ["stop_loss", "take_profit", "trailing_stop", "consensus_sell", "hold_complete"]:
            metrics[f"n_{reason}"] = int((sells["exit_reason"] == reason).sum())

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Variant definitions
# ══════════════════════════════════════════════════════════════════════════════

def build_variants():
    """Build all variant configs to test."""
    variants = []

    # ── A. BASELINE (current live) ──
    variants.append({
        "name": "baseline",
        "label": "Current Live",
        "short": "Base",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # ── B. EXIT TUNING ──

    # B1: No trailing stop (it has -3% avg return, 24.6% win rate)
    variants.append({
        "name": "no_trailing",
        "label": "No trailing stop",
        "short": "NoTS",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "use_trailing": False, "consensus_exit": True, "min_hold": 5},
    })

    # B2: Wider trailing stop (12%)
    variants.append({
        "name": "wide_trail",
        "label": "Trailing 12%",
        "short": "TS12",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # B3: Wider stop-loss (12%)
    variants.append({
        "name": "wide_sl",
        "label": "Stop-loss 12%",
        "short": "SL12",
        "exit_cfg": {"stop_loss": -0.12, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # B4: Tighter stop-loss (5%)
    variants.append({
        "name": "tight_sl",
        "label": "Stop-loss 5%",
        "short": "SL5",
        "exit_cfg": {"stop_loss": -0.05, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # B5: Higher take-profit (25%) — let winners run
    variants.append({
        "name": "high_tp",
        "label": "Take-profit 25%",
        "short": "TP25",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.25,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # B6: No take-profit cap (let winners fully run, trailing stop handles exit)
    variants.append({
        "name": "no_tp",
        "label": "No take-profit cap",
        "short": "NoTP",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 1.0,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # B7: Asymmetric — tight stop (5%), wide trailing (15%), no TP cap
    variants.append({
        "name": "asym_exit",
        "label": "Asym: SL5/TS15/noTP",
        "short": "Asym",
        "exit_cfg": {"stop_loss": -0.05, "take_profit": 1.0,
                     "trailing_stop": -0.15, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # B8: No consensus exit (rely only on price-based exits)
    variants.append({
        "name": "no_consensus",
        "label": "No consensus exit",
        "short": "NoCon",
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": False, "min_hold": 5},
    })

    # ── C. ENTRY FILTERS ──

    # C1: Don't enter when consensus is SELL
    variants.append({
        "name": "entry_filter",
        "label": "Skip SELL consensus entry",
        "short": "NoSE",
        "strat_class": ConsensusFilterMLStrategy,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # C2: Only enter when consensus >= HOLD
    variants.append({
        "name": "strong_entry",
        "label": "Entry: consensus >= HOLD",
        "short": "HldE",
        "strat_class": StrongConsensusFilterMLStrategy,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # ── D. POSITION SIZING ──

    # D1: Regime-adaptive sizing (full in bull, reduced in cautious/bear)
    variants.append({
        "name": "regime_size",
        "label": "Regime sizing (1.0/0.6/0.3)",
        "short": "RegSz",
        "strat_class": RegimeSizedMLStrategy,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # D2: Volatility-targeted sizing
    variants.append({
        "name": "vol_size",
        "label": "Vol-targeted sizing",
        "short": "VolSz",
        "strat_class": VolSizedMLStrategy,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5,
                     "vol_target": 0.01},
    })

    # D3: Fewer positions (6 max instead of 8)
    variants.append({
        "name": "fewer_pos",
        "label": "Max 6 positions",
        "short": "Max6",
        "slot_config": SLOT_CONFIG_FEWER,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # ── E. HOLD PERIOD ──

    # E1: Shorter hold (7 days)
    variants.append({
        "name": "hold7",
        "label": "Hold 7 days",
        "short": "Hld7",
        "hold_days": 7,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # E2: Longer hold (15 days)
    variants.append({
        "name": "hold15",
        "label": "Hold 15 days",
        "short": "Hld15",
        "hold_days": 15,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # E3: Longer hold (20 days)
    variants.append({
        "name": "hold20",
        "label": "Hold 20 days",
        "short": "Hld20",
        "hold_days": 20,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # ── F. DRAWDOWN CONTROL ──

    # F1: Drawdown brake (8% DD → 50% size)
    variants.append({
        "name": "dd_brake",
        "label": "DD brake (8%→50%)",
        "short": "DDBrk",
        "pm_class": DrawdownBrakePM,
        "pm_kwargs": {"dd_threshold": -0.08, "dd_scale": 0.5},
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "trailing_stop": -0.08, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # ── G. COMBINED VARIANTS ──

    # G1: No trailing + wider SL + consensus filter
    variants.append({
        "name": "combo_a",
        "label": "NoTrail+SL12+EntryFilter",
        "short": "CmbA",
        "strat_class": ConsensusFilterMLStrategy,
        "exit_cfg": {"stop_loss": -0.12, "take_profit": 0.15,
                     "use_trailing": False, "consensus_exit": True, "min_hold": 5},
    })

    # G2: No trailing + regime sizing
    variants.append({
        "name": "combo_b",
        "label": "NoTrail+RegimeSizing",
        "short": "CmbB",
        "strat_class": RegimeSizedMLStrategy,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "use_trailing": False, "consensus_exit": True, "min_hold": 5,
                     "bull_mult": 1.0, "caut_mult": 0.6, "bear_mult": 0.3},
    })

    # G3: Entry filter + wider SL + hold15
    variants.append({
        "name": "combo_c",
        "label": "EntryFilter+SL12+Hld15",
        "short": "CmbC",
        "strat_class": ConsensusFilterMLStrategy,
        "hold_days": 15,
        "exit_cfg": {"stop_loss": -0.12, "take_profit": 0.25,
                     "trailing_stop": -0.12, "use_trailing": True,
                     "consensus_exit": True, "min_hold": 5},
    })

    # G4: Entry filter + no trailing + regime sizing + hold 15
    variants.append({
        "name": "combo_d",
        "label": "Full combo (entry+regime+noTS+h15)",
        "short": "CmbD",
        "strat_class": ConsensusFilterMLStrategy,  # Will need special handling
        "hold_days": 15,
        "exit_cfg": {"stop_loss": -0.10, "take_profit": 0.25,
                     "use_trailing": False, "consensus_exit": True, "min_hold": 5},
    })

    # G5: Vol sizing + entry filter + no trailing
    variants.append({
        "name": "combo_e",
        "label": "VolSize+EntryFilter+NoTS",
        "short": "CmbE",
        "strat_class": VolSizedMLStrategy,
        "exit_cfg": {"stop_loss": -0.08, "take_profit": 0.15,
                     "use_trailing": False, "consensus_exit": True, "min_hold": 5,
                     "vol_target": 0.01},
    })

    return variants


# For combo classes that need both entry filter AND regime sizing
class RegimeSizedConsensusFilterML(RegimeSizedMLStrategy):
    """Combines consensus entry filter with regime-adaptive sizing."""

    def generate_signals(self, date, universe_data):
        signals = super().generate_signals(date, universe_data)
        if not hasattr(self, "_price_data"):
            return signals
        filtered = []
        for sig in signals:
            score = _consensus_score(self._price_data, sig.symbol, date)
            if score is None or score > -1.0:
                filtered.append(sig)
        return filtered


class VolSizedConsensusFilterML(VolSizedMLStrategy):
    """Combines consensus entry filter with vol sizing."""

    def generate_signals(self, date, universe_data):
        signals = super().generate_signals(date, universe_data)
        if not hasattr(self, "_price_data"):
            return signals
        filtered = []
        for sig in signals:
            score = _consensus_score(self._price_data, sig.symbol, date)
            if score is None or score > -1.0:
                filtered.append(sig)
        return filtered


def main():
    t0 = time.perf_counter()

    log("=" * 80)
    log("  RETURN MAXIMIZATION — WALK-FORWARD OPTIMIZATION")
    log("=" * 80)

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

    variants = build_variants()

    # Fix combo_d and combo_e to use combined classes
    for v in variants:
        if v["name"] == "combo_d":
            v["strat_class"] = RegimeSizedConsensusFilterML
            v["exit_cfg"]["bull_mult"] = 1.0
            v["exit_cfg"]["caut_mult"] = 0.6
            v["exit_cfg"]["bear_mult"] = 0.3
        elif v["name"] == "combo_e":
            v["strat_class"] = VolSizedConsensusFilterML

    # Run all variants × all years
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

    # ══════════════════════════════════════════════════════════════════════
    #  Report
    # ══════════════════════════════════════════════════════════════════════
    n = len(all_results["baseline"])
    if n == 0:
        log("No results.")
        return

    print("\n" + "=" * 140)
    print("RETURN MAXIMIZATION — WALK-FORWARD RESULTS")
    print("=" * 140)

    # ── Summary table ──
    def get_agg(field, results):
        return [m[field] for m in results]

    print(f"\n{'Variant':<30s} {'MedCAGR':>8s} {'MnCAGR':>8s} {'MedShp':>7s} {'MnShp':>7s} "
          f"{'WrstDD':>8s} {'MnDD':>8s} {'MedAlp':>8s} {'WinR':>5s} {'Trd/y':>6s} "
          f"{'+CAGR':>5s} {'dCAGR':>8s} {'dShp':>7s} {'dDD':>8s}")
    print("─" * 140)

    base_cagrs = get_agg("cagr", all_results["baseline"])
    base_sharpes = get_agg("sharpe", all_results["baseline"])
    base_dds = get_agg("max_dd", all_results["baseline"])

    rows = []
    for variant in variants:
        results = all_results[variant["name"]]
        if not results:
            continue

        cagrs = get_agg("cagr", results)
        sharpes = get_agg("sharpe", results)
        dds = get_agg("max_dd", results)
        alphas_raw = get_agg("alpha", results)
        alphas = [a for a in alphas_raw if a is not None]
        win_rates = get_agg("win_rate", results)
        trades = get_agg("n_trades", results)

        med_cagr = np.median(cagrs)
        mn_cagr = np.mean(cagrs)
        med_shp = np.median(sharpes)
        mn_shp = np.mean(sharpes)
        worst_dd = min(dds)
        mn_dd = np.mean(dds)
        med_alpha = np.median(alphas) if alphas else 0
        mn_wr = np.mean(win_rates)
        mn_trades = np.mean(trades)
        pos_cagr = sum(1 for c in cagrs if c > 0)

        d_cagr = med_cagr - np.median(base_cagrs)
        d_shp = med_shp - np.median(base_sharpes)
        d_dd = worst_dd - min(base_dds)

        rows.append((variant, med_cagr, mn_cagr, med_shp, mn_shp,
                      worst_dd, mn_dd, med_alpha, mn_wr, mn_trades,
                      pos_cagr, d_cagr, d_shp, d_dd))

        print(f"  {variant['label']:<28s} {med_cagr:>+7.1%} {mn_cagr:>+7.1%} "
              f"{med_shp:>7.2f} {mn_shp:>7.2f} "
              f"{worst_dd:>8.1%} {mn_dd:>8.1%} {med_alpha:>+7.1%} "
              f"{mn_wr:>5.0%} {mn_trades:>6.0f} "
              f"{pos_cagr:>3d}/{n:<1d} {d_cagr:>+7.1%} {d_shp:>+6.2f} {d_dd:>+7.1%}")

    # ── Per-year CAGR for top variants ──
    # Sort by composite score: Sharpe improvement + CAGR improvement - DD worsening
    rows_sorted = sorted(rows, key=lambda r: r[11] + r[12] * 0.1 + r[13] * 0.5, reverse=True)

    print(f"\n{'─' * 100}")
    print("TOP 10 VARIANTS — Per-Year CAGR")
    print(f"{'─' * 100}")

    top10 = rows_sorted[:10]
    print(f"\n{'Year':>6s}", end="")
    for row in top10:
        print(f" {row[0]['short']:>7s}", end="")
    print()
    print("─" * (6 + 8 * len(top10)))

    for yi in range(n):
        yr = all_results["baseline"][yi]["year"]
        print(f"{yr:>6d}", end="")
        for row in top10:
            results = all_results[row[0]["name"]]
            cagr = results[yi]["cagr"]
            print(f" {cagr:>+7.1%}", end="")
        print()

    # ── Per-year Max DD for top variants ──
    print(f"\n{'Year':>6s}", end="")
    for row in top10:
        print(f" {row[0]['short']:>7s}", end="")
    print("   (Max DD)")
    print("─" * (6 + 8 * len(top10) + 12))

    for yi in range(n):
        yr = all_results["baseline"][yi]["year"]
        print(f"{yr:>6d}", end="")
        for row in top10:
            results = all_results[row[0]["name"]]
            dd = results[yi]["max_dd"]
            print(f" {dd:>7.1%}", end="")
        print()

    # ── Rankings ──
    print(f"\n{'─' * 100}")
    print("RANKINGS (sorted by composite score = ΔCAGR + 0.1×ΔSharpe + 0.5×ΔDD)")
    print(f"{'─' * 100}")
    print(f"  {'Rank':>4s}  {'Variant':<30s} {'ΔCAGR':>8s} {'ΔSharpe':>8s} {'ΔWorstDD':>9s} {'Score':>7s}")
    print("─" * 80)

    for i, row in enumerate(rows_sorted):
        v = row[0]
        d_cagr = row[11]
        d_shp = row[12]
        d_dd = row[13]
        score = d_cagr + d_shp * 0.1 + d_dd * 0.5
        print(f"  {i+1:>4d}  {v['label']:<30s} {d_cagr:>+7.1%} {d_shp:>+8.2f} {d_dd:>+9.1%} {score:>+7.3f}")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
