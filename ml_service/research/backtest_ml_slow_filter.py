"""
ML Slow as Filter/Confirmation Layer
======================================
Tests ML Slow NOT as an independent strategy but as a modifier for ML Medium:

  Config 1: Path B baseline (no filter)
  Config 2: Veto filter — ML Medium buys vetoed when ML Slow disagrees
  Config 3: Conviction booster — size up when ML Slow agrees, no vetoes
  Config 4: Regime filter — halt ML Medium when avg ML Slow prob is low

Run with:
    python3 backtest_ml_slow_filter.py
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from unified_backtester import (
    INITIAL_CASH, HOLD_DAYS, POSITION_PCT,
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    SlotConfig, SLOT_ML_MOM_MR,
)
from strategy_base import Strategy, Signal
from backtest_utils import load_predictions, calc_metrics, calc_alpha_beta
from diagnose_combined import instrumented_run

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"


# ══════════════════════════════════════════════════════════════════════════════
#  Wrapper strategies that use ML Slow as a filter on ML Medium
# ══════════════════════════════════════════════════════════════════════════════

class MLMediumVetoFiltered(Strategy):
    """
    ML Medium strategy that vetoes buys when ML Slow disagrees.
    Only generates a signal if ML Slow prob >= slow_threshold for the same
    stock on the same day.
    """

    def __init__(self, med_predictions, slow_predictions, threshold=0.55,
                 slow_threshold=0.45, position_pct=POSITION_PCT):
        self._threshold = threshold
        self._position_pct = position_pct
        self._signals_by_date = {}
        self._slow_lookup = {}
        self._build_lookup(med_predictions)
        self._build_slow_lookup(slow_predictions, slow_threshold)

    @property
    def name(self):
        return "ml_medium"

    def _build_lookup(self, df):
        for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
            g = grp.sort_values("prob", ascending=False)
            self._signals_by_date[date] = [
                (row.symbol, row.prob, row.fwd_ret)
                for row in g.itertuples(index=False)
            ]

    def _build_slow_lookup(self, df, slow_threshold):
        """Build set of (date, symbol) pairs where ML Slow agrees."""
        for date, grp in df[["date", "symbol", "prob"]].groupby("date"):
            approved = set()
            for row in grp.itertuples(index=False):
                if row.prob >= slow_threshold:
                    approved.add(row.symbol)
            self._slow_lookup[date] = approved

    def generate_signals(self, date, universe_data):
        raw = self._signals_by_date.get(date, [])
        approved = self._slow_lookup.get(date, set())
        return [
            Signal(symbol=sym, confidence=prob,
                   strategy_name=self.name, fwd_ret=fwd_ret)
            for sym, prob, fwd_ret in raw
            if prob > self._threshold and sym in approved
        ]

    def check_exit(self, position, current_data):
        if current_data["idx"] >= position.exit_idx:
            return True, "hold_complete"
        return False, ""

    def get_position_size(self, signal, portfolio_value):
        ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
        return portfolio_value * self._position_pct * ml_mult


class MLMediumConvictionBoosted(Strategy):
    """
    ML Medium strategy with conviction-boosted sizing when ML Slow agrees.
    No vetoes — all ML Medium signals pass. But size is 1.25x when ML Slow
    also predicts positive.
    """

    def __init__(self, med_predictions, slow_predictions, threshold=0.55,
                 slow_threshold=0.45, boost=1.25, position_pct=POSITION_PCT):
        self._threshold = threshold
        self._position_pct = position_pct
        self._boost = boost
        self._signals_by_date = {}
        self._slow_approved = {}
        self._build_lookup(med_predictions)
        self._build_slow_lookup(slow_predictions, slow_threshold)

    @property
    def name(self):
        return "ml_medium"

    def _build_lookup(self, df):
        for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
            g = grp.sort_values("prob", ascending=False)
            self._signals_by_date[date] = [
                (row.symbol, row.prob, row.fwd_ret)
                for row in g.itertuples(index=False)
            ]

    def _build_slow_lookup(self, df, slow_threshold):
        for date, grp in df[["date", "symbol", "prob"]].groupby("date"):
            approved = set()
            for row in grp.itertuples(index=False):
                if row.prob >= slow_threshold:
                    approved.add(row.symbol)
            self._slow_approved[date] = approved

    def generate_signals(self, date, universe_data):
        raw = self._signals_by_date.get(date, [])
        approved = self._slow_approved.get(date, set())
        signals = []
        for sym, prob, fwd_ret in raw:
            if prob > self._threshold:
                # Tag signal with boost info via confidence adjustment
                # We encode boost in a small confidence bump that get_position_size reads
                boosted = sym in approved
                sig = Signal(symbol=sym, confidence=prob,
                             strategy_name=self.name, fwd_ret=fwd_ret)
                # Store boost flag in a way get_position_size can detect
                sig._boosted = boosted
                signals.append(sig)
        return signals

    def check_exit(self, position, current_data):
        if current_data["idx"] >= position.exit_idx:
            return True, "hold_complete"
        return False, ""

    def get_position_size(self, signal, portfolio_value):
        ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
        base = portfolio_value * self._position_pct * ml_mult
        if getattr(signal, '_boosted', False):
            return base * self._boost
        return base


class MLMediumRegimeFiltered(Strategy):
    """
    ML Medium strategy that halts buys when ML Slow's daily average
    prediction is below a threshold (bearish regime signal).
    """

    def __init__(self, med_predictions, slow_predictions, threshold=0.55,
                 regime_threshold=0.40, position_pct=POSITION_PCT):
        self._threshold = threshold
        self._position_pct = position_pct
        self._signals_by_date = {}
        self._regime_pass = {}
        self._daily_avg = {}
        self._build_lookup(med_predictions)
        self._build_regime_lookup(slow_predictions, regime_threshold)

    @property
    def name(self):
        return "ml_medium"

    def _build_lookup(self, df):
        for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
            g = grp.sort_values("prob", ascending=False)
            self._signals_by_date[date] = [
                (row.symbol, row.prob, row.fwd_ret)
                for row in g.itertuples(index=False)
            ]

    def _build_regime_lookup(self, df, regime_threshold):
        """Compute daily average ML Slow prob across all stocks."""
        for date, grp in df[["date", "symbol", "prob"]].groupby("date"):
            avg_prob = grp["prob"].mean()
            self._daily_avg[date] = avg_prob
            self._regime_pass[date] = avg_prob >= regime_threshold

    def generate_signals(self, date, universe_data):
        # If regime filter blocks, return no signals
        if not self._regime_pass.get(date, True):
            return []
        raw = self._signals_by_date.get(date, [])
        return [
            Signal(symbol=sym, confidence=prob,
                   strategy_name=self.name, fwd_ret=fwd_ret)
            for sym, prob, fwd_ret in raw
            if prob > self._threshold
        ]

    def check_exit(self, position, current_data):
        if current_data["idx"] >= position.exit_idx:
            return True, "hold_complete"
        return False, ""

    def get_position_size(self, signal, portfolio_value):
        ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
        return portfolio_value * self._position_pct * ml_mult


def fetch_ohlcv(symbols, start, end):
    all_syms = list(set(["SPY"] + symbols))
    raw = yf.download(all_syms, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume = None
    if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
        volume = raw["Volume"]
        volume.index = pd.to_datetime(volume.index).tz_localize(None)
    return close, volume


def main():
    t0 = time.perf_counter()
    print("=" * 90)
    print("  ML SLOW AS FILTER/CONFIRMATION LAYER")
    print("  Testing veto, conviction boost, and regime filter modes")
    print("=" * 90)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading predictions ...")
    df_med = load_predictions()

    pred_file = DATA_DIR / "predictions_slow.parquet"
    df_slow = pd.read_parquet(pred_file)
    df_slow["date"] = pd.to_datetime(df_slow["date"])
    df_slow = df_slow.sort_values(["date", "symbol"])
    print(f"  ML Slow: {len(df_slow):,} rows")

    common_dates = sorted(set(df_med["date"].unique()) & set(df_slow["date"].unique()))
    all_dates = common_dates
    universe_syms = sorted(set(df_med["symbol"].unique()) | set(df_slow["symbol"].unique()))
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    print(f"  {len(all_dates)} days | {years:.1f} years")

    # ── Fetch prices ─────────────────────────────────────────────────────
    print("\n  Fetching price & volume data ...")
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    close, volume = fetch_ohlcv(universe_syms, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")

    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close_aligned["SPY"].to_dict()

    # Non-ML strategies (shared across all configs)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ── Helper to run a config ───────────────────────────────────────────
    def run_config(label, ml_strat):
        vals, trades, diag = instrumented_run(
            [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
            close_aligned, hold_days=HOLD_DAYS, label=label)
        m = calc_metrics(vals, trades, years, label)
        a, _ = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
        m["alpha"] = a
        m["avg_pos"] = np.mean(diag["daily_pos_count"])
        # Count ML Medium trades specifically
        ml_trades = sum(1 for r in diag.get("trade_records", [])
                        if r.get("strategy") == "ml_medium")
        m["ml_trades"] = ml_trades
        return m, diag

    # ══════════════════════════════════════════════════════════════════════
    #  CONFIG 1: Path B baseline
    # ══════════════════════════════════════════════════════════════════════
    print("\n  [1] Running PATH B BASELINE ...")
    ml_baseline = MLMediumStrategy(df_med, threshold=0.55)
    m_base, d_base = run_config("Path B", ml_baseline)
    ml_base_trades = sum(1 for r in d_base.get("trade_records", [])
                         if r.get("strategy") == "ml_medium")
    m_base["ml_trades"] = ml_base_trades
    print(f"      CAGR {m_base['cagr']:+.2%} | Sharpe {m_base['sharpe']:.3f} | "
          f"Trades {m_base['n_trades']} | ML trades {ml_base_trades}")

    # ══════════════════════════════════════════════════════════════════════
    #  CONFIG 2: Veto filter
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*90}")
    print("  CONFIG 2: VETO FILTER — ML Medium vetoed when ML Slow disagrees")
    print(f"{'─'*90}")

    veto_thresholds = [0.40, 0.45, 0.50]
    veto_results = {}
    for st in veto_thresholds:
        label = f"Veto @{st:.2f}"
        print(f"\n  [2] {label} ...")
        ml_veto = MLMediumVetoFiltered(df_med, df_slow, threshold=0.55, slow_threshold=st)
        m, d = run_config(label, ml_veto)
        veto_results[st] = m

        # Count how many days the veto blocked all ML signals
        gen = d["signals_generated"].get("ml_medium", 0)
        acc = d["signals_accepted"].get("ml_medium", 0)
        print(f"      CAGR {m['cagr']:+.2%} | Sharpe {m['sharpe']:.3f} | "
              f"Trades {m['n_trades']} | ML signals {gen} → {acc} accepted")

    # ══════════════════════════════════════════════════════════════════════
    #  CONFIG 3: Conviction booster
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*90}")
    print("  CONFIG 3: CONVICTION BOOSTER — 1.25x size when ML Slow agrees")
    print(f"{'─'*90}")

    boost_thresholds = [0.40, 0.45, 0.50]
    boost_results = {}
    for st in boost_thresholds:
        label = f"Boost @{st:.2f}"
        print(f"\n  [3] {label} ...")
        ml_boost = MLMediumConvictionBoosted(df_med, df_slow, threshold=0.55,
                                              slow_threshold=st, boost=1.25)
        m, d = run_config(label, ml_boost)
        boost_results[st] = m
        print(f"      CAGR {m['cagr']:+.2%} | Sharpe {m['sharpe']:.3f} | "
              f"Trades {m['n_trades']} | ML trades {m['ml_trades']}")

    # ══════════════════════════════════════════════════════════════════════
    #  CONFIG 4: Regime filter
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*90}")
    print("  CONFIG 4: REGIME FILTER — halt ML Medium when avg ML Slow < threshold")
    print(f"{'─'*90}")

    regime_thresholds = [0.35, 0.40, 0.45]
    regime_results = {}

    # First show the daily avg distribution
    daily_avgs = df_slow.groupby("date")["prob"].mean()
    print(f"\n  ML Slow daily avg prob distribution:")
    print(f"    min={daily_avgs.min():.4f}  25%={daily_avgs.quantile(0.25):.4f}  "
          f"median={daily_avgs.median():.4f}  75%={daily_avgs.quantile(0.75):.4f}  "
          f"max={daily_avgs.max():.4f}")
    for rt in regime_thresholds:
        pct_blocked = (daily_avgs < rt).mean() * 100
        print(f"    Days blocked at {rt:.2f}: {pct_blocked:.1f}%")

    for rt in regime_thresholds:
        label = f"Regime @{rt:.2f}"
        print(f"\n  [4] {label} ...")
        ml_regime = MLMediumRegimeFiltered(df_med, df_slow, threshold=0.55,
                                            regime_threshold=rt)
        m, d = run_config(label, ml_regime)
        regime_results[rt] = m

        gen = d["signals_generated"].get("ml_medium", 0)
        acc = d["signals_accepted"].get("ml_medium", 0)
        print(f"      CAGR {m['cagr']:+.2%} | Sharpe {m['sharpe']:.3f} | "
              f"Trades {m['n_trades']} | ML signals {gen} → {acc} accepted")

    # ══════════════════════════════════════════════════════════════════════
    #  COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  FULL COMPARISON TABLE")
    print(f"{'='*90}")

    rows = []
    rows.append(("Path B (baseline)", m_base))
    for st in veto_thresholds:
        rows.append((f"Veto @{st:.2f}", veto_results[st]))
    for st in boost_thresholds:
        rows.append((f"Boost @{st:.2f}", boost_results[st]))
    for rt in regime_thresholds:
        rows.append((f"Regime @{rt:.2f}", regime_results[rt]))

    hdr = (f"  {'Config':<22} {'CAGR':>7} {'ΔCAGR':>7} {'Sharpe':>7} {'ΔSharp':>7} "
           f"{'MaxDD':>7} {'ΔDD':>6} {'WinR':>6} {'ΔWR':>6} {'Trades':>7} {'ΔTrd':>6}")
    print(hdr)
    print(f"  {'─'*22} " + " ".join(["─"*7]*4 + ["─"*7, "─"*6, "─"*6, "─"*6, "─"*7, "─"*6]))

    for label, m in rows:
        dcagr = m["cagr"] - m_base["cagr"]
        dsharpe = m["sharpe"] - m_base["sharpe"]
        ddd = m["max_dd"] - m_base["max_dd"]
        dwr = (m["win_rate"] - m_base["win_rate"]) if not np.isnan(m["win_rate"]) else 0
        dtrd = m["n_trades"] - m_base["n_trades"]

        marker = ""
        if label != "Path B (baseline)":
            # Check criteria
            sharpe_pass = dsharpe >= 0.05
            cagr_pass = dcagr >= -0.01
            dd_pass = ddd >= -0.02
            if sharpe_pass and cagr_pass and dd_pass:
                marker = " ✓"

        print(f"  {label:<22} {m['cagr']:>+6.2%} {dcagr:>+6.2%} {m['sharpe']:>7.3f} "
              f"{dsharpe:>+6.3f} {m['max_dd']:>6.1%} {ddd:>+5.1%} "
              f"{m['win_rate']:>5.1%} {dwr:>+5.1%} {m['n_trades']:>7} {dtrd:>+6}{marker}")

    # ══════════════════════════════════════════════════════════════════════
    #  GATE CHECKS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  DEPLOYMENT CRITERIA CHECK")
    print(f"  Sharpe improvement >= +0.05 | CAGR drop <= -1% | DD worsening <= -2pp")
    print(f"{'='*90}")

    any_pass = False
    best_config = None
    best_sharpe_delta = -999

    all_configs = []
    for st in veto_thresholds:
        all_configs.append((f"Veto @{st:.2f}", veto_results[st]))
    for st in boost_thresholds:
        all_configs.append((f"Boost @{st:.2f}", boost_results[st]))
    for rt in regime_thresholds:
        all_configs.append((f"Regime @{rt:.2f}", regime_results[rt]))

    print(f"\n  {'Config':<22} {'ΔSharpe':>8} {'≥+0.05':>8} {'ΔCAGR':>8} {'≥-1%':>6} "
          f"{'ΔDD':>7} {'≥-2pp':>7} {'Result':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*8}")

    for label, m in all_configs:
        dsharpe = m["sharpe"] - m_base["sharpe"]
        dcagr = m["cagr"] - m_base["cagr"]
        ddd = m["max_dd"] - m_base["max_dd"]

        g1 = dsharpe >= 0.05
        g2 = dcagr >= -0.01
        g3 = ddd >= -0.02

        g1s = "PASS" if g1 else "FAIL"
        g2s = "PASS" if g2 else "FAIL"
        g3s = "PASS" if g3 else "FAIL"

        all_pass_this = g1 and g2 and g3
        verdict = "PASS" if all_pass_this else "FAIL"

        if all_pass_this:
            any_pass = True
            if dsharpe > best_sharpe_delta:
                best_sharpe_delta = dsharpe
                best_config = label

        print(f"  {label:<22} {dsharpe:>+7.3f} {g1s:>8} {dcagr:>+7.2%} {g2s:>6} "
              f"{ddd:>+6.1%} {g3s:>7} {verdict:>8}")

    # ══════════════════════════════════════════════════════════════════════
    #  RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    if any_pass:
        print(f"  RECOMMENDATION: Deploy '{best_config}' (Sharpe improvement {best_sharpe_delta:+.3f})")
        m_best = dict(all_configs)[best_config]
        print(f"    CAGR: {m_best['cagr']:+.2%} (vs {m_base['cagr']:+.2%} baseline)")
        print(f"    Sharpe: {m_best['sharpe']:.3f} (vs {m_base['sharpe']:.3f} baseline)")
        print(f"    Max DD: {m_best['max_dd']:.1%} (vs {m_base['max_dd']:.1%} baseline)")
    else:
        print("  RECOMMENDATION: DROP ML Slow — no filter mode passes all 3 criteria")
        print()
        # Show closest
        closest = max(all_configs, key=lambda x: x[1]["sharpe"] - m_base["sharpe"])
        ds = closest[1]["sharpe"] - m_base["sharpe"]
        dc = closest[1]["cagr"] - m_base["cagr"]
        dd = closest[1]["max_dd"] - m_base["max_dd"]
        print(f"  Closest: '{closest[0]}' (ΔSharpe {ds:+.3f}, ΔCAGR {dc:+.2%}, ΔDD {dd:+.1%})")
        print(f"  The 30-day model does not improve the system in any configuration tested.")
    print(f"{'='*90}")

    print(f"\n  Runtime: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
