"""
Momentum Strategy Backtest & Comparison
========================================
Runs three configurations and compares against SPY B&H:

  1. Momentum alone  (5 primary + 2 flex = 7 slots)
  2. ML alone         (6 slots — baseline)
  3. ML + Momentum    (ML 2 + Mom 4 + 2 flex = max 8)

Prints full metrics table, correlation analysis, overlap frequency,
avg position count per day, and gate condition checks.

Run with:
    python3 backtest_momentum.py
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from unified_backtester import (
    INITIAL_CASH, HOLD_DAYS,
    MLMediumStrategy, MomentumStrategy,
    PortfolioManager,
    SLOT_ML_ONLY, SLOT_MOM_ONLY, SLOT_ML_MOM,
)
from backtest_utils import load_predictions, calc_metrics, calc_alpha_beta
from diagnose_combined import instrumented_run

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"

ML_THRESHOLD = 0.55


def fetch_ohlcv(symbols, start, end):
    """Download close prices and volume for all symbols."""
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
    print("=" * 80)
    print("  MOMENTUM STRATEGY BACKTEST & COMPARISON (FIXED LOGIC)")
    print("  Changes: 8 max slots, no overlap amplification,")
    print("           strategy-specific cooldowns")
    print("=" * 80)

    # ── 1. Load ML predictions ───────────────────────────────────────────
    print("\n  Loading ML predictions ...")
    df = load_predictions()
    all_dates = sorted(df["date"].unique().tolist())
    universe_syms = sorted(df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    n_dates = len(all_dates)

    print(f"  {n_dates} trading days | {years:.1f} years | {len(universe_syms)} symbols")

    # ── 2. Fetch OHLCV data ──────────────────────────────────────────────
    print("\n  Fetching price & volume data from Yahoo Finance ...")
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    close, volume = fetch_ohlcv(universe_syms,
                                 start.strftime("%Y-%m-%d"),
                                 end.strftime("%Y-%m-%d"))
    print(f"  Price data: {len(close)} days, {len(close.columns)} symbols")

    # Align to simulation dates
    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    volume_aligned = volume.reindex(sim_index, method="ffill") if volume is not None else None

    # SPY benchmark
    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close_aligned["SPY"].to_dict()
    m_spy = calc_metrics(spy_bh, [], years, "SPY B&H")

    # ── 3. Build strategies ──────────────────────────────────────────────
    print("\n  Building strategies ...")
    ml_strat = MLMediumStrategy(df, threshold=ML_THRESHOLD)
    mom_strat = MomentumStrategy(close, volume_data=volume)

    # ── 4. Run backtests (instrumented for position tracking) ────────────
    configs = {}
    diags = {}

    # 4a. Momentum alone
    print("\n  [1/3] Running MOMENTUM ALONE (7 max positions) ...")
    mom_vals, mom_trades, d_mom = instrumented_run(
        [mom_strat], SLOT_MOM_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=60, label="Momentum")
    m_mom = calc_metrics(mom_vals, mom_trades, years, "Momentum")
    a_mom, _ = calc_alpha_beta(mom_vals, spy_bh.reindex(mom_vals.index, method="ffill"))
    m_mom["alpha"] = a_mom
    m_mom["avg_pos"] = np.mean(d_mom["daily_pos_count"])
    configs["Momentum"] = (mom_vals, mom_trades, m_mom)
    diags["Momentum"] = d_mom
    print(f"    CAGR {m_mom['cagr']:+.2%} | Sharpe {m_mom['sharpe']:.3f} | "
          f"Trades {m_mom['n_trades']:,} | WR {m_mom['win_rate']:.1%}")

    # 4b. ML alone (baseline)
    print("\n  [2/3] Running ML ALONE (6 max positions, baseline) ...")
    ml_vals, ml_trades, d_ml = instrumented_run(
        [ml_strat], SLOT_ML_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML Medium")
    m_ml = calc_metrics(ml_vals, ml_trades, years, "ML Medium")
    a_ml, _ = calc_alpha_beta(ml_vals, spy_bh.reindex(ml_vals.index, method="ffill"))
    m_ml["alpha"] = a_ml
    m_ml["avg_pos"] = np.mean(d_ml["daily_pos_count"])
    configs["ML Medium"] = (ml_vals, ml_trades, m_ml)
    diags["ML Medium"] = d_ml
    print(f"    CAGR {m_ml['cagr']:+.2%} | Sharpe {m_ml['sharpe']:.3f} | "
          f"Trades {m_ml['n_trades']:,} | WR {m_ml['win_rate']:.1%}")

    # 4c. ML + Momentum combined
    print("\n  [3/3] Running ML + MOMENTUM COMBINED (8 max positions, fixed logic) ...")
    comb_vals, comb_trades, d_comb = instrumented_run(
        [ml_strat, mom_strat], SLOT_ML_MOM, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML + Momentum")
    m_comb = calc_metrics(comb_vals, comb_trades, years, "ML + Momentum")
    a_comb, _ = calc_alpha_beta(comb_vals, spy_bh.reindex(comb_vals.index, method="ffill"))
    m_comb["alpha"] = a_comb
    m_comb["avg_pos"] = np.mean(d_comb["daily_pos_count"])
    configs["ML + Momentum"] = (comb_vals, comb_trades, m_comb)
    diags["ML + Momentum"] = d_comb
    print(f"    CAGR {m_comb['cagr']:+.2%} | Sharpe {m_comb['sharpe']:.3f} | "
          f"Trades {m_comb['n_trades']:,} | WR {m_comb['win_rate']:.1%}")

    # ── 5. Comparison table ──────────────────────────────────────────────
    labels = ["Momentum", "ML Medium", "ML + Momentum", "SPY B&H"]
    all_metrics = {
        "Momentum": m_mom,
        "ML Medium": m_ml,
        "ML + Momentum": m_comb,
        "SPY B&H": m_spy,
    }
    m_spy["alpha"] = 0.0
    m_spy["avg_pos"] = 0.0

    metric_rows = [
        ("CAGR",          "cagr",          lambda v: f"{v:+.2%}"),
        ("Sharpe",        "sharpe",        lambda v: f"{v:.3f}"),
        ("Sortino",       "sortino",       lambda v: f"{v:.3f}"),
        ("Max Drawdown",  "max_dd",        lambda v: f"{v:.1%}"),
        ("Final Value",   "final_value",   lambda v: f"${v:,.0f}"),
        ("Alpha vs SPY",  "alpha",         lambda v: f"{v:+.2%}"),
        ("Total Trades",  "n_trades",      lambda v: f"{v:,}"),
        ("Win Rate",      "win_rate",      lambda v: f"{v:.1%}" if not np.isnan(v) else "—"),
        ("Profit Factor", "profit_factor",
         lambda v: f"{v:.2f}" if np.isfinite(v) else "inf" if not np.isnan(v) else "—"),
        ("Avg Trade Ret", "avg_trade_ret", lambda v: f"{v:+.2%}" if not np.isnan(v) else "—"),
        ("Avg Pos/Day",   "avg_pos",       lambda v: f"{v:.2f}" if v > 0 else "—"),
    ]

    col0 = 18
    col = 18
    print(f"\n{'='*80}")
    print("  FULL COMPARISON TABLE")
    print(f"{'='*80}")
    header = f"  {'Metric':<{col0}}" + "".join(f"{l:<{col}}" for l in labels)
    print(header)
    print(f"  {'─'*(col0-1)}" + "".join(f" {'─'*(col-2)} " for _ in labels))

    for name, key, fmt in metric_rows:
        row = f"  {name:<{col0}}"
        for label in labels:
            m = all_metrics[label]
            val = m.get(key, np.nan)
            row += f"{fmt(val):<{col}}"
        print(row)

    # ── 6. Correlation analysis ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  CORRELATION ANALYSIS")
    print(f"{'='*80}")

    mom_daily = mom_vals.pct_change().dropna()
    ml_daily = ml_vals.pct_change().dropna()
    comb_daily = comb_vals.pct_change().dropna()
    spy_daily = spy_bh.pct_change().dropna()

    combined = pd.concat([mom_daily, ml_daily, comb_daily, spy_daily], axis=1, join="inner")
    combined.columns = ["Momentum", "ML Medium", "ML+Mom Combined", "SPY"]

    corr = combined.corr()
    print(f"\n  Daily Return Correlations:")
    col_names = ["Momentum", "ML Medium", "ML+Mom Combined", "SPY"]
    print(f"  {'':18}" + "".join(f"{n:>14}" for n in col_names))
    for name in col_names:
        row = f"  {name:<18}"
        for name2 in col_names:
            row += f"{corr.loc[name, name2]:>14.3f}"
        print(row)

    ml_mom_corr = corr.loc["Momentum", "ML Medium"]
    print(f"\n  ML ↔ Momentum correlation: {ml_mom_corr:.3f}")

    # ── 7. Overlap & position analysis ───────────────────────────────────
    print(f"\n{'='*80}")
    print("  POSITION & ACTIVITY ANALYSIS")
    print(f"{'='*80}")

    # Avg position count by strategy in combined
    from collections import defaultdict
    strat_pos_sums = defaultdict(float)
    for day_dict in d_comb["daily_pos_by_strat"]:
        for strat, cnt in day_dict.items():
            strat_pos_sums[strat] += cnt
    print(f"\n  Combined system — avg positions per day by strategy:")
    for strat in sorted(strat_pos_sums.keys()):
        avg = strat_pos_sums[strat] / n_dates
        print(f"    {strat:<18} {avg:.2f}")
    print(f"    {'TOTAL':<18} {m_comb['avg_pos']:.2f}")

    # Signal acceptance rates
    print(f"\n  Combined system — signal acceptance:")
    gen = d_comb["signals_generated"]
    acc = d_comb["signals_accepted"]
    for strat in sorted(gen.keys()):
        g = gen[strat]
        a = acc.get(strat, 0)
        print(f"    {strat:<18} {a:>5} / {g:>6} accepted ({a/g:.1%})")

    # Block reasons
    br = d_comb["block_reasons"]
    if br:
        print(f"\n  Combined system — block reasons:")
        for reason, count in sorted(br.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25} {count:>6}")

    # Monthly activity
    print(f"\n  Monthly Trade Counts:")
    for label_name, (vals, trades_list, m) in configs.items():
        print(f"    {label_name}: {m['n_trades']:,} total "
              f"({m['n_trades']/years/12:.1f} trades/month)")

    # ── 8. Gate condition checks ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  GATE CONDITION CHECKS")
    print(f"{'='*80}")

    gates = []

    # Gate 1: Positive alpha (momentum alone)
    g1 = m_mom["alpha"] > 0
    gates.append(("Momentum alpha > 0", g1,
                  f"alpha = {m_mom['alpha']:+.2%}"))

    # Gate 2: Combined Sharpe > ML alone
    g2 = m_comb["sharpe"] > m_ml["sharpe"]
    gates.append(("Combined Sharpe > ML alone", g2,
                  f"{m_comb['sharpe']:.3f} vs {m_ml['sharpe']:.3f}"))

    # Gate 3: ML ↔ Momentum correlation < 0.6
    g3 = ml_mom_corr < 0.6
    gates.append(("ML ↔ Momentum corr < 0.6", g3,
                  f"corr = {ml_mom_corr:.3f}"))

    # Gate 4: Combined max DD within 3pp of ML alone
    dd_diff = m_comb["max_dd"] - m_ml["max_dd"]  # more negative = worse
    g4 = dd_diff > -0.03  # within 3 percentage points
    gates.append(("Combined max DD within 3pp of ML", g4,
                  f"diff = {dd_diff:+.1%} (ML {m_ml['max_dd']:.1%}, "
                  f"Combined {m_comb['max_dd']:.1%})"))

    all_pass = True
    for desc, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")
        print(f"         {detail}")

    print(f"\n{'='*80}")
    if all_pass:
        print("  VERDICT: ALL GATES PASS — Momentum strategy approved for production")
    else:
        failed = [desc for desc, passed, _ in gates if not passed]
        print(f"  VERDICT: {len(failed)} gate(s) FAILED — review before deploying")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*80}")

    print(f"\n  Runtime: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
