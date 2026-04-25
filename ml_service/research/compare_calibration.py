"""
Calibration Comparison
======================
Runs the backtest simulator on both baseline and calibrated predictions,
then prints CAGR, Sharpe, and lift side by side.

Run with:
    python3 compare_calibration.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_utils import calc_metrics, calc_alpha_beta
from backtest_ml import (
    INITIAL_CASH, HOLD_DAYS, SLIPPAGE,
    MAX_POSITIONS, POSITION_PCT, THRESHOLDS,
    SPY_IDLE_RESERVE_PCT, SPY_IDLE_THRESHOLD_PCT, SPY_IDLE_INVEST_PCT,
    run_simulation, make_ml_signal_fn, fetch_benchmarks,
)

warnings.filterwarnings("ignore")

DATA_DIR         = Path(__file__).resolve().parent / "data"
PRED_FILE        = DATA_DIR / "predictions.parquet"
CALIB_PRED_FILE  = DATA_DIR / "predictions_calibrated.parquet"
PROB_THRESH      = 0.55


def load_preds(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found — run train_model.py first.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])
    print(f"  [{label}] {len(df):,} rows  |  {df['symbol'].nunique()} symbols  |  "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def build_signal_lookup(df: pd.DataFrame) -> dict:
    sigs = {}
    for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
        grp_s = grp.sort_values("prob", ascending=False)
        sigs[date] = list(grp_s[["symbol", "prob", "fwd_ret"]].itertuples(
            index=False, name=None))
    return sigs


def compute_lift(df: pd.DataFrame, threshold: float) -> float:
    valid = df.dropna(subset=["fwd_ret"])
    if len(valid) == 0:
        return np.nan
    avg_all = valid["fwd_ret"].mean()
    picks = valid[valid["prob"] >= threshold]
    if len(picks) == 0:
        return np.nan
    return picks["fwd_ret"].mean() - avg_all


def main():
    t0 = time.perf_counter()
    print("=" * 75)
    print("  Calibration Comparison — Baseline vs Isotonic Calibrated")
    print("=" * 75)

    # ── Load both prediction sets ────────────────────────────────────────────
    df_base  = load_preds(PRED_FILE, "Baseline")
    df_calib = load_preds(CALIB_PRED_FILE, "Calibrated")

    # Use the intersection of dates present in both
    all_dates = sorted(set(df_base["date"].unique()) & set(df_calib["date"].unique()))
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    universe_syms = sorted(
        set(df_base["symbol"].unique()) | set(df_calib["symbol"].unique()))

    print(f"\n  {len(all_dates)} common trading days over {years:.1f} years")

    sigs_base  = build_signal_lookup(df_base)
    sigs_calib = build_signal_lookup(df_calib)

    # ── Fetch SPY for benchmarking ───────────────────────────────────────────
    start_str = pd.Timestamp(all_dates[0]).strftime("%Y-%m-%d")
    end_str   = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close_px  = fetch_benchmarks(start_str, end_str, universe_syms)
    close_px  = close_px.reindex(
        pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates]), method="ffill")

    spy_px     = close_px["SPY"].dropna()
    spy_bh     = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_px_dict = close_px["SPY"].to_dict()

    m_spy = calc_metrics(spy_bh, [], years, "SPY Buy & Hold")

    # ── Run backtests at threshold 0.55 (both pure ML and ML+SPY idle) ───────
    thresh = 0.55
    signal_fn = make_ml_signal_fn(thresh)

    print(f"\n  Running backtest at threshold >{thresh:.2f} ...")

    # Baseline — pure ML
    base_vals, base_trades = run_simulation(
        sigs_base, all_dates, signal_fn, years,
        label=f"Baseline >{thresh:.2f}", verbose=False)
    m_base = calc_metrics(base_vals, base_trades, years, f"Baseline >{thresh:.2f}")
    a_base, b_base = calc_alpha_beta(base_vals, spy_bh.reindex(base_vals.index, method="ffill"))
    m_base["alpha"] = a_base

    # Calibrated — pure ML
    calib_vals, calib_trades = run_simulation(
        sigs_calib, all_dates, signal_fn, years,
        label=f"Calibrated >{thresh:.2f}", verbose=False)
    m_calib = calc_metrics(calib_vals, calib_trades, years, f"Calibrated >{thresh:.2f}")
    a_calib, b_calib = calc_alpha_beta(calib_vals, spy_bh.reindex(calib_vals.index, method="ffill"))
    m_calib["alpha"] = a_calib

    # Baseline — ML + SPY idle
    base_spy_vals, base_spy_trades = run_simulation(
        sigs_base, all_dates, signal_fn, years,
        label=f"Baseline+SPY >{thresh:.2f}", verbose=False,
        spy_prices=spy_px_dict)
    m_base_spy = calc_metrics(base_spy_vals, base_spy_trades, years, f"Baseline+SPY >{thresh:.2f}")
    a_bs, _ = calc_alpha_beta(base_spy_vals, spy_bh.reindex(base_spy_vals.index, method="ffill"))
    m_base_spy["alpha"] = a_bs

    # Calibrated — ML + SPY idle
    calib_spy_vals, calib_spy_trades = run_simulation(
        sigs_calib, all_dates, signal_fn, years,
        label=f"Calibrated+SPY >{thresh:.2f}", verbose=False,
        spy_prices=spy_px_dict)
    m_calib_spy = calc_metrics(calib_spy_vals, calib_spy_trades, years, f"Calibrated+SPY >{thresh:.2f}")
    a_cs, _ = calc_alpha_beta(calib_spy_vals, spy_bh.reindex(calib_spy_vals.index, method="ffill"))
    m_calib_spy["alpha"] = a_cs

    # ── Lift ─────────────────────────────────────────────────────────────────
    lift_base  = compute_lift(df_base,  thresh)
    lift_calib = compute_lift(df_calib, thresh)

    # ── Side-by-side comparison table ────────────────────────────────────────
    def delta_str(base_val, calib_val, fmt="+.2%", pct=True):
        diff = calib_val - base_val
        if pct:
            return f"({diff:{fmt}})"
        return f"({diff:{fmt}})"

    col_w = [22, 18, 18, 14]
    headers = ("Metric", "Baseline", "Calibrated", "Delta")

    def fmt_row(vals):
        return "  " + "".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    print(f"\n{'='*75}")
    print(f"  PURE ML — SIDE-BY-SIDE @ THRESHOLD >{thresh:.2f}")
    print(f"{'='*75}")
    print(fmt_row(headers))
    print(fmt_row(tuple("─" * (w-1) for w in col_w)))

    rows = [
        ("CAGR",         f"{m_base['cagr']:+.2%}",         f"{m_calib['cagr']:+.2%}",
         f"{m_calib['cagr']-m_base['cagr']:+.2%}"),
        ("Sharpe",       f"{m_base['sharpe']:.3f}",         f"{m_calib['sharpe']:.3f}",
         f"{m_calib['sharpe']-m_base['sharpe']:+.3f}"),
        ("Sortino",      f"{m_base['sortino']:.3f}",        f"{m_calib['sortino']:.3f}",
         f"{m_calib['sortino']-m_base['sortino']:+.3f}"),
        ("Max Drawdown", f"{m_base['max_dd']:.1%}",         f"{m_calib['max_dd']:.1%}",
         f"{m_calib['max_dd']-m_base['max_dd']:+.1%}"),
        ("Final Value",  f"${m_base['final_value']:,.0f}",  f"${m_calib['final_value']:,.0f}",
         f"${m_calib['final_value']-m_base['final_value']:+,.0f}"),
        ("Alpha vs SPY", f"{m_base['alpha']:+.2%}",         f"{m_calib['alpha']:+.2%}",
         f"{m_calib['alpha']-m_base['alpha']:+.2%}"),
        ("Total Trades", f"{m_base['n_trades']:,}",         f"{m_calib['n_trades']:,}",
         f"{m_calib['n_trades']-m_base['n_trades']:+,}"),
        ("Win Rate",     f"{m_base['win_rate']:.1%}",       f"{m_calib['win_rate']:.1%}",
         f"{m_calib['win_rate']-m_base['win_rate']:+.1%}"),
        ("Avg Trade Ret",f"{m_base['avg_trade_ret']:+.2%}", f"{m_calib['avg_trade_ret']:+.2%}",
         f"{m_calib['avg_trade_ret']-m_base['avg_trade_ret']:+.2%}"),
        ("Lift",         f"{lift_base*100:+.2f}%",          f"{lift_calib*100:+.2f}%",
         f"{(lift_calib-lift_base)*100:+.2f}%"),
    ]
    for r in rows:
        print(fmt_row(r))

    # ── ML + SPY Idle ────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  ML + SPY IDLE — SIDE-BY-SIDE @ THRESHOLD >{thresh:.2f}")
    print(f"{'='*75}")
    print(fmt_row(headers))
    print(fmt_row(tuple("─" * (w-1) for w in col_w)))

    rows_spy = [
        ("CAGR",         f"{m_base_spy['cagr']:+.2%}",         f"{m_calib_spy['cagr']:+.2%}",
         f"{m_calib_spy['cagr']-m_base_spy['cagr']:+.2%}"),
        ("Sharpe",       f"{m_base_spy['sharpe']:.3f}",         f"{m_calib_spy['sharpe']:.3f}",
         f"{m_calib_spy['sharpe']-m_base_spy['sharpe']:+.3f}"),
        ("Sortino",      f"{m_base_spy['sortino']:.3f}",        f"{m_calib_spy['sortino']:.3f}",
         f"{m_calib_spy['sortino']-m_base_spy['sortino']:+.3f}"),
        ("Max Drawdown", f"{m_base_spy['max_dd']:.1%}",         f"{m_calib_spy['max_dd']:.1%}",
         f"{m_calib_spy['max_dd']-m_base_spy['max_dd']:+.1%}"),
        ("Final Value",  f"${m_base_spy['final_value']:,.0f}",  f"${m_calib_spy['final_value']:,.0f}",
         f"${m_calib_spy['final_value']-m_base_spy['final_value']:+,.0f}"),
        ("Alpha vs SPY", f"{m_base_spy['alpha']:+.2%}",         f"{m_calib_spy['alpha']:+.2%}",
         f"{m_calib_spy['alpha']-m_base_spy['alpha']:+.2%}"),
        ("Total Trades", f"{m_base_spy['n_trades']:,}",         f"{m_calib_spy['n_trades']:,}",
         f"{m_calib_spy['n_trades']-m_base_spy['n_trades']:+,}"),
        ("Win Rate",     f"{m_base_spy['win_rate']:.1%}",       f"{m_calib_spy['win_rate']:.1%}",
         f"{m_calib_spy['win_rate']-m_base_spy['win_rate']:+.1%}"),
        ("Avg Trade Ret",f"{m_base_spy['avg_trade_ret']:+.2%}", f"{m_calib_spy['avg_trade_ret']:+.2%}",
         f"{m_calib_spy['avg_trade_ret']-m_base_spy['avg_trade_ret']:+.2%}"),
        ("Lift",         f"{lift_base*100:+.2f}%",              f"{lift_calib*100:+.2f}%",
         f"{(lift_calib-lift_base)*100:+.2f}%"),
    ]
    for r in rows_spy:
        print(fmt_row(r))

    # ── SPY benchmark for reference ──────────────────────────────────────────
    print(f"\n  SPY Buy & Hold:  CAGR {m_spy['cagr']:+.2%}  |  "
          f"Sharpe {m_spy['sharpe']:.3f}  |  "
          f"Final ${m_spy['final_value']:,.0f}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    cagr_diff = m_calib_spy['cagr'] - m_base_spy['cagr']
    sharpe_diff = m_calib_spy['sharpe'] - m_base_spy['sharpe']
    lift_diff = lift_calib - lift_base

    wins = sum([cagr_diff > 0, sharpe_diff > 0, lift_diff > 0])
    if wins >= 2:
        print("  VERDICT: Calibration HELPS — improves majority of key metrics")
    elif wins == 0:
        print("  VERDICT: Calibration HURTS — degrades all key metrics")
    else:
        print("  VERDICT: Calibration is MIXED — some metrics improve, others degrade")

    print(f"    CAGR delta:   {cagr_diff:+.2%}")
    print(f"    Sharpe delta: {sharpe_diff:+.3f}")
    print(f"    Lift delta:   {lift_diff*100:+.2f}%")
    print(f"{'='*75}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
