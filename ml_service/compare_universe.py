"""
Universe Expansion Comparison
==============================
Backtests the 37-symbol baseline model vs the 60-symbol expanded model
side by side. Both use the same walk-forward validation, same features,
same threshold optimization.

Run with:
    python3 compare_universe.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_ml import (
    INITIAL_CASH, THRESHOLDS,
    run_simulation, calc_metrics, calc_alpha_beta, make_ml_signal_fn,
    fetch_benchmarks,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
PRED_37  = DATA_DIR / "predictions_37universe.parquet"
PRED_60  = DATA_DIR / "predictions_60universe.parquet"


def load_preds(path, label):
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])
    print(f"  [{label}] {len(df):,} rows  |  {df['symbol'].nunique()} symbols  |  "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def build_signal_lookup(df):
    sigs = {}
    for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
        grp_s = grp.sort_values("prob", ascending=False)
        sigs[date] = list(grp_s[["symbol", "prob", "fwd_ret"]].itertuples(
            index=False, name=None))
    return sigs


def compute_lift(df, threshold):
    valid = df.dropna(subset=["fwd_ret"])
    if len(valid) == 0:
        return np.nan
    avg_all = valid["fwd_ret"].mean()
    picks = valid[valid["prob"] >= threshold]
    if len(picks) == 0:
        return np.nan
    return picks["fwd_ret"].mean() - avg_all


def run_all_thresholds(sigs, all_dates, years, spy_bh, spy_px_dict, label_prefix):
    """Run pure ML and ML+SPY sweeps for all thresholds, return results."""
    pure_results = []
    spy_results = []

    for thresh in THRESHOLDS:
        signal_fn = make_ml_signal_fn(thresh)

        vals, trades = run_simulation(
            sigs, all_dates, signal_fn, years,
            label=f"{label_prefix} >{thresh:.2f}", verbose=False)
        m = calc_metrics(vals, trades, years, f"{label_prefix} >{thresh:.2f}")
        a, b = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
        m["alpha"] = a
        m["beta"] = b
        pure_results.append((thresh, vals, m))

        vals_s, trades_s = run_simulation(
            sigs, all_dates, signal_fn, years,
            label=f"{label_prefix}+SPY >{thresh:.2f}", verbose=False,
            spy_prices=spy_px_dict)
        m_s = calc_metrics(vals_s, trades_s, years, f"{label_prefix}+SPY >{thresh:.2f}")
        a_s, b_s = calc_alpha_beta(vals_s, spy_bh.reindex(vals_s.index, method="ffill"))
        m_s["alpha"] = a_s
        m_s["beta"] = b_s
        spy_results.append((thresh, vals_s, m_s))

    return pure_results, spy_results


def print_sweep_table(results_37, results_60, title):
    col_w = [8, 10, 9, 8, 9, 8, 10, 9, 8, 9]
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")

    headers = ("Thresh", "Model", "CAGR", "Sharpe", "Sortino", "MaxDD",
               "Trades", "WinRate", "PrftF", "Alpha")
    print("  " + "".join(str(h).ljust(w) for h, w in zip(headers, col_w)))
    print("  " + "".join(("─" * (w-1)).ljust(w) for w in col_w))

    for (t37, _, m37), (t60, _, m60) in zip(results_37, results_60):
        pf37 = f"{m37['profit_factor']:.2f}" if np.isfinite(m37['profit_factor']) else "inf"
        pf60 = f"{m60['profit_factor']:.2f}" if np.isfinite(m60['profit_factor']) else "inf"

        row37 = (f">{t37:.2f}", "37-sym",
                 f"{m37['cagr']:+.2%}", f"{m37['sharpe']:.3f}",
                 f"{m37['sortino']:.3f}", f"{m37['max_dd']:.1%}",
                 f"{m37['n_trades']:,}", f"{m37['win_rate']:.1%}",
                 pf37, f"{m37['alpha']:+.2%}")
        row60 = ("", "60-sym",
                 f"{m60['cagr']:+.2%}", f"{m60['sharpe']:.3f}",
                 f"{m60['sortino']:.3f}", f"{m60['max_dd']:.1%}",
                 f"{m60['n_trades']:,}", f"{m60['win_rate']:.1%}",
                 pf60, f"{m60['alpha']:+.2%}")

        # Delta
        cagr_d = m60['cagr'] - m37['cagr']
        sharpe_d = m60['sharpe'] - m37['sharpe']
        delta = ("", "delta",
                 f"{cagr_d:+.2%}", f"{sharpe_d:+.3f}",
                 "", "", "", "", "", "")

        print("  " + "".join(str(v).ljust(w) for v, w in zip(row37, col_w)))
        print("  " + "".join(str(v).ljust(w) for v, w in zip(row60, col_w)))
        print("  " + "".join(str(v).ljust(w) for v, w in zip(delta, col_w)))
        print("  " + "".join(("─" * (w-1)).ljust(w) for w in col_w))


def print_head_to_head(m37, m60, lift37, lift60, title):
    col_w = [22, 18, 18, 14]
    headers = ("Metric", "37-Symbol", "60-Symbol", "Delta")

    def fmt_row(vals):
        return "  " + "".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    print(fmt_row(headers))
    print(fmt_row(tuple("─" * (w-1) for w in col_w)))

    rows = [
        ("CAGR",         f"{m37['cagr']:+.2%}",         f"{m60['cagr']:+.2%}",
         f"{m60['cagr']-m37['cagr']:+.2%}"),
        ("Sharpe",       f"{m37['sharpe']:.3f}",         f"{m60['sharpe']:.3f}",
         f"{m60['sharpe']-m37['sharpe']:+.3f}"),
        ("Sortino",      f"{m37['sortino']:.3f}",        f"{m60['sortino']:.3f}",
         f"{m60['sortino']-m37['sortino']:+.3f}"),
        ("Max Drawdown", f"{m37['max_dd']:.1%}",         f"{m60['max_dd']:.1%}",
         f"{m60['max_dd']-m37['max_dd']:+.1%}"),
        ("Final Value",  f"${m37['final_value']:,.0f}",  f"${m60['final_value']:,.0f}",
         f"${m60['final_value']-m37['final_value']:+,.0f}"),
        ("Alpha vs SPY", f"{m37['alpha']:+.2%}",         f"{m60['alpha']:+.2%}",
         f"{m60['alpha']-m37['alpha']:+.2%}"),
        ("Total Trades", f"{m37['n_trades']:,}",         f"{m60['n_trades']:,}",
         f"{m60['n_trades']-m37['n_trades']:+,}"),
        ("Win Rate",     f"{m37['win_rate']:.1%}",       f"{m60['win_rate']:.1%}",
         f"{m60['win_rate']-m37['win_rate']:+.1%}"),
        ("Profit Factor",
         f"{m37['profit_factor']:.2f}" if np.isfinite(m37['profit_factor']) else "inf",
         f"{m60['profit_factor']:.2f}" if np.isfinite(m60['profit_factor']) else "inf",
         ""),
        ("Avg Trade Ret",f"{m37['avg_trade_ret']:+.2%}", f"{m60['avg_trade_ret']:+.2%}",
         f"{m60['avg_trade_ret']-m37['avg_trade_ret']:+.2%}"),
        ("Lift",         f"{lift37*100:+.2f}%",          f"{lift60*100:+.2f}%",
         f"{(lift60-lift37)*100:+.2f}%"),
    ]
    for r in rows:
        print(fmt_row(r))


def analyze_degradation(df_37, df_60):
    """Analyze WHY the 60-symbol model may perform differently."""
    print(f"\n{'='*75}")
    print("  DIAGNOSTIC ANALYSIS")
    print(f"{'='*75}")

    # 1. Signal quality by symbol group
    new_syms = set(df_60["symbol"].unique()) - set(df_37["symbol"].unique())
    old_syms = set(df_37["symbol"].unique())

    print(f"\n  Original symbols: {len(old_syms)}  |  New symbols: {len(new_syms)}")

    # Lift by group in the 60-symbol predictions
    valid_60 = df_60.dropna(subset=["fwd_ret"])
    avg_all = valid_60["fwd_ret"].mean()

    for group_name, syms in [("Original 37", old_syms), ("New 23", new_syms)]:
        grp = valid_60[valid_60["symbol"].isin(syms)]
        picks = grp[grp["prob"] >= 0.55]
        n_picks = len(picks)
        avg_ret = picks["fwd_ret"].mean() if n_picks > 0 else np.nan
        lift = avg_ret - avg_all if n_picks > 0 else np.nan
        hit = (picks["target"] == 1).mean() if n_picks > 0 else np.nan
        avg_prob = picks["prob"].mean() if n_picks > 0 else np.nan

        print(f"\n  [{group_name}] — {len(syms)} symbols")
        print(f"    Picks at p>0.55   : {n_picks:,}")
        print(f"    Avg return (picks): {avg_ret*100:+.2f}%" if n_picks > 0 else "    Avg return (picks): N/A")
        print(f"    Lift vs universe  : {lift*100:+.2f}%" if n_picks > 0 else "    Lift vs universe  : N/A")
        print(f"    Hit rate          : {hit*100:.1f}%" if n_picks > 0 else "    Hit rate          : N/A")
        print(f"    Avg probability   : {avg_prob:.3f}" if n_picks > 0 else "    Avg probability   : N/A")

    # 2. Sector breakdown of new symbols
    sector_map = {
        "CRM": "Tech", "ORCL": "Tech", "ADBE": "Tech", "CSCO": "Tech",
        "QCOM": "Semis", "COST": "Staples", "WMT": "Staples",
        "HD": "Consumer", "LOW": "Consumer",
        "LLY": "Health", "JNJ": "Health", "ABBV": "Health",
        "BAC": "Finance", "GS": "Finance", "MS": "Finance",
        "CVX": "Energy", "XOM": "Energy",
        "CAT": "Industrial", "DE": "Industrial", "BA": "Industrial",
        "VGK": "International", "VWO": "International", "IEFA": "International",
    }

    print(f"\n  Signal quality by sector (new symbols, p>0.55):")
    print(f"  {'Sector':<15} {'Picks':>6} {'Avg Ret':>10} {'Hit Rate':>10} {'Lift':>10}")
    print(f"  {'─'*14} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")

    for sector in sorted(set(sector_map.values())):
        sec_syms = [s for s, sec in sector_map.items() if sec == sector]
        sec_df = valid_60[(valid_60["symbol"].isin(sec_syms)) & (valid_60["prob"] >= 0.55)]
        if len(sec_df) == 0:
            continue
        sec_ret = sec_df["fwd_ret"].mean()
        sec_hit = (sec_df["target"] == 1).mean()
        sec_lift = sec_ret - avg_all
        print(f"  {sector:<15} {len(sec_df):>6} {sec_ret*100:>+9.2f}% {sec_hit*100:>9.1f}% {sec_lift*100:>+9.2f}%")

    # 3. Probability distribution comparison
    print(f"\n  Probability distribution (p>0.55 picks):")
    for label, df in [("37-sym", df_37), ("60-sym", df_60)]:
        picks = df[df["prob"] >= 0.55]
        if len(picks) == 0:
            continue
        print(f"    [{label}] n={len(picks):,}  mean={picks['prob'].mean():.3f}  "
              f"median={picks['prob'].median():.3f}  "
              f"p75={picks['prob'].quantile(0.75):.3f}  "
              f"p90={picks['prob'].quantile(0.90):.3f}")

    # 4. Conclusion
    print(f"\n  {'─'*75}")
    print("  LIKELY CAUSES:")

    picks_37 = df_37[df_37["prob"] >= 0.55]
    picks_60 = df_60[df_60["prob"] >= 0.55]
    new_picks = valid_60[(valid_60["symbol"].isin(new_syms)) & (valid_60["prob"] >= 0.55)]

    if len(new_picks) > 0:
        new_lift = new_picks["fwd_ret"].mean() - avg_all
        old_picks_in_60 = valid_60[(valid_60["symbol"].isin(old_syms)) & (valid_60["prob"] >= 0.55)]
        old_lift_in_60 = old_picks_in_60["fwd_ret"].mean() - avg_all if len(old_picks_in_60) > 0 else 0

        if new_lift < old_lift_in_60 * 0.5:
            print("  - NEW SYMBOLS CREATE FALSE SIGNALS: New symbols have significantly")
            print("    lower lift than original symbols, diluting signal quality.")
        if len(picks_60) > len(picks_37) * 1.3:
            print("  - MORE SYMBOLS = MORE NOISE: Expanded universe generates more picks,")
            print("    but marginal picks are lower quality.")
        if picks_60["prob"].mean() < picks_37["prob"].mean() - 0.01:
            print("  - PROBABILITY DILUTION: Mean confidence on picks dropped, suggesting")
            print("    the model spreads probability mass across more symbols.")
        if new_lift > old_lift_in_60:
            print("  - NEW SECTORS ADD VALUE: New symbols actually have better signal")
            print("    quality than original symbols in the expanded model.")


def main():
    t0 = time.perf_counter()
    print("=" * 75)
    print("  Universe Expansion Comparison — 37 vs 60 Symbols")
    print("=" * 75)

    df_37 = load_preds(PRED_37, "37-symbol")
    df_60 = load_preds(PRED_60, "60-symbol")

    # Use intersection of dates for fair comparison
    common_dates = sorted(set(df_37["date"].unique()) & set(df_60["date"].unique()))
    years = (common_dates[-1] - common_dates[0]).days / 365.25
    all_syms = sorted(set(df_37["symbol"].unique()) | set(df_60["symbol"].unique()))

    print(f"\n  {len(common_dates)} common trading days over {years:.1f} years")

    sigs_37 = build_signal_lookup(df_37)
    sigs_60 = build_signal_lookup(df_60)

    # Fetch benchmarks
    start_str = pd.Timestamp(common_dates[0]).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(common_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close_px = fetch_benchmarks(start_str, end_str, all_syms)
    close_px = close_px.reindex(
        pd.DatetimeIndex([pd.Timestamp(d) for d in common_dates]), method="ffill")

    spy_px = close_px["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_px_dict = close_px["SPY"].to_dict()
    m_spy = calc_metrics(spy_bh, [], years, "SPY Buy & Hold")

    # Run all thresholds for both universes
    print(f"\n  Running threshold sweeps for 37-symbol model ...")
    pure_37, spy_37 = run_all_thresholds(
        sigs_37, common_dates, years, spy_bh, spy_px_dict, "37sym")

    print(f"  Running threshold sweeps for 60-symbol model ...")
    pure_60, spy_60 = run_all_thresholds(
        sigs_60, common_dates, years, spy_bh, spy_px_dict, "60sym")

    # Print sweep tables
    print_sweep_table(pure_37, pure_60, "PURE ML — THRESHOLD SWEEP")
    print_sweep_table(spy_37, spy_60, "ML + SPY IDLE — THRESHOLD SWEEP")

    # Head-to-head at 0.55
    m37_pure = next(m for t, _, m in pure_37 if t == 0.55)
    m60_pure = next(m for t, _, m in pure_60 if t == 0.55)
    m37_spy = next(m for t, _, m in spy_37 if t == 0.55)
    m60_spy = next(m for t, _, m in spy_60 if t == 0.55)
    lift_37 = compute_lift(df_37, 0.55)
    lift_60 = compute_lift(df_60, 0.55)

    print_head_to_head(m37_pure, m60_pure, lift_37, lift_60,
                       "PURE ML HEAD-TO-HEAD @ THRESHOLD >0.55")
    print_head_to_head(m37_spy, m60_spy, lift_37, lift_60,
                       "ML + SPY IDLE HEAD-TO-HEAD @ THRESHOLD >0.55")

    # SPY benchmark
    print(f"\n  SPY Buy & Hold:  CAGR {m_spy['cagr']:+.2%}  |  "
          f"Sharpe {m_spy['sharpe']:.3f}  |  Final ${m_spy['final_value']:,.0f}")

    # Run diagnostic analysis
    analyze_degradation(df_37, df_60)

    # Deploy decision
    print(f"\n{'='*75}")
    cagr_ok = m60_spy["cagr"] >= m37_spy["cagr"]
    sharpe_ok = m60_spy["sharpe"] >= m37_spy["sharpe"]

    if cagr_ok and sharpe_ok:
        print("  VERDICT: 60-symbol model PASSES — CAGR and Sharpe both >= baseline")
        print("  Safe to deploy model_60universe.lgb as the production model.")
    else:
        failures = []
        if not cagr_ok:
            failures.append(f"CAGR ({m60_spy['cagr']:+.2%} < {m37_spy['cagr']:+.2%})")
        if not sharpe_ok:
            failures.append(f"Sharpe ({m60_spy['sharpe']:.3f} < {m37_spy['sharpe']:.3f})")
        print(f"  VERDICT: 60-symbol model FAILS — {' and '.join(failures)}")
        print("  DO NOT deploy. Keep model_37universe.lgb as production model.")
    print(f"{'='*75}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
