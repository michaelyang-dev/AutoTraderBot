"""
Three-way comparison: 37-sym baseline vs 60-sym v1 (broken) vs 60-sym v2 (fixed)
"""
import warnings, time
from pathlib import Path
import numpy as np, pandas as pd

from backtest_ml import (
    INITIAL_CASH, run_simulation, calc_metrics, calc_alpha_beta,
    make_ml_signal_fn, fetch_benchmarks,
)

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent / "data"

MODELS = [
    ("37-sym baseline", DATA / "predictions_37universe.parquet"),
    ("60-sym v1 (17t)", DATA / "predictions_60universe.parquet"),
    ("60-sym v2 (150t)", DATA / "predictions_60universe_v2.parquet"),
]

THRESH = 0.55


def load(path, label):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])
    print(f"  [{label}] {len(df):,} rows | {df['symbol'].nunique()} syms | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def build_sigs(df):
    sigs = {}
    for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
        g = grp.sort_values("prob", ascending=False)
        sigs[date] = list(g[["symbol", "prob", "fwd_ret"]].itertuples(index=False, name=None))
    return sigs


def main():
    t0 = time.perf_counter()
    print("=" * 85)
    print("  THREE-WAY COMPARISON @ threshold >0.55 (ML + SPY idle)")
    print("=" * 85)

    dfs = {}
    for label, path in MODELS:
        dfs[label] = load(path, label)

    # Common dates across all three
    date_sets = [set(df["date"].unique()) for df in dfs.values()]
    common = sorted(date_sets[0].intersection(*date_sets[1:]))
    years = (common[-1] - common[0]).days / 365.25
    print(f"\n  {len(common)} common days | {years:.1f} years")

    all_syms = set()
    for df in dfs.values():
        all_syms |= set(df["symbol"].unique())

    # Fetch SPY
    start = pd.Timestamp(common[0]).strftime("%Y-%m-%d")
    end = (pd.Timestamp(common[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close = fetch_benchmarks(start, end, sorted(all_syms))
    close = close.reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in common]), method="ffill")
    spy_px = close["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close["SPY"].to_dict()

    signal_fn = make_ml_signal_fn(THRESH)
    results = {}

    for label, df in dfs.items():
        sigs = build_sigs(df)
        vals, trades = run_simulation(
            sigs, common, signal_fn, years,
            label=label, verbose=False, spy_prices=spy_dict)
        m = calc_metrics(vals, trades, years, label)
        a, b = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
        m["alpha"] = a

        picks = df[df["prob"] >= THRESH]
        avg_all = df.dropna(subset=["fwd_ret"])["fwd_ret"].mean()
        lift = (picks["fwd_ret"].mean() - avg_all) if len(picks) > 0 else 0
        m["lift"] = lift
        m["n_picks"] = len(picks)
        m["avg_prob"] = picks["prob"].mean() if len(picks) > 0 else 0
        m["max_prob"] = picks["prob"].max() if len(picks) > 0 else 0
        results[label] = m

    # Print table
    labels = list(results.keys())
    metrics_rows = [
        ("CAGR",          lambda m: f"{m['cagr']:+.2%}"),
        ("Sharpe",        lambda m: f"{m['sharpe']:.3f}"),
        ("Sortino",       lambda m: f"{m['sortino']:.3f}"),
        ("Max Drawdown",  lambda m: f"{m['max_dd']:.1%}"),
        ("Final Value",   lambda m: f"${m['final_value']:,.0f}"),
        ("Alpha vs SPY",  lambda m: f"{m['alpha']:+.2%}"),
        ("Total Trades",  lambda m: f"{m['n_trades']:,}"),
        ("Win Rate",      lambda m: f"{m['win_rate']:.1%}"),
        ("Profit Factor", lambda m: f"{m['profit_factor']:.2f}" if np.isfinite(m['profit_factor']) else "inf"),
        ("Avg Trade Ret", lambda m: f"{m['avg_trade_ret']:+.2%}"),
        ("OOS Lift",      lambda m: f"{m['lift']*100:+.2f}%"),
        ("Picks (p>0.55)",lambda m: f"{m['n_picks']:,}"),
        ("Avg Prob",      lambda m: f"{m['avg_prob']:.3f}"),
        ("Max Prob",      lambda m: f"{m['max_prob']:.3f}"),
    ]

    col0 = 18
    col = 20
    print(f"\n{'='*78}")
    header = f"  {'Metric':<{col0}}" + "".join(f"{l:<{col}}" for l in labels)
    print(header)
    print(f"  {'─'*(col0-1)}" + "".join(f" {'─'*(col-2)} " for _ in labels))

    for name, fmt_fn in metrics_rows:
        row = f"  {name:<{col0}}"
        for label in labels:
            row += f"{fmt_fn(results[label]):<{col}}"
        print(row)

    # SPY benchmark
    m_spy = calc_metrics(spy_bh, [], years, "SPY B&H")
    print(f"\n  SPY Buy & Hold:  CAGR {m_spy['cagr']:+.2%}  |  Sharpe {m_spy['sharpe']:.3f}")

    # Verdict
    print(f"\n{'='*78}")
    m37 = results[labels[0]]
    m_v2 = results[labels[2]]
    cagr_ok = m_v2["cagr"] >= m37["cagr"]
    sharpe_ok = m_v2["sharpe"] >= m37["sharpe"]

    if cagr_ok and sharpe_ok:
        print("  VERDICT: 60-sym v2 PASSES — deploy as production model")
    else:
        fails = []
        if not cagr_ok:
            fails.append(f"CAGR ({m_v2['cagr']:+.2%} < {m37['cagr']:+.2%})")
        if not sharpe_ok:
            fails.append(f"Sharpe ({m_v2['sharpe']:.3f} < {m37['sharpe']:.3f})")
        print(f"  VERDICT: 60-sym v2 FAILS — {' and '.join(fails)}")
        print(f"  Keep 37-sym baseline as production model.")
    print(f"{'='*78}")

    print(f"\n  Runtime: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
