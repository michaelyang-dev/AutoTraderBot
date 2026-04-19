"""
Threshold sweep for the 60-symbol v2 model.
Tests multiple thresholds across pure ML and ML+SPY idle configurations.
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
PRED_FILE = DATA / "predictions_60universe_v2.parquet"

THRESHOLDS = [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65]

METRICS_COLS = [
    ("CAGR",          "cagr",           lambda v: f"{v:+.2%}"),
    ("Sharpe",        "sharpe",         lambda v: f"{v:.3f}"),
    ("Sortino",       "sortino",        lambda v: f"{v:.3f}"),
    ("Max DD",        "max_dd",         lambda v: f"{v:.1%}"),
    ("Win Rate",      "win_rate",       lambda v: f"{v:.1%}"),
    ("Profit F.",     "profit_factor",  lambda v: f"{v:.2f}" if np.isfinite(v) else "inf"),
    ("Avg Trade",     "avg_trade_ret",  lambda v: f"{v:+.2%}"),
    ("Trades",        "n_trades",       lambda v: f"{v:,}"),
    ("Alpha",         "alpha",          lambda v: f"{v:+.2%}"),
    ("Final $",       "final_value",    lambda v: f"${v:,.0f}"),
]


def load_preds():
    df = pd.read_parquet(PRED_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])
    print(f"  Loaded {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def build_sigs(df):
    sigs = {}
    for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
        g = grp.sort_values("prob", ascending=False)
        sigs[date] = list(g[["symbol", "prob", "fwd_ret"]].itertuples(index=False, name=None))
    return sigs


def run_sweep(sigs, all_dates, years, spy_bh, spy_dict, use_spy_idle):
    results = []
    for thresh in THRESHOLDS:
        signal_fn = make_ml_signal_fn(thresh)
        kw = dict(spy_prices=spy_dict) if use_spy_idle else {}
        vals, trades = run_simulation(
            sigs, all_dates, signal_fn, years,
            label=f">{thresh:.2f}", verbose=False, **kw)
        m = calc_metrics(vals, trades, years, f">{thresh:.2f}")
        a, b = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
        m["alpha"] = a
        results.append((thresh, m))
    return results


def print_table(results, title):
    col0 = 10
    col = 12

    w = len(title) + 6
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")

    header = f"  {'Thresh':<{col0}}" + "".join(f"{name:<{col}}" for name, _, _ in METRICS_COLS)
    print(header)
    print(f"  {'─'*(col0-1)}" + "".join(f" {'─'*(col-2)} " for _ in METRICS_COLS))

    for thresh, m in results:
        row = f"  >{thresh:<{col0-1}.2f}"
        for _, key, fmt in METRICS_COLS:
            row += f"{fmt(m[key]):<{col}}"
        print(row)

    # Highlight best per metric
    print(f"\n  {'BEST VALUES':}")
    print(f"  {'─'*60}")
    for name, key, fmt in METRICS_COLS:
        if key in ("n_trades", "final_value"):
            continue
        vals = [(t, m[key]) for t, m in results]
        if key == "max_dd":
            best_t, best_v = max(vals, key=lambda x: x[1])  # least negative
        else:
            best_t, best_v = max(vals, key=lambda x: x[1])
        print(f"    {name:<15} {fmt(best_v):<12} @ threshold >{best_t:.2f}")


def main():
    t0 = time.perf_counter()
    print("=" * 75)
    print("  THRESHOLD SWEEP — 60-Symbol v2 Model")
    print("=" * 75)

    df = load_preds()
    sigs = build_sigs(df)
    all_dates = sorted(df["date"].unique())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    # Fetch SPY
    start = pd.Timestamp(all_dates[0]).strftime("%Y-%m-%d")
    end = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close = fetch_benchmarks(start, end, sorted(df["symbol"].unique()))
    close = close.reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates]), method="ffill")
    spy_px = close["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close["SPY"].to_dict()
    m_spy = calc_metrics(spy_bh, [], years, "SPY B&H")

    print(f"\n  SPY Buy & Hold: CAGR {m_spy['cagr']:+.2%} | Sharpe {m_spy['sharpe']:.3f}")

    # Pure ML sweep
    print(f"\n  Running PURE ML sweep ...")
    pure = run_sweep(sigs, all_dates, years, spy_bh, spy_dict, use_spy_idle=False)
    print_table(pure, "PURE ML — THRESHOLD SWEEP (no SPY idle)")

    # ML + SPY idle sweep
    print(f"\n  Running ML + SPY IDLE sweep ...")
    spy_idle = run_sweep(sigs, all_dates, years, spy_bh, spy_dict, use_spy_idle=True)
    print_table(spy_idle, "ML + SPY IDLE — THRESHOLD SWEEP")

    # Summary: which threshold to pick
    print(f"\n{'='*75}")
    print("  OPTIMAL THRESHOLD RECOMMENDATIONS")
    print(f"{'='*75}")

    for label, results in [("Pure ML", pure), ("ML + SPY Idle", spy_idle)]:
        print(f"\n  [{label}]")

        best_cagr_t, best_cagr_m = max(results, key=lambda x: x[1]["cagr"])
        best_sharpe_t, best_sharpe_m = max(results, key=lambda x: x[1]["sharpe"])

        # Risk-adjusted: maximize (CAGR / -MaxDD) — return per unit of risk
        best_risk_t, best_risk_m = max(
            results,
            key=lambda x: x[1]["cagr"] / max(-x[1]["max_dd"], 0.01))

        # Composite score: normalize CAGR, Sharpe, Sortino, then average
        cagrs = [m["cagr"] for _, m in results]
        sharpes = [m["sharpe"] for _, m in results]
        sortinos = [m["sortino"] for _, m in results]
        c_range = max(cagrs) - min(cagrs) or 1
        s_range = max(sharpes) - min(sharpes) or 1
        so_range = max(sortinos) - min(sortinos) or 1

        best_composite_score = -999
        best_composite_t = None
        best_composite_m = None
        for t, m in results:
            score = ((m["cagr"] - min(cagrs)) / c_range * 0.4 +
                     (m["sharpe"] - min(sharpes)) / s_range * 0.3 +
                     (m["sortino"] - min(sortinos)) / so_range * 0.3)
            if score > best_composite_score:
                best_composite_score = score
                best_composite_t = t
                best_composite_m = m

        print(f"    Max CAGR          : >{best_cagr_t:.2f}  "
              f"(CAGR {best_cagr_m['cagr']:+.2%}, Sharpe {best_cagr_m['sharpe']:.3f}, "
              f"WR {best_cagr_m['win_rate']:.1%}, {best_cagr_m['n_trades']} trades)")
        print(f"    Max Sharpe        : >{best_sharpe_t:.2f}  "
              f"(CAGR {best_sharpe_m['cagr']:+.2%}, Sharpe {best_sharpe_m['sharpe']:.3f}, "
              f"WR {best_sharpe_m['win_rate']:.1%}, {best_sharpe_m['n_trades']} trades)")
        print(f"    Best risk-adj     : >{best_risk_t:.2f}  "
              f"(CAGR {best_risk_m['cagr']:+.2%}, MaxDD {best_risk_m['max_dd']:.1%}, "
              f"CAGR/DD {best_risk_m['cagr']/max(-best_risk_m['max_dd'],0.01):.2f})")
        print(f"    Best composite    : >{best_composite_t:.2f}  "
              f"(40% CAGR + 30% Sharpe + 30% Sortino, score={best_composite_score:.3f})")

    print(f"\n{'='*75}")
    print(f"  Runtime: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
