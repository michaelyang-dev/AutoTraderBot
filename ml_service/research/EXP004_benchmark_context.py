"""EXP-004 — the benchmark context every number in this repo has been missing.

The 26yr shuffle result (random picks through the deployed chassis: +6.36% / 0.37) only means
something against a passive alternative. Nowhere in this repo is the deployed strategy compared
to LEVERED SPY on matched terms -- same leverage, same 6.3% financing on the debit, same
period. Without that the whole program could be sophisticated beta.

Computed here, on the same price matrix the backtests use, so there is no data mismatch:

  1. SPY buy-and-hold, unlevered
  2. SPY at 1.49x, financing the 0.49x debit at 6.3%/yr, daily
  3. SPY at 1.49x with the SAME inverse-vol overlay the strategy runs (40d realised vol,
     target 0.15*1.49, clamp [0.30, 1.00]) -- i.e. the chassis applied to pure beta
  4. equal-weight SP1500 (PIT membership), unlevered -- the honest "random picks" ceiling

Every series is close-to-close on the identical calendar, so differences are attributable.

Run:  python3 research/EXP004_benchmark_context.py
"""
import os
import sys
import pickle

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIN = 0.063
LEV = 1.49
VOL_TARGET = 0.15 * LEV
LB = 40


def stat(v):
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
            dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0,
            ((v - v.cummax()) / v.cummax()).min())


def levered(rets, lev_series):
    """NAV path: gross = lev*NAV in SPY, debit = (lev-1)*NAV financed daily."""
    nav = 1.0
    out = []
    for r, L in zip(rets.values, lev_series.values):
        nav = nav * (1 + L * r) - nav * max(L - 1.0, 0.0) * (FIN / 252.0)
        out.append(max(nav, 1e-9))
    return pd.Series(out, index=rets.index)


def run(pkl, label, start, end):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    px = d["prices_df"]
    mem = {k: d[k] for k in ("sp500_mem", "sp400_mem", "sp600_mem")}
    idx = [x for x in px.index if pd.Timestamp(start) <= x <= pd.Timestamp(end)]
    spy = px["SPY"].reindex(idx).ffill().dropna()
    r = spy.pct_change().dropna()

    print(f"\n  === {label}  {r.index[0].date()} -> {r.index[-1].date()} ===", flush=True)
    print(f"  {'benchmark':<44}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}", flush=True)

    rows = []
    rows.append(("SPY buy & hold (unlevered)", levered(r, pd.Series(1.0, index=r.index))))
    rows.append((f"SPY {LEV}x, {FIN:.1%} financing",
                 levered(r, pd.Series(LEV, index=r.index))))

    # chassis on beta: same inverse-vol overlay the strategy uses, causal (shift 1)
    rv = r.rolling(LB).std() * np.sqrt(252)
    vs = (VOL_TARGET / rv).clip(0.30, 1.00).shift(1).fillna(1.0)
    rows.append((f"SPY {LEV}x + strategy's inverse-vol overlay",
                 levered(r, LEV * vs)))

    # ---- equal-weight SP1500, PIT membership --------------------------------------
    # THE FIRST VERSION OF THIS WAS AN INFLATED BENCHMARK and I nearly reported off it.
    # Taking a simple nanmean of daily returns each day is a DAILY-rebalanced equal-weight
    # portfolio. Daily rebalancing across 1,500 names harvests a large volatility/rebalancing
    # premium and is completely unimplementable -- it would trade the entire book every
    # session. A benchmark must be something a person could actually have held.
    # Corrected: reconstitute on schedule, then BUY AND HOLD in between, so weights drift
    # with prices exactly as they would in a real account. Names whose price disappears are
    # held flat and dropped at the next reconstitution -- the same convention the strategy
    # harness uses, so the two remain comparable.
    import bisect
    sub = px.reindex(idx)
    dr = sub.pct_change(fill_method=None)
    keys = {k: sorted(v.keys()) for k, v in mem.items()}

    def ew_series(freq):
        w, cur, out, last_p = None, [], [], None
        for i, dt in enumerate(sub.index):
            p = (dt.year, dt.month) if freq == "M" else (dt.year, (dt.month - 1) // 3)
            if p != last_p:
                names = set()
                for k, v in mem.items():
                    kk = keys[k]
                    j = bisect.bisect_right(kk, dt) - 1
                    if j >= 0:
                        names |= set(v[kk[j]])
                row0 = sub.iloc[i]
                cur = [x for x in names if x in sub.columns and np.isfinite(row0.get(x, np.nan))]
                w = np.full(len(cur), 1.0 / len(cur)) if cur else None
                last_p = p
            if i == 0 or w is None or not len(cur):
                out.append(0.0)
                continue
            r_i = dr.iloc[i].reindex(cur).values
            r_i = np.where(np.isfinite(r_i), r_i, 0.0)
            out.append(float((w * r_i).sum()))
            w = w * (1.0 + r_i)
            t = w.sum()
            if t > 0:
                w = w / t
        return pd.Series(out, index=sub.index).fillna(0.0)

    ewm, ewq = ew_series("M"), ew_series("Q")
    rows.append(("SP1500 EW PIT, monthly reconst., buy&hold", (1 + ewm).cumprod()))
    rows.append(("SP1500 EW PIT, quarterly reconst., buy&hold", (1 + ewq).cumprod()))
    rows.append((f"SP1500 EW quarterly {LEV}x, {FIN:.1%} financing",
                 levered(ewq, pd.Series(LEV, index=ewq.index))))


    for name, v in rows:
        c, s, dd = stat(v)
        print(f"  {name:<44}{c:>+9.2%}{s:>8.2f}{dd:>9.1%}", flush=True)


if __name__ == "__main__":
    run("data/wrds/complete_sp1500_universe.pkl", "8yr 2018-2025", "2018-01-03", "2025-12-31")
    run("data/wrds/sp1500_universe_2000.pkl", "26yr 2001-2025", "2001-01-03", "2025-12-31")
    print("\n  Strategy for comparison (12-start audited means):", flush=True)
    print("    8yr  +22.75% / 0.79 / -38.2%     26yr  +11.26% / 0.51 / -64.8%", flush=True)
