"""
EXECUTION-TIMING SENSITIVITY — how much does the strategy lose per trading day of delay
between the signal (prior close) and the actual trade? This brackets the open-vs-close
question: the live fills at the 9:30 OPEN (~0.7 sessions after the prior-close signal), the
backtest assumes the same-bar CLOSE (0 delay, a mild look-ahead). We have close-only data,
so we can't price the open directly — but if the strategy barely moves with a 1-day
execution delay, then open-vs-close (a fraction of a day) is a non-issue; if it bleeds
meaningfully per day, prompt (open) execution matters.

Post-processed from the backtest's recorded 20d target weights + daily returns: the SAME
targets are switched into `exec_lag` trading days later (0 = idealized same-bar, the current
backtest; 1 = execute next close; etc.). Real turnover cost. exec_lag=0 validated against
the backtest. Both periods incl. 2008+2020.
"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from main_production_backtest import FastBacktester  # noqa: E402
try:
    from main_production_backtest import COST_BPS
except ImportError:
    from strategies.multi_strategy_engine import COST_BPS
COST = COST_BPS / 10000.0

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
       "record_targets": True}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
LAGS = [0, 1, 2, 3, 5]
BASE_L, RATE = 1.49, 0.063


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(nav):
    r = nav.pct_change().dropna()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    c = (1 + r).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1,
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min())


def lever_stats(nav):
    """Apply constant 1.49x + financing to the 1x nav series, return (cagr, sharpe, dd)."""
    r1 = nav.pct_change().dropna()
    fin = max(0.0, BASE_L - 1.0) * RATE / 252
    rl = BASE_L * r1 - fin
    yrs = (rl.index[-1] - rl.index[0]).days / 365.25
    c = (1 + rl).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1,
            (rl.mean() / rl.std() * np.sqrt(252)) if rl.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min())


def simulate(rebal_log, rets, dates, exec_lag=0):
    """Switch to each rebalance's target `exec_lag` trading days later. Weight-based drift."""
    idx = {d: i for i, d in enumerate(dates)}
    switches = {}
    for rd, tgt in rebal_log.items():
        if rd in idx and idx[rd] + exec_lag < len(dates):
            switches[dates[idx[rd] + exec_lag]] = tgt
    w = {}
    nav = 1.0
    navs = []
    for date in dates:
        r = rets.loc[date] if date in rets.index else None
        if r is not None and w:
            pr = sum(w[s] * (r.get(s, 0.0) or 0.0) for s in w)
            nav *= (1 + pr)
            if (1 + pr) > 0:
                w = {s: w[s] * (1 + (r.get(s, 0.0) or 0.0)) / (1 + pr) for s in w}
        if date in switches:
            tgt = switches[date]
            to = sum(abs(tgt.get(s, 0.0) - w.get(s, 0.0)) for s in set(list(w) + list(tgt)))
            nav *= (1 - to * COST)
            w = {s: v for s, v in tgt.items() if v > 1e-6}
        navs.append((date, nav))
    return pd.Series({d: v for d, v in navs})


def main():
    t0 = time.time()
    for pname, path, starts, end in PERIODS:
        print("=" * 82, flush=True)
        print(f"PERIOD {pname} | {len(starts)}-start avg | signal fixed, execution delayed N days", flush=True)
        print("=" * 82, flush=True)
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        rets = bt.prices.pct_change().fillna(0.0)
        res1, resL = {n: [] for n in LAGS}, {n: [] for n in LAGS}
        bt_cagr = []
        for st in starts:
            bt._rebal_log = []
            m = bt.run(st, end, dict(V12))
            bt_cagr.append(stats(m["daily_values"])[0])
            rebal_log = {pd.Timestamp(d): wts for d, wts in bt._rebal_log}
            dates = list(m["daily_values"].index)
            for lag in LAGS:
                nav = simulate(rebal_log, rets, dates, exec_lag=lag)
                res1[lag].append(stats(nav))
                resL[lag].append(lever_stats(nav))
        hdr = (f"{'exec delay':<20}| {'1x CAGR':>8}{'1x Shrp':>8} | "
               f"{'1.49x CAGR':>11}{'1.49x Shrp':>11}{'1.49x DD':>9}{'vs same-bar':>12}")
        print(hdr); print("-" * len(hdr), flush=True)
        base1 = None
        for lag in LAGS:
            a1 = np.array(res1[lag]).mean(axis=0)
            aL = np.array(resL[lag]).mean(axis=0)
            if lag == 0:
                base1 = a1[0]
            label = {0: "0 = SAME-BAR close", 1: "1 day (next close)", 2: "2 days",
                     3: "3 days", 5: "5 days"}[lag]
            delta = "" if lag == 0 else f"{(a1[0]-base1)*100:>+7.1f}pp"
            print(f"{label:<20}| {a1[0]:>+8.1%}{a1[1]:>8.2f} | "
                  f"{aL[0]:>+11.1%}{aL[1]:>11.2f}{aL[2]:>+9.1%}{delta:>12}", flush=True)
        print(f"[validation] sim same-bar 1x CAGR {np.mean([res1[0][i][0] for i in range(len(starts))]):+.1%} "
              f"vs backtest {np.mean(bt_cagr):+.1%} (gap = trailing stop)", flush=True)
        print(f"  -> per-day bleed ≈ (same-bar - 1day)/1; the 9:30 OPEN is ~0.7 day, "
              f"the 4pm CLOSE ~1.0 day after the prior-close signal", flush=True)
        del bt
    print(f"\n({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
