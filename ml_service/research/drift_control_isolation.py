"""
DRIFT-CONTROL ISOLATION — the DIRECT test of the question: with the stock SELECTION held
fixed (re-select every 20d, exactly like live), is it better to let position sizes DRIFT
(current: PAYC runs from 8.9% -> 10.4%) or to SNAP them back to target between rebalances?

Post-processed from the backtest's recorded 20d target weights + daily returns, so the
SELECTION is byte-identical across methods — ONLY the re-sizing frequency differs. Real
trading costs (COST_BPS, per-side) charged on the extra turnover, so snapping isn't free.
Methods: drift (snap only at 20d rebal) vs snap-to-target every 10d / 5d / 1d.

Weight-based sim (cash weight = 1 - sum(w) earns 0). No trailing stop — it applies equally
to every method (it's about exiting losers, orthogonal to sizing), so it cancels in the
comparison. The 'drift' method is validated against the backtest's own CAGR. Both periods.
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
METHODS = [("drift (CURRENT)", None), ("snap 10d", 10), ("snap 5d", 5), ("snap 1d", 1)]


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
    return (c.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min(), yrs)


def simulate(rebal_log, rets, dates, resize_every=None):
    """Weight-based drift sim. resize_every=None -> only snap at rebalances (drift)."""
    w = {}
    nav = 1.0
    navs = []
    cur = {}
    dsr = 0
    turnover = 0.0
    rebal_dates = set(rebal_log)
    for date in dates:
        r = rets.loc[date] if date in rets.index else None
        if r is not None and w:                       # drift weights with returns
            pr = sum(w[s] * (r.get(s, 0.0) or 0.0) for s in w)
            nav *= (1 + pr)
            if (1 + pr) > 0:
                w = {s: w[s] * (1 + (r.get(s, 0.0) or 0.0)) / (1 + pr) for s in w}
        is_rebal = date in rebal_dates
        if is_rebal:
            cur = rebal_log[date]; dsr = 0
        else:
            dsr += 1
        is_resize = is_rebal or (resize_every and cur and dsr > 0 and dsr % resize_every == 0)
        if (is_rebal or is_resize) and cur:           # snap to target, charge turnover cost
            to = sum(abs(cur.get(s, 0.0) - w.get(s, 0.0)) for s in set(list(w) + list(cur)))
            turnover += to
            nav *= (1 - to * COST)
            w = {s: v for s, v in cur.items() if v > 1e-6}
        navs.append((date, nav))
    return pd.Series({d: v for d, v in navs}), turnover


def main():
    t0 = time.time()
    for pname, path, starts, end in PERIODS:
        print("=" * 80, flush=True)
        print(f"PERIOD {pname} | {len(starts)}-start avg | cost {COST_BPS}bps/side | "
              f"selection FIXED (20d) — only re-sizing differs", flush=True)
        print("=" * 80, flush=True)
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        rets = bt.prices.pct_change().fillna(0.0)
        res = {n: [] for n, _ in METHODS}
        turns = {n: [] for n, _ in METHODS}
        bt_cagr = []
        for st in starts:
            bt._rebal_log = []
            m = bt.run(st, end, dict(V12))
            bt_cagr.append(stats(m["daily_values"])[0])
            rebal_log = {pd.Timestamp(d): wts for d, wts in bt._rebal_log}
            dates = list(m["daily_values"].index)
            for name, re in METHODS:
                nav, turn = simulate(rebal_log, rets, dates, resize_every=re)
                c, v, s, d, yrs = stats(nav)
                res[name].append((c, v, s, d)); turns[name].append(turn / yrs)
        hdr = f"{'method':<18}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'turnovr/yr':>11}"
        print(hdr); print("-" * len(hdr), flush=True)
        drift_cagr = None
        for name, _ in METHODS:
            a = np.array(res[name]); to = np.mean(turns[name])
            c, v, s, d = a.mean(axis=0)
            if name.startswith("drift"):
                drift_cagr = c
            tag = f"  ({(c-drift_cagr)*100:+.1f}pp)" if drift_cagr is not None and not name.startswith("drift") else ""
            print(f"{name:<18}{c:>+8.1%}{v:>7.1%}{s:>8.2f}{d:>+8.1%}{to:>10.1f}x{tag}", flush=True)
        print(f"[validation] sim-drift CAGR {np.mean([res['drift (CURRENT)'][i][0] for i in range(len(starts))]):+.1%} "
              f"vs backtest CAGR {np.mean(bt_cagr):+.1%} (close = sim faithful; gap = trailing stop)", flush=True)
        del bt
    print(f"\n({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
