"""
REBALANCE FREQUENCY test — does letting positions DRIFT (current: re-select + resize only
every 20 trading days) beat resizing more often?

Sweeps rebal_days on the live v12 config, both periods, multi-start. Shorter rebal_days =
LESS drift (positions snapped back to target more often) + fresher selection, but MORE
turnover (real costs applied by the backtest). Longer = MORE drift, less cost. If the
short cadences win, snapping-to-target helps; if 20d+ wins, drift helps (momentum lets
winners run). vol-scaling OFF to isolate the frequency effect. Reports CAGR/Sharpe/MaxDD
+ annual turnover. Both periods incl. 2008+2020; a real answer holds in BOTH.
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

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
REBAL_DAYS = [5, 10, 20, 40, 60]
BASE_L, RATE = 1.49, 0.063   # deployed leverage + IBKR small-acct financing (on borrowed)


def lever(r1x):
    """Constant 1.49x + financing on the borrowed 0.49x (no vol-scaling — matches the 1x
    isolation; leverage should scale every cadence ~equally, so the RANKING is preserved)."""
    fin = max(0.0, BASE_L - 1.0) * RATE / 252
    return BASE_L * r1x - fin


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(vals):
    r = vals.pct_change().dropna()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    c = (1 + r).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min(), yrs)


def main():
    t0 = time.time()
    for pname, path, starts, end in PERIODS:
        print("=" * 78, flush=True)
        print(f"PERIOD {pname} | {len(starts)}-start avg | costs ON | vol-scaling OFF", flush=True)
        print("=" * 78, flush=True)
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        hdr = (f"{'rebal':<7}{'drift':<8}| {'1x CAGR':>8}{'1x Shrp':>8} | "
               f"{'1.49x CAGR':>11}{'1.49x Shrp':>11}{'1.49x DD':>9}{'turn/yr':>9}")
        print(hdr); print("-" * len(hdr), flush=True)
        for rd in REBAL_DAYS:
            cfg = dict(V12); cfg["rebal_days"] = rd
            c1s, s1s, cls, sls, dls, tos = [], [], [], [], [], []
            for st in starts:
                bt._gross_traded = 0.0
                m = bt.run(st, end, cfg)
                r1x = m["daily_values"].pct_change().dropna()
                yrs = (r1x.index[-1] - r1x.index[0]).days / 365.25
                c1 = (1 + r1x).cumprod().iloc[-1] ** (1 / yrs) - 1
                s1 = (r1x.mean() / r1x.std() * np.sqrt(252)) if r1x.std() > 0 else 0
                rl = lever(r1x); cl = (1 + rl).cumprod()
                clv = cl.iloc[-1] ** (1 / yrs) - 1
                slv = (rl.mean() / rl.std() * np.sqrt(252)) if rl.std() > 0 else 0
                dlv = ((cl - cl.cummax()) / cl.cummax()).min()
                mean_nav = float(m["daily_values"].mean())
                to = (getattr(bt, "_gross_traded", 0.0) / yrs / mean_nav) if mean_nav else 0
                c1s.append(c1); s1s.append(s1); cls.append(clv); sls.append(slv)
                dls.append(dlv); tos.append(to)
            drift = {5: "least", 10: "less", 20: "CURRENT", 40: "more", 60: "most"}.get(rd, "")
            print(f"{rd:<7}{drift:<8}| {np.mean(c1s):>+8.1%}{np.mean(s1s):>8.2f} | "
                  f"{np.mean(cls):>+11.1%}{np.mean(sls):>11.2f}{np.mean(dls):>+9.1%}"
                  f"{np.mean(tos):>8.1f}x", flush=True)
        del bt
    print(f"\n(shorter rebal_days = less drift + more cost; 20 = current live/backtest. "
          f"{time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
