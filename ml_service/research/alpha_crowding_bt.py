"""
Idea 3 — book-fragility exposure overlay: CONFIRMATORY BACKTEST.
===============================================================
Diagnostic (alpha_crowding_diag.py) found book vol/stretch z-scores predict
forward DRAWDOWN (corr -0.40 / -0.30) but NOT forward return (corr ~+0.05). So
the overlay can only be a RISK tool, like NAV vol-scaling. The fair question:
does it cut drawdown BEYOND the vol_scaling + SPY<SMA200 the system already runs?

Configs:
  A  v12 raw (no tail tools)
  B  v12 + vol_scaling + trend_scale   <- the deployed honest bar
  C  B + book-fragility overlay        <- additive value test
  D  v12 + trend_scale + book overlay (book REPLACES vol_scaling)  <- is book a
                                          better tail tool than NAV vol-scaling?
plus a small grid over the overlay (use=vol/strch/both).

Run: cd ml_service && ./venv/bin/python research/alpha_crowding_bt.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_production_backtest import FastBacktester

BASE = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
            top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40)
TREND = {"bear": 0.40, "caution": 0.75}
VOLSC = dict(vol_scaling=True, vol_target=0.20, vol_lookback=40)


def cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


def line(name, r):
    if r is None:
        print(f"{name:<46} None"); return
    print(f"{name:<46} CAGR {r['cagr']*100:6.1f}%  Sharpe {r['sharpe']:.2f}  "
          f"Sortino {r['sortino']:.2f}  MaxDD {r['max_dd']*100:6.1f}%  Vol {r['vol']*100:4.1f}%")


def run_set(bt, start, end, tag):
    print(f"\n==== {tag} ({start[:4]}-{end[:4]}) ====")
    line("A v12 raw", bt.run(start, end, cfg()))
    line("B v12 + volscale + trend (DEPLOYED BAR)",
         bt.run(start, end, cfg(trend_scale=TREND, **VOLSC)))
    # additive overlay
    for use in ("vol", "strch", "both"):
        ov = {"use": use, "thr": 1.0, "factor": 0.6, "window": 24, "min_periods": 8}
        line(f"C B + book[{use}] thr1.0 f0.6",
             bt.run(start, end, cfg(trend_scale=TREND, book_crash_scale=ov, **VOLSC)))
    # book replaces vol-scaling
    for use in ("vol", "strch", "both"):
        ov = {"use": use, "thr": 1.0, "factor": 0.6, "window": 24, "min_periods": 8}
        line(f"D trend + book[{use}] (NO volscale)",
             bt.run(start, end, cfg(trend_scale=TREND, book_crash_scale=ov)))


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    print(f"[loaded in {time.time()-t0:.0f}s]")
    run_set(bt, "2016-01-01", "2025-12-31", "default universe")
