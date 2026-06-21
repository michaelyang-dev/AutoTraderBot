"""
Alpha research — baselines (the bars to beat).
================================================
Establishes two honest baselines on the WRDS survivorship-free universe so every
new idea is measured apples-to-apples:

  1. PURE MOMENTUM sleeve (mom_w=1.0, signal-prop, no risk-parity) — the bar for
     the position-aging idea (which only touches the momentum book).
  2. FULL v12 deployed config — the bar for dispersion-sizing & crowding-exposure
     ideas (which touch sleeve weight / market exposure).

Run:
    cd ml_service && ./venv/bin/python research/alpha_baseline.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_production_backtest import FastBacktester


def summary(name, r):
    if r is None:
        print(f"{name:<34} -> None")
        return
    y = r.get("yearly", {})
    yrs = " ".join(f"{k}:{v['cagr']*100:+.0f}" for k, v in sorted(y.items()))
    print(f"{name:<34} CAGR {r['cagr']*100:6.1f}%  Sharpe {r['sharpe']:.2f}  "
          f"MaxDD {r['max_dd']*100:6.1f}%  Vol {r['vol']*100:4.1f}%  alpha {r['alpha']*100:+5.1f}")
    print(f"{'':34} yearly: {yrs}")


PURE_MOM = dict(universe="sp1500", mom_w=1.0, val_w=0.0, lv_w=0.0, sec_w=0.0,
                top_n=5, cap=0.15, rebal_days=20, use_rp=False,
                trailing_stop=0.40)

# v12 deployed (shared PROD weights are the default in run(); set explicitly here)
V12 = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
           top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40)


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    print(f"\n[loaded in {time.time()-t0:.0f}s]\n")

    for start, end, tag in [("2018-01-01", "2025-12-31", "2018-2025"),
                            ("2001-01-01", "2025-12-31", "2001-2025 (through-cycle)")]:
        print(f"==== {tag} ====")
        t = time.time()
        summary("Pure momentum (top5, rd20, stop40)", bt.run(start, end, PURE_MOM))
        summary("Full v12 (50/35/15)", bt.run(start, end, V12))
        print(f"[{time.time()-t:.0f}s]\n")
