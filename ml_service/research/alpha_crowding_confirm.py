"""
Idea 3 — DECISIVE confirmatory test, matching v12_ground_truth methodology:
  - DEPLOYED data condition (enhanced data OFF, short-interest OFF)
  - start-day averaged over several phases (kills rebalance-timing luck)
Compares the deployed tail bar (B) vs adding the book-fragility overlay (C).
A real win must beat B on Sharpe by MORE than the start-day std, at no CAGR cost.

Run: cd ml_service && ./venv/bin/python research/alpha_crowding_confirm.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_production_backtest import FastBacktester

BASE = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
            top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
            bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10})
TREND = {"bear": 0.40, "caution": 0.75}
VOLSC = dict(vol_scaling=True, vol_target=0.20, vol_lookback=40)
STARTS = ["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07", "2016-01-08",
          "2016-01-11", "2016-01-12", "2016-01-13"]
END = "2025-12-31"


def cfg(**kw):
    c = dict(BASE); c.update(kw); return c


def avg(bt, c):
    cc, ss, dd, so = [], [], [], []
    for st in STARTS:
        m = bt.run(st, END, c)
        if m:
            cc.append(m["cagr"]); ss.append(m["sharpe"]); dd.append(m["max_dd"]); so.append(m["sortino"])
    return (np.mean(cc)*100, np.std(cc)*100, np.mean(ss), np.std(ss),
            np.mean(so), np.mean(dd)*100)


def show(name, t):
    print(f"{name:<40} CAGR {t[0]:5.1f}% +/-{t[1]:.1f}  Sharpe {t[2]:.2f} +/-{t[3]:.2f}  "
          f"Sortino {t[4]:.2f}  MaxDD {t[5]:5.1f}%")


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    # DEPLOYED data condition: clear enhanced data + short interest
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals",
              "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    print(f"[loaded in {time.time()-t0:.0f}s; enhanced+SI cleared; {len(STARTS)} start days]\n")

    show("B  deployed bar (volscale+trend)", avg(bt, cfg(trend_scale=TREND, **VOLSC)))
    for use in ("vol", "strch", "both"):
        ov = {"use": use, "thr": 1.0, "factor": 0.6, "window": 24, "min_periods": 8}
        show(f"C  B + book[{use}] thr1.0 f0.6",
             avg(bt, cfg(trend_scale=TREND, book_crash_scale=ov, **VOLSC)))
    # a slightly more aggressive cut on stretch (the orthogonal-to-vol signal)
    for thr, f in [(0.75, 0.5), (1.25, 0.5)]:
        ov = {"use": "strch", "thr": thr, "factor": f, "window": 24, "min_periods": 8}
        show(f"C  B + book[strch] thr{thr} f{f}",
             avg(bt, cfg(trend_scale=TREND, book_crash_scale=ov, **VOLSC)))
