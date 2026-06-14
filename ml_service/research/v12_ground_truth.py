"""
v12 Ground Truth — loads universe ONCE, toggles data conditions.
Definitive deployed-v12 number + where the 25.4% claim came from.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from fast_backtest import FastBacktester
import numpy as np

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
STARTS = ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-08"]

bt = FastBacktester()

ENH = ["_fin_growth", "_ev", "_estimates", "_price_targets",
       "_revenue_surprise", "_beat_streak", "_earnings_signals"]
SAVE_UNI = {k: getattr(bt.uni, k, {}) for k in ENH + ["_short_interest_rank", "_si_change_rank"]}
SAVE_BT = {k: getattr(bt, k, None) for k in ["_si_ranks_by_month", "_si_change_ranks_by_month", "_si_months"]}


def restore():
    for k, v in SAVE_UNI.items():
        setattr(bt.uni, k, v)
    for k, v in SAVE_BT.items():
        setattr(bt, k, v)


def clear_enh():
    for k in ENH:
        setattr(bt.uni, k, {})


def clear_si():
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def avg():
    c, s, d = [], [], []
    for st in STARTS:
        m = bt.run(st, "2025-12-31", V12)
        if m:
            c.append(m["cagr"]); s.append(m["sharpe"]); d.append(m["max_dd"])
    return np.mean(c)*100, np.std(c)*100, np.mean(s), np.mean(d)*100


print("=" * 64, flush=True)
print("v12 GROUND TRUTH (one load, 4 data conditions, start-day avg)", flush=True)
print("=" * 64, flush=True)
for label, fns in [
    ("A. DEPLOYED (enhanced OFF, SI OFF)", [clear_enh, clear_si]),
    ("B. enhanced ON, SI OFF",            [clear_si]),
    ("C. enhanced OFF, SI ON",            [clear_enh]),
    ("D. EVERYTHING ON",                  []),
]:
    restore()
    for f in fns:
        f()
    cagr, std, shrp, dd = avg()
    print(f"\n  {label}\n     CAGR {cagr:5.1f}% +/- {std:.1f} | Sharpe {shrp:.2f} | MaxDD {dd:.1f}%", flush=True)
print("\n" + "=" * 64, flush=True)
print("README claims 25.4% / 1.00. Which condition matches?", flush=True)
