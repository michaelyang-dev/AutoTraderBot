"""
Phase 2 — Parameter sensitivity. Vary ONE param at a time around the deployed
value. Robust strategy = flat plateau near the chosen value; a cliff (chosen
value much better than neighbors) = overfit to that exact setting.
Single start day (2018-01-02) for speed; comparison across cells is apples-to-apples.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(v):
    r = v.pct_change().dropna()
    sh = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    peak = v.cummax(); dd = ((v - peak) / peak).min()
    cg = (v.iloc[-1] / v.iloc[0]) ** (252 / len(v)) - 1
    return cg, dd, sh


bt = FastBacktester(); clear(bt)
DEP = {"universe": "sp1500", "mom_w": .50, "val_w": .35, "lv_w": .15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

sweeps = [
    ("trailing_stop", [0.30, 0.35, 0.40, 0.45, 0.50], 0.40),
    ("rebal_days",    [15, 20, 25, 30],               20),
    ("top_n",         [3, 5, 8, 10],                  5),
    ("cap",           [0.12, 0.15, 0.20, 0.25],       0.15),
]

print("=" * 60, flush=True)
print("PARAMETER SENSITIVITY (1x, single start) — plateau vs cliff?")
print("=" * 60)
for param, values, deployed_val in sweeps:
    print(f"\n  {param}  (deployed = {deployed_val}):")
    print(f"    {'value':>8}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}")
    for val in values:
        cfg = {**DEP, param: val}
        m = bt.run("2018-01-01", "2025-12-31", cfg)
        cg, dd, sh = stats(m["daily_values"])
        mark = "  <-- deployed" if val == deployed_val else ""
        print(f"    {val:>8}{cg*100:>7.1f}%{dd*100:>7.1f}%{sh:>8.2f}{mark}", flush=True)
print("\nRead: if the deployed row is a peak with much-worse neighbors = overfit/fragile.")
print("If neighbors are similar (flat plateau) = robust choice.")
