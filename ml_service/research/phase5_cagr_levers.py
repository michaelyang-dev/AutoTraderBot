"""
"Are there more ways to increase CAGR?" — the accessible lever is AGGRESSIVENESS:
tilt to momentum (highest-returning sleeve) and/or concentrate (fewer names).
This is NOT free alpha — it's the same risk-for-return trade as leverage, just via
selection. Quantify the menu vs the deployed 50/35/15 n5 so the cost is explicit.
Selection-alpha levers (ML, factor blends, residual mom, frog-in-pan) already DEAD.
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
    dd = ((v - v.cummax()) / v.cummax()).min()
    cg = (v.iloc[-1] / v.iloc[0]) ** (252 / len(v)) - 1
    return cg, dd, sh


bt = FastBacktester(); clear(bt)
B = {"universe": "sp1500", "sec_w": 0.0, "cap": 0.15, "rebal_days": 20,
     "trailing_stop": 0.40, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

CONFIGS = [
    ("DEPLOYED  50/35/15  n5", {**B, "mom_w": .50, "val_w": .35, "lv_w": .15, "top_n": 5}),
    ("more mom  60/40       n5", {**B, "mom_w": .60, "val_w": .40, "lv_w": .00, "top_n": 5}),
    ("pure mom  100/0/0     n5", {**B, "mom_w": 1.0, "val_w": .00, "lv_w": .00, "top_n": 5}),
    ("concentr  50/35/15   n3", {**B, "mom_w": .50, "val_w": .35, "lv_w": .15, "top_n": 3}),
    ("aggr      60/40       n3", {**B, "mom_w": .60, "val_w": .40, "lv_w": .00, "top_n": 3}),
    ("pure mom  100/0/0     n3", {**B, "mom_w": 1.0, "val_w": .00, "lv_w": .00, "top_n": 3}),
]

print("=" * 64)
print("AGGRESSIVENESS MENU (1x, single-start 2018-2025; read deltas)")
print("=" * 64)
print(f"\n  {'config':<26}{'CAGR':>8}{'MaxDD':>9}{'Sharpe':>8}{'CAGR/DD':>9}")
for name, cfg in CONFIGS:
    cg, dd, sh = stats(bt.run("2018-01-02", "2025-12-31", cfg)["daily_values"])
    print(f"  {name:<26}{cg*100:>7.1f}%{dd*100:>8.1f}%{sh:>8.2f}{cg/abs(dd):>9.2f}", flush=True)
print("\n  CAGR/DD = return per unit of drawdown (higher = better trade). If aggressive")
print("  configs have HIGHER CAGR but LOWER CAGR/DD and Sharpe, the extra return is just")
print("  repriced risk, not new alpha — same deal as leverage, capped by your risk budget.")
