"""
Phase 1.4 (rigorous) — Vol-scaling vs current, START-DAY AVERAGED (5 offsets).
Gives honest absolute CAGR levels (~22% baseline), not single-start luck.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np

LEV = 1.49
STARTS = ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-08"]


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(vals, L=1.0):
    r = vals.pct_change().fillna(0.0)
    if L != 1.0:
        r = L * r - (L - 1) * 0.055 / 252
        vals = (1 + r).cumprod()
    peak = vals.cummax(); maxdd = ((vals - peak) / peak).min()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (252 / len(vals)) - 1
    m20 = vals.iloc[::20].pct_change().dropna()
    cvar = m20[m20 <= m20.quantile(0.05)].mean()
    return cagr, maxdd, cvar


bt = FastBacktester()
clear(bt)
base = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
        "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
configs = [
    ("CURRENT (no vol-scaling)", base),
    ("vol-scaling 20% target", {**base, "vol_scaling": True, "vol_target": 0.20}),
    ("vol-scaling 15% target", {**base, "vol_scaling": True, "vol_target": 0.15}),
]

print("=" * 78, flush=True)
print("VOL-SCALING vs CURRENT — START-DAY AVERAGED (5 offsets), 2018-2025, n=5")
print("=" * 78)
print(f"\n{'config':<26}{'CAGR 1x':>9}{'MaxDD 1x':>10}{'CVaR 1x':>9} | {'CAGR 1.49x':>11}{'MaxDD 1.49x':>12}")
print("-" * 78)
results = {}
for label, cfg in configs:
    c1s, dd1s, cv1s, cLs, ddLs = [], [], [], [], []
    for st in STARTS:
        m = bt.run(st, "2025-12-31", cfg)
        v = m["daily_values"]
        c1, dd1, cv1 = stats(v, 1.0); cL, ddL, _ = stats(v, LEV)
        c1s.append(c1); dd1s.append(dd1); cv1s.append(cv1); cLs.append(cL); ddLs.append(ddL)
    r = (np.mean(c1s), np.mean(dd1s), np.mean(cv1s), np.mean(cLs), np.mean(ddLs))
    results[label] = r
    print(f"{label:<26}{r[0]*100:>8.1f}%{r[1]*100:>9.1f}%{r[2]*100:>8.1f}% | {r[3]*100:>10.1f}%{r[4]*100:>11.1f}%", flush=True)

cur = results["CURRENT (no vol-scaling)"]
v15 = results["vol-scaling 15% target"]
print(f"\n15% vs CURRENT:  CAGR {(v15[0]-cur[0])*100:+.1f}pp | MaxDD@1x {(v15[1]-cur[1])*100:+.1f}pp | "
      f"MaxDD@1.49x {(v15[4]-cur[4])*100:+.1f}pp | CVaR {(v15[2]-cur[2])*100:+.1f}pp")
