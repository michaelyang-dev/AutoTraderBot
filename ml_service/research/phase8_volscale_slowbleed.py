"""
Phase 8 — Does vol-scaling help in SLOW-BLEED bears, or only sharp vol spikes?
The vol-scaling validation (-59->-47 GFC, -46->-31 COVID) was on two SHARP vol-spike
crashes — exactly what realized-vol scaling is built for. The honest test is the
slow-bleed regimes (esp. 2022: steady ~-25%, VIX mostly 20s-30s, no spike), where
realized-vol scaling may stay risk-on and add nothing.

Per window: drawdown WITHOUT vs WITH vol-scaling + the strategy's realized vol
(does it even cross the de-risk target?). Plus full-period Sharpe/Calmar AFTER costs
— a drawdown cut that costs CAGR may lose on risk-adjusted terms.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


bt = FastBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl"); clear(bt)
DEP = {"universe": "sp1500", "mom_w": .50, "val_w": .35, "lv_w": .15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

print("running base (no vol-scaling) + vol-scaled (target 15%)...", flush=True)
base = bt.run("2000-01-03", "2025-12-31", {**DEP, "vol_scaling": False})["daily_values"]
vs = bt.run("2000-01-03", "2025-12-31", {**DEP, "vol_scaling": True, "vol_target": 0.15})["daily_values"]


def win(v, a, b):
    return v[(v.index >= pd.Timestamp(a)) & (v.index <= pd.Timestamp(b))]


def dd(v):
    return ((v - v.cummax()) / v.cummax()).min()


def realized_vol(v):
    r = v.pct_change().dropna()
    return r.std() * np.sqrt(252) if len(r) > 2 else float("nan")


WINDOWS = [
    ("2011 EU/debt", "2011-07-01", "2011-10-31", "slow"),
    ("2015-16 corr", "2015-08-01", "2016-02-29", "slow"),
    ("2018 Q4", "2018-10-01", "2018-12-31", "slow"),
    ("2022 bear", "2022-01-01", "2022-12-31", "SLOW*"),
    ("GFC 08-09", "2007-10-01", "2009-03-31", "fast"),
    ("COVID 2020", "2020-02-01", "2020-04-30", "fast"),
]

print("\n" + "=" * 74)
print("VOL-SCALING IN SLOW-BLEED vs SHARP CRASHES (1x, 2000-2025, after 10bps costs)")
print("=" * 74)
print(f"\n  {'window':<14}{'type':>6}{'base DD':>9}{'volsc DD':>10}{'help':>7}{'strat vol':>11}")
for name, a, b, typ in WINDOWS:
    wb, wv = win(base, a, b), win(vs, a, b)
    ddb, ddv = dd(wb) * 100, dd(wv) * 100
    rv = realized_vol(wb) * 100
    help_pp = ddv - ddb   # positive = vol-scaling reduced the drawdown
    print(f"  {name:<14}{typ:>6}{ddb:>8.1f}%{ddv:>9.1f}%{help_pp:>+6.1f}{rv:>10.0f}%", flush=True)


def full(v):
    r = v.pct_change().dropna()
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    mdd = dd(v)
    calmar = cagr / abs(mdd) if mdd else 0
    return cagr, sh, mdd, calmar


print(f"\n  FULL PERIOD 2000-2025 (the cost-side check):")
print(f"  {'':<14}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}")
for lab, v in [("no vol-scale", base), ("vol-scaled", vs)]:
    cg, sh, md, ca = full(v)
    print(f"  {lab:<14}{cg*100:>7.1f}%{sh:>8.2f}{md*100:>8.1f}%{ca:>8.2f}")
print("\n  Read: 'help' ~0 in slow bleeds (esp 2022*) confirms the blind spot — vol-scaling")
print("  rides the bleed down. Big 'help' in GFC/COVID is the spike it's built for.")
print("  If vol-scaled Sharpe/Calmar <= base, the drawdown cut isn't worth the CAGR.")
