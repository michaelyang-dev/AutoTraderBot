"""
Rigorous test: 60/40 mom/val vs deployed 50/35/15, over 26 YEARS (2000-2025).
Why this matters:
  - 2000-2017 is genuine OUT-OF-SAMPLE (deployed config was tuned on 2018-2025).
  - It contains the dot-com bust (2000-02) and GFC (2008) — the sustained bears the
    recent sample lacks, which is exactly where the 15% low-vol sleeve earns its keep.
If 60/40 (no low-vol) still wins risk-adjusted across these crashes -> robust upgrade.
If it blows up in 2000-02/2008 -> the deployed config's caution is vindicated.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

bt = FastBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl")
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
          "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
    pass  # keep enhanced data empty? NO — long-history pkl has its own; leave as loaded

B = {"universe": "sp1500", "sec_w": 0.0, "cap": 0.15, "rebal_days": 20,
     "trailing_stop": 0.40, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
CONFIGS = {
    "DEPLOYED 50/35/15 n5": {**B, "mom_w": .50, "val_w": .35, "lv_w": .15, "top_n": 5},
    "60/40 mom/val   n5":   {**B, "mom_w": .60, "val_w": .40, "lv_w": .00, "top_n": 5},
    "pure mom 100    n5":   {**B, "mom_w": 1.0, "val_w": .00, "lv_w": .00, "top_n": 5},
}
STARTS = ["2000-01-03", "2000-01-04", "2000-01-05"]


def metrics(v):
    r = v.pct_change().dropna()
    sh = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    dd = ((v - v.cummax()) / v.cummax()).min()
    cg = (v.iloc[-1] / v.iloc[0]) ** (252 / len(v)) - 1
    return cg, dd, sh


def window(v, a, b):
    return v[(v.index >= pd.Timestamp(a)) & (v.index <= pd.Timestamp(b))]


# run each config: 3 start days, cache canonical (first) full series
series = {}
avg = {}
for name, cfg in CONFIGS.items():
    cgs, dds, shs = [], [], []
    for i, st in enumerate(STARTS):
        v = bt.run(st, "2025-12-31", cfg)["daily_values"]
        if i == 0:
            series[name] = v
        cg, dd, sh = metrics(v)
        cgs.append(cg); dds.append(dd); shs.append(sh)
    avg[name] = (np.mean(cgs), np.mean(dds), np.mean(shs))

print("=" * 70)
print("60/40 vs DEPLOYED — 26-YEAR TEST (2000-2025, 3-start avg)")
print("=" * 70)
print(f"\n  FULL PERIOD 2000-2025:")
print(f"  {'config':<24}{'CAGR':>8}{'MaxDD':>9}{'Sharpe':>8}")
for name in CONFIGS:
    cg, dd, sh = avg[name]
    print(f"  {name:<24}{cg*100:>7.1f}%{dd*100:>8.1f}%{sh:>8.2f}")

print(f"\n  SUB-PERIODS (canonical start):")
for a, b, lab in [("2000-01-01", "2017-12-31", "2000-2017  OUT-OF-SAMPLE"),
                  ("2018-01-01", "2025-12-31", "2018-2025  in-sample")]:
    print(f"\n    {lab}:")
    print(f"    {'config':<24}{'CAGR':>8}{'MaxDD':>9}{'Sharpe':>8}")
    for name in CONFIGS:
        cg, dd, sh = metrics(window(series[name], a, b))
        print(f"    {name:<24}{cg*100:>7.1f}%{dd*100:>8.1f}%{sh:>8.2f}")

print(f"\n  CRASH-WINDOW DRAWDOWNS (does dropping low-vol hurt in real bears?):")
print(f"    {'window':<22}", end="")
for name in CONFIGS:
    print(f"{name.split()[0][:8]:>11}", end="")
print()
for a, b, lab in [("2000-03-01", "2002-10-31", "dot-com 00-02"),
                  ("2007-10-01", "2009-03-31", "GFC 08-09"),
                  ("2020-02-01", "2020-04-30", "COVID 2020"),
                  ("2022-01-01", "2022-12-31", "2022 bear")]:
    print(f"    {lab:<22}", end="")
    for name in CONFIGS:
        w = window(series[name], a, b)
        dd = ((w - w.cummax()) / w.cummax()).min() if len(w) > 1 else 0
        print(f"{dd*100:>10.1f}%", end="")
    print()
print("\n  Verdict: 60/40 is a robust upgrade only if it holds CAGR+Sharpe in 2000-2017 OOS")
print("  AND its dot-com/GFC drawdowns aren't materially worse than deployed.")
