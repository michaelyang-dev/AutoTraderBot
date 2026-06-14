"""
Phase 2 — Walk-forward analysis (the real overfit test).
For each test YEAR, pick the best config on the prior 3-year TRAIN window
(by Sharpe), then record that config's OUT-OF-SAMPLE return on the test year.
Chain the OOS years -> walk-forward equity. Compare to:
  (a) the deployed v12 config's full-sample number, and
  (b) the best single config in hindsight (full-sample).
A large gap (walk-forward << full-sample best) = the config search was overfit.

Efficient: run each candidate ONCE (full period), cache daily values, slice windows.
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


def sharpe(vals):
    r = vals.pct_change().dropna()
    if r.std() == 0 or len(r) < 20:
        return -9
    return (r.mean() / r.std()) * np.sqrt(252)


def cagr(vals):
    if len(vals) < 2:
        return 0
    return (vals.iloc[-1] / vals.iloc[0]) ** (252 / len(vals)) - 1


def window(vals, y0, y1):
    return vals[(vals.index >= pd.Timestamp(f"{y0}-01-01")) & (vals.index <= pd.Timestamp(f"{y1}-12-31"))]


B = {"universe": "sp1500", "sec_w": 0.0, "cap": 0.15,
     "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
# Candidate grid — configs plausibly in the original search
CANDIDATES = {
    "DEPLOYED 50/35/15 n5 r20 s40": {**B, "mom_w": .50, "val_w": .35, "lv_w": .15, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40},
    "pure-mom 100/0/0 n5 r20 s40":  {**B, "mom_w": 1.0, "val_w": .00, "lv_w": .00, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40},
    "60/40 mom/val n5 r20 s40":     {**B, "mom_w": .60, "val_w": .40, "lv_w": .00, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40},
    "v11 35/25/40 n8 r20 s40":      {**B, "mom_w": .35, "val_w": .25, "lv_w": .40, "top_n": 8, "rebal_days": 20, "trailing_stop": 0.40},
    "50/35/15 n8 r20 s40":          {**B, "mom_w": .50, "val_w": .35, "lv_w": .15, "top_n": 8, "rebal_days": 20, "trailing_stop": 0.40},
    "50/35/15 n5 r20 s45":          {**B, "mom_w": .50, "val_w": .35, "lv_w": .15, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.45},
    "50/35/15 n5 r15 s40":          {**B, "mom_w": .50, "val_w": .35, "lv_w": .15, "top_n": 5, "rebal_days": 15, "trailing_stop": 0.40},
    "v10 85/15 n8 r10 s25":         {**B, "mom_w": .85, "val_w": .15, "lv_w": .00, "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25},
}

bt = FastBacktester(); clear(bt)
print("running candidates (once each)...", flush=True)
DV = {name: bt.run("2018-01-01", "2025-12-31", cfg)["daily_values"] for name, cfg in CANDIDATES.items()}

# Walk-forward: test years 2021..2025, train = prior 3 years
print("\n" + "=" * 72)
print("WALK-FORWARD (3yr train -> 1yr OOS test), config chosen by train Sharpe")
print("=" * 72)
print(f"\n{'test yr':>8}{'chosen config':<34}{'train Shrp':>11}{'OOS return':>11}")
wf_rets = []
for ty in [2021, 2022, 2023, 2024, 2025]:
    best, best_s = None, -99
    for name, v in DV.items():
        tw = window(v, ty - 3, ty - 1)
        s = sharpe(tw)
        if s > best_s:
            best_s, best = s, name
    test = window(DV[best], ty, ty)
    oos = test.iloc[-1] / test.iloc[0] - 1 if len(test) > 1 else 0
    wf_rets.append(oos)
    print(f"{ty:>8}  {best:<33}{best_s:>10.2f}{oos*100:>10.1f}%")

wf_cagr = np.prod([1 + r for r in wf_rets]) ** (1 / len(wf_rets)) - 1
wf_sharpe_proxy = np.mean(wf_rets) / (np.std(wf_rets) + 1e-9)
print(f"\nWALK-FORWARD (2021-2025 OOS): CAGR {wf_cagr*100:.1f}% | yr-return mean {np.mean(wf_rets)*100:+.1f}% std {np.std(wf_rets)*100:.1f}%")

# Full-sample comparisons (same 2021-2025 window for apples-to-apples)
dep = window(DV["DEPLOYED 50/35/15 n5 r20 s40"], 2021, 2025)
print(f"DEPLOYED full-sample (2021-2025):   CAGR {cagr(dep)*100:.1f}% | Sharpe {sharpe(dep):.2f}")
best_full = max(DV, key=lambda n: cagr(window(DV[n], 2021, 2025)))
bf = window(DV[best_full], 2021, 2025)
print(f"BEST-in-hindsight ({best_full[:20]}): CAGR {cagr(bf)*100:.1f}% | Sharpe {sharpe(bf):.2f}")
print(f"\nGap (best-hindsight - walk-forward): {(cagr(bf)-wf_cagr)*100:.1f}pp")
print("Small gap => robust selection. Large gap => the full-sample best is overfit.")
