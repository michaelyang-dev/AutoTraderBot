"""
The number that matters: after-tax CAGR at the LIVE leverage (~1.49x).
Builds on measured facts:
  - ~100% of realized gains are short-term (taxed at 24.05%) -> validated by
    phase0_tax_holding_period (avg hold 63 days).
  - 1x base: ~22% historical (start-day avg) / ~19% honest forward (rebal haircut).
Models leverage with a margin borrow cost on the levered portion, then taxes each
year's net gain at the short-term rate (losses carry/offset symmetrically).
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np

ST_RATE = 0.2405
L = 1.49
V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

bt = FastBacktester()
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
          "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
    setattr(bt.uni, k, {})
bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

# start-day-averaged yearly 1x returns
yr_acc = collections.defaultdict(list)
for st in ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-08"]:
    m = bt.run(st, "2025-12-31", V12)
    for y, d in m["yearly"].items():
        yr_acc[y].append(d.get("cagr", 0))
yearly = {y: np.mean(v) for y, v in yr_acc.items()}
years = sorted(yearly)
ny = len(years)


def cagr_from_yearly(yr):
    w = 1.0
    for y in years:
        w *= (1 + yr[y])
    return w ** (1 / ny) - 1


def aftertax_cagr(yr, t):
    w = 1.0
    for y in years:
        gain = w * yr[y]
        w = w + gain - gain * t          # losses give symmetric credit (offset future gains)
    return w ** (1 / ny) - 1


def lever(yr, borrow):
    return {y: L * yr[y] - (L - 1) * borrow for y in years}


base_1x = cagr_from_yearly(yearly)
print("=" * 66)
print("AFTER-TAX CAGR  (deployed v12, ~100% short-term @ 24.05%)")
print("=" * 66)
print(f"\n  Historical base 1x (2018-2025, start-day avg): {base_1x*100:.1f}% pre-tax")
print(f"\n  {'config':<26}{'pre-tax':>9}{'after-tax':>11}")
print(f"  {'1x':<26}{base_1x*100:>8.1f}%{aftertax_cagr(yearly, ST_RATE)*100:>10.1f}%")
for b in [0.03, 0.05, 0.06]:
    lv = lever(yearly, b)
    print(f"  {'1.49x  borrow '+str(int(b*100))+'%':<26}{cagr_from_yearly(lv)*100:>8.1f}%{aftertax_cagr(lv, ST_RATE)*100:>10.1f}%")

# Forward-looking: haircut 1x base to ~19% (rebal-interval luck), keep yearly SHAPE
print(f"\n  --- FORWARD (haircut 1x base 22%->19% for rebal-interval luck) ---")
scale = 0.19 / base_1x
fwd = {y: yearly[y] * scale for y in years}
print(f"  {'1x forward':<26}{cagr_from_yearly(fwd)*100:>8.1f}%{aftertax_cagr(fwd, ST_RATE)*100:>10.1f}%")
for b in [0.05, 0.06]:
    lv = lever(fwd, b)
    print(f"  {'1.49x forward borrow '+str(int(b*100))+'%':<26}{cagr_from_yearly(lv)*100:>8.1f}%{aftertax_cagr(lv, ST_RATE)*100:>10.1f}%", flush=True)
print("\nNote: leverage modeled at annual level (ignores ~0.5-1pp/yr daily vol drag);")
print("margin interest may be partly tax-deductible (not credited here) -> rough offsets.")
