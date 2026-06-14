"""
Phase 0.2 — After-tax CAGR vs after-tax SPY buy-and-hold.
Marginal rate: 24.05% (user-provided, short-term ordinary income).

Key asymmetry:
- The strategy has ~1100% turnover -> ~all gains are SHORT-term, taxed EVERY year.
- SPY buy-and-hold defers tax; the gain is taxed once at the end at LONG-term rates.
This is why high-turnover strategies lose more to taxes than their pre-tax edge suggests.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np

ST_RATE = 0.2405   # short-term (ordinary) — user-provided
LT_RATE = 0.15     # long-term cap gains assumption for SPY at sale (verify your bracket)

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

# SPY total-return by year (from README year table)
SPY = {2018: -0.052, 2019: 0.311, 2020: 0.173, 2021: 0.305,
       2022: -0.186, 2023: 0.267, 2024: 0.256, 2025: 0.180}

bt = FastBacktester()
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
          "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
    setattr(bt.uni, k, {})
bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

# average yearly returns across 5 start days
import collections
yr_acc = collections.defaultdict(list)
for st in ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-08"]:
    m = bt.run(st, "2025-12-31", V12)
    for y, d in m["yearly"].items():
        yr_acc[y].append(d.get("cagr", 0))
strat_yr = {y: np.mean(v) for y, v in yr_acc.items()}

print("=" * 64, flush=True)
print(f"AFTER-TAX: strategy (short-term {ST_RATE:.1%}/yr) vs SPY (LT deferred)")
print("=" * 64)


def aftertax_strategy(yearly, t):
    """Tax each year's gain at short-term rate (full realization). Losses give credit."""
    w = 1.0
    for y in sorted(yearly):
        r = yearly[y]
        gain = w * r
        tax = gain * t if gain > 0 else gain * t  # losses offset future gains -> symmetric approx
        w = w + gain - tax
    return w


def aftertax_spy(yearly, lt):
    """Buy-and-hold: defer all tax, pay LT once at the end on total gain."""
    w = 1.0
    for y in sorted(yearly):
        w *= (1 + yearly[y])
    total_gain = w - 1.0
    return 1.0 + total_gain * (1 - lt) if total_gain > 0 else w


years = sorted(set(strat_yr) & set(SPY))
ny = len(years)

# pre-tax
sp_pre = 1.0; spy_pre = 1.0
for y in years:
    sp_pre *= (1 + strat_yr[y]); spy_pre *= (1 + SPY[y])
sp_pre_cagr = sp_pre ** (1 / ny) - 1
spy_pre_cagr = spy_pre ** (1 / ny) - 1

# after-tax
sp_at = aftertax_strategy({y: strat_yr[y] for y in years}, ST_RATE)
spy_at = aftertax_spy({y: SPY[y] for y in years}, LT_RATE)
sp_at_cagr = sp_at ** (1 / ny) - 1
spy_at_cagr = spy_at ** (1 / ny) - 1

print(f"\nPeriod: {years[0]}-{years[-1]} ({ny} yrs), 1x, start-day averaged\n")
print(f"{'':22}{'PRE-TAX':>10}{'AFTER-TAX':>12}")
print(f"{'Strategy CAGR':22}{sp_pre_cagr*100:>9.1f}%{sp_at_cagr*100:>11.1f}%")
print(f"{'SPY buy-hold CAGR':22}{spy_pre_cagr*100:>9.1f}%{spy_at_cagr*100:>11.1f}%")
print(f"{'Edge over SPY':22}{(sp_pre_cagr-spy_pre_cagr)*100:>+8.1f}pp{(sp_at_cagr-spy_at_cagr)*100:>+10.1f}pp")
print(f"\nTax drag on strategy: {(sp_pre_cagr-sp_at_cagr)*100:.1f}pp/yr  |  on SPY: {(spy_pre_cagr-spy_at_cagr)*100:.1f}pp/yr")
print("\nNOTE: SPY uses LT 15% at sale. If your combined ST rate is higher than")
print("24.05% (e.g. with NY state + NYC), the strategy's after-tax edge shrinks further.")
