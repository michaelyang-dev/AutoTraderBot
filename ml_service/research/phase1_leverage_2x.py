"""
Should we raise leverage to 2x because vol-scaling improved drawdown?
Simulate the LIVE setup at each leverage: base leverage L, vol-scaling overlay
de-risking toward target = L*15% (40d realized NAV vol, floor 0.30, de-risk only),
margin borrow on the levered portion. Measure CAGR, MaxDD, after-tax CAGR, and
the worst single-day / margin-call proximity. The honest test: does 2x's extra
return justify the extra drawdown, and does vol-scaling actually contain it?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

BORROW = 0.05
ST_RATE = 0.2405
V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

bt = FastBacktester()
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
          "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
    setattr(bt.uni, k, {})
bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

r1x = bt.run("2018-01-02", "2025-12-31", V12)["daily_values"].pct_change().dropna()
idx = r1x.index
r1x = r1x.values


def sim(L):
    target = L * 0.15
    nav = 1.0; navs = []; recent = []
    for r in r1x:
        if len(recent) >= 20:
            rv = np.std(recent[-40:]) * np.sqrt(252)
            vs = min(1.0, max(0.30, target / rv)) if rv > 0 else 1.0
        else:
            vs = 1.0
        eff = L * vs
        rl = eff * r - max(eff - 1, 0) * BORROW / 252      # borrow on levered portion
        nav *= (1 + rl)
        navs.append(nav); recent.append(rl)
        if len(recent) > 40:
            recent.pop(0)
    s = pd.Series(navs, index=idx)
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = s.iloc[-1] ** (1 / yrs) - 1
    mdd = ((s - s.cummax()) / s.cummax()).min()
    dr = s.pct_change().dropna()
    worst = dr.min()
    # after-tax: yearly returns -> tax net gain at ST rate
    yr = s.resample("YE").last().pct_change().dropna()
    yr0 = s.iloc[0]; first_yr = s[s.index.year == idx[0].year].iloc[-1] / yr0 - 1
    yearly = [first_yr] + list(yr.values)
    w = 1.0
    for g in yearly:
        gain = w * g; w = w + gain - gain * ST_RATE
    at = w ** (1 / yrs) - 1
    return cagr, mdd, worst, at


print("=" * 70)
print("LEVERAGE COMPARISON (live vol-scaling overlay, single-start 2018-2025)")
print("=" * 70)
print(f"\n  {'leverage':<12}{'CAGR':>8}{'after-tax':>11}{'MaxDD':>9}{'worst day':>11}")
prev = None
for L in [1.00, 1.49, 1.75, 2.00]:
    cagr, mdd, worst, at = sim(L)
    tag = "  <- live" if abs(L - 1.49) < 0.01 else ""
    delta = ""
    if prev:
        d_at = (at - prev[1]) * 100; d_dd = (mdd - prev[0]) * 100
        delta = f"   (+{d_at:.1f}pp after-tax for {d_dd:+.0f}pp DD)"
    print(f"  {L:<12.2f}{cagr*100:>7.1f}%{at*100:>10.1f}%{mdd*100:>8.1f}%{worst*100:>10.1f}%{delta}")
    prev = (mdd, at)
print("\n  Note: single-start (runs ~4pp hot vs start-day-avg); read the DELTAS, not absolutes.")
print("  RISKS at 2x: (1) maintenance-margin liquidation if HOLDINGS fall ~33% (~66% NAV);")
print("  concentration add-ons make it easier -> a recoverable drawdown becomes forced ruin.")
print("  (2) Reg-T caps OVERNIGHT leverage at 2x, and the engine must target ABOVE effective")
print("  to beat rounding, so true 2x effective is unreachable on a $30K Reg-T account.")
print("  (3) Vol-scaling LAGS (40d) -> a fast crash lands before it de-risks. Sample has NO")
print("  2008-style bear, so these MaxDDs are the BENIGN case.")
