"""
Phase 0 (refine) — How much of the tax is REALLY short-term?
The 5pp drag estimate assumed worst case: 100% turnover, every gain short-term.
But positions carry over across rebalances, and some are held >1yr (long-term,
15% rate). This measures the actual FIFO realized-gain split via lot tracking.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np

ST_RATE, LT_RATE = 0.2405, 0.15
V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
       "track_tax": True}

bt = FastBacktester()
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
          "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
    setattr(bt.uni, k, {})
bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

m = bt.run("2018-01-01", "2025-12-31", V12)
ev = bt._tax_events
vals = m["daily_values"]
yrs = (vals.index[-1] - vals.index[0]).days / 365.25
avg_val = vals.mean()

# turnover (one-way): total $ traded / 2 / avg value / yrs
turnover = bt._gross_traded / 2 / avg_val / yrs

st = [e["gain"] for e in ev if e["days"] < 365]
lt = [e["gain"] for e in ev if e["days"] >= 365]
st_net, lt_net = sum(st), sum(lt)
tot_net = st_net + lt_net
days_w = np.array([e["days"] for e in ev])
gains_w = np.array([abs(e["gain"]) for e in ev])
avg_hold = (days_w * gains_w).sum() / gains_w.sum()      # $-weighted holding days
lt_share_dollars = lt_net / tot_net if tot_net else 0

print("=" * 64)
print("ACTUAL REALIZED-GAIN TAX PROFILE (deployed v12, 1x, 2018-2025)")
print("=" * 64)
print(f"\n  One-way turnover:        {turnover*100:.0f}% / yr")
print(f"  Realized-sale lots:      {len(ev):,}")
print(f"  $-weighted avg holding:  {avg_hold:.0f} days  ({avg_hold/365:.2f} yr)")
print(f"\n  Net realized gains split (by $):")
print(f"     short-term (<1yr):    {st_net/tot_net*100:5.1f}%   taxed @ {ST_RATE:.1%}")
print(f"     long-term  (>=1yr):   {lt_net/tot_net*100:5.1f}%   taxed @ {LT_RATE:.1%}")

# blended rate vs worst-case all-short-term
blended = (max(st_net,0)*ST_RATE + max(lt_net,0)*LT_RATE) / max(st_net+lt_net, 1e-9) \
    if (st_net+lt_net) > 0 else ST_RATE
print(f"\n  Blended effective rate on gains:  {blended*100:.1f}%")
print(f"  (vs worst-case all-short-term:    {ST_RATE*100:.1f}%)")
print(f"\n  => Long-term treatment saves ~{(ST_RATE-blended)*100:.1f}pp of tax rate on gains.")
print(f"     Most gains are still SHORT-term ({st_net/tot_net*100:.0f}%): turnover is high,")
print("     so the tax is LESS than worst-case but still the dominant drag.")
