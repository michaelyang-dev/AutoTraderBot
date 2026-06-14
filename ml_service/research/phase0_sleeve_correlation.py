"""
Phase 0.4 — Sleeve correlation.
Question: are Value (35%) and Low-Vol (15%) redundant? If their realized
return correlation is high (>~0.6), it's a 2-sleeve strategy in a 3-sleeve costume.

Builds each sleeve's standalone 20-day forward return series (its own picks,
its own weights) over 2018-2025 and correlates them. Matches deployed v12
(enhanced + SI cleared).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
from strategies.multi_strategy_engine import (
    strategy_value, strategy5_lowvol_quality, strategy1_momentum_reversal,
)
import numpy as np
import pandas as pd

bt = FastBacktester()
# match deployed v12
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
          "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
    setattr(bt.uni, k, {})
bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
bt.uni.get_sp500 = bt._get_sp1500

prices = bt.prices
dates = sorted([d for d in prices.index
                if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")])

rows = {"mom": [], "val": [], "lv": []}


def port_ret(picks, d, fwd):
    r = 0.0; tw = 0.0
    p0row = prices.loc[d]; p1row = prices.loc[fwd]
    for sym, w in picks.items():
        p0 = p0row.get(sym); p1 = p1row.get(sym)
        if p0 and p1 and p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
            r += w * (p1 / p0 - 1.0); tw += w
    return r / tw if tw > 0 else np.nan


for i in range(0, len(dates) - 21, 20):
    d = dates[i]; fwd = dates[i + 20]
    members = bt.uni.get_sp500(d)
    if len(members) < 50:
        continue
    t_mom = strategy1_momentum_reversal(d, bt.uni, i, top_n=5, rebal_days=20) or {}
    t_val = strategy_value(bt.uni, d, members, top_n=10) or {}
    t_lv = strategy5_lowvol_quality(d, bt.uni, i, top_n=10, rebal_days=20) or {}
    rows["mom"].append(port_ret(t_mom, d, fwd))
    rows["val"].append(port_ret(t_val, d, fwd))
    rows["lv"].append(port_ret(t_lv, d, fwd))

df = pd.DataFrame(rows).dropna()
print("=" * 60, flush=True)
print(f"SLEEVE CORRELATION (n={len(df)} non-overlapping 20d periods, 2018-2025)")
print("=" * 60)
print("\nReturn correlation matrix:")
print(df.corr().round(3).to_string())
print(f"\n>>> Value vs Low-Vol correlation: {df['val'].corr(df['lv']):.3f}")
print(f"    Momentum vs Value:           {df['mom'].corr(df['val']):.3f}")
print(f"    Momentum vs Low-Vol:         {df['mom'].corr(df['lv']):.3f}")
print("\nAvg 20d return per sleeve:")
for s in ["mom", "val", "lv"]:
    print(f"  {s}: {df[s].mean()*100:+.2f}%  (vol {df[s].std()*100:.2f}%)")
vlv = df['val'].corr(df['lv'])
print("\nVERDICT:")
if vlv > 0.6:
    print(f"  Value & Low-Vol correlation {vlv:.2f} > 0.6 — REDUNDANT. Effectively 2 sleeves.")
elif vlv > 0.4:
    print(f"  Value & Low-Vol correlation {vlv:.2f} — partially redundant.")
else:
    print(f"  Value & Low-Vol correlation {vlv:.2f} < 0.4 — genuinely distinct sleeves.")
