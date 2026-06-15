"""
Phase 4b — Fair trend construction: multi-timeframe signal (avg of 60/120/252d
trend), the standard CTA approach that's far less whipsaw-prone than a single
12m lookback. Plus a cash-dilution control to separate genuine hedging from
mere volatility dilution. Same instruments (SPY/TLT/GLD) — data-limited.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

LOOKS = [60, 120, 252]
VOLWIN, TGT_I, MAXLEV = 60, 0.10, 3.0
INSTR = ["SPY", "TLT", "GLD"]

bt = FastBacktester()
P = bt.prices[INSTR].dropna()
R = P.pct_change()
rvol = R.rolling(VOLWIN).std() * np.sqrt(252)

# multi-timeframe signal: average sign across lookbacks (continuous -1..1)
sig = sum(np.sign(P / P.shift(L) - 1.0) for L in LOOKS) / len(LOOKS)
pos = (sig * (TGT_I / rvol).clip(upper=MAXLEV)).shift(1)
trend_ret = (pos * R).mean(axis=1).dropna()
trend = trend_ret * (0.15 / (trend_ret.std() * np.sqrt(252)))

DEP = {"universe": "sp1500", "mom_w": .50, "val_w": .35, "lv_w": .15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
eq = bt.run("2018-01-01", "2025-12-31", DEP)["daily_values"].pct_change().dropna()
common = trend.index.intersection(eq.index)
trend, eq = trend.loc[common], eq.loc[common]


def stats(r):
    v = (1 + r).cumprod()
    return v.iloc[-1] ** (252 / len(r)) - 1, (r.mean() / r.std()) * np.sqrt(252), ((v - v.cummax()) / v.cummax()).min()


print("=" * 70)
print(f"PHASE 4b: MULTI-TIMEFRAME TREND (SPY/TLT/GLD) {common.min().year}-{common.max().year}")
print("=" * 70)
cg, sh, md = stats(trend)
print(f"\n  Multi-TF trend standalone: CAGR {cg*100:.1f}%  Sharpe {sh:.2f}  MaxDD {md*100:.1f}%")
print(f"  Correlation to v12 equity: {np.corrcoef(trend, eq)[0,1]:+.2f}")
print("\n  Crash-window returns:")
for wl, y0, y1 in [("COVID 2020", "2020-02-01", "2020-04-30"), ("2022", "2022-01-01", "2022-12-31"), ("2018 Q4", "2018-10-01", "2018-12-31")]:
    m = (common >= pd.Timestamp(y0)) & (common <= pd.Timestamp(y1))
    print(f"     {wl:12} equity {((1+eq[m]).prod()-1)*100:+6.1f}%   trend {((1+trend[m]).prod()-1)*100:+6.1f}%")

print(f"\n  Blend vs CASH-dilution control (same weight, 0%-return cash):")
print(f"    {'w':>5}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}   |  cash-blend MaxDD (dilution only)")
for w in [0.0, 0.10, 0.15, 0.20]:
    b = (1 - w) * eq + w * trend
    cg, sh, md = stats(b)
    _, _, md_cash = stats((1 - w) * eq)     # cash dilution: scaling returns leaves %DD ~unchanged
    print(f"    {w*100:>3.0f}%{cg*100:>7.1f}%{sh:>8.2f}{md*100:>7.1f}%   |  {md_cash*100:>6.1f}%", flush=True)
print("\nGenuine hedge => trend-blend MaxDD < cash-blend MaxDD AND Sharpe rises.")
