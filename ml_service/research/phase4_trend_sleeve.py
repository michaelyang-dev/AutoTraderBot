"""
Phase 4 — Diversified time-series-momentum (trend) sleeve as a TAIL hedge.
Moskowitz-Ooi-Pedersen (2012): position_i = sign(trailing-12m return_i),
vol-scaled per instrument. Instruments: SPY (equity), TLT (long bonds),
GLD (gold) — the core crash diversifiers (bonds/gold trend UP in equity selloffs).

The real question for the tail thread: in the equity-strategy's worst windows
(COVID 2020, 2022) does the trend sleeve PAY OFF, and does a small blend lower
the combined drawdown without giving up much CAGR? This is what residual
momentum failed to do.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

LOOK = 252       # 12m trend signal
VOLWIN = 60      # realized-vol window
TGT_I = 0.10     # per-instrument vol target
MAXLEV = 3.0     # cap per-instrument leverage when vol is tiny
INSTR = ["SPY", "TLT", "GLD"]

bt = FastBacktester()
P = bt.prices[INSTR].dropna()
R = P.pct_change()

# per-instrument TSMOM position (lagged 1d, no lookahead), vol-scaled
sig = np.sign(P / P.shift(LOOK) - 1.0)
rvol = R.rolling(VOLWIN).std() * np.sqrt(252)
pos = (sig * (TGT_I / rvol).clip(upper=MAXLEV)).shift(1)
trend_ret = (pos * R).mean(axis=1).dropna()          # equal-weight across instruments

# scale whole sleeve to ~15% annualized vol (comparable to 1x equity sleeve)
sleeve_vol = trend_ret.std() * np.sqrt(252)
trend = trend_ret * (0.15 / sleeve_vol)

# v12 equity strategy daily returns (deployed config, 1x)
DEP = {"universe": "sp1500", "mom_w": .50, "val_w": .35, "lv_w": .15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
eq_vals = bt.run("2018-01-01", "2025-12-31", DEP)["daily_values"]
eq = eq_vals.pct_change().dropna()

# align
common = trend.index.intersection(eq.index)
trend = trend.loc[common]; eq = eq.loc[common]


def stats(r):
    v = (1 + r).cumprod()
    cagr = v.iloc[-1] ** (252 / len(r)) - 1
    sh = (r.mean() / r.std()) * np.sqrt(252)
    mdd = ((v - v.cummax()) / v.cummax()).min()
    return cagr, sh, mdd, v


print("=" * 70)
print(f"PHASE 4: TREND SLEEVE (SPY/TLT/GLD TSMOM) {common.min().year}-{common.max().year}")
print("=" * 70)
cg, sh, md, _ = stats(trend)
print(f"\n  Trend sleeve standalone:  CAGR {cg*100:.1f}%  Sharpe {sh:.2f}  MaxDD {md*100:.1f}%")
print(f"  Correlation to v12 equity strategy: {np.corrcoef(trend, eq)[0,1]:+.2f}")

print("\n  Behavior in equity-strategy crash windows:")
for wl, y0, y1 in [("COVID 2020", "2020-02-01", "2020-04-30"), ("2022", "2022-01-01", "2022-12-31"),
                   ("2018 Q4", "2018-10-01", "2018-12-31")]:
    m = (common >= pd.Timestamp(y0)) & (common <= pd.Timestamp(y1))
    er = (1 + eq[m]).prod() - 1; tr = (1 + trend[m]).prod() - 1
    print(f"     {wl:12} equity {er*100:+6.1f}%   trend {tr*100:+6.1f}%")

print("\n  Blend (1-w)*equity + w*trend:")
print(f"    {'w_trend':>8}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}")
for w in [0.0, 0.10, 0.15, 0.20]:
    b = (1 - w) * eq + w * trend
    cg, sh, md, _ = stats(b)
    print(f"    {w*100:>6.0f}%{cg*100:>7.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)
print("\nTrend earns its slot if the blend lifts Sharpe / cuts MaxDD, esp. via crash-window gains.")
