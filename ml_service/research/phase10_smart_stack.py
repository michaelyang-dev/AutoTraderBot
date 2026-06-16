"""
Phase 10 — The orthogonal regime STACK: each layer catches a regime the others miss.
  - credit (HY OAS, primary)      -> credit-driven stress (GFC), efficient/leading
  - vol-spike (realized vol, hi target 30%) -> only the extreme panic tail (COVID)
  - rates (10y real-yield 6m momentum)      -> the rates-driven bear credit misses (2022)
Combined = min of the three (each de-risk only). Judged on Sharpe/Calmar AFTER overlay
turnover cost — and OUT-OF-SAMPLE (split halves), since a multi-layer overlay tuned on
one window is exactly what overfits.
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
base = bt.run("2000-01-03", "2025-12-31", {**DEP, "vol_scaling": False})["daily_values"]
ret = base.pct_change().dropna()

fred = pd.read_parquet("data/wrds/fred_interest_rates_spreads_daily.parquet",
                       columns=["date", "bamlh0a0hym2", "dfii10"])
fred["date"] = pd.to_datetime(fred["date"]); fred = fred.set_index("date")
oas = fred["bamlh0a0hym2"].reindex(base.index).ffill()
ry = fred["dfii10"].reindex(base.index).ffill()

rv = ret.rolling(40).std() * np.sqrt(252)
credit = (oas.rolling(126, min_periods=20).mean() / oas).clip(0.3, 1.0).reindex(ret.index)
vol15 = (0.15 / rv).clip(0.3, 1.0).reindex(ret.index)                    # aggressive (current)
vol_spike = (0.30 / rv).clip(0.3, 1.0).reindex(ret.index)                # extreme-only
ry_chg = (ry - ry.shift(126)).reindex(ret.index)
rates = (1 - 0.3 * ry_chg.clip(lower=0)).clip(0.5, 1.0)                   # de-risk when real yields rip up
stack = pd.concat([credit, vol_spike, rates], axis=1).min(axis=1)        # the smart combo

ONE_WAY = 0.0005


def apply(scale):
    s = scale.shift(1).fillna(1.0)
    return s * ret - s.diff().abs().fillna(0.0) * ONE_WAY


VARIANTS = {"base": ret, "credit only": apply(credit), "vol-15 (current)": apply(vol15),
            "SMART stack": apply(stack)}


def stats(r):
    v = (1 + r).cumprod(); yrs = len(r) / 252
    cg = v.iloc[-1] ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    md = ((v - v.cummax()) / v.cummax()).min()
    return cg, sh, md, (cg / abs(md) if md else 0)


print("=" * 70)
print("SMART REGIME STACK (1x, after overlay cost)")
print("=" * 70)
for lab, (a, b) in [("FULL 2000-2025", ("2000", "2026")),
                    ("OOS H1 2000-2012", ("2000", "2012-12-31")),
                    ("OOS H2 2013-2025", ("2013-01-01", "2026"))]:
    print(f"\n  {lab}:")
    print(f"  {'variant':<18}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}")
    for name, r in VARIANTS.items():
        rs = r[(r.index >= a) & (r.index <= b)]
        cg, sh, md, ca = stats(rs)
        print(f"  {name:<18}{cg*100:>7.1f}%{sh:>8.2f}{md*100:>8.1f}%{ca:>8.2f}", flush=True)

print(f"\n  Per-window drawdown (does each layer cover its regime?):")
print(f"  {'window':<9}" + "".join(f"{n.split()[0][:8]:>11}" for n in VARIANTS))
for wn, a, b in [("2018Q4", "2018-10-01", "2018-12-31"), ("2022", "2022-01-01", "2022-12-31"),
                 ("GFC", "2007-10-01", "2009-03-31"), ("COVID", "2020-02-01", "2020-04-30")]:
    row = f"  {wn:<9}"
    for name, r in VARIANTS.items():
        v = (1 + r[(r.index >= a) & (r.index <= b)]).cumprod()
        row += f"{((v - v.cummax())/v.cummax()).min()*100:>10.1f}%"
    print(row)
print("\n  STACK wins only if Sharpe/Calmar beat the best single overlay in BOTH halves.")
