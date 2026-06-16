"""
Phase 9 — Does a credit-spread (HY OAS) overlay beat vol-scaling on RISK-ADJUSTED
return (not just drawdown)? Credit widening leads equity stress, so it may de-risk
with better TIMING — and fewer false de-risks in calm markets — which is the only
way to raise Sharpe/Calmar above vol-scaling's (which Phase 8 showed is ~neutral).

Overlays applied post-hoc to the deployed strategy's daily returns (an exposure
overlay just scales the book), lagged 1 day (no look-ahead), AFTER overlay-turnover
cost (5bps one-way per unit exposure change). Judged on Sharpe + Calmar.
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

# HY OAS (ICE BofA US High Yield OAS); ffill tail (FRED file ends ~Feb 2025)
fred = pd.read_parquet("data/wrds/fred_interest_rates_spreads_daily.parquet", columns=["date", "bamlh0a0hym2"])
fred["date"] = pd.to_datetime(fred["date"])
oas = fred.set_index("date")["bamlh0a0hym2"].reindex(base.index).ffill()

# de-risk overlays (both: scale in [0.3, 1.0], de-risk only)
rv = ret.rolling(40).std() * np.sqrt(252)
vol_scale = (0.15 / rv).clip(0.3, 1.0).reindex(ret.index)
# credit: de-risk when OAS is above its 6-month trailing average (spreads widening)
oas_base = oas.rolling(126, min_periods=20).mean()
credit_scale = (oas_base / oas).clip(0.3, 1.0).reindex(ret.index)

ONE_WAY = 0.0005  # 5bps per unit exposure change (overlay turnover cost)


def apply(scale):
    s = scale.shift(1).fillna(1.0)                      # signal lagged 1 day
    turn = s.diff().abs().fillna(0.0)                   # exposure change per day
    return s * ret - turn * ONE_WAY, turn.sum() / (len(ret) / 252)  # net return, ann. turnover


VARIANTS = {
    "base (no overlay)": (ret, 0.0),
    "vol-scale only": apply(vol_scale),
    "credit only": apply(credit_scale),
    "vol + credit (min)": apply(pd.concat([vol_scale, credit_scale], axis=1).min(axis=1)),
}


def stats(r):
    v = (1 + r).cumprod(); yrs = len(r) / 252
    cagr = v.iloc[-1] ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    mdd = ((v - v.cummax()) / v.cummax()).min()
    return cagr, sh, mdd, (cagr / abs(mdd) if mdd else 0), v


print("=" * 78)
print("CREDIT-SPREAD OVERLAY vs VOL-SCALING (1x, 2000-2025, after overlay turnover cost)")
print("=" * 78)
print(f"\n  {'variant':<20}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'Calmar':>8}{'turnover':>10}")
curves = {}
for name, (r, turn) in VARIANTS.items():
    cg, sh, md, ca, v = stats(r); curves[name] = v
    print(f"  {name:<20}{cg*100:>7.1f}%{sh:>8.2f}{md*100:>8.1f}%{ca:>8.2f}{turn:>9.1f}x", flush=True)

WINDOWS = [("2011", "2011-07-01", "2011-10-31"), ("2015-16", "2015-08-01", "2016-02-29"),
           ("2018Q4", "2018-10-01", "2018-12-31"), ("2022", "2022-01-01", "2022-12-31"),
           ("GFC", "2007-10-01", "2009-03-31"), ("COVID", "2020-02-01", "2020-04-30")]
print(f"\n  Per-window drawdown:")
print(f"  {'window':<10}" + "".join(f"{n.split()[0][:9]:>13}" for n in VARIANTS))
for wn, a, b in WINDOWS:
    row = f"  {wn:<10}"
    for name in VARIANTS:
        w = curves[name][(curves[name].index >= a) & (curves[name].index <= b)]
        d = ((w - w.cummax()) / w.cummax()).min() * 100 if len(w) > 1 else 0
        row += f"{d:>12.1f}%"
    print(row)
print("\n  Credit earns its slot only if Sharpe/Calmar beat 'vol-scale only'. Watch 2022")
print("  (rates-driven, credit only widened to ~6%) vs GFC (credit-driven, OAS ~20%).")
