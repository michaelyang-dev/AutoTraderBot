"""
Phase 11b — ROBUSTNESS: does swapping FF-UMD for price-based UMD change the FULL
STRATEGY's behavior? Inject each UMD series into the backtest's crash detector and
compare CAGR/Sharpe/MaxDD + the crash windows. Also a 'no UMD-crash' baseline to
size how much the detector even matters. Only swap if FF and price give ~identical
strategy results in the period where FF data WAS current (apples-to-apples).
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
bt.uni.get_sp500 = bt._get_sp1500
prices, rets, allidx = bt.prices, bt.prices.pct_change(), list(bt.prices.index)
ff_umd_20 = bt.umd_20d.copy()   # the FF-based series the backtest normally uses

# ── build price-based UMD (monthly-formed tercile momentum long-short, daily) ──
dts = [d for d in prices.index if pd.Timestamp("2000-06-01") <= d <= pd.Timestamp("2025-12-31")]
umd_daily, longs, shorts, lfm = {}, set(), set(), None
for d in dts:
    loc = allidx.index(d)
    if loc < 260:
        continue
    if lfm != (d.year, d.month):
        lfm = (d.year, d.month)
        mem = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
        p0, p20, p252 = prices.loc[d], prices.loc[allidx[loc - 20]], prices.loc[allidx[loc - 252]]
        mom = {m: (p0[m] / p252[m] - 1) - (p0[m] / p20[m] - 1) for m in mem
               if p0.get(m) and p252.get(m) and p20.get(m)
               and not (np.isnan(p0[m]) or np.isnan(p252[m]) or np.isnan(p20[m]))}
        if len(mom) > 30:
            s = sorted(mom, key=mom.get); k = len(s) // 3
            shorts, longs = set(s[:k]), set(s[-k:])
    if longs and shorts:
        r = rets.loc[d]
        rl = np.nanmean([r.get(x) for x in longs]); rs = np.nanmean([r.get(x) for x in shorts])
        if np.isfinite(rl) and np.isfinite(rs):
            umd_daily[d] = rl - rs
price_umd_20 = pd.Series(umd_daily).sort_index().rolling(20).sum().reindex(prices.index)

DEP = {"universe": "sp1500", "mom_w": .50, "val_w": .35, "lv_w": .15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}


def run_with(umd_series):
    bt.umd_20d = umd_series
    return bt.run("2000-01-03", "2025-12-31", DEP)["daily_values"]


def stats(v):
    r = v.pct_change().dropna(); yrs = (v.index[-1] - v.index[0]).days / 365.25
    cg = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    md = ((v - v.cummax()) / v.cummax()).min()
    return cg, sh, md


variants = {
    "FF-UMD (current)": ff_umd_20,
    "PRICE-UMD (new)": price_umd_20,
    "no UMD-crash": pd.Series(0.0, index=prices.index),
}
curves = {}
print("=" * 64)
print("FULL STRATEGY: FF-UMD vs PRICE-UMD crash detector (2000-2025)")
print("=" * 64)
print(f"\n  {'crash signal':<20}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}")
for name, s in variants.items():
    v = run_with(s); curves[name] = v
    cg, sh, md = stats(v)
    print(f"  {name:<20}{cg*100:>7.1f}%{sh:>8.2f}{md*100:>8.1f}%", flush=True)

print(f"\n  Crash-window drawdowns:")
print(f"  {'window':<11}" + "".join(f"{n.split()[0][:9]:>12}" for n in variants))
for wn, a, b in [("2009 crash", "2009-03-01", "2009-09-30"), ("2008 GFC", "2008-09-01", "2009-03-31"),
                 ("2022", "2022-01-01", "2022-12-31"), ("COVID", "2020-02-01", "2020-04-30")]:
    row = f"  {wn:<11}"
    for name in variants:
        w = curves[name][(curves[name].index >= a) & (curves[name].index <= b)]
        row += f"{((w - w.cummax())/w.cummax()).min()*100:>11.1f}%"
    print(row)
print("\n  Swap is safe if PRICE-UMD ~= FF-UMD on CAGR/Sharpe/MaxDD and crash windows.")
