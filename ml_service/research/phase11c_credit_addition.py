"""
Phase 11c — Does adding a CREDIT (HY OAS) trigger to the momentum-crash detector help?
Today: crash weights fire when umd_20d < -0.05. Test OR-ing in a credit-stress trigger
so the strategy also goes defensive when high-yield spreads blow out (credit leads
equity in GFC/COVID). Judge on CAGR/Sharpe/MaxDD + crash windows vs the price-UMD-only
detector. Credit earns a slot ONLY if it cuts drawdown without bleeding CAGR.
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

# ── price-based UMD (same construction as 11b) ──
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

# ── HY OAS ──
fred = pd.read_parquet("data/wrds/fred_interest_rates_spreads_daily.parquet", columns=["date", "bamlh0a0hym2"])
fred["date"] = pd.to_datetime(fred["date"])
oas = fred.set_index("date")["bamlh0a0hym2"].reindex(prices.index).ffill()
oas_chg20 = oas - oas.shift(20)                      # 20-day spread change (pp)
oas_ratio = oas / oas.rolling(126, min_periods=20).mean()   # vs 6-month trailing mean

DEP = {"universe": "sp1500", "mom_w": .50, "val_w": .35, "lv_w": .15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}


def synth(credit_fire):
    """price-UMD, but forced below -0.05 on days the credit trigger fires (OR logic)."""
    s = price_umd_20.copy()
    if credit_fire is not None:
        s = s.where(~credit_fire.reindex(s.index).fillna(False), -0.10)
    return s


def run_with(series):
    bt.umd_20d = series
    return bt.run("2000-01-03", "2025-12-31", DEP)["daily_values"]


def stats(v):
    r = v.pct_change().dropna(); yrs = (v.index[-1] - v.index[0]).days / 365.25
    cg = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    md = ((v - v.cummax()) / v.cummax()).min()
    return cg, sh, md


# credit-trigger definitions to test as ADDITIONS (lagged 1d, no look-ahead)
variants = {
    "price-UMD only":            None,
    "+credit chg>+1.5pp/20d":    (oas_chg20 > 1.5).shift(1),
    "+credit lvl>7%":            (oas > 7.0).shift(1),
    "+credit >1.3x 6m-avg":      (oas_ratio > 1.3).shift(1),
    "+credit chg>1.5 OR lvl>8":  ((oas_chg20 > 1.5) | (oas > 8.0)).shift(1),
}
curves = {}
print("=" * 70)
print("CREDIT as an ADDITION to the momentum-crash trigger (full strategy, 2000-25)")
print("=" * 70)
print(f"\n  {'crash trigger':<27}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'days':>7}")
for name, cf in variants.items():
    v = run_with(synth(cf)); curves[name] = v
    cg, sh, md = stats(v)
    nd = int(cf.fillna(False).sum()) if cf is not None else 0
    print(f"  {name:<27}{cg*100:>7.1f}%{sh:>8.2f}{md*100:>8.1f}%{nd:>7}", flush=True)

print(f"\n  Crash-window drawdowns:")
hdr = "".join(f"{n.split('+')[-1][:9]:>11}" for n in variants)
print(f"  {'window':<10}" + hdr.replace("price-UMD", "  base"))
for wn, a, b in [("2008 GFC", "2008-09-01", "2009-03-31"), ("COVID", "2020-02-01", "2020-04-30"),
                 ("2022", "2022-01-01", "2022-12-31"), ("2011", "2011-07-01", "2011-10-31"),
                 ("2015-16", "2015-08-01", "2016-02-29"), ("2018Q4", "2018-10-01", "2018-12-31")]:
    row = f"  {wn:<10}"
    for name in variants:
        w = curves[name][(curves[name].index >= a) & (curves[name].index <= b)]
        row += f"{((w - w.cummax())/w.cummax()).min()*100:>10.1f}%"
    print(row)
print("\n  Credit earns a slot only if it cuts GFC/COVID DD materially without bleeding CAGR.")
