"""
Phase 11 — Does a REAL-TIME, price-based UMD (computed from our own universe) track
the Ken French UMD well enough to replace it for momentum-crash detection?
The live FF file lags ~46 days, so the crash detector is blind in real time. If a
price-based UMD (top-tercile momentum minus bottom-tercile, monthly-formed) correlates
with FF UMD and agrees on the -5% crash signal, we swap it (fresh data, same signal).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

bt = FastBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl")
bt.uni.get_sp500 = bt._get_sp1500
prices = bt.prices
rets = prices.pct_change()
allidx = list(prices.index)

# ── price-based UMD: monthly-formed momentum long-short, daily returns ──
dates = [d for d in prices.index if pd.Timestamp("2000-06-01") <= d <= pd.Timestamp("2025-12-31")]
umd_daily = {}
long_set, short_set = set(), set()
last_form_month = None
for d in dates:
    loc = allidx.index(d)
    if loc < 260:
        continue
    # reform monthly
    if last_form_month != (d.year, d.month):
        last_form_month = (d.year, d.month)
        members = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
        p0, p20, p252 = prices.loc[d], prices.loc[allidx[loc - 20]], prices.loc[allidx[loc - 252]]
        mom = {}
        for m in members:
            a, a20, a252 = p0.get(m), p20.get(m), p252.get(m)
            if a and a252 and a20 and not (np.isnan(a) or np.isnan(a252) or np.isnan(a20)):
                mom[m] = (a / a252 - 1) - (a / a20 - 1)
        if len(mom) > 30:
            srt = sorted(mom, key=mom.get)
            k = len(srt) // 3
            short_set = set(srt[:k])      # bottom-tercile momentum
            long_set = set(srt[-k:])      # top-tercile momentum
    if long_set and short_set:
        r = rets.loc[d]
        rl = np.nanmean([r.get(s) for s in long_set if pd.notna(r.get(s))])
        rs = np.nanmean([r.get(s) for s in short_set if pd.notna(r.get(s))])
        if np.isfinite(rl) and np.isfinite(rs):
            umd_daily[d] = rl - rs

price_umd = pd.Series(umd_daily).sort_index()

# ── FF UMD ──
ff = pd.read_parquet("data/wrds/fama_french_5factors_momentum_daily.parquet", columns=["date", "umd"])
ff["date"] = pd.to_datetime(ff["date"]); ff_umd = ff.set_index("date")["umd"]

# align + 20-day sums
idx = price_umd.index.intersection(ff_umd.index)
p20 = price_umd.reindex(idx).rolling(20).sum()
f20 = ff_umd.reindex(idx).rolling(20).sum()
both = pd.concat([p20, f20], axis=1, keys=["price", "ff"]).dropna()

print("=" * 66)
print(f"PRICE-BASED UMD vs FAMA-FRENCH UMD ({both.index.min().year}-{both.index.max().year})")
print("=" * 66)
print(f"\n  Daily UMD correlation:        {price_umd.reindex(idx).corr(ff_umd.reindex(idx)):+.2f}")
print(f"  20-day-sum UMD correlation:   {both['price'].corr(both['ff']):+.2f}")

# crash-signal agreement at FF's -5% threshold; find equivalent price threshold
FF_THR = -0.05
ff_crash = both["ff"] < FF_THR
# pick price threshold matching FF crash frequency
price_thr = both["price"].quantile(ff_crash.mean())
pr_crash = both["price"] < price_thr
agree = (ff_crash == pr_crash).mean()
both_fire = (ff_crash & pr_crash).sum()
print(f"\n  FF crash days (<{FF_THR}): {ff_crash.sum()} ({ff_crash.mean()*100:.1f}% of days)")
print(f"  Matched price threshold:    {price_thr:+.3f}  (same firing frequency)")
print(f"  Signal agreement:           {agree*100:.0f}%  | both fire together on {both_fire} days")

print(f"\n  Did each fire in known momentum-stress windows? (min 20d-sum)")
for wn, a, b in [("2009 mom crash", "2009-03-01", "2009-06-30"), ("COVID 2020", "2020-02-15", "2020-04-15"),
                 ("2022", "2022-01-01", "2022-12-31"), ("2008 GFC", "2008-09-01", "2008-12-31")]:
    w = both[(both.index >= a) & (both.index <= b)]
    if len(w):
        print(f"    {wn:16} FF min {w['ff'].min():+.3f} {'CRASH' if w['ff'].min()<FF_THR else '  -  '}"
              f"   price min {w['price'].min():+.3f} {'CRASH' if w['price'].min()<price_thr else '  -  '}")
print("\n  Replace if correlation is high AND it fires in the same crash windows.")
