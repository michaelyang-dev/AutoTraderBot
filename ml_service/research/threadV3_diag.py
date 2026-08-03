"""
THREAD V3 — integrity diagnostics for the V1/V2 momentum-pick census.

1. Why are 75.6% of 26yr-pickle picks labelled OTHER (not in clean PIT SP500/400/600)?
   Ticker-namespace mismatch, or a genuinely wider universe?
2. Is ret_252d (the momentum feature) consistent with the actual price series? A median
   12-1 score of +220% on the top-5 is extreme; confirm it is real, not a corrupt feature.
3. Outlier sensitivity of the 2018-25 non-SP500 edge.
"""
import os, sys, pickle
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402

R = os.path.dirname(os.path.abspath(__file__))
CLEAN = "data/wrds/complete_sp1500_universe.pkl"


def hdr(s):
    print("\n" + "=" * 96); print(s); print("=" * 96, flush=True)


hdr("1. OTHER-label diagnosis (26yr pickle)")
c = pickle.load(open(CLEAN, "rb"))
ever = set()
for k in ["sp500_mem", "sp400_mem", "sp600_mem"]:
    for v in c[k].values():
        ever |= set(v)
clean_cols = set(c["prices_df"].columns)
del c
o = pickle.load(open("data/wrds/sp1500_universe_2000.pkl", "rb"))
old_cols = set(o["prices_df"].columns)
old_ever = set()
for k in ["sp500_mem", "sp400_mem", "sp600_mem"]:
    for v in o[k].values():
        old_ever |= set(v)
del o
print(f"clean pickle: {len(clean_cols)} price columns, {len(ever)} ever-SP1500 tickers")
print(f"26yr  pickle: {len(old_cols)} price columns, {len(old_ever)} ever-'member' tickers")
print(f"ticker overlap clean-vs-26yr price cols: {len(clean_cols & old_cols)} "
      f"({len(clean_cols & old_cols)/len(clean_cols):.1%} of clean)")

df = pd.read_parquet(os.path.join(R, "_v2_mom_picks.parquet"))
long = df[df.period == "2001-25"]
oth = long[long.tier == "OTHER"]["sym"].unique()
print(f"\n26yr picks labelled OTHER: {len(oth)} unique tickers")
print(f"  ...of which EVER in clean SP1500 membership (any date): "
      f"{len([s for s in oth if s in ever])} ({len([s for s in oth if s in ever])/len(oth):.1%})")
print(f"  ...of which present in the CLEAN price matrix at all:  "
      f"{len([s for s in oth if s in clean_cols])} ({len([s for s in oth if s in clean_cols])/len(oth):.1%})")
print("  sample OTHER tickers:", sorted(oth)[:25])
print("\n=> if 'ever in clean SP1500' is LOW, the 26yr pickle universe is genuinely wider than")
print("   SP1500 and its tier split is not interpretable. If HIGH, it is a PIT-date artifact.")

hdr("2. ret_252d feature vs price-derived 252d return (clean pickle, on the actual picks)")
bt = FastBacktester(universe_path=CLEAN)
px = bt.prices
pos = {d: i for i, d in enumerate(px.index)}
short = df[df.period == "2018-25"]
recs = []
for date, g in short.groupby("date"):
    if date not in pos or pos[date] < 252:
        continue
    fm = bt.uni.get_feature_map(date, "ret_252d")
    i = pos[date]
    for sym in g["sym"]:
        if sym not in px.columns:
            continue
        s = px[sym]
        p0, p1 = s.iloc[i - 252], s.iloc[i]
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            recs.append((sym, date, fm.get(sym, np.nan), p1 / p0 - 1))
chk = pd.DataFrame(recs, columns=["sym", "date", "feat", "price_derived"]).dropna()
print(f"n={len(chk)}  corr(feature, price-derived 252d ret) = {chk['feat'].corr(chk['price_derived']):.4f}")
print(f"  median feature {chk['feat'].median():+.1%}   median price-derived {chk['price_derived'].median():+.1%}")
print(f"  median abs diff {(chk['feat']-chk['price_derived']).abs().median():.4f}")
print(f"  fraction within 5pp: {((chk['feat']-chk['price_derived']).abs()<0.05).mean():.1%}")
print("\n  10 highest-momentum picks (feature vs price-derived), 2018-25:")
for _, r in chk.nlargest(10, "feat").iterrows():
    print(f"    {r['sym']:<6} {str(r['date'])[:10]}  feat {r['feat']:>+8.1%}  price {r['price_derived']:>+8.1%}")

hdr("3. Outlier sensitivity, 2018-25 non-SP500 vs SP500 (fwd60)")
for tag, sub in [("SP500", short[short.tier == "SP500"]), ("NON-SP500", short[short.tier != "SP500"])]:
    x = sub["fwd60"].dropna().values
    xs = np.sort(x)
    print(f"  {tag:<10} n={len(x)}  mean {x.mean():+.2%}  "
          f"ex-top5 {xs[:-5].mean():+.2%}  ex-top10 {xs[:-10].mean():+.2%}  "
          f"median {np.median(x):+.2%}  "
          f"top-5 contribution to mean: {(x.mean()-xs[:-5].mean()):+.2%}")
print("\n  biggest single fwd60 winners among NON-SP500 picks:")
w = short[short.tier != "SP500"].nlargest(8, "fwd60")
for _, r in w.iterrows():
    print(f"    {r['sym']:<6} {str(r['date'])[:10]} {r['tier']}  mom {r['mom_raw']:>+7.0%}  fwd60 {r['fwd60']:>+8.1%}")
print("\n  biggest single fwd60 losers among NON-SP500 picks:")
for _, r in short[short.tier != "SP500"].nsmallest(8, "fwd60").iterrows():
    print(f"    {r['sym']:<6} {str(r['date'])[:10]} {r['tier']}  mom {r['mom_raw']:>+7.0%}  fwd60 {r['fwd60']:>+8.1%}")
