"""The diversification prize: combine the distinct futures premia into one book and
see if it beats trend alone — especially whether it cuts the recent calm-market bleed
while keeping crisis convexity. Equal-risk-weight (each sleeve pre-scaled to 12% vol),
then re-scale the blend to 12%."""
import numpy as np, pandas as pd
import futures_lib as fl

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
eq = R["ES"].dropna()
vol = R.ewm(span=60, min_periods=20).std() * np.sqrt(252)
div = [m for m in mkts if m not in fl.EQUITY]

# sleeves (same methodology)
sig_tr = sum(np.sign(CCB.diff(L)) for L in [21, 63, 252]) / 3
trend, _ = fl.backtest(fl.vol_target(sig_tr, R), R)
trend_div, _ = fl.backtest(fl.vol_target(sig_tr[div], R[div]), R[div])
ramom = (CCB.diff(252) / NON.shift(252).abs()) / vol
xsmom, _ = fl.backtest(fl.vol_target(np.sign(ramom.sub(ramom.median(axis=1), axis=0)), R), R)
carry = pd.read_parquet("_sleeve_carry.parquet")["carry"]


def blend(sleeves, tgt=0.12):
    S = pd.concat(sleeves, axis=1).dropna()
    raw = S.mean(axis=1)                       # equal weight
    sc = (tgt / (raw.rolling(252, min_periods=60).std().shift(1) * np.sqrt(252))).clip(upper=3).fillna(1)
    return (raw * sc).dropna()


books = {
    "trend only":              trend,
    "trend + carry":           blend([trend, carry]),
    "trend + xsmom":           blend([trend, xsmom]),
    "trend + xsmom + carry":   blend([trend, xsmom, carry]),
    "trend_div + xsmom + carry": blend([trend_div, xsmom, carry]),
}

print("=== FULL SAMPLE ===")
for nm, r in books.items():
    fl.stats(r, nm)
print("\n=== RECENT (2015-2026) — does the blend fix the bleed? ===")
for nm, r in books.items():
    fl.stats(r[r.index.year >= 2015], nm)
print("\n=== 2020s only ===")
for nm, r in books.items():
    fl.stats(r[r.index.year >= 2020], nm)

best = books["trend + xsmom + carry"]
print("\n=== correlation among the three sleeves ===")
print(pd.concat([trend.rename("trend"), xsmom.rename("xsmom"), carry.rename("carry")],
                axis=1).dropna().corr().round(2).to_string())
fl.crisis_table(best, eq, "3-sleeve")
j = best.index.intersection(eq.index)
print(f"\n  corr(3-sleeve book, equity): {np.corrcoef(best[j], eq[j])[0,1]:+.2f}")

# save the best book for the equity-integration test
pd.DataFrame({"book": best}).to_parquet("_futures_book.parquet")
print("  [saved _futures_book.parquet]")
