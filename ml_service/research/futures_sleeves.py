"""Value, cross-sectional momentum, and sub-universe trend — all on the continuous
panel, same methodology as trend. Measures each standalone + cross-correlations +
crisis behaviour, to see which premia are distinct and worth combining."""
import numpy as np, pandas as pd
import futures_lib as fl

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
eq = R["ES"].dropna()
vol = R.ewm(span=60, min_periods=20).std() * np.sqrt(252)

# TREND (baseline)
sig_tr = sum(np.sign(CCB.diff(L)) for L in [21, 63, 252]) / 3
trend, _ = fl.backtest(fl.vol_target(sig_tr, R), R)

# TREND, long-horizon only (decayed less?)
sig_tr_lt = sum(np.sign(CCB.diff(L)) for L in [126, 252, 504]) / 3
trend_lt, _ = fl.backtest(fl.vol_target(sig_tr_lt, R), R)

# TREND on diversifying markets only (rates/FX/commodities — purer hedge)
div = [m for m in mkts if m not in fl.EQUITY]
trend_div, _ = fl.backtest(fl.vol_target(sig_tr[div], R[div]), R[div])

# CROSS-SECTIONAL MOMENTUM: risk-adj 12m return, cross-sectionally demeaned
ramom = (CCB.diff(252) / NON.shift(252).abs()) / vol
xs = ramom.sub(ramom.median(axis=1), axis=0)
xsmom, _ = fl.backtest(fl.vol_target(np.sign(xs), R), R)

# VALUE: 5-year reversal (buy what fell, sell what rose over ~5y)
sig_val = -np.sign(CCB.diff(1260))
value, _ = fl.backtest(fl.vol_target(sig_val, R), R)

print("=== STANDALONE SLEEVES (vol-scaled 12% target) ===")
fl.stats(trend,     "trend (all, 1/3/12m)")
fl.stats(trend_lt,  "trend (6/12/24m)")
fl.stats(trend_div, "trend (diversifying only)")
fl.stats(xsmom,     "cross-sec momentum")
fl.stats(value,     "value (5y reversal)")

S = pd.DataFrame({"trend": trend, "trend_lt": trend_lt, "xsmom": xsmom, "value": value}).dropna()
print("\n=== cross-correlation of sleeves ===")
print(S.corr().round(2).to_string())
print("\ncorr to equity (ES):")
for c in S:
    j = S[c].index.intersection(eq.index)
    print(f"  {c:<10} {np.corrcoef(S[c][j], eq[j])[0, 1]:+.2f}")

print("\n=== recent-era Sharpe (2015-2026) ===")
for nm, r in [("trend", trend), ("trend_lt", trend_lt), ("trend_div", trend_div),
              ("xsmom", xsmom), ("value", value)]:
    fl.stats(r[r.index.year >= 2015], nm)

for nm, r in [("trend", trend), ("xsmom", xsmom), ("value", value)]:
    fl.crisis_table(r, eq, nm)
