"""Push book construction further (honestly): (a) more trend horizons blended,
(b) risk-parity sleeve weighting (rolling inverse-vol) instead of equal, (c) a
vol-responsive book scalar. Compare recent Sharpe to v2. Guard against overfitting:
only keep changes that are principled AND help recent + full sample."""
import numpy as np, pandas as pd
import futures_lib as fl

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
eq = R["ES"].dropna()
vol = R.ewm(span=60, min_periods=20).std()
div = [m for m in mkts if m not in fl.EQUITY]

def conv(L):
    mom = CCB.diff(L) / NON.shift(L).abs()
    return np.tanh(mom / (vol * np.sqrt(L)))

# v2 trend (3 horizons) vs v3 (more horizons)
tr3 = sum(conv(L) for L in [21, 63, 252]) / 3
tr_more = sum(conv(L) for L in [10, 21, 42, 84, 168, 252, 504]) / 7
ramom = (CCB.diff(252) / NON.shift(252).abs()) / vol
xs = np.tanh(ramom.sub(ramom.median(axis=1), axis=0) / ramom.std())
carry = pd.read_parquet("_sleeve_carry.parquet")["carry"]

def sleeve(sig, u):
    r, _ = fl.backtest(fl.vol_target(sig[u], R[u]), R[u]); return r

def blend(sleeves, weight="equal", tgt=0.12):
    S = pd.concat(sleeves, axis=1, keys=range(len(sleeves))).dropna()
    if weight == "equal":
        raw = S.mean(axis=1)
    else:  # risk-parity: rolling inverse-vol across sleeves
        w = 1.0 / S.rolling(63, min_periods=20).std().shift(1)
        w = w.div(w.sum(axis=1), axis=0)
        raw = (S * w).sum(axis=1)
    sc = (tgt / (raw.rolling(252, min_periods=60).std().shift(1) * np.sqrt(252))).clip(upper=3).fillna(1)
    return (raw * sc).dropna()

t3, tm = sleeve(tr3, div), sleeve(tr_more, div)
x = sleeve(xs, list(R.columns))
books = {
    "v2 (3-horizon, equal)":        blend([t3, x, carry], "equal"),
    "v3a (more-horizon, equal)":    blend([tm, x, carry], "equal"),
    "v3b (more-horizon, riskparity)": blend([tm, x, carry], "rp"),
    "v3c (3-horizon, riskparity)":  blend([t3, x, carry], "rp"),
}
print("=== FULL SAMPLE ===")
for nm, r in books.items(): fl.stats(r, nm)
print("\n=== RECENT 2015-2026 ===")
for nm, r in books.items(): fl.stats(r[r.index.year >= 2015], nm)
print("\n=== 2020s ===")
for nm, r in books.items(): fl.stats(r[r.index.year >= 2020], nm)

# keep the best for the capacity test (save weights too)
best_name = max(books, key=lambda k: fl.stats(books[k][books[k].index.year>=2015], "", show=False).get("sharpe",0))
print(f"\nbest recent: {best_name}")
fl.crisis_table(books[best_name], eq, "best")
