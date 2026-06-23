"""Last novel angle: does any futures market LEAD equities short-term? Lead-lag is the
one genuinely less-arbitraged edge (micro-structure / slow information diffusion across
assets). Test: does a market's recent return predict ES forward 1d/5d return, OOS?
Honest: if ICs ~0, equities are efficient w.r.t. cross-asset moves (consistent with the
crash-timing & factor-timing nulls)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import futures_lib as fl

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON).loc["2000-01-01":]
es = R["ES"]
cands = [m for m in ["ZN","ZB","DX","HG","CL","GC","VX","6E","6J","HSI","FESX","SR3","BRN"] if m in R.columns]

for h, lbl in [(1, "fwd 1d"), (5, "fwd 5d")]:
    fwd = es.rolling(h).sum().shift(-h)
    print(f"\n=== predicting ES {lbl} (IC of past-week return, 2000-2026) ===")
    rows = []
    for m in cands:
        past = R[m].rolling(5).sum()
        d = pd.concat([past.rename("x"), fwd.rename("y")], axis=1).dropna()
        if len(d) < 500: continue
        ic = spearmanr(d.x, d.y).statistic
        rows.append((m, ic, len(d)))
    for m, ic, n in sorted(rows, key=lambda r: -abs(r[1])):
        flag = "  <-- notable" if abs(ic) > 0.05 else ""
        print(f"  {m:<5} IC {ic:+.3f}  (n={n}){flag}")

# also: does ES's OWN past predict it (baseline autocorr)?
fwd1 = es.shift(-1)
own = pd.concat([es.rolling(5).sum().rename("x"), fwd1.rename("y")], axis=1).dropna()
print(f"\n  [baseline] ES own 5d->1d autocorr IC {spearmanr(own.x, own.y).statistic:+.3f}")
print("  (cross-asset ICs near 0 => equities efficient w.r.t. futures moves)")
