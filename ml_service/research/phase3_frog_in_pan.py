"""
Phase 3 (cont.) — Frog-in-the-Pan, Da-Gurun-Warachka (2014).
Momentum driven by CONTINUOUS information (steady drift, many small same-sign
days) underreacts more and persists longer than momentum from DISCRETE jumps
(few big days that grab attention and mean-revert). Information Discreteness:
    ID = sign(PRET) * (%neg_days - %pos_days)
Low/negative ID = smooth/continuous winner (preferred). High ID = jumpy winner.

Test as a SELECTION refinement on the top-5 momentum sleeve: among momentum
winners (top pool), does picking the SMOOTHEST 5 beat plain top-5, and does it
have lower drawdown (the persistence/reversal claim)? Jumpy-5 shown as control.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

FORM, SKIP, POOL, TOPN = 252, 21, 50, 5

bt = FastBacktester(); prices = bt.prices; rets = prices.pct_change()
bt.uni.get_sp500 = bt._get_sp1500
dates = [d for d in prices.index if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")]
allidx = list(prices.index)

recs = []
for i in range(0, len(dates) - 21, 20):
    d = dates[i]; loc = allidx.index(d)
    if loc < FORM + 5:
        continue
    members = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
    win = allidx[loc - FORM: loc - SKIP]
    if len(win) < 100:
        continue
    Y = rets.loc[win, members].values
    valid = np.isfinite(Y).sum(axis=0) > 0.8 * len(win)
    cols = [members[j] for j in range(len(members)) if valid[j]]
    Yv = np.nan_to_num(Y[:, valid])
    mom = Yv.sum(axis=0)
    pos = (Yv > 0).mean(axis=0); neg = (Yv < 0).mean(axis=0)
    ID = np.sign(mom) * (neg - pos)               # low = continuous winner
    fwd = dates[i + 20]
    p0 = prices.loc[d, cols].values; p1 = prices.loc[fwd, cols].values
    fr = np.where((p0 > 0) & np.isfinite(p0) & np.isfinite(p1), p1 / p0 - 1, np.nan)
    recs.append((fwd, np.array(cols), mom, ID, fr))


def curve(kind):
    out = []
    for fwd, cols, mom, ID, fr in recs:
        pool = np.argsort(-mom)[:POOL]            # momentum winners pool
        if kind == "plain":
            pick = np.argsort(-mom)[:TOPN]
        elif kind == "smooth":                    # lowest ID within pool
            pick = pool[np.argsort(ID[pool])[:TOPN]]
        else:                                     # jumpy: highest ID within pool
            pick = pool[np.argsort(-ID[pool])[:TOPN]]
        out.append((fwd, np.nanmean(fr[pick])))
    s = pd.Series([r for _, r in out], index=[f for f, _ in out]).dropna()
    v = (1 + s).cumprod()
    cagr = v.iloc[-1] ** (252 / (len(s) * 20)) - 1
    sh = (s.mean() / s.std()) * np.sqrt(252 / 20)
    mdd = ((v - v.cummax()) / v.cummax()).min()
    return cagr, sh, mdd


print("=" * 64)
print(f"PHASE 3: FROG-IN-THE-PAN (top-5 from {POOL}-name momentum pool, 2018-2025)")
print("=" * 64)
print(f"\n  {'variant':<28}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}")
for kind, lab in [("plain", "plain top-5 momentum"), ("smooth", "smoothest-5 (low ID, FIP)"), ("jumpy", "jumpiest-5 (high ID, ctrl)")]:
    cg, sh, md = curve(kind)
    print(f"  {lab:<28}{cg*100:>7.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)
print("\nFIP helps if smoothest-5 > plain on Sharpe/MaxDD and jumpy-5 is clearly worse.")
