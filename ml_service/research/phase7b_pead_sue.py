"""Phase 7b — SUE / post-earnings drift (PEAD), same 3-gate test as revisions.
Signal: suescore (standardized unexpected earnings) from the most recent EPS
announcement within 90 days (the PEAD drift window). Point-in-time via anndats.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np, pandas as pd
from scipy.stats import spearmanr

bt = FastBacktester(); prices = bt.prices; bt.uni.get_sp500 = bt._get_sp1500
allidx = list(prices.index)
print("loading IBES SUE...", flush=True)
su = pd.read_parquet("data/wrds/ibes_surprise.parquet",
                     columns=["OFTIC", "MEASURE", "anndats", "suescore"],
                     filters=[("MEASURE", "=", "EPS")])
su["anndats"] = pd.to_datetime(su["anndats"])
su = su[su["anndats"] >= "2016-06-01"].dropna(subset=["OFTIC", "suescore"]).sort_values("anndats")
SUE = {t: (g["anndats"].values, g["suescore"].values) for t, g in su.groupby("OFTIC") if t in prices.columns}
print(f"tickers with SUE in panel: {len(SUE)}", flush=True)

def sue_asof(t, d):
    a = SUE.get(t)
    if a is None: return np.nan
    ad, sc = a; i = np.searchsorted(ad, np.datetime64(d), "right") - 1
    if i < 0 or (np.datetime64(d) - ad[i]) > np.timedelta64(90, "D"): return np.nan
    return sc[i]

dates = [d for d in prices.index if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")]
ics, corrs, g3h, g3l, plain5, filt5 = [], [], [], [], [], []
for k in range(0, len(dates) - 21, 20):
    d = dates[k]; loc = allidx.index(d)
    if loc < 260: continue
    members = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
    p0, p20, p252, pf = prices.loc[d], prices.loc[allidx[loc-20]], prices.loc[allidx[loc-252]], prices.loc[dates[k+20]]
    rows = []
    for m in members:
        a, a20, a252, af = p0.get(m), p20.get(m), p252.get(m), pf.get(m)
        if not (a and a252 and af) or np.isnan(a) or np.isnan(a252) or np.isnan(af): continue
        mom = (a/a252-1) - (a/a20-1) if a20 and not np.isnan(a20) else (a/a252-1)
        rows.append((m, mom, sue_asof(m, d), af/a-1))
    if len(rows) < 50: continue
    df = pd.DataFrame(rows, columns=["sym","mom","sue","fwd"]); v = df.dropna(subset=["sue"])
    if len(v) > 30:
        ics.append(spearmanr(v["sue"], v["fwd"]).correlation); corrs.append(spearmanr(v["sue"], v["mom"]).correlation)
        tq = v[v["mom"] >= v["mom"].quantile(0.80)]
        if len(tq) >= 10:
            med = tq["sue"].median(); g3h.append(tq[tq["sue"]>med]["fwd"].mean()); g3l.append(tq[tq["sue"]<=med]["fwd"].mean())
    plain5.append(df.nlargest(5,"mom")["fwd"].mean())
    pos = df[df["sue"]>0]; filt5.append(pos.nlargest(5,"mom")["fwd"].mean() if len(pos)>=5 else df.nlargest(5,"mom")["fwd"].mean())

def ann(s):
    s = pd.Series(s).dropna(); eq=(1+s).cumprod(); return eq.iloc[-1]**(1/(len(s)*20/252))-1, (s.mean()/s.std())*np.sqrt(252/20)
print("\n"+"="*64); print("PHASE 7b: SUE / PEAD (post-earnings drift), 2018-2025"); print("="*64)
print(f"\n  GATE 1 IC vs fwd        = {np.nanmean(ics):+.4f}  (>0.02 useful)")
print(f"  GATE 2 corr w/ momentum = {np.nanmean(corrs):+.3f}")
hi, lo = np.nanmean(g3h), np.nanmean(g3l)
print(f"  GATE 3 winners: high-SUE {hi*100:+.2f}% vs low-SUE {lo*100:+.2f}% (spread {(hi-lo)*100:+.2f}pp)")
cp, sp = ann(plain5); cf, sf = ann(filt5)
print(f"\n  top-5 mom plain {cp*100:.1f}% Sh {sp:.2f}  |  +positive-SUE {cf*100:.1f}% Sh {sf:.2f}")
