"""Pairwise promotion gate: GATE059's checks applied to ARM vs any BASE arm (both cached at the same stage).
Run: python3 research/GATE059_pair.py <arm> <base> [stage]"""
import os, sys, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARM, BASE = sys.argv[1], sys.argv[2]; STAGE = sys.argv[3] if len(sys.argv) > 3 else "stage2"; rng = np.random.default_rng(1)
def st(v):
    v = v.dropna(); r = v.pct_change().dropna(); y = (v.index[-1]-v.index[0]).days/365.25
    return (v.iloc[-1]/v.iloc[0])**(1/y)-1, r.mean()/r.std()*np.sqrt(252), float(((v-v.cummax())/v.cummax()).min())
def yearly(c):
    c = c.dropna(); out = {}
    for y, seg in c.groupby(c.index.year):
        pr = c[c.index < pd.Timestamp(f"{y}-01-01")]
        if len(seg) >= 150 and not pr.empty: out[y] = seg.iloc[-1]/pr.iloc[-1]-1
    return out
def sh(r): return r.mean()/r.std()*np.sqrt(252)
SUBS = {"8yr": [("2018-20",2018,2020),("2021-22",2021,2022),("2023-26",2023,2026)], "26yr": [("2001-08",2001,2008),("2009-16",2009,2016),("2017-22",2017,2022),("2023-26",2023,2026)]}
CRISIS = {"8yr": [2020], "26yr": [2008, 2020]}
for hz in ("8yr", "26yr"):
    fb, fa = f"research/_v2_{hz}/exp059/{STAGE}_{BASE}.parquet", f"research/_v2_{hz}/exp059/{STAGE}_{ARM}.parquet"
    if not (os.path.exists(fb) and os.path.exists(fa)): print(f"{hz}: not cached"); continue
    B, A = pd.read_parquet(fb), pd.read_parquet(fa); cols = [c for c in A.columns if c in B.columns]
    b = np.array([st(B[c]) for c in cols]); a = np.array([st(A[c]) for c in cols]); d = a - b
    shd = []
    for _ in range(1500):
        c = cols[rng.integers(0, len(cols))]; rb_ = B[c].dropna().pct_change().dropna(); ra_ = A[c].dropna().pct_change().dropna(); j = rb_.index.intersection(ra_.index); rb_, ra_ = rb_[j].values, ra_[j].values
        Bk = 60; idx = rng.integers(0, len(rb_) - Bk, size=len(rb_)//Bk + 1); sel = np.concatenate([np.arange(i, i+Bk) for i in idx])[:len(rb_)]
        x, y = rb_[sel], ra_[sel]; shd.append(y.mean()/y.std()*np.sqrt(252) - x.mean()/x.std()*np.sqrt(252))
    shd = np.array(shd); ci = (np.percentile(shd, 5), np.percentile(shd, 95)); p = float(np.mean(shd > 0))
    yb = pd.DataFrame([yearly(B[c]) for c in cols]).mean(); ya = pd.DataFrame([yearly(A[c]) for c in cols]).mean(); dy = ya - yb; pos = dy[dy > 0]
    conc = pos.max()/pos.sum() if len(pos) else float("nan")
    subs = {}
    for nm, y0, y1 in SUBS[hz]:
        vals = []
        for c in cols:
            rb_ = B[c].dropna().pct_change().dropna(); ra_ = A[c].dropna().pct_change().dropna(); j = rb_.index.intersection(ra_.index); rb_, ra_ = rb_[j], ra_[j]
            m = (rb_.index.year >= y0) & (rb_.index.year <= y1)
            if m.sum() > 120: vals.append(sh(ra_[m]) - sh(rb_[m]))
        subs[nm] = float(np.mean(vals)) if vals else float("nan")
    odd = [c for c in cols if pd.Timestamp(c).month % 2 == 1]; even = [c for c in cols if pd.Timestamp(c).month % 2 == 0]
    de = np.mean([st(A[c])[1]-st(B[c])[1] for c in even]) if even else float("nan"); do = np.mean([st(A[c])[1]-st(B[c])[1] for c in odd]) if odd else float("nan")
    exc = []
    for c in cols:
        rb_ = B[c].dropna().pct_change().dropna(); ra_ = A[c].dropna().pct_change().dropna(); j = rb_.index.intersection(ra_.index); rb_, ra_ = rb_[j], ra_[j]
        m = ~rb_.index.year.isin(CRISIS[hz]); exc.append(sh(ra_[m]) - sh(rb_[m]))
    ok = (p >= 0.70) and ((dy > 0).mean() >= 0.5) and (not (conc == conc) or conc <= 0.40) and (min(subs.values()) >= -0.02) and (np.sign(de) == np.sign(do))
    print(f"{hz}: {'PASS' if ok else 'FAIL'}  {ARM} vs {BASE} · dCAGR {d[:,0].mean()*100:+.2f}pp dSharpe {d[:,1].mean():+.3f} ({int((d[:,1]>0).sum())}/{len(cols)}) dMaxDD {d[:,2].mean()*100:+.2f}pp · P {p:.0%} CI [{ci[0]:+.3f},{ci[1]:+.3f}] · years {int((dy>0).sum())}/{len(dy)} conc {conc:.0%} · subs {' '.join(f'{k} {v:+.3f}' for k,v in subs.items())} · odd {do:+.3f} even {de:+.3f} · ex-crisis dSharpe {np.mean(exc):+.3f} ({int((np.array(exc)>0).sum())}/{len(cols)})")
