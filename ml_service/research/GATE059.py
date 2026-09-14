"""PROMOTION GATE for EXP-059 candidates, on cached curves. A candidate must beat the base (FINAL@1.49) on:
  (1) 24 starts x 2 horizons: mean dSharpe > 0 with the block-bootstrap 90% CI not crossing zero on at least one horizon
      and P(>0) >= 70% on both;  (2) year-by-year: better in >= 50% of calendar years on both horizons, and no single
  year contributes > 40% of the total improvement;  (3) every sub-period dSharpe >= -0.02;  (4) untouched even-month
  starts agree in sign;  (5) best-5-days removed from BOTH: dSharpe still > 0;  (6) cost 2x (if cached): delta intact.
Run: python3 research/GATE059.py <arm> [stage]"""
import os, sys, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, "research")
ARM = sys.argv[1]; STAGE = sys.argv[2] if len(sys.argv) > 2 else "stage2"
rng = np.random.default_rng(1)
def st(v):
    v = v.dropna(); r = v.pct_change().dropna(); y = (v.index[-1]-v.index[0]).days/365.25
    return (v.iloc[-1]/v.iloc[0])**(1/y)-1, r.mean()/r.std()*np.sqrt(252), float(((v-v.cummax())/v.cummax()).min())
def yearly(c):
    c = c.dropna(); out = {}
    for y, seg in c.groupby(c.index.year):
        pr = c[c.index < pd.Timestamp(f"{y}-01-01")]
        if len(seg) >= 150 and not pr.empty: out[y] = seg.iloc[-1]/pr.iloc[-1]-1
    return out
def drop_best(v, k):
    r = v.dropna().pct_change().dropna(); r2 = r.drop(r.nlargest(k).index); return (1+r2).cumprod()*v.dropna().iloc[0]
SUBS = {"8yr": [("2018-20",2018,2020),("2021-22",2021,2022),("2023-26",2023,2026)], "26yr": [("2001-08",2001,2008),("2009-16",2009,2016),("2017-22",2017,2022),("2023-26",2023,2026)]}
verdict = {}
for hz in ("8yr", "26yr"):
    fb, fa = f"research/_v2_{hz}/exp059/{STAGE}_base.parquet", f"research/_v2_{hz}/exp059/{STAGE}_{ARM}.parquet"
    if not (os.path.exists(fb) and os.path.exists(fa)): print(f"{hz}: not cached"); continue
    B, A = pd.read_parquet(fb), pd.read_parquet(fa); cols = [c for c in A.columns if c in B.columns]
    b = np.array([st(B[c]) for c in cols]); a = np.array([st(A[c]) for c in cols]); d = a - b
    print(f"\n===== {hz} · {ARM} vs base · {len(cols)} starts =====")
    print(f"  levels: base {b[:,0].mean():+.2%}/{b[:,1].mean():.3f}/{b[:,2].mean():.1%}  arm {a[:,0].mean():+.2%}/{a[:,1].mean():.3f}/{a[:,2].mean():.1%}  d {d[:,0].mean()*100:+.2f}pp / {d[:,1].mean():+.3f} ({int((d[:,1]>0).sum())}/{len(cols)}) / {d[:,2].mean()*100:+.2f}pp ({int((d[:,2]>0).sum())}/{len(cols)})")
    # (0) exposure / level check: realized vol and vol-matched CAGR (a lower-exposure arm must still win per unit of risk)
    vb = np.mean([B[c].dropna().pct_change().std()*np.sqrt(252) for c in cols]); va = np.mean([A[c].dropna().pct_change().std()*np.sqrt(252) for c in cols])
    print(f"  (0) realized vol base {vb:.1%} arm {va:.1%} -> arm CAGR vol-matched to base ≈ {a[:,0].mean()*vb/va:+.2%} vs base {b[:,0].mean():+.2%} ({(a[:,0].mean()*vb/va-b[:,0].mean())*100:+.2f}pp); MaxDD/vol base {b[:,2].mean()/vb:.2f} arm {a[:,2].mean()/va:.2f}")
    # (1) bootstrap
    shd = []
    for _ in range(1500):
        c = cols[rng.integers(0, len(cols))]; rb_ = B[c].dropna().pct_change().dropna(); ra_ = A[c].dropna().pct_change().dropna(); j = rb_.index.intersection(ra_.index); rb_, ra_ = rb_[j].values, ra_[j].values
        Bk = 60; idx = rng.integers(0, len(rb_) - Bk, size=len(rb_)//Bk + 1); sel = np.concatenate([np.arange(i, i+Bk) for i in idx])[:len(rb_)]
        x, y = rb_[sel], ra_[sel]; shd.append(y.mean()/y.std()*np.sqrt(252) - x.mean()/x.std()*np.sqrt(252))
    shd = np.array(shd); ci = (np.percentile(shd, 5), np.percentile(shd, 95)); p = float(np.mean(shd > 0))
    print(f"  (1) bootstrap dSharpe {shd.mean():+.3f} 90% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] P(>0)={p:.0%}")
    # (2) year-by-year + concentration of the improvement
    yb = [yearly(B[c]) for c in cols]; ya = [yearly(A[c]) for c in cols]; ys = sorted(set().union(*[set(x) for x in yb]))
    dy = {y: np.mean([xa[y]-xb[y] for xa, xb in zip(ya, yb) if y in xa and y in xb]) for y in ys}
    wins = sum(1 for y in ys if dy[y] > 0); pos = {y: v for y, v in dy.items() if v > 0}; conc = (max(pos.values()) / sum(pos.values())) if pos and sum(dy.values()) > 0 else float("nan")
    print(f"  (2) years better {wins}/{len(ys)}; mean {np.mean(list(dy.values()))*100:+.2f}pp; largest positive year's share of total positive delta {conc:.0%}; " + " ".join(f"{y}:{v*100:+.1f}" for y, v in dy.items()))
    # (3) sub-periods
    subs = []
    for lab, y0, y1 in SUBS[hz]:
        ds = [st(A[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"])[1] - st(B[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"])[1] for c in cols if len(A[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"]) > 60]
        subs.append((lab, np.mean(ds) if ds else float("nan")))
    print(f"  (3) sub-period dSharpe: " + " ".join(f"{l} {v:+.3f}" for l, v in subs))
    # (4) even-month (untouched) starts
    ev = [c for c in cols if pd.Timestamp(c).month % 2 == 0]; od = [c for c in cols if pd.Timestamp(c).month % 2 == 1]
    de = np.mean([st(A[c])[1]-st(B[c])[1] for c in ev]) if ev else float("nan"); do = np.mean([st(A[c])[1]-st(B[c])[1] for c in od]) if od else float("nan")
    print(f"  (4) dSharpe odd-month (used) starts {do:+.3f} ({len(od)}) vs even-month (untouched) {de:+.3f} ({len(ev)})")
    # (5) best-5-days removed from both
    d5 = np.mean([st(drop_best(A[c], 5))[1] - st(drop_best(B[c], 5))[1] for c in cols])
    print(f"  (5) dSharpe with best-5 days removed from both: {d5:+.3f}")
    verdict[hz] = dict(dsh=d[:,1].mean(), p=p, ci_lo=ci[0], wins=wins/len(ys), conc=conc, subs=min(v for _, v in subs), even=de, d5=d5, ddd=d[:,2].mean(), dc=d[:,0].mean())
print("\nGATE:")
for hz, v in verdict.items():
    ok = v["dsh"] > 0 and v["p"] >= 0.70 and v["wins"] >= 0.5 and (np.isnan(v["conc"]) or v["conc"] <= 0.40) and v["subs"] >= -0.02 and (np.isnan(v["even"]) or v["even"] > 0) and v["d5"] > 0
    print(f"  {hz}: {'PASS' if ok else 'FAIL'}  (dSharpe {v['dsh']:+.3f}, P {v['p']:.0%}, CI_lo {v['ci_lo']:+.3f}, years {v['wins']:.0%}, conc {v['conc']:.0%}, min sub {v['subs']:+.3f}, even {v['even']:+.3f}, best5 {v['d5']:+.3f}, dMaxDD {v['ddd']*100:+.1f}pp, dCAGR {v['dc']*100:+.2f}pp)")
