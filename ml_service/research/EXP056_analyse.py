"""Analyse the full final-base ladder (1.00/1.10/1.25/1.40/1.49 x gate 0.00) vs LIVE from cache.
Answers the return-spend question: CAGR at LIVE's OWN drawdown, and year-by-year at high leverage."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def st(v):
    v=v.dropna(); r=v.pct_change(); y=max((v.index[-1]-v.index[0]).days/365.25,1)
    return ((v.iloc[-1]/v.iloc[0])**(1/y)-1, r.mean()/r.std()*np.sqrt(252), float(((v-v.cummax())/v.cummax()).min()))
def yearly(c):
    c=c.dropna(); out={}
    for y,seg in c.groupby(c.index.year):
        if len(seg)<200: continue
        pr=c[c.index<pd.Timestamp(f"{y}-01-01")]
        if pr.empty: continue
        out[y]=seg.iloc[-1]/pr.iloc[-1]-1
    return out
LEVS=["1.00","1.10","1.25","1.40","1.49"]
for HZ in ("8yr","26yr"):
    D=f"research/_gw_{HZ}"
    load=lambda nm: pd.concat([pd.read_parquet(f"{D}/{nm}_used.parquet"),pd.read_parquet(f"{D}/{nm}_hold.parquet")],axis=1)
    L=load("LIVE"); cols=list(L.columns); l=np.array([st(L[c]) for c in cols]); n=len(cols)
    print(f"\n================ {HZ} — FINAL BASE LADDER vs LIVE ({n} starts: 12 used + 12 untouched) ================")
    print(f"  LIVE 1.49x+ovl+gate0.50   CAGR {l[:,0].mean():+.2%}  Sharpe {l[:,1].mean():.3f}  MaxDD {l[:,2].mean():.1%}  worst-start DD {l[:,2].min():.1%}")
    print(f"  {'lev':<6}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'worstDD':>9}{'dCAGR':>9}{'dShrp':>8}{'dMaxDD':>9}{'+Shrp':>7}{'+CAGR':>7}{'+DD':>6}")
    pts={}
    for lev in LEVS:
        f=f"{D}/L{lev}_g0.00_used.parquet"
        if not os.path.exists(f) or not os.path.exists(f"{D}/L{lev}_g0.00_hold.parquet"): print(f"  {lev:<6} (missing)"); continue
        F=load(f"L{lev}_g0.00"); x=np.array([st(F[c]) for c in cols]); d=x-l; pts[lev]=(x,F)
        print(f"  {lev:<6}{x[:,0].mean():>+9.2%}{x[:,1].mean():>8.3f}{x[:,2].mean():>8.1%}{x[:,2].min():>9.1%}{d[:,0].mean()*100:>+8.2f}p{d[:,1].mean():>+8.3f}{d[:,2].mean()*100:>+8.2f}p{int((d[:,1]>0).sum()):>4}/{n}{int((d[:,0]>0).sum()):>4}/{n}{int((d[:,2]>0).sum()):>3}/{n}")
    # iso-drawdown: interpolate CAGR (and Sharpe) at LIVE's own MaxDD
    xs=[(pts[k][0][:,2].mean(), pts[k][0][:,0].mean(), pts[k][0][:,1].mean(), k) for k in pts]; xs.sort()
    dd=[p[0] for p in xs]; cg=[p[1] for p in xs]; sh=[p[2] for p in xs]; target=l[:,2].mean()
    if min(dd)<=target<=max(dd):
        print(f"  ISO-DRAWDOWN at LIVE's MaxDD {target:.1%}: FINAL base delivers CAGR {np.interp(target,dd,cg):+.2%} (LIVE {l[:,0].mean():+.2%}, +{(np.interp(target,dd,cg)-l[:,0].mean())*100:.2f}pp) at Sharpe {np.interp(target,dd,sh):.3f} (LIVE {l[:,1].mean():.3f})")
    else:
        print(f"  ISO-DRAWDOWN: LIVE's MaxDD {target:.1%} is outside the ladder range [{min(dd):.1%},{max(dd):.1%}] — nearest: {xs[-1][3]}x at {xs[-1][0]:.1%} / {xs[-1][1]:+.2%}")
    # year-by-year for the top two leverages
    yl=[yearly(L[c]) for c in cols]
    for lev in ("1.40","1.49"):
        if lev not in pts: continue
        F=pts[lev][1]; yf=[yearly(F[c]) for c in cols]; rows=[]
        print(f"  YEAR-BY-YEAR at {lev}x vs LIVE:")
        for y in sorted(set().union(*[set(a) for a in yl])):
            pr=[(a[y],b[y]) for a,b in zip(yl,yf) if y in a and y in b]
            if not pr: continue
            A=np.mean([p[0] for p in pr]); B=np.mean([p[1] for p in pr]); w=sum(1 for a,b in pr if b>a); rows.append((y,B-A,w,len(pr)))
        line="    "+"  ".join(f"{y}:{d*100:+.1f}({w}/{m})" for y,d,w,m in rows); print(line)
        dd_=np.array([r[1] for r in rows]); k=int(np.argmax(dd_)); rest=np.delete(dd_,k)
        print(f"    years better {int((dd_>0).sum())}/{len(dd_)}  median {np.median(dd_)*100:+.2f}pp  mean {dd_.mean()*100:+.2f}pp  | drop best ({rows[k][0]}): {rest.mean()*100:+.2f}pp, positive {int((rest>0).sum())}/{len(rest)}")
