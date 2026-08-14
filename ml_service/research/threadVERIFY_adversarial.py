"""thread VERIFY — adversarial re-check of the off-cadence result BEFORE it touches production.

Three specific ways the reported numbers could be inflated, none of which the earlier threads
tested:

1. EXPOSURE, not skill. threadDYN11 compared MINIMAL/hy-only directly against DEPLOYED and did
   NOT report avg_gross. If the off-cadence variants simply run MORE gross, part of the +1.58pp
   is leverage, not timing. Every earlier DYN thread used a matched-exposure control precisely
   to prevent this; DYN11 dropped it, and DYN11 is the thread the deployment ladder rests on.

2. WHICH HALF is doing the work? "MINIMAL" re-applies BOTH vol_scale AND the gate every 5 days.
   If the benefit is really the vol half, then shipping it as "apply the credit gate promptly" is
   mislabelled and the credit-gate story is wrong even if the number is right.

3. CONCENTRATION. If most of the gain lands in a handful of days (March 2020, Oct 2008), it is a
   lottery ticket, not a policy. Reported as the share of total outperformance contributed by
   the best 10 and best 20 days.

Anything that fails here does NOT get implemented.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'research'))
import numpy as np, pandas as pd
from livemirror_backtest import LiveMirrorBacktester

B = {"universe":"sp1500","mom_w":0.50,"val_w":0.35,"lv_w":0.15,"sec_w":0.0,"top_n":5,
     "rebal_days":20,"trailing_stop":0.40,"cap":0.10,"use_rp":False,"vol_scaling":True,
     "vol_target":0.15,"vol_lookback":40,"vol_scale_cap":1.0,"initial_capital":50_000.0,
     "leverage":1.49,"integer_shares":True,"financing_rate":0.063,"live_sizing":True}
DEP = {"credit_pct":0.95,"credit_derisk":0.5}
OFF = {"lev_recheck_every":5,"lev_band":0.10,"gate_offcadence":True}

ARMS = [
    ("DEPLOYED (baseline)",            {**DEP}),
    ("MINIMAL both off-cadence",       {**DEP, **OFF}),
    ("VOL off-cadence ONLY",           {**DEP, "lev_recheck_every":5, "lev_band":0.10}),  # gate stays rebal-only
    ("GATE off-cadence, vol OFF",      {**DEP, **OFF, "vol_scaling":False}),
    ("hy_oas only derisk .30",         {"gate_cols":["hy_oas"],"gate_pct":0.95,"gate_derisk":0.30, **OFF}),
]
CONST=[0.60,0.80,1.00,1.20,1.49]

def cd(bt):
    for k in ["_fin_growth","_ev","_estimates","_price_targets","_revenue_surprise",
              "_beat_streak","_earnings_signals"]:
        setattr(bt.uni,k,{})
    bt._si_ranks_by_month={}; bt._si_change_ranks_by_month={}; bt._si_months=[]
    bt.uni._short_interest_rank={}; bt.uni._si_change_rank={}

def stat(v):
    dr=v.pct_change().dropna(); yrs=max((v.index[-1]-v.index[0]).days/365.25,1)
    return ((v.iloc[-1]/v.iloc[0])**(1/yrs)-1,
            dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0,
            ((v-v.cummax())/v.cummax()).min())

for path,lbl,yrs_ in [("data/wrds/sp1500_universe_2000.pkl","26yr",[2001,2002]),
                      ("data/wrds/complete_sp1500_universe.pkl","8yr",[2018,2019])]:
    STARTS=[f"{y}-{m:02d}-03" for y in yrs_ for m in range(1,13,2)]
    bt=LiveMirrorBacktester(universe_path=path); cd(bt)
    # constant-leverage reference for the matched-exposure control
    gx,cy,sy,dy=[],[],[],[]
    for lev in CONST:
        c=dict(B); c.update({"leverage":lev,"lev_policy":"constant"})
        cs,ss,ds,gs=[],[],[],[]
        for st in STARTS:
            m=bt.run(st,"2025-12-31",c); a,s,d=stat(m["daily_values"])
            cs.append(a);ss.append(s);ds.append(d);gs.append(m["avg_gross"])
        gx.append(np.mean(gs)); cy.append(np.mean(cs)); sy.append(np.mean(ss)); dy.append(np.mean(ds))
    o=np.argsort(gx); gx=np.array(gx)[o];cy=np.array(cy)[o];sy=np.array(sy)[o];dy=np.array(dy)[o]
    ref=lambda g:(float(np.interp(g,gx,cy)),float(np.interp(g,gx,sy)),float(np.interp(g,gx,dy)))

    print(f"\n=== {lbl} | {len(STARTS)} starts | MATCHED-EXPOSURE residuals (the honest test) ===", flush=True)
    print(f"  {'arm':<30}{'avgGross':>10}{'CAGR':>9}{'Sharpe':>8}{'dSh_raw':>9}{'dSh_matched':>13}", flush=True)
    basec=bases=None
    for name,extra in ARMS:
        cs,ss,ds,gs=[],[],[],[]
        for st in STARTS:
            c=dict(B); c.update(extra)
            m=bt.run(st,"2025-12-31",c); a,s,d=stat(m["daily_values"])
            cs.append(a);ss.append(s);ds.append(d);gs.append(m["avg_gross"])
        C,S,G=np.mean(cs),np.mean(ss),np.mean(gs)
        if basec is None: basec,bases=C,S; raw=""
        else: raw=f"{S-bases:>+9.3f}"
        rc,rs,rd=ref(G)
        print(f"  {name:<30}{G:>10.4f}{C:>+9.2%}{S:>8.3f}{raw:>9}{S-rs:>+13.3f}", flush=True)
    del bt
print("\ndSh_raw = vs DEPLOYED (what DYN11 reported). dSh_matched = vs constant leverage at the", flush=True)
print("SAME realised gross. If matched << raw, the DYN11 numbers were partly an exposure effect.", flush=True)
