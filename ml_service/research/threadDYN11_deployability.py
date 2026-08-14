"""thread DYN11 — can the off-cadence gate work WITHOUT the new FRED feed?

The validated finding uses OR(hy_oas, baa_aaa). baa_aaa is the single biggest deployment
blocker: it comes from fred_rates, which ends 2025-02 in the pickle, so shipping it means wiring
and monitoring a NEW live feed (DBAA/DAAA). hy_oas by contrast is ALREADY pulled daily by the
existing credit-gate cron -- zero new data infrastructure.

DYN4 hinted hy-alone might be enough (+0.052/+0.019 vs the OR's +0.043/+0.043) but that was 4
starts, which threadCANON has shown cannot resolve differences of this size. If hy-alone holds
up at 12 starts, the biggest blocker disappears and the change becomes engine-code-only.

Also tests whether the OFF-CADENCE part alone (no gate change at all -- the deployed hy_oas
gate, just applied every 5 days instead of every 20) captures most of the benefit. That would be
the minimal possible change: no new data, no new signal, only WHEN the existing rule is applied.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'research'))
import numpy as np
from livemirror_backtest import LiveMirrorBacktester

B = {"universe":"sp1500","mom_w":0.50,"val_w":0.35,"lv_w":0.15,"sec_w":0.0,"top_n":5,
     "rebal_days":20,"trailing_stop":0.40,"cap":0.10,"use_rp":False,"vol_scaling":True,
     "vol_target":0.15,"vol_lookback":40,"vol_scale_cap":1.0,"initial_capital":50_000.0,
     "leverage":1.49,"integer_shares":True,"financing_rate":0.063,"live_sizing":True}
DEPLOYED = {"credit_pct":0.95,"credit_derisk":0.5}
OFF = {"lev_recheck_every":5,"lev_band":0.10,"gate_offcadence":True}

VARIANTS = [
    ("MINIMAL: deployed gate, off-cadence", {**DEPLOYED, **OFF}),
    ("hy_oas only, derisk .30",            {"gate_cols":["hy_oas"],"gate_pct":0.95,
                                            "gate_derisk":0.30, **OFF}),
    ("OR(hy,baa) derisk .30 (validated)",  {"gate_cols":["hy_oas","baa_aaa"],"gate_pct":0.95,
                                            "gate_derisk":0.30, **OFF}),
]

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
    base=[]
    for st in STARTS:
        c=dict(B); c.update(DEPLOYED)
        base.append(stat(bt.run(st,"2025-12-31",c)["daily_values"]))
    print(f"\n=== {lbl} | vs DEPLOYED (rebal-only hy gate), {len(STARTS)} starts ===", flush=True)
    print(f"  {'variant':<38}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+Sh':>7}", flush=True)
    for name,extra in VARIANTS:
        dc,ds,dd=[],[],[]
        for i,st in enumerate(STARTS):
            c=dict(B); c.update(extra)
            a,s,d=stat(bt.run(st,"2025-12-31",c)["daily_values"])
            dc.append(a-base[i][0]); ds.append(s-base[i][1]); dd.append(d-base[i][2])
        a,b,c2=np.array(dc),np.array(ds),np.array(dd)
        print(f"  {name:<38}{a.mean()*100:>+8.2f}p{b.mean():>+9.3f}{c2.mean()*100:>+8.2f}p"
              f"{int((b>0).sum()):>4}/{len(b)}", flush=True)
    del bt
print("\nIf MINIMAL is close to the validated config, the change is engine-code-only:", flush=True)
print("no new FRED feed, no new signal -- only WHEN the existing gate is applied.", flush=True)
