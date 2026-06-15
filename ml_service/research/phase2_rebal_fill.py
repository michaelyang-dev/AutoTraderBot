import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))); os.environ["OMP_NUM_THREADS"]="1"
from main_production_backtest import FastBacktester
import numpy as np
STARTS=["2018-01-02","2018-01-03","2018-01-04","2018-01-05","2018-01-08"]
def clear(bt):
    for k in ["_fin_growth","_ev","_estimates","_price_targets","_revenue_surprise","_beat_streak","_earnings_signals"]:
        setattr(bt.uni,k,{})
    bt._si_ranks_by_month={};bt._si_change_ranks_by_month={};bt._si_months=[]
    bt.uni._short_interest_rank={};bt.uni._si_change_rank={}
def st(v):
    r=v.pct_change().dropna(); sh=(r.mean()/r.std())*np.sqrt(252) if r.std()>0 else 0
    return (v.iloc[-1]/v.iloc[0])**(252/len(v))-1, sh
bt=FastBacktester(); clear(bt)
DEP={"universe":"sp1500","mom_w":.50,"val_w":.35,"lv_w":.15,"sec_w":0.0,"top_n":5,"trailing_stop":0.40,"cap":0.15,"bear_weights":{"mom":0.10,"val":0.30,"s5":0.50,"s3":0.10}}
print("REBAL fill-in 19/21 (cliff vs noisy plateau?), start-day avg:", flush=True)
print(f"  {'rebal':>6}{'CAGR':>8}{'Sharpe':>8}", flush=True)
res={}
for rd in [18,19,20,21,22]:
    cs=[];ss=[]
    for s in STARTS:
        m=bt.run(s,"2025-12-31",{**DEP,"rebal_days":rd}); c,sh=st(m["daily_values"]); cs.append(c);ss.append(sh)
    res[rd]=(np.mean(cs),np.mean(ss))
    print(f"  {rd:>6}{np.mean(cs)*100:>7.1f}%{np.mean(ss):>8.2f}", flush=True)
neigh=np.mean([res[r][0] for r in [18,19,21,22]])*100
print(f"\n  20d peak: {res[20][0]*100:.1f}% | mean of 18/19/21/22 neighbors: {neigh:.1f}%", flush=True)
print(f"  honest forward base (haircut toward neighborhood): ~{neigh:.0f}% at 1x", flush=True)
