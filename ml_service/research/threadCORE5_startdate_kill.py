import os, sys
os.chdir("/Users/michaelslyanggmail.com/Downloads/auto-trader 2/ml_service")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(),'research'))
import numpy as np
from livemirror_backtest import LiveMirrorBacktester
B = {"universe":"sp1500","mom_w":0.50,"val_w":0.35,"lv_w":0.15,"sec_w":0.0,"top_n":5,
     "rebal_days":20,"trailing_stop":0.40,"cap":0.10,"use_rp":False,"vol_scaling":True,
     "vol_target":0.15,"vol_lookback":40,"vol_scale_cap":1.0,"initial_capital":50_000.0,
     "leverage":1.49,"integer_shares":True,"financing_rate":0.063}
SIMPLE = {"mom_w":0.85,"val_w":0.00,"lv_w":0.15}
def cd(bt):
    for k in ["_fin_growth","_ev","_estimates","_price_targets","_revenue_surprise","_beat_streak","_earnings_signals"]:
        setattr(bt.uni,k,{})
    bt._si_ranks_by_month={}; bt._si_change_ranks_by_month={}; bt._si_months=[]
    bt.uni._short_interest_rank={}; bt.uni._si_change_rank={}
def stat(v):
    dr=v.pct_change().dropna(); yrs=max((v.index[-1]-v.index[0]).days/365.25,1)
    return ((v.iloc[-1]/v.iloc[0])**(1/yrs)-1, dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0)
# 12 monthly starts spanning 2017 and 2018 -- if the effect is real the DELTA should be
# consistently positive, not flip with the entry month.
STARTS=[f"2017-{m:02d}-03" for m in range(1,13,2)] + [f"2018-{m:02d}-02" for m in range(1,13,2)]
bt=LiveMirrorBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl"); cd(bt)
print("=== START-DATE SENSITIVITY of the bull-only value removal (26yr universe) ===", flush=True)
print(f"  {'start':<12}{'dep CAGR':>10}{'simple':>10}{'dCAGR':>9}{'dSharpe':>9}", flush=True)
ds=[]
for st in STARTS:
    try:
        m0=bt.run(st,"2025-12-31",dict(B)); c0,s0=stat(m0["daily_values"])
        c1=dict(B); c1.update(SIMPLE)
        m1=bt.run(st,"2025-12-31",c1); c1v,s1=stat(m1["daily_values"])
    except Exception as e:
        print(f"  {st:<12} ERR {e}", flush=True); continue
    ds.append((c1v-c0, s1-s0))
    print(f"  {st:<12}{c0:>+10.2%}{c1v:>+10.2%}{(c1v-c0)*100:>+8.2f}p{s1-s0:>+9.3f}", flush=True)
a=np.array([d[0] for d in ds]); b=np.array([d[1] for d in ds])
print(f"\n  dCAGR  mean {a.mean()*100:+.2f}p  median {np.median(a)*100:+.2f}p  min {a.min()*100:+.2f}p  max {a.max()*100:+.2f}p", flush=True)
print(f"  dSharpe mean {b.mean():+.3f}  positive on {int((b>0).sum())}/{len(b)} starts", flush=True)
print("\n  A real effect is positive on MOST starts. If it flips sign with the entry month,", flush=True)
print("  the 4-start averages I have been reporting were luck of the calendar.", flush=True)
