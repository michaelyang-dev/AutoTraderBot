"""thread GM4 — re-test the gross_margin bound A/B under the many-start standard.

threadGM reported the [-1,1] gm bound costing -1.50pp CAGR on the 8yr and gaining +0.88pp on the
26yr, from 3 and 2 starts. threadCANON has since measured the per-start sigma of the 8yr CAGR at
7.07pp and CORE5 measured a delta's per-start sigma at ~2.30pp. A -1.50pp difference off 3 starts
is inside that noise, so the number I reported was not measurable at the precision I implied.

This does not change whether the fix ships -- it removes definitionally-impossible values
(gm = 4561.72 on a ratio bounded by 1) and it fixed a live book that had collapsed to 15 names.
Correctness, not performance, is the justification. But the PERFORMANCE claim needs correcting.

12 monthly starts per horizon; sign-consistency is the test, not the mean.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'research'))
import numpy as np
import strategies.multi_strategy_engine as M
from livemirror_backtest import LiveMirrorBacktester

B = {"universe":"sp1500","mom_w":0.50,"val_w":0.35,"lv_w":0.15,"sec_w":0.0,"top_n":5,
     "rebal_days":20,"trailing_stop":0.40,"cap":0.10,"use_rp":False,"vol_scaling":True,
     "vol_target":0.15,"vol_lookback":40,"vol_scale_cap":1.0,"initial_capital":50_000.0,
     "leverage":1.49,"integer_shares":True,"financing_rate":0.063}

_SHIPPED = M._sane_gross_margin
def _identity(gm): return gm

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

for path,lbl,yrs_ in [("data/wrds/complete_sp1500_universe.pkl","8yr",[2018,2019]),
                      ("data/wrds/sp1500_universe_2000.pkl","26yr",[2001,2002])]:
    STARTS=[f"{y}-{m:02d}-03" for y in yrs_ for m in range(1,13,2)]
    bt=LiveMirrorBacktester(universe_path=path); cd(bt)
    print(f"\n=== {lbl}: gm bound ON minus OFF, {len(STARTS)} starts ===", flush=True)
    print(f"  {'start':<12}{'OFF CAGR':>10}{'ON CAGR':>10}{'dCAGR':>9}{'dSharpe':>9}", flush=True)
    dc,ds=[],[]
    for st in STARTS:
        M._sane_gross_margin=_identity
        m0=bt.run(st,"2025-12-31",dict(B)); a0,s0,_=stat(m0["daily_values"])
        M._sane_gross_margin=_SHIPPED
        m1=bt.run(st,"2025-12-31",dict(B)); a1,s1,_=stat(m1["daily_values"])
        dc.append(a1-a0); ds.append(s1-s0)
        print(f"  {st:<12}{a0:>+10.2%}{a1:>+10.2%}{(a1-a0)*100:>+8.2f}p{s1-s0:>+9.3f}", flush=True)
    a,b=np.array(dc),np.array(ds)
    print(f"\n  dCAGR   mean {a.mean()*100:+.2f}p  median {np.median(a)*100:+.2f}p  "
          f"sigma {a.std(ddof=1)*100:.2f}pp  positive {int((a>0).sum())}/{len(a)}", flush=True)
    print(f"  dSharpe mean {b.mean():+.3f}  positive {int((b>0).sum())}/{len(b)}", flush=True)
    del bt
M._sane_gross_margin=_SHIPPED
print("\nIf this is a coin flip, the honest statement is 'no measurable performance effect',", flush=True)
print("and the fix stands purely on correctness.", flush=True)
