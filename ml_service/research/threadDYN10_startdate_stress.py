"""thread DYN10 — apply the test that KILLED the value finding to the LEVERAGE finding.

threadCORE5 showed that a 4-start average hides enormous start-date variance: the value-weight
delta ranged -3.91pp to +3.70pp across 12 monthly starts and was positive on only 6 of 12. The
per-start sigma was ~2.30pp, roughly 4x the repo's documented ~0.6pp/start noise floor.

The leverage result (off-cadence credit gate) was measured the same way -- 3 to 4 starts. It has
one thing the value finding never had, a walk-forward OOS test, but the in-sample deltas deserve
the same scrutiny. If it is also a coin flip across starts, it goes the same way.

Comparison: DEPLOYED (rebalance-only credit gate) vs the DYN8 configuration
(OR(hy_oas,baa_aaa), gate_pct .95, derisk .30, 5-day re-check, band .10, gate off-cadence),
from 12 monthly starts. Sign consistency is the test, not the mean.
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
WIN = {"gate_cols":["hy_oas","baa_aaa"],"gate_pct":0.95,"gate_derisk":0.30,
       "lev_recheck_every":5,"lev_band":0.10,"gate_offcadence":True}

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

for path, lbl, yrs_ in [("data/wrds/sp1500_universe_2000.pkl","26yr",[2001,2002]),
                        ("data/wrds/complete_sp1500_universe.pkl","8yr",[2018,2019])]:
    STARTS=[f"{y}-{m:02d}-03" for y in yrs_ for m in range(1,13,2)]
    bt=LiveMirrorBacktester(universe_path=path); cd(bt)
    print(f"\n=== {lbl}: DEPLOYED vs off-cadence gate, {len(STARTS)} starts ===", flush=True)
    print(f"  {'start':<12}{'dep CAGR':>10}{'win CAGR':>10}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}", flush=True)
    dc,ds,dd=[],[],[]
    for st in STARTS:
        try:
            c0=dict(B); c0.update(DEPLOYED); m0=bt.run(st,"2025-12-31",c0); a0,s0,d0=stat(m0["daily_values"])
            c1=dict(B); c1.update(WIN);      m1=bt.run(st,"2025-12-31",c1); a1,s1,d1=stat(m1["daily_values"])
        except Exception as e:
            print(f"  {st:<12} ERR {type(e).__name__}", flush=True); continue
        dc.append(a1-a0); ds.append(s1-s0); dd.append(d1-d0)
        print(f"  {st:<12}{a0:>+10.2%}{a1:>+10.2%}{(a1-a0)*100:>+8.2f}p{s1-s0:>+9.3f}{(d1-d0)*100:>+8.2f}p", flush=True)
    a,b,c=np.array(dc),np.array(ds),np.array(dd)
    print(f"\n  dCAGR   mean {a.mean()*100:+.2f}p  median {np.median(a)*100:+.2f}p  "
          f"range {a.min()*100:+.2f}p..{a.max()*100:+.2f}p  positive {int((a>0).sum())}/{len(a)}", flush=True)
    print(f"  dSharpe mean {b.mean():+.3f}  positive {int((b>0).sum())}/{len(b)}", flush=True)
    print(f"  dMaxDD  mean {c.mean()*100:+.2f}p  positive {int((c>0).sum())}/{len(c)}", flush=True)
    del bt
print("\nSign consistency across starts is the test. 6/12 = coin flip = dead, like the value result.", flush=True)
