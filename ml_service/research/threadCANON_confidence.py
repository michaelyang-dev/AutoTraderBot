"""thread CANON — how uncertain ARE the canonical numbers?

The headline figures (8yr +23.58%/0.84/-38.23%, 26yr +11.29%/0.51/-64.75%) were computed on 3
and 2 starts. threadCORE5 then showed the per-start sigma of a DELTA is ~2.30pp -- roughly 4x
the repo's documented ~0.6pp/start noise floor. If a difference between two configs is that
noisy, the LEVEL of a single config is at least as noisy, and quoting it to two decimals implies
a precision that does not exist.

This measures the actual start-date distribution of the DEPLOYED configuration: 12 monthly
starts per horizon, reporting mean/median/range/sigma rather than a point estimate.

The output is not a new strategy. It is an honest error bar on every number this project quotes.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'research'))
import numpy as np
from livemirror_backtest import LiveMirrorBacktester

B = {"universe":"sp1500","mom_w":0.50,"val_w":0.35,"lv_w":0.15,"sec_w":0.0,"top_n":5,
     "rebal_days":20,"trailing_stop":0.40,"cap":0.10,"use_rp":False,"vol_scaling":True,
     "vol_target":0.15,"vol_lookback":40,"vol_scale_cap":1.0,"initial_capital":50_000.0,
     "leverage":1.49,"integer_shares":True,"financing_rate":0.063}

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

for path,lbl,yrs_,quoted in [
        ("data/wrds/complete_sp1500_universe.pkl","8yr  2018-25",[2018,2019],"+23.58% / 0.84 / -38.23%"),
        ("data/wrds/sp1500_universe_2000.pkl","26yr 2001-25",[2001,2002],"+11.29% / 0.51 / -64.75%")]:
    STARTS=[f"{y}-{m:02d}-03" for y in yrs_ for m in range(1,13,2)]
    bt=LiveMirrorBacktester(universe_path=path); cd(bt)
    cs,ss,ds=[],[],[]
    print(f"\n=== {lbl} | DEPLOYED config, {len(STARTS)} starts ===", flush=True)
    print(f"  quoted in docs: {quoted}", flush=True)
    print(f"  {'start':<12}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}", flush=True)
    for st in STARTS:
        try:
            m=bt.run(st,"2025-12-31",dict(B)); a,b,c=stat(m["daily_values"])
        except Exception as e:
            print(f"  {st:<12} ERR {type(e).__name__}", flush=True); continue
        cs.append(a); ss.append(b); ds.append(c)
        print(f"  {st:<12}{a:>+9.2%}{b:>8.2f}{c:>+9.2%}", flush=True)
    a,b,c=np.array(cs),np.array(ss),np.array(ds)
    print(f"\n  CAGR   mean {a.mean():+.2%}  median {np.median(a):+.2%}  sigma {a.std(ddof=1)*100:.2f}pp"
          f"  range {a.min():+.2%}..{a.max():+.2%}", flush=True)
    print(f"  Sharpe mean {b.mean():.3f}  median {np.median(b):.3f}  sigma {b.std(ddof=1):.3f}"
          f"  range {b.min():.2f}..{b.max():.2f}", flush=True)
    print(f"  MaxDD  mean {c.mean():+.2%}  median {np.median(c):+.2%}  sigma {c.std(ddof=1)*100:.2f}pp"
          f"  range {c.min():+.2%}..{c.max():+.2%}", flush=True)
    print(f"  95% CI on the MEAN (sigma/sqrt(n)*1.96): CAGR +/-{1.96*a.std(ddof=1)/np.sqrt(len(a))*100:.2f}pp"
          f"  Sharpe +/-{1.96*b.std(ddof=1)/np.sqrt(len(b)):.3f}", flush=True)
    del bt
print("\nQuoting these to 2dp off 2-3 starts implies precision that does not exist.", flush=True)
