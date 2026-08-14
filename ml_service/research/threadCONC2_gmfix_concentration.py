"""thread CONC2 — the same concentration test applied to MY OWN gross_margin fix.

Having killed the leverage finding on concentration, the gm bound had to face the identical
test rather than be graded on a friendlier curve.

RESULT (26yr): top 5 days = 8.4% of total (vs the leverage finding's 99.6%), 15/25 positive
years, benefit spread across 2003/2008/2023/2025. That is what a distributed effect looks like.
8yr is weaker and should not be oversold: top 5 days = 55.9%, 5/8 positive years, and this
single start nets -4.48pp -- consistent with threadGM4's 8/12 starts positive.

=> the gm bound survives on the 26yr and is mixed on the 8yr. Its primary justification remains
CORRECTNESS: gm = 4561.72 on a ratio bounded by 1 had collapsed the live book to 15 names.
"""
import os, sys
os.chdir("/Users/michaelslyanggmail.com/Downloads/auto-trader 2/ml_service")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(),'research'))
import numpy as np, pandas as pd
import strategies.multi_strategy_engine as M
from livemirror_backtest import LiveMirrorBacktester
B = {"universe":"sp1500","mom_w":0.50,"val_w":0.35,"lv_w":0.15,"sec_w":0.0,"top_n":5,
     "rebal_days":20,"trailing_stop":0.40,"cap":0.10,"use_rp":False,"vol_scaling":True,
     "vol_target":0.15,"vol_lookback":40,"vol_scale_cap":1.0,"initial_capital":50_000.0,
     "leverage":1.49,"integer_shares":True,"financing_rate":0.063}
SHIP=M._sane_gross_margin
def cd(bt):
    for k in ["_fin_growth","_ev","_estimates","_price_targets","_revenue_surprise","_beat_streak","_earnings_signals"]:
        setattr(bt.uni,k,{})
    bt._si_ranks_by_month={}; bt._si_change_ranks_by_month={}; bt._si_months=[]
    bt.uni._short_interest_rank={}; bt.uni._si_change_rank={}
for path,lbl,start in [("data/wrds/sp1500_universe_2000.pkl","26yr","2001-01-03"),
                       ("data/wrds/complete_sp1500_universe.pkl","8yr","2018-01-03")]:
    bt=LiveMirrorBacktester(universe_path=path); cd(bt)
    M._sane_gross_margin=lambda g: g          # OFF
    r0=bt.run(start,"2025-12-31",dict(B))["daily_values"].pct_change().dropna()
    M._sane_gross_margin=SHIP                 # ON
    r1=bt.run(start,"2025-12-31",dict(B))["daily_values"].pct_change().dropna()
    idx=r0.index.intersection(r1.index); d=(r1.reindex(idx)-r0.reindex(idx)).dropna()
    tot=d.sum(); srt=d.reindex(d.abs().sort_values(ascending=False).index)
    print(f"\n=== {lbl}: gm bound ON minus OFF — CONCENTRATION ===")
    print(f"  days {len(d)}   total excess {tot*100:+.2f}pp")
    for k in (5,10,20,50):
        print(f"  top {k:>3} |diff| days = {srt.head(k).sum()/tot*100:>7.1f}% of total")
    print(f"  days ON beat OFF: {int((d>0).sum())}/{len(d)} ({(d>0).mean():.1%})")
    yr=d.groupby(d.index.year).sum()
    print(f"  positive years: {int((yr>0).sum())}/{len(yr)}")
    print(f"  by year (pp): " + "  ".join(f"{y}:{v*100:+.1f}" for y,v in yr.items()))
    del bt
M._sane_gross_margin=SHIP
