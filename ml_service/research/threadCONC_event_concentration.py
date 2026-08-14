"""thread CONC — the test that KILLED the off-cadence leverage finding.

Start-date consistency (23/24, threadDYN10) and walk-forward OOS (threadDYN7) both looked
decisive. They share a blind spot: every start window inside a horizon contains the SAME crisis.
24 starts on the 26yr are not 24 independent samples of "does this help in a crisis" -- they are
24 views of ONE 2008. Sign-consistency across starts measures robustness to ENTRY TIMING, not to
REGIME, and I treated it as if it measured both.

This measures where the outperformance actually comes from, day by day.

RESULT (26yr): top 5 days = 99.6% of total excess; excluding 2008 the remaining 25 years sum to
-0.8pp; the winner beats deployed on only 48.1% of days; 2001 (dot-com) was -10.2pp while 2008
was +10.9pp. 8yr: top 5 days = 304% of total, 2020 +7.3pp but 2022 -8.4pp, 4/8 positive years.

=> helped in 2 crises (2008, 2020), HURT in 2 others (2001, 2022). Not reliable insurance --
a bet that future crises rhyme with those two. NOT IMPLEMENTED.
"""
import os, sys
os.chdir("/Users/michaelslyanggmail.com/Downloads/auto-trader 2/ml_service")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(),'research'))
import numpy as np, pandas as pd
from livemirror_backtest import LiveMirrorBacktester
B = {"universe":"sp1500","mom_w":0.50,"val_w":0.35,"lv_w":0.15,"sec_w":0.0,"top_n":5,
     "rebal_days":20,"trailing_stop":0.40,"cap":0.10,"use_rp":False,"vol_scaling":True,
     "vol_target":0.15,"vol_lookback":40,"vol_scale_cap":1.0,"initial_capital":50_000.0,
     "leverage":1.49,"integer_shares":True,"financing_rate":0.063,"live_sizing":True}
DEP={"credit_pct":0.95,"credit_derisk":0.5}
WIN={"gate_cols":["hy_oas"],"gate_pct":0.95,"gate_derisk":0.30,
     "lev_recheck_every":5,"lev_band":0.10,"gate_offcadence":True}
def cd(bt):
    for k in ["_fin_growth","_ev","_estimates","_price_targets","_revenue_surprise","_beat_streak","_earnings_signals"]:
        setattr(bt.uni,k,{})
    bt._si_ranks_by_month={}; bt._si_change_ranks_by_month={}; bt._si_months=[]
    bt.uni._short_interest_rank={}; bt.uni._si_change_rank={}
bt=LiveMirrorBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl"); cd(bt)
c0=dict(B); c0.update(DEP); c1=dict(B); c1.update(WIN)
r0=bt.run("2001-01-03","2025-12-31",c0)["daily_values"].pct_change().dropna()
r1=bt.run("2001-01-03","2025-12-31",c1)["daily_values"].pct_change().dropna()
idx=r0.index.intersection(r1.index); d=(r1.reindex(idx)-r0.reindex(idx)).dropna()
tot=d.sum()
srt=d.reindex(d.abs().sort_values(ascending=False).index)
print("=== CONCENTRATION of outperformance (26yr, single start) ===")
print(f"  trading days: {len(d)}   total excess return (sum of daily diffs): {tot*100:+.2f}pp")
for k in (5,10,20,50):
    print(f"  top {k:>3} |diff| days contribute {srt.head(k).sum()/tot*100:>7.1f}% of the total")
print(f"\n  days winner BEAT deployed: {int((d>0).sum())}/{len(d)} ({(d>0).mean():.1%})")
print(f"  largest single-day diff  : {d.abs().max()*100:.2f}pp on {d.abs().idxmax().date()}")
yr=d.groupby(d.index.year).sum()
print(f"\n  by year (pp): " + "  ".join(f"{y}:{v*100:+.1f}" for y,v in yr.items()))
print(f"  positive years: {int((yr>0).sum())}/{len(yr)}")
