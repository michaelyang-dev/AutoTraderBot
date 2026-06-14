"""
Pre-deployment stress on vol-scaling, addressing 3 objections:
 #1 Don't anchor on 5pp: which DRAWDOWN EVENTS drive the improvement? (n? )
 #2 Principled target + robustness: natural vol; sweep target x lookback (plateau?)
 #3 Turnover/tax: extra turnover the overlay adds, and net benefit after cost.
Single start day (2018-01-02) for speed; this is about direction/robustness.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

BORROW = 0.055


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def lever(vals, L):
    r = vals.pct_change().fillna(0.0)
    if L != 1.0:
        r = L * r - (L - 1) * BORROW / 252
        vals = (1 + r).cumprod()
    return vals


def maxdd(vals):
    peak = vals.cummax(); return ((vals - peak) / peak).min()


def cagr(vals):
    return (vals.iloc[-1] / vals.iloc[0]) ** (252 / len(vals)) - 1


def top_drawdowns(vals, n=3):
    """Find the n worst non-overlapping drawdown troughs (year, depth)."""
    peak = vals.cummax(); dd = (vals - peak) / peak
    out = []
    ddc = dd.copy()
    for _ in range(n):
        t = ddc.idxmin(); d = ddc.min()
        if d > -0.05:
            break
        out.append((pd.Timestamp(t), d))
        # zero out a +/-60 trading day window around the trough
        loc = vals.index.get_loc(t)
        lo = max(0, loc - 60); hi = min(len(vals), loc + 60)
        ddc.iloc[lo:hi] = 0
    return out


bt = FastBacktester(); clear(bt)
base = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
        "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
ST = "2018-01-02"

# ---- baseline ----
mb = bt.run(ST, "2025-12-31", base)
vb = mb["daily_values"]; base_traded = bt._gross_traded
avg_pv = vb.mean()
years = len(vb) / 252
nat_vol = vb.pct_change().std() * np.sqrt(252)
base_turn = base_traded / avg_pv / years

print("=" * 66, flush=True)
print("VOL-SCALING PRE-DEPLOYMENT STRESS")
print("=" * 66)
print(f"\nStrategy natural realized vol (1x): {nat_vol*100:.1f}%")
print(f"  -> a 'leverage-neutral' target sits near {nat_vol*100:.0f}%; targets below it")
print(f"     also reduce AVERAGE exposure (not pure re-timing).")
print(f"Baseline annual turnover: {base_turn*100:.0f}%")

# ---- #1 DD event attribution (flat 1.49x vs vs15 1.49x) ----
mv15 = bt.run(ST, "2025-12-31", {**base, "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40})
vv15 = mv15["daily_values"]; vs15_traded = bt._gross_traded
vb_L = lever(vb, 1.49); vv_L = lever(vv15, 1.49)
print(f"\n#1 WORST DRAWDOWN EVENTS at 1.49x (where does the gap come from?):")
print(f"   {'event(trough)':<16}{'flat 1.49x':>12}{'vol-scaled':>12}")
for t, d in top_drawdowns(vb_L, 4):
    # vol-scaled dd around same window
    loc = vv_L.index.get_loc(t); lo = max(0, loc - 60); hi = min(len(vv_L), loc + 60)
    w = vv_L.iloc[lo:hi]; wdd = ((w - w.cummax()) / w.cummax()).min()
    print(f"   {str(t.date()):<16}{d*100:>11.1f}%{wdd*100:>11.1f}%")

# ---- #2 robustness: target x lookback ----
print(f"\n#2 ROBUSTNESS GRID (MaxDD@1.49x / CAGR@1.49x) — flat plateau = robust:")
print(f"   {'tgt / lookbk':<16}{'20d':>14}{'40d':>14}{'60d':>14}")
for tgt in [0.15, 0.18, 0.21]:
    row = f"   {int(tgt*100)}%{'':<13}"
    for lb in [20, 40, 60]:
        m = bt.run(ST, "2025-12-31", {**base, "vol_scaling": True, "vol_target": tgt, "vol_lookback": lb})
        vL = lever(m["daily_values"], 1.49)
        row += f"{maxdd(vL)*100:>6.0f}%/{cagr(vL)*100:>5.0f}% "
    print(row, flush=True)

# ---- #3 turnover + cost/tax of the overlay ----
vs15_turn = vs15_traded / vv15.mean() / years
extra_turn = vs15_turn - base_turn
print(f"\n#3 TURNOVER & COST of the overlay (vs15, 15% target):")
print(f"   baseline turnover {base_turn*100:.0f}%/yr -> vol-scaled {vs15_turn*100:.0f}%/yr "
      f"(extra {extra_turn*100:+.0f}pp)")
print(f"   Backtest already charges 10bps on ALL trades (incl. resizing), so the")
print(f"   CAGR figures are NET of trading cost. Extra cost at a higher 20bps:")
print(f"   extra {extra_turn*100:.0f}% turnover x 10bps add'l = {extra_turn*0.0010*100:.2f}pp/yr more drag")
print(f"   Tax: base strategy is ALREADY ~100% short-term (20d holds), so the overlay")
print(f"   adds little NET new tax — it re-times exposure, not net realization.")
