"""
Phase 1.3 — Whipsaw audit of the regime overlay.
The regime switch (20-day UMD crash + breadth blend) is backward-looking.
Does it actually help, or did it get lucky once? Compare:
  A. v12 WITH regime overlay (bull/bear blend + UMD crash weights)
  B. v12 with NO regime overlay (always bull weights 50/35/15)
across the full period AND in the specific stress windows.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

LEV = 1.49


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(vals, L=1.0):
    r = vals.pct_change().fillna(0.0)
    if L != 1.0:
        r = L * r - (L - 1) * 0.055 / 252
        vals = (1 + r).cumprod()
    peak = vals.cummax(); maxdd = ((vals - peak) / peak).min()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (252 / len(vals)) - 1
    return cagr, maxdd


bt = FastBacktester()
clear(bt)
base = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
        "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

# WITH overlay = the normal v12 (regime blend active).
# NO overlay = force bull weights always: set bear_weights == bull weights and
#   disable crash by making crash == bull. We approximate "always bull" by setting
#   bear_weights equal to the bull 50/35/15/0 so the breadth blend is a no-op,
#   and note the UMD-crash override still fires (can't disable via config) -- so this
#   isolates the BREADTH blend; UMD crash is rare (a few days).
no_overlay = {**base, "bear_weights": {"mom": 0.50, "val": 0.35, "s5": 0.15, "s3": 0.00}}

print("=" * 64, flush=True)
print("REGIME OVERLAY WHIPSAW AUDIT (v12, n=5, 2018-2025)")
print("=" * 64)
for label, cfg in [("WITH regime overlay (live v12)", base),
                   ("NO breadth overlay (always bull)", no_overlay)]:
    m = bt.run("2018-01-01", "2025-12-31", cfg)
    vals = m["daily_values"]
    c1, dd1 = stats(vals, 1.0)
    cL, ddL = stats(vals, LEV)
    print(f"\n  {label}")
    print(f"     1x:    CAGR {c1*100:5.1f}%  MaxDD {dd1*100:6.1f}%")
    print(f"     1.49x: CAGR {cL*100:5.1f}%  MaxDD {ddL*100:6.1f}%")
    # stress-window returns
    for wlabel, s, e in [("COVID 2020 Q1", "2020-02-01", "2020-04-30"),
                         ("2022 mom reversal", "2022-01-01", "2022-12-31")]:
        w = vals[(vals.index >= s) & (vals.index <= e)]
        if len(w) > 2:
            wr = w.iloc[-1] / w.iloc[0] - 1
            wpeak = w.cummax(); wdd = ((w - wpeak) / wpeak).min()
            print(f"       {wlabel}: return {wr*100:+.1f}%  intra-DD {wdd*100:.1f}%")
print("\nVERDICT: if WITH-overlay has materially better stress-window drawdowns,")
print("the regime switch earns its keep. If similar/worse, it's whipsaw / luck.")
