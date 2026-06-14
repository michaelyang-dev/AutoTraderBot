"""
Phase 0.3 — Trailing-stop attribution.
Question: is the 40% trailing stop helping or decorative?
- How often does it fire per year?
- After a name is stopped, what's its forward 20/60-day return?
  If stopped names tend to RECOVER (positive forward return), the stop costs money.
  If they keep FALLING, the stop helped.
Matches deployed v12 (enhanced + SI cleared).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15, "log_stops": True,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

bt = FastBacktester()
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
          "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
    setattr(bt.uni, k, {})
bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

m = bt.run("2018-01-01", "2025-12-31", V12)
events = bt._stop_events
prices = bt.prices
pidx = prices.index

print("=" * 60, flush=True)
print("TRAILING-STOP ATTRIBUTION (v12, 40% stop, 2018-2025)")
print("=" * 60)
n = len(events)
years = 8.0
print(f"\nTotal stop fires: {n}  (~{n/years:.1f} per year)")
print(f"Backtest CAGR with stop: {m['cagr']*100:.1f}% | MaxDD {m['max_dd']*100:.1f}%")

fwd20, fwd60 = [], []
for e in events:
    d = e["date"]; sym = e["sym"]; sp = e["stop_px"]
    try:
        loc = pidx.get_loc(d)
    except Exception:
        continue
    for horizon, bucket in [(20, fwd20), (60, fwd60)]:
        j = loc + horizon
        if j < len(pidx):
            p1 = prices.iloc[j].get(sym)
            if p1 and sp and sp > 0 and not np.isnan(p1):
                bucket.append((p1 - sp) / sp)

print(f"\nForward return of STOPPED names (from the stop price):")
if fwd20:
    print(f"  +20 days: mean {np.mean(fwd20)*100:+.2f}%  median {np.median(fwd20)*100:+.2f}%  (n={len(fwd20)})")
if fwd60:
    print(f"  +60 days: mean {np.mean(fwd60)*100:+.2f}%  median {np.median(fwd60)*100:+.2f}%  (n={len(fwd60)})")

print("\nVERDICT:")
if n / years < 3:
    print(f"  Stop fires rarely (~{n/years:.1f}/yr) — mostly decorative, only acts in sharp drops.")
if fwd20 and np.mean(fwd20) > 0.03:
    print(f"  Stopped names rebound +{np.mean(fwd20)*100:.1f}% in 20d — the stop is COSTING money (selling the bottom).")
elif fwd20 and np.mean(fwd20) < -0.03:
    print(f"  Stopped names keep falling {np.mean(fwd20)*100:.1f}% in 20d — the stop is HELPING.")
elif fwd20:
    print(f"  Stopped names roughly flat ({np.mean(fwd20)*100:+.1f}% in 20d) — stop is ~neutral.")

# Compare: run WITHOUT the stop to see the actual return impact
print("\n=== Same config, NO trailing stop ===", flush=True)
V12_nostop = dict(V12); V12_nostop["trailing_stop"] = None; V12_nostop["log_stops"] = False
m2 = bt.run("2018-01-01", "2025-12-31", V12_nostop)
print(f"  WITH 40% stop: CAGR {m['cagr']*100:.1f}% | MaxDD {m['max_dd']*100:.1f}%")
print(f"  NO stop:       CAGR {m2['cagr']*100:.1f}% | MaxDD {m2['max_dd']*100:.1f}%")
print(f"  Stop impact:   CAGR {(m['cagr']-m2['cagr'])*100:+.1f}pp | MaxDD {(m['max_dd']-m2['max_dd'])*100:+.1f}pp")
