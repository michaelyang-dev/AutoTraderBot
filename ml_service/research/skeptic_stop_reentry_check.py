"""
Skeptic check — stop-out re-entry exclusion window idea.
Measures, in deployed v12 (2018-2025):
  1. How many trailing-stop fires total.
  2. How many stopped names RE-ENTER the book at a later rebalance within
     10/20/40/63 trading days (the events the proposed exclusion would block).
  3. For those re-entries: was the stock still falling (re-entry px vs stop px),
     and what was the fwd 20/60d return from re-entry? If positive, an exclusion
     window HURTS; if negative, it helps.
  4. Baseline: fwd 20/60d return of ALL stopped names from stop price.
Matches deployed v12 (enhanced + SI cleared), same as phase0_stop_attribution.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15, "log_stops": True, "record_targets": True,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

bt = FastBacktester()
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
          "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
    setattr(bt.uni, k, {})
bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

m = bt.run("2018-01-01", "2025-12-31", V12)
events = bt._stop_events
rebals = bt._rebal_log  # list of (date, {sym: weight})
prices = bt.prices
pidx = prices.index

print("=" * 64, flush=True)
print("STOP-OUT RE-ENTRY MEASUREMENT (v12, 40% trailing stop, 2018-25)")
print("=" * 64)
print(f"CAGR {m['cagr']*100:.1f}% | MaxDD {m['max_dd']*100:.1f}%")
print(f"Total stop fires: {len(events)}  (~{len(events)/8:.1f}/yr)")
print(f"Total rebalances logged: {len(rebals)}")

def fwd_ret(sym, d, horizon, base_px):
    try:
        loc = pidx.get_loc(d)
    except Exception:
        return None
    j = loc + horizon
    if j >= len(pidx):
        return None
    p1 = prices.iloc[j].get(sym)
    if p1 and base_px and base_px > 0 and not np.isnan(p1):
        return (p1 - base_px) / base_px
    return None

# Baseline: forward return from stop price (the "freefall" premise)
fwd20_all = [r for e in events
             if (r := fwd_ret(e["sym"], e["date"], 20, e["stop_px"])) is not None]
fwd60_all = [r for e in events
             if (r := fwd_ret(e["sym"], e["date"], 60, e["stop_px"])) is not None]
print(f"\nALL stopped names, fwd from stop px:")
if fwd20_all:
    print(f"  +20d: mean {np.mean(fwd20_all)*100:+.2f}% med {np.median(fwd20_all)*100:+.2f}% (n={len(fwd20_all)})")
if fwd60_all:
    print(f"  +60d: mean {np.mean(fwd60_all)*100:+.2f}% med {np.median(fwd60_all)*100:+.2f}% (n={len(fwd60_all)})")

# Re-entry detection
windows = [10, 20, 40, 63]
reentries = []  # (event, reentry_date, gap_days)
for e in events:
    d, sym = e["date"], e["sym"]
    try:
        loc0 = pidx.get_loc(d)
    except Exception:
        continue
    for rd, tgt in rebals:
        if rd <= d:
            continue
        if sym in tgt and tgt[sym] > 0:
            try:
                gap = pidx.get_loc(rd) - loc0
            except Exception:
                break
            reentries.append((e, rd, gap))
            break  # first re-entry only

print(f"\nRe-entries (stopped name re-picked at a later rebalance): {len(reentries)}")
for w in windows:
    n_w = sum(1 for (_, _, g) in reentries if g <= w)
    print(f"  within {w:>2} trading days: {n_w}  (~{n_w/8:.2f}/yr)")

# For re-entries within 40d (the widest proposed window): what happened?
print(f"\nDetail of re-entries within 63 trading days:")
helped, hurt = [], []
for e, rd, gap in sorted(reentries, key=lambda x: x[2]):
    if gap > 63:
        continue
    sym, sp = e["sym"], e["stop_px"]
    re_px = prices.loc[rd].get(sym)
    if re_px is None or np.isnan(re_px):
        continue
    drop_since_stop = (re_px - sp) / sp
    f20 = fwd_ret(sym, rd, 20, re_px)
    f60 = fwd_ret(sym, rd, 60, re_px)
    print(f"  {e['date'].date()} stop {sym:<6} @{sp:8.2f} -> re-enter {rd.date()} "
          f"(+{gap}td) @{re_px:8.2f} ({drop_since_stop*100:+.1f}% since stop) "
          f"fwd20 {'' if f20 is None else f'{f20*100:+.1f}%'} "
          f"fwd60 {'' if f60 is None else f'{f60*100:+.1f}%'}")
    if f20 is not None:
        (hurt if f20 < 0 else helped).append(f20)

nn = len(helped) + len(hurt)
if nn:
    allf = helped + hurt
    print(f"\nRe-entered names fwd20 from re-entry px: mean {np.mean(allf)*100:+.2f}% "
          f"(n={nn}; {len(helped)} up, {len(hurt)} down)")
    print("If mean > 0, an exclusion window would have FORGONE gains (hurts).")
