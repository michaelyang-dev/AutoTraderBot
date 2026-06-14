"""
Is vol-scaling genuinely better, or just lower average leverage?
Compare vol-scaling@1.49x against simply running a LOWER FLAT leverage.
Efficient: run base + vol-scaling once each (per start day), cache daily values,
apply leverage in post-processing. Start-day averaged.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np

STARTS = ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-08"]
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
    peak = vals.cummax(); maxdd = ((vals - peak) / peak).min()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (252 / len(vals)) - 1
    return cagr, maxdd


bt = FastBacktester(); clear(bt)
base = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
        "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

# run each config ONCE per start day, cache daily values
print("running base + vol-scaling backtests...", flush=True)
base_dv = [bt.run(st, "2025-12-31", base)["daily_values"] for st in STARTS]
vs20_dv = [bt.run(st, "2025-12-31", {**base, "vol_scaling": True, "vol_target": 0.20})["daily_values"] for st in STARTS]
vs15_dv = [bt.run(st, "2025-12-31", {**base, "vol_scaling": True, "vol_target": 0.15})["daily_values"] for st in STARTS]


def avg(dvs, L):
    cs, dds = zip(*[lever(v, L) for v in dvs])
    return np.mean(cs) * 100, np.mean(dds) * 100


print("\n" + "=" * 56)
print("FLAT DE-LEVERAGING FRONTIER vs VOL-SCALING@1.49x")
print("=" * 56)
print("\nCURRENT config at flat leverage levels:")
print(f"{'lev':>8}{'CAGR':>9}{'MaxDD':>9}")
flat = {}
for L in [1.49, 1.40, 1.30, 1.20, 1.10, 1.00]:
    c, dd = avg(base_dv, L); flat[L] = (c, dd)
    print(f"{L:>7.2f}x{c:>8.1f}%{dd:>8.1f}%")

print("\nVOL-SCALING @ 1.49x base — does it beat the flat line at equal CAGR?")
for name, dv in [("vs20%", vs20_dv), ("vs15%", vs15_dv)]:
    c, dd = avg(dv, 1.49)
    match = min(flat.items(), key=lambda kv: abs(kv[1][0] - c))
    verdict = f"BETTER by {match[1][1]-dd:+.1f}pp" if dd > match[1][1] + 0.3 else (
        f"WORSE by {match[1][1]-dd:+.1f}pp" if dd < match[1][1] - 0.3 else "SAME (~flat line)")
    print(f"  {name}: CAGR {c:.1f}%  MaxDD {dd:.1f}%  | nearest flat {match[0]:.2f}x "
          f"(CAGR {match[1][0]:.1f}%, MaxDD {match[1][1]:.1f}%) -> {verdict}", flush=True)
