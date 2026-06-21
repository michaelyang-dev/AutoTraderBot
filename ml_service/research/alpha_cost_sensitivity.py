"""
Cost sensitivity -> honest denominator for the turnover/hysteresis prize.
CAGR of the deployed config at several round-trip cost levels. The CAGR gap per
cost level = total cost drag; hysteresis can recover at most ~21% of it (the
measured re-entry fraction). Patches the cost constants (they're module globals,
not config).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import main_production_backtest as mpb
from strategies import multi_strategy_engine as mse
from main_production_backtest import FastBacktester

DEPLOYED = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
                top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
                bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
                trend_scale={"bear": 0.40, "caution": 0.75},
                vol_scaling=True, vol_target=0.20, vol_lookback=40)

t0 = time.time()
bt = FastBacktester()
for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
          "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
    setattr(bt.uni, k, {})
bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
print(f"[loaded {time.time()-t0:.0f}s]\n")

base = None
print("  round-trip cost   CAGR     drag vs 0bps")
for total_bps in (0, 10, 20, 30):
    # run() reads main_production_backtest's own COST_BPS + SLIPPAGE_BPS module globals
    mpb.COST_BPS = total_bps
    mpb.SLIPPAGE_BPS = 0
    mse.COST_BPS = total_bps
    r = bt.run("2016-01-01", "2025-12-31", DEPLOYED)
    if base is None:
        base = r["cagr"]
    print(f"  {total_bps:3d} bps/trade    {r['cagr']*100:5.1f}%   {(base-r['cagr'])*100:+5.2f}pp")
print("\nHysteresis recovers ~21% of the drag (the re-entry fraction); the rest is structural.")
