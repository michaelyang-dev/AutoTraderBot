"""
Turnover measurement + hysteresis-addressable churn (decides if hysteresis is
worth building). Measures, for the deployed config:
  - annual turnover (gross $ traded / yr / avg NAV) and implied cost drag
  - per-rebalance NAME churn of the combined book and weight turnover
  - BOUNDARY oscillation: names that exit then RE-ENTER within K rebalances
    (this is the churn hysteresis can recover; structural exits can't)

Run: cd ml_service && ./venv/bin/python research/alpha_turnover.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import COST_BPS

DEPLOYED = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
                top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
                bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
                trend_scale={"bear": 0.40, "caution": 0.75},
                vol_scaling=True, vol_target=0.20, vol_lookback=40,
                record_targets=True)


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals",
              "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    print(f"[loaded {time.time()-t0:.0f}s]")

    r = bt.run("2016-01-01", "2025-12-31", DEPLOYED)
    nav = r["daily_values"]
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    avg_nav = nav.mean()
    gt = bt._gross_traded
    cost_bps = COST_BPS + SLIPPAGE_BPS
    ann_turn = gt / yrs / avg_nav
    drag = ann_turn * cost_bps / 10000
    print(f"\n=== TURNOVER (deployed, 2016-2025) ===")
    print(f"  CAGR {r['cagr']*100:.1f}%  cost model {cost_bps}bps/trade")
    print(f"  gross traded ${gt/1e6:.1f}M  avg NAV ${avg_nav/1e3:.0f}k  over {yrs:.1f}y")
    print(f"  ANNUAL TURNOVER: {ann_turn*100:.0f}%   implied cost drag: {drag*100:.2f}%/yr")

    # name churn between consecutive rebalances
    log = bt._rebal_log  # list of (date, {sym: w})
    churns, wturn, reentry = [], [], 0
    history = []  # list of sets of held names
    for i in range(1, len(log)):
        prev = set(log[i-1][1]); cur = set(log[i][1])
        if not prev:
            continue
        changed = len(prev.symmetric_difference(cur)) / 2
        churns.append(changed / max(len(prev), 1))
        # weight turnover (sum |dw|/2)
        allk = prev | cur
        wt = sum(abs(log[i][1].get(s, 0) - log[i-1][1].get(s, 0)) for s in allk) / 2
        wturn.append(wt)
    # boundary oscillation: exited at i then re-entered within K
    K = 3
    held_seq = [set(d[1]) for d in log]
    osc = 0; exits = 0
    for i in range(1, len(held_seq)):
        gone = held_seq[i-1] - held_seq[i]
        exits += len(gone)
        for s in gone:
            if any(s in held_seq[j] for j in range(i+1, min(i+1+K, len(held_seq)))):
                osc += 1
    print(f"\n=== NAME CHURN (combined book, {len(log)} rebalances) ===")
    print(f"  avg book size: {np.mean([len(d[1]) for d in log]):.0f} names")
    print(f"  avg name churn / rebalance: {np.mean(churns)*100:.0f}%  "
          f"(weight turnover {np.mean(wturn)*100:.0f}%)")
    print(f"  exits that RE-ENTER within {K} rebalances: {osc}/{exits} = {osc/max(exits,1)*100:.0f}%")
    print(f"  -> hysteresis can address ~the re-entry fraction; structural exits cannot")
    print(f"\n[total {time.time()-t0:.0f}s]")
