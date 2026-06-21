"""
Reframe — event-driven EXIT layer (a DIFFERENT job than stock selection).
========================================================================
Question that decides whether a news/ML exit layer can ever help: when a HELD
top-momentum name gaps down hard on an idiosyncratic (market-relative) move
(proxy for a bad-news event), does it CONTINUE down (structural break -> fast exit
wins) or BOUNCE (overreaction -> exit loses)?

If hard gaps continue down -> a news/ML layer that flags them adds value (its job
= separate breakers from bouncers, which the 40% price stop can't do).
If hard gaps bounce -> even this use is limited (consistent w/ the aging finding
that dips mean-revert).

Method: at each 20d rebalance take the top-5 momentum book. Over the next 20d,
flag the FIRST day a name's 1d return minus SPY < -thresh (idiosyncratic gap).
Measure that name's MARKET-RELATIVE forward return 5/10/20d AFTER the gap, vs the
no-gap baseline forward return of held names. Splits by gap size.

Run: cd ml_service && ./venv/bin/python research/alpha_event_exit_diag.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester
from strategies.multi_strategy_engine import strategy1_momentum_reversal

THRESH = [0.07, 0.10, 0.15]   # idiosyncratic 1d drop sizes (name - SPY)
FWD = [5, 10, 20]


def fwd_excess(dr, spy_r, sym, gi, h, allidx):
    """market-relative forward return of sym over h days starting AFTER day gi."""
    j = gi + h
    if j >= len(allidx):
        return None
    s = dr[sym].iloc[gi+1:j+1].sum() if sym in dr.columns else np.nan
    m = spy_r.iloc[gi+1:j+1].sum()
    if np.isnan(s):
        return None
    return s - m


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni.get_sp500 = bt._get_sp1500
    print(f"[loaded {time.time()-t0:.0f}s]")

    prices = bt.prices
    dr = prices.pct_change()
    spy_r = dr["SPY"]
    allidx = list(prices.index)
    di_map = {d: i for i, d in enumerate(allidx)}
    dates = [d for d in allidx if pd.Timestamp("2016-01-01") <= d <= pd.Timestamp("2025-12-31")]
    rebal = list(range(0, len(dates), 20))

    # collect baseline (held-name daily) and event-conditioned forward excess
    base = {h: [] for h in FWD}
    ev = {th: {h: [] for h in FWD} for th in THRESH}
    n_events = {th: 0 for th in THRESH}
    n_holdmonths = 0

    for k in rebal[:-1]:
        d = dates[k]; gi = di_map[d]
        w = strategy1_momentum_reversal(d, bt.uni, k, top_n=5, rebal_days=20)
        if not w:
            continue
        window = allidx[gi:di_map[dates[rebal[rebal.index(k)+1]]]] if False else allidx[gi:gi+20]
        for sym in w:
            n_holdmonths += 1
            # baseline forward excess from rebalance day
            for h in FWD:
                fe = fwd_excess(dr, spy_r, sym, gi, h, allidx)
                if fe is not None:
                    base[h].append(fe)
            # find first idiosyncratic gap in the next 20d, per threshold
            if sym not in dr.columns:
                continue
            seg = dr[sym].iloc[gi+1:gi+21] - spy_r.iloc[gi+1:gi+21]
            for th in THRESH:
                hit = seg[seg < -th]
                if len(hit) == 0:
                    continue
                gday = hit.index[0]; ggi = di_map[gday]
                n_events[th] += 1
                for h in FWD:
                    fe = fwd_excess(dr, spy_r, sym, ggi, h, allidx)
                    if fe is not None:
                        ev[th][h].append(fe)

    print(f"\n[top-5 holdings tracked over {len(rebal)-1} rebalances, {n_holdmonths} name-holds, 2016-2025]\n")
    print("=== BASELINE: market-relative forward return of held names (from rebalance) ===")
    for h in FWD:
        a = np.array(base[h])
        print(f"  +{h:2d}d: mean {a.mean()*100:+5.2f}%  median {np.median(a)*100:+5.2f}%  hit {(a>0).mean()*100:4.0f}%  n={len(a)}")

    print("\n=== AFTER an idiosyncratic gap (name - SPY < -thresh): forward MKT-REL return ===")
    print("    (negative => CONTINUES down => exit wins; positive => BOUNCES => exit loses)")
    for th in THRESH:
        print(f"\n  gap < -{int(th*100)}%  ({n_events[th]} events = {n_events[th]/max(n_holdmonths,1)*100:.1f}% of holds)")
        for h in FWD:
            a = np.array(ev[th][h])
            if len(a) == 0:
                continue
            print(f"     +{h:2d}d: mean {a.mean()*100:+6.2f}%  median {np.median(a)*100:+6.2f}%  "
                  f"hit {(a>0).mean()*100:4.0f}%  n={len(a)}")
    print(f"\n[total {time.time()-t0:.0f}s]")
