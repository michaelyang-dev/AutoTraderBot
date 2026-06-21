"""
Thread #3 — market-level RISK/sentiment as a de-risk confirmer.
Note: true market NEWS-sentiment needs GDELT history; what's testable now is
market-RISK signals (VIX, VIX term structure, HY credit spread, yield curve) used
as the de-risk trigger. Compared at MATCHED avg exposure vs incumbents
(SPY<SMA200, vol-scaling) on the deployed book.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester
from research.exposure_lib import (metrics, exposure_from_signal, spy_trend_exposure,
                                    vol_scale_exposure, apply_overlay)

BASE = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
            top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
            bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10})
# NOTE: NO trend_scale, NO vol_scaling -> base book returns (overlays applied here)


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def show(name, m, avg_e):
    print(f"  {name:<34} CAGR {m['cagr']*100:5.1f}%  Sharpe {m['sharpe']:.2f}  "
          f"Sortino {m['sortino']:.2f}  MaxDD {m['mdd']*100:6.1f}%  avgExpo {avg_e*100:3.0f}%")


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester(); clear(bt)
    print(f"[loaded {time.time()-t0:.0f}s]")

    r = bt.run("2016-01-01", "2025-12-31", BASE)
    nav = r["daily_values"].dropna()
    base_r = nav.pct_change().fillna(0).values
    idx = nav.index

    # ---- signals ----
    vix = pd.read_parquet("data/enhanced_data/vix_cache.parquet")
    vix.index = pd.to_datetime(vix.index)
    fred = pd.read_parquet("data/macro_fred.parquet")
    fred.index = pd.to_datetime(fred.index)

    vix_lvl = vix["^VIX"]
    vix_term = vix["^VIX"] / vix["^VIX3M"]           # >1 = backwardation = stress
    hy_oas = fred["BAMLH0A0HYM2"]                     # high-yield credit spread
    yc = fred["T10Y2Y"]                              # yield curve (low/neg = stress)

    # ---- incumbents (set the matched-exposure target) ----
    e_spy = spy_trend_exposure(bt.prices, idx)
    r_spy, a_spy = apply_overlay(base_r, idx, e_spy)
    target = a_spy   # match everything to SPY<SMA200's average exposure
    e_vol = vol_scale_exposure(base_r, idx)
    r_vol, a_vol = apply_overlay(base_r, idx, e_vol)

    print("\n=== exposure de-risk signals @ matched avg exposure (2016-2025) ===")
    show("NONE (always 100%)", metrics(base_r, idx), 1.0)
    show("SPY<SMA200 (incumbent trend)", metrics(r_spy, idx), a_spy)
    show("vol-scaling (incumbent vol)", metrics(r_vol, idx), a_vol)
    print(f"  [matching continuous signals to avg exposure = {target*100:.0f}%]")

    for nm, sig, hir in [("VIX level", vix_lvl, True),
                         ("VIX term (VIX/VIX3M)", vix_term, True),
                         ("HY credit spread", hy_oas, True),
                         ("yield curve (T10Y2Y)", yc, False)]:
        e = exposure_from_signal(sig, idx, floor=0.40, target_avg=target, high_is_risk=hir)
        rr, ae = apply_overlay(base_r, idx, e)
        show(nm, metrics(rr, idx), ae)

    # combos: de-risk if SPY-bear OR signal fires (union), matched after
    print("\n=== SPY<SMA200 + VIX-term (de-risk if EITHER) ===")
    e_combo = pd.concat([e_spy, exposure_from_signal(vix_term, idx, 0.40, target, True)], axis=1).min(axis=1)
    rc, ac = apply_overlay(base_r, idx, e_combo)
    show("SPY OR VIXterm", metrics(rc, idx), ac)
    print(f"\n[total {time.time()-t0:.0f}s]")
