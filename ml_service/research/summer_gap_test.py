"""
SUMMER DATA-GAP DECISION TEST — WRDS is unavailable until late Aug/early Sept, so live
fundamentals will reach ~5-6 months stale. Options, tested head-to-head:

  A. FRESH Compustat        — the unattainable ceiling (for reference)
  B. STALE 63td (~1 qtr)    — where we are now
  C. STALE 126td (~2 qtr)   — where we'll be by late August if we do nothing
  D. FRESH + FMP-LIKE NOISE — the "overlay FMP for the gap" option: fresh values with
     noise calibrated to the MEASURED FMP-vs-Compustat rank corr (~0.84 ROE / 0.73 GM):
     v_noisy = v + k*sigma_cross*z with k=0.62 -> Spearman(v_noisy, v) ~ 0.85.

If D > C, overlaying FMP during the gap is expected to help; if C >= D, staying
stale-but-clean is right. 8yr, 2 starts, live config (1x, no vol flags), deployed cond.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
import numpy as np
from main_production_backtest import FastBacktester
from live_config import V12_LIVE_BACKTEST_CONFIG

FUND_KEYS = {"roe", "gross_margin", "debt_to_equity", "eps_surprise_last"}
STARTS = ["2018-01-02", "2018-01-17"]
END = "2025-12-31"
NOISE_K = 0.62   # calibrated: Spearman(fresh, noisy) ~ 0.85 = measured FMP-vs-Compustat


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def main():
    bt = FastBacktester()
    clear_deployed(bt)
    dates = sorted(bt.uni._feat_by_date.keys())
    idx = {d: i for i, d in enumerate(dates)}
    orig = bt.uni.get_feature_map

    def make_lagged(lag):
        def f(date, feature, members=None):
            if feature in FUND_KEYS and date in idx:
                date = dates[max(0, idx[date] - lag)]
            return orig(date, feature, members)
        return f

    # deterministic per-(symbol, feature, quarter) noise — stable within a quarter like a
    # vendor's value would be, resampled when the underlying quarter rolls
    def noisy_map(date, feature, members=None):
        vals = orig(date, feature, members)
        if feature not in FUND_KEYS or not vals:
            return vals
        arr = np.array(list(vals.values()), dtype=float)
        sigma = np.nanstd(arr)
        if not np.isfinite(sigma) or sigma == 0:
            return vals
        q = idx.get(date, 0) // 63
        out = {}
        for sym, v in vals.items():
            rng = np.random.default_rng(abs(hash((sym, feature, q))) % (2**32))
            out[sym] = v + NOISE_K * sigma * rng.standard_normal()
        return out

    cfg = dict(V12_LIVE_BACKTEST_CONFIG)
    for k in ("vol_scaling", "vol_target", "vol_lookback"):
        cfg.pop(k, None)

    variants = [("A FRESH (ceiling)", orig), ("B STALE 63td (now)", make_lagged(63)),
                ("C STALE 126td (late Aug)", make_lagged(126)), ("D FRESH+FMP-noise (overlay)", noisy_map)]
    print(f"{'variant':<30}{'start':<13}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}", flush=True)
    for label, fn in variants:
        bt.uni.get_feature_map = fn
        for st in STARTS:
            m = bt.run(st, END, cfg)
            print(f"{label:<30}{st:<13}{m['cagr']:>+8.1%}{m['sharpe']:>8.2f}{m['max_dd']:>+8.1%}", flush=True)
    bt.uni.get_feature_map = orig


if __name__ == "__main__":
    main()
