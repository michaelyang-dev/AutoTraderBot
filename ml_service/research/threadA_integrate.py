"""
THREAD A — equity + micro-futures integration (A2 overlay / A4 realloc).

A1 passed: micro-only book is tradeable at $50k (net Sharpe ~0.4-0.5, corr~0 to equity).
A3: crisis convexity partial (2020/2022/2018 yes; 2008/2011 no — bonds gone).
Now: does adding the micro stream to the v12 equity book Pareto-improve or give a DD-first
trade? Modern era 2010-2025 (micro book realism window), multi-start equity.

  A2 overlay : r = r_eq_lev + w * r_micro   (futures are MARGIN-funded; small cash carve-out
               ~$5-6k covers micro margin at $50k, so equity stays ~fully invested)
  A4 realloc : r = (1-w) * r_eq_lev + w * r_micro   (carve cash from equity into futures)

Honest caveat: at $50k TOTAL, overlay stacks the equity 1.49x margin with futures margin —
realistic only for small w. Both reported; correlation shown.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML); sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np, pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
BASE_L, RATE = 1.49, 0.063
STARTS = ["2010-01-04", "2010-01-19", "2010-02-01"]
END = "2025-12-31"


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(r):
    r = r.dropna(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    c = (1 + r).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min())


def main():
    micro = pd.read_parquet(os.path.join(ML, "research", "_threadA_micro50k.parquet"))["micro50k"]
    micro.index = pd.to_datetime(micro.index)
    bt = FastBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl"); clear_deployed(bt)
    eq_runs = []
    for st in STARTS:
        r = bt.run(st, END, dict(V12))["daily_values"].pct_change().dropna()
        r_lev = BASE_L * r - max(0.0, BASE_L - 1.0) * (RATE / 252)
        eq_runs.append(r_lev)

    ms, es = stats(micro), None
    print("=" * 92, flush=True)
    print("THREAD A integration | equity v12 @1.49x + micro-$50k book | 2010-2025, 3-start eq", flush=True)
    print("=" * 92, flush=True)
    print(f"micro book standalone: CAGR {ms[0]:+.1%}  Vol {ms[1]:.1%}  Sharpe {ms[2]:.2f}  MaxDD {ms[3]:+.1%}", flush=True)

    def combine(w, mode):
        cs, vs, ss, ds, corrs = [], [], [], [], []
        for r_eq in eq_runs:
            idx = r_eq.index.intersection(micro.index)
            re = r_eq.reindex(idx); mi = micro.reindex(idx).fillna(0)
            if mode == "overlay":
                rc = re + w * mi
            else:  # realloc
                rc = (1 - w) * re + w * mi
            c, v, s, d = stats(rc)
            cs.append(c); vs.append(v); ss.append(s); ds.append(d); corrs.append(re.corr(mi))
        return np.mean(cs), np.mean(vs), np.mean(ss), np.mean(ds), np.mean(corrs)

    # equity-only baseline (aligned window)
    b = combine(0.0, "realloc")
    print(f"\n{'variant':<26}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'corr(eq,mi)':>12}", flush=True)
    print(f"{'EQUITY ONLY 1.49x':<26}{b[0]:>+8.1%}{b[1]:>7.1%}{b[2]:>8.2f}{b[3]:>+8.1%}{b[4]:>12.2f}", flush=True)
    print("-" * 69, flush=True)
    for mode in ("overlay", "realloc"):
        for w in (0.1, 0.2, 0.3, 0.5):
            c, v, s, d, cr = combine(w, mode)
            better = (s > b[2] + 0.02) or (d > b[3] + 0.01 and c > b[0] - 0.005)
            tag = "  <-improve" if better else ""
            print(f"{mode+' w='+str(w):<26}{c:>+8.1%}{v:>7.1%}{s:>8.2f}{d:>+8.1%}{cr:>12.2f}{tag}", flush=True)
    print("\nGATE: GO only if a w gives Pareto-improve or a clean DD-first trade vs equity-only,"
          "\n      remembering crisis convexity is partial (no 2008/2011 hedge without bonds).", flush=True)


if __name__ == "__main__":
    main()
