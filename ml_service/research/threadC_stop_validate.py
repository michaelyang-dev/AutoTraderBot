"""
THREAD C — confirmatory backtest after the C1 gate.

C1 graduated ONLY rollover (held names that pulled back 1m OUTPERFORM -> "hold the dip"),
and killed every per-name feature for a vol/regime-scaled stop (vol_20d flat both periods).
Prediction from C1: the uniform 40% trailing stop is deliberately DEEP to avoid cutting
bounce-prone dips; a TIGHTER stop should HURT, a looser/none should not help much.

This validates that in the full run() (not an overlay): sweep trailing_stop and confirm
40% is at/near the frontier. If nothing beats it -> Thread C closes as a NULL with backtest
evidence, consistent with the C1 gate and prior 'aging flat/inverted' work.
Both periods, multi-start, deployed data OFF, levered 1.49x + financing (overlay math).
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
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
BASE_L, RATE = 1.49, 0.063
STOPS = [0.25, 0.40, 0.55, None]  # None -> no trailing stop


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


def lever_flat(r1x, L=BASE_L, rate=RATE):
    """flat 1.49x + financing on borrowed (matches how the levered baseline is reported)."""
    return L * r1x - max(0.0, L - 1.0) * (rate / 252)


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 92, flush=True)
        print(f"PERIOD {pname} | {len(starts)}-start | 1.49x flat + fin {RATE:.1%}", flush=True)
        print("=" * 92, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'trailing_stop':<16}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}"
        print(hdr); print("-" * len(hdr), flush=True)
        for stop in STOPS:
            cs, vs_, ss, ds = [], [], [], []
            for st in starts:
                cfg = dict(V12); cfg["trailing_stop"] = stop
                r = bt.run(st, end, cfg)["daily_values"].pct_change().dropna()
                rl = lever_flat(r)
                c, v, s, d = stats(rl); cs.append(c); vs_.append(v); ss.append(s); ds.append(d)
            lbl = f"{stop:.0%}" if stop is not None else "NONE"
            star = "  <- LIVE" if stop == 0.40 else ""
            print(f"{lbl:<16}{np.mean(cs):>+8.1%}{np.mean(vs_):>7.1%}{np.mean(ss):>8.2f}"
                  f"{np.mean(ds):>+8.1%}{star}", flush=True)
        del bt
    print("\nC1 predicted tighter=worse (cuts bounce-prone dips). If 40% not beaten -> Thread C NULL.", flush=True)


if __name__ == "__main__":
    main()
