"""
Opportunity-set test driver: identical lean 12-1 momentum on SMALL-CAP (R2000)
vs LARGE/MID (SP1500). Same signal, same params, same period -> any gap is the
opportunity set. Loads universes sequentially to cap memory.

Run: cd ml_service && ./venv/bin/python research/alpha_universe_test.py
"""
import os, sys, pickle, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.alpha_lean_momentum import lean_momentum, show

DATA = "data/wrds"


def combine_membership(dicts):
    """Union of several {date: members} dicts via as-of at each present date."""
    all_dates = sorted(set().union(*[set(d.keys()) for d in dicts]))
    keyed = []
    for d in dicts:
        kd = sorted(d.keys())
        keyed.append((kd, d))
    out = {}
    for dt in all_dates:
        u = set()
        for kd, d in keyed:
            i = np.searchsorted(kd, dt, side="right") - 1
            if i >= 0:
                u.update(d[kd[i]])
        out[dt] = u
    return out


def run_grid(prices, members, label, periods):
    for (start, end) in periods:
        print(f"\n  --- {label}  {start[:4]}-{end[:4]} ---")
        for top_n in (15, 25, 50):
            for cost in (10, 30):
                r = lean_momentum(prices, members, start, end, top_n=top_n,
                                  rebal_days=20, cost_bps=cost, min_price=5.0)
                show(f"  top{top_n} rd20 cost{cost}bps", r)


if __name__ == "__main__":
    t0 = time.time()

    # ---- R2000 small caps (2006-2025) ----
    print("=" * 70 + "\nSMALL-CAP  (R2000, true Norgate membership)\n" + "=" * 70)
    with open(os.path.join(DATA, "r2000_universe.pkl"), "rb") as f:
        d = pickle.load(f)
    px_r2k, mem_r2k = d["prices_df"], d["membership"]
    print(f"[r2000 loaded {time.time()-t0:.0f}s; prices {px_r2k.shape}]")
    run_grid(px_r2k, mem_r2k, "R2000",
             [("2006-01-01", "2025-12-31"), ("2016-06-01", "2025-12-31")])
    del d, px_r2k, mem_r2k

    # ---- SP1500 large/mid (2016-2025, same lean method) ----
    print("\n" + "=" * 70 + "\nLARGE/MID  (SP1500, union of SP500/400/600)\n" + "=" * 70)
    t1 = time.time()
    with open(os.path.join(DATA, "complete_sp1500_universe.pkl"), "rb") as f:
        d = pickle.load(f)
    px_sp = d["prices_df"]
    mem_sp = combine_membership([d["sp500_mem"], d["sp400_mem"], d["sp600_mem"]])
    print(f"[sp1500 loaded {time.time()-t1:.0f}s; prices {px_sp.shape}]")
    run_grid(px_sp, mem_sp, "SP1500", [("2016-06-01", "2025-12-31")])
    print(f"\n[total {time.time()-t0:.0f}s]")
