"""EXP-006 (IDEAS I-06) — PREMISE CHECK: do recent INDEX ADDITIONS underperform?

MECHANISM BEING TESTED
  Index funds are FORCED buyers into an addition's effective date -- they must buy at the
  deadline regardless of price. That demand is mechanical and temporary, so the pop it creates
  should reverse. Our momentum sleeve is structurally attracted to exactly these names: a stock
  that just popped on index demand scores well on 12-1 momentum for a reason that carries no
  future return. If additions do underperform afterwards, excluding recent adds from the
  momentum pool is a free improvement using data we already hold (PIT membership dicts).

  Who is on the other side: index funds, by construction.

WHAT WOULD FALSIFY IT
  If recently-added names' forward returns match the rest of the pool WITHIN THE SAME
  cross-section, the idea is void and no backtest arm gets built.

LESSON APPLIED FROM EXP-002
  Every comparison here is WITHIN-DATE. EXP-002's first pass compared an "in" group whose
  membership share swung 2%-85% across the calendar against an "out" group, and produced
  t = +20 that was pure calendar confounding. Additions cluster at quarterly rebalance dates,
  so the identical trap is present here. Per-date means, then a t-stat across dates.

CAUSALITY
  Membership is read from the SAME PIT dicts the backtest trades on. A name counts as "added on
  date d" the first date it appears in a membership set, so the flag uses only information
  available at d. No forward knowledge of index changes is used.

Run:  python3 research/EXP006_index_addition_premise.py [8yr|26yr]
"""
import os
import sys
import pickle
import bisect

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PKL = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
       else "data/wrds/sp1500_universe_2000.pkl")
HORIZONS = (20, 60)
WINDOWS = (20, 60, 120)      # "added within the last N sessions"


def main():
    print(f"loading {PKL} ...", flush=True)
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    px = d["prices_df"]
    mems = {k: d[k] for k in ("sp500_mem", "sp400_mem", "sp600_mem")}
    dates = px.index
    print(f"  prices {px.shape}  membership snapshot counts: "
          + ", ".join(f"{k}={len(v)}" for k, v in mems.items()), flush=True)

    # ---- union SP1500 membership per trading date (PIT, as-of) ----
    keys = {k: sorted(v.keys()) for k, v in mems.items()}
    memb = []
    for dt in dates:
        s = set()
        for k, v in mems.items():
            kk = keys[k]
            j = bisect.bisect_right(kk, dt) - 1
            if j >= 0:
                s |= set(v[kk[j]])
        memb.append(s)
    sizes = [len(s) for s in memb]
    print(f"  SP1500 membership size: min {min(sizes)} med {int(np.median(sizes))} "
          f"max {max(sizes)}", flush=True)

    # ---- first date each symbol appears (= the addition event, causally observable) ----
    first_seen = {}
    for i, s in enumerate(memb):
        for sym in s:
            if sym not in first_seen:
                first_seen[sym] = i
    # names present on day 0 are pre-existing, not additions
    day0 = memb[0]
    adds = {s: i for s, i in first_seen.items() if s not in day0}
    print(f"  addition events observed after day 0: {len(adds):,}", flush=True)

    fwd = {h: (px.shift(-h) / px - 1) for h in HORIZONS}
    sample = list(range(252, len(dates) - max(HORIZONS), 5))
    print(f"  {len(sample)} cross-sections\n", flush=True)

    for W in WINDOWS:
        for H in HORIZONS:
            per_date = []
            n_flag = []
            for i in sample:
                mem_i = memb[i]
                r = fwd[H].iloc[i]
                a_vals, b_vals = [], []
                for sym in mem_i:
                    v = r.get(sym)
                    if v is None or not np.isfinite(v):
                        continue
                    j = adds.get(sym)
                    (a_vals if (j is not None and 0 < i - j <= W) else b_vals).append(v)
                if len(a_vals) >= 3 and len(b_vals) >= 50:
                    per_date.append(np.mean(a_vals) - np.mean(b_vals))
                    n_flag.append(len(a_vals))
            pdv = np.array(per_date)
            if len(pdv) < 20:
                print(f"  W={W:>3}d H={H:>3}d  too few usable dates ({len(pdv)})", flush=True)
                continue
            t = pdv.mean() / (pdv.std(ddof=1) / np.sqrt(len(pdv)))
            print(f"  added within {W:>3}d, forward {H:>3}d:  "
                  f"mean diff {pdv.mean()*100:+7.3f}pp   t={t:+6.2f}   "
                  f"dates {len(pdv):>4}  avg flagged/date {np.mean(n_flag):5.1f}  "
                  f"{100*(pdv<0).mean():.0f}% of dates negative", flush=True)

    print("\n  VERDICT GUIDE", flush=True)
    print("   mean diff clearly NEGATIVE (t < -2, most dates negative)", flush=True)
    print("     -> additions underperform; build the exclusion arm.", flush=True)
    print("   |t| < 2  -> void. Do not build anything; log and move on.", flush=True)
    print("   mean diff POSITIVE -> additions OUTperform; the momentum sleeve is right to", flush=True)
    print("     want them and an exclusion would destroy return.", flush=True)


if __name__ == "__main__":
    main()
