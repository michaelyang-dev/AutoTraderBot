"""EXP-034 — IMPROVING OPTION 2. Two ideas, both pointed at the gap the year-by-year exposed.

WHAT THE YEAR-BY-YEAR SHOWED
  Option 2 @1.00x : drawdown better in 25/25 years, but NO return edge (13/25, arithmetic
                    sum -20.4%). Equal CAGR comes from compounding a smoother path.
  Option 2 @1.25x : return edge appears (+49.2% summed, survives dropping the best TWO years)
                    but drawdown goes WORSE in exactly the crises that matter --
                    2008 -4.7pp, 2020 -2.9pp.

  So the leverage dial trades the one thing Option 2 is genuinely good at (crisis drawdown) for
  return. What we actually want is the SMOOTHNESS OF 1.00x WITH THE RETURN OF 1.25x, and the
  only way to get it is to restore crisis protection from a source that does NOT cost ~1pp/yr
  the way the deleted vol overlay did.

IDEA 1 — MORE TRANCHES (K sweep)
  K=4 was chosen arbitrarily (20/4 = a clean 5-day stride), never swept. If the benefit is
  genuine variance reduction it should keep improving with K at a decaying rate (~1/sqrt(K)),
  and flatten. At $50k, K=8 is $6,250 per sub-book -- still ~$270/position at 1.0x, feasible.
  If K=8 is no better than K=4, the mechanism is saturated and K=4 is the right build.
  FALSIFIABLE: sd CAGR should fall monotonically in K. If it does not, the variance story is
  wrong and the K=4 result was luck.

IDEA 2 — BUY BACK CRISIS PROTECTION WITH THE CREDIT GATE, NOT THE VOL OVERLAY
  EXP-022 measured the credit gate at +0.61pp CAGR / +0.021 Sharpe / +8.54pp MaxDD, 12/12 on
  ALL THREE -- it is the single most valuable and least costly risk control in the system, and
  EXP-029 showed it cannot be improved by better signals. It is currently set to halve leverage
  (derisk 0.50) when HY-OAS is at its p95 expanding percentile.
  So: run Option 2 at the HIGHER leverage that produces the return, and make the gate bite
  HARDER when it fires. That converts a permanent ~1pp/yr tax (the overlay) into a conditional
  one that only pays in the ~9% of days credit is actually stressed.
  PRIOR SUPPORT, not a fresh fit: the DYN walk-forward independently selected derisk 0.30 in
  all 22 out-of-sample years.

MULTIPLE-TESTING DISCIPLINE
  This tests 4 K values and 3 derisk values = 7 new configs on top of a candidate already
  chosen from prior work. Any single winning cell must be treated as one of seven. The K sweep
  is judged on MONOTONICITY (a shape, not a peak) and the derisk sweep on whether the crisis
  years specifically recover -- both harder to fake than a best-cell.

Run:  python3 research/EXP034_improve_option2.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from livemirror_tranche import JointTrancheBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
LIVE = {**DEPLOYED, **GATE, "financing_curve": True, "live_sizing": True,
        "initial_capital": 50_000.0}
O2 = {**LIVE, "vol_scaling": False}

ARMS = [
    ("LIVE (1.49x + overlay)", dict(LIVE)),
    # --- idea 1: K sweep at 1.00x (stride = 20/K) ---
    ("O2 K=2 @1.00", {**O2, "leverage": 1.00, "tranches": 2, "tranche_stride": 10}),
    ("O2 K=4 @1.00", {**O2, "leverage": 1.00, "tranches": 4, "tranche_stride": 5}),
    ("O2 K=5 @1.00", {**O2, "leverage": 1.00, "tranches": 5, "tranche_stride": 4}),
    ("O2 K=10 @1.00", {**O2, "leverage": 1.00, "tranches": 10, "tranche_stride": 2}),
    # --- idea 2: return leverage + harder credit gate ---
    ("O2 K=4 @1.25 d0.50", {**O2, "leverage": 1.25, "tranches": 4, "credit_derisk": 0.50}),
    ("O2 K=4 @1.25 d0.30", {**O2, "leverage": 1.25, "tranches": 4, "credit_derisk": 0.30}),
    ("O2 K=4 @1.25 d0.20", {**O2, "leverage": 1.25, "tranches": 4, "credit_derisk": 0.20}),
    ("O2 K=4 @1.40 d0.30", {**O2, "leverage": 1.40, "tranches": 4, "credit_derisk": 0.30}),
]

CRISIS = [2001, 2002, 2008, 2020, 2022]


def crisis_dd(v):
    """worst within-year drawdown across the crisis years present in this path."""
    out = []
    for y, seg in v.groupby(v.index.year):
        if int(y) in CRISIS and len(seg) > 20:
            out.append(float(((seg - seg.cummax()) / seg.cummax()).min()))
    return float(np.mean(out)) if out else float("nan")


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*104}\nEXP-034 IMPROVING OPTION 2 — {HZ}, {len(STARTS)} starts, $50k"
          f"\n{'='*104}", flush=True)
    res = {}
    for name, cfg in ARMS:
        rows, cds = [], []
        for st in STARTS:
            m = bt.run(st, END, dict(cfg))
            rows.append(stat(m["daily_values"])); cds.append(crisis_dd(m["daily_values"]))
        res[name] = (rows, float(np.nanmean(cds)))
        c = np.array([r["cagr"] for r in rows])
        print(f"  {name:<22} CAGR {c.mean():+6.2%}   ({time.time()-t0:.0f}s)", flush=True)

    base = res["LIVE (1.49x + overlay)"][0]
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base]); bcd = res["LIVE (1.49x + overlay)"][1]

    print(f"\n  {'arm':<22}{'CAGR':>9}{'sdCAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'worst':>8}"
          f"{'CRISIS ddn':>12}{'dCAGR':>8}{'dShrp':>8}{'dMaxDD':>8}{'+Shrp':>7}{'verdict':>10}",
          flush=True)
    for name, _cfg in ARMS:
        rows, cd = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        tag = "" if name.startswith("LIVE") else v
        print(f"  {name:<22}{c.mean():>+9.2%}{c.std(ddof=1):>8.2%}{s.mean():>8.3f}"
              f"{d.mean():>8.1%}{d.min():>8.1%}{cd:>12.1%}"
              f"{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}"
              f"{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}{tag:>10}", flush=True)

    print(f"\n  === IDEA 1: does sd CAGR fall MONOTONICALLY in K? (the falsifiable shape) ===",
          flush=True)
    for k in ("O2 K=2 @1.00", "O2 K=4 @1.00", "O2 K=5 @1.00", "O2 K=10 @1.00"):
        r = res[k][0]
        print(f"    {k:<16} sdCAGR {np.std([x['cagr'] for x in r], ddof=1):.2%}   "
              f"Sharpe {np.mean([x['sharpe'] for x in r]):.3f}", flush=True)
    print(f"    (LIVE sdCAGR {bc.std(ddof=1):.2%}) — if sd stops falling, the mechanism is",
          flush=True)
    print(f"    saturated and the extra sub-books are pure operational cost.", flush=True)

    print(f"\n  === IDEA 2: does a harder credit gate restore CRISIS drawdown at 1.25x? ===",
          flush=True)
    print(f"    LIVE crisis-year drawdown: {bcd:.1%}", flush=True)
    for k in ("O2 K=4 @1.00", "O2 K=4 @1.25 d0.50", "O2 K=4 @1.25 d0.30",
              "O2 K=4 @1.25 d0.20", "O2 K=4 @1.40 d0.30"):
        rows, cd = res[k]
        print(f"    {k:<20} crisis ddn {cd:>7.1%} ({cd-bcd:+.1f}pp vs LIVE)   "
              f"CAGR {np.mean([x['cagr'] for x in rows]):+.2%}", flush=True)
    print(f"\n  WANT: a cell with crisis drawdown <= LIVE's AND CAGR clearly above LIVE's.",
          flush=True)
    print(f"  That is the 'smoothness of 1.00x with the return of 1.25x' target. Anything else",
          flush=True)
    print(f"  is just re-picking a point on the leverage line already mapped in EXP-031.",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
