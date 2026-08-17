"""EXP-038 — is derisk=0.20 a real interior optimum, or just the edge of what I tested?

THE PROBLEM
  EXP-034/037 swept credit-gate derisk at 0.50 / 0.30 / 0.20 and found monotone improvement:
      26yr @1.25x   CAGR +14.46 -> +14.74 -> +14.87%   MaxDD -57.4 -> -52.8 -> -51.2%
  0.20 was the BEST value tested -- which means it sits at the EDGE of the range. That is
  precisely where an overfit optimum hides: if the true function keeps improving past 0.20, the
  real finding is "cut exposure to ~zero while credit is stressed", not "0.20 is special", and
  quoting 0.20 as a tuned parameter would be a fitted artefact.

  The 8yr cannot answer this: there all three values converge (+27.27 / +27.25 / +27.18) because
  2018-25 contains no genuine credit crisis, so the gate barely fires and every derisk collapses
  to the same book. The 26yr is the only discriminating horizon -- 2001, 2002 and 2008 actually
  trigger it.

WHAT WOULD MEAN WHAT
  keeps improving through 0.10 to 0.00  -> the finding is "go flat in credit stress"; 0.20 is
                                           arbitrary and should not be quoted as tuned
  flattens between 0.20 and 0.00        -> the gate is saturating; anything <=0.20 is equivalent
                                           and the choice is operational, not statistical
  reverses below 0.20                   -> 0.20 is a genuine interior optimum

  Only the third outcome justifies calling 0.20 a chosen parameter. The first two mean the
  honest description is a RANGE, not a point.

Run:  python3 research/EXP038_derisk_edge.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_tranche import JointTrancheBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
LIVE = {**DEPLOYED, "credit_pct": 0.95, "credit_derisk": 0.5, "financing_curve": True,
        "live_sizing": True, "initial_capital": 50_000.0}
O2 = {**LIVE, "vol_scaling": False, "tranches": 4, "tranche_stride": 5, "leverage": 1.25}
DERISKS = [0.50, 0.30, 0.20, 0.10, 0.00]
CRISIS = {2001, 2002, 2008, 2020, 2022}


def crisis_dd(v):
    o = [float(((s - s.cummax()) / s.cummax()).min())
         for y, s in v.groupby(v.index.year) if int(y) in CRISIS and len(s) > 20]
    return float(np.mean(o)) if o else float("nan")


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*94}\nEXP-038 DERISK EDGE CHECK — {HZ}, {len(STARTS)} starts, O2 @1.25x"
          f"\n{'='*94}", flush=True)
    base = [stat(bt.run(st, END, dict(LIVE))["daily_values"]) for st in STARTS]
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  LIVE: CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  MaxDD {bd.mean():.1%}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  {'derisk':<9}{'gross':>7}{'CAGR':>9}{'sdCAGR':>8}{'Sharpe':>8}{'MaxDD':>8}"
          f"{'worst':>8}{'crisis':>8}{'dCAGR':>8}{'dShrp':>8}{'+Shrp':>7}", flush=True)
    prev = None
    for d in DERISKS:
        rows, cds, gs = [], [], []
        for st in STARTS:
            m = bt.run(st, END, {**O2, "credit_derisk": d})
            rows.append(stat(m["daily_values"])); cds.append(crisis_dd(m["daily_values"]))
            gs.append(m["avg_gross"])
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        dd = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        arrow = "" if prev is None else ("  ^" if c.mean() > prev else "  v")
        prev = c.mean()
        print(f"  {d:<9.2f}{np.mean(gs):>7.3f}{c.mean():>+9.2%}{c.std(ddof=1):>8.2%}"
              f"{s.mean():>8.3f}{dd.mean():>8.1%}{dd.min():>8.1%}{np.nanmean(cds):>8.1%}"
              f"{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}{ns:>4}/{len(s)}{arrow}",
              flush=True)
    print(f"\n  keeps improving to 0.00 -> the finding is 'go FLAT in credit stress';"
          f"\n     0.20 is arbitrary and must be quoted as a RANGE, not a tuned value."
          f"\n  flattens below 0.20     -> saturating; any value <=0.20 equivalent."
          f"\n  reverses below 0.20     -> 0.20 is a genuine interior optimum.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
