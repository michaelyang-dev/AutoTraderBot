"""EXP-015 (IDEAS I-20) — DOES THE DEPLOYED INVERSE-VOL OVERLAY DO ANYTHING?

THE CLAIM ON FILE
  DYNAMIC_LEVERAGE_FINDINGS states the deployed vol overlay has "~no timing skill"
  (+0.018 Sharpe on 8yr, +0.001 on 26yr, and negative with more starts) and that its value is
  purely a LEVEL effect -- it runs less gross on average, and less gross is what helps.

  If that is true, the overlay is REPLACEABLE BY A NUMBER. Running constant leverage at the
  overlay's own realised avg_gross would deliver the same risk with:
    - no 40-day estimator,
    - no leverage churn (and its transaction costs),
    - one fewer moving part in the live engine,
    - no stale-vol failure mode.
  Simplification with equal performance is a legitimate win, and this repo has never tested it
  directly at the 12-start standard.

  It also matters for a second reason: EXP-008 killed a better vol ESTIMATOR partly on the
  argument that "there is no timing skill to improve". That argument is only sound if this
  experiment confirms the premise. Testing your own reasoning is the point.

DESIGN
  A  DEPLOYED           1.49x with vol_scaling on            -> realised avg_gross G
  B  CONSTANT-MATCHED   vol_scaling OFF, leverage tuned per horizon so avg_gross ~ G
  C  a small leverage grid with scaling off, so the match can be interpolated rather than
     assumed, and so the sensitivity of the comparison to the match is visible

  If A - B is ~0 on Sharpe and MaxDD, the overlay is decorative and should be replaced by a
  constant. If A - B is clearly positive, the overlay has real timing skill and the repo's own
  documentation is wrong -- which would ALSO be a finding, and would reopen EXP-008.

PRE-REGISTERED
  1. Judge on Sharpe and MaxDD at MATCHED gross. CAGR alone is meaningless here.
  2. Must agree on both horizons.
  3. Report the realised avg_gross of every arm so the match is auditable, not asserted.

Run:  python3 research/EXP015_overlay_vs_constant.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GRID = [0.90, 1.00, 1.10, 1.20, 1.30, 1.49]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*94}\nEXP-015 OVERLAY vs CONSTANT LEVERAGE — {HZ}, {len(STARTS)} starts"
          f"\n{'='*94}", flush=True)

    dep, dep_g = [], []
    for st in STARTS:
        m = bt.run(st, END, dict(DEPLOYED))
        dep.append(stat(m["daily_values"])); dep_g.append(m["avg_gross"])
    G = float(np.mean(dep_g))
    dc = np.array([r["cagr"] for r in dep]); ds = np.array([r["sharpe"] for r in dep])
    dd = np.array([r["dd"] for r in dep]); dso = np.array([r["sortino"] for r in dep])
    print(f"  A DEPLOYED (1.49x + inverse-vol overlay)", flush=True)
    print(f"    CAGR {dc.mean():+.2%}  Sharpe {ds.mean():.3f}  Sortino {dso.mean():.3f}  "
          f"MaxDD {dd.mean():.1%}  worst {dd.min():.1%}  avgGross {G:.4f}", flush=True)

    print(f"\n  C constant-leverage grid (vol_scaling OFF)", flush=True)
    print(f"  {'lev':<8}{'avgGross':>10}{'CAGR':>10}{'Sharpe':>9}{'Sortino':>9}"
          f"{'MaxDD':>9}{'worstDD':>9}", flush=True)
    pts = []
    for L in GRID:
        rows, grs = [], []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, "leverage": L, "vol_scaling": False})
            rows.append(stat(m["daily_values"])); grs.append(m["avg_gross"])
        g = float(np.mean(grs))
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
        pts.append((g, c.mean(), s.mean(), so.mean(), d.mean(), d.min()))
        print(f"  {L:<8.2f}{g:>10.4f}{c.mean():>+10.2%}{s.mean():>9.3f}{so.mean():>9.3f}"
              f"{d.mean():>9.1%}{d.min():>9.1%}", flush=True)

    pts.sort()
    xs = np.array([p[0] for p in pts])
    def interp(j):
        return float(np.interp(G, xs, np.array([p[j] for p in pts])))
    print(f"\n  B CONSTANT-MATCHED at the overlay's own avgGross {G:.4f} (interpolated)",
          flush=True)
    print(f"    CAGR {interp(1):+.2%}  Sharpe {interp(2):.3f}  Sortino {interp(3):.3f}  "
          f"MaxDD {interp(4):.1%}  worst {interp(5):.1%}", flush=True)

    print(f"\n  === A - B (the overlay's TIMING skill, level effect removed) ===", flush=True)
    print(f"    dCAGR   {(dc.mean()-interp(1))*100:+.2f}pp", flush=True)
    print(f"    dSharpe {ds.mean()-interp(2):+.4f}", flush=True)
    print(f"    dSortino{dso.mean()-interp(3):+.4f}", flush=True)
    print(f"    dMaxDD  {(dd.mean()-interp(4))*100:+.2f}pp", flush=True)
    print(f"    dWorstDD{(dd.min()-interp(5))*100:+.2f}pp", flush=True)
    v = ds.mean() - interp(2)
    print(f"\n  READING: |dSharpe| < 0.02 => the overlay is DECORATIVE and replaceable by a "
          f"constant.\n           got {v:+.4f} -> "
          f"{'decorative' if abs(v) < 0.02 else ('real timing skill' if v > 0 else 'HARMFUL')}",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
