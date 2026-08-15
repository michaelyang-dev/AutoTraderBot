"""EXP-012 — BOOK BREADTH. The architectural question the last four cycles converged on.

THE CHAIN OF EVIDENCE THAT MAKES THIS THE RIGHT EXPERIMENT
  EXP-003 (26yr, 12 starts, sticky-random controls, 24 paths per control):
      A deployed (filter+rank)      +11.26% / 0.509
      B filter only, random inside   +8.13% / 0.444
      C no filter, random             +8.19% / 0.445
    -> RANKING is worth +3.13pp CAGR (SEM 0.87) and +0.065 Sharpe (SEM 0.027).
    -> FILTERS are worth -0.06pp and -0.001 Sharpe. Literally nothing.
    The composite score carries 100% of the selection edge.

  EXP-004: over 26yr, passive quarterly EW SP1500 = +11.17% / 0.59 / -58.5%, versus the
    deployed +11.26% / 0.51 / -64.8%. We are at parity on return and BEHIND on risk.

  EXP-005: de-levering to 1.00x recovers +0.044 Sharpe (0.509 -> 0.553), 12/12 sign-consistent.
    Still short of passive's 0.59.

  Putting those together: the ranking genuinely adds value, but the book is so CONCENTRATED
  (top-5 of a ~1,100-name eligible pool) that idiosyncratic variance eats more Sharpe than the
  ranking creates. A broad index earns a better Sharpe with no skill at all, purely through
  diversification.

THE HYPOTHESIS
  If the ranking has real information, then holding MORE ranked names harvests the same edge on
  a larger, better-diversified base. Expected: CAGR falls slowly (later names are worse), vol
  falls faster (idiosyncratic risk diversifies as ~1/sqrt(N)), so Sharpe and MaxDD improve up to
  some optimum, then flatten as the book converges on the index.

WHY THIS IS NOT CONTRADICTED BY EXP-007
  EXP-007 (partial adjustment) also diluted the book and was catastrophic -- but it diluted with
  STALE, decaying legacy positions. This dilutes with FRESH, TOP-RANKED ones. Opposite thing.
  If breadth also fails, then the ranking's value is concentrated in the very top names and the
  strategy is correctly specified -- which is itself a clean answer.

PRE-REGISTERED
  1. Judged on SHARPE and MAXDD. CAGR is expected to fall; that alone does not kill it.
  2. Must improve on BOTH horizons.
  3. Expect a smooth interior optimum. A jagged profile = noise, not a curve.
  4. Live constraint: at $33k and 1.49x, 40 names is ~$1,230/position -- feasible. 80 is not.

Run:  python3 research/EXP012_book_breadth.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)

# top_n drives the momentum sleeve; `cap` must be loosened as N grows or the 10%-of-book cap
# binds and silently re-concentrates. Scaled so the cap never binds before ~2x equal weight.
ARMS = [("top_n=%d" % n, {"top_n": n, "cap": max(0.02, min(0.10, 2.0 / n))})
        for n in (3, 5, 8, 12, 20, 30, 40)]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*100}\nEXP-012 BOOK BREADTH — {HZ}, {len(STARTS)} starts\n{'='*100}",
          flush=True)

    res = {}
    for name, extra in ARMS:
        rows, grs = [], []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, **extra})
            rows.append(stat(m["daily_values"])); grs.append(m["avg_gross"])
        res[name] = (rows, np.mean(grs))
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        print(f"  {name:<10} cap={extra['cap']:.3f}  CAGR {c.mean():+6.2%}  "
              f"Sharpe {s.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'arm':<10}{'cap':>7}{'meanCAGR':>10}{'sdCAGR':>8}{'Sharpe':>9}{'sdShrp':>8}"
          f"{'Sortino':>9}{'vol':>8}{'meanDD':>9}{'worstDD':>9}{'avgGross':>10}", flush=True)
    for name, extra in ARMS:
        rows, g = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
        v = np.array([r["vol"] for r in rows])
        print(f"  {name:<10}{extra['cap']:>7.3f}{c.mean():>+10.2%}{c.std(ddof=1):>8.2%}"
              f"{s.mean():>9.3f}{s.std(ddof=1):>8.3f}{so.mean():>9.3f}{v.mean():>8.1%}"
              f"{d.mean():>9.1%}{d.min():>9.1%}{g:>10.3f}", flush=True)

    base_rows = res["top_n=5"][0]
    bc = np.array([r["cagr"] for r in base_rows]); bs = np.array([r["sharpe"] for r in base_rows])
    bd = np.array([r["dd"] for r in base_rows])
    print(f"\n  --- paired vs deployed top_n=5 ---", flush=True)
    print(f"  {'arm':<10}{'dCAGR':>9}{'dSharpe':>9}{'dSortino':>10}{'dMaxDD':>9}"
          f"{'+CAGR':>8}{'+Shrp':>8}{'verdict':>11}", flush=True)
    for name, _ in ARMS:
        rows, _g = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
        bso = np.array([r["sortino"] for r in base_rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<10}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
              f"{(so-bso).mean():>+10.3f}{(d-bd).mean()*100:>+8.2f}p"
              f"{nc:>5}/{len(c)}{ns:>5}/{len(s)}{v:>11}", flush=True)
    print(f"\n  passive reference (EXP-004, single path): 26yr EW SP1500 qtrly "
          f"+11.17% / 0.59 / -58.5%   |   8yr +9.79% / 0.52 / -41.5%", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
