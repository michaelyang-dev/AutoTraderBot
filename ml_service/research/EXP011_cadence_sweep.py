"""EXP-011 — REBALANCE CADENCE, re-measured at the 12-start standard, with cost sensitivity.

WHERE THIS CAME FROM
  EXP-007's CONTROL arm was meant to be a throwaway: 5-day cadence with a FULL move, included
  only to separate "cadence" from "partial move". It came back essentially free --
      base 20d:  +22.75% / 0.793 / -39.4%   sdCAGR 7.07pp
      5d      :  +23.50% / 0.785 / -38.3%   sdCAGR 4.69pp   (ratio 0.663)
  i.e. +0.75pp CAGR, -0.008 Sharpe, +1.04pp MaxDD, and a 34% cut in start-date dispersion,
  from a ONE-LINE config change with no new live engineering.

  Note this also corrects my own reading: from the first 4 starts the control looked like it
  cost -4 to -7pp. Over 12 starts it is +0.75pp. Per-start values ranged -9.2pp to +12.8pp.
  Exactly the failure mode the many-start rule exists to prevent.

  It does NOT contradict the inherited "shortening cadence under stress is catastrophic"
  finding -- that was `rebal_stress`, a cadence change applied ONLY during stress, which is a
  different (and conditional) rule. This is a uniform cadence.

WHAT IS AND IS NOT BEING CLAIMED
  Lower dispersion ACROSS START DATES means the outcome depends less on when you happened to
  begin. That is worth something, but it is NOT alpha and NOT a within-path improvement. The
  within-path deltas here (Sharpe -0.008, MaxDD +1.04pp) are ~zero. Say so plainly.

THE OBVIOUS SUSPICION
  4x more rebalances should cost 4x more turnover. That it does not implies most 5-day
  rebalances are near-no-ops (the no-trade band is |delta| < 0.3% of NAV, and top-5 momentum
  picks are persistent). If that is the explanation, the result must survive higher costs --
  and if it does not, it was never there. Hence the cost sweep.

Run:  python3 research/EXP011_cadence_sweep.py [8yr|26yr]
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

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)

ARMS = []
for rd in (5, 10, 20):
    for cm in (1, 2, 3):
        ARMS.append((f"{rd:>2}d cost x{cm}", {"rebal_days": rd, "cost_mult": cm}))


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*98}\nEXP-011 CADENCE SWEEP x COST — {HZ}, {len(STARTS)} starts\n{'='*98}",
          flush=True)

    res = {}
    for name, extra in ARMS:
        rows = []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, **extra})
            rows.append(stat(m["daily_values"]))
        res[name] = rows
        c = np.array([r["cagr"] for r in rows])
        print(f"  {name}: CAGR {c.mean():+6.2%}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'arm':<16}{'meanCAGR':>10}{'sdCAGR':>9}{'Sharpe':>9}{'sdShrp':>8}"
          f"{'Sortino':>9}{'meanDD':>9}{'worstDD':>9}", flush=True)
    for name, _ in ARMS:
        rows = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
        print(f"  {name:<16}{c.mean():>+10.2%}{c.std(ddof=1):>9.2%}{s.mean():>9.3f}"
              f"{s.std(ddof=1):>8.3f}{so.mean():>9.3f}{d.mean():>9.1%}{d.min():>9.1%}",
              flush=True)

    print(f"\n  --- vs deployed (20d, cost x1), paired per start ---", flush=True)
    base = res["20d cost x1"]
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  {'arm':<16}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>8}{'+Shrp':>8}"
          f"{'sdCAGR ratio':>14}{'verdict':>11}", flush=True)
    for name, _ in ARMS:
        rows = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<16}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
              f"{(d-bd).mean()*100:>+8.2f}p{nc:>5}/{len(c)}{ns:>5}/{len(s)}"
              f"{c.std(ddof=1)/bc.std(ddof=1):>13.3f}{v:>11}", flush=True)
    print(f"\n  reference: K=4 phase tranching, 8yr — sdCAGR ratio 0.484, "
          f"dCAGR +2.04pp, dSharpe +0.107, dMaxDD +3.18pp", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
