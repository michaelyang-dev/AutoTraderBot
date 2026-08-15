"""EXP-018 — corrected cost audit for the constant-leverage candidate (fixes BUGS A8b).

EXP-017 established, paired per start, 12 starts:

    26yr  C const 1.10x no-overlay vs A deployed 1.49x+overlay:
          dCAGR +0.66pp   dSharpe +0.049   dMaxDD +2.52pp
          sign: 10/12 CAGR, 12/12 SHARPE, 12/12 DRAWDOWN          <- decisive
    26yr  B const 1.00x no-overlay:
          dCAGR +0.12pp   dSharpe +0.060   dMaxDD +6.55pp
          sign:  7/12 CAGR, 12/12 SHARPE, 12/12 DRAWDOWN, top-5 day share 8%
     8yr  C: dCAGR -0.20pp  dSharpe +0.026  8/12 Sharpe            <- marginal

But EXP-017's cost row was WRONG: it compared the candidate at cost x2 against the DEPLOYED at
cost x1, which measures the cost increase (it hits both arms) rather than the difference between
them. That row said nothing and must not be quoted.

This runs the comparison correctly: every cost multiplier is paired against the deployed arm at
the SAME multiplier. It also uses the guarded event_concentration (BUGS A8a), which returns NaN
when the total excess is too small for a share to be meaningful.

WHY THE COST TEST MATTERS HERE SPECIFICALLY
  Removing the vol overlay removes leverage churn -- the overlay re-scales every position each
  rebalance as its 40-day estimate drifts. So the candidate should trade LESS, and its advantage
  should GROW with cost. If instead the advantage shrinks with cost, something is wrong with my
  understanding of where the gain comes from, and that is worth knowing.

PRE-REGISTERED
  1. If dSharpe grows (or holds) as cost rises, the mechanism is confirmed: this is a
     turnover-reducing simplification, not a return bet.
  2. If dSharpe shrinks materially with cost, the gain is coming from somewhere I have not
     identified and the result should not be promoted.

Run:  python3 research/EXP018_constant_cost_audit.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import (DEPLOYED, HORIZONS, END, starts, stat,  # noqa: E402
                     event_concentration)

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
CANDS = [("B const 1.00x", {"leverage": 1.00, "vol_scaling": False}),
         ("C const 1.10x", {"leverage": 1.10, "vol_scaling": False})]
COSTS = [1, 2, 3]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*102}\nEXP-018 CORRECTED COST AUDIT — {HZ}, {len(STARTS)} starts, "
          f"REAL financing\n{'='*102}", flush=True)
    print(f"  every candidate is paired against the DEPLOYED arm at the SAME cost multiplier",
          flush=True)

    print(f"\n  {'cost':<6}{'arm':<16}{'CAGR':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}"
          f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}{'+Shrp':>7}{'+DD':>7}"
          f"{'top5%':>9}", flush=True)
    for cm in COSTS:
        base, bcv = [], []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, "cost_mult": cm, "financing_curve": True})
            base.append(stat(m["daily_values"])); bcv.append(m["daily_values"])
        bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
        bd = np.array([r["dd"] for r in base]); bso = np.array([r["sortino"] for r in base])
        print(f"  {'x%d' % cm:<6}{'A deployed':<16}{bc.mean():>+9.2%}{bs.mean():>9.3f}"
              f"{bso.mean():>9.3f}{bd.mean():>9.1%}", flush=True)
        for name, extra in CANDS:
            rows, cvs = [], []
            for st in STARTS:
                m = bt.run(st, END, {**DEPLOYED, **extra, "cost_mult": cm,
                                     "financing_curve": True})
                rows.append(stat(m["daily_values"])); cvs.append(m["daily_values"])
            c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
            d = np.array([r["dd"] for r in rows]); so = np.array([r["sortino"] for r in rows])
            cc = [event_concentration(bcv[i], cvs[i]) for i in range(len(rows))]
            ok = [x for x in cc if x["excess_meaningful"]]
            t5 = (f"{np.mean([x['top5_share'] for x in ok]):.0%}" if len(ok) >= len(cc) // 2
                  else f"n/a({len(ok)}/{len(cc)})")
            print(f"  {'':<6}{name:<16}{c.mean():>+9.2%}{s.mean():>9.3f}{so.mean():>9.3f}"
                  f"{d.mean():>9.1%}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
                  f"{(d-bd).mean()*100:>+8.2f}p{int((c>bc).sum()):>4}/{len(c)}"
                  f"{int((s>bs).sum()):>4}/{len(s)}{int((d>bd).sum()):>4}/{len(d)}"
                  f"{t5:>9}", flush=True)
        print(f"    ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  READING: dSharpe should HOLD or GROW with cost -- removing the vol overlay",
          flush=True)
    print(f"           removes leverage churn, so the candidate trades LESS. If dSharpe",
          flush=True)
    print(f"           SHRINKS with cost, the gain is not where I think it is.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
