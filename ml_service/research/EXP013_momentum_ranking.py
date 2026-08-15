"""EXP-013 (IDEAS I-09 / I-10) — ALTERNATIVE MOMENTUM RANKINGS.

WHY THIS EXPERIMENT AND NOT ANOTHER
  EXP-003 decomposed the strategy's selection edge over 26 years:
      RANKING  +3.13pp CAGR (SEM 0.87), +0.065 Sharpe (SEM 0.027)   <- 102% of the edge
      FILTERS  -0.06pp,                 -0.001 Sharpe               <- nothing
  The composite score is the only part of security selection that works. So it is the only
  part worth trying to improve.

  It also bounds the prize: ALL of the current ranking is worth +0.065 Sharpe. A better
  ordering might add a similar amount. Nothing here can be a large win, and any arm that
  claims one is a bug.

VARIANTS (all causal, all read the same trailing panel the sleeve reads)
  multi   rank-average of 12-1, 6-1 and 3-1 skip-month momentum. Any single lookback is an
          arbitrary choice with large sampling error; averaging RANKS is variance reduction on
          the ESTIMATOR, not a new bet. Falsifiable claim: the ensemble beats every member.
  hi52    George-Hwang proximity to the 52-week high. Anchoring is a documented behavioural
          mechanism, and nearness is far more robust to one outlier month than a 12-month
          return. Same anomaly, less noisy estimator.
  voladj  (12-1)/vol_60d. AUDIT01 found vol_60d the strongest feature in the entire 29-feature
          panel (IC -0.050, beating every return feature); the sleeve currently harvests it
          only through a soft x1.15 nudge.
  blend   50/50 rank blend of multi and hi52.

DESIGN — WHY THIS IS SAFE
  Nothing is reimplemented. Each variant only RE-ORDERS the pool that the real
  `strategy1_momentum_reversal` already produced, so every filter, boost, regime rule and
  bear-sector exclusion is preserved bit-for-bit. `mom_pool` sets how wide that pool is:
    pool=4   -> refine the top 20 (a tweak to the existing ordering)
    pool=100 -> re-rank essentially the whole eligible pool (a genuine replacement)
  Running both separates "polish" from "replace".

PRE-REGISTERED
  1. Any arm beating base by more than ~+0.10 Sharpe is presumed BUGGY, not brilliant --
     that would exceed the total value of all existing ranking.
  2. Must hold on BOTH horizons.
  3. `multi` must beat its own single-horizon members or its stated mechanism is false.

Run:  python3 research/EXP013_momentum_ranking.py [8yr|26yr]
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
                     event_concentration, sign_verdict)

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)

ARMS = []
for kind in ("multi", "hi52", "voladj", "blend"):
    for pool in (4, 100):
        ARMS.append((f"{kind} pool={pool}", {"mom_rescore": kind, "mom_pool": pool}))
# control: does the deployed ordering itself survive being EQUAL-weighted? Separates
# "the ordering is better" from "the signal-proportional weighting is better".
ARMS.append(("orig-order eqwt", {"mom_rescore": "orig", "mom_pool": 4}))


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*102}\nEXP-013 ALTERNATIVE MOMENTUM RANKINGS — {HZ}, {len(STARTS)} starts"
          f"\n{'='*102}", flush=True)

    a = bt.run(STARTS[0], END, dict(DEPLOYED))
    b = bt.run(STARTS[0], END, {**DEPLOYED, "mom_rescore": None})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  PARITY mom_rescore=None vs untouched: {'IDENTICAL' if same else 'DIFFERENT'}",
          flush=True)
    if not same:
        print("  ABORT — hook changed the default path.")
        return

    base, base_curves = [], []
    for st in STARTS:
        m = bt.run(st, END, dict(DEPLOYED))
        base.append(stat(m["daily_values"])); base_curves.append(m["daily_values"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE (12-1 composite)  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  "
          f"MaxDD {bd.mean():.1%}", flush=True)

    print(f"\n  {'arm':<18}{'CAGR':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>8}"
          f"{'dCAGR':>8}{'dShrp':>8}{'dDD':>8}{'+CAGR':>7}{'+Shrp':>7}{'top5%':>7}"
          f"{'verdict':>11}", flush=True)
    for name, extra in ARMS:
        rows, curves = [], []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, **extra})
            rows.append(stat(m["daily_values"])); curves.append(m["daily_values"])
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
        conc = [event_concentration(base_curves[i], curves[i]) for i in range(len(rows))]
        t5 = np.mean([x["top5_share"] for x in conc])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        flag = "  <-- TOO GOOD, SUSPECT BUG" if (s - bs).mean() > 0.10 else ""
        print(f"  {name:<18}{c.mean():>+9.2%}{s.mean():>8.3f}{so.mean():>9.3f}{d.mean():>8.1%}"
              f"{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}{(d-bd).mean()*100:>+7.2f}p"
              f"{nc:>4}/{len(c)}{ns:>4}/{len(s)}{t5:>7.0%}{v:>11}{flag}", flush=True)
    print(f"\n  ceiling reminder: ALL existing ranking is worth +0.065 Sharpe (EXP-003).",
          flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
