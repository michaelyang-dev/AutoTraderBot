"""EXP-033 — is Option 2's gain BROAD, or is it one or two lucky years?

THE QUESTION THIS ANSWERS
  Every number so far is a 26-year (or 8-year) aggregate. An aggregate can be produced by a
  broad, small, persistent edge OR by one enormous year. Those are completely different things
  to own, and only the first is worth deploying.

  This repo has already been burned by exactly this failure: the off-cadence credit gate passed
  23/24 start-consistency, walk-forward OOS, a sensitivity plateau AND all three sub-periods --
  and then 99.6% of its 26yr excess turned out to come from FIVE DAYS. It helped in 2008 and
  2020 and hurt in 2001 and 2022. The daily event-concentration test caught that; this is the
  same question asked at CALENDAR-YEAR resolution, which is the scale an owner actually
  experiences.

METHOD
  For every start, compute each arm's return and max-drawdown WITHIN each calendar year, then
  average across all starts that cover that year. Report:
    - per-year return for baseline vs Option 2, and the difference
    - how many years Option 2 wins
    - the contribution of the single best year to the total excess (drop-the-best-year test)
  A broad edge shows a majority of positive years and survives dropping its best one.

Run:  python3 research/EXP033_year_by_year.py [8yr|26yr]
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
from evalkit import DEPLOYED, HORIZONS, END, starts  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
LIVE = {**DEPLOYED, **GATE, "financing_curve": True, "live_sizing": True,
        "initial_capital": 50_000.0}                       # deployed: 1.49x + vol overlay
# leverage is settable: the 1.00x comparison is at MUCH lower exposure than the deployed
# 1.49x, so it conflates "Option 2 is different" with "Option 2 holds less". OPT2_LEV=1.25
# runs it at roughly the deployed book's DRAWDOWN, which is the fair return comparison.
_LEV = float(os.environ.get("OPT2_LEV", 1.00))
OPT2 = {**LIVE, "vol_scaling": False, "leverage": _LEV, "tranches": 4}


def yearly(v):
    """calendar-year return and within-year max drawdown from a NAV path."""
    out = {}
    for y, seg in v.groupby(v.index.year):
        if len(seg) < 20:
            continue
        out[int(y)] = (seg.iloc[-1] / seg.iloc[0] - 1,
                       float(((seg - seg.cummax()) / seg.cummax()).min()))
    return out


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*92}\nEXP-033 YEAR BY YEAR — {HZ}, {len(STARTS)} starts, $50k, gate ON, "
          f"OPT2 leverage {_LEV:.2f}x vs LIVE 1.49x"
          f"\n{'='*92}", flush=True)

    accA, accB = {}, {}
    for si, st in enumerate(STARTS):
        a = yearly(bt.run(st, END, dict(LIVE))["daily_values"])
        b = yearly(bt.run(st, END, dict(OPT2))["daily_values"])
        for y in set(a) & set(b):
            accA.setdefault(y, []).append(a[y]); accB.setdefault(y, []).append(b[y])
        print(f"  [{si+1:2d}/{len(STARTS)}] {st}   ({time.time()-t0:.0f}s)", flush=True)

    yrs = sorted(accA)
    print(f"\n  {'year':<6}{'LIVE ret':>10}{'OPT2 ret':>10}{'diff':>9}   "
          f"{'LIVE ddn':>10}{'OPT2 ddn':>10}{'diff':>9}{'  n':>4}", flush=True)
    wins = wins_dd = 0
    diffs = {}
    for y in yrs:
        ra = np.mean([x[0] for x in accA[y]]); rb = np.mean([x[0] for x in accB[y]])
        da = np.mean([x[1] for x in accA[y]]); db = np.mean([x[1] for x in accB[y]])
        diffs[y] = rb - ra
        wins += rb > ra; wins_dd += db > da
        flag = "  <<<" if abs(rb - ra) > 0.06 else ""
        print(f"  {y:<6}{ra:>+10.1%}{rb:>+10.1%}{rb-ra:>+9.1%}   "
              f"{da:>10.1%}{db:>10.1%}{db-da:>+9.1%}{len(accA[y]):>4}{flag}", flush=True)

    n = len(yrs)
    tot = sum(diffs.values())
    best = max(diffs, key=lambda k: diffs[k])
    worst = min(diffs, key=lambda k: diffs[k])
    print(f"\n  YEARS WON on return : {wins}/{n}   on drawdown: {wins_dd}/{n}", flush=True)
    print(f"  best year for OPT2  : {best} ({diffs[best]:+.1%})", flush=True)
    print(f"  worst year for OPT2 : {worst} ({diffs[worst]:+.1%})", flush=True)
    print(f"  sum of yearly diffs : {tot:+.1%}", flush=True)
    print(f"  DROP the best year  : {tot - diffs[best]:+.1%}  "
          f"({'still positive — broad edge' if tot - diffs[best] > 0 else 'goes NEGATIVE — one-year artefact'})",
          flush=True)
    print(f"  DROP best TWO years : "
          f"{tot - diffs[best] - sorted(diffs.values())[-2]:+.1%}", flush=True)
    print(f"\n  READ: >60% of years won AND positive after dropping the best year = broad.",
          flush=True)
    print(f"        Concentrated in 1-2 years = the off-cadence-gate failure mode.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
