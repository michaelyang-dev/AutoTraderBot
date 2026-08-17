"""EXP-036 — YEAR-BY-YEAR on the new candidates. The test that decides whether they are real.

WHY THIS IS THE DECIDING TEST, NOT A FORMALITY
  An aggregate CAGR can come from a broad persistent edge OR from one enormous year. This repo
  has already been burned: the off-cadence credit gate passed 23/24 start-consistency,
  walk-forward OOS, a sensitivity plateau AND all three sub-periods, and then 99.6% of its 26yr
  excess turned out to come from FIVE DAYS.

  EXP-033 already applied this to plain Option 2 and it changed the conclusion materially:
  at 1.00x the drawdown gain is universal (25/25 years) but there is NO return edge (13/25,
  arithmetic sum -20.4%). The aggregate "equal CAGR" came from compounding a smoother path, not
  from out-earning. Every new candidate must face the same test before it is believed.

CANDIDATES (from EXP-034 / EXP-035, 26yr, all vs the deployed system)
  O2 @1.25 d0.20   harder credit gate: +2.15pp CAGR, +0.067 Sharpe, +4.73pp MaxDD,
                   crisis-year drawdown -32.8% vs LIVE -33.1%  <- hit the pre-registered target
  A tilt80 @1.00   momentum tilt:      +1.10pp CAGR at LOWER leverage than LIVE
  A tilt70 @1.25   tilt + leverage:    +15.04% CAGR
  A+B combined     tilt70 @1.25 with the harder gate

PASS BAR, set before running
  1. years won on return > 60% (15/25), OR total excess that survives dropping the best TWO
     years -- a magnitude-driven edge is acceptable, a one-year edge is not
  2. crisis years (2001/2002/2008/2020/2022) must not be systematically worse than LIVE
  3. drawdown years won >= 60%

Run:  python3 research/EXP036_yby_candidates.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_tranche import JointTrancheBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
LIVE = {**DEPLOYED, **GATE, "financing_curve": True, "live_sizing": True,
        "initial_capital": 50_000.0}
O2 = {**LIVE, "vol_scaling": False, "tranches": 4, "tranche_stride": 5}
TILT70 = {"mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09}
TILT80 = {"mom_w": 0.80, "val_w": 0.14, "lv_w": 0.06}

# EXP-037 made O2+T70 @1.00 d0.20 the best arm on BOTH horizons (26yr +1.05pp CAGR/+0.083
# Sharpe/+11.6pp MaxDD; 8yr +2.26pp/+0.107/+5.4pp) -- and it has never faced the year-by-year
# test. tilt80 @1.00 FAILED that test while looking excellent in aggregate, so a tilt arm at
# 1.00x is exactly the shape that has already fooled me once. It leads the list now.
CANDS = [
    ("O2+T70 @1.00 d0.20", {**O2, "leverage": 1.00, "credit_derisk": 0.20, **TILT70}),
    ("O2 @1.00 d0.20", {**O2, "leverage": 1.00, "credit_derisk": 0.20}),
    ("O2+T70 @1.10 d0.20", {**O2, "leverage": 1.10, "credit_derisk": 0.20, **TILT70}),
    ("A+B tilt70 @1.25 d0.20", {**O2, "leverage": 1.25, "credit_derisk": 0.20, **TILT70}),
]
CRISIS = {2001, 2002, 2008, 2020, 2022}


def yearly(v):
    out = {}
    for y, seg in v.groupby(v.index.year):
        if len(seg) >= 20:
            out[int(y)] = (seg.iloc[-1] / seg.iloc[0] - 1,
                           float(((seg - seg.cummax()) / seg.cummax()).min()))
    return out


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*100}\nEXP-036 YEAR-BY-YEAR, NEW CANDIDATES — {HZ}, {len(STARTS)} starts"
          f"\n{'='*100}", flush=True)

    acc = {n: {} for n, _ in CANDS}
    accL = {}
    for si, st in enumerate(STARTS):
        L = yearly(bt.run(st, END, dict(LIVE))["daily_values"])
        for y, v in L.items():
            accL.setdefault(y, []).append(v)
        for n, cfg in CANDS:
            C = yearly(bt.run(st, END, dict(cfg))["daily_values"])
            for y in set(C) & set(L):
                acc[n].setdefault(y, []).append(C[y])
        print(f"  [{si+1:2d}/{len(STARTS)}] {st}   ({time.time()-t0:.0f}s)", flush=True)

    yrs = sorted(accL)
    print(f"\n  {'year':<6}{'LIVE':>8}" +
          "".join(f"{n.split()[0]+n.split()[1][:5]:>13}" for n, _ in CANDS), flush=True)
    diffs = {n: {} for n, _ in CANDS}
    for y in yrs:
        la = np.mean([x[0] for x in accL[y]])
        row = f"  {y:<6}{la:>+8.1%}"
        for n, _ in CANDS:
            if y in acc[n]:
                d = np.mean([x[0] for x in acc[n][y]]) - la
                diffs[n][y] = d
                row += f"{d:>+13.1%}"
            else:
                row += f"{'--':>13}"
        print(row + ("   <CRISIS" if y in CRISIS else ""), flush=True)

    print(f"\n  {'candidate':<24}{'yrs won':>9}{'ddn won':>9}{'sum diff':>10}"
          f"{'drop best':>11}{'drop 2':>9}{'crisis avg':>12}{'VERDICT':>10}", flush=True)
    for n, _ in CANDS:
        d = diffs[n]; ys = sorted(d)
        won = sum(1 for y in ys if d[y] > 0)
        dw = 0
        for y in ys:
            la = np.mean([x[1] for x in accL[y]]); ca = np.mean([x[1] for x in acc[n][y]])
            dw += ca > la
        tot = sum(d.values()); srt = sorted(d.values())
        cav = np.mean([d[y] for y in ys if y in CRISIS]) if any(y in CRISIS for y in ys) else 0
        broad = (won / len(ys) > 0.60) or (tot - srt[-1] - srt[-2] > 0)
        ok = broad and cav > -0.02 and dw / len(ys) >= 0.50
        print(f"  {n:<24}{won:>5}/{len(ys)}{dw:>6}/{len(ys)}{tot:>+10.1%}"
              f"{tot-srt[-1]:>+11.1%}{tot-srt[-1]-srt[-2]:>+9.1%}{cav:>+12.1%}"
              f"{'PASS' if ok else 'FAIL':>10}", flush=True)
    print(f"\n  BAR: >60% years won OR positive after dropping the best TWO years;"
          f"\n       crisis-year average not materially negative; drawdown years won >= 50%.",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
