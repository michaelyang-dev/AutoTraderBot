"""EXP-014 — THE FULL AUDIT GATE on the one surviving candidate: phase tranching.

STATUS COMING IN (both horizons, 12 starts each, paired per start):
    8yr   K=4 vs base:  dCAGR +2.04pp  dSharpe +0.107  dMaxDD +3.18pp  sdCAGR ratio 0.484
    26yr  K=4 vs base:  dCAGR +0.52pp  dSharpe +0.031  dMaxDD +0.28pp  sdCAGR ratio 0.409
    worst-case MaxDD:   8yr -46.2% -> -38.8%   |   26yr -71.1% -> -64.5%
    capital control:    -0.35pp (default sizing) / +0.51pp (live sizing) -> effect is PHASE
    no phase is systematically better (4pp spread vs 2.04pp SEM) -> a nuisance parameter

THIS RESULT MAY BE INFLATED. Remaining attacks, all run here:

  1. REAL ACCOUNT SIZE. Everything so far used $50,000. The live IBKR account is ~$33,000.
     K=4 means $8,250 per tranche; at 1.49x that is ~$12,300 of gross across ~23 names, so
     ~$535/position. Names above ~$500/share become unbuyable and whole-share truncation bites
     hardest exactly where the idea needs it not to. This is the most likely killer.

  2. COST SENSITIVITY. Four sub-books each trade their own schedule. Turnover per dollar should
     be unchanged, but if it is not, the edge should die at higher costs. x1 / x2 / x3.

  3. EVENT CONCENTRATION of the EXCESS return vs base. A finding in this repo once passed 23/24
     start-consistency, walk-forward OOS, a sensitivity plateau and all three sub-periods -- and
     99.6% of its 26yr excess came from 5 days. >50% from top-5 days = reject.

  4. WHAT THE CLAIM ACTUALLY IS. Sign consistency is 8/12, not 11/12. That is EXPECTED: base is
     one draw from the phase distribution and K=4 is its average, so base must win ~half the
     starts by construction. The pre-registered statistic was the sd ratio, and it landed on
     0.484 / 0.409 against a theoretical 0.500. This is a VARIANCE result. Do not let the CAGR
     headline be quoted on its own.

Run:  python3 research/EXP014_tranching_audit_gate.py [8yr|26yr]
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

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
CAPITALS = [50_000.0, 33_000.0]
SCHEMES = [("K=1", [0]), ("K=2", [0, 10]), ("K=4", [0, 5, 10, 15])]
COSTS = [1, 2, 3]


def combine(cs):
    return sum(c.reindex(cs[0].index).ffill() for c in cs)


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*104}\nEXP-014 TRANCHING AUDIT GATE — {HZ}, {len(STARTS)} starts\n{'='*104}",
          flush=True)

    out = {}
    curves = {}
    for cap in CAPITALS:
        for cm in COSTS:
            for tag, phases in SCHEMES:
                rows, cvs = [], []
                for st in STARTS:
                    ck = cap / len(phases)
                    per = [bt.run(st, END, {**DEPLOYED, "initial_capital": ck,
                                            "rebal_phase": p, "cost_mult": cm,
                                            "live_sizing": True})["daily_values"]
                           for p in phases]
                    c = combine(per)
                    rows.append(stat(c)); cvs.append(c)
                out[(cap, cm, tag)] = rows
                curves[(cap, cm, tag)] = cvs
            print(f"  ${cap:,.0f} cost x{cm} done ({time.time()-t0:.0f}s)", flush=True)

    for cap in CAPITALS:
        print(f"\n  === capital ${cap:,.0f} (live_sizing, the closed-loop quantity calibration "
              f"the engine actually uses) ===", flush=True)
        print(f"  {'arm':<16}{'meanCAGR':>10}{'sdCAGR':>9}{'Sharpe':>9}{'sdShrp':>8}"
              f"{'meanDD':>9}{'worstDD':>9}{'dCAGR':>9}{'dShrp':>9}{'+Shrp':>7}"
              f"{'sdRatio':>9}{'top5%':>8}", flush=True)
        for cm in COSTS:
            b = out[(cap, cm, "K=1")]
            bc = np.array([r["cagr"] for r in b]); bsh = np.array([r["sharpe"] for r in b])
            for tag, _ph in SCHEMES:
                rows = out[(cap, cm, tag)]
                c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
                d = np.array([r["dd"] for r in rows])
                if tag == "K=1":
                    t5 = float("nan")
                else:
                    cc = [event_concentration(curves[(cap, cm, "K=1")][i],
                                              curves[(cap, cm, tag)][i])
                          for i in range(len(rows))]
                    t5 = np.mean([x["top5_share"] for x in cc])
                print(f"  {f'{tag} costx{cm}':<16}{c.mean():>+10.2%}{c.std(ddof=1):>9.2%}"
                      f"{s.mean():>9.3f}{s.std(ddof=1):>8.3f}{d.mean():>9.1%}{d.min():>9.1%}"
                      f"{(c-bc).mean()*100:>+8.2f}p{(s-bsh).mean():>+9.3f}"
                      f"{int((s>bsh).sum()):>4}/{len(s)}"
                      f"{c.std(ddof=1)/bc.std(ddof=1):>9.3f}"
                      f"{'' if tag=='K=1' else f'{t5:>8.0%}'}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
