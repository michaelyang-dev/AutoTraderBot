"""EXP-022 — fix the baseline (BUGS A9), then re-test both candidates against the REAL one.

THE DEFECT
  evalkit.DEPLOYED has no `credit_pct`, so cycles 1-19 compared everything against a baseline
  WITHOUT the HY-OAS credit gate. The live engine runs it: ibkr_engine.compute_credit_derisk()
  halves the gross-leverage target while HY-OAS >= its p95 expanding percentile
  (credit_gate.PCT=0.95, DERISK=0.5, live since 2026-07-18).

WHY IT MATTERS MOST FOR THE EXP-017/018 CLAIM
  That claim is "removing the vol overlay and running constant leverage improves drawdown,
  12/12 on both horizons". The credit gate ALSO buys crisis drawdown protection. If the two are
  SUBSTITUTES, then most of what constant-lower-leverage appeared to add was simply replacing a
  control my baseline was missing -- and the finding shrinks or disappears. That is the specific
  way this could be inflated, and it is my own error, not a data problem.

  EXP-014 (tranching) is far less exposed: variance reduction across rebalance phase does not
  compete with a credit gate for the same job. Testing that expectation is part of the point.

ARMS (12 starts, REAL financing, live_sizing)
  A0  baseline as used in cycles 1-19        (gate OFF)  <- what I have been comparing against
  A1  baseline as ACTUALLY DEPLOYED          (gate ON)   <- the honest baseline
  C0  const 1.10x no overlay, gate OFF
  C1  const 1.10x no overlay, gate ON
  B1  const 1.00x no overlay, gate ON
  T1  tranching K=4, gate ON

  gate effect on baseline      = A1 - A0
  candidate vs HONEST baseline = C1 - A1, B1 - A1, T1 - A1
  substitution                 = (C0 - A0) - (C1 - A1).  Large positive => the gate and constant
                                 leverage were buying the same protection and the earlier result
                                 was partly an artefact of the missing gate.

Run:  python3 research/EXP022_credit_gate_baseline.py [8yr|26yr]
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
TOTAL = 50_000.0
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}     # credit_gate.PCT / DERISK

ARMS = [
    ("A0 base gate-OFF",   [0],              {}),
    ("A1 base gate-ON",    [0],              GATE),
    ("C0 const1.10 gOFF",  [0],              {"leverage": 1.10, "vol_scaling": False}),
    ("C1 const1.10 gON",   [0],              {"leverage": 1.10, "vol_scaling": False, **GATE}),
    ("B1 const1.00 gON",   [0],              {"leverage": 1.00, "vol_scaling": False, **GATE}),
    ("T1 tranche K=4 gON", [0, 5, 10, 15],   GATE),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*96}\nEXP-022 CREDIT-GATE BASELINE FIX — {HZ}, {len(STARTS)} starts, "
          f"REAL financing\n{'='*96}", flush=True)

    res = {}
    for name, phases, extra in ARMS:
        rows = []
        ck = TOTAL / len(phases)
        for st in STARTS:
            per = [bt.run(st, END, {**DEPLOYED, **extra, "initial_capital": ck,
                                    "rebal_phase": p, "live_sizing": True,
                                    "financing_curve": True})["daily_values"]
                   for p in phases]
            rows.append(stat(sum(x.reindex(per[0].index).ffill() for x in per)))
        res[name] = rows
        c = np.array([r["cagr"] for r in rows]); d = np.array([r["dd"] for r in rows])
        print(f"  {name:<20} CAGR {c.mean():+6.2%}  MaxDD {d.mean():6.1%}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'arm':<20}{'CAGR':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}{'worstDD':>9}",
          flush=True)
    for name, _p, _e in ARMS:
        r = res[name]
        print(f"  {name:<20}{np.mean([x['cagr'] for x in r]):>+9.2%}"
              f"{np.mean([x['sharpe'] for x in r]):>9.3f}"
              f"{np.mean([x['sortino'] for x in r]):>9.3f}"
              f"{np.mean([x['dd'] for x in r]):>9.1%}"
              f"{np.min([x['dd'] for x in r]):>9.1%}", flush=True)

    def delta(a, b, lbl):
        ra, rb = res[a], res[b]
        c = np.array([ra[i]["cagr"] - rb[i]["cagr"] for i in range(len(ra))])
        s = np.array([ra[i]["sharpe"] - rb[i]["sharpe"] for i in range(len(ra))])
        d = np.array([ra[i]["dd"] - rb[i]["dd"] for i in range(len(ra))])
        print(f"  {lbl:<34}{c.mean()*100:>+8.2f}p{s.mean():>+9.3f}{d.mean()*100:>+8.2f}p"
              f"{int((c>0).sum()):>5}/{len(c)}{int((s>0).sum()):>5}/{len(s)}"
              f"{int((d>0).sum()):>5}/{len(d)}", flush=True)
        return c.mean(), s.mean(), d.mean()

    print(f"\n  {'paired delta':<34}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}"
          f"{'+Shrp':>7}{'+DD':>7}", flush=True)
    g = delta("A1 base gate-ON", "A0 base gate-OFF", "GATE effect on baseline (A1-A0)")
    c1 = delta("C1 const1.10 gON", "A1 base gate-ON", "C const1.10 vs HONEST base")
    delta("B1 const1.00 gON", "A1 base gate-ON", "B const1.00 vs HONEST base")
    delta("T1 tranche K=4 gON", "A1 base gate-ON", "T tranching K=4 vs HONEST base")
    c0 = delta("C0 const1.10 gOFF", "A0 base gate-OFF", "C vs OLD base (for comparison)")

    print(f"\n  SUBSTITUTION (C0-A0) - (C1-A1):  dCAGR {(c0[0]-c1[0])*100:+.2f}pp  "
          f"dSharpe {c0[1]-c1[1]:+.3f}  dMaxDD {(c0[2]-c1[2])*100:+.2f}pp", flush=True)
    print(f"  Large positive => the gate and constant leverage buy the SAME protection and the",
          flush=True)
    print(f"  EXP-017/018 result was partly an artefact of my missing-gate baseline.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
