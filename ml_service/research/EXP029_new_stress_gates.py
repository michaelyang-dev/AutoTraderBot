"""EXP-029 (IDEAS I-31) — NEW LEVERAGE-GATE SIGNALS from the local FRED mirror.
Escalation-ladder level 9: cross-asset / credit-microstructure data.

THE DEPLOYED GATE uses HY-OAS *level* >= p95 expanding percentile. EXP-022 measured it at
+0.61pp CAGR / +0.021 Sharpe / +8.54pp MaxDD, 12/12 on all three -- it is the single most
valuable risk control in the system. The obvious question nobody has asked: is the LEVEL of the
aggregate spread the best available credit signal, or just the first one tried?

CANDIDATE: `ccc_bb` = (CCC-and-lower yield) - (BB yield), the QUALITY SPREAD INSIDE HIGH YIELD.
  Mechanism: the aggregate HY-OAS level rises for two very different reasons -- a broad
  repricing of credit risk (often benign, e.g. rates), or DISTRESS CONCENTRATING in the worst
  credits (the thing that precedes defaults and forced selling). ccc_bb isolates the second.
  It is 0.779 correlated with hy_oas, so ~22% of its variation is genuinely new information,
  and it is built from two series FRED still publishes daily (BAMLH0A3HYCEY, BAMLH0A1HYBBEY),
  so the live credit_gate cron could compute it with no new vendor.
  WHO LOSES: leveraged credit holders forced to delever when the tail of the distribution
  gaps -- and equity beta follows credit in exactly those episodes.

REFERENCE ARM, NOT DEPLOYABLE: `tedrate` (LIBOR - T-bill funding stress) is only 0.348
  correlated with hy_oas -- much more independent -- but the series ENDS 2022-01-21 because
  LIBOR was discontinued. It cannot be computed live going forward. It is included ONLY to
  bound how much a genuinely independent funding signal could be worth; any positive result
  from it is unusable and must be labelled so. Checking this BEFORE building saved testing a
  signal that could never ship.

CONTROLS (this whole family has a bad record -- the off-cadence gate passed 23/24 starts,
walk-forward OOS, a sensitivity plateau and all three sub-periods, then died at 99.6% event
concentration):
  - matched exposure (a gate that merely de-risks must not score for it)
  - guarded event concentration
  - both horizons, 12 starts
  - OR-combination with the deployed hy_oas gate, to test whether it ADDS or merely duplicates

Run:  python3 research/EXP029_new_stress_gates.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import (DEPLOYED, HORIZONS, END, starts, stat, sign_verdict,  # noqa: E402
                     matched_exposure_curve, event_concentration)

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
BASE = {**DEPLOYED, **GATE, "financing_curve": True}

ARMS = [
    ("ccc_bb only p95",      {"credit_pct": None, "gate_cols": ["ccc_bb"],
                              "gate_pct": 0.95, "gate_derisk": 0.5}),
    ("ccc_bb only p90",      {"credit_pct": None, "gate_cols": ["ccc_bb"],
                              "gate_pct": 0.90, "gate_derisk": 0.5}),
    ("OR(hy_oas, ccc_bb)",   {"credit_pct": None, "gate_cols": ["hy_oas", "ccc_bb"],
                              "gate_pct": 0.95, "gate_derisk": 0.5}),
    ("tedrate p95 [NOT DEPLOYABLE]", {"credit_pct": None, "gate_cols": ["tedrate"],
                                      "gate_pct": 0.95, "gate_derisk": 0.5}),
    ("OR(hy,ted) [NOT DEPLOYABLE]",  {"credit_pct": None, "gate_cols": ["hy_oas", "tedrate"],
                                      "gate_pct": 0.95, "gate_derisk": 0.5}),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*116}\nEXP-029 NEW STRESS GATES — {HZ}, {len(STARTS)} starts vs the DEPLOYED "
          f"hy_oas p95 gate\n{'='*116}", flush=True)

    base, bcv, bg = [], [], []
    for st in STARTS:
        m = bt.run(st, END, BASE)
        base.append(stat(m["daily_values"])); bcv.append(m["daily_values"])
        bg.append(m["avg_gross"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE (deployed hy_oas gate)  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  "
          f"MaxDD {bd.mean():.1%}  worst {bd.min():.1%}  gross {np.mean(bg):.4f}", flush=True)

    print(f"\n  {'arm':<32}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'worst':>8}{'gross':>8}"
          f"{'dCAGR':>8}{'dShrp':>8}{'dMaxDD':>8}{'+Shrp':>7}{'+DD':>7}"
          f"{'MATCHED dShrp':>15}{'top5%':>9}{'verdict':>10}", flush=True)
    for name, extra in ARMS:
        rows, cvs, gs, ms = [], [], [], []
        for st in STARTS:
            m = bt.run(st, END, {**BASE, **extra})
            r = stat(m["daily_values"]); rows.append(r); cvs.append(m["daily_values"])
            gs.append(m["avg_gross"])
            ref = matched_exposure_curve(bt, st, END, m["avg_gross"], cfg=BASE)
            ms.append(r["sharpe"] - ref["sharpe"])
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        ec = [event_concentration(bcv[i], cvs[i]) for i in range(len(rows))]
        ok = [x for x in ec if x["excess_meaningful"]]
        t5 = (f"{np.mean([x['top5_share'] for x in ok]):.0%}" if len(ok) >= len(ec) // 2
              else f"n/a({len(ok)}/{len(ec)})")
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<32}{c.mean():>+9.2%}{s.mean():>8.3f}{d.mean():>8.1%}{d.min():>8.1%}"
              f"{np.mean(gs):>8.4f}{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}"
              f"{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}{int((d>bd).sum()):>4}/{len(d)}"
              f"{np.mean(ms):>+15.3f}{t5:>9}{v:>10}", flush=True)

    print(f"\n  NOTE: the tedrate arms are RESEARCH-ONLY. LIBOR was discontinued and the series",
          flush=True)
    print(f"  ends 2022-01-21, so it cannot be computed live. Any positive result there bounds",
          flush=True)
    print(f"  what an independent FUNDING signal would be worth -- it does not ship.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
