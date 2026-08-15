"""EXP-009 (IDEAS I-05) — VOL-NORMALISED TRAILING STOP.

THE DEFECT
  The stop is a flat 40% for every holding. Annualised vol across the SP1500 runs from ~15% to
  ~80%. A 40% drawdown is therefore a ~2-sigma annual event on a quiet name and an ordinary
  fluctuation on a volatile one: the SAME rule applies a completely different confidence level
  to every position. It fires on high-vol names for no informational reason, and never protects
  low-vol ones at all.

THE FIX
  threshold_i = clamp(k * annualised vol_i, lo, hi), using the panel's causal vol feature.

WHY IT SHOULD HELP AND WHAT IT COSTS
  Normalising equalises the false-positive rate across names. Expect fewer whipsaw exits on
  volatile momentum winners (which is where the strategy's return lives -- EXP-007 showed
  returns collapse when the book is diluted away from fresh concentrated winners) and genuine
  protection on quiet names that currently have none.
  Cost: a tighter stop on low-vol names raises turnover; full modelled costs are charged inside.

WHY THIS TARGETS THE RIGHT AXIS
  EXP-004 found the 26yr strategy is at PARITY with passive equal-weight SP1500 on return
  (+11.26% vs +11.17%) and BEHIND on drawdown (-64.8% vs -58.5%). Drawdown, not CAGR, is the
  axis with room. A stop rule is the most direct instrument on it.

PRE-REGISTERED
  1. Judge on Sharpe and MaxDD. A CAGR-only win is not a win here.
  2. Must hold on BOTH horizons -- single-horizon results have been wrong before (umd_crash).
  3. Must be robust across k. If one k works and its neighbours do not, it is fitted.
  4. Event-concentration on the excess: >50% from 5 days => reject.

Run:  python3 research/EXP009_vol_normalised_stop.py [8yr|26yr]
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

ARMS = [
    ("vol k=1.0 [.25,.60]", {"stop_mode": "vol", "stop_k": 1.0}),
    ("vol k=1.2 [.25,.60]", {"stop_mode": "vol", "stop_k": 1.2}),
    ("vol k=1.5 [.25,.60]", {"stop_mode": "vol", "stop_k": 1.5}),
    ("vol k=1.2 [.30,.70]", {"stop_mode": "vol", "stop_k": 1.2,
                             "stop_lo": 0.30, "stop_hi": 0.70}),
    ("vol k=1.2 vol_20d",   {"stop_mode": "vol", "stop_k": 1.2, "stop_feat": "vol_20d"}),
    ("flat 30% (control)",  {"trailing_stop": 0.30}),
    ("flat 50% (control)",  {"trailing_stop": 0.50}),
    ("no stop (control)",   {"trailing_stop": None}),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*104}\nEXP-009 VOL-NORMALISED TRAILING STOP — {HZ}, {len(STARTS)} starts"
          f"\n{'='*104}", flush=True)

    a = bt.run(STARTS[0], END, dict(DEPLOYED))
    b = bt.run(STARTS[0], END, {**DEPLOYED, "stop_mode": "flat"})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  PARITY stop_mode='flat' vs untouched: {'IDENTICAL' if same else 'DIFFERENT'}",
          flush=True)
    if not same:
        print("  ABORT — hook changed the default path.")
        return

    base, base_curves = [], []
    for st in STARTS:
        m = bt.run(st, END, dict(DEPLOYED))
        base.append(stat(m["daily_values"])); base_curves.append(m["daily_values"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base]); bso = np.array([r["sortino"] for r in base])
    print(f"  BASE (flat 40%)  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  "
          f"Sortino {bso.mean():.3f}  MaxDD {bd.mean():.1%}  worst {bd.min():.1%}", flush=True)

    print(f"\n  {'arm':<24}{'CAGR':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>8}{'worstDD':>9}"
          f"{'dCAGR':>8}{'dShrp':>8}{'dDD':>8}{'+Shrp':>7}{'top5%':>7}{'verdict':>11}",
          flush=True)
    for name, extra in ARMS:
        rows, curves = [], []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, **extra})
            rows.append(stat(m["daily_values"])); curves.append(m["daily_values"])
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
        conc = [event_concentration(base_curves[i], curves[i]) for i in range(len(rows))]
        t5 = np.mean([x["top5_share"] for x in conc])
        nc, ns, verdict = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<24}{c.mean():>+9.2%}{s.mean():>8.3f}{so.mean():>9.3f}"
              f"{d.mean():>8.1%}{d.min():>9.1%}{(c-bc).mean()*100:>+7.2f}p"
              f"{(s-bs).mean():>+8.3f}{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}"
              f"{t5:>7.0%}{verdict:>11}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
