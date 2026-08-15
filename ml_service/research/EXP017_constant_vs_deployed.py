"""EXP-017 — THE DECISIVE TEST: constant leverage, no vol overlay, vs the deployed engine.

WHY
  EXP-015 compared the deployed inverse-vol overlay against constant leverage at MATCHED average
  gross and found the overlay mildly HARMFUL:
      26yr  dCAGR -0.91pp  dSharpe -0.0241  dSortino -0.0428  dMaxDD +2.89pp
       8yr  (interpolated) dCAGR ~-1.0pp    dSharpe ~-0.016   dMaxDD ~+0.6pp
  i.e. it costs ~1pp of CAGR and ~0.02 of Sharpe, and buys ~0.6-2.9pp of drawdown.

  It also showed something stronger. Plain CONSTANT 1.10x leverage with vol_scaling OFF beat the
  deployed 1.49x+overlay on ALL THREE axes, on BOTH horizons:
      26yr  +11.93% / 0.558 / -61.7%   vs deployed  +11.26% / 0.509 / -64.2%
       8yr  +22.77% / 0.824 / -37.1%   vs deployed  +22.75% / 0.793 / -39.4%

  If that survives a proper paired test it is the most valuable result available here, because
  it makes the engine SIMPLER: it deletes the 40-day vol estimator, the leverage churn it
  causes, and the stale-vol failure mode -- and it is a pure config change, not new live code.

  EXP-015 IS NOT SUFFICIENT TO CONCLUDE THIS. It compared MEANS across starts. It never checked
  sign consistency per start, event concentration, cost sensitivity, or honest financing. A
  mean difference with 5/12 sign consistency is noise, and this repo has been burned by exactly
  that (the value-weight finding, retracted at 6/12).

WHAT THIS RUNS
  Paired per start, 12 starts, both financing assumptions:
    A  deployed            1.49x + inverse-vol overlay
    B  constant 1.00x      vol_scaling OFF
    C  constant 1.10x      vol_scaling OFF   <- the candidate
    D  constant 1.20x      vol_scaling OFF   <- robustness: is 1.10 a knife-edge?
  plus cost x2 on the candidate, and event concentration of C's excess over A.

PRE-REGISTERED
  1. C must beat A on Sharpe in >=9/12 starts on BOTH horizons. Mean improvement is not enough.
  2. C must not be a knife-edge: B and D should also be respectable, or the result is fitted
     to one leverage value.
  3. Event concentration of the excess <50% from top-5 days.
  4. Must survive honest time-varying financing, which FAVOURS the deployed arm (it runs more
     gross, so it gains more from the corrected rate).

Run:  python3 research/EXP017_constant_vs_deployed.py [8yr|26yr]
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

ARMS = [
    ("A deployed 1.49x+overlay", {}),
    ("B const 1.00x no-overlay", {"leverage": 1.00, "vol_scaling": False}),
    ("C const 1.10x no-overlay", {"leverage": 1.10, "vol_scaling": False}),
    ("D const 1.20x no-overlay", {"leverage": 1.20, "vol_scaling": False}),
    ("C cost x2",                {"leverage": 1.10, "vol_scaling": False, "cost_mult": 2}),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*104}\nEXP-017 CONSTANT LEVERAGE vs DEPLOYED — {HZ}, {len(STARTS)} starts"
          f"\n{'='*104}", flush=True)

    for fin in (False, True):
        tag = "REAL DFF+1.5pp" if fin else "FLAT 6.3%"
        res, curves, gross = {}, {}, {}
        for name, extra in ARMS:
            rows, cvs, gs = [], [], []
            for st in STARTS:
                m = bt.run(st, END, {**DEPLOYED, **extra, "financing_curve": fin})
                rows.append(stat(m["daily_values"])); cvs.append(m["daily_values"])
                gs.append(m["avg_gross"])
            res[name] = rows; curves[name] = cvs; gross[name] = float(np.mean(gs))

        print(f"\n  === financing: {tag} ===", flush=True)
        print(f"  {'arm':<26}{'CAGR':>9}{'sdCAGR':>8}{'Sharpe':>9}{'Sortino':>9}"
              f"{'MaxDD':>9}{'worstDD':>9}{'avgGross':>10}", flush=True)
        for name, _ in ARMS:
            r = res[name]
            c = np.array([x["cagr"] for x in r]); s = np.array([x["sharpe"] for x in r])
            so = np.array([x["sortino"] for x in r]); d = np.array([x["dd"] for x in r])
            print(f"  {name:<26}{c.mean():>+9.2%}{c.std(ddof=1):>8.2%}{s.mean():>9.3f}"
                  f"{so.mean():>9.3f}{d.mean():>9.1%}{d.min():>9.1%}{gross[name]:>10.4f}",
                  flush=True)

        A = res["A deployed 1.49x+overlay"]
        ac = np.array([x["cagr"] for x in A]); as_ = np.array([x["sharpe"] for x in A])
        ad = np.array([x["dd"] for x in A]); aso = np.array([x["sortino"] for x in A])
        print(f"\n  {'paired vs A (deployed)':<26}{'dCAGR':>9}{'dSharpe':>9}{'dSortino':>10}"
              f"{'dMaxDD':>9}{'+CAGR':>8}{'+Shrp':>8}{'+DD':>7}{'top5%':>8}", flush=True)
        for name, _ in ARMS[1:]:
            r = res[name]
            c = np.array([x["cagr"] for x in r]); s = np.array([x["sharpe"] for x in r])
            so = np.array([x["sortino"] for x in r]); d = np.array([x["dd"] for x in r])
            cc = [event_concentration(curves["A deployed 1.49x+overlay"][i], curves[name][i])
                  for i in range(len(r))]
            t5 = np.mean([x["top5_share"] for x in cc])
            print(f"  {name:<26}{(c-ac).mean()*100:>+8.2f}p{(s-as_).mean():>+9.3f}"
                  f"{(so-aso).mean():>+10.3f}{(d-ad).mean()*100:>+8.2f}p"
                  f"{int((c>ac).sum()):>5}/{len(c)}{int((s>as_).sum()):>5}/{len(s)}"
                  f"{int((d>ad).sum()):>4}/{len(d)}{t5:>8.0%}", flush=True)

    print(f"\n  BAR: candidate C must beat A on Sharpe in >=9/12 starts on BOTH horizons,",
          flush=True)
    print(f"       with B and D also respectable (not a knife-edge) and top5 < 50%.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
