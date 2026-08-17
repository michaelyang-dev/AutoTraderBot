"""EXP-037 — FINAL aggregate stats for the candidates that survived the year-by-year test.

EXP-036 (year-by-year, 25 calendar years, 26yr horizon) verdicts:
  O2 @1.25 d0.20          13/25 yrs, sum +58.4%, survives dropping best TWO (+20.7%)   PASS
  A tilt80 @1.00          11/25 yrs, sum +14.4%, DIES on dropping best (-1.6%)         FAIL
  A tilt70 @1.25          15/25 yrs, sum +67.8%, survives dropping best TWO (+25.5%)   PASS
  A+B tilt70 @1.25 d0.20  15/25 yrs, sum +77.2%, survives dropping best TWO (+35.4%)   PASS  <- best

  The tilt80 @1.00 failure is important: its RETURN edge was one or two years, though its
  DRAWDOWN record was the best of all (23/25 years, crisis +6.9pp). It is a risk instrument,
  not a return one, and must not be sold as the latter.

  The combined arm wins 60% of years, has the largest total excess, and still shows +35.4%
  after removing its two best years -- that is magnitude-driven breadth, not a spike.

WHAT IS STILL MISSING and is the point of this run
  EXP-036 reports only year-by-year DIFFERENCES. The aggregate MaxDD of the combined arm was
  never measured (EXP-035's A+B cell used the BROKEN sigexit, not the gate). And the tilt70
  @1.25 arm that WAS measured had MaxDD -59.0% vs LIVE -55.9% -- i.e. WORSE. So the combined
  arm's headline risk numbers are unknown, and without them "it passes" is not a recommendation.

  This measures the survivors' full aggregates on BOTH horizons, plus lower-leverage variants,
  so the drawdown cost of the tilt is visible rather than assumed away.

Run:  python3 research/EXP037_final_candidates.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_tranche import JointTrancheBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
LIVE = {**DEPLOYED, **GATE, "financing_curve": True, "live_sizing": True,
        "initial_capital": 50_000.0}
O2 = {**LIVE, "vol_scaling": False, "tranches": 4, "tranche_stride": 5}
T70 = {"mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09}

ARMS = [
    ("LIVE deployed", dict(LIVE)),
    ("O2 @1.00 (plain)", {**O2, "leverage": 1.00}),
    ("O2 @1.00 d0.20", {**O2, "leverage": 1.00, "credit_derisk": 0.20}),
    ("O2 @1.10 d0.20", {**O2, "leverage": 1.10, "credit_derisk": 0.20}),
    ("O2 @1.25 d0.20", {**O2, "leverage": 1.25, "credit_derisk": 0.20}),
    ("O2+T70 @1.00 d0.20", {**O2, "leverage": 1.00, "credit_derisk": 0.20, **T70}),
    ("O2+T70 @1.10 d0.20", {**O2, "leverage": 1.10, "credit_derisk": 0.20, **T70}),
    ("O2+T70 @1.25 d0.20", {**O2, "leverage": 1.25, "credit_derisk": 0.20, **T70}),
    # d0.20 was the BEST of the three derisk values tested (0.50/0.30/0.20) -- i.e. it sits at
    # the EDGE of the tested range, which is where an overfit optimum hides. If the improvement
    # keeps going at 0.10 and 0.00, the real finding is "cut exposure to ~zero while credit is
    # stressed", not "0.20 is special". If it reverses, 0.20 is a genuine interior optimum.
    ("O2 @1.25 d0.10", {**O2, "leverage": 1.25, "credit_derisk": 0.10}),
    ("O2 @1.25 d0.00", {**O2, "leverage": 1.25, "credit_derisk": 0.00}),
]
CRISIS = {2001, 2002, 2008, 2020, 2022}


def crisis_dd(v):
    o = [float(((s - s.cummax()) / s.cummax()).min())
         for y, s in v.groupby(v.index.year) if int(y) in CRISIS and len(s) > 20]
    return float(np.mean(o)) if o else float("nan")


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*106}\nEXP-037 FINAL CANDIDATES — {HZ}, {len(STARTS)} starts, $50k\n{'='*106}",
          flush=True)
    res = {}
    for name, cfg in ARMS:
        rows, cds, gs = [], [], []
        for st in STARTS:
            m = bt.run(st, END, dict(cfg))
            rows.append(stat(m["daily_values"])); cds.append(crisis_dd(m["daily_values"]))
            gs.append(m["avg_gross"])
        res[name] = (rows, float(np.nanmean(cds)), float(np.mean(gs)))
        print(f"  {name:<22} CAGR {np.mean([r['cagr'] for r in rows]):+6.2%}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    base = res["LIVE deployed"][0]
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base]); bcd = res["LIVE deployed"][1]
    print(f"\n  {'arm':<22}{'gross':>7}{'CAGR':>9}{'sdCAGR':>8}{'Sharpe':>8}{'Sortino':>8}"
          f"{'MaxDD':>8}{'worst':>8}{'crisis':>8}{'dCAGR':>8}{'dShrp':>8}{'dMaxDD':>8}"
          f"{'+Shrp':>7}", flush=True)
    for name, _c in ARMS:
        rows, cd, g = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows]); so = np.array([r["sortino"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<22}{g:>7.3f}{c.mean():>+9.2%}{c.std(ddof=1):>8.2%}{s.mean():>8.3f}"
              f"{so.mean():>8.3f}{d.mean():>8.1%}{d.min():>8.1%}{cd:>8.1%}"
              f"{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}"
              f"{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}", flush=True)
    print(f"\n  LIVE reference: CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  "
          f"MaxDD {bd.mean():.1%}  crisis {bcd:.1%}", flush=True)
    print(f"  A candidate is only worth deploying if it beats LIVE on CAGR *and* Sharpe *and*",
          flush=True)
    print(f"  does not give up crisis drawdown. Read the 'crisis' column, not just MaxDD.",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
