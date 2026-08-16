"""EXP-028 — are the ROE filter and overlay-removal SUBSTITUTES or COMPLEMENTS?

EXP-024's matched-exposure control found the ROE arm's Sharpe edge vanishes against a
constant-leverage benchmark (dSharpe -0.003, 8/12). Read together with EXP-015 (the vol overlay
is worth ~-0.024 Sharpe vs constant leverage), that says ROE's +0.026 over the deployed baseline
is about the same size as simply DELETING the overlay -- and does not beat it.

If they are substitutes, adding ROE on top of a no-overlay book buys ~nothing, and the right
answer is the simpler change alone (two config values, deletes code) rather than both.
If they are complements, they stack and both belong.

This is the same question EXP-019 asked of tranching and overlay-removal, where the answer was
"additive, they fix independent problems". No reason to assume the same holds here: overlay
removal is a SIZING change and ROE is a SELECTION change, so a priori they should be independent
-- but EXP-024's matched control is direct evidence against that, which is why it gets measured.

ARMS (12 starts, gate ON, REAL financing)
  A  deployed
  B  ROE only               deployed + mom_quality_filter=roe, pool 3
  C  no-overlay only        vol_scaling OFF, leverage 1.10x
  D  BOTH
  interaction = (D-A) - [(B-A) + (C-A)];  ~0 => complements/additive, strongly negative => substitutes

Run:  python3 research/EXP028_roe_plus_nooverlay.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
BASE = {**DEPLOYED, **GATE, "financing_curve": True}
ROE = {"mom_quality_filter": "roe", "mom_quality_pool": 3}
NOOV = {"leverage": 1.10, "vol_scaling": False}

ARMS = [("A deployed", {}), ("B roe only", ROE), ("C no-overlay only", NOOV),
        ("D both", {**ROE, **NOOV})]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*96}\nEXP-028 ROE x NO-OVERLAY: substitutes or complements? — {HZ}, "
          f"{len(STARTS)} starts\n{'='*96}", flush=True)
    res = {}
    for name, extra in ARMS:
        rows = []
        for st in STARTS:
            rows.append(stat(bt.run(st, END, {**BASE, **extra})["daily_values"]))
        res[name] = rows
        print(f"  {name:<20} CAGR {np.mean([r['cagr'] for r in rows]):+6.2%}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'arm':<20}{'CAGR':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}{'worst':>9}",
          flush=True)
    for name, _ in ARMS:
        r = res[name]
        print(f"  {name:<20}{np.mean([x['cagr'] for x in r]):>+9.2%}"
              f"{np.mean([x['sharpe'] for x in r]):>9.3f}"
              f"{np.mean([x['sortino'] for x in r]):>9.3f}"
              f"{np.mean([x['dd'] for x in r]):>9.1%}"
              f"{np.min([x['dd'] for x in r]):>9.1%}", flush=True)

    A = res["A deployed"]
    D_ = {}
    print(f"\n  {'paired vs A':<20}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}"
          f"{'+Shrp':>7}{'+DD':>7}{'verdict':>10}", flush=True)
    for name, _ in ARMS[1:]:
        r = res[name]
        c = np.array([r[i]["cagr"] - A[i]["cagr"] for i in range(len(A))])
        s = np.array([r[i]["sharpe"] - A[i]["sharpe"] for i in range(len(A))])
        d = np.array([r[i]["dd"] - A[i]["dd"] for i in range(len(A))])
        D_[name] = (c.mean(), s.mean(), d.mean())
        nc, ns, v = sign_verdict(c, s, len(A))
        print(f"  {name:<20}{c.mean()*100:>+8.2f}p{s.mean():>+9.3f}{d.mean()*100:>+8.2f}p"
              f"{nc:>4}/{len(c)}{ns:>4}/{len(s)}{int((d>0).sum()):>4}/{len(d)}{v:>10}",
              flush=True)

    print(f"\n  === INTERACTION: (D-A) - [(B-A)+(C-A)] ===", flush=True)
    for j, lbl, sc in ((0, "CAGR", 100), (1, "Sharpe", 1), (2, "MaxDD", 100)):
        add = D_["B roe only"][j] + D_["C no-overlay only"][j]
        got = D_["D both"][j]
        print(f"  {lbl:<8} additive {add*sc:+7.3f}   actual {got*sc:+7.3f}   "
              f"interaction {(got-add)*sc:+7.3f}", flush=True)
    print(f"\n  ~0 => independent, both belong.  strongly NEGATIVE => substitutes, take the",
          flush=True)
    print(f"  simpler one (overlay removal is 2 config values and DELETES code; the ROE filter",
          flush=True)
    print(f"  adds a fundamental dependency to the signal path).", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
