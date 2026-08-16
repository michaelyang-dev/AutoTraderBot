"""EXP-019 — do the two audited candidates STACK, or do they overlap?

CANDIDATE 1 (tranching): K=4 phase sub-books.
    26yr +0.52pp CAGR / +0.031 Sharpe / worst DD -71.1%->-64.5%, sd ratio 0.409
CANDIDATE 2 (no overlay): vol_scaling OFF, constant leverage 1.10x.
    26yr +0.14pp CAGR / +0.034 Sharpe (12/12) / +2.49pp MaxDD (12/12)

WHY THEY MIGHT NOT STACK
  Both reduce the same thing: variance from a nuisance. Tranching averages away the phase
  lottery; removing the overlay stops leverage from lurching on a stale 40-day estimate. If a
  large part of what tranching fixes IS the overlay's staleness -- four sub-books each carry
  their own vol estimate, so averaging them smooths exactly the noise the overlay injects --
  then applying both should deliver much less than the sum, and the honest conclusion would be
  "do the cheap one (config change), skip the expensive one (new engine code)".

  That is the specific way this could disappoint, and it is worth knowing BEFORE anyone builds
  four-book netting into ibkr_engine.

ARMS (all live_sizing, REAL financing, 12 starts, both horizons)
  A  deployed
  B  tranching only          K=4, overlay ON
  C  no-overlay only         K=1, constant 1.10x
  D  BOTH                    K=4, constant 1.10x
  E  BOTH at 1.00x           K=4, constant 1.00x   (max drawdown protection)

Stacking is measured as  (D - A) - [(B - A) + (C - A)].  Zero => perfectly additive.
Negative => they fix overlapping problems.

Run:  python3 research/EXP019_stack_candidates.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, event_concentration  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
TOTAL = 50_000.0
P4 = [0, 5, 10, 15]

GATE = ({"credit_pct": 0.95, "credit_derisk": 0.5}
        if os.environ.get("STACK_GATE", "1") == "1" else {})   # BUGS A9: honest baseline

ARMS = [
    ("A deployed",        [0],  {}),
    ("B tranche K=4",     P4,   {}),
    ("C const1.10 K=1",   [0],  {"leverage": 1.10, "vol_scaling": False}),
    ("D BOTH 1.10",       P4,   {"leverage": 1.10, "vol_scaling": False}),
    ("E BOTH 1.00",       P4,   {"leverage": 1.00, "vol_scaling": False}),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*100}\nEXP-019 DO THE CANDIDATES STACK — {HZ}, {len(STARTS)} starts, "
          f"REAL financing, credit gate "
          f"{'ON (honest, BUGS A9)' if GATE else 'OFF'}\n{'='*100}", flush=True)

    res, curves = {}, {}
    for name, phases, extra in ARMS:
        rows, cvs = [], []
        ck = TOTAL / len(phases)
        for st in STARTS:
            per = [bt.run(st, END, {**DEPLOYED, **GATE, **extra, "initial_capital": ck,
                                    "rebal_phase": p, "live_sizing": True,
                                    "financing_curve": True})["daily_values"]
                   for p in phases]
            c = sum(x.reindex(per[0].index).ffill() for x in per)
            rows.append(stat(c)); cvs.append(c)
        res[name] = rows; curves[name] = cvs
        cc = np.array([r["cagr"] for r in rows])
        print(f"  {name:<18} CAGR {cc.mean():+6.2%}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'arm':<18}{'CAGR':>9}{'sdCAGR':>8}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}"
          f"{'worstDD':>9}", flush=True)
    for name, _p, _e in ARMS:
        r = res[name]
        c = np.array([x["cagr"] for x in r]); s = np.array([x["sharpe"] for x in r])
        so = np.array([x["sortino"] for x in r]); d = np.array([x["dd"] for x in r])
        print(f"  {name:<18}{c.mean():>+9.2%}{c.std(ddof=1):>8.2%}{s.mean():>9.3f}"
              f"{so.mean():>9.3f}{d.mean():>9.1%}{d.min():>9.1%}", flush=True)

    A = res["A deployed"]
    ac = np.array([x["cagr"] for x in A]); as_ = np.array([x["sharpe"] for x in A])
    ad = np.array([x["dd"] for x in A])
    print(f"\n  {'paired vs A':<18}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}"
          f"{'+Shrp':>7}{'+DD':>7}{'sdRatio':>9}{'top5%':>9}", flush=True)
    D = {}
    for name, _p, _e in ARMS[1:]:
        r = res[name]
        c = np.array([x["cagr"] for x in r]); s = np.array([x["sharpe"] for x in r])
        d = np.array([x["dd"] for x in r])
        cc = [event_concentration(curves["A deployed"][i], curves[name][i])
              for i in range(len(r))]
        ok = [x for x in cc if x["excess_meaningful"]]
        t5 = (f"{np.mean([x['top5_share'] for x in ok]):.0%}" if len(ok) >= len(cc) // 2
              else f"n/a({len(ok)}/{len(cc)})")
        D[name] = (c - ac, s - as_, d - ad)
        print(f"  {name:<18}{(c-ac).mean()*100:>+8.2f}p{(s-as_).mean():>+9.3f}"
              f"{(d-ad).mean()*100:>+8.2f}p{int((c>ac).sum()):>4}/{len(c)}"
              f"{int((s>as_).sum()):>4}/{len(s)}{int((d>ad).sum()):>4}/{len(d)}"
              f"{c.std(ddof=1)/ac.std(ddof=1):>9.3f}{t5:>9}", flush=True)

    print(f"\n  === STACKING: (D-A) - [(B-A)+(C-A)] ===", flush=True)
    for tgt in ("D BOTH 1.10", "E BOTH 1.00"):
        for j, lbl in ((0, "CAGR"), (1, "Sharpe"), (2, "MaxDD")):
            add = D["B tranche K=4"][j].mean() + D["C const1.10 K=1"][j].mean()
            got = D[tgt][j].mean()
            sc = 100 if lbl != "Sharpe" else 1
            print(f"  {tgt:<14} {lbl:<7} additive {add*sc:+7.3f}  actual {got*sc:+7.3f}  "
                  f"interaction {(got-add)*sc:+7.3f}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
