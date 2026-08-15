"""EXP-010 — the leverage question, re-asked with a HONEST financing rate.

EXP-005 measured, over 26 years and 12 starts, that 1.49x leverage buys +1.89pp of CAGR and
costs -0.044 of Sharpe and -16.01pp of MaxDD, with Sharpe better at 1.00x in 12/12 starts.
That is the most sign-consistent result this program has produced -- and it was computed with a
FLAT 6.3%/yr financing charge applied across all of 2001-2025.

That assumption is wrong, and wrong in the direction that manufactures the result:
  measured (research/build_financing_curve.py, DFF + 1.5pp = IBKR Pro small-balance tier)
    2001-2025 actual mean 3.25%   -> flat rate OVERCHARGES by 3.05pp/yr on the debit
    2009-2015 actual mean 1.63%
    2018-2025 actual mean 3.86%   -> overcharges by 2.44pp/yr
The levered arm borrows ~0.40 of NAV more than the unlevered one, so ~3pp/yr of excess charge
is ~1.2pp/yr of CAGR handed to the unlevered arm for free. That is most of the measured gap.

This re-runs the sweep under BOTH assumptions so the financing effect is explicit rather than
buried, on BOTH horizons.

NOTE ON WHAT IS AND IS NOT A FINDING
  Leverage MUST lower Sharpe whenever financing > 0: Sharpe(L) = mu/sigma - (L-1)*rf/(L*sigma).
  Observing that is arithmetic, not research. The real question is whether the CAGR bought is
  worth the drawdown paid -- and whether the answer is the same on both horizons.

Run:  python3 research/EXP010_financing_and_leverage.py [8yr|26yr]
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
LEVS = [1.00, 1.25, 1.49, 1.75]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*96}\nEXP-010 LEVERAGE under FLAT vs REAL financing — {HZ}, "
          f"{len(STARTS)} starts\n{'='*96}", flush=True)

    a = bt.run(STARTS[0], END, dict(DEPLOYED))
    b = bt.run(STARTS[0], END, {**DEPLOYED, "financing_curve": False})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  PARITY financing_curve=False vs untouched: "
          f"{'IDENTICAL' if same else 'DIFFERENT'}", flush=True)
    if not same:
        print("  ABORT — hook changed the default path.")
        return

    res = {}
    for curve in (False, True):
        for L in LEVS:
            key = (curve, L)
            res[key] = []
            for st in STARTS:
                m = bt.run(st, END, {**DEPLOYED, "leverage": L,
                                     "financing_curve": curve})
                r = stat(m["daily_values"]); r["gross"] = m["avg_gross"]
                r["fin"] = getattr(bt, "_fin_paid", 0.0)
                res[key].append(r)
            c = np.array([x["cagr"] for x in res[key]])
            print(f"  [{'REAL ' if curve else 'FLAT '}{L:.2f}x] CAGR {c.mean():+6.2%}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    for curve in (False, True):
        tag = "REAL DFF+1.5pp" if curve else "FLAT 6.3%"
        print(f"\n  === financing: {tag} ===", flush=True)
        print(f"  {'arm':<14}{'meanCAGR':>10}{'sdCAGR':>8}{'Sharpe':>9}{'Sortino':>9}"
              f"{'meanDD':>9}{'worstDD':>9}{'avgGross':>10}", flush=True)
        for L in LEVS:
            k = (curve, L)
            c = np.array([x["cagr"] for x in res[k]]); s = np.array([x["sharpe"] for x in res[k]])
            so = np.array([x["sortino"] for x in res[k]]); d = np.array([x["dd"] for x in res[k]])
            print(f"  {('%.2fx' % L):<14}{c.mean():>+10.2%}{c.std(ddof=1):>8.2%}"
                  f"{s.mean():>9.3f}{so.mean():>9.3f}{d.mean():>9.1%}{d.min():>9.1%}"
                  f"{np.mean([x['gross'] for x in res[k]]):>10.3f}", flush=True)
        base = res[(curve, 1.00)]
        print(f"  {'--- paired deltas vs 1.00x ---':<40}", flush=True)
        for L in LEVS[1:]:
            v = res[(curve, L)]
            dc = np.array([v[i]["cagr"] - base[i]["cagr"] for i in range(len(STARTS))])
            ds = np.array([v[i]["sharpe"] - base[i]["sharpe"] for i in range(len(STARTS))])
            dso = np.array([v[i]["sortino"] - base[i]["sortino"] for i in range(len(STARTS))])
            dd = np.array([v[i]["dd"] - base[i]["dd"] for i in range(len(STARTS))])
            print(f"  {L:.2f}x: dCAGR {dc.mean()*100:+6.2f}pp  dSharpe {ds.mean():+6.3f} "
                  f"({int((ds>0).sum())}/{len(STARTS)})  dSortino {dso.mean():+6.3f}  "
                  f"dMaxDD {dd.mean()*100:+7.2f}pp", flush=True)

    print(f"\n  === what the financing assumption alone is worth (REAL - FLAT) ===", flush=True)
    for L in LEVS:
        f_ = res[(False, L)]; r_ = res[(True, L)]
        dc = np.array([r_[i]["cagr"] - f_[i]["cagr"] for i in range(len(STARTS))])
        ds = np.array([r_[i]["sharpe"] - f_[i]["sharpe"] for i in range(len(STARTS))])
        print(f"  {L:.2f}x: dCAGR {dc.mean()*100:+6.2f}pp  dSharpe {ds.mean():+6.3f}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
