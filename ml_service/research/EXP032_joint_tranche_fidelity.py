"""EXP-032 — IMPLEMENTATION-FIDELITY CHECK before writing any live code for Option 2.

THE PROBLEM I FOUND WHILE DESIGNING THE LIVE ENGINE
  Every tranching number in this program (EXP-001, -014, -019, -027, -031) was produced by
  running K sub-books as SEPARATE backtests, each with capital/K, and SUMMING their equity
  curves. That means the sub-books never share capital: a tranche that does well stays big, one
  that lags stays small, and the mix drifts for 26 years.

  The LIVE engine cannot work that way. All four sub-books live in ONE margin account, and when
  a tranche rebalances it necessarily sizes off NAV/4 of the CURRENT TOTAL NAV. That is
  continuous reallocation between tranches -- the opposite of what I backtested.

  Cross-phase CAGR sigma on the 26yr is 1.94pp. Compounded over 26 years that is roughly a
  1.6x spread between the best and worst tranche, so the drift is NOT negligible and the two
  designs are genuinely different products. Shipping the live version while quoting the summed
  version's numbers would be exactly the backtest-vs-live divergence class in BUGS section B.

WHAT THIS MEASURES
  A  SUMMED-INDEPENDENT   K runs at capital/K, curves added        <- what every prior number is
  B  JOINT-REALLOCATED    one account, one shared NAV, tranche p rebalances every K-th cycle
                          and sizes off NAV/K of the CURRENT total  <- what the live engine does

  If B ~= A, the live design is faithful and the published numbers stand.
  If B differs materially, the published numbers describe a product we cannot build, and either
  the live design must track per-tranche NAV or the numbers must be restated.

  Reallocation is not obviously worse -- it moves capital from lucky phases to unlucky ones, and
  phase is pure noise, so it may add a small rebalancing bonus. But that is a hypothesis; this
  measures it.

PARITY GATE
  With tranches=1 the joint simulator must reproduce the untouched single-book path BIT-FOR-BIT.
  Without that, any difference between A and B could be a reimplementation bug rather than the
  design difference being studied.

Run:  python3 research/EXP032_joint_tranche_fidelity.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from livemirror_tranche import JointTrancheBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
BASE = {**DEPLOYED, **GATE, "financing_curve": True, "live_sizing": True,
        "initial_capital": 50_000.0, "vol_scaling": False, "leverage": 1.00}
P4 = [0, 5, 10, 15]


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*96}\nEXP-032 JOINT-TRANCHE FIDELITY — {HZ}, {len(STARTS)} starts\n{'='*96}",
          flush=True)

    # ---- PARITY GATE 1: no regression (K=1 delegates to the validated parent) --------
    a = bt.run(STARTS[0], END, dict(BASE))
    b = bt.run(STARTS[0], END, {**BASE, "tranches": 1})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  GATE 1  tranches=1 vs untouched: {'IDENTICAL' if same else 'DIFFERENT'}  "
          f"({a['daily_values'].iloc[-1]:,.2f})", flush=True)
    if not same:
        print("  ABORT — K=1 must delegate to the parent unchanged."); return

    # ---- PARITY GATE 2: the one that actually validates the TRANCHE MACHINERY ---------
    # K tranches all on the SAME phase is just one book split K ways, so it must approximately
    # reproduce the single-book run. Gate 1 cannot test this -- it never enters the new code.
    # Exact equality is impossible: K books of NAV/K each floor their share counts separately,
    # so rounding drag is larger. A few percent is expected; a large gap means a real bug.
    c = bt.run(STARTS[0], END, {**BASE, "tranches": 1, "force_joint": True,
                                "tranche_stride": DEPLOYED["rebal_days"]})
    ref, got = a["daily_values"].iloc[-1], c["daily_values"].iloc[-1]
    err = abs(got / ref - 1)
    print(f"  GATE 2  joint loop as ONE book (K=1, stride=20) vs parent: {got:,.2f} vs "
          f"{ref:,.2f}  ({err:+.2%}) {'OK' if err < 0.03 else 'FAIL — reimplementation bug'}",
          flush=True)
    if err >= 0.03:
        print("  ABORT — the tranche loop does not reproduce a split single book.", flush=True)
        return

    rows = {"A summed-independent": [], "B joint-reallocated": []}
    for si, st in enumerate(STARTS):
        # A: separate runs, capital/K each, curves summed (the historical method)
        per = [bt.run(st, END, {**BASE, "initial_capital": 50_000.0 / len(P4),
                                "rebal_phase": p})["daily_values"] for p in P4]
        rows["A summed-independent"].append(
            stat(sum(x.reindex(per[0].index).ffill() for x in per)))
        # B: one account, shared NAV, tranche p sized off NAV/K of the CURRENT total
        m = bt.run(st, END, {**BASE, "tranches": len(P4)})
        rows["B joint-reallocated"].append(stat(m["daily_values"]))
        print(f"  [{si+1:2d}/{len(STARTS)}] {st}  A {rows['A summed-independent'][-1]['cagr']:+7.2%}"
              f"   B {rows['B joint-reallocated'][-1]['cagr']:+7.2%}   ({time.time()-t0:.0f}s)",
              flush=True)

    print(f"\n  {'arm':<24}{'meanCAGR':>10}{'sdCAGR':>9}{'Sharpe':>9}{'Sortino':>9}"
          f"{'MaxDD':>9}{'worstDD':>9}", flush=True)
    for k in ("A summed-independent", "B joint-reallocated"):
        r = rows[k]
        c = np.array([x["cagr"] for x in r]); s = np.array([x["sharpe"] for x in r])
        print(f"  {k:<24}{c.mean():>+10.2%}{c.std(ddof=1):>9.2%}{s.mean():>9.3f}"
              f"{np.mean([x['sortino'] for x in r]):>9.3f}"
              f"{np.mean([x['dd'] for x in r]):>9.1%}{np.min([x['dd'] for x in r]):>9.1%}",
              flush=True)
    A, B = rows["A summed-independent"], rows["B joint-reallocated"]
    dc = np.array([B[i]["cagr"] - A[i]["cagr"] for i in range(len(A))])
    ds = np.array([B[i]["sharpe"] - A[i]["sharpe"] for i in range(len(A))])
    dd = np.array([B[i]["dd"] - A[i]["dd"] for i in range(len(A))])
    nc, ns, v = sign_verdict(dc, ds, len(A))
    print(f"\n  B - A:  dCAGR {dc.mean()*100:+.2f}pp   dSharpe {ds.mean():+.4f}   "
          f"dMaxDD {dd.mean()*100:+.2f}pp   ({nc}/{len(dc)} CAGR, {ns}/{len(ds)} Sharpe)",
          flush=True)
    print(f"\n  |dSharpe| < 0.010 => the live design is FAITHFUL, published numbers stand.",
          flush=True)
    print(f"  Larger => the published tranching numbers describe a product we cannot build.",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
