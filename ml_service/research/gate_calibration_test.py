"""
GATE CALIBRATION — make strictness PROVABLE instead of guessed.

Simulates the exact live situation on history: anchor the gate on PAST quarter(s),
resolve each item's extraction method, extract the NEXT quarter, then score it against
Compustat's KNOWN value for that quarter (it's in the file — this is a true holdout).

For gate strictness g in {1, 2, 3} anchor quarters, per item and per feature:
    coverage      = share of symbols where a method passes the g-quarter gate
    next-q accuracy = of those, share whose NEXT-quarter extraction matches Compustat
                      (close_enough, same tolerance as production)
    jump-caught   = of the WRONG extractions, share the emission jump-guard would stop

Decision rule this enables: pick, per feature, the loosest gate whose measured
next-quarter accuracy clears a chosen bar (e.g. 99%). No vibes.

Uses the EDGAR facts cache (no network). Run on AWS:
    venv/bin/python research/gate_calibration_test.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "scripts"))
from edgar_fundamentals_patch import (  # noqa: E402
    ABS_TOL, CONCEPTS, DATA, FEATURES, FLOWS, FUND, ZERO_OK,
    close_enough, fetch_facts, load_cik_map, spec_value)

GATES = [1, 2, 3]


def main():
    members = json.load(open(DATA / "sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    hist = {t: g.sort_values("datadate").tail(5) for t, g in fund.groupby("tic")}
    cik_map = load_cik_map()

    # results[item][gate] = [coverage_hits, correct, wrong, jump_caught]
    res = {it: {g: [0, 0, 0, 0] for g in GATES} for it in CONCEPTS}
    fres = {f: {g: [0, 0, 0, 0] for g in GATES} for f in FEATURES}
    t0 = time.time()
    n_done = 0
    for i, sym in enumerate(syms):
        if i % 200 == 0:
            print(f"  {i}/{len(syms)} ({time.time()-t0:.0f}s)", flush=True)
        g5 = hist.get(sym)
        if g5 is None or len(g5) < 4:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            continue
        facts = fetch_facts(cik)          # cache hit — no network
        if facts is None:
            continue
        n_done += 1
        rows = list(g5.itertuples())
        target = rows[-1]                  # the "future" quarter with KNOWN truth
        anchors = rows[:-1][::-1]          # most recent first: T-1, T-2, T-3

        resolved_by_gate = {}
        for g in GATES:
            if len(anchors) < g:
                continue
            resolved = {}
            for item, candidates in CONCEPTS.items():
                hit = None
                for spec in candidates:
                    ok = True
                    for a in anchors[:g]:
                        v, _ = spec_value(facts, spec, a.datadate, item in FLOWS)
                        if not close_enough(v, getattr(a, item)):
                            ok = False
                            break
                    if ok:
                        hit = spec
                        break
                if hit is None and item in ZERO_OK:
                    a0 = anchors[0]
                    if pd.isna(getattr(a0, item)) or abs(getattr(a0, item)) <= ABS_TOL:
                        hit = "ZERO"
                resolved[item] = hit
            resolved_by_gate[g] = resolved

            # score items at the target quarter
            extracted = {}
            for item, spec in resolved.items():
                if spec is None:
                    continue
                res[item][g][0] += 1
                truth = getattr(target, item)
                if spec == "ZERO":
                    v = 0.0
                else:
                    v, _ = spec_value(facts, spec, target.datadate, item in FLOWS)
                if v is None:
                    continue                      # fails closed live — not an error
                extracted[item] = v
                if close_enough(v, truth):
                    res[item][g][1] += 1
                else:
                    res[item][g][2] += 1
                    prev = getattr(anchors[0], item)
                    if (pd.notna(prev) and abs(v) > 5 * max(abs(prev), 0.01)
                            and abs(v) > 0.5):
                        res[item][g][3] += 1

            # score features
            for feat, (inputs, formula) in FEATURES.items():
                if any(resolved.get(it) is None for it in inputs):
                    continue
                fres[feat][g][0] += 1
                vals, truthvals, complete = {}, {}, True
                for it in inputs:
                    if resolved[it] == "ZERO":
                        vals[it] = 0.0
                    elif it in extracted:
                        vals[it] = extracted[it]
                    else:
                        complete = False
                        break
                    tv = getattr(target, it)
                    truthvals[it] = 0.0 if pd.isna(tv) else tv
                if not complete:
                    continue
                fv = formula(vals)
                tv = formula(truthvals)
                if fv is None or tv is None or not np.isfinite(fv) or not np.isfinite(tv):
                    continue
                if abs(fv - tv) <= max(0.02, 0.05 * abs(tv)):     # feature-level tolerance
                    fres[feat][g][1] += 1
                else:
                    fres[feat][g][2] += 1
                    old = formula({it: (getattr(anchors[0], it) if pd.notna(getattr(anchors[0], it)) else 0.0)
                                   for it in inputs})
                    if (old is not None and np.isfinite(old)
                            and abs(fv) > 5 * max(abs(old), 0.01) and abs(fv) > 0.5):
                        fres[feat][g][3] += 1

    print(f"\n===== GATE CALIBRATION ({time.time()-t0:.0f}s, {n_done} symbols scored) =====")
    hdr = f"{'':<18}{'gate':>5}{'coverage':>10}{'next-q acc':>12}{'wrong':>7}{'jump-caught':>12}"
    print(hdr)
    for item in CONCEPTS:
        for g in GATES:
            c, ok, bad, jc = res[item][g]
            if c == 0:
                continue
            scored = ok + bad
            acc = ok / scored if scored else float("nan")
            print(f"{item:<18}{g:>5}{c/n_done:>10.1%}{acc:>12.1%}{bad:>7}{(jc/bad if bad else 0):>12.0%}")
    print("─" * len(hdr))
    for feat in FEATURES:
        for g in GATES:
            c, ok, bad, jc = fres[feat][g]
            if c == 0:
                continue
            scored = ok + bad
            acc = ok / scored if scored else float("nan")
            print(f"{feat:<18}{g:>5}{c/n_done:>10.1%}{acc:>12.1%}{bad:>7}{(jc/bad if bad else 0):>12.0%}")
    print("\nread: coverage = gate pass-rate; next-q acc = of passed methods, how often the")
    print("NEXT quarter's extraction matches Compustat truth; jump-caught = wrong ones the")
    print("emission guard would have withheld anyway.")


if __name__ == "__main__":
    main()
