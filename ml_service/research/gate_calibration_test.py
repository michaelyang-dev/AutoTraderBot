"""
GATE CALIBRATION — make strictness PROVABLE instead of guessed.

Simulates the exact live situation on history: anchor the gate on PAST quarter(s),
resolve each item's extraction method, extract the NEXT quarter, then score it against
Compustat's KNOWN value for that quarter (a true holdout).

MULTI-QUARTER HOLDOUT (2026-07-12): the last FOUR quarters per symbol are scored as
successive targets, each anchored only on its own past — ~4x the scoring events,
accuracy measured across different reporting regimes, not one lucky quarter.

For gate strictness g in {1, 2, 3} anchor quarters, per item and per feature:
    coverage       = share of (symbol, quarter) events where a method passes the gate
    next-q accuracy = of those, share whose NEXT-quarter extraction matches Compustat
    jump-caught    = of the WRONG extractions, share the emission jump-guard stops

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
N_TARGETS = 4   # holdout quarters per symbol


def score_target(facts, rows, res, fres):
    """rows = history up to and including the target quarter (last element)."""
    target = rows[-1]
    anchors = rows[:-1][::-1]          # most recent first: T-1, T-2, T-3
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
            if abs(fv - tv) <= max(0.02, 0.05 * abs(tv)):
                fres[feat][g][1] += 1
            else:
                fres[feat][g][2] += 1
                old = formula({it: (getattr(anchors[0], it)
                                    if pd.notna(getattr(anchors[0], it)) else 0.0)
                               for it in inputs})
                if (old is not None and np.isfinite(old)
                        and abs(fv) > 5 * max(abs(old), 0.01) and abs(fv) > 0.5):
                    fres[feat][g][3] += 1


def main():
    members = json.load(open(DATA / "sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    hist = {t: g.sort_values("datadate").tail(4 + N_TARGETS) for t, g in fund.groupby("tic")}
    cik_map = load_cik_map()

    res = {it: {g: [0, 0, 0, 0] for g in GATES} for it in CONCEPTS}
    fres = {f: {g: [0, 0, 0, 0] for g in GATES} for f in FEATURES}
    t0 = time.time()
    n_events = 0
    for i, sym in enumerate(syms):
        if i % 200 == 0:
            print(f"  {i}/{len(syms)} ({time.time()-t0:.0f}s, events={n_events})", flush=True)
        gh = hist.get(sym)
        if gh is None or len(gh) < 5:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            continue
        facts = fetch_facts(cik)          # cache hit — no network
        if facts is None:
            continue
        allrows = list(gh.itertuples())
        first_target = max(4, len(allrows) - N_TARGETS)
        for tgt_i in range(first_target, len(allrows)):
            score_target(facts, allrows[:tgt_i + 1], res, fres)
            n_events += 1

    print(f"\n===== GATE CALIBRATION ({time.time()-t0:.0f}s, {n_events} (symbol,quarter) events) =====")
    hdr = f"{'':<18}{'gate':>5}{'coverage':>10}{'next-q acc':>12}{'wrong':>7}{'jump-caught':>12}"
    print(hdr)
    for table in (res, fres):
        for key in table:
            for g in GATES:
                c, ok, bad, jc = table[key][g]
                if c == 0:
                    continue
                scored = ok + bad
                acc = ok / scored if scored else float("nan")
                print(f"{key:<18}{g:>5}{c/n_events:>10.1%}{acc:>12.1%}{bad:>7}"
                      f"{(jc/bad if bad else 0):>12.0%}")
        print("─" * len(hdr))
    print("read: coverage = gate pass-rate over events; next-q acc = of passed methods,")
    print("how often the NEXT quarter's extraction matches Compustat truth.")


if __name__ == "__main__":
    main()
