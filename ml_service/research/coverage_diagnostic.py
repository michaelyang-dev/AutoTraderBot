"""
COVERAGE + DRIFT DIAGNOSTIC for the EDGAR overlay. Answers two questions before we
spend effort building fixes:

1. ROE coverage (~93%): WHY do the uncovered symbols fail? Categorize each as
   no_cik (foreign/ADR not in SEC ticker map), no_facts (CIK has no companyfacts),
   or spec_fail (facts exist but niq/seqq specs don't reproduce Compustat). For
   spec_fail, dump the us-gaap tags they DO expose that look like net-income /
   equity — the raw material for new CONCEPTS variants.

2. SPEC DRIFT (the B++ suspicion): for gate-passing gm/d2e symbols, is the
   anchor-resolved spec still accurate at OLDER quarters? Compare reproduction rate
   at the anchor vs 8 quarters back. If it degrades, the B++ backtest (anchor-once
   specs applied to 8yr history) was UNFAIR and a walk-forward test is warranted.

Run on AWS (cache-hot): venv/bin/python research/coverage_diagnostic.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "scripts"))
from edgar_fundamentals_patch import (  # noqa: E402
    CONCEPTS, DATA, FUND, GATE_DEPTH, close_enough, fetch_facts, load_cik_map,
    spec_value)


def resolve_spec(facts, item, is_flow, anchor_rows):
    depth = GATE_DEPTH.get(item, 1)
    for spec in CONCEPTS[item]:
        ok = True
        for r in anchor_rows[:depth]:
            v, _ = spec_value(facts, spec, r["datadate"], is_flow)
            truth = r.get(item)
            if item in ("dlcq", "dlttq") and pd.isna(truth):
                truth = 0.0
            if not close_enough(v, truth):
                ok = False
                break
        if ok:
            return spec
    return None


def main():
    members = json.load(open(DATA / "sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    by_tic = {t: g.sort_values("datadate") for t, g in fund.groupby("tic")}
    cik_map = load_cik_map()

    roe_fail = {"no_cik": [], "no_facts": [], "spec_fail_niq": [], "spec_fail_seqq": []}
    tag_hints = Counter()
    drift = {"anchor_ok": 0, "q8_ok": 0, "n": 0}   # gm reproduction now vs 8q ago

    for i, sym in enumerate(syms):
        if i % 250 == 0:
            print(f"  {i}/{len(syms)}", flush=True)
        g = by_tic.get(sym)
        if g is None or len(g) < 3:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            roe_fail["no_cik"].append(sym)
            continue
        facts = fetch_facts(cik)
        if facts is None:
            roe_fail["no_facts"].append(sym)
            continue
        anchor = [g.iloc[-1], g.iloc[-2]]
        spec_ni = resolve_spec(facts, "niq", True, anchor)
        spec_se = resolve_spec(facts, "seqq", False, anchor)
        if spec_ni is None:
            roe_fail["spec_fail_niq"].append(sym)
            gaap = facts.get("facts", {}).get("us-gaap", {})
            for k in gaap:
                if "NetIncome" in k or "ProfitLoss" in k:
                    tag_hints[k] += 1
        elif spec_se is None:
            roe_fail["spec_fail_seqq"].append(sym)

        # drift probe on gm (saleq/cogsq gate-passers)
        spec_sa = resolve_spec(facts, "saleq", True, anchor)
        spec_co = resolve_spec(facts, "cogsq", True, anchor)
        if spec_sa and spec_co and len(g) >= 10:
            drift["n"] += 1
            # anchor quarter reproduction (already known ok) vs 8 quarters back
            old = g.iloc[-9]
            sa_o, _ = spec_value(facts, spec_sa, old["datadate"], True)
            co_o, _ = spec_value(facts, spec_co, old["datadate"], True)
            if close_enough(sa_o, old.get("saleq")) and close_enough(co_o, old.get("cogsq")):
                drift["q8_ok"] += 1
            drift["anchor_ok"] += 1

    total = len(syms)
    covered = total - sum(len(v) for v in roe_fail.values())
    print("\n" + "=" * 64)
    print(f"ROE COVERAGE: {covered}/{total} = {covered/total:.1%}")
    for k, v in roe_fail.items():
        print(f"  {k:<16} {len(v):>4}   e.g. {', '.join(v[:8])}")
    print("\nTOP net-income-ish tags on spec_fail_niq symbols (new-variant candidates):")
    for tag, c in tag_hints.most_common(12):
        print(f"  {c:>3}  {tag}")
    if drift["n"]:
        print("\n" + "=" * 64)
        print(f"SPEC DRIFT (gm, {drift['n']} gate-passing symbols):")
        print(f"  anchor quarter reproduced: {drift['anchor_ok']}/{drift['n']} = "
              f"{drift['anchor_ok']/drift['n']:.1%}")
        print(f"  8-quarters-back reproduced: {drift['q8_ok']}/{drift['n']} = "
              f"{drift['q8_ok']/drift['n']:.1%}  <-- if << anchor, drift is real")


if __name__ == "__main__":
    main()
