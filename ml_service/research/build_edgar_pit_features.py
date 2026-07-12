"""
Build the FULL EDGAR-PIT feature dataset (roe + gross_margin + debt_to_equity) for the
"can 92-96%-accurate fresh gm/d2e beat stale" A/B — the untested assumption behind the
99% bar. Same methodology as build_edgar_pit_roe.py, generalized:

- per symbol, resolve production specs per ITEM at the current anchor (pit=False =
  production gate semantics, honoring GATE_DEPTH: niq/seqq/saleq depth-1, cogsq/dlttq/
  dlcq depth-2 two-consecutive-quarter reproduction)
- extract EVERY quarter as-first-reported (pit=True, earliest-filed) with original
  filed dates -> honest availability
- values carry their REAL extraction errors — that's the point of the test
- loader plausibility bounds mirrored: gm in [-1,1], |roe|<=25, |d2e|<=25

d2e convention (production _fill_fundamentals): (dlttq + dlcq) / seqq, dlcq/dlttq
NaN->0 — here both specs must RESOLVE at anchor (gate), but a missing historical
quarter for one debt item falls back to 0 only if the anchor Compustat value is also
0/NaN (else skip the quarter — can't distinguish "zero debt" from "tag not yet used").

Run on AWS (cache-hot): venv/bin/python research/build_edgar_pit_features.py
Output: data/edgar_pit_features.parquet (symbol, feature, period_end, filed, value)
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "scripts"))
from edgar_fundamentals_patch import (  # noqa: E402
    CONCEPTS, DATA, FUND, GATE_DEPTH, close_enough, fetch_facts, load_cik_map,
    spec_value)

FEATURES = {
    "roe": {"items": [("niq", True), ("seqq", False)],
            "formula": lambda v: 4 * v["niq"] / v["seqq"] if v["seqq"] else None,
            "bound": 25},
    "gross_margin": {"items": [("saleq", True), ("cogsq", True)],
                     "formula": lambda v: (v["saleq"] - v["cogsq"]) / v["saleq"] if v["saleq"] else None,
                     "bound": 1},
    "debt_to_equity": {"items": [("dlttq", False), ("dlcq", False), ("seqq", False)],
                       "formula": lambda v: (v["dlttq"] + v["dlcq"]) / v["seqq"] if v["seqq"] else None,
                       "bound": 25},
}


def resolve_spec(facts, item, is_flow, anchor_rows):
    """Production gate: spec must reproduce Compustat at GATE_DEPTH consecutive anchors."""
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

    rows = []
    covered = {f: 0 for f in FEATURES}
    t0 = time.time()
    for i, sym in enumerate(syms):
        if i % 200 == 0:
            print(f"  {i}/{len(syms)} ({time.time()-t0:.0f}s) covered={covered}", flush=True)
        g = by_tic.get(sym)
        if g is None or len(g) < 3:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            continue
        facts = fetch_facts(cik)
        if facts is None:
            continue
        anchor_rows = [g.iloc[-1], g.iloc[-2]]   # newest first, for depth-2 gates

        # resolve each ITEM once (shared across features, e.g. seqq)
        item_specs = {}
        for feat, cfg in FEATURES.items():
            for item, is_flow in cfg["items"]:
                if item not in item_specs:
                    item_specs[item] = resolve_spec(facts, item, is_flow, anchor_rows)

        for feat, cfg in FEATURES.items():
            if any(item_specs[item] is None for item, _ in cfg["items"]):
                continue   # feature not gate-covered for this symbol
            covered[feat] += 1
            for _, r in g.iterrows():
                vals, fileds, bad = {}, [], False
                for item, is_flow in cfg["items"]:
                    v, f = spec_value(facts, item_specs[item], r["datadate"], is_flow, pit=True)
                    if v is None:
                        if item in ("dlcq", "dlttq") and (pd.isna(r.get(item)) or r.get(item) == 0):
                            v = 0.0   # genuinely no debt of this class that quarter
                        else:
                            bad = True
                            break
                    else:
                        fileds.append(f)
                    vals[item] = v
                if bad or not fileds:
                    continue
                out = cfg["formula"](vals)
                if out is None or abs(out) > cfg["bound"]:
                    continue
                rows.append({"symbol": sym, "feature": feat, "period_end": r["datadate"],
                             "filed": pd.Timestamp(max(fileds)), "value": out})

    df = pd.DataFrame(rows)
    out = DATA / "edgar_pit_features.parquet"
    df.to_parquet(out)
    lag = (df.filed - df.period_end).dt.days
    print(f"\ncoverage: {covered} of {len(syms)}")
    print(df.groupby("feature").size().to_string())
    print(f"filed-lag days: median {lag.median():.0f} p10 {lag.quantile(.1):.0f} "
          f"p90 {lag.quantile(.9):.0f}  (sane ~30-45d)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
