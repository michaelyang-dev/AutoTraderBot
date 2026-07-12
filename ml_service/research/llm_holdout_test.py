"""
LLM HOLDOUT — is the LLM extractor ACTUALLY good? Same yardstick as XBRL (99% bar).

For every symbol the LLM backfill passed, run the true holdout: validate the LLM's
reading on the SECOND-last Compustat quarter (anchor), then extract the LAST Compustat
quarter (target, truth KNOWN) and score it. Produces "LLM next-quarter accuracy" per
feature, directly comparable to the XBRL calibration (roe 99.2% / gm 92% / d2e 93%).

Decision rule: an LLM-sourced feature earns live eligibility ONLY at >=99% measured
accuracy — identical bar to everything else.
Run (AWS): venv/bin/python research/llm_holdout_test.py   (~2h, ~$10 of haiku calls)
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
from edgar_llm_extract import (  # noqa: E402
    PROMPT, candidates_from_llm, filing_statements, llm)

MODEL = "haiku"


def main():
    ov_f = DATA / "edgar_llm_overlay.json"
    if not ov_f.exists():
        print("no LLM overlay — run the backfill first")
        return
    llm_syms = sorted(json.load(open(ov_f)).get("features", {}).keys())
    print(f"holdout over {len(llm_syms)} LLM-passing symbols", flush=True)

    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    hist = {t: g.sort_values("datadate").tail(6) for t, g in fund.groupby("tic")}
    cik_map = load_cik_map()

    item_score = {k: [0, 0] for k in CONCEPTS}          # [correct, wrong]
    feat_score = {f: [0, 0] for f in FEATURES}
    stats = {"symbols": 0, "anchor_pass": 0, "anchor_fail": 0, "no_docs": 0}
    t0 = time.time()
    for si, sym in enumerate(llm_syms):
        gh = hist.get(sym)
        if gh is None or len(gh) < 3:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            continue
        facts = fetch_facts(cik)
        if facts is None:
            continue
        rows = list(gh.itertuples())
        target, anchor = rows[-1], rows[-2]
        stats["symbols"] += 1

        # XBRL-resolve at the anchor; the LLM only covers what XBRL cannot prove there
        resolved = {}
        for item, cands in CONCEPTS.items():
            hit = None
            for spec in cands:
                v, _ = spec_value(facts, spec, anchor.datadate, item in FLOWS)
                if close_enough(v, getattr(anchor, item)):
                    hit = spec
                    break
            if hit is None and item in ZERO_OK and (
                    pd.isna(getattr(anchor, item)) or abs(getattr(anchor, item)) <= ABS_TOL):
                hit = "ZERO"
            resolved[item] = hit
        missing = [it for it, h in resolved.items() if h is None]
        if not missing:
            continue

        # LLM anchor validation
        atext, _ = filing_statements(cik, anchor.datadate)
        if not atext:
            stats["no_docs"] += 1
            continue
        try:
            avals = candidates_from_llm(llm(MODEL, PROMPT.format(end=anchor.datadate.date(), text=atext)))
        except Exception:
            continue
        passed = [it for it in missing
                  if any(close_enough(cv, getattr(anchor, it)) for cv in avals.get(it, []))]
        if not passed:
            stats["anchor_fail"] += 1
            continue
        stats["anchor_pass"] += 1

        # extract the TARGET quarter (truth known) and score
        ttext, _ = filing_statements(cik, target.datadate)
        if not ttext:
            stats["no_docs"] += 1
            continue
        try:
            tvals = candidates_from_llm(llm(MODEL, PROMPT.format(end=target.datadate.date(), text=ttext)))
        except Exception:
            continue
        got = {}
        for it in CONCEPTS:
            if it in passed and tvals.get(it):
                v = tvals[it][0]
                got[it] = v
                truth = getattr(target, it)
                if close_enough(v, truth):
                    item_score[it][0] += 1
                else:
                    item_score[it][1] += 1
            elif resolved.get(it) == "ZERO":
                got[it] = 0.0
            elif resolved.get(it) not in (None, "ZERO"):
                v, _ = spec_value(facts, resolved[it], target.datadate, it in FLOWS)
                if v is not None:
                    got[it] = v
        for feat, (inputs, formula) in FEATURES.items():
            if not any(it in passed for it in inputs):
                continue                      # only score features the LLM contributed to
            if not all(it in got for it in inputs):
                continue
            fv = formula(got)
            tvals_truth = {it: (getattr(target, it) if pd.notna(getattr(target, it)) else 0.0)
                           for it in inputs}
            tv = formula(tvals_truth)
            if fv is None or tv is None or not np.isfinite(fv) or not np.isfinite(tv):
                continue
            if abs(fv - tv) <= max(0.02, 0.05 * abs(tv)):
                feat_score[feat][0] += 1
            else:
                feat_score[feat][1] += 1

        if si % 20 == 0:
            print(f"  {si}/{len(llm_syms)} ({time.time()-t0:.0f}s) "
                  f"anchor_pass={stats['anchor_pass']}", flush=True)

    print(f"\n===== LLM HOLDOUT ({MODEL}) {time.time()-t0:.0f}s =====")
    print(f"  symbols tried: {stats['symbols']} | anchor pass: {stats['anchor_pass']} "
          f"| anchor fail: {stats['anchor_fail']} | no docs: {stats['no_docs']}")
    print(f"  {'':<18}{'n':>6}{'next-q accuracy':>18}")
    for it, (ok, bad) in item_score.items():
        if ok + bad:
            print(f"  item {it:<13}{ok+bad:>6}{ok/(ok+bad):>17.1%}")
    for f, (ok, bad) in feat_score.items():
        if ok + bad:
            print(f"  feat {f:<13}{ok+bad:>6}{ok/(ok+bad):>17.1%}")
    print("\nbar: >=99% for live eligibility (same as XBRL).")


if __name__ == "__main__":
    main()
