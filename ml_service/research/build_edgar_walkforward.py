"""
WALK-FORWARD EDGAR PIT features — the fair-to-gm/d2e rebuild.

build_edgar_pit_features.py resolved each symbol's tag-spec ONCE at the 2026 anchor
and applied it to 2016-2025. The coverage diagnostic proved 20% of gm specs that
reproduce Compustat TODAY do NOT reproduce it 8 quarters back (real tagging drift) —
so that anchor-once dataset injected ~20% wrong historical values, which is what made
the B++ A/B look like gm/d2e "hurts drawdowns."

This rebuild re-resolves the spec PER YEAR against the latest-known Compustat quarter
of that year (walk-forward — exactly how the live overlay operates: it re-certifies
against each fresh WRDS upload). Values stay strictly PIT (earliest-filed date). If
walk-forward gm/d2e stops hurting the A/B, the ~2.4pp gm/d2e prize is real and
harvestable; if it still hurts, gm/d2e is genuinely unusable and we stop.

Also applies the recovered ROE specs (non-null-truth anchor) so the roe column here
matches the improved coverage.

Run on AWS (cache-hot): venv/bin/python research/build_edgar_walkforward.py
Output: data/edgar_walkforward_features.parquet (symbol, feature, period_end, filed, value)
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
    CONCEPTS, DATA, FEATURES, FUND, GATE_DEPTH, close_enough, fetch_facts,
    load_cik_map, spec_value)


def non_null_anchor(rows, item, depth):
    """Most-recent `depth` rows with non-null truth for `item` (debt: NaN==0 counts)."""
    out = []
    for r in reversed(rows):
        t = r.get(item)
        if pd.notna(t) or item in ("dlttq", "dlcq"):
            out.append(r)
        if len(out) >= depth:
            break
    return out


def resolve_spec(facts, item, is_flow, anchor_rows):
    depth = GATE_DEPTH.get(item, 1)
    rows = non_null_anchor(anchor_rows, item, depth)
    if len(rows) < depth:
        return None
    for spec in CONCEPTS[item]:
        ok = True
        for r in rows:
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
    by_tic = {t: g.sort_values("datadate").reset_index(drop=True) for t, g in fund.groupby("tic")}
    cik_map = load_cik_map()

    rows_out = []
    cov = {f: 0 for f in FEATURES}
    t0 = time.time()
    for i, sym in enumerate(syms):
        if i % 200 == 0:
            print(f"  {i}/{len(syms)} ({time.time()-t0:.0f}s) cov={cov}", flush=True)
        g = by_tic.get(sym)
        if g is None or len(g) < 4:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            continue
        facts = fetch_facts(cik)
        if facts is None:
            continue
        rows_list = [r for _, r in g.iterrows()]
        covered_feat = set()
        # WALK-FORWARD but resolved PER YEAR (specs are stable within a year; drift is
        # a multi-year phenomenon). For each quarter, use the spec resolved from the
        # most recent prior quarter of a DIFFERENT year -> ~10 resolutions not ~40.
        spec_cache = {}   # (item, year) -> spec
        for qi, r in enumerate(rows_list):
            if qi < 4:
                continue
            prior = rows_list[:qi]
            yr = r["datadate"].year
            for feat, (items, formula) in FEATURES.items():
                vals, fileds, bad = {}, [], False
                for item in items:
                    is_flow = item in ("niq", "saleq", "cogsq")
                    key = (item, yr)
                    if key not in spec_cache:
                        spec_cache[key] = resolve_spec(facts, item, is_flow, prior)
                    spec = spec_cache[key]
                    if spec is None:
                        bad = True
                        break
                    v, f = spec_value(facts, spec, r["datadate"], is_flow, pit=True)
                    if v is None:
                        if item in ("dlttq", "dlcq") and (pd.isna(r.get(item)) or r.get(item) == 0):
                            v = 0.0
                        else:
                            bad = True
                            break
                    else:
                        fileds.append(f)
                    vals[item] = v
                if bad or not fileds:
                    continue
                out = formula(vals)
                if out is None:
                    continue
                bound = 25 if feat != "gross_margin" else 1
                if abs(out) > bound:
                    continue
                covered_feat.add(feat)
                rows_out.append({"symbol": sym, "feature": feat, "period_end": r["datadate"],
                                 "filed": pd.Timestamp(max(fileds)), "value": out})
        for f in covered_feat:
            cov[f] += 1

    df = pd.DataFrame(rows_out)
    out = DATA / "edgar_walkforward_features.parquet"
    df.to_parquet(out)
    lag = (df.filed - df.period_end).dt.days
    print(f"\nwalk-forward coverage (>=1 quarter): {cov} of {len(syms)}")
    print(df.groupby("feature").size().to_string())
    print(f"filed-lag median {lag.median():.0f}d -> {out}")


if __name__ == "__main__":
    main()
