"""
Build the EDGAR-PIT ROE dataset for the true assistant A/B backtest.

For every SP1500 symbol whose roe inputs (niq, seqq) pass the production gate, extract
niq/seqq for EVERY quarter in the Compustat window (2016+) from cached EDGAR facts,
stamped with the REAL SEC `filed` date -> a point-in-time series: (symbol, period_end,
filed_date, roe). The A/B backtest serves arm-B roe as "latest value whose filed date
<= signal date" — exactly what the live assistant does.

Honest simplification (disclosed): the tag-spec per symbol is resolved ONCE at the
current anchor (the production method) and applied historically. Method-selection thus
uses today's knowledge; extraction VALUES remain strictly PIT via filed dates. The
calibration showed specs are stable (98.9% across 4 quarters), so the bias is small.

Run on AWS (cache-hot, no network): venv/bin/python research/build_edgar_pit_roe.py
Output: data/edgar_pit_roe.parquet
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
    CONCEPTS, DATA, FLOWS, FUND, close_enough, fetch_facts, load_cik_map,
    series_for_concept, spec_value, value_at)


def main():
    members = json.load(open(DATA / "sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    by_tic = {t: g.sort_values("datadate") for t, g in fund.groupby("tic")}
    cik_map = load_cik_map()

    rows = []
    covered = 0
    t0 = time.time()
    for i, sym in enumerate(syms):
        if i % 200 == 0:
            print(f"  {i}/{len(syms)} ({time.time()-t0:.0f}s) covered={covered}", flush=True)
        g = by_tic.get(sym)
        if g is None or len(g) < 2:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            continue
        facts = fetch_facts(cik)
        if facts is None:
            continue
        anchor = g.iloc[-1]
        # resolve production specs (gate depth 1, same as deployed)
        spec_ni = spec_seq = None
        for spec in CONCEPTS["niq"]:
            v, _ = spec_value(facts, spec, anchor["datadate"], True)
            if close_enough(v, anchor["niq"]):
                spec_ni = spec
                break
        for spec in CONCEPTS["seqq"]:
            v, _ = spec_value(facts, spec, anchor["datadate"], False)
            if close_enough(v, anchor["seqq"]):
                spec_seq = spec
                break
        if spec_ni is None or spec_seq is None:
            continue
        covered += 1
        # extract every Compustat quarter with real filed dates
        for _, r in g.iterrows():
            ni, f1 = spec_value(facts, spec_ni, r["datadate"], True)
            se, f2 = spec_value(facts, spec_seq, r["datadate"], False)
            if ni is None or se is None or not se:
                continue
            filed = max(f1 or "1900-01-01", f2 or "1900-01-01")
            rows.append({"symbol": sym, "period_end": r["datadate"],
                         "filed": pd.Timestamp(filed), "roe": 4 * ni / se})
    df = pd.DataFrame(rows)
    out = DATA / "edgar_pit_roe.parquet"
    df.to_parquet(out)
    print(f"\ncovered symbols: {covered} | rows: {len(df)} | "
          f"period range {df.period_end.min().date()} -> {df.period_end.max().date()}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
