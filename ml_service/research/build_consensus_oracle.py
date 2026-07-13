"""
ORACLE-CONSENSUS filter — the decisive gm/d2e test.

The walk-forward A/B proved gm/d2e extraction hurts (deep drawdowns) even with drift
fixed and coverage tripled — so per-value ACCURACY (~92-96%) is the binding constraint,
not drift/coverage. But DIAG (PERFECT gm/d2e) clearly helps (+1.8pp). The missing cell:
PARTIAL coverage at ~100% accuracy. This filter keeps a walk-forward gm/d2e value ONLY
where it matches Compustat's own value at that period (2%/$0.02 tol) — i.e. the subset
a perfect second-source consensus (XBRL n an independent source) would certify.

If the resulting B-CONS arm HELPS -> a real consensus gate (XBRL n FMP/LLM) is worth
building. If it still HURTS -> partial coverage itself is the problem and gm/d2e
freshness is genuinely the irreducible cost of WRDS's quarterly cadence. Either way,
decisive. This uses Compustat as the oracle second source (look-ahead, clearly an
UPPER BOUND on what real consensus could achieve).

Run local: python3 research/build_consensus_oracle.py
Output: data/edgar_consensus_features.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))

FUND = ML / "data/wrds/compustat_fundamentals_quarterly.parquet"
WF = ML / "data/edgar_walkforward_features.parquet"
OUT = ML / "data/edgar_consensus_features.parquet"


def cs_feature(row, feat):
    if feat == "roe":
        return 4 * row["niq"] / row["seqq"] if row.get("seqq") else None
    if feat == "gross_margin":
        return (row["saleq"] - row["cogsq"]) / row["saleq"] if row.get("saleq") else None
    if feat == "debt_to_equity":
        d = (0 if pd.isna(row.get("dlttq")) else row["dlttq"]) + \
            (0 if pd.isna(row.get("dlcq")) else row["dlcq"])
        return d / row["seqq"] if row.get("seqq") else None
    return None


def main():
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    truth = {}   # (tic, datadate) -> {feat: value}
    for _, r in fund.iterrows():
        d = {}
        for feat in ("roe", "gross_margin", "debt_to_equity"):
            v = cs_feature(r, feat)
            if v is not None and np.isfinite(v):
                d[feat] = v
        truth[(r["tic"], r["datadate"])] = d

    wf = pd.read_parquet(WF)
    keep = []
    for _, r in wf.iterrows():
        t = truth.get((r["symbol"], r["period_end"]))
        if not t or r["feature"] not in t:
            continue
        cs = t[r["feature"]]
        if abs(r["value"] - cs) <= max(0.02, 0.02 * abs(cs)):
            keep.append(r)
    out = pd.DataFrame(keep)
    out.to_parquet(OUT)
    tot = wf.groupby("feature").size()
    kept = out.groupby("feature").size()
    print("oracle-consensus kept / walk-forward total (accurate subset):")
    for f in ("roe", "gross_margin", "debt_to_equity"):
        print(f"  {f:<16} {int(kept.get(f, 0)):>6} / {int(tot.get(f, 0)):>6}  "
              f"({kept.get(f, 0)/max(tot.get(f, 1),1):.0%} period-accurate)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
