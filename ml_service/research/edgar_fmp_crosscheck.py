"""
CROSS-VENDOR CHECK of the live EDGAR ROE overlay against FMP — a THIRD, independent
source (not WRDS, not our own EDGAR tags). Answers "are the values we are serving RIGHT
NOW correct?" without waiting for the September Compustat upload. For each roe entry in
edgar_feature_overlay.json, recompute roe from FMP's quarterly netIncome / equity at the
matching period_end and compare. Read-only, research artifact; prints agreement + the
worst disagreers to eyeball. Run on AWS (uses the box FMP cache + key).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
from fmp_fundamentals_pipeline import _fetch_cached  # noqa: E402

DATA = ML / "data"
MATCH_DAYS = 20
AGREE_TOL = 0.10       # |served - fmp| roe agreement band (looser: 3rd-vendor conventions differ)


def fmp_roe_at(sym, period_end):
    """4 * quarterly netIncome / totalStockholdersEquity at the quarter nearest period_end."""
    pe = pd.Timestamp(period_end)
    try:
        inc = _fetch_cached(sym, "income-statement", "period=quarter&limit=24")
        bal = _fetch_cached(sym, "balance-sheet-statement", "period=quarter&limit=24")
    except Exception:
        return None
    if not inc or not bal:
        return None
    def nearest(rows, field):
        best = None
        for r in rows:
            d = r.get("date") or r.get("period")
            v = r.get(field)
            if d is None or v is None:
                continue
            dd = abs((pd.Timestamp(d) - pe).days)
            if dd <= MATCH_DAYS and (best is None or dd < best[0]):
                best = (dd, v)
        return best[1] if best else None
    ni = nearest(inc, "netIncome")
    eq = nearest(bal, "totalStockholdersEquity")
    if ni is None or eq in (None, 0):
        return None
    return 4.0 * ni / eq


def main():
    ov = json.load(open(DATA / "edgar_feature_overlay.json")).get("features", {})
    roe_entries = [(s, e["roe"]) for s, e in ov.items() if "roe" in e]
    print(f"live roe overlay entries: {len(roe_entries)}")
    rows, no_fmp = [], 0
    for i, (sym, e) in enumerate(roe_entries):
        if i % 50 == 0:
            print(f"  {i}/{len(roe_entries)}", flush=True)
        fmp = fmp_roe_at(sym, e["period_end"])
        if fmp is None:
            no_fmp += 1
            continue
        served = float(e["value"])
        rows.append((sym, e["period_end"], served, fmp, served - fmp,
                     abs(served - fmp) <= max(0.02, AGREE_TOL * abs(fmp))))
    if not rows:
        print("no FMP matches (cache cold / key missing) — run after an FMP cache warm")
        return
    df = pd.DataFrame(rows, columns=["sym", "period_end", "served", "fmp", "diff", "agree"])
    print(f"\n===== EDGAR-vs-FMP ROE CROSS-CHECK =====")
    print(f"  compared: {len(df)}  (no FMP data: {no_fmp})")
    print(f"  agree within {AGREE_TOL:.0%}: {df['agree'].mean():.1%}")
    print(f"  median |diff|: {df['diff'].abs().median():.4f}   mean |diff|: {df['diff'].abs().mean():.4f}")
    worst = df.reindex(df["diff"].abs().sort_values(ascending=False).index).head(12)
    print("  worst disagreers (served vs fmp):")
    for _, r in worst.iterrows():
        print(f"    {r['sym']:<6} {r['period_end']}  served={r['served']:+.3f}  "
              f"fmp={r['fmp']:+.3f}  diff={r['diff']:+.3f}")
    df.to_csv(DATA / "edgar_fmp_crosscheck.csv", index=False)
    print(f"  full table -> {DATA / 'edgar_fmp_crosscheck.csv'}")


if __name__ == "__main__":
    main()
