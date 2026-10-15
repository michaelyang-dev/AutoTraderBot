"""ONE rule for "the Compustat quarter the live signals use" (compustat_rows.live_rows) — 2026-10-14 fix.

WHY THESE TESTS EXIST
  Compustat's quarterly file carries a PRELIMINARY placeholder row for a company's newest quarter (datadate stamped,
  no rdq, NaN niq/seqq) before the figures arrive. The live loader skipped it (newest by report date), but the EDGAR
  vintage guard and the EDGAR patcher took the newest DATADATE — the placeholder — so for 51 S&P 1500 names (ADBE,
  PAYX, MU, ORCL, COST, NKE, FDX, ACN ...) the quarter they had FILED (ADBE's Aug-28 10-Q, filed 09-22) was never
  extracted and would have been rejected anyway: their ROE stayed one quarter stale with no alarm. All three places now
  call the shared rule; these tests pin the rule and reproduce the ADBE case end to end on synthetic data.
Run: python3 tests/test_compustat_live_row.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from compustat_rows import live_rows, live_datadate  # noqa: E402
import signal_builder as sb  # noqa: E402
import edgar_fundamentals_patch as P  # noqa: E402  (importing runs nothing: main() is guarded)

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


T = pd.Timestamp
# ADBE-like: the newest row is a placeholder (datadate 08-31, no rdq, NaN figures); the live quarter is 05-31.
# NORM: a normal name whose newest row is filled and reported.  TIE: two rows reported the same day (restatement +
# new quarter) -> the later quarter wins.  NORDQ: only placeholder rows -> no live row at all.
fund = pd.DataFrame([
    dict(tic="ADBE", datadate=T("2026-02-28"), rdq=T("2026-03-12"), niq=1600.0, seqq=11000.0, saleq=5700.0, cogsq=600.0, dlttq=5600.0, dlcq=np.nan),
    dict(tic="ADBE", datadate=T("2026-05-31"), rdq=T("2026-06-11"), niq=1712.0, seqq=11518.0, saleq=5900.0, cogsq=620.0, dlttq=5600.0, dlcq=np.nan),
    dict(tic="ADBE", datadate=T("2026-08-31"), rdq=pd.NaT, niq=np.nan, seqq=np.nan, saleq=np.nan, cogsq=np.nan, dlttq=np.nan, dlcq=np.nan),
    dict(tic="NORM", datadate=T("2026-03-31"), rdq=T("2026-04-25"), niq=100.0, seqq=1000.0, saleq=500.0, cogsq=300.0, dlttq=0.0, dlcq=0.0),
    dict(tic="NORM", datadate=T("2026-06-30"), rdq=T("2026-07-28"), niq=120.0, seqq=1050.0, saleq=520.0, cogsq=310.0, dlttq=0.0, dlcq=0.0),
    dict(tic="TIE", datadate=T("2026-03-31"), rdq=T("2026-07-30"), niq=5.0, seqq=100.0, saleq=50.0, cogsq=30.0, dlttq=0.0, dlcq=0.0),
    dict(tic="TIE", datadate=T("2026-06-30"), rdq=T("2026-07-30"), niq=6.0, seqq=104.0, saleq=52.0, cogsq=31.0, dlttq=0.0, dlcq=0.0),
    dict(tic="NORDQ", datadate=T("2026-06-30"), rdq=pd.NaT, niq=np.nan, seqq=np.nan, saleq=np.nan, cogsq=np.nan, dlttq=np.nan, dlcq=np.nan),
])
# shuffle the row order so nothing relies on file order
fund = fund.sample(frac=1.0, random_state=3).reset_index(drop=True)
snapshot = fund.copy()

# ── 1. the rule ──────────────────────────────────────────────────────────────────────────────────────────
rows = live_rows(fund)
check("placeholder newest row is skipped: ADBE's live quarter is 2026-05-31", rows.loc["ADBE", "datadate"] == T("2026-05-31") and rows.loc["ADBE", "niq"] == 1712.0)
check("normal name: newest reported quarter", rows.loc["NORM", "datadate"] == T("2026-06-30"))
check("same report date twice: the later quarter wins", rows.loc["TIE", "datadate"] == T("2026-06-30") and rows.loc["TIE", "niq"] == 6.0)
check("a name with only placeholder rows has no live row", "NORDQ" not in rows.index)
check("input frame untouched", fund.equals(snapshot))
check("live_datadate maps tic -> that quarter", live_datadate(fund)["ADBE"] == T("2026-05-31") and "NORDQ" not in live_datadate(fund))
nordq = live_rows(fund.drop(columns=["rdq"]))
check("without an rdq column: newest by datadate (legacy files)", nordq.loc["ADBE", "datadate"] == T("2026-08-31") and nordq.loc["NORDQ", "datadate"] == T("2026-06-30"))
check("empty input -> empty result", live_rows(fund.iloc[0:0]).empty)
# the rule is byte-identical to what _fill_fundamentals did before (sort by rdq, tail(1)) whenever there is no tie
old = fund.dropna(subset=["rdq"]).sort_values("rdq").groupby("tic").tail(1).set_index("tic")
check("same rows as the pre-fix loader rule for every non-tied name", all(rows.loc[t, "datadate"] == old.loc[t, "datadate"] for t in ("ADBE", "NORM")))

# ── 2. the live loader uses it: ROE for ADBE comes from the 05-31 row, not NaN from the placeholder ───────
tmp = Path(tempfile.mkdtemp())
(tmp / "wrds").mkdir()
fund.to_parquet(tmp / "wrds" / "compustat_fundamentals_quarterly.parquet")
feats = pd.DataFrame({"symbol": ["ADBE", "NORM", "TIE", "NORDQ"], "roe": np.nan, "gross_margin": np.nan,
                      "debt_to_equity": np.nan, "operating_margin": np.nan, "net_margin": np.nan, "roa": np.nan,
                      "current_ratio": np.nan, "eps_surprise_last": np.nan})
out = sb._fill_fundamentals(feats.copy(), tmp).set_index("symbol")
check("_fill_fundamentals: ADBE roe = 4 x 1712 / 11518 (the reported quarter)", abs(out.loc["ADBE", "roe"] - 4 * 1712.0 / 11518.0) < 1e-12, str(out.loc["ADBE", "roe"]))
check("_fill_fundamentals: TIE uses the later quarter (6/104)", abs(out.loc["TIE", "roe"] - 4 * 6.0 / 104.0) < 1e-12)
check("_fill_fundamentals: a placeholder-only name gets no fundamentals", np.isnan(out.loc["NORDQ", "roe"]))
check("_fill_fundamentals: debt_to_equity NaN dlcq counts as 0 (unchanged behaviour)", abs(out.loc["ADBE", "debt_to_equity"] - 5600.0 / 11518.0) < 1e-12)

# ── 3. the EDGAR vintage guard uses it: ADBE's Aug-28 filing is NEWER than the live quarter -> applied ────
json.dump({"features": {
    "ADBE": {"roe": {"value": 0.621, "period_end": "2026-08-28", "rdq": "2026-09-22"}},          # the real case
    "NORM": {"roe": {"value": 0.5, "period_end": "2026-06-30", "rdq": "2026-07-28"}},            # same quarter live has -> stale
    "TIE": {"roe": {"value": 0.9, "period_end": "2026-09-30", "rdq": "2026-10-10", "source": "llm_x"}},  # LLM: research-only
    "NORDQ": {"roe": {"value": 0.2, "period_end": "2026-06-30", "rdq": "2026-07-30"}},           # no live row -> applied
    "ADBE2": {"roe": {"value": 99.0, "period_end": "2026-08-28", "rdq": "2026-09-22"}},          # absurd -> bounded out
}}, open(tmp / "edgar_feature_overlay.json", "w"))
sb._edgar_cache["mtime"] = None
ov = sb._load_edgar_overlay(tmp)
check("guard: ADBE's filed quarter (08-28) beats the live quarter (05-31) -> overlay applied", ov.get("roe", {}).get("ADBE") == 0.621, str(ov))
check("guard: a value for the quarter live already has is NOT applied", "NORM" not in ov.get("roe", {}))
check("guard: LLM-sourced entries stay research-only", "TIE" not in ov.get("roe", {}))
check("guard: a name without any live Compustat row takes the EDGAR value", ov.get("roe", {}).get("NORDQ") == 0.2)
check("guard: plausibility bound still applies", "ADBE2" not in ov.get("roe", {}))
# the OLD guard (max datadate incl. the placeholder) would have rejected ADBE — pin the regression explicitly
old_dd = fund.groupby("tic")["datadate"].max()
check("regression: the pre-fix guard (max datadate = placeholder 08-31) would have rejected ADBE's 08-28 filing", not (T("2026-08-28") > old_dd["ADBE"]))

# ── 4. the patcher's overlap end uses it ──────────────────────────────────────────────────────────────────
last_rows, live = P.overlap_rows(fund)
check("patcher: ADBE overlap end = 2026-05-31 (live quarter), not the 08-31 placeholder", P.overlap_row("ADBE", last_rows, live)["datadate"] == T("2026-05-31"))
check("patcher: a placeholder-only name falls back to its newest datadate row", P.overlap_row("NORDQ", last_rows, live)["datadate"] == T("2026-06-30"))
check("patcher: ADBE's Aug-28 period now counts as a fresher quarter", P.is_fresher(T("2026-08-28"), T("2026-05-31")))
check("regression: against the placeholder date it did not", not P.is_fresher(T("2026-08-28"), T("2026-08-31")))
check("patcher: a period inside the 20-day fiscal-calendar slack is NOT fresher (4-4-5 ends)", not P.is_fresher(T("2026-06-15"), T("2026-05-31")))
check("patcher: the jump-guard's stale inputs come from the live row (figures, not NaN)", pd.notna(P.overlap_row("ADBE", last_rows, live)["niq"]))

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all compustat-live-row tests passed"))
sys.exit(1 if FAILS else 0)
