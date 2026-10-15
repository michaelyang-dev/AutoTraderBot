"""THE Compustat quarter the live signals use — one rule, three consumers.

Compustat's quarterly file carries a PRELIMINARY row for a company's newest quarter before the figures arrive: the
datadate is stamped, the report date (rdq) is empty and niq/seqq/saleq are NaN (51 S&P 1500 names on 2026-10-14:
ADBE, PAYX, MU, ORCL, COST, NKE, FDX, ACN ...). Three places decide which row is "current" and they MUST agree:

  signal_builder._fill_fundamentals   the row whose fields become the live roe / gross_margin / debt_to_equity
  signal_builder._load_edgar_overlay  the vintage guard: an EDGAR value applies only if its period is NEWER than that row
  scripts/edgar_fundamentals_patch.py the overlap end: EDGAR periods after it are "fresher quarters" worth extracting

Found 2026-10-14: the loader used the newest row BY REPORT DATE (skipping the placeholder) while the guard and the
patcher used the newest DATADATE (the placeholder). So for those 51 names the filed quarter in between (ADBE's Aug-28
10-Q, filed 09-22) was never extracted and would have been rejected anyway — their ROE stayed one quarter stale with no
alarm. Everything now calls live_rows(); the rule is tested in tests/test_compustat_live_row.py.
"""
import pandas as pd

DATE_COL_PRIMARY = "rdq"       # report date: point-in-time honest; a placeholder row has none
DATE_COL_FALLBACK = "datadate"


def live_rows(fund):
    """One row per ticker, indexed by `tic`: the newest quarter by REPORT DATE (rdq). Rows without an rdq — Compustat's
    preliminary placeholders — are excluded. If the frame has no rdq column at all, the newest by datadate is used.
    Ties on the date are broken by datadate (the later quarter wins) with a stable sort, so the choice is deterministic.
    Returns an empty frame (same columns) when nothing qualifies. Never mutates the input."""
    date_col = DATE_COL_PRIMARY if DATE_COL_PRIMARY in fund.columns else DATE_COL_FALLBACK
    f = fund.copy()
    f[date_col] = pd.to_datetime(f[date_col], errors="coerce")
    f = f.dropna(subset=[date_col])
    if f.empty:
        return f.set_index("tic") if "tic" in f.columns else f
    keys = [date_col] + ([DATE_COL_FALLBACK] if date_col != DATE_COL_FALLBACK and DATE_COL_FALLBACK in f.columns else [])
    return f.sort_values(keys, kind="mergesort").groupby("tic").tail(1).set_index("tic")


def live_datadate(fund):
    """{tic: datadate of the live row} — the vintage a fresher EDGAR period must beat."""
    rows = live_rows(fund)
    if rows.empty or DATE_COL_FALLBACK not in rows.columns:
        return {}
    return pd.to_datetime(rows[DATE_COL_FALLBACK]).to_dict()
