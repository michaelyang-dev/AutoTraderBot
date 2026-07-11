"""
EDGAR XBRL FUNDAMENTALS FRESHNESS PATCH — Compustat-quality gate, stale-clean fallback.

Purpose: WRDS Compustat refreshes quarterly by manual download (unavailable in summer).
This patcher freshens ONLY the newest missing quarter per symbol, directly from SEC EDGAR
companyfacts (the primary source Compustat itself derives from), under a hard per-symbol
validation gate:

  A symbol is patched ONLY if our EDGAR extraction REPRODUCES Compustat's own last
  overlapping quarter for ALL six items (niq, saleq, cogsq, seqq, dlttq, dlcq) within
  2% relative / $2M absolute. Any mismatch, missing tag, or ambiguity -> the symbol
  stays stale-but-clean (measured: stale-clean beats fresh-noisy). Quality is proven
  per symbol, never assumed.

Items patched are exactly the inputs of the three sleeve-consumed features
(roe = 4*niq/seqq, gross_margin = (saleq-cogsq)/saleq, debt_to_equity = (dlttq+dlcq)/seqq,
formulas per signal_builder._fill_fundamentals). All other columns in the synthesized row
carry forward the symbol's previous values (they feed no sleeve; NaN would be worse).
Flows use quarterly-duration XBRL entries (80-100d); fiscal-Q4 flows are derived as
FY minus the three sibling quarters. Instants (equity/debt) are taken at period end.
`rdq` (the point-in-time selector used by signal_builder) = the SEC `filed` date.

Usage (AWS):
  venv/bin/python scripts/edgar_fundamentals_patch.py                 # VALIDATE-ONLY report
  venv/bin/python scripts/edgar_fundamentals_patch.py --emit          # + write patched parquet
Output: data/wrds/compustat_fundamentals_quarterly_patched.parquet (original untouched)
        data/edgar_patch_report.json
NEVER wired into live here — signal_builder keeps reading the original file until the
shadow-diff phase proves the patched picks are sane and the user flips it.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ML = Path(__file__).resolve().parent.parent
DATA = ML / "data"
FUND = DATA / "wrds" / "compustat_fundamentals_quarterly.parquet"
OUT = DATA / "wrds" / "compustat_fundamentals_quarterly_patched.parquet"
REPORT = DATA / "edgar_patch_report.json"
UA = {"User-Agent": "AutoTrader research michaelslyang@gmail.com"}

# Compustat item -> ordered candidate SPECS: (base_tag, [add_alternatives...], [sub_alternatives...]).
# value = base + sum(first-available of each add-list, else 0) - sum(same for sub-lists).
# Derived combos encode Compustat conventions: cogsq EXCLUDES D&A (verified: AAPL/MU/PAYX
# match to the dollar after subtracting D&A); dlcq = current LTD + commercial paper/ST borrowings.
# The per-symbol validation gate still decides — a wrong combo simply fails and stays stale.
_DA = ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
       "DepreciationAmortizationAndAccretionNet", "Depreciation"]
_STB = ["CommercialPaper", "ShortTermBorrowings", "OtherShortTermBorrowings", "ShortTermBankLoansAndNotesPayable"]
CONCEPTS = {
    "niq":   [("NetIncomeLoss", [], []), ("ProfitLoss", [], []),
              ("NetIncomeLossAvailableToCommonStockholdersBasic", [], [])],
    "saleq": [("RevenueFromContractWithCustomerExcludingAssessedTax", [], []), ("Revenues", [], []),
              ("SalesRevenueNet", [], []), ("RevenueFromContractWithCustomerIncludingAssessedTax", [], [])],
    "cogsq": [("CostOfRevenue", [], [_DA]), ("CostOfGoodsAndServicesSold", [], [_DA]),
              ("CostOfRevenue", [], []), ("CostOfGoodsAndServicesSold", [], []),
              ("CostOfGoodsSold", [], [_DA]), ("CostOfGoodsSold", [], [])],
    "seqq":  [("StockholdersEquity", [], []),
              ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", [], [])],
    "dlttq": [("LongTermDebtNoncurrent", [], []), ("LongTermDebtAndCapitalLeaseObligations", [], []),
              ("LongTermDebt", [], [])],
    "dlcq":  [("LongTermDebtCurrent", [_STB], []), ("DebtCurrent", [], []),
              ("LongTermDebtCurrent", [], []), ("LongTermDebtAndCapitalLeaseObligationsCurrent", [_STB], []),
              ("LongTermDebtAndCapitalLeaseObligationsCurrent", [], [])],
}
FLOWS = {"niq", "saleq", "cogsq"}          # duration items (need quarterly windows)
ZERO_OK = {"dlttq", "dlcq", "cogsq"}       # items legitimately 0/absent for some firms
REL_TOL, ABS_TOL = 0.02, 2.0               # 2% relative or $2M absolute (Compustat is $MM)


def load_cik_map():
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30)
    r.raise_for_status()
    return {v["ticker"].upper(): int(v["cik_str"]) for v in r.json().values()}


def fetch_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return None


def series_for_concept(facts, concept):
    """{period_end(Timestamp): (value_$MM, filed_date)} — latest-filed wins per period.
    For FLOW concepts also returns annual windows for Q4 derivation."""
    node = facts.get("facts", {}).get("us-gaap", {}).get(concept)
    if not node:
        return None, None
    units = node.get("units", {}).get("USD")
    if not units:
        return None, None
    quarters, annuals = {}, {}
    for e in units:
        try:
            end = pd.Timestamp(e["end"])
            val = float(e["val"]) / 1e6
            filed = e.get("filed", "1900-01-01")
        except (KeyError, TypeError, ValueError):
            continue
        if "start" in e and e.get("start"):
            days = (end - pd.Timestamp(e["start"])).days
            bucket = quarters if 80 <= days <= 100 else annuals if 350 <= days <= 380 else None
            if bucket is None:
                continue
        else:
            bucket = quarters   # instant
        prev = bucket.get(end)
        if prev is None or filed >= prev[1]:
            bucket[end] = (val, filed)
    return quarters, annuals


def value_at(quarters, annuals, end, is_flow):
    """Value for the quarter ending `end` (±5d). Flows: direct quarterly window, else
    fiscal-Q4 = annual(end) - three sibling quarters inside that annual window."""
    for d, (v, filed) in (quarters or {}).items():
        if abs((d - end).days) <= 5:
            return v, filed
    if is_flow and annuals:
        for d, (fy_val, filed) in annuals.items():
            if abs((d - end).days) <= 5:
                sibs = [v for q, (v, _) in quarters.items()
                        if pd.Timedelta(days=50) < (d - q) < pd.Timedelta(days=330)]
                if len(sibs) == 3:
                    return fy_val - sum(sibs), filed
    return None, None


def spec_value(facts, spec, end, is_flow):
    """Evaluate a candidate spec (base, add_alt_lists, sub_alt_lists) at a period end.
    Add/sub components are flows/instants matching the base item's nature; absent
    optional components contribute 0. Returns (value, filed) or (None, None)."""
    base, adds, subs = spec
    q, a = series_for_concept(facts, base)
    if q is None:
        return None, None
    v, filed = value_at(q, a, end, is_flow)
    if v is None:
        return None, None
    total = v
    for alt_list in adds + [["___SUB___"] + x for x in subs]:
        sign = 1
        if alt_list and alt_list[0] == "___SUB___":
            sign, alt_list = -1, alt_list[1:]
        for tag in alt_list:
            q2, a2 = series_for_concept(facts, tag)
            if q2 is None:
                continue
            v2, _ = value_at(q2, a2, end, is_flow)
            if v2 is not None:
                total += sign * v2
                break
    return total, filed


def close_enough(edgar, compustat):
    if pd.isna(compustat):
        return edgar is None or abs(edgar) <= ABS_TOL
    if edgar is None:
        return False
    return abs(edgar - compustat) <= max(ABS_TOL, REL_TOL * abs(compustat))


def main():
    emit = "--emit" in sys.argv
    members = json.load(open(DATA / "sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    last_rows = fund.sort_values("datadate").drop_duplicates("tic", keep="last").set_index("tic")
    cik_map = load_cik_map()

    stats = {"patched": [], "validated_no_new": [], "failed_validation": [],
             "no_cik": [], "no_facts": [], "no_overlap": []}
    new_rows = []
    t0 = time.time()
    for i, sym in enumerate(syms):
        if i % 150 == 0:
            print(f"  {i}/{len(syms)} ({time.time()-t0:.0f}s) patched={len(stats['patched'])} "
                  f"failed={len(stats['failed_validation'])}", flush=True)
        if sym not in last_rows.index:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            stats["no_cik"].append(sym)
            continue
        facts = fetch_facts(cik)
        time.sleep(0.12)
        if facts is None:
            stats["no_facts"].append(sym)
            continue

        row = last_rows.loc[sym]
        overlap_end = row["datadate"]
        # resolve, per item, the first concept that validates on the overlap quarter
        resolved, ok = {}, True
        for item, candidates in CONCEPTS.items():
            cs_val = row[item]
            hit = None
            for spec in candidates:
                v, _ = spec_value(facts, spec, overlap_end, item in FLOWS)
                if close_enough(v, cs_val):
                    hit = spec
                    break
            if hit is None:
                if item in ZERO_OK and (pd.isna(cs_val) or abs(cs_val) <= ABS_TOL):
                    resolved[item] = None     # legitimately absent; treat as 0/NaN downstream
                    continue
                ok = False
                stats["failed_validation"].append(f"{sym}:{item}")
                break
            resolved[item] = hit
        if not ok:
            continue

        # find the newest quarter-end strictly after Compustat's last, where every
        # resolved concept has a value (validated tags only — no re-guessing)
        cand_ends = set()
        for item, spec in resolved.items():
            if spec is None:
                continue
            q, a = series_for_concept(facts, spec[0])
            cand_ends |= {d for d in (q or {}) if d > overlap_end + pd.Timedelta(days=20)}
        newer = sorted(cand_ends)
        patched = None
        for end in reversed(newer):
            vals, rdq = {}, None
            complete = True
            for item, spec in resolved.items():
                if spec is None:
                    vals[item] = np.nan if item == "dlttq" else 0.0
                    continue
                v, filed = spec_value(facts, spec, end, item in FLOWS)
                if v is None:
                    complete = False
                    break
                vals[item] = v
                rdq = max(rdq or filed, filed)
            if complete:
                patched = (end, vals, rdq)
                break
        if patched is None:
            stats["validated_no_new"].append(sym)
            continue

        end, vals, rdq = patched
        nr = row.copy()                      # carry forward unconsumed columns
        nr["datadate"] = end
        if "rdq" in nr.index:
            nr["rdq"] = pd.Timestamp(rdq)
        for item, v in vals.items():
            nr[item] = v
        nr["tic"] = sym
        new_rows.append(nr)
        stats["patched"].append(f"{sym}:{end.date()}")

    print(f"\n===== EDGAR PATCH REPORT ({time.time()-t0:.0f}s) =====")
    for k in stats:
        print(f"  {k}: {len(stats[k])}")
    coverage = len(stats["patched"]) / max(1, len(syms))
    print(f"  freshness coverage: {coverage:.1%} of SP1500 gets a newer validated quarter")
    REPORT.parent.mkdir(exist_ok=True)
    json.dump({k: v for k, v in stats.items()}, open(REPORT, "w"), indent=1, default=str)
    print(f"  report -> {REPORT}")

    if emit and new_rows:
        add = pd.DataFrame(new_rows).reset_index(drop=True)
        out = pd.concat([fund, add], ignore_index=True)
        out.to_parquet(OUT)
        print(f"  emitted {len(add)} synthesized rows -> {OUT} (original untouched)")
    elif not emit:
        print("  VALIDATE-ONLY mode (pass --emit to write the patched parquet)")


if __name__ == "__main__":
    main()
