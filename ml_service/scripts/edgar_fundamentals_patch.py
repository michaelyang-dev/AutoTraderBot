"""
EDGAR XBRL FRESHNESS OVERLAY — per-FEATURE validation gate, Compustat-quality or nothing.

v3 redesign (2026-07-11): instead of synthesizing whole Compustat rows (all-or-nothing on
6 items -> 8% pass rate, blocked by COGS/debt conventions), validate and patch PER FEATURE.
Each sleeve-consumed feature has its own input set and its own gate:

    roe            = 4*niq/seqq          inputs: niq, seqq            (~96% provable)
    gross_margin   = (saleq-cogsq)/saleq inputs: saleq, cogsq
    debt_to_equity = (dlttq+dlcq)/seqq   inputs: dlttq, dlcq, seqq

GATE (unchanged in spirit): a feature for a symbol is patched ONLY if every one of ITS
inputs, extracted from EDGAR companyfacts with a candidate tag-spec, REPRODUCES Compustat's
own last overlapping quarter within 2% / $2M. All inputs of a feature must come from the
SAME (newer) period end — no mixed-quarter features. Anything unproven stays stale-clean.

Output: data/edgar_feature_overlay.json — INERT until signal_builder integration (flagged,
shadow-tested). Formulas mirror signal_builder._fill_fundamentals verbatim, including
dlcq/dlttq NaN->0. `rdq` = SEC filed date (point-in-time honest).

Facts are cached in data/edgar_cache/*.json.gz (20h TTL) so re-runs take seconds.
Usage (AWS): venv/bin/python scripts/edgar_fundamentals_patch.py
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ML = Path(__file__).resolve().parent.parent
DATA = ML / "data"
FUND = DATA / "wrds" / "compustat_fundamentals_quarterly.parquet"
OVERLAY = DATA / "edgar_feature_overlay.json"
REPORT = DATA / "edgar_patch_report.json"
CACHE = DATA / "edgar_cache"
UA = {"User-Agent": "AutoTrader research michaelslyang@gmail.com"}

# Candidate specs: (base_tag, [add_alternative_lists], [sub_alternative_lists]).
# value = base + sum(first-available of each add-list else 0) - sum(same for subs).
# Derived combos encode Compustat conventions; the per-symbol gate arbitrates.
_DA = ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
       "DepreciationAmortizationAndAccretionNet", "Depreciation"]
_STB = ["CommercialPaper", "ShortTermBorrowings", "OtherShortTermBorrowings",
        "ShortTermBankLoansAndNotesPayable"]
_FLN = ["FinanceLeaseLiabilityNoncurrent"]
_FLC = ["FinanceLeaseLiabilityCurrent"]
CONCEPTS = {
    "niq":   [("NetIncomeLoss", [], []), ("ProfitLoss", [], []),
              ("NetIncomeLossAvailableToCommonStockholdersBasic", [], [])],
    "saleq": [("RevenueFromContractWithCustomerExcludingAssessedTax", [], []),
              ("Revenues", [], []), ("SalesRevenueNet", [], []),
              ("RevenueFromContractWithCustomerIncludingAssessedTax", [], []),
              ("RevenuesNetOfInterestExpense", [], []),      # banks/financials
              ("InterestAndDividendIncomeOperating", [], [])],
    "cogsq": [("CostOfRevenue", [], [_DA]), ("CostOfGoodsAndServicesSold", [], [_DA]),
              ("CostOfRevenue", [], []), ("CostOfGoodsAndServicesSold", [], []),
              ("CostOfGoodsSold", [], [_DA]), ("CostOfGoodsSold", [], [])],
    "seqq":  [("StockholdersEquity", [], []),
              ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", [], [])],
    "dlttq": [("LongTermDebtNoncurrent", [], []), ("LongTermDebtNoncurrent", [_FLN], []),
              ("LongTermDebtAndCapitalLeaseObligations", [], []), ("LongTermDebt", [], [])],
    "dlcq":  [("LongTermDebtCurrent", [_STB], []), ("DebtCurrent", [], []),
              ("LongTermDebtCurrent", [], []), ("LongTermDebtCurrent", [_FLC], []),
              ("LongTermDebtAndCapitalLeaseObligationsCurrent", [], [])],
}
FLOWS = {"niq", "saleq", "cogsq"}
# CALIBRATED GATE DEPTHS (gate_calibration_test.py, 1,504 symbols, next-quarter holdout):
# roe inputs at 1 anchor = 99.2% next-q accuracy (2 anchors adds NOTHING, costs coverage);
# cogs/debt items never exceed ~88-93% at any depth -> their features are generated for
# research but EXCLUDED from the live overlay by the loader (see signal_builder).
GATE_DEPTH = {"niq": 1, "seqq": 1, "saleq": 1, "cogsq": 2, "dlttq": 2, "dlcq": 2}
ZERO_OK = {"dlttq", "dlcq"}            # legitimately 0/absent (debt-free firms)
REL_TOL, ABS_TOL = 0.02, 2.0           # 2% relative or $2M absolute ($MM units)

# The sleeve-consumed features and EXACT live formulas (signal_builder._fill_fundamentals)
FEATURES = {
    "roe":            (["niq", "seqq"],
                       lambda v: 4 * v["niq"] / v["seqq"] if v.get("seqq") else None),
    "gross_margin":   (["saleq", "cogsq"],
                       lambda v: (v["saleq"] - v["cogsq"]) / v["saleq"] if v.get("saleq") else None),
    "debt_to_equity": (["dlttq", "dlcq", "seqq"],
                       lambda v: ((v.get("dlttq") or 0) + (v.get("dlcq") or 0)) / v["seqq"]
                       if v.get("seqq") else None),
}


def load_cik_map():
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30)
    r.raise_for_status()
    return {v["ticker"].upper(): int(v["cik_str"]) for v in r.json().values()}


def fetch_facts(cik):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{cik}.json.gz"
    if f.exists() and time.time() - f.stat().st_mtime < 20 * 3600:
        try:
            return json.load(gzip.open(f, "rt"))
        except Exception:
            pass
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                data = r.json()
                with gzip.open(f, "wt") as fh:
                    json.dump(data, fh)
                time.sleep(0.12)
                return data
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return None


def series_for_concept(facts, concept, pit=False):
    """{period_end: (value_$MM, filed)} quarterly + annual buckets; latest-filed wins.

    pit=True: EARLIEST-filed wins (as-first-reported). Production wants the freshest
    restated value (pit=False); historical point-in-time research needs the value and
    filed date of the ORIGINAL filing — later 10-K/10-Q comparatives re-report old
    quarters and inflate availability by ~16 months (bug found 2026-07-12).
    """
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
        if e.get("start"):
            days = (end - pd.Timestamp(e["start"])).days
            bucket = quarters if 80 <= days <= 100 else annuals if 350 <= days <= 380 else None
            if bucket is None:
                continue
        else:
            bucket = quarters
        prev = bucket.get(end)
        if prev is None or (filed < prev[1] if pit else filed >= prev[1]):
            bucket[end] = (val, filed)
    return quarters, annuals


def value_at(quarters, annuals, end, is_flow):
    """Quarter value NEAREST `end` within 20d; fiscal-Q4 flows derived as FY - 3
    sibling quarters. Retail 4-4-5 fiscal ends sit up to ~2 weeks off Compustat's
    month-end datadate (e.g. AZO quarter ends 2/14 vs Compustat 2/28) — the old ±5d
    window silently dropped them. Duration was already validated to 80-100d in
    series_for_concept, so nearest-within-20 cannot match a wrong-length period, and
    the caller still gates every value against Compustat."""
    best = None
    for d, (v, filed) in (quarters or {}).items():
        dd = abs((d - end).days)
        if dd <= 20 and (best is None or dd < best[0]):
            best = (dd, v, filed)
    if best is not None:
        return best[1], best[2]
    if is_flow and annuals:
        for d, (fy_val, filed) in annuals.items():
            if abs((d - end).days) <= 20:
                sibs = [v for q, (v, _) in quarters.items()
                        if pd.Timedelta(days=50) < (d - q) < pd.Timedelta(days=330)]
                if len(sibs) == 3:
                    return fy_val - sum(sibs), filed
    return None, None


def spec_value(facts, spec, end, is_flow, pit=False):
    base, adds, subs = spec
    q, a = series_for_concept(facts, base, pit)
    if q is None:
        return None, None
    v, filed = value_at(q, a, end, is_flow)
    if v is None:
        return None, None
    total = v
    for sign, groups in ((1, adds), (-1, subs)):
        for alt_list in groups:
            for tag in alt_list:
                q2, a2 = series_for_concept(facts, tag, pit)
                if q2 is None:
                    continue
                v2, f2 = value_at(q2, a2, end, is_flow)
                if v2 is not None:
                    total += sign * v2
                    if pit and f2 is not None and (filed is None or f2 > filed):
                        filed = f2   # PIT availability = when ALL components existed
                    break
    return total, filed


def close_enough(edgar, compustat):
    if pd.isna(compustat):
        return edgar is None or abs(edgar) <= ABS_TOL
    if edgar is None:
        return False
    return abs(edgar - compustat) <= max(ABS_TOL, REL_TOL * abs(compustat))


def item_anchors(g, item, depth):
    """Most recent `depth` quarters with NON-NULL Compustat truth for `item`, newest
    first. The latest WRDS quarter is often preliminary (NaN fields) during the summer
    stale window — anchoring the gate there fails even when earlier quarters would
    validate (measured: 21 symbols have NaN seqq at the latest quarter; recovers 19).
    Debt items count NaN as a legitimate 0."""
    rows = []
    for _, r in g.iloc[::-1].iterrows():
        if pd.notna(r.get(item)) or item in ZERO_OK:
            rows.append(r)
        if len(rows) >= depth:
            break
    return rows


def main():
    members = json.load(open(DATA / "sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    last_rows = fund.sort_values("datadate").drop_duplicates("tic", keep="last").set_index("tic")
    by_tic = {t: g.sort_values("datadate") for t, g in fund.groupby("tic")}
    cik_map = load_cik_map()

    stats = {f: {"validated": 0, "patched": 0, "failed": 0} for f in FEATURES}
    fail_items = {}
    overlay = {}
    t0 = time.time()
    for i, sym in enumerate(syms):
        if i % 150 == 0:
            print(f"  {i}/{len(syms)} ({time.time()-t0:.0f}s) roe_ok={stats['roe']['validated']} "
                  f"roe_patched={stats['roe']['patched']}", flush=True)
        if sym not in last_rows.index:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            continue
        facts = fetch_facts(cik)
        if facts is None:
            continue
        row = last_rows.loc[sym]
        overlap_end = row["datadate"]
        g = by_tic[sym]

        # resolve each ITEM once. TWO-QUARTER GATE (2026-07-11 hardening): the spec must
        # reproduce Compustat on the last two overlapping quarters (where a second
        # exists) — protects against unstable/fluke tag mappings and mid-stream
        # reclassifications. Anchors are the most recent NON-NULL-truth quarters per
        # item (skips preliminary NaN latest quarters during the summer stale window).
        resolved = {}
        for item, candidates in CONCEPTS.items():
            depth = GATE_DEPTH.get(item, 1)
            anchors = item_anchors(g, item, depth)
            a0 = anchors[0] if anchors else row
            cs_val = a0[item]
            hit = None
            for spec in candidates:
                v, _ = spec_value(facts, spec, a0["datadate"], item in FLOWS)
                if not close_enough(v, cs_val):
                    continue
                if depth >= 2 and len(anchors) >= 2:
                    v2, _ = spec_value(facts, spec, anchors[1]["datadate"], item in FLOWS)
                    if not close_enough(v2, anchors[1][item]):
                        continue
                hit = spec
                break
            if hit is None and item in ZERO_OK and (pd.isna(cs_val) or abs(cs_val) <= ABS_TOL):
                resolved[item] = "ZERO"      # provably ~0 at overlap; treat as 0 going forward
                continue
            if hit is None:
                fail_items[item] = fail_items.get(item, 0) + 1
            resolved[item] = hit

        sym_overlay = {}
        for feat, (inputs, formula) in FEATURES.items():
            if any(resolved.get(it) is None for it in inputs):
                stats[feat]["failed"] += 1
                continue
            stats[feat]["validated"] += 1
            # newest period end AFTER the overlap where ALL of this feature's inputs exist
            base_ends = set()
            for it in inputs:
                if resolved[it] == "ZERO":
                    continue
                q, _a = series_for_concept(facts, resolved[it][0])
                base_ends |= {d for d in (q or {}) if d > overlap_end + pd.Timedelta(days=20)}
            for end in sorted(base_ends, reverse=True):
                vals, rdq, complete = {}, None, True
                for it in inputs:
                    if resolved[it] == "ZERO":
                        vals[it] = 0.0
                        continue
                    v, filed = spec_value(facts, resolved[it], end, it in FLOWS)
                    if v is None:
                        complete = False
                        break
                    vals[it] = v
                    rdq = max(rdq or filed, filed)
                if complete:
                    fv = formula(vals)
                    if fv is not None and np.isfinite(fv):
                        # JUMP GUARD: a reclassification mid-stream shows up as an
                        # implausible leap vs the stale value — withhold those for review
                        stale_inputs = {it: (row[it] if pd.notna(row[it]) else 0.0)
                                        for it in inputs}
                        old = formula(stale_inputs) if all(
                            it in stale_inputs for it in inputs) else None
                        if (old is not None and np.isfinite(old)
                                and abs(fv) > 5 * max(abs(old), 0.01) and abs(fv) > 0.5):
                            stats[feat]["jump_guarded"] = stats[feat].get("jump_guarded", 0) + 1
                        else:
                            sym_overlay[feat] = {"value": round(float(fv), 6),
                                                 "period_end": str(end.date()), "rdq": rdq}
                            stats[feat]["patched"] += 1
                    break
        if sym_overlay:
            overlay[sym] = sym_overlay

    # ── eps_surprise_last freshness: FMP announcement events NEWER than the IBES vintage.
    # Validated 2026-07-11 vs IBES on 57,864 matched events: Spearman 0.841, beat/miss sign
    # agreement 91.5% — adequate for this minor boost feature. Formula matches
    # signal_builder: (actual - estimate) / |estimate|.
    eps_stats = {"fresh": 0, "skipped_stale": 0}
    try:
        ib = pd.read_parquet(DATA / "wrds" / "ibes_summary_latest.parquet",
                             columns=["OFTIC", "ANNDATS_ACT"]).dropna()
        ib["ANNDATS_ACT"] = pd.to_datetime(ib["ANNDATS_ACT"])
        ibes_vintage = ib.groupby("OFTIC")["ANNDATS_ACT"].max()
        fe = pd.read_parquet(DATA / "fundamentals_earnings.parquet").dropna(
            subset=["eps_actual", "eps_estimated"])
        fe["date"] = pd.to_datetime(fe["date"])
        fe = fe[fe["date"] <= pd.Timestamp.now()]
        fel = fe.sort_values("date").groupby("symbol").last()
        for sym in syms:
            if sym not in fel.index:
                continue
            ev = fel.loc[sym]
            vint = ibes_vintage.get(sym)
            if vint is not None and ev["date"] <= vint:
                eps_stats["skipped_stale"] += 1
                continue
            est = ev["eps_estimated"]
            if est == 0 or pd.isna(est):
                continue
            surp = float((ev["eps_actual"] - est) / abs(est))
            overlay.setdefault(sym, {})["eps_surprise_last"] = {
                "value": round(surp, 6), "period_end": str(ev["date"].date()),
                "rdq": str(ev["date"].date()), "source": "fmp_event"}
            eps_stats["fresh"] += 1
    except Exception as e:
        print(f"  eps_surprise extension failed (skipped): {e}")
    stats["eps_surprise_last"] = {"validated": eps_stats["fresh"] + eps_stats["skipped_stale"],
                                  "patched": eps_stats["fresh"], "failed": 0}

    print(f"\n===== EDGAR OVERLAY REPORT ({time.time()-t0:.0f}s, {len(syms)} symbols) =====")
    for feat, s in stats.items():
        print(f"  {feat:<16} validated {s['validated']:>4} ({s['validated']/len(syms):.0%})"
              f"  | fresher-quarter available now: {s['patched']:>4}")
    print(f"  item failures: {fail_items}")
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "compustat_max_datadate": str(fund['datadate'].max().date()),
               "stats": stats, "fail_items": fail_items,
               "features": overlay}, open(OVERLAY, "w"), indent=1)
    json.dump({"stats": stats, "fail_items": fail_items}, open(REPORT, "w"), indent=1)
    print(f"  overlay -> {OVERLAY} ({len(overlay)} symbols with >=1 fresh feature)")
    print("  INERT: nothing reads this file until the flagged signal_builder integration + shadow pass.")


if __name__ == "__main__":
    main()
