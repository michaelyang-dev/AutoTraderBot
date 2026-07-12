"""
EDGAR LLM EXTRACTION CANDIDATE — raise GM/D2E coverage; same proof gate as XBRL.

For symbols where XBRL tag-candidates FAILED the Compustat gate (COGS conventions, debt
classification), an LLM reads the actual filed financial statements (the R-file exhibits
of the 10-Q/10-K) and reports the line items. The extraction is trusted ONLY if it
reproduces Compustat's overlap quarter (same close_enough tolerance) — model quality
therefore affects COVERAGE, never CORRECTNESS. Passing symbols get their newest-quarter
value extracted from the latest filing, emitted to data/edgar_llm_overlay.json
(same schema as the XBRL overlay; loader merges, XBRL wins conflicts).

Pilot usage:  venv/bin/python scripts/edgar_llm_extract.py --model haiku --limit 25
Full run:     venv/bin/python scripts/edgar_llm_extract.py --model haiku
Models: haiku = claude-haiku-4-5-20251001, sonnet = claude-sonnet-4-6.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edgar_fundamentals_patch import (  # noqa: E402
    CONCEPTS, DATA, FEATURES, FLOWS, FUND, UA, ZERO_OK,
    close_enough, fetch_facts, load_cik_map, spec_value)

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
API_KEY = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
MODELS = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6"}
OUT = DATA / "edgar_llm_overlay.json"
STATE = DATA / "edgar_llm_processed.json"   # {sym: last_processed_period_end} — makes daily
# runs INCREMENTAL: a symbol is re-attempted only when a NEWER filing exists, so the
# recurring cost is ~10-50 new filings/day in earnings season (~$0.2-1/day), not a full
# $20 backfill. Delete the file to force a full re-run.
LLM_ITEMS = {"saleq", "cogsq", "niq", "seqq", "dlttq", "dlcq"}

PROMPT = """From the financial statements below, extract these values for the fiscal
quarter ended {end} (use the THREE-MONTH column, not year-to-date; balance-sheet items
as of {end}). Normalize everything to USD MILLIONS (the statements say if figures are in
thousands or millions). Reply with ONLY a JSON object, keys:
  revenue            total revenue / net sales for the quarter
  cogs_presented     cost of revenue / cost of sales AS PRESENTED on the income statement
  da_in_costs        depreciation & amortization included in cost of revenue if separately
                     disclosed on these statements, else null
  da_total           total depreciation & amortization for the quarter (usually on the
                     cash flow statement), else null
  net_income         net income attributable to the company (loss negative)
  stockholders_equity  total stockholders' equity (parent, excl. noncontrolling interests
                     if broken out)
  lt_debt_noncurrent long-term debt, noncurrent portion
  current_debt_total current portion of long-term debt PLUS short-term borrowings /
                     commercial paper PLUS current finance-lease liabilities (sum the
                     components you can see; null only if none are shown)
Use null for anything not determinable from these statements. No commentary.

STATEMENTS:
{text}"""


def llm(model, content):
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"},
                      json={"model": MODELS[model], "max_tokens": 400, "temperature": 0,
                            "messages": [{"role": "user", "content": content}]},
                      timeout=90)
    r.raise_for_status()
    txt = r.json()["content"][0]["text"]
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def sec_get(url):
    time.sleep(0.12)
    r = requests.get(url, headers=UA, timeout=30)
    return r if r.status_code == 200 else None


def filing_statements(cik, report_date):
    """Fetch income statement + balance sheet text for the filing covering report_date."""
    sub = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if not sub:
        return None, None
    recent = sub.json().get("filings", {}).get("recent", {})
    best = None
    for form, acc, rdate, fdate in zip(recent.get("form", []), recent.get("accessionNumber", []),
                                       recent.get("reportDate", []), recent.get("filingDate", [])):
        if form in ("10-Q", "10-K") and rdate:
            if abs((pd.Timestamp(rdate) - report_date).days) <= 5:
                best = (acc.replace("-", ""), fdate)
                break
    if not best:
        return None, None
    acc, fdate = best
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"
    fs = sec_get(f"{base}/FilingSummary.xml")
    if not fs:
        return None, None
    reports = re.findall(r"<Report[^>]*>(.*?)</Report>", fs.text, re.DOTALL)
    picks = []
    for rep in reports:
        name_m = re.search(r"<ShortName>(.*?)</ShortName>", rep, re.DOTALL)
        file_m = re.search(r"<HtmlFileName>(.*?)</HtmlFileName>", rep)
        if not name_m or not file_m:
            continue
        name = name_m.group(1).upper()
        if "PARENTHETICAL" in name or "COMPREHENSIVE" in name:
            continue
        if re.search(r"STATEMENTS? OF (OPERATIONS|INCOME)|INCOME STATEMENTS?", name) or \
           re.search(r"BALANCE SHEETS?|FINANCIAL POSITION", name) or \
           re.search(r"CASH FLOWS?", name):
            picks.append(file_m.group(1))
        if len(picks) >= 3:
            break
    if not picks:
        return None, None
    text = ""
    for f in picks[:3]:
        page = sec_get(f"{base}/{f}")
        if page:
            t = page.text
            t = re.sub(r"</t[dh]>", " | ", t, flags=re.I)     # keep column boundaries
            t = re.sub(r"</tr>", "\n", t, flags=re.I)         # keep row boundaries
            t = re.sub(r"<[^>]+>", " ", t)
            t = re.sub(r"[ \t]+", " ", t)
            t = re.sub(r"\n\s*\n+", "\n", t)
            text += t[:20000] + "\n\n"
    return (text if text else None), fdate


def candidates_from_llm(vals):
    """Compustat-convention candidates per item from the LLM's raw line items."""
    out = {}
    g = lambda k: vals.get(k) if isinstance(vals.get(k), (int, float)) else None
    if g("revenue") is not None:
        out["saleq"] = [g("revenue")]
    if g("cogs_presented") is not None:
        c = g("cogs_presented")
        cands = []
        if g("da_in_costs"):
            cands.append(c - g("da_in_costs"))
        if g("da_total"):
            cands.append(c - g("da_total"))
        cands.append(c)
        out["cogsq"] = cands
    if g("net_income") is not None:
        out["niq"] = [g("net_income")]
    if g("stockholders_equity") is not None:
        out["seqq"] = [g("stockholders_equity")]
    if g("lt_debt_noncurrent") is not None:
        out["dlttq"] = [g("lt_debt_noncurrent")]
    if g("current_debt_total") is not None:
        out["dlcq"] = [g("current_debt_total")]
    return out


def main():
    model = "haiku"
    limit = None
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    assert API_KEY, "ANTHROPIC_API_KEY missing"

    members = json.load(open(DATA / "sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    last_rows = fund.sort_values("datadate").drop_duplicates("tic", keep="last").set_index("tic")
    cik_map = load_cik_map()

    stats = {"targets": 0, "gate_pass": 0, "gate_fail": 0, "no_docs": 0, "fresh": 0}
    item_pass = {k: 0 for k in LLM_ITEMS}
    overlay = {}
    t0 = time.time()
    done = 0
    for sym in syms:
        if limit and done >= limit:
            break
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

        # XBRL resolution first — LLM only targets what XBRL could NOT prove
        resolved = {}
        for item, cands in CONCEPTS.items():
            hit = None
            for spec in cands:
                v, _ = spec_value(facts, spec, overlap_end, item in FLOWS)
                if close_enough(v, row[item]):
                    hit = spec
                    break
            if hit is None and item in ZERO_OK and (pd.isna(row[item]) or abs(row[item]) <= 2.0):
                hit = "ZERO"
            resolved[item] = hit
        missing = [it for it, h in resolved.items() if h is None]
        if not missing:
            continue
        # incremental guard: skip if no filing newer than what we already processed
        try:
            processed = json.load(open(STATE)) if STATE.exists() else {}
        except Exception:
            processed = {}
        stats["targets"] += 1
        done += 1

        last_done = processed.get(sym)
        if last_done and not any(
                pd.Timestamp(r) > pd.Timestamp(last_done)
                for r in (sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
                          .get("filings", {}).get("recent", {}).get("reportDate", [])[:12] if True else [])
                if r):
            stats["targets"] -= 1
            done -= 1
            continue
        text, _ = filing_statements(cik, overlap_end)
        if not text:
            stats["no_docs"] += 1
            continue
        try:
            vals = llm(model, PROMPT.format(end=overlap_end.date(), text=text))
        except Exception as e:
            print(f"  {sym}: llm error {e}", flush=True)
            continue
        cands = candidates_from_llm(vals)
        passed = {}
        for item in missing:
            for cv in cands.get(item, []):
                if close_enough(cv, row[item]):
                    passed[item] = True
                    item_pass[item] += 1
                    break
        if passed:
            stats["gate_pass"] += 1
        else:
            stats["gate_fail"] += 1
            continue

        # fresh extraction: newest filing strictly after the overlap quarter
        sub = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if not sub:
            continue
        recent = sub.json().get("filings", {}).get("recent", {})
        newest = None
        for form, rdate, fdate in zip(recent.get("form", []), recent.get("reportDate", []),
                                      recent.get("filingDate", [])):
            if form in ("10-Q", "10-K") and rdate and pd.Timestamp(rdate) > overlap_end + pd.Timedelta(days=20):
                if newest is None or pd.Timestamp(rdate) > newest[0]:
                    newest = (pd.Timestamp(rdate), fdate)
        if newest is None:
            continue
        ntext, nfiled = filing_statements(cik, newest[0])
        if not ntext:
            continue
        try:
            nvals = llm(model, PROMPT.format(end=newest[0].date(), text=ntext))
        except Exception:
            continue
        ncands = candidates_from_llm(nvals)

        # assemble features whose inputs are now complete at the NEW period
        got = {}
        for item in LLM_ITEMS:
            if item in passed and ncands.get(item):
                got[item] = ncands[item][0]
            elif resolved.get(item) == "ZERO":
                got[item] = 0.0
            elif resolved.get(item) not in (None, "ZERO"):
                v, _ = spec_value(facts, resolved[item], newest[0], item in FLOWS)
                if v is not None:
                    got[item] = v
        sym_out = {}
        for feat, (inputs, formula) in FEATURES.items():
            if all(it in got for it in inputs):
                fv = formula(got)
                if fv is not None and pd.notna(fv):
                    sym_out[feat] = {"value": round(float(fv), 6),
                                     "period_end": str(newest[0].date()),
                                     "rdq": nfiled, "source": f"llm-{model}"}
        if sym_out:
            overlay[sym] = sym_out
            stats["fresh"] += 1
            try:
                processed = json.load(open(STATE)) if STATE.exists() else {}
            except Exception:
                processed = {}
            processed[sym] = str(newest[0].date())
            json.dump(processed, open(STATE, "w"))
        if done % 5 == 0:
            print(f"  {done} targets ({time.time()-t0:.0f}s) pass={stats['gate_pass']} "
                  f"fail={stats['gate_fail']}", flush=True)

    print(f"\n===== LLM EXTRACT ({model}) {time.time()-t0:.0f}s =====")
    print(f"  targets tried: {stats['targets']} | gate PASS: {stats['gate_pass']} "
          f"| gate FAIL: {stats['gate_fail']} | no docs: {stats['no_docs']}")
    print(f"  per-item gate passes: { {k: v for k, v in item_pass.items() if v} }")
    print(f"  fresh-quarter features emitted: {stats['fresh']} symbols")
    if overlay:
        existing = {}
        if OUT.exists():
            try:
                existing = json.load(open(OUT)).get("features", {})
            except Exception:
                pass
        existing.update(overlay)
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "model": MODELS[model],
                   "features": existing}, open(OUT, "w"), indent=1)
        print(f"  -> {OUT} ({len(existing)} total symbols)")


if __name__ == "__main__":
    main()
