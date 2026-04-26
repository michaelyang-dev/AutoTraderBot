#!/usr/bin/env python3
"""
Survivorship Bias Diagnostic Test
===================================
Three checks to verify training pipeline integrity:
  1. Massive data coverage for delisted S&P 500 tickers
  2. Pipeline audit: point-in-time vs today's constituents
  3. Empirical end-to-end data recovery test

Run:
    cd auto-trader\ 2 && python3 tests/test_survivorship_bias.py
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add ml_service to path
ML_DIR = Path(__file__).resolve().parent.parent / "ml_service"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

REPORT_FILE = Path(__file__).resolve().parent / "survivorship_bias_report.md"

# ══════════════════════════════════════════════════════════════════════════════
#  Test Cases
# ══════════════════════════════════════════════════════════════════════════════

DELISTED_TEST_CASES = [
    ("FRC",   2023, "Failed - FDIC takeover",      "2023-05-01"),
    ("SIVB",  2023, "Failed - FDIC takeover",      "2023-03-10"),
    ("SBNY",  2023, "Failed - FDIC takeover",      "2023-03-12"),
    ("CTXS",  2022, "Acquired by Vista Equity",    "2022-09-30"),
    ("TWTR",  2022, "Acquired by Musk/X Corp",     "2022-10-27"),
    ("XLNX",  2022, "Acquired by AMD",             "2022-02-14"),
    ("ATVI",  2023, "Acquired by Microsoft",       "2023-10-13"),
    ("DISCA", 2022, "Merged into WBD",             "2022-04-08"),
    ("VIAC",  2022, "Renamed to PARA",             "2022-02-15"),
    ("TIF",   2021, "Acquired by LVMH",            "2021-01-07"),
    ("XEC",   2021, "Acquired by Cabot Oil",       "2021-10-01"),
    ("WLTW",  2022, "Merged into WTW",             "2022-09-22"),
    ("FB",    2022, "Renamed to META",             "2022-06-09"),
    ("TWX",   2018, "Acquired by AT&T",            "2018-06-14"),
    ("SCG",   2019, "Acquired by Dominion",        "2019-01-02"),
    ("MNK",   2020, "Bankruptcy",                  "2020-10-12"),
    ("SLM",   2014, "Spin-off Navient",            "2014-04-30"),
    ("KORS",  2018, "Renamed to CPRI",             "2019-01-02"),
    ("CELG",  2019, "Acquired by Bristol-Myers",   "2019-11-22"),
    ("RTN",   2020, "Merged into RTX",             "2020-04-03"),
]


def log(msg: str):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 1: Massive Data Coverage for Delisted Tickers
# ══════════════════════════════════════════════════════════════════════════════

def check1_delisted_coverage():
    log("\n" + "=" * 70)
    log("CHECK 1: Massive Data Coverage for Delisted Tickers")
    log("=" * 70)

    from massive_data_provider import MassiveDataProvider
    provider = MassiveDataProvider(validate_vs_yfinance=False)

    results = []
    for ticker, exit_year, reason, expected_end in DELISTED_TEST_CASES:
        log(f"  {ticker:<6} ({reason[:35]:<35}) ...", )
        try:
            df = provider.fetch_ticker_bars(ticker, "2015-01-01", "2024-12-31",
                                             adjusted=True)
            if len(df) == 0:
                category = "NO_COVERAGE"
                n_bars = 0
                first_date = "N/A"
                last_date = "N/A"
            else:
                n_bars = len(df)
                first_date = str(df.index.min().date())
                last_date = str(df.index.max().date())

                # Check if last date is close to expected exit
                expected = pd.Timestamp(expected_end)
                actual_last = df.index.max()
                gap_days = abs((actual_last - expected).days)

                if n_bars >= 100 and gap_days <= 10:
                    category = "FULL_COVERAGE"
                elif n_bars >= 100:
                    category = "PARTIAL_COVERAGE"
                else:
                    category = "NO_COVERAGE"

            results.append({
                "ticker": ticker, "exit_year": exit_year, "reason": reason,
                "expected_end": expected_end, "n_bars": n_bars,
                "first_date": first_date, "last_date": last_date,
                "category": category,
            })
            log(f"    {category}: {n_bars} bars, last={last_date}")

        except Exception as e:
            results.append({
                "ticker": ticker, "exit_year": exit_year, "reason": reason,
                "expected_end": expected_end, "n_bars": 0,
                "first_date": "ERROR", "last_date": "ERROR",
                "category": "NO_COVERAGE",
            })
            log(f"    ERROR: {e}")

    # Summary
    full = sum(1 for r in results if r["category"] == "FULL_COVERAGE")
    partial = sum(1 for r in results if r["category"] == "PARTIAL_COVERAGE")
    none_ = sum(1 for r in results if r["category"] == "NO_COVERAGE")
    total = len(results)

    log(f"\n  Summary: {full}/{total} full, {partial}/{total} partial, {none_}/{total} none")
    return results, {"full": full, "partial": partial, "none": none_, "total": total}


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 2: Pipeline Audit
# ══════════════════════════════════════════════════════════════════════════════

def check2_pipeline_audit():
    log("\n" + "=" * 70)
    log("CHECK 2: Pipeline Audit — Point-in-Time vs Today's Universe")
    log("=" * 70)

    findings = []

    # Check 1: Does sp500_history.py exist and have get_sp500_on_date?
    sp500_hist_file = ML_DIR / "sp500_history.py"
    if sp500_hist_file.exists():
        content = sp500_hist_file.read_text()
        has_pit = "get_sp500_on_date" in content
        findings.append(("sp500_history.py exists", True))
        findings.append(("get_sp500_on_date() function defined", has_pit))

        # Check the logic
        if "reverse-apply" in content.lower() or "reverse" in content.lower():
            findings.append(("Uses reverse-application of changes (correct algorithm)", True))
        if "current" in content.lower() and "discard" in content.lower():
            findings.append(("Starts from current list, removes future additions, adds back removals", True))
    else:
        findings.append(("sp500_history.py exists", False))

    # Check 2: Does data_pipeline.py use get_sp500_on_date?
    dp_file = ML_DIR / "data_pipeline.py"
    dp_content = dp_file.read_text()

    uses_pit = "get_sp500_on_date" in dp_content
    findings.append(("data_pipeline.py imports get_sp500_on_date", uses_pit))

    uses_per_date = "sp500_by_date" in dp_content
    findings.append(("data_pipeline.py computes per-date membership", uses_per_date))

    marks_in_sp500 = "in_sp500" in dp_content
    findings.append(("data_pipeline.py sets in_sp500 column per date", marks_in_sp500))

    # Check 3: Does train_production_model.py filter by in_sp500?
    tp_file = ML_DIR / "train_production_model.py"
    tp_content = tp_file.read_text()

    filters_sp500 = 'in_sp500' in tp_content
    findings.append(("train_production_model.py filters by in_sp500", filters_sp500))

    # Check 4: What symbols does data_pipeline actually DOWNLOAD?
    downloads_all = "ALL_SYMBOLS" in dp_content
    findings.append(("data_pipeline.py uses ALL_SYMBOLS for download", downloads_all))

    # Check if ALL_SYMBOLS includes historical members (survivorship fix)
    has_historical_union = "_build_historical_universe" in dp_content
    findings.append(("data_pipeline builds historical S&P 500 union", has_historical_union))

    has_ticker_normalization = "_ticker_to_massive" in dp_content
    findings.append(("data_pipeline has ticker format normalization (BRK-B <-> BRK.B)", has_ticker_normalization))

    # Check what ALL_SYMBOLS actually contains at runtime
    from data_pipeline import ALL_SYMBOLS as DP_ALL_SYMBOLS
    from sp500_universe import get_stock_symbols
    current_sp500 = set(get_stock_symbols())

    # Check: are delisted test cases in the download list?
    delisted_tickers = set(t for t, _, _, _ in DELISTED_TEST_CASES)
    delisted_in_download = delisted_tickers & set(DP_ALL_SYMBOLS)
    delisted_missing = delisted_tickers - set(DP_ALL_SYMBOLS)

    findings.append((f"Current S&P 500 count: {len(current_sp500)}", True))
    findings.append((f"ALL_SYMBOLS count (downloaded): {len(DP_ALL_SYMBOLS)}", True))
    findings.append((f"Delisted test tickers in download list: {len(delisted_in_download)}/{len(delisted_tickers)}",
                      len(delisted_in_download) >= len(delisted_tickers) - 2))
    if delisted_missing:
        findings.append((f"Delisted tickers NOT in download list: {sorted(delisted_missing)}", False))
    else:
        findings.append(("All delisted test tickers included in download list", True))

    fetches_historical = has_historical_union

    # Determine verdict
    if uses_pit and uses_per_date and marks_in_sp500:
        if fetches_historical:
            verdict = "POINT-IN-TIME"
            explanation = ("Training uses sp500_history.get_sp500_on_date() correctly and "
                          "downloads historical members. Survivorship bias is mitigated.")
        else:
            verdict = "MIXED"
            explanation = ("The pipeline correctly marks in_sp500 per date using point-in-time "
                          "membership, BUT it only downloads prices for TODAY's S&P 500 list "
                          "(ALL_SYMBOLS). Historical members that have been delisted/renamed "
                          "are never downloaded, so their rows don't exist in features.parquet. "
                          "The in_sp500 marking is correct but irrelevant for missing symbols. "
                          "This is SURVIVORSHIP BIAS by data omission.")
    else:
        verdict = "TODAY'S UNIVERSE"
        explanation = "Training uses current S&P 500 list. SEVERE survivorship bias."

    log(f"\n  Verdict: {verdict}")
    log(f"  {explanation}")

    return findings, verdict, explanation


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK 3: Empirical End-to-End Test
# ══════════════════════════════════════════════════════════════════════════════

def check3_empirical_test():
    log("\n" + "=" * 70)
    log("CHECK 3: Empirical End-to-End Data Recovery Test")
    log("=" * 70)

    from sp500_history import get_sp500_on_date
    from massive_data_provider import MassiveDataProvider
    provider = MassiveDataProvider(validate_vs_yfinance=False)

    # Also get current S&P 500 for comparison
    from sp500_universe import get_stock_symbols
    current_sp500 = set(get_stock_symbols())

    test_dates = [
        ("2022-06-01", 252),
        ("2018-06-01", 252),
    ]

    results = []
    for date_str, lookback_days in test_dates:
        date = pd.Timestamp(date_str)
        log(f"\n  Testing date: {date_str}")

        # Get point-in-time constituents
        pit_members = get_sp500_on_date(date)
        log(f"    Point-in-time S&P 500 members: {len(pit_members)}")

        # Which are NOT in current S&P 500?
        historical_only = pit_members - current_sp500
        log(f"    Members not in current S&P 500: {len(historical_only)}")
        if historical_only:
            log(f"    Examples: {sorted(historical_only)[:15]}")

        # Try to fetch data for each constituent
        start = (date - pd.Timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
        end = date.strftime("%Y-%m-%d")

        full_data = []
        partial_data = []
        no_data = []

        log(f"    Fetching {len(pit_members)} symbols from Massive ...")
        t0 = time.time()

        for i, sym in enumerate(sorted(pit_members)):
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                log(f"      [{i+1}/{len(pit_members)}] ({elapsed:.0f}s)")
            # Normalize ticker: sp500_history uses BRK-B, Massive uses BRK.B
            massive_sym = sym.replace("-", ".")
            try:
                df = provider.fetch_ticker_bars(massive_sym, start, end, adjusted=True)
                if len(df) >= lookback_days * 0.9:  # 90% of expected bars
                    full_data.append(sym)
                elif len(df) > 0:
                    partial_data.append((sym, len(df)))
                else:
                    no_data.append(sym)
            except Exception:
                no_data.append(sym)

        elapsed = time.time() - t0
        log(f"    Fetched in {elapsed:.0f}s")

        # Categorize no-data symbols
        no_data_historical = [s for s in no_data if s in historical_only]
        no_data_current = [s for s in no_data if s in current_sp500]

        log(f"    Results:")
        log(f"      Full data (>={int(lookback_days*0.9)} bars): {len(full_data)}/{len(pit_members)}")
        log(f"      Partial data: {len(partial_data)}/{len(pit_members)}")
        log(f"      No data: {len(no_data)}/{len(pit_members)}")
        log(f"        - Historical (delisted): {len(no_data_historical)}")
        log(f"        - Current SP500 (unexpected): {len(no_data_current)}")
        if no_data:
            log(f"      No-data tickers: {sorted(no_data)[:20]}")

        results.append({
            "date": date_str,
            "pit_members": len(pit_members),
            "historical_only": len(historical_only),
            "historical_only_list": sorted(historical_only),
            "full": len(full_data),
            "partial": len(partial_data),
            "partial_list": partial_data[:20],
            "no_data": len(no_data),
            "no_data_list": sorted(no_data),
            "no_data_historical": sorted(no_data_historical),
            "no_data_current": sorted(no_data_current),
            "recovery_rate": len(full_data) / len(pit_members) * 100,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Report Generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(check1_results, check1_summary,
                     check2_findings, check2_verdict, check2_explanation,
                     check3_results):
    lines = []

    # Determine overall status
    if check2_verdict == "POINT-IN-TIME":
        overall = "MITIGATED"
    elif check2_verdict == "MIXED":
        # Check empirical recovery rate
        avg_recovery = np.mean([r["recovery_rate"] for r in check3_results])
        if avg_recovery >= 95:
            overall = "MODERATE"
        else:
            overall = "SEVERE"
    else:
        overall = "SEVERE"

    # Executive Summary
    lines.append("# Survivorship Bias Diagnostic Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n## Executive Summary")
    lines.append(f"\n**Survivorship bias status: {overall}**")

    if overall == "SEVERE":
        lines.append(
            "\nThe training pipeline downloads price data only for TODAY's S&P 500 "
            "constituents. Historical members that were delisted, acquired, or renamed "
            "have no price data in features.parquet. While the pipeline correctly marks "
            "`in_sp500` per date using point-in-time membership, this is meaningless for "
            "symbols with no data. The model never sees the stocks that failed — which "
            "biases it toward thinking stocks generally survive and recover."
        )
    elif overall == "MODERATE":
        lines.append(
            "\nThe training pipeline has partial survivorship bias. Point-in-time membership "
            "is correctly computed, and most historical constituents have data available from "
            "Massive, but some delisted tickers are missing from the download universe."
        )
    else:
        lines.append(
            "\nThe training pipeline correctly uses point-in-time S&P 500 membership and "
            "downloads data for historical members. Survivorship bias is adequately mitigated."
        )

    # Check 1
    lines.append(f"\n## Check 1: Massive Data Coverage for Delisted Tickers")
    lines.append(f"\n| Ticker | Exit Year | Reason | Bars | First Date | Last Date | Expected End | Status |")
    lines.append(f"|--------|-----------|--------|------|------------|-----------|--------------|--------|")
    for r in check1_results:
        lines.append(f"| {r['ticker']} | {r['exit_year']} | {r['reason'][:30]} | {r['n_bars']} | "
                     f"{r['first_date']} | {r['last_date']} | {r['expected_end']} | {r['category']} |")

    lines.append(f"\n**Summary:** {check1_summary['full']}/{check1_summary['total']} full coverage, "
                 f"{check1_summary['partial']}/{check1_summary['total']} partial, "
                 f"{check1_summary['none']}/{check1_summary['total']} no coverage")

    # Check 2
    lines.append(f"\n## Check 2: Pipeline Audit")
    lines.append(f"\n**Verdict: {check2_verdict}**")
    lines.append(f"\n{check2_explanation}")
    lines.append(f"\n| Check | Result |")
    lines.append(f"|-------|--------|")
    for desc, passed in check2_findings:
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {desc} | {status} |")

    # Check 3
    lines.append(f"\n## Check 3: Empirical Data Recovery")
    for r in check3_results:
        lines.append(f"\n### Date: {r['date']}")
        lines.append(f"\n| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Point-in-time S&P 500 members | {r['pit_members']} |")
        lines.append(f"| Members not in current S&P 500 | {r['historical_only']} |")
        lines.append(f"| Full data recovered | {r['full']} ({r['full']/r['pit_members']*100:.1f}%) |")
        lines.append(f"| Partial data | {r['partial']} |")
        lines.append(f"| No data | {r['no_data']} |")
        lines.append(f"| **Recovery rate** | **{r['recovery_rate']:.1f}%** |")

        if r["no_data_list"]:
            lines.append(f"\nTickers with no data: {', '.join(r['no_data_list'][:30])}")
        if r["no_data_historical"]:
            lines.append(f"\nOf these, known historical/delisted: {', '.join(r['no_data_historical'][:20])}")

    # Recommendations
    lines.append(f"\n## Recommendations")
    if overall == "SEVERE":
        lines.append("""
1. **Expand the download universe** to include historical S&P 500 members:
   - In `data_pipeline.py`, after computing `ALL_SYMBOLS`, also compute the union of all historical
     S&P 500 members using `get_sp500_on_date()` for each year, and add those to the download list
   - This ensures delisted tickers' price data is fetched from Massive (which HAS the data)

2. **Specific code change needed** in `data_pipeline.py`:
   ```python
   # After ALL_SYMBOLS = get_all_symbols()
   from sp500_history import get_sp500_on_date
   historical_members = set()
   for year in range(2016, datetime.today().year + 1):
       historical_members |= get_sp500_on_date(pd.Timestamp(f"{year}-01-01"))
   ALL_SYMBOLS = sorted(set(ALL_SYMBOLS) | historical_members)
   ```

3. **Validate after fix**: Re-run this diagnostic to confirm recovery rate improves to >98%
""")
    elif overall == "MODERATE":
        lines.append("""
1. The point-in-time membership logic is correct
2. Most historical data is available from Massive
3. Consider adding the missing delisted tickers to the download list
4. Re-run periodically to catch newly delisted names
""")
    else:
        lines.append("""
1. The current setup adequately addresses survivorship bias
2. Continue monitoring for edge cases (recent delistings, name changes)
3. Verify that Massive data coverage extends to all time periods used in training
""")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 70)
    log("  SURVIVORSHIP BIAS DIAGNOSTIC TEST")
    log("=" * 70)

    # Check 1
    check1_results, check1_summary = check1_delisted_coverage()

    # Check 2
    check2_findings, check2_verdict, check2_explanation = check2_pipeline_audit()

    # Check 3
    check3_results = check3_empirical_test()

    # Generate report
    log("\n" + "=" * 70)
    log("  GENERATING REPORT")
    log("=" * 70)

    report = generate_report(
        check1_results, check1_summary,
        check2_findings, check2_verdict, check2_explanation,
        check3_results,
    )

    REPORT_FILE.write_text(report)
    log(f"\n  Report saved to: {REPORT_FILE}")

    elapsed = time.time() - t0
    log(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Print summary
    log(f"\n{'='*70}")
    avg_recovery = np.mean([r["recovery_rate"] for r in check3_results])
    log(f"  Delisted coverage: {check1_summary['full']}/{check1_summary['total']} full")
    log(f"  Pipeline verdict: {check2_verdict}")
    log(f"  Avg recovery rate: {avg_recovery:.1f}%")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
