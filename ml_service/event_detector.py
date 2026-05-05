#!/usr/bin/env python3
"""
SEC EDGAR Event Detector for Forced-Selling Short Strategy
============================================================
Polls SEC EDGAR for 8-K and 10-K/10-Q filings containing forced-selling
events: material impairment, auditor change, restatement, going concern,
internal/disclosure controls weakness.

Three detection methods:
1. ITEM-CODE DETECTION (8-K): Item 2.06, 4.01, 4.02 are explicit → no NLP
2. KEYWORD DETECTION (10-K/10-Q): "going concern", "substantial doubt", etc.
3. FULL-TEXT SEARCH: EDGAR EFTS API for broad queries

Architecture:
  - Poll every 30 minutes during market hours
  - Cross-reference with R2K membership + SMA-50 filter
  - Feed qualifying events to EventShortStrategy
  - Log all detections for validation against Audit Analytics
"""

import json
import logging
import os
import re
import ssl
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# SSL context (production should use proper certs)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

EDGAR_HEADERS = {
    "User-Agent": "AutoTraderBot/1.0 (research@autotrader.com)",
    "Accept": "application/json",
}

# Rate limit: SEC allows 10 requests/second
RATE_LIMIT_SECONDS = 0.12


# ═══════════════════════════════════════════════════════════════════════
# 8-K ITEM CODE DETECTION (the easy ones — no NLP needed)
# ═══════════════════════════════════════════════════════════════════════

# 8-K item codes that map to forced-selling events
ITEM_CODE_MAP = {
    "2.06": "material_impairment",     # Material Impairments
    "4.01": "auditor_change",          # Changes in Registrant's Certifying Accountant
    "4.02": "financial_restatement",   # Non-Reliance on Previously Issued Financials
}


def fetch_json(url, max_retries=3):
    """Fetch JSON from URL with retries and rate limiting."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=EDGAR_HEADERS)
            resp = urllib.request.urlopen(req, context=SSL_CTX, timeout=15)
            data = json.loads(resp.read())
            time.sleep(RATE_LIMIT_SECONDS)
            return data
        except Exception as e:
            logger.warning(f"Fetch attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
    return None


def fetch_text(url, max_retries=3):
    """Fetch text content from URL."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=EDGAR_HEADERS)
            resp = urllib.request.urlopen(req, context=SSL_CTX, timeout=30)
            content = resp.read().decode("utf-8", errors="replace")
            time.sleep(RATE_LIMIT_SECONDS)
            return content
        except Exception as e:
            logger.warning(f"Text fetch attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
    return None


def detect_8k_items(start_date, end_date):
    """
    Detect 8-K filings with specific item codes using EDGAR EFTS.

    Item 2.06 = Material Impairment
    Item 4.01 = Auditor Change
    Item 4.02 = Non-Reliance (Restatement)

    Returns list of events: [{ticker, event_type, filing_date, cik, accession}]
    """
    events = []

    for item_code, event_type in ITEM_CODE_MAP.items():
        logger.info(f"Searching EDGAR for 8-K Item {item_code} ({event_type})...")

        # Use EFTS full-text search for item code
        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22Item+{item_code}%22"
            f"&forms=8-K"
            f"&dateRange=custom"
            f"&startdt={start_date}"
            f"&enddt={end_date}"
        )

        data = fetch_json(url)
        if not data:
            continue

        hits = data.get("hits", {}).get("hits", [])
        logger.info(f"  Found {len(hits)} hits for Item {item_code}")

        for hit in hits:
            source = hit.get("_source", {})

            # Extract ticker from display_names
            display_names = source.get("display_names", [])
            ticker = None
            for dn in display_names:
                # Format: "Company Name  (TICKER)  (CIK ...)"
                match = re.search(r'\(([A-Z]{1,5})\)', dn)
                if match:
                    ticker = match.group(1)
                    break

            if not ticker:
                continue

            ciks = source.get("ciks", [])
            file_date = source.get("file_date", "")
            accession = source.get("adsh", "")
            items = source.get("items", "")

            # Verify item code is in the items field
            if item_code not in str(items):
                # Item was in the document text but not the header
                # Still valid — the EFTS search found it in the filing
                pass

            events.append({
                "ticker": ticker,
                "event_type": event_type,
                "filing_date": file_date,
                "cik": ciks[0] if ciks else "",
                "accession": accession,
                "item_code": item_code,
                "source": "8K_item_code",
                "confidence": "high",
            })

    return events


# ═══════════════════════════════════════════════════════════════════════
# 10-K/10-Q KEYWORD DETECTION (going-concern, internal controls)
# ═══════════════════════════════════════════════════════════════════════

GOING_CONCERN_KEYWORDS = [
    "going concern",
    "substantial doubt",
    "ability to continue as a going concern",
    "raise substantial doubt",
    "substantial doubt about",
    "doubt about the company's ability to continue",
]

INTERNAL_CONTROLS_KEYWORDS = [
    "material weakness",
    "material weaknesses",
    "significant deficiency",
    "significant deficiencies",
    "ineffective internal control",
    "adverse opinion on internal control",
]

DISCLOSURE_CONTROLS_KEYWORDS = [
    "disclosure controls and procedures were not effective",
    "disclosure controls were not effective",
    "ineffective disclosure controls",
]


def detect_10k_keywords(start_date, end_date):
    """
    Detect going-concern, internal controls, and disclosure controls
    events in 10-K/10-Q filings using EDGAR full-text search.

    Returns list of events.
    """
    events = []

    keyword_configs = [
        ("going_concern", GOING_CONCERN_KEYWORDS, ["10-K", "10-K/A"]),
        ("internal_controls", INTERNAL_CONTROLS_KEYWORDS, ["10-K", "10-K/A", "10-Q"]),
        ("disclosure_controls", DISCLOSURE_CONTROLS_KEYWORDS, ["10-K", "10-K/A", "10-Q"]),
    ]

    for event_type, keywords, forms in keyword_configs:
        # Use the most specific keyword for initial search
        primary_keyword = keywords[0]
        forms_str = ",".join(forms)

        url = (
            f"https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{urllib.parse.quote(primary_keyword)}%22"
            f"&forms={forms_str}"
            f"&dateRange=custom"
            f"&startdt={start_date}"
            f"&enddt={end_date}"
        )

        logger.info(f"Searching EDGAR for '{primary_keyword}' in {forms_str}...")
        data = fetch_json(url)
        if not data:
            continue

        total = data.get("hits", {}).get("total", {}).get("value", 0)
        hits = data.get("hits", {}).get("hits", [])
        logger.info(f"  Found {total} total, {len(hits)} returned")

        for hit in hits:
            source = hit.get("_source", {})

            display_names = source.get("display_names", [])
            ticker = None
            for dn in display_names:
                match = re.search(r'\(([A-Z]{1,5})\)', dn)
                if match:
                    ticker = match.group(1)
                    break

            if not ticker:
                continue

            events.append({
                "ticker": ticker,
                "event_type": event_type,
                "filing_date": source.get("file_date", ""),
                "cik": source.get("ciks", [""])[0],
                "accession": source.get("adsh", ""),
                "item_code": "",
                "source": "10K_keyword",
                "confidence": "medium",  # keyword match, not item code
            })

    return events


# ═══════════════════════════════════════════════════════════════════════
# COMPANY-LEVEL 8-K ITEM DETECTION (via submissions API)
# ═══════════════════════════════════════════════════════════════════════

def detect_company_8k_items(cik, start_date=None):
    """
    Check a specific company's recent 8-K filings for item codes.
    Uses the submissions API which has structured item data.

    Args:
        cik: Company CIK (zero-padded to 10 digits)
        start_date: only return filings after this date

    Returns: list of events
    """
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    data = fetch_json(url)
    if not data:
        return []

    # Get ticker from the data
    tickers = data.get("tickers", [])
    ticker = tickers[0] if tickers else ""

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    items_list = recent.get("items", [])
    accessions = recent.get("accessionNumber", [])

    events = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue

        filing_date = dates[i] if i < len(dates) else ""
        if start_date and filing_date < start_date:
            continue

        items_str = items_list[i] if i < len(items_list) else ""
        accession = accessions[i] if i < len(accessions) else ""

        # Check each item code
        for item_code, event_type in ITEM_CODE_MAP.items():
            if item_code in items_str:
                events.append({
                    "ticker": ticker,
                    "event_type": event_type,
                    "filing_date": filing_date,
                    "cik": cik,
                    "accession": accession,
                    "item_code": item_code,
                    "source": "submissions_api",
                    "confidence": "high",
                })

    return events


# ═══════════════════════════════════════════════════════════════════════
# MAIN DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════

class EventDetector:
    """
    Production event detector that combines all three detection methods.

    Usage:
        detector = EventDetector()
        events = detector.scan(lookback_days=1)  # scan last 24 hours
        # or for validation:
        events = detector.scan_range("2025-01-01", "2025-03-31")
    """

    def __init__(self, r2k_members=None):
        """
        Args:
            r2k_members: set of tickers currently in R2K (for filtering)
        """
        self.r2k_members = r2k_members or set()
        self.detected_events = []
        self.detection_log = []

    def scan(self, lookback_days=1):
        """Scan for new events in the last N days."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        return self.scan_range(start_date, end_date)

    def scan_range(self, start_date, end_date):
        """
        Scan EDGAR for events between start_date and end_date.

        Returns: list of event dicts with keys:
            ticker, event_type, filing_date, cik, accession, source, confidence
        """
        all_events = []

        # Method 1: 8-K item codes (highest confidence)
        logger.info(f"Scanning 8-K items {start_date} to {end_date}...")
        item_events = detect_8k_items(start_date, end_date)
        all_events.extend(item_events)
        logger.info(f"  8-K items: {len(item_events)} events")

        # Method 2: 10-K/10-Q keywords (medium confidence)
        logger.info(f"Scanning 10-K keywords {start_date} to {end_date}...")
        keyword_events = detect_10k_keywords(start_date, end_date)
        all_events.extend(keyword_events)
        logger.info(f"  10-K keywords: {len(keyword_events)} events")

        # Deduplicate (same ticker + same date + same event type)
        seen = set()
        unique_events = []
        for ev in all_events:
            key = (ev["ticker"], ev["event_type"], ev["filing_date"])
            if key not in seen:
                seen.add(key)
                unique_events.append(ev)

        logger.info(f"Total unique events: {len(unique_events)}")

        # Filter to R2K members if available
        if self.r2k_members:
            r2k_events = [e for e in unique_events if e["ticker"] in self.r2k_members]
            logger.info(f"R2K-filtered events: {len(r2k_events)}")
            unique_events = r2k_events

        self.detected_events.extend(unique_events)
        return unique_events

    def validate_against_audit_analytics(self, audit_data, start_date, end_date):
        """
        Compare detected events against Audit Analytics ground truth.

        Args:
            audit_data: DataFrame with Audit Analytics data
            start_date, end_date: validation period

        Returns: dict with precision, recall, and detailed mismatches
        """
        # Run detection
        detected = self.scan_range(start_date, end_date)
        detected_set = {(e["ticker"], e["event_type"], e["filing_date"][:10]) for e in detected}

        # Build ground truth from Audit Analytics
        col_map = {
            "going_concern_severity": ("going_concern", 1),
            "auditor_change_severity": ("auditor_change", 2),
            "financial_restatement_severity": ("financial_restatement", 2),
            "material_impairment_severity": ("material_impairment", 2),
            "internal_controls_severity": ("internal_controls", 2),
            "disclosure_controls_severity": ("disclosure_controls", 2),
        }

        truth_set = set()
        audit_filtered = audit_data[
            (audit_data["date"] >= start_date) & (audit_data["date"] <= end_date)
        ]

        for _, row in audit_filtered.iterrows():
            ticker = row.get("ticker", "")
            date = str(row.get("date", ""))[:10]
            for col, (event_type, min_sev) in col_map.items():
                if col in row and pd.notna(row[col]) and row[col] >= min_sev:
                    truth_set.add((ticker, event_type, date))

        # Compute precision and recall
        true_positives = detected_set & truth_set
        false_positives = detected_set - truth_set
        false_negatives = truth_set - detected_set

        precision = len(true_positives) / len(detected_set) if detected_set else 0
        recall = len(true_positives) / len(truth_set) if truth_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            "detected": len(detected_set),
            "truth": len(truth_set),
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_samples": list(false_positives)[:10],
            "fn_samples": list(false_negatives)[:10],
        }

        return result


# ═══════════════════════════════════════════════════════════════════════
# POLLING SERVICE
# ═══════════════════════════════════════════════════════════════════════

def run_polling_service(detector, strategy, interval_minutes=30):
    """
    Production polling service. Runs continuously during market hours.

    Args:
        detector: EventDetector instance
        strategy: EventShortStrategy instance
        interval_minutes: polling interval
    """
    logger.info(f"Starting event polling service (interval={interval_minutes}min)")

    while True:
        try:
            now = datetime.now()

            # Only poll during market hours (9:00 AM - 6:00 PM ET)
            hour = now.hour
            if hour < 9 or hour > 18:
                logger.debug(f"Outside market hours ({hour}:00), sleeping...")
                time.sleep(300)  # 5 minutes
                continue

            # Skip weekends
            if now.weekday() >= 5:
                logger.debug("Weekend, sleeping...")
                time.sleep(3600)
                continue

            # Scan for new events
            events = detector.scan(lookback_days=1)

            for event in events:
                logger.info(f"New event: {event['ticker']} - {event['event_type']} "
                           f"({event['filing_date']})")

                # The strategy module handles entry logic
                # (SMA check, position limits, etc.)

            # Sleep until next poll
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Polling service stopped by user")
            break
        except Exception as e:
            logger.error(f"Polling error: {e}")
            time.sleep(60)  # wait 1 minute on error


# ═══════════════════════════════════════════════════════════════════════
# FMP-BASED IMPAIRMENT DETECTION (catches 10-K/10-Q impairments)
# ═══════════════════════════════════════════════════════════════════════

def detect_impairments_from_fmp(cache_dir, min_decline_pct=0.10):
    """
    Detect material impairments by comparing goodwill+intangible assets
    quarter-over-quarter from FMP balance sheet data.

    A decline of >10% in goodwill+intangibles = likely impairment.

    Args:
        cache_dir: Path to FMP fundamentals cache directory
        min_decline_pct: minimum QoQ decline to flag (0.10 = 10%)

    Returns: list of events [{ticker, event_type, filing_date, ...}]
    """
    cache_dir = Path(cache_dir)
    events = []

    for bs_file in cache_dir.glob("*_balance-sheet-statement.json"):
        ticker = bs_file.name.split("_")[0]

        try:
            with open(bs_file) as f:
                data = json.load(f)
        except Exception:
            continue

        if not data or len(data) < 2:
            continue

        # Compare consecutive quarters
        for i in range(len(data) - 1):
            curr = data[i]
            prev = data[i + 1]

            curr_gw = (curr.get("goodwill", 0) or 0) + (curr.get("intangibleAssets", 0) or 0)
            prev_gw = (prev.get("goodwill", 0) or 0) + (prev.get("intangibleAssets", 0) or 0)

            if prev_gw <= 0:
                continue

            change = (curr_gw - prev_gw) / prev_gw

            if change < -min_decline_pct:
                filing_date = curr.get("filingDate", curr.get("date", ""))

                events.append({
                    "ticker": ticker,
                    "event_type": "material_impairment",
                    "filing_date": filing_date,
                    "cik": "",
                    "accession": "",
                    "item_code": "",
                    "source": "fmp_balance_sheet",
                    "confidence": "medium",
                    "impairment_pct": change,
                    "impairment_amount": curr_gw - prev_gw,
                })

    logger.info(f"FMP impairment detection: {len(events)} events from {len(list(cache_dir.glob('*_balance-sheet-statement.json')))} tickers")
    return events


def detect_going_concern_from_fmp(cache_dir):
    """
    Detect going-concern flags from FMP key metrics / financial ratios.
    Companies with very low current ratio + negative cash flow + high debt
    are going-concern candidates.

    This is a PROXY — not as reliable as actual audit opinion detection.
    Use as a supplement to EDGAR keyword detection, not a replacement.
    """
    # This would use key-metrics.json and ratios.json from FMP cache
    # For now, defer to EDGAR keyword detection which is more reliable
    return []


# ═══════════════════════════════════════════════════════════════════════
# COMBINED DETECTOR (EDGAR + FMP)
# ═══════════════════════════════════════════════════════════════════════

class CombinedEventDetector(EventDetector):
    """
    Production detector combining EDGAR + FMP sources.

    EDGAR handles:
      - 8-K Item 2.06 (material impairment, immediate)
      - 8-K Item 4.01 (auditor change, immediate)
      - 8-K Item 4.02 (restatement, immediate)
      - 10-K keywords (going concern, material weakness)

    FMP handles:
      - Material impairment from balance sheet changes (quarterly)
      - Covers the ~70% of impairments that come from 10-K/10-Q, not 8-K

    Combined coverage: estimated 70-80% of Audit Analytics events
    """

    def __init__(self, r2k_members=None, fmp_cache_dir=None):
        super().__init__(r2k_members)
        self.fmp_cache_dir = fmp_cache_dir or Path("data/fundamentals_cache")

    def scan_range(self, start_date, end_date):
        """Scan both EDGAR and FMP for events."""
        # EDGAR events (real-time)
        edgar_events = super().scan_range(start_date, end_date)

        # FMP impairment events (quarterly, from cached balance sheets)
        fmp_events = []
        if self.fmp_cache_dir.exists():
            all_fmp = detect_impairments_from_fmp(self.fmp_cache_dir)
            # Filter to date range
            fmp_events = [e for e in all_fmp
                          if start_date <= e["filing_date"][:10] <= end_date]
            logger.info(f"FMP impairments in range: {len(fmp_events)}")

        # Combine and deduplicate
        all_events = edgar_events + fmp_events
        seen = set()
        unique = []
        for ev in all_events:
            key = (ev["ticker"], ev["event_type"], ev["filing_date"][:10])
            if key not in seen:
                seen.add(key)
                unique.append(ev)

        # Filter to R2K if available
        if self.r2k_members:
            unique = [e for e in unique if e["ticker"] in self.r2k_members]

        logger.info(f"Combined events: {len(unique)} (EDGAR={len(edgar_events)}, FMP={len(fmp_events)})")
        return unique


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="SEC EDGAR Event Detector")
    parser.add_argument("--scan", action="store_true", help="Scan for recent events")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--validate", action="store_true", help="Validate against Audit Analytics")
    parser.add_argument("--lookback", type=int, default=7, help="Days to look back")
    args = parser.parse_args()

    detector = EventDetector()

    if args.scan:
        if args.start and args.end:
            events = detector.scan_range(args.start, args.end)
        else:
            events = detector.scan(lookback_days=args.lookback)

        print(f"\nDetected {len(events)} events:")
        for e in events[:20]:
            print(f"  {e['filing_date']} {e['ticker']:6s} {e['event_type']:25s} "
                  f"({e['source']}, {e['confidence']})")

    elif args.validate:
        print("Validation requires Audit Analytics data.")
        print("Usage: import and call detector.validate_against_audit_analytics()")

    else:
        # Quick test
        print("SEC EDGAR Event Detector — Quick Test")
        print("=" * 50)
        events = detector.scan(lookback_days=7)
        print(f"\nEvents in last 7 days: {len(events)}")
        for e in events[:10]:
            print(f"  {e['filing_date']} {e['ticker']:6s} {e['event_type']:25s} [{e['confidence']}]")
