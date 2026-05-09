#!/usr/bin/env python3
"""
XBRL-Based Impairment Detector
================================
Detects material impairments from SEC XBRL structured data.
Uses the Company Facts API — no text parsing needed.

This recovers ~80% of material impairment events that the
8-K Item 2.06 search misses (because most impairments are
disclosed in 10-K/10-Q, not 8-K).
"""

import json
import logging
import ssl
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE
HEADERS = {
    "User-Agent": "AutoTraderBot/1.0 research@autotrader.com",
    "Accept": "application/json",
}

# XBRL tags that indicate impairment charges
IMPAIRMENT_TAGS = [
    "GoodwillImpairmentLoss",
    "AssetImpairmentCharges",
    "ImpairmentOfIntangibleAssetsExcludingGoodwill",
    "ImpairmentOfLongLivedAssetsHeldForUse",
    "ImpairmentOfRealEstate",
    "GoodwillAndIntangibleAssetImpairment",
    "OtherAssetImpairmentCharges",
]

# Minimum impairment to flag (in dollars)
MIN_IMPAIRMENT_USD = 1_000_000  # $1M


def fetch_json(url, max_retries=3):
    """Fetch JSON with retries and rate limiting."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, context=SSL_CTX, timeout=15)
            data = json.loads(resp.read())
            time.sleep(0.12)  # SEC rate limit: 10 req/sec
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                logger.warning(f"Failed to fetch {url}: {e}")
    return None


def load_ticker_to_cik():
    """Load SEC ticker-to-CIK mapping."""
    cache_path = Path("data/sec_ticker_cik.json")

    # Try cache first
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    # Fetch from SEC
    url = "https://www.sec.gov/files/company_tickers.json"
    data = fetch_json(url)
    if not data:
        return {}

    mapping = {}
    for entry in data.values():
        ticker = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker:
            mapping[ticker] = cik

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(mapping, f)

    logger.info(f"Loaded {len(mapping)} ticker-CIK mappings")
    return mapping


def detect_impairments(cik, start_date=None, min_amount=MIN_IMPAIRMENT_USD):
    """
    Detect material impairments for a single company using XBRL Company Facts.

    Args:
        cik: Company CIK (zero-padded string)
        start_date: only return events after this date (YYYY-MM-DD)
        min_amount: minimum impairment value in USD

    Returns: list of event dicts
    """
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"

    data = fetch_json(url)
    if not data:
        return []

    company_name = data.get("entityName", "")
    us_gaap = data.get("facts", {}).get("us-gaap", {})

    events = []
    seen = set()  # deduplicate by (tag, period_end, form)

    for tag in IMPAIRMENT_TAGS:
        if tag not in us_gaap:
            continue

        usd_entries = us_gaap[tag].get("units", {}).get("USD", [])

        for entry in usd_entries:
            val = entry.get("val", 0)
            if val < min_amount:
                continue

            filed = entry.get("filed", "")
            end_date = entry.get("end", "")
            form = entry.get("form", "")
            accession = entry.get("accn", "")

            # Filter by date
            if start_date and filed < start_date:
                continue

            # Only 10-K, 10-Q, 8-K
            if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A", "8-K"):
                continue

            # Deduplicate
            key = (tag, end_date, form)
            if key in seen:
                continue
            seen.add(key)

            events.append({
                "cik": cik,
                "company": company_name,
                "tag": tag,
                "amount": val,
                "filing_date": filed,
                "period_end": end_date,
                "form": form,
                "accession": accession,
                "source": "xbrl",
                "event_type": "material_impairment",
                "confidence": "high",
            })

    return events


def scan_universe(tickers, start_date=None, min_amount=MIN_IMPAIRMENT_USD):
    """
    Scan a list of tickers for impairment events.

    Args:
        tickers: list of ticker symbols
        start_date: only return events after this date
        min_amount: minimum impairment value

    Returns: list of event dicts with ticker added
    """
    ticker_to_cik = load_ticker_to_cik()

    all_events = []
    scanned = 0
    found = 0

    for ticker in tickers:
        cik = ticker_to_cik.get(ticker.upper())
        if not cik:
            continue

        events = detect_impairments(cik, start_date, min_amount)
        for e in events:
            e["ticker"] = ticker

        if events:
            found += 1
            all_events.extend(events)

        scanned += 1
        if scanned % 100 == 0:
            logger.info(f"XBRL scan: {scanned} tickers, {found} with impairments, "
                        f"{len(all_events)} total events")

    logger.info(f"XBRL scan complete: {scanned} tickers, {found} with impairments, "
                f"{len(all_events)} total events")
    return all_events


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, help="Single ticker to check")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date")
    args = parser.parse_args()

    if args.ticker:
        tcm = load_ticker_to_cik()
        cik = tcm.get(args.ticker.upper())
        if cik:
            events = detect_impairments(cik, args.start)
            print(f"\n{args.ticker} impairments since {args.start}:")
            for e in events:
                print(f"  {e['filing_date']} {e['form']:5s} {e['tag']:40s} ${e['amount']/1e6:,.0f}M")
        else:
            print(f"CIK not found for {args.ticker}")
    else:
        print("Usage: python xbrl_detector.py --ticker AAPL --start 2024-01-01")
