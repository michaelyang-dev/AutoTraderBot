#!/usr/bin/env python3
"""
S&P 1500 Constituent Scraper
==============================
Scrapes current SP500/SP400/SP600 membership from Wikipedia.
Saves to data/sp1500_members.json for the live signal server.

Run weekly via cron on AWS:
    0 8 * * 0  cd /home/ubuntu/auto-trader/ml_service && python3 scrape_sp1500.py

Wikipedia updates these lists within hours of S&P announcements.
"""

import json
import logging
import re
import ssl
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("scrape_sp1500")

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = DATA_DIR / "sp1500_members.json"

# SSL context: try default first, fall back to unverified if Wikipedia blocks
try:
    _SSL_CTX = ssl.create_default_context()
    # Test with a simple request
    urllib.request.urlopen("https://en.wikipedia.org/robots.txt",
                           context=_SSL_CTX, timeout=5)
except Exception:
    _SSL_CTX = ssl.create_default_context()
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE
    log.warning("Using unverified SSL (default context failed for Wikipedia)")

URLS = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

EXPECTED_COUNTS = {"sp500": (490, 510), "sp400": (390, 410), "sp600": (590, 620)}


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
        return resp.read().decode()


def _tickers_from_table(table_html: str) -> list[str]:
    """First-column tickers from a table body. Rows are matched as <tr ...> WITH
    attributes — Wikipedia's 2026-07 markup change added row attributes, and the old
    bare '<tr>' pattern silently matched ZERO rows (broke sp400 7/2, sp500 7/6)."""
    tickers = []
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL):
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
        if cells:
            ticker = re.sub(r'<[^>]+>', '', cells[0]).strip()
            ticker = re.sub(r'\[.*?\]', '', ticker).strip()
            if ticker and re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker):
                tickers.append(ticker)
    return sorted(set(tickers))


def _extract_constituents(html: str) -> list[str]:
    """All three index pages now carry id="constituents" on the main table; try that
    first, then fall back to scanning every wikitable for a plausibly-sized result."""
    m = re.search(r'<table[^>]*id="constituents"[^>]*>(.*?)</table>', html, re.DOTALL)
    if m:
        tickers = _tickers_from_table(m.group(1))
        if len(tickers) > 100:
            return tickers
    for table in re.findall(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>',
                            html, re.DOTALL):
        tickers = _tickers_from_table(table)
        if len(tickers) > 100:
            return tickers
    raise ValueError("Could not find constituents table")


def scrape() -> dict:
    # Per-index resilience: one broken page must NOT freeze the whole file (the
    # 2026-07 format change froze sp1500_members.json for 5 days because a single
    # sp400 failure aborted all three). On failure, reuse that index's last-good
    # list from the existing members file and log loudly.
    previous = {}
    if OUTPUT_FILE.exists():
        try:
            previous = json.load(open(OUTPUT_FILE))
        except Exception:
            pass

    results = {}
    failures = []
    for name, url in URLS.items():
        log.info("Fetching %s from Wikipedia...", name)
        try:
            html = _fetch(url)
            tickers = _extract_constituents(html)
            lo, hi = EXPECTED_COUNTS[name]
            if not (lo <= len(tickers) <= hi):
                raise ValueError(
                    f"got {len(tickers)} tickers, expected {lo}-{hi}. "
                    "Wikipedia format may have changed."
                )
            results[name] = tickers
            log.info("  %s: %d tickers", name, len(tickers))
        except Exception as e:
            last_good = previous.get(name) or []
            if not last_good:
                raise  # no fallback available — keep the old fail-loud behavior
            failures.append(name)
            results[name] = last_good
            log.error("FAILED %s (%s) — reusing %d last-good tickers from %s",
                      name, e, len(last_good), previous.get("updated", "?"))

    total = sum(len(v) for v in results.values())
    log.info("Total SP1500: %d%s", total,
             f" (STALE: {', '.join(failures)})" if failures else "")

    data = {"updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stale": failures, **results}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    log.info("Saved to %s", OUTPUT_FILE)

    return data


def update_sector_map(fmp_api_key: str) -> None:
    """Fill missing sectors in cache_sectors.json from FMP profiles.

    Called after scrape() so any new SP1500 members get sector mappings.
    """
    sector_file = DATA_DIR / "cache_sectors.json"
    members_file = DATA_DIR / "sp1500_members.json"

    if not members_file.exists():
        log.warning("No sp1500_members.json — skipping sector update")
        return

    sectors = {}
    if sector_file.exists():
        with open(sector_file) as f:
            sectors = json.load(f)

    with open(members_file) as f:
        sp1500 = json.load(f)

    all_tickers = set(sp1500["sp500"] + sp1500["sp400"] + sp1500["sp600"])
    missing = sorted(t for t in all_tickers if t not in sectors)

    if not missing:
        log.info("Sector map complete — all %d symbols covered", len(sectors))
        return

    log.info("Fetching sectors for %d new symbols from FMP...", len(missing))
    import time

    fetched = 0
    for sym in missing:
        url = (
            f"https://financialmodelingprep.com/stable/profile"
            f"?symbol={sym}&apikey={fmp_api_key}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
            with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
            if isinstance(data, list) and data:
                sector = data[0].get("sector", "")
                if sector:
                    sectors[sym] = sector
                    fetched += 1
        except Exception:
            pass
        time.sleep(0.1)

    with open(sector_file, "w") as f:
        json.dump(sectors, f, indent=2)

    still_missing = sum(1 for t in all_tickers if t not in sectors)
    log.info("Sector update done: %d fetched, %d total, %d still missing",
             fetched, len(sectors), still_missing)


if __name__ == "__main__":
    import os
    try:
        scrape()

        # Auto-update sector map if FMP key is available
        # Load .env via dotenv if available, fall back to manual parse
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        except ImportError:
            pass
        fmp_key = os.environ.get("FMP_API_KEY", "")
        if fmp_key:
            update_sector_map(fmp_key)
        else:
            log.warning("No FMP_API_KEY — skipping sector map update")
    except Exception as e:
        log.error("FAILED: %s", e)
        sys.exit(1)
