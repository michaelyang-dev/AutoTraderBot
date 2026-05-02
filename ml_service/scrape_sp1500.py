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

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

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


def _extract_sp500(html: str) -> list[str]:
    """SP500 page has a table with id='constituents'."""
    table_match = re.search(
        r'<table[^>]*id="constituents"[^>]*>(.*?)</table>', html, re.DOTALL
    )
    if not table_match:
        raise ValueError("Could not find SP500 constituents table")

    tickers = []
    for row in re.findall(r'<tr>(.*?)</tr>', table_match.group(1), re.DOTALL):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if cells:
            ticker = re.sub(r'<[^>]+>', '', cells[0]).strip()
            ticker = re.sub(r'\[.*?\]', '', ticker).strip()
            if ticker and re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker):
                tickers.append(ticker)
    return sorted(set(tickers))


def _extract_wikitable(html: str) -> list[str]:
    """SP400/SP600 pages use a standard wikitable."""
    tables = re.findall(
        r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>', html, re.DOTALL
    )
    for table in tables:
        tickers = []
        for row in re.findall(r'<tr>(.*?)</tr>', table, re.DOTALL):
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if cells:
                ticker = re.sub(r'<[^>]+>', '', cells[0]).strip()
                ticker = re.sub(r'\[.*?\]', '', ticker).strip()
                if ticker and re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', ticker):
                    tickers.append(ticker)
        if len(tickers) > 100:
            return sorted(set(tickers))
    raise ValueError("Could not find constituents wikitable")


def scrape() -> dict:
    results = {}

    for name, url in URLS.items():
        log.info("Fetching %s from Wikipedia...", name)
        html = _fetch(url)

        if name == "sp500":
            tickers = _extract_sp500(html)
        else:
            tickers = _extract_wikitable(html)

        lo, hi = EXPECTED_COUNTS[name]
        if not (lo <= len(tickers) <= hi):
            raise ValueError(
                f"{name}: got {len(tickers)} tickers, expected {lo}-{hi}. "
                "Wikipedia format may have changed."
            )

        results[name] = tickers
        log.info("  %s: %d tickers", name, len(tickers))

    total = sum(len(v) for v in results.values())
    log.info("Total SP1500: %d", total)

    data = {"updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **results}

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
        fmp_key = os.environ.get("FMP_API_KEY", "")
        if not fmp_key:
            env_file = Path(__file__).resolve().parent.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("FMP_API_KEY="):
                        fmp_key = line.split("=", 1)[1].strip()
        if fmp_key:
            update_sector_map(fmp_key)
        else:
            log.warning("No FMP_API_KEY — skipping sector map update")
    except Exception as e:
        log.error("FAILED: %s", e)
        sys.exit(1)
