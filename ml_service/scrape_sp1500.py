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


if __name__ == "__main__":
    try:
        scrape()
    except Exception as e:
        log.error("FAILED: %s", e)
        sys.exit(1)
