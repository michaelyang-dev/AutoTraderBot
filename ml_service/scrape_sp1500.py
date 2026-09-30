#!/usr/bin/env python3
"""
S&P 1500 Constituent Scraper
==============================
Current SP500/SP400/SP600 membership: SSGA ETF holdings (SPY/MDY/SPSM) first, Wikipedia fallback.
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

def _ssl_context():
    """Verified SSL unless certificate verification ITSELF fails. Until 2026-09-30 the probe sent Python's
    default User-Agent, Wikipedia answered 403, and ANY exception switched every fetch — including the ETF
    holdings that define the tradable universe — to an unverified context. Only an SSL error does now."""
    ctx = ssl.create_default_context()
    try:
        urllib.request.urlopen(urllib.request.Request("https://en.wikipedia.org/robots.txt",
                                                      headers={"User-Agent": "Mozilla/5.0"}),
                               context=ctx, timeout=10)
    except Exception as e:
        if isinstance(e, ssl.SSLError) or isinstance(getattr(e, "reason", None), ssl.SSLError):
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            log.warning("Using unverified SSL (certificate verification failed: %s)", e)
        else:
            log.info("SSL probe inconclusive (%s: %s) — keeping certificate verification ON", type(e).__name__, e)
    return ctx


_SSL_CTX = _ssl_context()

URLS = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}

EXPECTED_COUNTS = {"sp500": (490, 510), "sp400": (390, 410), "sp600": (590, 620)}

# PRIMARY SOURCE (2026-09-30): the daily holdings of State Street's full-replication index ETFs. Wikipedia's
# S&P 600 page had not applied the September 2026 rebalance on 2026-09-29 (15 adds / 11 deletes missing vs
# SPSM's 09-28 holdings; its CWEN.A is the wrong share class — the index holds CWEN). That changed live
# picks: with the ETF-based list, ATRC and AXTI replace LGND and VICR in the 2026-09-29 BUY list. Its S&P
# 500 and 400 pages matched SPY / MDY exactly. ETF holdings switch on the index's effective date (the fund
# rebalances at the prior close), so a 06:00 scrape sees that morning's membership. Wikipedia remains the
# fallback, then the last-good list. Parsed with the standard library (no xlsx dependency on the box).
SSGA_ETF = {"sp500": "spy", "sp400": "mdy", "sp600": "spsm"}
SSGA_URL = ("https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/"
            "holdings-daily-us-en-{etf}.xlsx")
MIN_OVERLAP = 0.90       # an ETF list sharing < 90% of names with the reference list is treated as a bad parse
TICKER_RE = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
        return resp.read().decode()


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
        return resp.read()


def _xlsx_rows(data: bytes):
    """Yield {column_letter: cell_text} for each row of the first worksheet. Standard library only."""
    import io
    import xml.etree.ElementTree as ET
    import zipfile
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{ns}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{ns}t")))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    for row in ET.fromstring(z.read(sheet)).iter(f"{ns}row"):
        cells = {}
        for c in row.iter(f"{ns}c"):
            col = re.match(r"[A-Z]+", c.get("r", "")).group(0) if c.get("r") else None
            v, t = c.find(f"{ns}v"), c.get("t")
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{ns}t"))
            else:
                val = v.text if v is not None else None
            if col:
                cells[col] = val
        yield cells


def _ssga_constituents(data: bytes) -> tuple:
    """(sorted tickers, 'As of ...' text) from an SSGA daily-holdings xlsx. Cash, money-market, futures,
    rights and earn-out lines have non-ticker symbols ('-', 'RTYZ6', '2200963D') and are dropped by
    TICKER_RE; share classes are normalised to the dot form the rest of the system uses (BRK.B, MOG.A)."""
    rows = list(_xlsx_rows(data))
    as_of = next((v for r in rows[:8] for v in r.values() if v and "As of" in v), "")
    hdr = next(i for i, r in enumerate(rows[:25]) if "Ticker" in r.values())
    col = next(k for k, v in rows[hdr].items() if v == "Ticker")
    out = set()
    for r in rows[hdr + 1:]:
        s = (r.get(col) or "").strip().upper().replace("/", ".").replace(" ", ".").replace("-", ".")
        if TICKER_RE.match(s):
            out.add(s)
    return sorted(out), as_of.strip()


def _overlap(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(len(a), len(b), 1)


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
    sources = {}
    for name, url in URLS.items():
        lo, hi = EXPECTED_COUNTS[name]
        wiki = None
        log.info("Fetching %s from Wikipedia...", name)
        try:
            html = _fetch(url)
            tickers = _extract_constituents(html)
            if not (lo <= len(tickers) <= hi):
                raise ValueError(
                    f"got {len(tickers)} tickers, expected {lo}-{hi}. "
                    "Wikipedia format may have changed."
                )
            wiki = tickers
            log.info("  %s (Wikipedia): %d tickers", name, len(tickers))
        except Exception as e:
            log.error("  %s: Wikipedia FAILED (%s)", name, e)

        etf = SSGA_ETF[name]
        ssga, as_of = None, ""
        try:
            ssga, as_of = _ssga_constituents(_fetch_bytes(SSGA_URL.format(etf=etf)))
            ref = wiki or previous.get(name) or []
            ov = _overlap(ssga, ref) if ref else 1.0
            if not (lo <= len(ssga) <= hi):
                raise ValueError(f"{len(ssga)} holdings, expected {lo}-{hi}")
            if ov < MIN_OVERLAP:
                raise ValueError(f"only {ov:.1%} overlap with the reference list (bad parse?)")
            log.info("  %s (%s holdings, %s): %d tickers, %.1f%% overlap with %s", name, etf.upper(), as_of,
                     len(ssga), ov * 100, "Wikipedia" if wiki else "the last-good list")
        except Exception as e:
            log.error("  %s: %s holdings unusable (%s) — falling back", name, etf.upper(), e)
            ssga = None

        if ssga:
            results[name] = ssga
            sources[name] = f"ssga:{etf.upper()} {as_of}"
            if wiki:
                add, rem = sorted(set(ssga) - set(wiki)), sorted(set(wiki) - set(ssga))
                if add or rem:
                    log.warning("  %s: Wikipedia differs from %s — ETF-only %s | Wikipedia-only %s",
                                name, etf.upper(), add, rem)
        elif wiki:
            results[name] = wiki
            sources[name] = "wikipedia"
        else:
            last_good = previous.get(name) or []
            if not last_good:
                raise RuntimeError(f"{name}: no ETF, Wikipedia or last-good list available")
            failures.append(name)
            results[name] = last_good
            sources[name] = f"last-good {previous.get('updated', '?')}"
            log.error("FAILED %s — reusing %d last-good tickers from %s",
                      name, len(last_good), previous.get("updated", "?"))

    total = sum(len(v) for v in results.values())
    log.info("Total SP1500: %d%s", total,
             f" (STALE: {', '.join(failures)})" if failures else "")

    data = {"updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stale": failures, "source": sources, **results}

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
