#!/usr/bin/env python3
"""
Real-Time EDGAR Filing Monitor
================================
Polls EDGAR EFTS every 60 seconds for new 8-K filings with forced-selling
event triggers. Sends alerts via Telegram and generates short signals.

Much faster than the 30-minute scan cycle — captures events within
1-2 minutes of filing, before most of the price drop happens.

Event types detected:
  - 8-K Item 4.01: Auditor change (going concern trigger)
  - 8-K Item 4.02: Financial restatement
  - 8-K Item 2.06: Material impairment
  - 8-K Item 3.01: Delisting notice
  - 8-K Item 5.02: CEO/CFO departure (with SMA50 filter for quality)

SEC rate limit: 10 requests/second. We use 1 request/minute = safe.

Usage:
    cd ml_service && python3 edgar_realtime.py

Deployment:
    pm2 start edgar_realtime.py --name edgar-monitor \
        --interpreter /path/to/venv/bin/python3
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Setup
ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
os.chdir(ML_DIR)

from dotenv import load_dotenv
load_dotenv(ML_DIR.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("edgar_realtime")

# ── Configuration ──
POLL_INTERVAL = 60  # seconds between checks
USER_AGENT = "AutoTraderBot research@autotrader.com"

# Event types that trigger forced selling
TRIGGER_ITEMS = {
    "4.01": "auditor_change",
    "4.02": "financial_restatement",
    "2.06": "material_impairment",
    "3.01": "delisting_notice",
}

# Lower-conviction events (need additional filters)
SECONDARY_ITEMS = {
    "5.02": "executive_departure",
}

# EDGAR EFTS endpoint
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

# Telegram (optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# Signal output
SIGNAL_FILE = ML_DIR / "data" / "short_sleeve" / "realtime_events.json"


def send_telegram(msg):
    """Send alert via Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=10)
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def query_edgar_filings(form_type="8-K", days_back=1):
    """Query EDGAR EFTS for recent filings matching our trigger keywords."""
    headers = {"User-Agent": USER_AGENT}

    now = datetime.utcnow()
    today = now.strftime("%Y-%m-%d")
    start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # EFTS requires a keyword query — we search for our trigger item codes
    # Each query targets specific 8-K item types
    queries = [
        '"4.01"',  # auditor change
        '"4.02"',  # financial restatement
        '"2.06"',  # material impairment
        '"3.01"',  # delisting notice
        '"going concern"',
        '"material weakness"',
    ]

    all_hits = []
    seen_ids = set()

    for q in queries:
        try:
            url = (f"{EFTS_URL}?q={q}&forms={form_type}"
                   f"&dateRange=custom&startdt={start}&enddt={today}")
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                for h in hits:
                    fid = h.get("_id", "")
                    if fid not in seen_ids:
                        all_hits.append(h)
                        seen_ids.add(fid)
            time.sleep(0.12)  # SEC rate limit: 10/sec
        except Exception as e:
            log.error(f"EDGAR query '{q}' failed: {e}")

    return all_hits


def extract_events(hits, seen_ids):
    """Extract forced-selling events from EDGAR filing hits."""
    events = []

    for hit in hits:
        filing_id = hit.get("_id", "")
        if filing_id in seen_ids:
            continue

        source = hit.get("_source", {})
        form = source.get("form", "")
        items = source.get("items", [])
        file_date = source.get("file_date", "")
        display_names = source.get("display_names", [])
        ciks = source.get("ciks", [])
        adsh = source.get("adsh", "")

        if form != "8-K":
            continue

        # Extract ticker from display name (format: "Company Name (TICKER) (CIK ...)")
        ticker = None
        for dn in display_names:
            if "(" in dn:
                parts = dn.split("(")
                for p in parts:
                    p = p.strip().rstrip(")")
                    if p.isupper() and 1 <= len(p) <= 5 and p.isalpha():
                        ticker = p
                        break
            if ticker:
                break

        if not ticker:
            continue

        # Check for trigger items
        triggered = []
        for item in items:
            if item in TRIGGER_ITEMS:
                triggered.append((item, TRIGGER_ITEMS[item]))
            elif item in SECONDARY_ITEMS:
                triggered.append((item, SECONDARY_ITEMS[item]))

        if not triggered:
            continue

        # Determine if this is high-conviction (primary) or needs filtering (secondary)
        is_primary = any(item in TRIGGER_ITEMS for item, _ in triggered)

        event = {
            "filing_id": filing_id,
            "ticker": ticker,
            "cik": ciks[0] if ciks else "",
            "company": display_names[0] if display_names else "",
            "file_date": file_date,
            "items": items,
            "triggered_items": [(i, t) for i, t in triggered],
            "event_types": [t for _, t in triggered],
            "is_primary": is_primary,
            "adsh": adsh,
            "detected_at": datetime.utcnow().isoformat(),
            "url": f"https://www.sec.gov/Archives/edgar/data/{ciks[0]}/{adsh.replace('-', '')}" if ciks else "",
        }
        events.append(event)
        seen_ids.add(filing_id)

    return events


def save_events(events, existing_events):
    """Save detected events to signal file for the short sleeve manager."""
    all_events = existing_events + events

    # Keep last 30 days
    cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
    all_events = [e for e in all_events if e.get("detected_at", "") > cutoff]

    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_FILE, "w") as f:
        json.dump(all_events, f, indent=2, default=str)

    return all_events


def load_existing_events():
    """Load previously detected events."""
    if SIGNAL_FILE.exists():
        try:
            with open(SIGNAL_FILE) as f:
                return json.load(f)
        except:
            return []
    return []


def run_monitor():
    """Main monitoring loop."""
    log.info("=" * 60)
    log.info("EDGAR Real-Time Filing Monitor")
    log.info(f"Polling every {POLL_INTERVAL}s for 8-K forced-selling events")
    log.info(f"Trigger items: {list(TRIGGER_ITEMS.keys())}")
    log.info(f"Secondary items: {list(SECONDARY_ITEMS.keys())}")
    log.info(f"Telegram: {'enabled' if TELEGRAM_TOKEN else 'disabled'}")
    log.info("=" * 60)

    seen_ids = set()
    existing_events = load_existing_events()

    # Pre-populate seen IDs from existing events
    for e in existing_events:
        seen_ids.add(e.get("filing_id", ""))

    log.info(f"Loaded {len(existing_events)} existing events, {len(seen_ids)} seen IDs")

    # Initial scan: get today's filings to build seen set
    log.info("Initial scan: loading today's filings...")
    hits = query_edgar_filings(days_back=1)  # last 24 hours
    initial_events = extract_events(hits, seen_ids)
    if initial_events:
        log.info(f"Found {len(initial_events)} events in initial scan (not alerting)")
        existing_events = save_events(initial_events, existing_events)
    log.info(f"Monitoring started. Seen {len(seen_ids)} filings today.")

    consecutive_errors = 0

    while True:
        try:
            time.sleep(POLL_INTERVAL)

            # Only poll during extended market hours (7 AM - 8 PM ET, weekdays)
            now = datetime.utcnow()
            et_hour = (now.hour - 4) % 24  # rough ET conversion
            if now.weekday() >= 5:  # weekend
                continue
            if et_hour < 7 or et_hour > 20:  # outside extended hours
                continue

            hits = query_edgar_filings(days_back=1)
            new_events = extract_events(hits, seen_ids)

            if new_events:
                for event in new_events:
                    # Log and alert
                    types = ", ".join(event["event_types"])
                    items_str = ", ".join(event["items"])
                    primary = "🔴 PRIMARY" if event["is_primary"] else "🟡 SECONDARY"

                    log.info(f"{primary} {event['ticker']}: {types} (items: {items_str})")
                    log.info(f"  Company: {event['company']}")
                    log.info(f"  URL: {event['url']}")

                    # Telegram alert for primary events
                    if event["is_primary"]:
                        msg = (
                            f"🔴 SHORT SIGNAL: {event['ticker']}\n"
                            f"Event: {types}\n"
                            f"8-K Items: {items_str}\n"
                            f"Company: {event['company']}\n"
                            f"Filed: {event['file_date']}\n"
                            f"Detected: {event['detected_at'][:19]}\n"
                            f"URL: {event['url']}"
                        )
                        send_telegram(msg)

                existing_events = save_events(new_events, existing_events)
                log.info(f"Total events stored: {len(existing_events)}")

            consecutive_errors = 0

        except KeyboardInterrupt:
            log.info("Monitor stopped by user")
            break
        except Exception as e:
            consecutive_errors += 1
            log.error(f"Error in monitoring loop: {e}")
            if consecutive_errors > 10:
                log.error("Too many consecutive errors, backing off 5 minutes")
                time.sleep(300)
                consecutive_errors = 0


if __name__ == "__main__":
    run_monitor()
