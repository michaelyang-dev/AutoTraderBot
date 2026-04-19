"""
Point-in-Time S&P 500 Membership
==================================
Fetches historical S&P 500 constituent changes from FMP and provides
a function to get the exact S&P 500 membership on any given date.

This avoids survivorship bias in backtests — we only trade stocks that
were actually in the S&P 500 at the time.

Usage:
    from sp500_history import get_sp500_on_date, load_sp500_changes

    # Get S&P 500 members on a specific date
    members = get_sp500_on_date(pd.Timestamp("2020-01-15"))

Run standalone to fetch and cache the history:
    python3 sp500_history.py
"""

import json
import os
import ssl
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_FILE = DATA_DIR / "sp500_changes.json"
CACHE_TTL_DAYS = 7

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# Module-level cache for the computed membership sets
_changes_cache = None


def _fetch_historical_changes() -> list[dict]:
    """Fetch historical S&P 500 constituent changes from FMP."""
    if not FMP_API_KEY:
        print("  WARNING: FMP_API_KEY not set — cannot fetch S&P 500 history")
        return []

    url = f"https://financialmodelingprep.com/stable/historical-sp500-constituent?apikey={FMP_API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list):
            print(f"  Fetched {len(data)} S&P 500 historical changes from FMP")
            return data
        return []
    except Exception as e:
        print(f"  FMP historical fetch failed: {e}")
        return []


def load_sp500_changes(force_refresh: bool = False) -> list[dict]:
    """Load S&P 500 changes (cached to disk)."""
    global _changes_cache

    if not force_refresh and CACHE_FILE.exists():
        age_days = (datetime.today() - datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)).days
        if age_days < CACHE_TTL_DAYS:
            try:
                with open(CACHE_FILE) as f:
                    _changes_cache = json.load(f)
                return _changes_cache
            except (json.JSONDecodeError, OSError):
                pass

    changes = _fetch_historical_changes()
    if changes:
        DATA_DIR.mkdir(exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(changes, f)
    _changes_cache = changes
    return changes


def get_sp500_on_date(date: pd.Timestamp) -> set[str]:
    """
    Get the set of S&P 500 constituent symbols on a given date.

    Works by starting with the current S&P 500 and reverse-applying
    all changes after the target date:
    - If a symbol was added after the target date, remove it
    - If a symbol was removed after the target date, add it back

    Returns a set of ticker symbols.
    """
    from sp500_universe import get_stock_symbols

    changes = load_sp500_changes()
    if not changes:
        # No history available — return current constituents
        return set(get_stock_symbols())

    # Start with current S&P 500
    current = set(get_stock_symbols())

    # Sort changes by date descending (most recent first)
    dated_changes = []
    for c in changes:
        try:
            change_date = pd.Timestamp(c.get("dateAdded") or c.get("date", ""))
            if pd.isna(change_date):
                continue
            dated_changes.append((change_date, c))
        except (ValueError, TypeError):
            continue

    dated_changes.sort(key=lambda x: x[0], reverse=True)

    # Reverse-apply changes that happened AFTER the target date
    for change_date, c in dated_changes:
        if change_date <= date:
            break

        added = c.get("symbol", "")
        removed = c.get("removedTicker", c.get("replacedBy", ""))

        # Reverse: if it was added after target, remove it from our set
        if added:
            current.discard(added)
        # Reverse: if it was removed after target, add it back
        if removed:
            current.add(removed)

    return current


if __name__ == "__main__":
    print("=" * 60)
    print("S&P 500 Historical Membership")
    print("=" * 60)

    changes = load_sp500_changes(force_refresh=True)
    print(f"\n  Total historical changes: {len(changes)}")

    if changes:
        # Show sample
        print(f"\n  Recent changes (last 10):")
        sorted_changes = sorted(changes, key=lambda c: c.get("dateAdded", c.get("date", "")), reverse=True)
        for c in sorted_changes[:10]:
            date = c.get("dateAdded", c.get("date", "?"))
            added = c.get("symbol", "?")
            removed = c.get("removedTicker", c.get("replacedBy", "?"))
            print(f"    {date}: +{added:<6} -{removed}")

    # Test point-in-time lookup
    test_dates = [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2022-01-03"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2018-01-02"),
        pd.Timestamp("2015-01-02"),
    ]

    print(f"\n  Point-in-time membership:")
    for d in test_dates:
        members = get_sp500_on_date(d)
        print(f"    {d.date()}: {len(members)} members")

    print(f"\n  Cache file: {CACHE_FILE}")
    print("  Done.")
