"""
Point-in-Time S&P 1500 Membership (Live)
==========================================
Reads SP1500 constituent data exported weekly from Norgate (via Parallels).
Falls back to FMP SP500-only if the Norgate export is missing or stale.

Usage:
    from sp1500_membership import get_sp1500_on_date

    members = get_sp1500_on_date()  # returns set of ticker strings
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from sp500_history import get_sp500_on_date

log = logging.getLogger("sp1500_membership")

DATA_DIR = Path(__file__).resolve().parent / "data"
MEMBERS_FILE = DATA_DIR / "sp1500_members.json"
STALE_DAYS = 14  # warn if file older than 14 days

# Module-level cache
_cache = None
_cache_mtime = None


def _load_members() -> dict | None:
    """Load sp1500_members.json, with caching based on file mtime."""
    global _cache, _cache_mtime

    if not MEMBERS_FILE.exists():
        return None

    mtime = MEMBERS_FILE.stat().st_mtime
    if _cache is not None and _cache_mtime == mtime:
        return _cache

    with open(MEMBERS_FILE) as f:
        data = json.load(f)

    _cache = data
    _cache_mtime = mtime

    # Check staleness
    updated = datetime.strptime(data["updated"], "%Y-%m-%d %H:%M:%S")
    age_days = (datetime.now() - updated).days
    if age_days > STALE_DAYS:
        log.warning("sp1500_members.json is %d days old — run Norgate export", age_days)

    return data


def get_sp1500_on_date(date=None) -> set[str]:
    """Get current SP1500 members. Returns set of ticker strings.

    Uses Norgate export if available, falls back to FMP SP500-only.
    The `date` param is accepted for API compatibility but the Norgate
    export only contains the current membership (not historical).
    """
    data = _load_members()

    if data is not None:
        members = set(data.get("sp500", []))
        members.update(data.get("sp400", []))
        members.update(data.get("sp600", []))
        if members:
            return members

    # Fallback: FMP SP500 only
    log.warning("No Norgate SP1500 data — falling back to FMP SP500 only")
    return get_sp500_on_date(date)


def get_membership_status() -> dict:
    """Return status info for monitoring/debugging."""
    data = _load_members()
    if data is None:
        return {"source": "fmp_sp500_fallback", "count": 0, "updated": None}

    updated = data.get("updated", "unknown")
    sp500 = len(data.get("sp500", []))
    sp400 = len(data.get("sp400", []))
    sp600 = len(data.get("sp600", []))

    return {
        "source": "norgate",
        "updated": updated,
        "sp500": sp500,
        "sp400": sp400,
        "sp600": sp600,
        "total": sp500 + sp400 + sp600,
    }
