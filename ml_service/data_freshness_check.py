#!/usr/bin/env python3
"""
Unified data-freshness checker. Inspects every data source the strategy depends on
and fires ONE Telegram alert if anything critical is stale. Runs on a cron
(pre-open + midday) and is also callable on-demand from the Telegram /data command.

Two deliberate carve-outs:
  - WRDS Compustat/IBES are uploaded MANUALLY (quarterly). Next upload is ~Sept 2026,
    so their staleness is SUPPRESSED (reported 'expected', no alarm) until WRDS_EXPECTED_BY.
  - FMP fundamentals are a flaky FALLBACK (WRDS is primary) — informational, never alarm.
"""
import os
import json
from datetime import datetime, date
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
WRDS = DATA / "wrds"
WRDS_EXPECTED_BY = date(2026, 9, 15)   # manual upload due ~Sept; suppress alarm until then


def _load_env():
    for f in (BASE / ".env", BASE.parent / ".env"):
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _send_telegram(text):
    _load_env()
    tok = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not tok or not chat:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=10)
    except Exception:
        pass


def _age_days(path):
    """Newest mtime under a file/dir, in days; None if missing/empty."""
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        newest = p.stat().st_mtime
    else:
        mtimes = [f.stat().st_mtime for f in p.rglob("*") if f.is_file()]
        if not mtimes:
            return None
        newest = max(mtimes)
    return (datetime.now().timestamp() - newest) / 86400


def _parquet_last_date(path, col):
    try:
        import pandas as pd
        return pd.to_datetime(pd.read_parquet(path, columns=[col])[col]).max().date()
    except Exception:
        return None


def _fmt(age):
    return f"{age*24:.0f}h" if age is not None and age < 1 else (f"{age:.1f}d" if age is not None else "?")


def check_all():
    """Returns (list of (name, status, detail), any_critical_stale)."""
    checks, crit = [], False

    # 1. Prices (Massive/Polygon cache) — refreshes every 15 min in market hours
    age = _age_days(DATA / "massive_cache")
    if age is None:
        checks.append(("Prices (Massive)", "STALE", "missing")); crit = True
    elif age > 2.5:
        checks.append(("Prices (Massive)", "STALE", f"{_fmt(age)} old")); crit = True
    else:
        checks.append(("Prices (Massive)", "OK", _fmt(age)))

    # 2. Signals (signal server)
    try:
        d = requests.get("http://localhost:5001/signals", timeout=6).json()
        buys = len([x for x in d.get("signals", []) if x.get("signal") == "BUY"])
        if d.get("is_stale"):
            checks.append(("Signals", "STALE", "server reports STALE")); crit = True
        elif buys == 0:
            checks.append(("Signals", "STALE", "0 BUY signals")); crit = True
        else:
            checks.append(("Signals", "OK", f"{buys} BUYs, fresh"))
    except Exception:
        checks.append(("Signals", "STALE", "server unreachable")); crit = True

    # 3. Fama-French factors (last DATE in parquet; ~3-5d publication lag is normal)
    ffd = _parquet_last_date(WRDS / "fama_french_5factors_momentum_daily.parquet", "date")
    if ffd is None:
        checks.append(("Fama-French", "STALE", "unreadable")); crit = True
    else:
        old = (date.today() - ffd).days
        if old > 7:
            checks.append(("Fama-French", "STALE", f"last {ffd} ({old}d)")); crit = True
        else:
            checks.append(("Fama-French", "OK", f"last {ffd}"))

    # 4. WRDS Compustat (manual quarterly — suppressed until ~Sept upload)
    cqd = _parquet_last_date(WRDS / "compustat_quarterly.parquet", "datadate")
    if cqd is None:
        checks.append(("WRDS Compustat", "STALE", "missing")); crit = True
    else:
        old = (date.today() - cqd).days
        if date.today() < WRDS_EXPECTED_BY:
            checks.append(("WRDS Compustat", "EXPECTED", f"datadate {cqd} — next upload ~Sept"))
        elif old > 200:
            checks.append(("WRDS Compustat", "STALE", f"datadate {cqd} ({old}d) — upload overdue")); crit = True
        else:
            checks.append(("WRDS Compustat", "OK", f"datadate {cqd}"))

    # 5. FMP fundamentals (flaky FALLBACK — WRDS is primary; informational only)
    age = _age_days(DATA / "fundamentals_cache")
    if age is None:
        checks.append(("FMP fundamentals", "INFO", "missing (fallback only)"))
    elif age > 3:
        checks.append(("FMP fundamentals", "INFO", f"{_fmt(age)} stale (fallback; WRDS primary)"))
    else:
        checks.append(("FMP fundamentals", "OK", _fmt(age)))

    # 6. IBKR NAV history (only records when the engine is connected)
    try:
        hist = json.load(open(DATA / "ibkr_nav_history.json"))
        last = date.fromisoformat(hist[-1][0]); old = (date.today() - last).days
        if old > 4:
            checks.append(("IBKR NAV history", "STALE", f"last {last} ({old}d) — engine recording?")); crit = True
        else:
            checks.append(("IBKR NAV history", "OK", f"last {last}"))
    except Exception:
        checks.append(("IBKR NAV history", "STALE", "unreadable")); crit = True

    # 7. Sentiment archive (6h collector cron)
    age = _age_days(DATA / "sentiment_archive" / "raw")
    if age is None:
        checks.append(("Sentiment", "INFO", "no data yet"))
    elif age > 1.5:
        checks.append(("Sentiment", "INFO", f"{_fmt(age)} stale (collector)"))
    else:
        checks.append(("Sentiment", "OK", _fmt(age)))

    # 8. SP1500 membership (daily 6 AM scrape)
    age = _age_days(DATA / "sp1500_members.json")
    if age is None:
        checks.append(("SP1500 membership", "STALE", "missing")); crit = True
    elif age > 4:
        checks.append(("SP1500 membership", "STALE", f"{_fmt(age)} old")); crit = True
    else:
        checks.append(("SP1500 membership", "OK", _fmt(age)))

    return checks, crit


def format_report(checks):
    icons = {"OK": "✅", "STALE": "🔴", "INFO": "ℹ️", "EXPECTED": "🟡"}
    return "📋 DATA FRESHNESS\n\n" + "\n".join(
        f"{icons.get(s, '?')} {n}: {m}" for n, s, m in checks)


def main():
    checks, crit = check_all()
    report = format_report(checks)
    print(report)
    if crit:
        _send_telegram("⚠️ STALE DATA DETECTED\n\n" + report)


if __name__ == "__main__":
    main()
