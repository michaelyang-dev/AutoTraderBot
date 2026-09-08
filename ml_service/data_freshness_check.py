#!/usr/bin/env python3
"""
Unified data-freshness checker. Inspects every data source the strategy depends on
and fires ONE Telegram alert if anything critical is stale. Runs on a cron
(pre-open + midday + post-patcher) and is also callable on-demand from /data.

HARDENED 2026-07-13 (user: "make sure nothing will silently fail"):
  - EVERY check runs in ISOLATION (its own try/except). A bug in one check reports
    ERROR for that check but never hides the others or crashes the whole report.
  - Added the EDGAR OVERLAY / PATCHER check — the overlay now feeds LIVE signals, so a
    silently-broken daily patcher = live signals drifting on stale fresh-data.
  - A check that cannot verify its source returns STALE/ERROR (never silently OK).

Two deliberate carve-outs:
  - WRDS Compustat/IBES are uploaded MANUALLY (quarterly). Next upload ~Sept 2026, so
    their staleness is SUPPRESSED ('expected', no alarm) until WRDS_EXPECTED_BY.
  - FMP fundamentals are a flaky FALLBACK (WRDS primary) — informational, never alarm.
"""
import os
import json
from datetime import datetime, date, timedelta
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


def _min_age_days(path, pattern="*"):
    """OLDEST (stalest) mtime among files matching pattern under a dir, in days —
    catches PARTIAL staleness that max-mtime (_age_days) would hide (a few fresh files
    masking many stale ones). (count, oldest_age_days) or (0, None) if empty."""
    p = Path(path)
    if not p.exists() or p.is_file():
        return (0, None)
    files = [f for f in p.rglob(pattern) if f.is_file()]
    if not files:
        return (0, None)
    oldest = min(f.stat().st_mtime for f in files)
    return (len(files), (datetime.now().timestamp() - oldest) / 86400)


def _parquet_last_date(path, col):
    import pandas as pd
    return pd.to_datetime(pd.read_parquet(path, columns=[col])[col]).max().date()


def _fmt(age):
    return f"{age*24:.0f}h" if age is not None and age < 1 else (f"{age:.1f}d" if age is not None else "?")


def _last_expected_patcher_run():
    """Most recent weekday 18:40 (server ET) that the patcher cron should have finished.
    Handles the weekend gap so Monday morning doesn't false-alarm on Friday's run."""
    now = datetime.now()
    d = now if now.hour >= 19 else now - timedelta(days=1)   # today's 18:40 may not be done
    while d.weekday() >= 5:                                   # skip Sat/Sun back to Fri
        d = d - timedelta(days=1)
    return d.replace(hour=18, minute=40, second=0, microsecond=0)


# ── individual checks: each returns (name, status, detail); raising is caught by the
#    runner and reported as ERROR (never a silent skip). status in OK/STALE/INFO/EXPECTED/ERROR
def _check_prices():
    age = _age_days(DATA / "massive_cache")
    if age is None:
        return ("Prices (Massive)", "STALE", "missing")
    if age > 2.5:
        return ("Prices (Massive)", "STALE", f"{_fmt(age)} old")
    return ("Prices (Massive)", "OK", _fmt(age))


def _check_signals():
    d = requests.get("http://localhost:5001/signals", timeout=6).json()
    buys = len([x for x in d.get("signals", []) if x.get("signal") == "BUY"])
    if d.get("is_stale"):
        return ("Signals", "STALE", "server reports STALE")
    if buys == 0:
        return ("Signals", "STALE", "0 BUY signals")
    return ("Signals", "OK", f"{buys} BUYs, fresh")


def _check_fama_french():
    ff = WRDS / "fama_french_5factors_momentum_daily.parquet"
    age = _age_days(ff)
    try:
        ffd = _parquet_last_date(ff, "date")
    except Exception:
        ffd = None
    note = f", data to {ffd}" if ffd else ""
    if age is None:
        return ("Fama-French", "STALE", "missing")
    if age > 4:
        return ("Fama-French", "STALE", f"download {_fmt(age)} old{note}")
    return ("Fama-French", "OK", f"{_fmt(age)}{note}")


def _check_wrds_compustat():
    try:
        cqd = _parquet_last_date(WRDS / "compustat_fundamentals_quarterly.parquet", "datadate")
    except Exception:
        cqd = None
    if cqd is None:
        return ("WRDS Compustat", "STALE", "missing / unreadable")
    old = (date.today() - cqd).days
    # Next upload is derived from the data, not a hardcoded date (the old "~Sept" hint kept
    # showing after the 2026-09-07 upload had already landed). Quarterly filings for the quarter
    # ending after `cqd` are mostly in ~6 weeks after that quarter end, so suggest uploading then.
    q_end_month = ((cqd.month - 1) // 3 + 1) * 3            # calendar quarter containing cqd
    q_end = (date(cqd.year + (q_end_month == 12), 1 if q_end_month == 12 else q_end_month + 1, 1) - timedelta(days=1))
    if cqd >= q_end:                                         # cqd IS a quarter end -> that quarter is in hand, look to the next
        m = q_end.month + 3
        q_end = date(q_end.year + (m > 12), (m - 1) % 12 + 1, 1) + timedelta(days=31)
        q_end = date(q_end.year, q_end.month, 1) - timedelta(days=1)
    next_upload = (q_end + timedelta(days=45)).replace(day=15)   # ~6 weeks after that quarter end
    if old > 200:
        return ("WRDS Compustat", "STALE", f"datadate {cqd} ({old}d) — upload overdue")
    if old > 120:
        return ("WRDS Compustat", "EXPECTED", f"datadate {cqd} ({old}d) — upload due ~{next_upload:%b %d}")
    return ("WRDS Compustat", "OK", f"datadate {cqd} (uploaded; next ~{next_upload:%b %d})")


def _check_edgar_overlay():
    """EDGAR overlay / daily patcher — LIVE-feeding since 2026-07-13, so a silently
    broken patcher = live signals drifting on stale fresh-data. Verifies: the overlay
    was regenerated by the most recent expected patcher run, the ROE patched count
    hasn't collapsed, and the patcher log has no recent traceback."""
    ov = DATA / "edgar_feature_overlay.json"
    if not ov.exists():
        return ("EDGAR overlay", "STALE", "overlay MISSING — patcher never ran")
    o = json.load(open(ov))                       # raise -> runner reports ERROR
    gen = o.get("generated")
    roe_patched = o.get("stats", {}).get("roe", {}).get("patched", 0)
    gen_dt = None
    if gen:
        try:
            gen_dt = datetime.fromisoformat(gen)
        except Exception:
            pass
    if gen_dt is None:
        return ("EDGAR overlay", "STALE", "no valid 'generated' timestamp")
    expected = _last_expected_patcher_run()
    if gen_dt < expected - timedelta(hours=1):
        age_h = (datetime.now() - gen_dt).total_seconds() / 3600
        return ("EDGAR overlay", "STALE",
                f"generated {age_h:.0f}h ago — MISSED the {expected:%a %H:%M} patcher run")
    # HEALTH is "did the patcher evaluate the universe", NOT "how many names it patched".
    # 2026-09-08 false alarm: the 09-07 WRDS refresh brought Compustat to datadate 2026-08-31,
    # so EDGAR had newer data for only 4 names (validated 1,339, failed 157, no errors; the
    # 18:55 reconcile matched the overlay's earlier patches to the new Compustat values at 99.6%).
    # A patched count near zero is therefore EXPECTED while Compustat is current, and a genuine
    # breakage shows up as a collapsed validated+patched count or a traceback, both checked here.
    roe_stats = o.get("stats", {}).get("roe", {})
    roe_evaluated = int(roe_stats.get("validated", 0)) + int(roe_patched)
    if roe_evaluated < 1000:
        return ("EDGAR overlay", "STALE",
                f"roe evaluated COLLAPSED to {roe_evaluated} (validated {roe_stats.get('validated', 0)}, "
                f"patched {roe_patched}) — patcher broken?")
    cq_days = None
    try:
        cq_days = (datetime.now() - datetime.fromisoformat(str(o.get("compustat_max_datadate")))).days
    except Exception:
        pass
    if roe_patched < 30 and (cq_days is None or cq_days > 75):
        return ("EDGAR overlay", "STALE",
                f"roe patched COLLAPSED to {roe_patched} while Compustat is {cq_days}d old — patcher broken?")
    # recent patcher-log traceback (only if the log was written this cycle)
    logf = BASE.parent / "logs" / "edgar_patch.log"
    if logf.exists() and (datetime.now().timestamp() - logf.stat().st_mtime) < 30 * 3600:
        tail = logf.read_text(errors="ignore").splitlines()[-80:]
        if any(("Traceback" in ln or "Error" in ln) for ln in tail):
            return ("EDGAR overlay", "STALE", f"patcher LOG has a recent error (roe {roe_patched})")
    note = " (Compustat current — few patches needed)" if roe_patched < 30 else ""
    return ("EDGAR overlay", "OK",
            f"roe validated {roe_stats.get('validated', 0)}, patched {roe_patched}{note}, {gen_dt:%b %d %H:%M}")


def _check_fmp():
    age = _age_days(DATA / "fundamentals_cache")
    if age is None:
        return ("FMP fundamentals", "INFO", "missing (fallback only)")
    if age > 3:
        return ("FMP fundamentals", "INFO", f"{_fmt(age)} stale (fallback; WRDS primary)")
    return ("FMP fundamentals", "OK", _fmt(age))


def _check_ibkr_nav():
    hist = json.load(open(DATA / "ibkr_nav_history.json"))
    last = date.fromisoformat(hist[-1][0])
    old = (date.today() - last).days
    if old > 4:
        return ("IBKR NAV history", "STALE", f"last {last} ({old}d) — engine recording?")
    return ("IBKR NAV history", "OK", f"last {last}")


def _check_sentiment():
    age = _age_days(DATA / "sentiment_archive" / "raw")
    if age is None:
        return ("Sentiment", "INFO", "no data yet")
    if age > 1.5:
        return ("Sentiment", "INFO", f"{_fmt(age)} stale (collector)")
    return ("Sentiment", "OK", _fmt(age))


def _check_sp1500():
    age = _age_days(DATA / "sp1500_members.json")
    if age is None:
        return ("SP1500 membership", "STALE", "missing")
    if age > 4:
        return ("SP1500 membership", "STALE", f"{_fmt(age)} old")
    return ("SP1500 membership", "OK", _fmt(age))


_CHECKS = [_check_prices, _check_signals, _check_fama_french, _check_wrds_compustat,
           _check_edgar_overlay, _check_fmp, _check_ibkr_nav, _check_sentiment, _check_sp1500]


def check_all():
    """Run EVERY check in isolation. Returns (list of (name, status, detail),
    any_critical). A check that raises is reported as ERROR (critical) — never
    silently skipped, so one broken check can't hide a real staleness elsewhere."""
    checks, crit = [], False
    for fn in _CHECKS:
        try:
            name, status, detail = fn()
        except Exception as e:
            name = fn.__name__.replace("_check_", "").replace("_", " ")
            status, detail = "ERROR", f"check crashed: {type(e).__name__}: {e}"
        checks.append((name, status, detail))
        if status in ("STALE", "ERROR"):
            crit = True
    return checks, crit


def format_report(checks):
    icons = {"OK": "✅", "STALE": "🔴", "INFO": "ℹ️", "EXPECTED": "🟡", "ERROR": "🔴"}
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
