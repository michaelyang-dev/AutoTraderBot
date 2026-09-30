"""Price-cache validity: a Massive per-symbol cache file is reused only if it is < 18h old AND was
written after the most recent settled close (weekday 17:00 ET).

WHY THIS TEST EXISTS (2026-09-30)
  Verifying the 2026-09-30 book-3 rebuild found that the age-only rule (18h) made
    * every post-close build on 2026-09-29 (17:50 cron restart, 18:33 refresh restart) reuse files
      fetched that morning -> evening signals computed on 2026-09-28's close, a day stale;
    * Monday's refetch land at ~15:20 (18h after the Sunday 21:20 restart) and cache a ~15:05
      intraday snapshot as Monday's bar, reused by the evening builds AND Tuesday's 09:18 pre-open
      build (2026-09-29 09:18:51 did exactly this) until the ~09:33 refetch.
  The cases below are those exact timestamps.
Run: python3 tests/test_cache_settle.py
"""
import os
import sys
import tempfile
import types
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MASSIVE_API_KEY", "test-key-not-used")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import massive_data_provider as M  # noqa: E402

ET = ZoneInfo("America/New_York")
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def ts(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=ET).timestamp()


def utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=ZoneInfo("UTC")).timestamp()


print("last settled close")
check("Wed 09:18 -> Tue 17:00", M.last_settle_ts(ts(2026, 9, 30, 9, 18)) == ts(2026, 9, 29, 17))
check("Tue 17:50 -> Tue 17:00 (same day, after settle)", M.last_settle_ts(ts(2026, 9, 29, 17, 50)) == ts(2026, 9, 29, 17))
check("Tue 16:59 -> Mon 17:00 (bell rung, not settled)", M.last_settle_ts(ts(2026, 9, 29, 16, 59)) == ts(2026, 9, 28, 17))
check("exactly 17:00 counts as settled", M.last_settle_ts(ts(2026, 9, 29, 17, 0)) == ts(2026, 9, 29, 17))
check("Mon 09:18 -> Fri 17:00 (skips the weekend)", M.last_settle_ts(ts(2026, 9, 28, 9, 18)) == ts(2026, 9, 25, 17))
check("Sun 21:20 -> Fri 17:00", M.last_settle_ts(ts(2026, 9, 27, 21, 20)) == ts(2026, 9, 25, 17))
check("Sat 12:00 -> Fri 17:00", M.last_settle_ts(ts(2026, 9, 26, 12)) == ts(2026, 9, 25, 17))
check("DST end: Mon 2026-11-02 09:18 EST -> Fri 10-30 17:00 EDT = 21:00Z",
      M.last_settle_ts(ts(2026, 11, 2, 9, 18)) == utc(2026, 10, 30, 21))
check("DST end: Mon 2026-11-02 17:30 EST -> same day 17:00 EST = 22:00Z",
      M.last_settle_ts(ts(2026, 11, 2, 17, 30)) == utc(2026, 11, 2, 22))
check("DST start: Mon 2027-03-15 09:18 EDT -> Fri 03-12 17:00 EST = 22:00Z",
      M.last_settle_ts(ts(2027, 3, 15, 9, 18)) == utc(2027, 3, 12, 22))
check("DST start weekend: Sun 2027-03-14 12:00 EDT -> Fri 03-12 17:00 EST",
      M.last_settle_ts(ts(2027, 3, 14, 12)) == utc(2027, 3, 12, 22))

import re  # noqa: E402
_sb_src = (Path(__file__).resolve().parent.parent / "signal_builder.py").read_text()
_m = re.search(r"^SESSION_SETTLE_HOUR_ET\s*=\s*(\d+)", _sb_src, re.M)
check("price-cache settle hour == signal_builder partial-session settle hour (they must move together)",
      _m is not None and int(_m.group(1)) == M.SETTLE_HOUR_ET, _m.group(1) if _m else "not found")

print("cache validity — the 2026-09-28/29 timeline")
f, why = M.cache_file_fresh(ts(2026, 9, 29, 9, 34), ts(2026, 9, 29, 17, 50))
check("09-29 17:50 restart: morning file is REFETCHED (was reused -> evening signals a day stale)", (f, why) == (False, "pre-settle"), (f, why))
f, why = M.cache_file_fresh(ts(2026, 9, 28, 15, 33), ts(2026, 9, 29, 9, 18))
check("09-29 09:18 pre-open: Monday 15:33 snapshot file is REFETCHED (was reused at 17.75h)", (f, why) == (False, "pre-settle"), (f, why))
f, why = M.cache_file_fresh(ts(2026, 9, 29, 17, 52), ts(2026, 9, 30, 9, 18))
check("pre-open build reuses last evening's post-settle file (no vendor call before the open)", f is True, (f, why))
f, why = M.cache_file_fresh(ts(2026, 9, 29, 17, 52), ts(2026, 9, 30, 11, 53))
check("the 18h age cap still applies", (f, why) == (False, "age"), (f, why))
f, why = M.cache_file_fresh(ts(2026, 9, 25, 17, 52), ts(2026, 9, 27, 21, 20))
check("Sunday restart: Friday-evening file expired by age", (f, why) == (False, "age"), (f, why))
f, why = M.cache_file_fresh(ts(2026, 9, 27, 21, 22), ts(2026, 9, 28, 9, 18))
check("Monday pre-open reuses the Sunday-night file (Friday's final bars)", f is True, (f, why))
f, why = M.cache_file_fresh(ts(2026, 9, 25, 17, 52), ts(2026, 9, 26, 10, 0))
check("Saturday morning reuses the Friday-evening file", f is True, (f, why))

print("strictly more conservative than the old age-only rule")
rng = np.random.default_rng(930)
viol = 0
for _ in range(20000):
    now = ts(2026, 9, 1, 0) + rng.uniform(0, 60 * 86400)
    mt = now - rng.uniform(0, 40 * 3600)
    new_ok, _w = M.cache_file_fresh(mt, now)
    old_ok = (now - mt) / 3600 < 18
    if new_ok and not old_ok:
        viol += 1
check("never reuses a file the old rule would have refetched (20,000 random cases)", viol == 0, f"{viol} violations")

print("fetch path uses the rule (temp cache dir, vendor stubbed)")
tmp = Path(tempfile.mkdtemp(prefix="cache_settle_test_"))
M.CACHE_DIR = tmp
now_fixed = ts(2026, 9, 29, 17, 50)
idx = pd.bdate_range(end="2026-09-28", periods=380)
frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": np.linspace(10, 20, 380), "volume": 1e6}, index=idx)
for sym, mt in [("PRE", ts(2026, 9, 29, 9, 34)), ("POST", ts(2026, 9, 29, 17, 5)), ("OLD", ts(2026, 9, 28, 9, 0))]:
    p = tmp / f"{sym}_adj.parquet"
    frame.to_parquet(p)
    os.utime(p, (mt, mt))
calls = []
_real_time = M.time
M.time = types.SimpleNamespace(time=lambda: now_fixed, sleep=_real_time.sleep, perf_counter=_real_time.perf_counter)
try:
    prov = M.MassiveDataProvider(api_key="x", validate_vs_yfinance=False)
    prov.fetch_grouped_daily = lambda *a, **k: {}

    def _fake(sym, start, end, adjusted=True):
        calls.append(sym)
        return frame
    prov.fetch_ticker_bars = _fake
    out = prov.fetch_bars_batch(["PRE", "POST", "OLD"], warmup_days=550)
finally:
    M.time = _real_time
check("pre-settle file refetched, post-settle file reused, aged file refetched", sorted(calls) == ["OLD", "PRE"], calls)
check("all three symbols returned", all(len(out.get(s, [])) == 380 for s in ("PRE", "POST", "OLD")))

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all cache-settle tests passed"))
sys.exit(1 if FAILS else 0)
