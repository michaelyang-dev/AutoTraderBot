"""data_freshness_check._check_edgar_overlay: health = the patcher evaluated the universe, not the patched count.

WHY: 2026-09-08 19:30 the check alarmed "roe patched COLLAPSED to 4 — patcher broken?" right after the WRDS
refresh made Compustat current (datadate 2026-08-31). The patcher had validated 1,339 names with no errors;
there was simply nothing left to patch. A genuine breakage collapses validated+patched (or leaves a traceback).
Run: python3 tests/test_freshness_edgar_rule.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import data_freshness_check as F  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


tmp = Path(tempfile.mkdtemp())
F.DATA = tmp
F._last_expected_patcher_run = lambda: datetime.now() - timedelta(hours=2)


def run(validated, patched, cq_date, gen=None):
    json.dump({"generated": (gen or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"), "compustat_max_datadate": cq_date,
               "stats": {"roe": {"validated": validated, "patched": patched, "failed": 1504 - validated - patched}}, "features": {}},
              open(tmp / "edgar_feature_overlay.json", "w"))
    return F._check_edgar_overlay()


today = datetime.now()
r = run(1339, 4, (today - timedelta(days=8)).strftime("%Y-%m-%d"))
check("2026-09-08 state (validated 1339, patched 4, Compustat 8d old) -> OK, not an alarm", r[1] == "OK", r[2])
check("...OK message says why the count is low", "few patches" in r[2], r[2])
r = run(10, 4, (today - timedelta(days=8)).strftime("%Y-%m-%d"))
check("genuine collapse (validated 10) -> STALE", r[1] == "STALE", r[2])
r = run(1339, 4, (today - timedelta(days=120)).strftime("%Y-%m-%d"))
check("patched 4 while Compustat is 120d old -> STALE (EDGAR should be patching)", r[1] == "STALE", r[2])
r = run(400, 900, (today - timedelta(days=120)).strftime("%Y-%m-%d"))
check("normal pre-refresh state (patched 900) -> OK", r[1] == "OK", r[2])
r = run(1339, 4, "not-a-date")
check("unparseable Compustat date with low patched -> STALE (fail loud)", r[1] == "STALE", r[2])
r = run(1339, 4, (today - timedelta(days=8)).strftime("%Y-%m-%d"), gen=today - timedelta(hours=30))
check("missed patcher run still -> STALE", r[1] == "STALE", r[2])

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all freshness-rule tests passed"))
sys.exit(1 if FAILS else 0)
