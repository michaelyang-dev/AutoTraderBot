"""Verify the rebalance-clock persistence fix: a restart must RESUME the 20-day
cadence from disk, not reset it (which caused an extra rebalance at every restart).
Safe to run alongside the live engine — it only round-trips the state file (backed
up + restored) and never connects to IBKR."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date, datetime
from ibkr_engine import IBKREngine, REBAL_STATE_FILE, REBAL_DAYS

bak = str(REBAL_STATE_FILE) + ".verifybak"
if os.path.exists(REBAL_STATE_FILE):
    os.rename(REBAL_STATE_FILE, bak)

try:
    # TEST 1 — fresh deploy (no state file) keeps the immediate-first-rebal default
    e = IBKREngine()
    assert e._trading_days_since_rebal == REBAL_DAYS, f"fresh={e._trading_days_since_rebal}"
    print(f"PASS  fresh deploy (no file) -> counter={e._trading_days_since_rebal} (allows immediate first rebal to converge)")

    # TEST 2 — a mid-cycle clock survives a 'restart' (new instance reloads it)
    e._trading_days_since_rebal = 7
    e._last_rebal_date = date(2026, 6, 3)
    e._last_counted_day = date(2026, 6, 15)
    e.last_rebalance = datetime(2026, 6, 15, 9, 16)
    e._save_rebal_state()
    e2 = IBKREngine()                       # simulate a restart
    assert e2._trading_days_since_rebal == 7, f"reload counter={e2._trading_days_since_rebal}"
    assert e2._last_rebal_date == date(2026, 6, 3), f"reload date={e2._last_rebal_date}"
    assert e2._last_counted_day == date(2026, 6, 15)
    assert e2.last_rebalance == datetime(2026, 6, 15, 9, 16)
    print(f"PASS  restart RESUMES -> counter={e2._trading_days_since_rebal}/{REBAL_DAYS} (NOT reset to {REBAL_DAYS}); 7<{REBAL_DAYS} => NO extra rebalance")

    # TEST 3 — a genuinely due rebalance is still preserved across restart
    e2._trading_days_since_rebal = REBAL_DAYS
    e2._save_rebal_state()
    e3 = IBKREngine()
    assert e3._trading_days_since_rebal == REBAL_DAYS
    print(f"PASS  due rebalance preserved -> counter={e3._trading_days_since_rebal} (>= {REBAL_DAYS} => rebalances, correct)")

    print("\nALL TESTS PASSED — restart now resumes the cadence instead of resetting it.")
finally:
    if os.path.exists(bak):
        os.replace(bak, REBAL_STATE_FILE)
    elif os.path.exists(REBAL_STATE_FILE):
        os.remove(REBAL_STATE_FILE)
