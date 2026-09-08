"""Partial-session guard: clock rule AND data rule.

WHY THIS TEST EXISTS
  Third dist_sma200 coverage alarm, 2026-09-08 16:13 ET. The guard was clock-only (`hour < 16`), so at
  16:13 it declared the day final while the vendor had published the day's bar for 421 of 1,504 names.
  The other 1,083 were NaN on the last row -> rolling(200) NaN -> dist_sma200 coverage 28% -> names
  silently dropped from the momentum/lowvol sleeves. A last row materially emptier than the sessions
  before it is not a completed session, whatever the clock says.
Run: python3 tests/test_partial_row_guard.py
"""
import os
import sys
import types
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# signal_builder imports heavy optional deps at module import; stub what is not installed locally
for _m in ("ib_insync",):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import signal_builder as SB  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# synthetic 10-session close matrix for 1,504 names, last row only 28% populated (the 09-08 state)
dates = pd.bdate_range("2026-08-25", periods=10)
n = 1504
full = pd.DataFrame(np.random.default_rng(0).uniform(10, 100, (10, n)), index=dates, columns=[f"S{i}" for i in range(n)])
partial = full.copy()
partial.iloc[-1, 421:] = np.nan          # 421 printed, 1,083 missing
cov_full, cov_partial = SB._last_row_coverage(full), SB._last_row_coverage(partial)
check("row coverage 1.0 on a normal day", abs(cov_full - 1.0) < 1e-9, f"{cov_full:.3f}")
check("row coverage ~0.28 on the 2026-09-08 16:13 state", abs(cov_partial - 421 / 1504) < 1e-9, f"{cov_partial:.3f}")
check("data rule: 28% row is PARTIAL even after 16:00 (yesterday's date, so the clock rule is off)", SB._is_partial_session(partial.index, cov_partial) is True)
check("data rule: 100% row is NOT partial (yesterday's date)", SB._is_partial_session(full.index, cov_full) is False)
check("threshold: 89% row partial, 91% row not", SB._is_partial_session(full.index, 0.89) is True and SB._is_partial_session(full.index, 0.91) is False)
check("None coverage falls back to the clock rule only", SB._is_partial_session(full.index, None) is False)
check("len<2 never partial (caller is never left with nothing)", SB._is_partial_session(full.index[:1], 0.1) is False)

# clock rule preserved: index ending TODAY before 16:00 ET is partial regardless of coverage
from zoneinfo import ZoneInfo  # noqa: E402
now = datetime.now(ZoneInfo("US/Eastern"))
today_idx = pd.DatetimeIndex([pd.Timestamp(now.date() - timedelta(days=1)), pd.Timestamp(now.date())])
expect_clock = now.hour < 16
check("clock rule unchanged: today's row before 16:00 ET is partial", SB._is_partial_session(today_idx, 1.0) is expect_clock, f"hour={now.hour}")

# raw-dict measure agrees with the matrix measure
raw = {}
for i, s in enumerate(partial.columns):
    col = partial[s].dropna()
    raw[s] = pd.DataFrame({"close": col.values}, index=col.index)
raw_cov = SB._raw_last_row_coverage(raw, dates[-1])
check("raw-dict coverage equals the matrix coverage (the two decisions cannot diverge)", abs(raw_cov - cov_partial) < 1e-9, f"raw {raw_cov:.3f} matrix {cov_partial:.3f}")
raw_full = {s: pd.DataFrame({"close": full[s].values}, index=full.index) for s in full.columns[:50]}
check("raw-dict coverage 1.0 on a normal day", abs(SB._raw_last_row_coverage(raw_full, dates[-1]) - 1.0) < 1e-9)

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all partial-row guard tests passed"))
sys.exit(1 if FAILS else 0)
