"""Regression tests for the daily data-refresh step runner (scripts/refresh_data.py), added after the
2026-09-29 false "DATA REFRESH PARTIAL: enhanced_data" alert.

Failure modes of the old SIGALRM wrapper that these tests pin down:
  * a timeout raised INSIDE a step can be swallowed by the step's own `except Exception`;
  * a step that catches its error and RETURNS False was never retried;
  * a step could report success without writing its outputs.
Plus a parity guard: the refresh step table classifies the FMP 'enhanced' data as NOT feeding live
trading because signal_server builds live signals with enhanced_data=None. If that call ever changes,
this test fails so the classification (and the alert severity) is revisited.

Run: python3 tests/test_refresh_runner.py
"""
import ast
import os
import sys
import tempfile
import time
from pathlib import Path

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "scripts"))
import refresh_data as R  # noqa: E402

FAILS = []
TMP = Path(tempfile.mkdtemp(prefix="refresh_runner_test_"))
COUNTER = TMP / "attempts"


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _bump():
    n = int(COUNTER.read_text()) if COUNTER.exists() else 0
    COUNTER.write_text(str(n + 1))
    return n + 1


def _attempts():
    return int(COUNTER.read_text()) if COUNTER.exists() else 0


def _reset():
    COUNTER.unlink(missing_ok=True)


# ---- step bodies (module level so the forked child can run them) -------------------------------
def step_ok():
    _bump()
    return True


def step_returns_false():
    _bump()
    return False


def step_raises():
    _bump()
    raise RuntimeError("boom")


def step_flaky():
    return _bump() >= 2          # fails on attempt 1, succeeds on attempt 2


def step_hangs_and_swallows():
    _bump()
    while True:                  # the exact pattern that defeated SIGALRM: a broad except in a loop
        try:
            time.sleep(0.1)
        except Exception:
            pass


OUT = TMP / "out.parquet"


def step_writes_output():
    _bump()
    OUT.write_text("x")
    return True


def step_claims_ok_writes_nothing():
    _bump()
    return True


print("refresh step runner")
_reset(); ok, d = R._run_step(step_ok, "ok", 10, retries=1)
check("clean step succeeds on the first attempt", ok and _attempts() == 1, f"{ok} {d} attempts={_attempts()}")

_reset(); ok, d = R._run_step(step_returns_false, "false", 10, retries=1)
check("a step that RETURNS False is retried (old wrapper never retried it)", (not ok) and _attempts() == 2, f"{d} attempts={_attempts()}")
check("…and the failure reason says it returned False", "returned False" in d, d)

_reset(); ok, d = R._run_step(step_raises, "raises", 10, retries=1)
check("a raising step is retried and reported as an exception", (not ok) and _attempts() == 2 and "exception" in d, f"{d} attempts={_attempts()}")

_reset(); ok, d = R._run_step(step_flaky, "flaky", 10, retries=1)
check("a flaky step succeeds on its retry", ok and _attempts() == 2 and "attempt 2" in d, f"{ok} {d}")

_reset(); t0 = time.time(); ok, d = R._run_step(step_hangs_and_swallows, "hang", 2, retries=0)
el = time.time() - t0
check("a hung step that swallows exceptions is KILLED at its budget", (not ok) and "timed out" in d and el < 20, f"{d} elapsed={el:.1f}s")

_reset(); OUT.unlink(missing_ok=True)
ok, d = R._run_step(step_writes_output, "writes", 10, retries=0, outputs=[])
check("output-less step with no declared outputs passes", ok, d)

# declared outputs are resolved relative to ML_DIR; point at the temp file with a relative path
_reset(); OUT.unlink(missing_ok=True)
rel = os.path.relpath(OUT, R.ML_DIR)
ok, d = R._run_step(step_writes_output, "writes", 10, retries=0, outputs=[(rel, 1)])
check("a step that writes its declared output passes the post-condition", ok, d)

_reset(); OUT.unlink(missing_ok=True)
ok, d = R._run_step(step_claims_ok_writes_nothing, "liar", 10, retries=1, outputs=[(rel, 1)])
check("a step that claims success but writes nothing FAILS the post-condition (and is retried)",
      (not ok) and "outputs not fresh" in d and _attempts() == 2, f"{d} attempts={_attempts()}")

print("step table")
names = {row[0]: row for row in R.STEPS}
check("every step's function exists in refresh_data", all(callable(getattr(R, row[1], None)) for row in R.STEPS),
      [row[1] for row in R.STEPS if not callable(getattr(R, row[1], None))])
check("every budget is positive and retries >= 0", all(row[2] > 0 and row[3] >= 0 for row in R.STEPS))
check("enhanced_fmp budget covers the weekly FMP refresh (>= 1200s)", names["enhanced_fmp"][2] >= 1200)
check("enhanced_fmp no longer sweeps options (no fetch_options_snapshots / fetch_all_data.main call)",
      "fetch_options_snapshots" not in R.refresh_enhanced_data.__code__.co_names
      and "main" not in R.refresh_enhanced_data.__code__.co_names)
check("VIX is classified as a LIVE input", names["VIX"][4] is True)
check("enhanced_fmp is classified as NOT live", names["enhanced_fmp"][4] is False)

print("parity guard: live signals are built WITHOUT enhanced data")
tree = ast.parse((ML / "signal_server.py").read_text())
calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and node.args and any(
            isinstance(a, ast.Name) and a.id == "build_signals_v9" for a in node.args):
        args = node.args
        i = [k for k, a in enumerate(args) if isinstance(a, ast.Name) and a.id == "build_signals_v9"][0]
        # run_in_executor(None, build_signals_v9, raw, ENHANCED, top_n, ...): enhanced_data is 2 after the fn
        enh = args[i + 2] if len(args) > i + 2 else None
        calls.append(isinstance(enh, ast.Constant) and enh.value is None)
check("signal_server calls build_signals_v9 at least once", len(calls) >= 1, calls)
check("EVERY live build_signals_v9 call passes enhanced_data=None "
      "(if this fails: re-classify enhanced_fmp as live in refresh_data.STEPS)", calls and all(calls), calls)

print("price archive: final bars only, labelled by their own session (2026-09-30)")
import types  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datetime import datetime as _dt  # noqa: E402
_ET = ZoneInfo("America/New_York")


def _ets(y, mo, d, h, mi=0):
    return _dt(y, mo, d, h, mi, tzinfo=_ET).timestamp()


def _frame(end, n=380, last_close=None):
    ix = pd.bdate_range(end=end, periods=n)
    c = np.linspace(10, 20, n)
    if last_close is not None:
        c[-1] = last_close
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": c, "volume": 1e6}, index=ix)


fb = R._final_bar(_frame("2026-09-29", last_close=77.0), _ets(2026, 9, 29, 17, 52))
check("a file written after the 17:00 settle: its last bar is final", fb is not None and str(fb[0])[:10] == "2026-09-29" and fb[1] == 77.0, fb)
fb = R._final_bar(_frame("2026-09-29", last_close=77.0), _ets(2026, 9, 29, 9, 52))
check("a file written mid-session: the in-progress bar is NOT final; the previous session is", fb is not None and str(fb[0])[:10] == "2026-09-28", fb)

_real_ml = R.ML_DIR
tmpml = Path(tempfile.mkdtemp(prefix="archive_test_"))
(tmpml / "data" / "massive_cache").mkdir(parents=True)
for sym, end, mt, lc in [("A", "2026-09-29", _ets(2026, 9, 29, 17, 52), 101.0), ("B", "2026-09-29", _ets(2026, 9, 29, 17, 52), 202.0),
                         ("C", "2026-09-29", _ets(2026, 9, 29, 17, 53), 303.0), ("PART", "2026-09-29", _ets(2026, 9, 29, 9, 52), 999.0)]:
    fp = tmpml / "data" / "massive_cache" / f"{sym}_adj.parquet"
    _frame(end, last_close=lc).to_parquet(fp)
    os.utime(fp, (mt, mt))
try:
    R.ML_DIR = tmpml
    ok1 = R.archive_daily_prices()
    arch = tmpml / "data" / "price_archive" / "closes_2026-09-29.parquet"
    a = pd.read_parquet(arch) if arch.exists() else None
    check("archive labelled by the bar's own session (2026-09-29), not the wall clock", ok1 and a is not None and a["date"].iloc[0] == "2026-09-29")
    check("only settled bars archived: A/B/C in, the mid-session snapshot (999.0) out",
          a is not None and {"A", "B", "C"} <= set(a.columns) and "PART" not in a.columns and a["A"].iloc[0] == 101.0, list(a.columns) if a is not None else None)
    before = arch.stat().st_mtime
    ok2 = R.archive_daily_prices()
    check("second run is a no-op (session already archived)", ok2 and arch.stat().st_mtime == before)
finally:
    R.ML_DIR = _real_ml

print("data_gaps yfinance patch: includes the latest session, never overwrites with older data (2026-09-30)")
tmpg = Path(tempfile.mkdtemp(prefix="gaps_test_"))
(tmpg / "data" / "massive_cache").mkdir(parents=True)
import json as _json  # noqa: E402
(tmpg / "data" / "sp1500_members.json").write_text(_json.dumps({"sp500": ["NEWT", "OLDT"], "sp400": [], "sp600": []}))
for sym in ("NEWT", "OLDT"):
    _frame("2026-09-29", n=100).to_parquet(tmpg / "data" / "massive_cache" / f"{sym}_adj.parquet")
seen = {}


def _yf_dl(tickers, start=None, end=None, **k):
    seen["end"] = end
    parts = {}
    for t, e in (("NEWT", "2026-09-29"), ("OLDT", "2026-09-28")):
        ix = pd.bdate_range(end=e, periods=377)
        parts[t] = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": np.linspace(5, 6, 377), "Volume": 1e5}, index=ix)
    return pd.concat(parts, axis=1).swaplevel(0, 1, axis=1)


_fake = types.ModuleType("yfinance"); _fake.download = _yf_dl
_real_yf = sys.modules.get("yfinance"); sys.modules["yfinance"] = _fake
try:
    R.ML_DIR = tmpg
    R.check_data_gaps()
finally:
    R.ML_DIR = _real_ml
    if _real_yf is not None:
        sys.modules["yfinance"] = _real_yf
    else:
        sys.modules.pop("yfinance", None)
from datetime import timedelta as _tdl  # noqa: E402
check("yfinance end = tomorrow (includes the session that just closed)", seen.get("end") == (_dt.today() + _tdl(days=1)).strftime("%Y-%m-%d"), seen)
n_new = len(pd.read_parquet(tmpg / "data" / "massive_cache" / "NEWT_adj.parquet"))
o = pd.read_parquet(tmpg / "data" / "massive_cache" / "OLDT_adj.parquet")
check("patch that reaches the latest session is written (100 -> 377 bars)", n_new == 377, n_new)
check("patch that ends a session EARLY is not written over newer data", len(o) == 100 and str(pd.to_datetime(o.index).max())[:10] == "2026-09-29", (len(o), str(o.index.max())))

import re as _re  # noqa: E402
_mp = (ML / "massive_data_provider.py").read_text()
_mm = _re.search(r"^SETTLE_HOUR_ET\s*=\s*(\d+)", _mp, _re.M)
check("archive settle hour == price-cache settle hour", _mm is not None and int(_mm.group(1)) == R.ARCHIVE_SETTLE_HOUR_ET)

print(f"\n{len(FAILS)} failures" + ("" if not FAILS else f": {FAILS}"))
sys.exit(1 if FAILS else 0)
