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

print(f"\n{len(FAILS)} failures" + ("" if not FAILS else f": {FAILS}"))
sys.exit(1 if FAILS else 0)
