"""Ticker-reuse splice guard (signal_builder._splice_guard).

WHY: 2026-09-08 the live history for "BNY" (BNY Mellon since 2026-05-21) began with the ~$10.5 closed-end fund
that held the ticker before, giving a fake +1,608% 12-month return and the #2 momentum rank. The PERMNO-keyed
backtest never sees this. The guard discards everything up to the last one-day close ratio > 4x.
Run: python3 tests/test_splice_guard.py
"""
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if "ib_insync" not in sys.modules:
    sys.modules["ib_insync"] = types.ModuleType("ib_insync")
import signal_builder as SB  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


idx = pd.bdate_range("2025-03-07", periods=306)
fund = np.full(200, 10.5) * np.cumprod(1 + np.random.default_rng(0).normal(0, 0.002, 200))
bank = np.full(106, 137.0) * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 106))
bny = pd.DataFrame({"close": np.concatenate([fund, bank])}, index=idx)          # the live BNY splice
gme = pd.DataFrame({"close": np.concatenate([np.full(100, 20.0), [46.9], np.full(50, 45.0)])}, index=pd.bdate_range("2020-09-01", periods=151))   # +134.8%, genuine
clean = pd.DataFrame({"close": np.linspace(50, 80, 300)}, index=pd.bdate_range("2025-01-01", periods=300))
tiny = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.bdate_range("2026-01-01", periods=2))
raw = {"BNY": bny, "GME": gme, "CLEAN": clean, "TINY": tiny, "EMPTY": pd.DataFrame(), "NOCLOSE": pd.DataFrame({"open": [1, 2, 3]}, index=pd.bdate_range("2026-01-01", periods=3))}
out, spliced = SB._splice_guard(raw)
check("BNY detected as a splice at the jump date", [s for s, _, _ in spliced] == ["BNY"] and spliced[0][1] == str(idx[200].date()), str(spliced))
check("BNY history truncated to the bars after the jump (106 bars, too short for 200/252-day features)", len(out["BNY"]) == 106 and float(out["BNY"]["close"].iloc[0]) == float(bank[0]))
check("GME's genuine +134.8% day is NOT treated as a splice", len(out["GME"]) == 151)
check("clean, tiny, empty and no-close frames pass through unchanged", len(out["CLEAN"]) == 300 and len(out["TINY"]) == 2 and len(out["EMPTY"]) == 0 and len(out["NOCLOSE"]) == 3)
check("input frames not mutated", len(raw["BNY"]) == 306)
check("threshold is 4x", SB.SPLICE_MAX_RATIO == 4.0)
# a double splice keeps only the history after the LAST jump
dbl = pd.DataFrame({"close": np.concatenate([np.full(50, 1.0), np.full(50, 10.0), np.full(50, 100.0)])}, index=pd.bdate_range("2025-01-01", periods=150))
o2, s2 = SB._splice_guard({"X": dbl})
check("double splice -> keep only bars after the last jump", len(o2["X"]) == 50 and s2[0][2] == 10.0, str(s2))

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all splice-guard tests passed"))
sys.exit(1 if FAILS else 0)
