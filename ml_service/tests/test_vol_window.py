"""Vol-scale window: 40 close-to-close NAV returns ending at the LAST COMPLETED close.

WHY THIS TEST EXISTS (2026-09-30)
  record_nav() seeds today's entry whenever the engine starts. On 2026-09-30 the engine restarted at
  00:21 ET after IBKR's nightly reset and wrote the previous evening's after-hours NAV as "2026-09-30".
  compute_vol_scale() included it, so the 09:32 book-3 target was 1.46x instead of 1.41x: the leverage
  depended on whether the engine happened to restart overnight. The logged 2026-09-09 value (realized
  26.1%, scale 0.86) is reproduced exactly only by the completed-sessions window.
Run: python3 tests/test_vol_window.py
"""
import os
import sys
import types
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if "ib_insync" not in sys.modules:
    _stub = types.ModuleType("ib_insync")
    _stub.__all__ = ["IB", "MarketOrder", "Position", "Stock", "util"]
    for _n in _stub.__all__:
        setattr(_stub, _n, type(_n, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules["ib_insync"] = _stub
sys.argv = ["x"]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import ibkr_engine as E  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


ET = ZoneInfo("US/Eastern")
days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end="2026-09-29", periods=60)]
rng = np.random.default_rng(7)
nav = 50_000 * np.cumprod(1 + rng.normal(0, 0.022, len(days)))   # ~35% ann. vol: scale below the 1.0 cap
nav[-41] *= 0.96                                   # a big day right at the window edge (makes the bug visible)
hist = [[d, float(v)] for d, v in zip(days, nav)]
seed = [["2026-09-30", float(nav[-1]) * 1.002]]   # the pre-session seed (last night's after-hours NAV)


def ref(h):
    last = h[-41:]
    r = [last[i][1] / last[i - 1][1] - 1 for i in range(1, len(last))]
    m = sum(r) / len(r)
    v = (sum((x - m) ** 2 for x in r) / len(r)) ** 0.5 * 252 ** 0.5
    return min(1.0, max(E.VOL_SCALE_FLOOR, E.VOL_TARGET / v)), v


class _Pin(datetime):
    PIN = None

    @classmethod
    def now(cls, tz=None):
        return cls.PIN.astimezone(tz) if tz else cls.PIN.replace(tzinfo=None)


eng = E.IBKREngine.__new__(E.IBKREngine)
eng._load_flows = lambda: []
eng._is_trading_day = lambda d=None: True
_real = E.datetime
try:
    E.datetime = _Pin
    eng._load_nav_history = lambda: hist + seed
    _Pin.PIN = datetime(2026, 9, 30, 9, 32, tzinfo=ET)
    vs, rv = eng.compute_vol_scale()
    want, wv = ref(hist)
    check("09:32 rebalance ignores today's pre-session seed (window ends at yesterday's close)", abs(vs - want) < 1e-12 and abs(rv - wv) < 1e-12, f"{vs} vs {want}")
    bad, _ = ref(hist + seed)
    check("…and that differs from the seeded window (the bug was real in this fixture)", abs(bad - want) > 1e-4, f"{bad} vs {want}")
    _Pin.PIN = datetime(2026, 9, 30, 16, 15, tzinfo=ET)
    vs2, _ = eng.compute_vol_scale()
    check("after the 16:05 EOD close mark, today's entry counts", abs(vs2 - ref(hist + seed)[0]) < 1e-12)
    eng._load_nav_history = lambda: hist
    _Pin.PIN = datetime(2026, 9, 30, 9, 32, tzinfo=ET)
    vs3, _ = eng.compute_vol_scale()
    check("no seed present: unchanged behaviour", abs(vs3 - want) < 1e-12)
finally:
    E.datetime = _real

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all vol-window tests passed"))
sys.exit(1 if FAILS else 0)
