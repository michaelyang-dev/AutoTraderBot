"""/daily must still price the IBKR book when Alpaca's data API is down (2026-09-11: HTTP 504 all morning,
/daily showed no stocks). Fallback: px = IBKR portfolio mark, prev = last 4pm close snapshot value/qty.
Run: python3 tests/test_daily_fallback.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if "ib_insync" not in sys.modules:
    _stub = types.ModuleType("ib_insync")
    _stub.__all__ = ["IB", "MarketOrder", "Position", "Stock", "util"]
    for _n in _stub.__all__:
        setattr(_stub, _n, type(_n, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules["ib_insync"] = _stub
import ibkr_engine as E  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class _C:
    def __init__(self, s):
        self.symbol = s


class _Item:
    def __init__(self, s, pos, px):
        self.contract, self.position, self.marketPrice = _C(s), pos, px


items = [_Item("SNDK", 5, 1700.0), _Item("MU", 4, 990.0), _Item("NEWCO", 3, 10.0), _Item("ZEROPX", 2, 0.0)]
snap = {"date": "2026-09-10", "nav": 60549.24, "positions": [
    {"symbol": "SNDK", "qty": 5, "value": 8814.0}, {"symbol": "MU", "qty": 4, "value": 4112.0}, {"symbol": "ZEROPX", "qty": 2, "value": 100.0}]}
fb = E.IBKREngine._ibkr_fallback_bars(items, snap, "2026-09-11")
check("prices held names from IBKR mark vs yesterday's close", set(fb) == {"SNDK", "MU"}, str(sorted(fb)))
check("prev = snapshot value/qty, px = IBKR mark", abs(fb["SNDK"]["prev"] - 1762.8) < 1e-6 and fb["SNDK"]["px"] == 1700.0 and fb["MU"]["prev"] == 1028.0)
check("names bought today (no prior close) and zero marks are skipped", "NEWCO" not in fb and "ZEROPX" not in fb)
check("bar_date is today and src tags the fallback", fb["MU"]["bar_date"] == "2026-09-11" and fb["MU"]["src"] == "ibkr")
check("a snapshot dated TODAY is not a previous close -> nothing", E.IBKREngine._ibkr_fallback_bars(items, dict(snap, date="2026-09-11"), "2026-09-11") == {})
check("no snapshot / empty items -> nothing, no exception", E.IBKREngine._ibkr_fallback_bars(items, None, "2026-09-11") == {} and E.IBKREngine._ibkr_fallback_bars([], snap, "2026-09-11") == {})
# merge semantics used by /daily: Alpaca rows win, fallback fills the gaps only
bars = {"SNDK": {"px": 1701.0, "close": 1701.0, "prev": 1762.8, "bar_date": "2026-09-11"}}
used = [s for s in fb if s not in bars]
for s in used:
    bars[s] = fb[s]
check("fallback fills only the symbols Alpaca could not price", used == ["MU"] and bars["SNDK"]["px"] == 1701.0)

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all /daily fallback tests passed"))
sys.exit(1 if FAILS else 0)
