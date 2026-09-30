"""Stock splits must be invisible to the live book, as they are to the (split-adjusted) backtest.

WHY THIS TEST EXISTS (2026-09-30)
  The engine never adjusted its per-book ledger or its trailing-stop peaks for a split. After a 2:1 split the price
  halves overnight against a pre-split peak (-50% < -40%), so the stop fired in EVERY book holding the name at the next
  open, and _reconcile_books credited all the new shares to one book. The control case below reproduces that with
  split handling OFF; the rest pin the fix. No network, no broker.
Run: python3 tests/test_split_handling.py
"""
import asyncio
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
sys.argv = ["x"]
os.environ.setdefault("IBKR_TRANCHES", "4")
import ibkr_engine as E  # noqa: E402

ALERTS = []
E.send_telegram = lambda m: ALERTS.append(m)
E.STOP_AT_CLOSE = False   # these tests exercise stop LOGIC; the close-window timing is tests/test_stop_at_close.py
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class _Stock:
    def __init__(self, symbol, *a, **k):
        self.symbol = symbol


E.Stock = _Stock

print("pure split adjustment")
books = {"0": {"MU": 1, "X": 5}, "1": {}, "2": {"MU": 1}, "3": {"MU": 3}}
peaks = {"0:MU": 1100.0, "2:MU": 1080.0, "3:MU": 1066.0, "0:X": 50.0}
b2, p2 = E.IBKREngine._split_adjust(books, peaks, "MU", 2.0)
check("2-for-1: every book's shares doubled", b2["0"]["MU"] == 2 and b2["2"]["MU"] == 2 and b2["3"]["MU"] == 6, str(b2))
check("2-for-1: every book's peak halved", p2["0:MU"] == 550.0 and p2["2:MU"] == 540.0 and p2["3:MU"] == 533.0, str(p2))
check("other names untouched; inputs not mutated", b2["0"]["X"] == 5 and p2["0:X"] == 50.0 and books["0"]["MU"] == 1 and peaks["0:MU"] == 1100.0)
b3, p3 = E.IBKREngine._split_adjust({"0": {"Y": 3}, "1": {"Y": 5}}, {"0:Y": 90.0, "1:Y": 60.0}, "Y", 1.5)
check("3-for-2: shares floored (cash in lieu), peaks / 1.5", b3["0"]["Y"] == 4 and b3["1"]["Y"] == 7 and abs(p3["0:Y"] - 60.0) < 1e-9, str((b3, p3)))
b4, p4 = E.IBKREngine._split_adjust({"0": {"Z": 25}, "1": {"Z": 5}}, {"0:Z": 2.0, "1:Z": 3.0}, "Z", 0.1)
check("1-for-10 reverse: 25 -> 2 shares, peak x10; a 5-share slice becomes 0 and leaves the book",
      b4["0"]["Z"] == 2 and abs(p4["0:Z"] - 20.0) < 1e-9 and "Z" not in b4["1"] and "1:Z" not in p4, str((b4, p4)))
check("split-like detector", E.IBKREngine._split_like({"0": {"A": 3}, "1": {"B": 10}}, {"A": 6, "B": 9}) == {"A"})


class Eng(E.IBKREngine):
    def __init__(self, books, peaks, qty, px, splits=None, cal_error=False):
        self._tranche = {"initialized": True, "books": {k: dict(v) for k, v in books.items()}, "next_tranche": 3}
        self.trailing_peaks = dict(peaks)
        self.positions = {s: {"qty": q, "avg_cost": 1.0, "contract": None} for s, q in qty.items()}
        self._px, self._splits, self._cal_error = px, splits or {}, cal_error
        self.sold = []

    def _fetch_splits_on(self, day):
        if self._cal_error:
            raise IOError("calendar down")
        return self._splits

    def _save_tranche_state(self):
        pass

    def _save_trailing_peaks(self):
        pass

    async def get_market_price(self, contract):
        return self._px[contract.symbol]

    async def sell_position(self, symbol, qty, reason="rebalance"):
        self.sold.append((symbol, qty, reason))
        self.positions[symbol]["qty"] -= qty


BOOKS = {"0": {"MU": 1}, "1": {}, "2": {"MU": 1}, "3": {"MU": 1}}
PEAKS = {"0:MU": 1100.0, "2:MU": 1080.0, "3:MU": 1066.0}
AFTER = {"MU": 6}                                      # the broker after a 2-for-1 on 3 shares
PX = {"MU": 533.0}                                     # the pre-split close 1066 / 2

print("the bug, reproduced (split handling OFF)")
E.SPLIT_CHECK = False
e = Eng(BOOKS, PEAKS, AFTER, PX)
asyncio.run(e._check_trailing_stops_tranched())
check("CONTROL: without the fix the stop fires in every book holding the name", len(e.sold) == 3 and all(r.startswith("trailing_stop") for _, _, r in e.sold), str(e.sold))
E.SPLIT_CHECK = True

print("the fix")
ALERTS.clear()
e = Eng(BOOKS, PEAKS, AFTER, PX, splits={"MU": 2.0, "OTHER": 4.0})
asyncio.run(e._check_trailing_stops_tranched())
check("no false stop on the split day", e.sold == [], str(e.sold))
check("every book's shares doubled (not all credited to one book)", {t: b.get("MU") for t, b in e._tranche["books"].items() if b.get("MU")} == {"0": 2, "2": 2, "3": 2}, str(e._tranche["books"]))
check("peaks divided by the ratio", e.trailing_peaks == {"0:MU": 550.0, "2:MU": 540.0, "3:MU": 533.0}, str(e.trailing_peaks))
check("owner told about the split", any("MU" in a and "split" in a for a in ALERTS), str(ALERTS))
check("a split of a name we do not hold is ignored", "OTHER" not in str(e._tranche["books"]))
asyncio.run(e._check_trailing_stops_tranched())
check("idempotent: a second check the same day does not re-apply it", {t: b.get("MU") for t, b in e._tranche["books"].items() if b.get("MU")} == {"0": 2, "2": 2, "3": 2} and e.trailing_peaks["3:MU"] == 533.0)
e._px = {"MU": 533.0 * 0.59}                          # a REAL post-split crash still stops out
asyncio.run(e._check_trailing_stops_tranched())
check("a genuine 41% fall after the split still triggers the stop", len(e.sold) == 3, str(e.sold))

print("calendar unavailable: fail-safe")
ALERTS.clear()
e = Eng(BOOKS, PEAKS, AFTER, PX, cal_error=True)
asyncio.run(e._check_trailing_stops_tranched())
check("split-like name is NOT sold", e.sold == [], str(e.sold))
check("its ledger is NOT reconciled into one book", {t: b.get("MU") for t, b in e._tranche["books"].items() if b.get("MU")} == {"0": 1, "2": 1, "3": 1}, str(e._tranche["books"]))
check("owner alerted", any("split" in a.lower() for a in ALERTS), str(ALERTS))
e2 = Eng({"0": {"A": 10}, "1": {}, "2": {}, "3": {}}, {"0:A": 100.0}, {"A": 10}, {"A": 55.0}, cal_error=True)
asyncio.run(e2._check_trailing_stops_tranched())
check("calendar down does not block a normal stop (qty unchanged, -45%)", e2.sold == [("A", 10, e2.sold[0][2])] if e2.sold else False, str(e2.sold))

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all split-handling tests passed"))
sys.exit(1 if FAILS else 0)
