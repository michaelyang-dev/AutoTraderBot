"""Trailing stops are evaluated once, at the close, with the backtest's rule (EXP-062).

WHY THIS TEST EXISTS (2026-09-30)
  The validated backtest updates each book's peak with the day's CLOSE and stops when the close is <= 60% of it.
  The engine polled all session: peaks rose with intraday highs and it sold on intraday dips. On the deployed
  package (8yr, 12 starts) that cost -4.88pp CAGR / -0.104 Sharpe on 12/12 starts with no drawdown benefit.
Run: python3 tests/test_stop_at_close.py
"""
import asyncio
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
os.environ.setdefault("IBKR_TRANCHES", "4")
import ibkr_engine as E  # noqa: E402

E.send_telegram = lambda m: None
ET = ZoneInfo("US/Eastern")
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class _Stock:
    def __init__(self, symbol, *a, **k):
        self.symbol = symbol


E.Stock = _Stock


class _Pin(datetime):
    PIN = None

    @classmethod
    def now(cls, tz=None):
        return cls.PIN.astimezone(tz) if tz else cls.PIN.replace(tzinfo=None)


class Eng(E.IBKREngine):
    def __init__(self, px, close=(16, 0)):
        self._tranche = {"initialized": True, "next_tranche": 1,
                         "books": {"0": {"A": 10}, "1": {}, "2": {"B": 4}, "3": {}}}
        self.trailing_peaks = {"0:A": 100.0, "2:B": 50.0}
        self.positions = {"A": {"qty": 10, "avg_cost": 1.0, "contract": None}, "B": {"qty": 4, "avg_cost": 1.0, "contract": None}}
        self._px, self._close = dict(px), close
        self.sold = []

    def _fetch_splits_on(self, day):
        return {}

    def _close_time_et(self, d):
        return self._close

    def _save_tranche_state(self):
        pass

    def _save_trailing_peaks(self):
        pass

    async def get_market_price(self, contract):
        return self._px[contract.symbol]

    async def sell_position(self, symbol, qty, reason="rebalance"):
        self.sold.append((symbol, qty, reason))
        self.positions[symbol]["qty"] -= qty


def run_at(eng, hh, mm, day=30):
    _Pin.PIN = datetime(2026, 9, day, hh, mm, tzinfo=ET)
    asyncio.run(eng._check_trailing_stops_tranched())


_real = E.datetime
E.datetime = _Pin
try:
    E.STOP_AT_CLOSE = True
    print("intraday: no stop, no peak change")
    e = Eng({"A": 55.0, "B": 60.0})                     # A -45% intraday (would have stopped before); B a new high
    run_at(e, 11, 0)
    check("a -45% intraday dip does NOT sell (the backtest only looks at the close)", e.sold == [], str(e.sold))
    check("an intraday high does NOT raise the peak", e.trailing_peaks == {"0:A": 100.0, "2:B": 50.0}, str(e.trailing_peaks))
    run_at(e, 15, 49)
    check("15:49 is still outside the 10-minute close window", e.sold == [] and e.trailing_peaks["2:B"] == 50.0)

    print("close window: backtest rule, once")
    e._px = {"A": 60.0, "B": 60.0}                      # A exactly 0.60 x peak -> stop (backtest uses <=)
    run_at(e, 15, 52)
    check("in the window a close <= 60% of the peak sells that book's slice", e.sold == [("A", 10, e.sold[0][2])] if e.sold else False, str(e.sold))
    check("…using the backtest's <= comparison (exactly -40% stops)", bool(e.sold) and "-40.0%" in e.sold[0][2], str(e.sold))
    check("a new closing high raises the peak", e.trailing_peaks.get("2:B") == 60.0, str(e.trailing_peaks))
    check("the day is stamped as evaluated", e._tranche.get("stop_eval_day") == "2026-09-30")
    e._px = {"B": 30.0}
    run_at(e, 15, 57)
    check("evaluated once per day: a second window call does nothing", len(e.sold) == 1 and e.trailing_peaks.get("2:B") == 60.0, str(e.sold))
    _Pin.PIN = datetime(2026, 10, 1, 15, 55, tzinfo=ET)
    asyncio.run(e._check_trailing_stops_tranched())
    check("next day's window evaluates again (B 30 <= 0.6 x 60 -> stop)", any(s == "B" for s, _, _ in e.sold), str(e.sold))

    print("half day (13:00 close)")
    h = Eng({"A": 50.0, "B": 40.0}, close=(13, 0))
    run_at(h, 12, 45)
    check("12:45 is outside a half day's window", h.sold == [])
    run_at(h, 12, 55)
    check("12:55 is inside a half day's window and evaluates", ("A", 10) in [(s, q) for s, q, _ in h.sold], str(h.sold))

    print("flag off = the old intraday behaviour")
    E.STOP_AT_CLOSE = False
    o = Eng({"A": 55.0, "B": 60.0})
    run_at(o, 11, 0)
    check("IBKR_STOP_AT_CLOSE=0: intraday dip sells and intraday high raises the peak", ("A", 10) in [(s, q) for s, q, _ in o.sold] and o.trailing_peaks.get("2:B") == 60.0, str((o.sold, o.trailing_peaks)))
finally:
    E.datetime = _real
    E.STOP_AT_CLOSE = True

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all stop-at-close tests passed"))
sys.exit(1 if FAILS else 0)
