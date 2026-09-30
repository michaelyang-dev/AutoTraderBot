"""Tranched rebalance path (decision 2026-09-08: deploy FINAL @1.49x).

WHY THESE TESTS EXIST
  The live book becomes 4 virtual sub-books rebalanced on a 5-session stagger. The failure modes
  that matter are bookkeeping ones: a book selling another book's shares, the books drifting away
  from the broker's actual position after a stop or a manual trade, the transition split losing
  or inventing shares, the cadence firing on the wrong day, a stop firing on the wrong book, and
  the legacy single-book path changing when TRANCHES=1. Every test below is offline: ib_insync is
  stubbed and no network or broker call is made.
Run: python3 tests/test_tranche_engine.py
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
os.environ.setdefault("IBKR_TRANCHES", "4")
os.environ.setdefault("IBKR_TRANCHE_STRIDE", "5")

import ibkr_engine as E  # noqa: E402

E.send_telegram = lambda m: None
E.IBKREngine._fetch_splits_on = lambda self, day: {}   # no network in tests; split handling: tests/test_split_handling.py
E.STOP_AT_CLOSE = False   # these tests exercise stop LOGIC; the close-window timing is tests/test_stop_at_close.py
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ── 1. transition split: whole shares, nothing lost, remainder to the lowest books ──────────
books = E.IBKREngine._split_books({"AAPL": 10, "MSFT": 3, "NVDA": 1}, 4)
check("split conserves shares", all(sum(b.get(s, 0) for b in books.values()) == q for s, q in {"AAPL": 10, "MSFT": 3, "NVDA": 1}.items()))
check("split remainder goes to the lowest books", books["0"] == {"AAPL": 3, "MSFT": 1, "NVDA": 1} and books["3"] == {"AAPL": 2}, str(books))

# ── 2. reconcile: shortfall (a stop fired) removed from the biggest holder; surplus credited ──
bk = {"0": {"AAPL": 5, "X": 2}, "1": {"AAPL": 3}, "2": {}, "3": {"AAPL": 4}}
r = E.IBKREngine._reconcile_books(bk, {"AAPL": 8, "X": 2, "NEW": 7}, "2")
check("reconcile shortfall taken from the largest holder first", r["0"]["AAPL"] == 1 and r["1"]["AAPL"] == 3 and r["3"]["AAPL"] == 4, str(r))
check("reconcile sums equal the broker", sum(b.get("AAPL", 0) for b in r.values()) == 8)
check("reconcile surplus credited to the given book", r["2"] == {"NEW": 7})
r2 = E.IBKREngine._reconcile_books(bk, {"X": 2}, "0")
check("reconcile a fully-sold name disappears from every book", all("AAPL" not in b for b in r2.values()), str(r2))
check("reconcile is pure (input untouched)", bk["0"]["AAPL"] == 5)

# ── 3. orders for one book: bounded sells, tiny-delta skip, other books untouched ───────────
book = {"AAPL": 10, "OLD": 6, "TRIM": 20}
targets = {"AAPL": 12, "TRIM": 15, "NEWCO": 9}
prices = {"AAPL": 100.0, "TRIM": 50.0, "NEWCO": 30.0, "OLD": 10.0}
positions = {"AAPL": 25, "OLD": 4, "TRIM": 40}          # OLD: broker holds only 4 (a stop sold 2 earlier)
others = {"AAPL": 15, "OLD": 0, "TRIM": 20}
orders = E.IBKREngine._tranche_orders(book, targets, prices, 15_000.0, positions, others)
od = {s: (d, why) for s, d, why in orders}
check("exit sells only what the broker still holds beyond other books", od["OLD"] == (-4, "tranche_exit"), str(orders))
check("trim bounded by actual minus other books", od["TRIM"] == (-5, "tranche_trim"))
check("buy delta on an existing holding", od["AAPL"] == (2, "tranche_rebalance"))
check("new position bought in full", od["NEWCO"] == (9, "tranche_rebalance"))
check("sells are emitted before buys", [d for _, d, _ in orders][:2] == [-4, -5] or all(d < 0 for _, d, _ in orders[:2]))
tiny = E.IBKREngine._tranche_orders({"AAPL": 100}, {"AAPL": 101}, {"AAPL": 30.0}, 15_000.0, {"AAPL": 100}, {})
check("tiny delta on an existing holding is skipped (0.3% of book NAV)", tiny == [], str(tiny))
never = E.IBKREngine._tranche_orders({"AAPL": 10}, {}, {}, 15_000.0, {"AAPL": 10}, {"AAPL": 10})
check("a book can never sell shares that belong to the other books", never == [], str(never))

# ── 4. sizing on a book: NAV/4 with the 15% cap and closed-loop 1.49x target ────────────────
sigs = [{"symbol": f"S{i}", "probability": 0.9 - i * 0.02} for i in range(25)]
px = {f"S{i}": 40.0 + i for i in range(25)}
q, m, gross = E.IBKREngine._calibrate_quantities(sigs, px, 15_000.0, 1.0, 1.0)
check("book sizing hits ~1.49x of the BOOK NAV", 1.35 <= gross / 15_000.0 <= 1.55, f"gross/book {gross/15000:.2f}, mult {m:.2f}")
check("no position above 15% of the book NAV", max(q[s] * px[s] for s in q) <= 0.15 * 15_000.0 * 1.0001 + max(px.values()))
q0, _, g0 = E.IBKREngine._calibrate_quantities(sigs, px, 15_000.0, 1.0, 0.0)
check("credit gate x0.00 sizes the book to zero", g0 == 0.0 and not q0)


# ── 5. cadence + the rebalance flow with a fake broker ──────────────────────────────────────
class FakeEngine(E.IBKREngine):
    def __init__(self, positions, nav, signals, prices):
        self.positions = {s: {"qty": q, "avg_cost": 1.0, "contract": None} for s, q in positions.items()}
        self.trailing_peaks = {}
        self.last_signals = []
        self.last_rebalance = None
        self.account_id = "DU1"
        self._trading_days_since_rebal = 19
        self._last_rebal_date = None
        self._last_counted_day = None
        self._tranche = self._tranche_default()
        self._nav = nav
        self._signals = signals
        self._prices = prices
        self.sold, self.bought = [], []
        self._saved = 0

    def _save_tranche_state(self):
        self._saved += 1

    def _save_rebal_state(self):
        pass

    def _save_trailing_peaks(self):
        pass

    def fetch_signals(self):
        return self._signals

    async def update_positions(self):
        pass

    async def get_portfolio_value(self):
        return self._nav

    async def get_market_price(self, contract):
        return self._prices.get(getattr(contract, "symbol", None), None) or self._cur_sym_price

    def compute_vol_scale(self):
        return 1.0, 0.20

    def compute_credit_derisk(self):
        return 1.0, {"ok": True, "latest": 3.0, "pctile": 0.5}

    async def sell_position(self, symbol, qty, reason="rebalance"):
        actual = self.positions.get(symbol, {}).get("qty", 0)
        q = min(qty, actual)
        self.sold.append((symbol, q, reason))
        self.positions[symbol]["qty"] = actual - q
        if self.positions[symbol]["qty"] <= 0:
            self.positions.pop(symbol)

    async def buy_position(self, symbol, qty, reason="signal"):
        self.bought.append((symbol, qty, reason))
        self.positions.setdefault(symbol, {"qty": 0, "avg_cost": 1.0, "contract": None})["qty"] += qty


# Stock() stub must expose .symbol so the fake price lookup works
class _Stock:
    def __init__(self, symbol, *a, **k):
        self.symbol = symbol


E.Stock = _Stock
signals = [{"symbol": s, "signal": "BUY", "probability": p} for s, p in
           [("AAPL", 0.9), ("MSFT", 0.8), ("NVDA", 0.7), ("AMD", 0.6), ("GOOG", 0.5)]]
prices = {"AAPL": 100.0, "MSFT": 200.0, "NVDA": 50.0, "AMD": 25.0, "GOOG": 150.0, "OLDCO": 10.0}
eng = FakeEngine({"AAPL": 120, "OLDCO": 80, "MSFT": 40}, 60_000.0, signals, prices)
eng._cur_sym_price = None
E.datetime = __import__("datetime").datetime            # real clock for the day-count
st = eng._tranche
check("fresh state: counter one short of the stride (fires on the first NEW trading day, never mid-session on deploy day)",
      st["stride_counter"] == E.TRANCHE_STRIDE - 1)
# deploy-day guard: if the engine starts DURING a session, today is pre-marked as counted -> no rebuild today
from zoneinfo import ZoneInfo
_now = E.datetime.now(ZoneInfo("US/Eastern"))
_in_session = _now.weekday() < 5 and (9, 30) <= (_now.hour, _now.minute) < (16, 0)
check("deploy-day guard marks today as counted iff started in-session", (st["last_counted_day"] == _now.date().isoformat()) == _in_session, f"in_session={_in_session} last_counted_day={st['last_counted_day']}")
if _in_session:
    asyncio.run(eng.rebalance_tranche())
    check("started in-session: no trade on the deploy day", not eng.sold and not eng.bought and not st["initialized"])
# simulate the next trading day: clear the day mark so the count runs
st["last_counted_day"] = None
eng.trailing_peaks["AAPL"] = 130.0                      # legacy single-book peak to be carried into the books
asyncio.run(eng.rebalance_tranche())
check("legacy peak carried into every book slice at the transition (plain key pruned at the first stop check)",
      all(eng.trailing_peaks.get(f"{t}:AAPL") == 130.0 for t in ("0", "1", "2", "3")), str({k: v for k, v in eng.trailing_peaks.items() if "AAPL" in k}))
st = eng._tranche
check("transition initialized and book 0 rebuilt", st["initialized"] and st["last_tranche"] == 0 and st["next_tranche"] == 1)
check("counter reset after the tranche day", st["stride_counter"] == 0)
b0, b_others = st["books"]["0"], [st["books"][k] for k in ("1", "2", "3")]
check("books 1-3 keep their transition shares untouched", b_others[0] == {"AAPL": 30, "OLDCO": 20, "MSFT": 10} and b_others[2] == {"AAPL": 30, "OLDCO": 20, "MSFT": 10}, str(b_others))
check("book 0 sold its OLDCO slice only (20 of 80)", ("OLDCO", 20, "tranche_exit") in eng.sold and eng.positions["OLDCO"]["qty"] == 60, str(eng.sold))
total_books = {}
for b in st["books"].values():
    for s_, q_ in b.items():
        total_books[s_] = total_books.get(s_, 0) + q_
actual = {s_: p_["qty"] for s_, p_ in eng.positions.items()}
check("after the tranche day, books sum to the broker's positions", total_books == actual, f"books {total_books} vs broker {actual}")
gross0 = sum(q_ * prices[s_] for s_, q_ in b0.items())
# only 5 signal names here, so the 15% cap binds: max gross = 5 x 15% = 0.75x of the book NAV
check("book 0 gross = cap-bound 5 x 15% of NAV/4 (25-name sizing hits 1.49x, tested above)", 0.70 * 15_000 <= gross0 <= 0.75 * 15_000, f"{gross0:,.0f}")
check("book 0 holds only signal names", set(b0) <= {s["symbol"] for s in signals}, str(b0))
check("per-book peaks set for book 0 BUYS (trimmed transition holdings get theirs at the first stop check)",
      all(f"0:{s_}" in eng.trailing_peaks for s_, _, _ in eng.bought))
# second call the same day: nothing happens (counter 0/5, same calendar day)
n_sold, n_bought = len(eng.sold), len(eng.bought)
asyncio.run(eng.rebalance_tranche())
check("same-day second call does not trade", (len(eng.sold), len(eng.bought)) == (n_sold, n_bought))

# ── 6. per-book trailing stop sells only that book's slice ───────────────────────────────────
eng.trailing_peaks["1:AAPL"] = 200.0            # book 1 bought at a higher peak; book 0/2/3 peaks at 100
eng.trailing_peaks["0:AAPL"] = 100.0
eng._cur_sym_price = None
eng._prices = dict(prices)
E.TRAILING_STOP = 0.40
before = eng.positions["AAPL"]["qty"]
asyncio.run(eng._check_trailing_stops_tranched())
sold_aapl = [x for x in eng.sold if x[0] == "AAPL"]
check("stop fires for the book whose own peak is breached (book 1: 100 vs peak 200 = -50%)", any(x[2].startswith("trailing_stop book 1") for x in sold_aapl), str(sold_aapl))
check("stop sold exactly book 1's slice (30 shares), other books keep theirs", any(x[1] == 30 for x in sold_aapl) and "AAPL" not in eng._tranche["books"]["1"] and eng._tranche["books"]["2"]["AAPL"] == 30, str(eng._tranche["books"]))
check("book 0 (peak 100, price 100) not stopped", "AAPL" in eng._tranche["books"]["0"])

# ── 7. legacy single-book path untouched when TRANCHES=1 ──────────────────────────────────
E.TRANCHES = 1
check("_rebal_left falls back to the 20-day clock", eng._rebal_left() == max(0, E.REBAL_DAYS - eng._trading_days_since_rebal))
E.TRANCHES = 4

# ── 8. multi-week scenario with a fake clock: rotation, external sell, restart, invariants ──
import datetime as _dt
import json as _json
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo as _ZI


class FakeDT(_dt.datetime):
    _now = _dt.datetime(2026, 9, 9, 10, 0, tzinfo=_ZI("US/Eastern"))

    @classmethod
    def now(cls, tz=None):
        return cls._now


def advance_session():
    d = FakeDT._now + _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    FakeDT._now = d


E.datetime = FakeDT
tmp = Path(tempfile.mkdtemp()) / "tranche_state.json"
E.TRANCHE_STATE_FILE = tmp


class ScenarioEngine(FakeEngine):
    def _save_tranche_state(self):
        E.IBKREngine._save_tranche_state(self)

    def _save_trailing_peaks(self):
        pass


N = 25
sig25 = [{"symbol": f"N{i:02d}", "signal": "BUY", "probability": 0.95 - i * 0.02} for i in range(N)]
px25 = {f"N{i:02d}": 20.0 + 2 * i for i in range(N)}
NAV = 60_000.0
legacy = {f"N{i:02d}": int(1.49 * NAV / N / px25[f"N{i:02d}"]) for i in range(N)}
eng = ScenarioEngine(legacy, NAV, sig25, px25)
eng._cur_sym_price = None
eng._tranche = eng._tranche_default(); eng._tranche["last_counted_day"] = None   # deployed after the close
rebuilt = []
inv_ok = True


def invariants(e, label):
    global inv_ok
    books = e._tranche["books"]
    tot = {}
    for b in books.values():
        for s_, q_ in b.items():
            tot[s_] = tot.get(s_, 0) + q_
            if q_ <= 0:
                inv_ok = False; print(f"    non-positive book qty at {label}: {s_} {q_}")
    actual = {s_: p_["qty"] for s_, p_ in e.positions.items() if p_["qty"] > 0}
    if tot != actual:
        inv_ok = False; print(f"    books != broker at {label}: {tot} vs {actual}")


for session in range(1, 23):
    before = dict(eng._tranche)
    asyncio.run(eng.rebalance_tranche())
    if eng._tranche["last_tranche_rebal"] == FakeDT._now.date().isoformat() and eng._tranche.get("last_tranche") is not None and eng._tranche["stride_counter"] == 0 and before.get("stride_counter") != 0:
        rebuilt.append((session, eng._tranche["last_tranche"]))
        b = eng._tranche["books"][str(eng._tranche["last_tranche"])]
        gross = sum(q_ * px25[s_] for s_, q_ in b.items())
        if not (1.30 <= gross / (NAV / 4) <= 1.60):
            inv_ok = False; print(f"    rebuilt book gross {gross/(NAV/4):.2f}x at session {session}")
        if not set(b) <= set(px25):
            inv_ok = False
    if session == 8:                                    # external event: broker shows a stop/manual sell of N03
        eng.positions["N03"]["qty"] = max(1, eng.positions["N03"]["qty"] // 2)
    if eng._tranche.get("initialized"):                 # the engine runs the (reconciling) stop check every cycle
        asyncio.run(eng._check_trailing_stops_tranched())
    invariants(eng, f"session {session}")
    if session == 12:                                   # restart: a NEW engine instance loads the persisted state
        saved = _json.load(open(tmp))
        eng2 = ScenarioEngine({s_: p_["qty"] for s_, p_ in eng.positions.items()}, NAV, sig25, px25)
        eng2._cur_sym_price = None
        eng2._tranche = eng2._load_tranche_state()
        eng2.trailing_peaks = dict(eng.trailing_peaks)
        check("restart reloads the persisted tranche state", eng2._tranche["next_tranche"] == saved["next_tranche"] and eng2._tranche["books"] == {k: {s: int(q) for s, q in v.items()} for k, v in saved["books"].items()})
        eng = eng2
    advance_session()

check("rotation: books rebuilt on sessions 1,6,11,16,21 as 0,1,2,3,0", rebuilt == [(1, 0), (6, 1), (11, 2), (16, 3), (21, 0)], str(rebuilt))
check("invariants held every session (books == broker, positive qty, rebuilt book ~1.49x NAV/4, signal names only)", inv_ok)
check("external sell on session 8 was absorbed by reconcile (N03 books == broker)", sum(b.get("N03", 0) for b in eng._tranche["books"].values()) == eng.positions.get("N03", {"qty": 0})["qty"])
sells_by_kind = {}
for _, _, why in eng.sold:
    sells_by_kind[why.split(" ")[0]] = sells_by_kind.get(why.split(" ")[0], 0) + 1
check("no legacy-style full-book liquidation occurred (sells are tranche_exit/trim only)", set(sells_by_kind) <= {"tranche_exit", "tranche_trim"}, str(sells_by_kind))

# ── 9. INCIDENT 2026-09-16: a crash mid-rebuild must leave an accurate ledger (persisted after every fill) ──
class CrashEngine(ScenarioEngine):
    """buy_position raises on the 3rd buy, emulating the watchdog kill mid-rebuild."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self._nbuy = 0; self.saves = 0; self.beats = 0

    def _save_tranche_state(self):
        self.saves += 1; E.IBKREngine._save_tranche_state(self)

    async def buy_position(self, symbol, qty, reason="signal"):
        self._nbuy += 1
        if self._nbuy == 3:
            raise RuntimeError("watchdog kill")
        return await super().buy_position(symbol, qty, reason)


tmp2 = Path(tempfile.mkdtemp()) / "tranche_state.json"; E.TRANCHE_STATE_FILE = tmp2
FakeDT._now = _dt.datetime(2026, 9, 16, 9, 35, tzinfo=_ZI("US/Eastern"))
legacy2 = {f"N{i:02d}": int(1.49 * NAV / N / px25[f"N{i:02d}"]) for i in range(N)}
ce = CrashEngine(legacy2, NAV, sig25, px25); ce._cur_sym_price = None
ce._tranche = ce._tranche_default(); ce._tranche["last_counted_day"] = None; ce._tranche["initialized"] = True
ce._tranche["books"] = E.IBKREngine._split_books(legacy2, 4)
ce._last_progress = 0.0
try:
    asyncio.run(ce.rebalance_tranche())
    crashed = False
except RuntimeError:
    crashed = True
check("crash mid-rebuild reproduced", crashed)
saved = _json.load(open(tmp2)); book0_saved = {s: int(q) for s, q in saved["books"]["0"].items()}
actual = {s_: p_["qty"] for s_, p_ in ce.positions.items() if p_["qty"] > 0}
tot = {}
for b in saved["books"].values():
    for s_, q_ in b.items(): tot[s_] = tot.get(s_, 0) + int(q_)
check("ledger on disk reflects every fill executed before the crash (books == broker)", tot == actual, f"diffs {[(s_, tot.get(s_), actual.get(s_)) for s_ in set(tot)|set(actual) if tot.get(s_) != actual.get(s_)][:5]}")
check("ledger was persisted after each fill, not only at the end", ce.saves >= len(ce.sold) + len(ce.bought))
check("watchdog heartbeat updated during the rebuild", ce._last_progress > 0)
check("the interrupted rebuild did NOT advance the counter (retry will complete it)", saved["stride_counter"] >= E.TRANCHE_STRIDE and saved["next_tranche"] == 0)
# the retry: a fresh engine loads the persisted ledger, reconciles against the broker, and only completes what is missing
re_ = ScenarioEngine({s_: p_["qty"] for s_, p_ in ce.positions.items()}, NAV, sig25, px25); re_._cur_sym_price = None
re_._tranche = re_._load_tranche_state(); re_.trailing_peaks = dict(ce.trailing_peaks)
asyncio.run(re_.rebalance_tranche())
resold = [s_ for s_, q_, _ in re_.sold if s_ in {x[0] for x in ce.sold}]
check("retry does not re-sell what the first attempt already sold (no phantom exits)", resold == [], str(resold))
check("retry completes book 0 and the account reconciles", re_._tranche["last_tranche"] == 0 and re_._tranche["stride_counter"] == 0)
tot2 = {}
for b in re_._tranche["books"].values():
    for s_, q_ in b.items(): tot2[s_] = tot2.get(s_, 0) + q_
check("after the retry, books == broker", tot2 == {s_: p_["qty"] for s_, p_ in re_.positions.items() if p_["qty"] > 0})

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all tranche tests passed"))
sys.exit(1 if FAILS else 0)
