"""ACCOUNT-LEVEL ROUNDING (EXP-063, IBKR_ACCOUNT_ROUNDING = floor | round; default off).

WHY THESE TESTS EXIST
  With the switch on, the 4 books keep FRACTIONAL virtual shares and only the account rounds to whole shares. The
  failure modes that matter are bookkeeping ones again, plus new ones: the account drifting away from the books'
  rounded sum, a switch-on that trades by itself, a stop or an overlay trim that BUYS, a round-up through the 15%
  cap, a restart losing the fractions, a crash mid-rebuild double-trading on the retry, and a rollback to "off"
  that leaves the ledger inconsistent. The rounding rule itself must be the research harness's
  (research/EXP063_account_rounding.py, acct_floor / acct_round) or the backtest evidence does not apply.
  Every test is offline: ib_insync is stubbed, no network or broker call is made.
Run: python3 tests/test_account_rounding.py
"""
import asyncio
import datetime as _dt
import json as _json
import os
import random
import sys
import tempfile
import types
from pathlib import Path
from zoneinfo import ZoneInfo as _ZI

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
E.IBKREngine._fetch_splits_on = lambda self, day: {}
E.STOP_AT_CLOSE = False
X = E.IBKREngine
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ── 1. the rounding rule ───────────────────────────────────────────────────────────────────────────────
check("floor: 4 x 0.58 = 2.32 -> 2 (LITE)", X._acct_whole(2.32, 980.0, 62_000, "floor") == 2)
check("floor: float noise 0.9999999999 -> 1", X._acct_whole(0.9999999999, 50.0, 62_000, "floor") == 1)
check("floor: 0.58 -> 0", X._acct_whole(0.58, 980.0, 62_000, "floor") == 0)
check("round: half up (2.5 -> 3, 2.49 -> 2)", X._acct_whole(2.5, 10.0, 62_000, "round") == 3 and X._acct_whole(2.49, 10.0, 62_000, "round") == 2)
check("round: never UP through 15% of NAV ($1,000 x 3 = $3,000 > 15% of $15,000)",
      X._acct_whole(2.6, 1000.0, 15_000, "round") == 2 and X._acct_whole(2.6, 1000.0, 30_000, "round") == 3)
check("round: no price / NAV -> a round-up is refused", X._acct_whole(2.6, None, 62_000, "round") == 2 and X._acct_whole(2.6, 10.0, None, "round") == 2)
check("round: rounding DOWN never needs a price", X._acct_whole(2.4, None, None, "round") == 2)
check("zero / negative -> 0", X._acct_whole(0.0, 10.0, 1e5, "round") == 0 and X._acct_whole(-1.0, 10.0, 1e5, "floor") == 0)

# LITE across the four books: per-book truncation holds 0 at every rebuild; the account rules do not
slot = 0.58
seq_floor = [X._acct_whole(slot * k, 980.0, 62_000, "floor") for k in range(1, 5)]
seq_round = [X._acct_whole(slot * k, 980.0, 62_000, "round") for k in range(1, 5)]
check("LITE path as books rebuild: floor 0,1,1,2 / round 1,1,2,2 (per-book int(): 0,0,0,0)",
      seq_floor == [0, 1, 1, 2] and seq_round == [1, 1, 2, 2], f"floor {seq_floor} round {seq_round}")

# ── 2. account orders ──────────────────────────────────────────────────────────────────────────────────
before = {"0": {"A": 1.2, "B": 3.0}, "1": {"A": 0.9}, "2": {}, "3": {"C": 0.4}}
after = {"0": {"A": 1.3, "D": 2.0}, "1": {"A": 0.9}, "2": {}, "3": {"C": 0.4}}
tgt = {"A": 2, "B": 3}
o = X._acct_orders(before, after, tgt, {"A": 50.0, "B": 20.0, "D": 10.0}, 62_000, "floor")
od = {s: (d, n) for s, d, n in o}
check("a name whose whole number does not change is not traded (A: 2.1 -> 2.2, still 2)", "A" not in od, str(o))
check("an exited name is sold to 0 (B)", od.get("B") == (-3, 0))
check("a new name is bought to floor(total) (D 2.0 -> 2)", od.get("D") == (2, 2))
check("an unchanged virtual-only name is untouched (C)", "C" not in od)
check("sells are emitted before buys", [d for _, d, _ in o] == sorted([d for _, d, _ in o], key=lambda d: d >= 0))

# ── 3. reconciliation against the broker ───────────────────────────────────────────────────────────────
books = {"0": {"L": 0.58, "A": 10.0}, "1": {"L": 0.58}, "2": {"L": 0.58}, "3": {"L": 0.58, "V": 0.3}}
rb, rt = X._reconcile_virtual(books, {"L": 2, "A": 10}, {"L": 1, "A": 10, "M": 4}, "2")
check("shortfall of 1 whole share removes 1.0 virtual share (largest books first)",
      abs(sum(b.get("L", 0) for b in rb.values()) - 1.32) < 1e-9 and rt["L"] == 1, str(rb))
check("surplus (manual buy of M) credited to the given book and adopted", rb["2"].get("M") == 4.0 and rt["M"] == 4)
check("virtual-only holdings (V 0.3, account 0) survive reconciliation", rb["3"].get("V") == 0.3)
check("reconcile is pure (input untouched)", books["0"]["L"] == 0.58)
rb2, rt2 = X._reconcile_virtual({"0": {"Z": 5.0}, "1": {}, "2": {}, "3": {}}, {"Z": 5}, {}, "0")
check("a name fully sold at the broker leaves the books and the target", "Z" not in rb2["0"] and "Z" not in rt2)

# ── 4. the rebuilt book's virtual targets (the clean room's no-trade band) ─────────────────────────────
nb = X._virtual_book({"A": 10.0, "OLD": 3.0, "T": 1.0}, {"A": 10.01, "NEW": 0.5, "T": 2.0, "TINY": 0.001},
                     {"A": 100.0, "NEW": 900.0, "T": 50.0, "TINY": 30.0}, 15_000.0)
check("existing holding moved < 0.3% of the book keeps its size", nb.get("A") == 10.0)
check("a fractional new target is kept virtually (NEW 0.5 of a $900 share)", nb.get("NEW") == 0.5)
check("a resize above the band is applied", nb.get("T") == 2.0)
check("a tiny NEW position below the band is not opened (clean room)", "TINY" not in nb)
check("a name no longer targeted leaves the book", "OLD" not in nb)

# ── 5. sizing without truncation ───────────────────────────────────────────────────────────────────────
sigs = [{"symbol": f"S{i}", "probability": 0.9 - i * 0.02} for i in range(25)]
px = {f"S{i}": 40.0 + 60 * i for i in range(25)}                    # up to $1,480 a share
qw, _, gw = X._calibrate_quantities(sigs, px, 15_500.0, 0.98, 1.0)
qf, _, gf = X._calibrate_quantities(sigs, px, 15_500.0, 0.98, 1.0, whole=False)
check("fractional sizing hits the target gross exactly", abs(gf - 15_500.0 * 1.49 * 0.98) < 1e-6, f"{gf:.2f}")
check("fractional sizing keeps the expensive names whole-share sizing drops",
      len(qf) == 25 and len(qw) < 25, f"whole {len(qw)} names, fractional {len(qf)}")
check("whole=True (default) unchanged: integers", all(isinstance(v, int) for v in qw.values()))

# ── 6. the de-risk overlay on virtual books, incl. the credit gate at 0 ────────────────────────────────
vb = {"0": {"A": 10.0}, "1": {"A": 20.0, "B": 5.0}, "2": {"A": 2.0}, "3": {}}
pr = {"A": 100.0, "B": 100.0}
ov = X._virtual_overlay(vb, "0", pr, 1_000.0, 1.0)                # target $1,000/book
check("a book above target by > 5% is scaled to the target (book 1: $2,500 -> $1,000)",
      abs(ov["1"]["A"] * 100 + ov["1"]["B"] * 100 - 1_000.0) < 1e-6, str(ov["1"]))
check("the rebuilt book and books within 5% are untouched", ov["0"] == vb["0"] and ov["2"] == vb["2"])
ov0 = X._virtual_overlay(vb, "0", pr, 1_000.0, 0.0)
check("credit gate flat (target 0): every other book is flattened, as the clean room does",
      ov0["1"] == {} and ov0["2"] == {} and ov0["0"] == vb["0"], str(ov0))
check("overlay is pure", vb["1"]["A"] == 20.0)

# ── 7. splits on virtual books ─────────────────────────────────────────────────────────────────────────
sb, sp = X._split_adjust({"0": {"N": 0.58}, "1": {"N": 1.4}}, {"0:N": 1000.0}, "N", 2.0, virtual=True)
check("virtual split scales exactly (0.58 -> 1.16, 1.4 -> 2.8) and divides the peak",
      abs(sb["0"]["N"] - 1.16) < 1e-12 and abs(sb["1"]["N"] - 2.8) < 1e-12 and sp["0:N"] == 500.0)
check("split-like detection uses the account target with virtual books",
      X._split_like({"0": {"N": 0.58}}, {"N": 4}, {"N": 2}) == {"N"} and X._split_like({"0": {"N": 0.58}}, {"N": 2}, {"N": 2}) == set())

# ── 8. PARITY with the research harness's rule (EXP063 ENDDAY, acct_floor / acct_round, real == adjusted px) ─
def harness_step(H0, V, price, nav, mode):
    n0 = round(H0)
    fr = V
    n1 = int(fr + 0.5) if mode == "round" else int(fr + 1e-9)
    if mode == "round" and n1 > fr and n1 * price > 0.15 * nav * 1.0000001:
        n1 -= 1
    return H0 if n1 == n0 else n1


rng = random.Random(7)
mism = 0
for mode in ("floor", "round"):
    for _ in range(5000):
        price = rng.choice([5.0, 37.0, 140.0, 980.0, 1733.0])
        nav = rng.choice([15_000.0, 62_000.0, 250_000.0])
        Vb = rng.random() * 6
        Va = max(0.0, Vb + rng.uniform(-2, 2))
        H0 = X._acct_whole(Vb, price, nav, mode)                    # the position held before the change
        live = X._acct_orders({"0": {"S": Vb}}, {"0": {"S": Va}}, {"S": H0}, {"S": price}, nav, mode)
        live_new = live[0][2] if live else H0
        if live_new != harness_step(H0, Va, price, nav, mode):
            mism += 1
check("live rounding == research harness rule on 10,000 random steps (floor + round)", mism == 0, f"{mism} mismatches")

# ── 9. engine flows with a fake broker ─────────────────────────────────────────────────────────────────
class FakeDT(_dt.datetime):
    _now = _dt.datetime(2026, 10, 7, 10, 0, tzinfo=_ZI("US/Eastern"))

    @classmethod
    def now(cls, tz=None):
        return cls._now


def advance_session():
    d = FakeDT._now + _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    FakeDT._now = d


class _Stock:
    def __init__(self, symbol, *a, **k):
        self.symbol = symbol


E.Stock = _Stock
E.datetime = FakeDT


class Eng(E.IBKREngine):
    def __init__(self, positions, nav, signals, prices, state_file):
        self.positions = {s: {"qty": q, "avg_cost": 1.0, "contract": None} for s, q in positions.items() if q > 0}
        self.trailing_peaks = {}
        self.last_signals = []
        self.last_rebalance = None
        self.account_id = "DU1"
        self._trading_days_since_rebal = 0
        self._last_rebal_date = None
        self._last_counted_day = None
        self._nav, self._signals, self._prices = nav, signals, prices
        self.sold, self.bought = [], []
        self._state_file = state_file
        self._tranche = self._tranche_default()

    def _save_tranche_state(self):
        E.TRANCHE_STATE_FILE = self._state_file
        E.IBKREngine._save_tranche_state(self)

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
        return self._prices.get(getattr(contract, "symbol", None))

    def compute_vol_scale(self):
        return 0.98, 0.22

    def compute_credit_derisk(self):
        return getattr(self, "_derisk", 1.0), {"ok": True, "latest": 3.0, "pctile": 0.1}

    async def sell_position(self, symbol, qty, reason="rebalance"):
        actual = self.positions.get(symbol, {}).get("qty", 0)
        q = min(qty, actual)
        if q <= 0:
            return None
        self.sold.append((symbol, q, reason))
        self.positions[symbol]["qty"] = actual - q
        if self.positions[symbol]["qty"] <= 0:
            self.positions.pop(symbol)

    async def buy_position(self, symbol, qty, reason="signal"):
        if qty <= 0:
            return None
        self.bought.append((symbol, qty, reason))
        self.positions.setdefault(symbol, {"qty": 0, "avg_cost": 1.0, "contract": None})["qty"] += qty


N = 25
SIG = [{"symbol": f"N{i:02d}", "signal": "BUY", "probability": 0.95 - i * 0.02} for i in range(N)]
PX = {f"N{i:02d}": 20.0 + 75.0 * i for i in range(N)}               # $20 .. $1,820 a share
NAV = 62_000.0


def broker(e):
    return {s: p["qty"] for s, p in e.positions.items() if p["qty"] > 0}


def consistent(e, mode, label):
    """After every event: broker == acct_target, and acct_target == the rule applied to the books' sum
    (a name whose whole number was cap-held in round mode may sit one share below)."""
    st = e._tranche
    tot = X._book_totals(st["books"])
    if broker(e) != {s: q for s, q in st["acct_target"].items() if q > 0}:
        return f"{label}: broker {broker(e)} != acct_target {st['acct_target']}"
    for s in set(tot) | set(st["acct_target"]):
        want = X._acct_whole(tot.get(s, 0.0), PX.get(s), NAV, mode)
        have = st["acct_target"].get(s, 0)
        if have != want and not (mode == "round" and have == want - 1):
            return f"{label}: {s} target {have} vs rule {want} (sum {tot.get(s, 0):.3f})"
    if any(q <= 0 for b in st["books"].values() for q in b.values()):
        return f"{label}: non-positive virtual quantity"
    return None


held_by_mode = {}
for MODE in ("off", "floor", "round"):
    E.ACCOUNT_ROUNDING = MODE
    tmp = Path(tempfile.mkdtemp()) / "tranche_state.json"
    E.TRANCHE_STATE_FILE = tmp
    FakeDT._now = _dt.datetime(2026, 10, 7, 10, 0, tzinfo=_ZI("US/Eastern"))
    # the LIVE situation: a whole-share per-book ledger from the old regime, books == broker
    legacy = {f"N{i:02d}": max(0, int(1.49 * NAV / N / PX[f"N{i:02d}"])) for i in range(N)}
    e = Eng(legacy, NAV, SIG, PX, tmp)
    e._tranche["books"] = X._split_books({s: q for s, q in legacy.items() if q > 0}, 4)
    e._tranche["initialized"] = True
    e._tranche["last_counted_day"] = None
    e._tranche["stride_counter"] = 0                 # not a tranche day: only the stop check runs
    held_before = broker(e)
    asyncio.run(e._check_trailing_stops_tranched())
    if MODE != "off":
        check(f"[{MODE}] switching on trades nothing and adopts the broker as the target",
              not e.sold and not e.bought and e._tranche["acct_target"] == held_before and e._tranche["acct_mode"] == MODE)
    problems = []
    rebuilt = []
    for session in range(1, 23):
        before_counter = e._tranche["stride_counter"]
        asyncio.run(e.rebalance_tranche())
        if e._tranche["stride_counter"] == 0 and before_counter != 0:
            rebuilt.append(e._tranche["last_tranche"])
        if MODE == "off":
            if session == 8 and e.positions.get("N20", {}).get("qty", 0) > 0:
                e.positions["N20"]["qty"] -= 1
                if e.positions["N20"]["qty"] == 0:
                    e.positions.pop("N20")
            asyncio.run(e._check_trailing_stops_tranched())
            advance_session()
            continue
        p = consistent(e, MODE, f"session {session} after rebuild")
        if p:
            problems.append(p)
        if session == 8:                             # external sell (manual / stop) of a whole share of N20
            if e.positions.get("N20", {}).get("qty", 0) > 0:
                e.positions["N20"]["qty"] -= 1
                if e.positions["N20"]["qty"] == 0:
                    e.positions.pop("N20")
        asyncio.run(e._check_trailing_stops_tranched())
        p = consistent(e, MODE, f"session {session} after stop check")
        if p:
            problems.append(p)
        if session == 12:                            # restart: a NEW engine loads the persisted state
            e2 = Eng(broker(e), NAV, SIG, PX, tmp)
            e2._tranche = e2._load_tranche_state()
            e2.trailing_peaks = dict(e.trailing_peaks)
            check(f"[{MODE}] restart reloads fractional books and the account target",
                  e2._tranche["acct_target"] == e._tranche["acct_target"]
                  and all(abs(e2._tranche["books"][t].get(s, 0) - q) < 1e-12 for t, b in e._tranche["books"].items() for s, q in b.items()))
            e = e2
        advance_session()
    held_by_mode[MODE] = broker(e)
    if MODE == "off":
        check("[off] rotation rebuilt books 0,1,2,3 (counter started at 0: rebuilds on sessions 5/10/15/20)",
              rebuilt == [0, 1, 2, 3], str(rebuilt))
        continue
    check(f"[{MODE}] rotation rebuilt books 0,1,2,3 (sessions 5/10/15/20)", rebuilt == [0, 1, 2, 3], str(rebuilt))
    check(f"[{MODE}] invariants every session (broker == target == rule(books), positive virtual qty)",
          not problems, "; ".join(problems[:3]))
    unheld_off = sorted(s for s in PX if held_by_mode["off"].get(s, 0) == 0)
    unheld_now = sorted(s for s in PX if broker(e).get(s, 0) == 0)
    check(f"[{MODE}] names per-book truncation drops are held at the account level "
          f"(unheld: per-book {len(unheld_off)} {unheld_off} -> {len(unheld_now)} {unheld_now})",
          len(unheld_off) > 0 and len(unheld_now) < len(unheld_off))
    check(f"[{MODE}] no orders for zero shares and no stop/overlay buys",
          all(q > 0 for _, q, _ in e.sold + e.bought) and all(r.startswith("tranche") for _, _, r in e.bought))

    # a stop on one book's fractional slice sells only what the account's whole number loses
    st = e._tranche
    name = next(s for s in sorted(PX, key=PX.get, reverse=True) if sum(1 for b in st["books"].values() if b.get(s, 0) > 0) >= 2)
    holders = [t for t, b in st["books"].items() if b.get(name, 0) > 0]
    e.trailing_peaks[f"{holders[0]}:{name}"] = PX[name] * 2         # only this book's peak is breached
    tot_before = X._book_totals(st["books"])[name]
    acct_before = st["acct_target"].get(name, 0)
    n_sold_before = len(e.sold)
    FakeDT._now = FakeDT._now.replace(hour=15, minute=55)
    asyncio.run(e._check_trailing_stops_tranched())
    sold_now = [x for x in e.sold[n_sold_before:] if x[0] == name]
    want = X._acct_whole(tot_before - (tot_before - X._book_totals(st["books"]).get(name, 0.0)), PX[name], NAV, MODE)
    check(f"[{MODE}] stop on book {holders[0]}'s slice of {name}: that slice leaves the book, the account sells "
          f"{acct_before - st['acct_target'].get(name, 0)} share(s)",
          name not in st["books"][holders[0]] and st["acct_target"].get(name, 0) == want
          and sum(q for _, q, _ in sold_now) == acct_before - st["acct_target"].get(name, 0), str(sold_now))
    FakeDT._now = FakeDT._now.replace(hour=10, minute=0)

    # credit gate flat on a tranche day: the rebuilt book AND (overlay on) every other book go to zero
    E.TRANCHE_OVERLAY_DOWN = True
    e._derisk = 0.0
    e._tranche["stride_counter"] = E.TRANCHE_STRIDE
    e._tranche["last_counted_day"] = FakeDT._now.date().isoformat()
    asyncio.run(e.rebalance_tranche())
    check(f"[{MODE}] credit gate at 0: every book flattened and the account sold out (clean-room overlay)",
          all(not b for b in e._tranche["books"].values()) and broker(e) == {} and e._tranche["acct_target"] == {},
          f"left {broker(e)}")
    e._derisk = 1.0
    E.TRANCHE_OVERLAY_DOWN = False

# ── 10. crash mid-rebuild: the retry completes without double trading ─────────────────────────────────
class CrashEng(Eng):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._n = 0

    async def buy_position(self, symbol, qty, reason="signal"):
        self._n += 1
        if self._n == 2:
            raise RuntimeError("watchdog kill")
        return await super().buy_position(symbol, qty, reason)


E.ACCOUNT_ROUNDING = "floor"
tmp3 = Path(tempfile.mkdtemp()) / "tranche_state.json"
E.TRANCHE_STATE_FILE = tmp3
FakeDT._now = _dt.datetime(2026, 10, 7, 10, 0, tzinfo=_ZI("US/Eastern"))
legacy3 = {f"N{i:02d}": max(0, int(1.49 * NAV / N / PX[f"N{i:02d}"])) for i in range(N)}
# three NEW cheap names enter the signal list, so book 0's rebuild has to buy (and the 2nd buy crashes)
SIG3 = SIG[:22] + [{"symbol": f"M{i}", "signal": "BUY", "probability": 0.9} for i in range(3)]
PX.update({f"M{i}": 30.0 + i for i in range(3)})
SIG = SIG3
ce = CrashEng(legacy3, NAV, SIG, PX, tmp3)
ce._tranche["books"] = X._split_books({s: q for s, q in legacy3.items() if q > 0}, 4)
ce._tranche["initialized"] = True
ce._tranche["last_counted_day"] = None
ce._tranche["stride_counter"] = E.TRANCHE_STRIDE - 1
try:
    asyncio.run(ce.rebalance_tranche())
    crashed = False
except RuntimeError:
    crashed = True
check("crash mid-rebuild reproduced (account-level)", crashed)
saved = _json.load(open(tmp3))
check("persisted account target == broker after the crash", {s: int(q) for s, q in saved["acct_target"].items()} == broker(ce),
      f"{saved['acct_target']} vs {broker(ce)}")
re_ = Eng(broker(ce), NAV, SIG, PX, tmp3)
re_._tranche = re_._load_tranche_state()
asyncio.run(re_.rebalance_tranche())
dup = [s for s, _, _ in re_.sold if s in {x[0] for x in ce.bought}] + [s for s, _, _ in re_.bought if s in {x[0] for x in ce.bought}]
check("retry does not re-trade what the first attempt filled", dup == [], str(dup))
check("retry completes the book and the account is consistent", re_._tranche["last_tranche"] == 0
      and consistent(re_, "floor", "retry") is None, str(consistent(re_, "floor", "retry")))

# ── 11. rollback to off: the ledger is truncated to whole shares and reconciled to the broker ──────────
E.ACCOUNT_ROUNDING = "off"
rb_eng = Eng(broker(re_), NAV, SIG, PX, tmp3)
rb_eng._tranche = rb_eng._load_tranche_state()
check("rollback: books are whole shares again and the account target is dropped",
      all(isinstance(q, int) and q > 0 for b in rb_eng._tranche["books"].values() for q in b.values())
      and rb_eng._tranche["acct_target"] is None)
asyncio.run(rb_eng._check_trailing_stops_tranched())
tot_rb = {}
for b in rb_eng._tranche["books"].values():
    for s, q in b.items():
        tot_rb[s] = tot_rb.get(s, 0) + q
check("rollback: after the next reconciliation the per-book ledger equals the broker, nothing traded",
      tot_rb == broker(rb_eng) and not rb_eng.sold and not rb_eng.bought)

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all account-rounding tests passed"))
sys.exit(1 if FAILS else 0)
