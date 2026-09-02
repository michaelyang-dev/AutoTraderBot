"""The engine must refuse to rebalance when feature coverage is degraded.

WHY THIS TEST EXISTS
  The coverage guard fired twice (dist_sma200 30.6% on 2026-08-18, 19.7% on 2026-09-01). On BOTH
  occasions the signals were perfectly FRESH, so the engine's pre-existing staleness check passed
  and it would have traded a book built from ~20% of the universe. Freshness and correctness are
  different properties. Neither incident landed on a rebalance day -- that was luck, not design.
"""
import os
import sys
import types  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Stub the IBKR client so the gate logic is testable without the broker dependency.
# We are exercising fetch_signals' decision, which touches no IB API.
if "ib_insync" not in sys.modules:
    _stub = types.ModuleType("ib_insync")
    # exactly the names ibkr_engine star-imports; `from x import *` needs a real __all__
    _stub.__all__ = ["IB", "MarketOrder", "Position", "Stock", "util"]
    for _n in _stub.__all__:
        setattr(_stub, _n, type(_n, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules["ib_insync"] = _stub


class _Resp:
    def __init__(self, payload, code=200):
        self._p, self.status_code = payload, code

    def json(self):
        return self._p


def run_case(name, health, expect_trade):
    import ibkr_engine as E
    sent = []
    E.send_telegram = lambda m: sent.append(m)
    sigs = [{"symbol": "AAPL", "signal": "BUY", "probability": 0.9},
            {"symbol": "MSFT", "signal": "BUY", "probability": 0.8}]

    def fake_get(url, timeout=None):
        return _Resp(health) if "health" in url else _Resp({"signals": sigs})
    E.requests = types.SimpleNamespace(get=fake_get)

    eng = E.IBKREngine.__new__(E.IBKREngine)
    eng.last_signals = []
    out = E.IBKREngine.fetch_signals(eng)
    traded = len(out) > 0
    ok = traded == expect_trade
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} traded={traded} "
          f"(expected {expect_trade}){'  alert sent' if sent else ''}")
    return ok


print("=== ENGINE FEATURE-COVERAGE GATE ===")
allok = True
allok &= run_case("healthy coverage -> trades",
                  {"is_stale": False, "coverage_ok": True,
                   "coverage": {"dist_sma200": 0.981}}, True)
allok &= run_case("DEGRADED coverage (19.7%) -> REFUSES",
                  {"is_stale": False, "coverage_ok": False,
                   "coverage": {"dist_sma200": 0.197}}, False)
allok &= run_case("degraded AND fresh (the real incident) -> REFUSES",
                  {"is_stale": False, "coverage_ok": False,
                   "coverage": {"dist_sma200": 0.306, "roe": 0.945}}, False)
allok &= run_case("stale -> REFUSES (pre-existing behaviour intact)",
                  {"is_stale": True, "coverage_ok": True}, False)
allok &= run_case("older server, no coverage field -> trades (back-compat)",
                  {"is_stale": False}, True)
print("\n" + ("ALL TESTS PASSED" if allok else "SOME TESTS FAILED"))
sys.exit(0 if allok else 1)
