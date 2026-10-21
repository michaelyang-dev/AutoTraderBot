"""Price-data quality gate: OHLCV problems are handled PER SYMBOL, never by sending the whole universe to yfinance.

WHY THIS TEST EXISTS (2026-10-21)
  The 2026-10-20 18:22 signal build failed the gate on ONE vendor bar (CALY: open 13.84 < low 13.86, a name the
  book does not hold) and fell back to yfinance for all 1,541 symbols. The fallback (a) is a different vendor,
  (b) could not fetch dotted tickers (BF.B / BRK.B / MOG.A -> "possibly delisted"), and (c) used an EXCLUSIVE
  end=today, so the evening build was a session stale: it served 10-19's regime (breadth 34.9%, blend 0.00 = full
  bear weights, momentum 11% of the book) instead of 10-20's (42.5%, 0.30, momentum 32%) into the 10-21 book-2
  rebuild. DMC did the same on 10-19 18:24 and five times intraday on 10-20.
  Now: range violations are repaired in place (low/high widened; close untouched), symbols with a dead last-5-bar
  window are quarantined (count as missing for coverage), the universe-level fallback remains for coverage /
  staleness / null / recency failures, and the fallback itself uses end=tomorrow and dash-spelled tickers.
Run: python3 tests/test_quality_gate.py
"""
import os
import sys
import types
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MASSIVE_API_KEY", "test-key-not-used")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import massive_data_provider as M  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def gate(bars, syms):
    """Call the gate unbound (it does not use `self`) so no provider is constructed."""
    return M.MassiveDataProvider.run_quality_gate(None, bars, expected_symbols=syms)


def mk_bars(n_syms=40, n=300, seed=0, end=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=n)
    out = {}
    for i in range(n_syms):
        c = np.maximum(50 + np.cumsum(rng.normal(0, 0.5, n)), 5.0)
        o = c + rng.normal(0, 0.2, n)
        h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3, n)) + 0.01
        l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3, n)) - 0.01
        v = rng.integers(10_000, 1_000_000, n).astype(float)
        out[f"S{i:02d}"] = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    return out


SYMS = [f"S{i:02d}" for i in range(40)]

# ── 1. clean bars pass, nothing repaired or quarantined ───────────────────────────────────────────────────
b = mk_bars(); r = gate(b, SYMS)
check("clean bars: gate passes, 0 repaired, 0 quarantined, 40 symbols kept",
      r["passed"] and r["n_repaired"] == 0 and r["n_quarantined"] == 0 and len(b) == 40 and r["issues"] == [], str(r["issues"]))

# ── 2. the CALY case: open 2c below low on the LAST bar -> repaired, gate passes, close untouched ───────
b = mk_bars(); df = b["S00"]; last = df.index[-1]
df.loc[last, "open"] = df.loc[last, "low"] - 0.02; close_before = df.loc[last, "close"]; open_before = df.loc[last, "open"]
r = gate(b, SYMS)
check("CALY case (open < low on the last bar): gate PASSES (no universe fallback)", r["passed"], str(r["issues"]))
check("  ... the bar is repaired: low widened to the open, close untouched, symbol kept",
      "S00" in b and abs(b["S00"].loc[last, "low"] - open_before) < 1e-9 and b["S00"].loc[last, "close"] == close_before
      and b["S00"].loc[last, "high"] >= b["S00"].loc[last, "close"], f"low {b['S00'].loc[last, 'low']:.4f} open {open_before:.4f}")
check("  ... reported: n_repaired 1, repaired [('S00', 1)], n_bad_ohlcv 1", r["n_repaired"] == 1 and r["repaired"] == [("S00", 1)] and r["n_bad_ohlcv"] == 1)

# ── 3. high < low and close > high on older rows -> repaired, passes ────────────────────────────────────
b = mk_bars(); df = b["S05"]; i1, i2 = df.index[-100], df.index[-7]
df.loc[i1, ["high", "low"]] = [df.loc[i1, "low"], df.loc[i1, "high"]]      # swapped
df.loc[i2, "close"] = df.loc[i2, "high"] + 1.0                             # close above high
r = gate(b, SYMS)
check("high<low (swapped) and close>high on older rows: repaired (2 rows), gate passes, high>=low everywhere",
      r["passed"] and r["repaired"] == [("S05", 2)] and bool((b["S05"]["high"] >= b["S05"]["low"]).all())
      and bool((b["S05"]["close"] <= b["S05"]["high"]).all()) and bool((b["S05"]["close"] >= b["S05"]["low"]).all()))

# ── 4. zero volume on the LAST bar -> symbol quarantined, counts as missing, gate still passes (39/40) ───
b = mk_bars(); b["S01"].loc[b["S01"].index[-1], "volume"] = 0.0
r = gate(b, SYMS)
check("zero volume on the last bar: symbol QUARANTINED (removed from bars), gate passes at 97.5% coverage",
      r["passed"] and "S01" not in b and r["quarantined"] == ["S01"] and r["n_missing"] == 1 and len(b) == 39, str(r["issues"]))

# ── 5. non-positive close ANYWHERE -> quarantined; old zero volume -> kept, reported only ───────────────
b = mk_bars()
b["S02"].loc[b["S02"].index[-3], "close"] = 0.0        # recent zero close
b["S04"].loc[b["S04"].index[-200], "close"] = 0.0      # zero close 200 bars ago (would corrupt SMA200 / returns)
b["S03"].loc[b["S03"].index[-200], "volume"] = 0.0     # zero volume 200 bars ago (legitimate illiquid day)
r = gate(b, SYMS)
check("close<=0 recent (S02) and 200 bars ago (S04): both quarantined; zero volume 200 bars ago (S03): kept, reported only",
      r["passed"] and r["quarantined"] == ["S02", "S04"] and "S03" in b and "S02" not in b and "S04" not in b and r["n_bad_ohlcv"] == 3,
      f"quarantined {r['quarantined']} n_bad_ohlcv {r['n_bad_ohlcv']} issues {r['issues']}")

# ── 6. the universe-level fallback STILL exists: enough quarantines -> LOW_COVERAGE fails the gate ──────
b = mk_bars()
for s_ in ("S10", "S11", "S12"):
    b[s_].loc[b[s_].index[-1], "volume"] = 0.0
r = gate(b, SYMS)
check("3 of 40 quarantined (92.5% < 95%): gate FAILS with LOW_COVERAGE — the coverage fallback is intact",
      (not r["passed"]) and r["issues"] and r["issues"][0].startswith("LOW_COVERAGE") and r["n_quarantined"] == 3, str(r["issues"]))

# ── 7. the other checks are unchanged ───────────────────────────────────────────────────────────────────
b = mk_bars(); b["S06"].loc[b["S06"].index[-5:], "close"] = 42.0
r = gate(b, SYMS); check("stale prices (5 identical closes) still fail the gate", (not r["passed"]) and any(i.startswith("STALE_PRICES") for i in r["issues"]))
b = mk_bars(); b["S07"].loc[b["S07"].index[-1], "close"] = np.nan
r = gate(b, SYMS); check("NaN last close still fails the gate (NULL_PRICES)", (not r["passed"]) and any(i.startswith("NULL_PRICES") for i in r["issues"]))
b = mk_bars(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=10))
r = gate(b, SYMS); check("10-day-old data still fails the gate (DATA_STALE)", (not r["passed"]) and any(i.startswith("DATA_STALE") for i in r["issues"]))
b = mk_bars(); [b.pop(s_) for s_ in SYMS[:3]]
r = gate(b, SYMS); check("3 missing symbols (92.5%) still fail the gate (LOW_COVERAGE)", (not r["passed"]) and r["issues"][0].startswith("LOW_COVERAGE"))

# ── 8. end to end: fetch_bars_batch_massive keeps the Massive bars on a repaired bar, falls back on coverage ─
calls = []
def _fake_fallback(symbols, warmup_days=550):
    calls.append(("fallback", len(symbols))); return {"FALLBACK": pd.DataFrame()}
orig_get, orig_fb = M.get_provider, M._fetch_bars_yfinance_fallback
try:
    M._fetch_bars_yfinance_fallback = _fake_fallback
    b = mk_bars(); b["S00"].loc[b["S00"].index[-1], "open"] = b["S00"].loc[b["S00"].index[-1], "low"] - 0.02
    M.get_provider = lambda: types.SimpleNamespace(
        fetch_bars_batch=lambda symbols, warmup_days=550, adjusted=True: b,
        run_quality_gate=lambda bars, expected_symbols=None: M.MassiveDataProvider.run_quality_gate(None, bars, expected_symbols),
        validate_vs_yfinance=False)
    out = M.fetch_bars_batch_massive(SYMS, warmup_days=550)
    check("fetch_bars_batch_massive: one repaired bar -> the MASSIVE bars are returned (40 symbols), no fallback call",
          out is b and len(out) == 40 and calls == [], f"calls {calls}")
    b2 = mk_bars(); [b2.pop(s_) for s_ in SYMS[:3]]
    M.get_provider = lambda: types.SimpleNamespace(
        fetch_bars_batch=lambda symbols, warmup_days=550, adjusted=True: b2,
        run_quality_gate=lambda bars, expected_symbols=None: M.MassiveDataProvider.run_quality_gate(None, bars, expected_symbols),
        validate_vs_yfinance=False)
    out = M.fetch_bars_batch_massive(SYMS, warmup_days=550)
    check("fetch_bars_batch_massive: coverage failure -> the yfinance fallback is still used", "FALLBACK" in out and calls == [("fallback", 40)], f"calls {calls}")
finally:
    M.get_provider, M._fetch_bars_yfinance_fallback = orig_get, orig_fb

# ── 9. the fallback itself: end = tomorrow (inclusive of the latest session) and dash-spelled tickers ──
rec = {}
def _fake_download(tickers, start=None, end=None, auto_adjust=True, progress=False, threads=True):
    rec["tickers"], rec["start"], rec["end"] = list(tickers), start, end
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=30)
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], list(tickers)])
    data = np.tile(np.array([[10.0, 11.0, 9.0, 10.5, 1000.0]]).repeat(len(tickers), axis=0).T.reshape(1, -1), (30, 1))
    return pd.DataFrame(data, index=idx, columns=cols)
fake_yf = types.ModuleType("yfinance"); fake_yf.download = _fake_download
saved = sys.modules.get("yfinance"); sys.modules["yfinance"] = fake_yf
try:
    out = M._fetch_bars_yfinance_fallback(["BRK.B", "AAPL", "BF.B"], warmup_days=550)
finally:
    if saved is not None:
        sys.modules["yfinance"] = saved
    else:
        sys.modules.pop("yfinance", None)
tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
check("fallback requests end=tomorrow (yfinance end is exclusive -> the latest session is INCLUDED)", rec.get("end") == tomorrow, f"end {rec.get('end')} vs {tomorrow}")
check("fallback requests dash-spelled tickers (BRK-B, BF-B) and returns them under the dotted symbols",
      rec.get("tickers") == ["BRK-B", "AAPL", "BF-B"] and len(out.get("BRK.B", [])) == 30 and len(out.get("BF.B", [])) == 30 and len(out.get("AAPL", [])) == 30
      and list(out["BRK.B"].columns) == ["open", "high", "low", "close", "volume"], f"tickers {rec.get('tickers')}")

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all quality-gate tests passed"))
sys.exit(1 if FAILS else 0)
