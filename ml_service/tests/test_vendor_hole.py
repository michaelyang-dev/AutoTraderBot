"""ROOT-CAUSE regression test for the dist_sma200 coverage collapses (2026-08-18, 2026-09-01).

Reproduces the exact failure from 16 bad-day cache files: a REAL trading date (2026-08-28)
missing from most symbols' history while present in others. Before the fix, every symbol
lacking the date got a NaN row from the union-index reindex, rolling(200) went NaN, and the
name silently vanished from the momentum sleeve -- while pct_change-based features padded
through it and looked healthy. The test asserts the fill restores dist_sma200 WITHOUT
touching the formula, leaves leading NaNs and long gaps alone, and that the provider-level
hole scan flags the date.
"""
import os, sys, io, logging, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd

import signal_builder as sb
import massive_data_provider as M

def mk(dates, seed=0):
    rng = np.random.default_rng(seed)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": 1e6}, index=dates)

full = pd.bdate_range("2025-06-02", "2026-09-01")            # 326 bdays, > 252, spans the hole
hole = pd.Timestamp("2026-08-28")
assert hole in full
d_full = full
d_hole = full.drop(hole)                                      # the vendor hole
d_gap5 = full.drop(full[-40:-35])                             # a 5-day halt: must NOT fill
raw = {"GOOD": mk(d_full, 1), "HOLE1": mk(d_hole, 2), "HOLE2": mk(d_hole, 3),
       "HOLE3": mk(d_hole, 4), "HALT": mk(d_gap5, 5),
       "SPY": mk(d_full, 6)}
prices = pd.DataFrame({s: d["close"] for s, d in raw.items()})   # union index == full
last = full[-1]

feats = sb._compute_features_from_raw(raw, prices)
f_last = feats[feats["date"] == last].set_index("symbol")

ok = True
def chk(c, m):
    global ok; print(("  PASS  " if c else "  FAIL  ") + m); ok &= c

print("=== VENDOR-HOLE FILL ===")
chk(hole in prices.index, "union index contains the hole date (some symbols have it)")
chk(not np.isnan(f_last.loc["GOOD", "dist_sma200"]), "GOOD: dist_sma200 present")
for s in ("HOLE1", "HOLE2", "HOLE3"):
    chk(not np.isnan(f_last.loc[s, "dist_sma200"]), f"{s}: dist_sma200 RESTORED despite missing 2026-08-28")
chk(np.isnan(f_last.loc["HALT", "dist_sma200"]), "HALT (5-day gap): NOT filled -> still excluded (limit=3 respected)")
chk(sb._HOLE_STATS["symbols"] == 4 and sb._HOLE_STATS["dates"].get(hole) == 3,
    f"hole stats: 4 symbols touched, 3 missing 2026-08-28 -> {sb._HOLE_STATS['dates'].get(hole)}")
chk(sb._HOLE_STATS["unfilled"] == 2, f"unfillable cells = {sb._HOLE_STATS['unfilled']} (5-day halt: 3 carried, 2 left NaN -> excluded)")

# formula untouched: with NO holes, output must be identical to plain rolling(200)
c = prices["GOOD"]; ref = ((c - c.rolling(200).mean()) / c.rolling(200).mean()).iloc[-1]
chk(abs(f_last.loc["GOOD", "dist_sma200"] - ref) < 1e-12, "formula unchanged: matches raw rolling(200) exactly for a clean symbol")

print("\n=== PROVIDER HOLE SCAN ===")
class P(M.MassiveDataProvider):
    def __init__(self): self.api_key="x"; self.validate_vs_yfinance=False
    def fetch_ticker_bars(self, *a, **k): return pd.DataFrame()
    def fetch_grouped_daily(self, *a, **k): return {}
import tempfile, shutil, pathlib
tmp = tempfile.mkdtemp(); old = M.CACHE_DIR; M.CACHE_DIR = pathlib.Path(tmp)
# 60 symbols: 45 with the hole, 15 without  -> 75% missing -> must flag
syms = []
for i in range(60):
    s = f"S{i:02d}"; syms.append(s)
    mk(d_hole if i < 45 else d_full, i).to_parquet(M.CACHE_DIR / f"{s}_adj.parquet")
buf = io.StringIO(); h = logging.StreamHandler(buf); M.log.addHandler(h); M.log.setLevel(logging.INFO)
P().fetch_bars_batch(syms, warmup_days=550)
M.log.removeHandler(h); M.CACHE_DIR = old; shutil.rmtree(tmp, ignore_errors=True)
out = buf.getvalue()
chk("VENDOR HOLE(S)" in out and "2026-08-28" in out, "provider scan flags 2026-08-28 as a vendor hole (45/60 missing)")
print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")); sys.exit(0 if ok else 1)
