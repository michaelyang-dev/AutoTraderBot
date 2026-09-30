"""Live universe rules that must match the backtest (2026-09-30 parity audit).

1. Minimum history: the backtest keeps a symbol once it has a valid price and ret_20d (>= 21 bars), long-lookback
   features NaN. Live required 252 bars and dropped new listings / spin-offs entirely, so they could never reach the
   lowvol sleeve (the validated backtest held one on 7.8% of sampled rebalance dates).
2. Breadth: over index MEMBERS only — the backtest's breadth set holds no ETFs.
Run: python3 tests/test_universe_parity.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _m in ("ib_insync",):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import signal_builder as SB  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


print("minimum history = the backtest's rule")
idx = pd.bdate_range(end="2026-09-29", periods=320)
rng = np.random.default_rng(1)


def bars(n):
    c = 50 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx[-n:])


raw = {"LONG": bars(320), "SHORT": bars(100), "TINY": bars(15)}
prices = pd.DataFrame({s: d["close"] for s, d in raw.items()})
SB._fill_fundamentals = lambda f, d: f            # fundamentals are not under test here (no data files needed)
SB._fill_sector_relative = lambda f, d: f
feat = SB._compute_features_from_raw(raw, prices)
last = feat[feat["date"] == idx[-1]].set_index("symbol")
check("MIN_FEATURE_BARS is 21 (valid ret_20d)", SB.MIN_FEATURE_BARS == 21)
check("a 100-bar listing is IN the feature set (was dropped under the 252-bar rule)", "SHORT" in last.index)
check("…with its short-window features", "SHORT" in last.index and np.isfinite(last.loc["SHORT", "vol_60d"]) and np.isfinite(last.loc["SHORT", "ret_20d"]))
check("…and NaN long-lookback features (keeps it out of momentum/value, as in the backtest)",
      "SHORT" in last.index and np.isnan(last.loc["SHORT", "ret_252d"]) and np.isnan(last.loc["SHORT", "dist_sma200"]))
check("a 15-bar listing is still excluded (no ret_20d yet)", "TINY" not in last.index)
check("a full-history name has every feature", np.isfinite(last.loc["LONG", "ret_252d"]) and np.isfinite(last.loc["LONG", "dist_sma200"]))

print("breadth over members only")


class FakeUni:
    def __init__(self, fmap):
        self.fmap = fmap

    def get_feature_map(self, date, feature, members=None):
        d = self.fmap
        return {s: v for s, v in d.items() if members is None or s in members}


u = FakeUni({"A": 0.1, "B": -0.1, "C": -0.2, "D": -0.3, "SPY": 0.05, "GLD": 0.2, "TLT": 0.1})
b = SB._breadth(u, None, {"A", "B", "C", "D"})
check("ETFs excluded from the breadth denominator and numerator (1 of 4 members above = 25%)", abs(b - 0.25) < 1e-12, b)
check("with the ETFs it would have read 4/7 = 57% — the class of error removed", abs(sum(v > 0 for v in u.fmap.values()) / 7 - 4 / 7) < 1e-12)
check("nothing measurable -> 0.5 (unchanged fallback)", SB._breadth(FakeUni({}), None, {"A"}) == 0.5)

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all universe-parity tests passed"))
sys.exit(1 if FAILS else 0)
