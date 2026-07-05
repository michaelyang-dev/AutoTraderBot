"""Unit test for IBKREngine._calibrate_quantities (closed-loop sizing).

Validates, at several account sizes, that the calibrated integer share counts
produce realized gross within one-share tolerance of the EFFECTIVE_LEVERAGE
target, that the multiplier respects the LEVERAGE ceiling, that it adapts DOWN
as NAV grows (the whole point of the fix), and that degenerate inputs are safe.
Run with the engine venv: venv/bin/python research/test_sizing_calibration.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_engine import IBKREngine, EFFECTIVE_LEVERAGE, LEVERAGE

# realistic book: prices modeled on the actual live holdings (incl $2k/share names)
PRICES = {"WDC": 598, "MU": 1033, "SNDK": 2034, "LITE": 801, "PAYC": 134, "STX": 916,
          "BOX": 27, "SABR": 2.1, "PAYX": 103, "SLM": 26, "QLYS": 144, "META": 613,
          "SFM": 87, "V": 351, "MORN": 162, "MO": 71, "INCY": 114, "EA": 205,
          "VRSN": 256, "FFIV": 424, "VICI": 26, "MSCI": 582, "VRTX": 498}
SIGNALS = [{"symbol": s, "probability": 1.0 - i * 0.02} for i, s in enumerate(PRICES)]


def main():
    print(f"{'NAV':>10} {'vscale':>7} {'mult':>6} {'gross/NAV':>10} {'target':>7} {'names':>6}")
    ok = True
    for nav in [33_000, 100_000, 300_000, 1_400_000]:
        for vs in [1.0, 0.7]:
            qty, m, gross = IBKREngine._calibrate_quantities(SIGNALS, PRICES, nav, vs)
            eff = gross / nav
            tgt = EFFECTIVE_LEVERAGE * vs
            tol = sum(PRICES.values()) / nav * 1.2 + 0.01   # one-share granularity bound
            good = abs(eff - tgt) <= tol and m <= LEVERAGE + 1e-9
            ok &= good
            print(f"{nav:>10,} {vs:>7.1f} {m:>6.2f} {eff:>10.3f} {tgt:>7.2f} "
                  f"{len(qty):>6} {'OK' if good else 'FAIL'}")

    m_small = IBKREngine._calibrate_quantities(SIGNALS, PRICES, 33_000, 1.0)[1]
    m_big = IBKREngine._calibrate_quantities(SIGNALS, PRICES, 1_400_000, 1.0)[1]
    print(f"\nmult @33K = {m_small:.3f}  vs  @1.4M = {m_big:.3f}  (must be >=)")
    ok &= m_small >= m_big - 1e-9

    assert IBKREngine._calibrate_quantities([], {}, 33_000, 1.0) == ({}, 0.0, 0.0)
    assert IBKREngine._calibrate_quantities(SIGNALS, {}, 33_000, 1.0) == ({}, 0.0, 0.0)
    assert IBKREngine._calibrate_quantities(SIGNALS, PRICES, 0, 1.0) == ({}, 0.0, 0.0)
    print("degenerate cases: OK")

    print("\nALL PASS" if ok else "\nFAILURES — DO NOT RESTART ENGINE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
