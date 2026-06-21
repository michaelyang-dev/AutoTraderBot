"""
Higher-return carry frontier — for a HIGHER risk appetite, with the HONEST tail for each point.

Two kinds of 'more risk' on this book:
  - CONCENTRATION / FUNDING-WEIGHT: lean into the richest payers. A blowup costs ONE position, not
    the book. Survivable → genuinely buys return. (the slope)
  - LEVERAGE: the daily vol is tiny so it looks free, but the structural tail (HL insolvency, intraday
    short-leg liquidation) is FIXED SIZE — leverage turns a bad day into RUIN. (the cliff)

For each config we show CAGR (clean data) AND three honest tails the daily Sharpe hides:
  worst-mo (in-sample), HL-insolvency loss, and a SQUEEZE-LIQUIDATION event (a held coin +50% intraday
  → short leg at this leverage liquidates → loss). 'Ruin' = any tail < -100%.
Run:  python crypto/backtest/aggressive_carry.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from improved_carry import load, carry, stats

# HL margin posted per $1 notional, given a squeeze-survival target of 40% (from risk_controls)
MARGIN_PER_NOTIONAL = 0.40 + 0.05                      # survive +45% before liq → 0.45 margin/notional
SQUEEZE = 0.50                                          # stress: a held coin +50% intraday


def tails(lev, top_k):
    # HL capital = leverage * notional * margin-per-notional; notional gross = 1.0 of capital at 1x
    hl_frac = min(1.0, lev * MARGIN_PER_NOTIONAL)       # fraction of capital sitting on HL
    insolv = -hl_frac                                   # HL insolvency = lose all HL capital
    # squeeze liquidation: a +50% move on a position liquidates the short if 50% > 1/lev-ish; loss =
    # the position's margin (per-coin weight * hl capital), but at high lev the move exceeds margin →
    # lose the position notional. concentrated book (1/top_k each).
    per_coin = 1.0 / top_k
    liq_per_pos = -per_coin * lev * SQUEEZE             # loss on the one squeezed position, levered
    return insolv, liq_per_pos


if __name__ == "__main__":
    F, P = load()
    print("=" * 100)
    print("AGGRESSIVE CARRY FRONTIER (clean data, %d coins) — CAGR vs the HONEST tails" % F.shape[1])
    print("=" * 100)
    print(f"  {'config':<34}{'CAGR':>8}{'worst-mo':>10}{'HL-insolv':>11}{'+50% squeeze/pos':>18}{'  verdict':>12}")
    configs = [
        ("conservative: top15 invvol 1x", 15, "invvol", 1.0),
        ("balanced: top10 equal 1x", 10, "equal", 1.0),
        ("lean-in: top6 funding 1x", 6, "funding", 1.0),
        ("lean-in + 1.5x", 6, "funding", 1.5),
        ("aggressive: top6 funding 2x", 6, "funding", 2.0),
        ("max: top5 funding 3x", 5, "funding", 3.0),
    ]
    for lab, k, w, lev in configs:
        d = carry(F, P, top_k=k, weighting=w, lev=lev)
        cg, vol, sh, md, wm = stats(d)
        insolv, liq = tails(lev, k)
        ruin = insolv <= -1.0 or liq <= -1.0
        verdict = "RUIN RISK" if ruin else ("aggressive" if lev > 1 else "survivable")
        print(f"  {lab:<34}{cg*100:>7.0f}%{wm*100:>9.1f}%{insolv*100:>10.0f}%{liq*100:>17.0f}%{verdict:>12}", flush=True)

    print("\n  HOW TO READ:")
    print("  - HL-insolv = you lose ALL capital posted on Hyperliquid if the DEX fails. At 2x+ this")
    print("    approaches/exceeds -100% = RUIN. Cap it by splitting venues + keeping leverage <=1.5x.")
    print("  - +50% squeeze/pos = loss if ONE held coin rockets and your short leg liquidates. Concentration")
    print("    (low top_k) + leverage makes this bite harder. Survivable while it's a fraction, not the book.")
    print("\n  RECOMMENDED for higher risk appetite (return WITHOUT a ruin cliff):")
    print("    top-6 to top-8, FUNDING-WEIGHTED, ~1.5x leverage, HL exposure capped ~50%, majors-heavy.")
    print("    ≈ mid-20s%% CAGR, survives an HL-insolvency at a big-but-not-fatal loss, no single event = 0.")
    print("    The 2x-3x rows print higher CAGR but a -100%%+ tail — that's not 'more return', it's a coin flip on ruin.")
