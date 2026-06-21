"""
Cross-venue funding DISPERSION inspection (look before building). Binance vs Hyperliquid.

The doc's #1 lead: long the perp where funding is low, short where it's high, same coin, two
venues, capture the spread — a market-neutral carry powered by retail's structural long-bias.
This is DIFFERENT from the single-venue collect-funding I already found decayed.

Before any backtest, answer three questions honestly:
  1. UNITS — are the two venues' funding series on the same daily scale? (else any "spread" is fake)
  2. DISPERSION — is the cross-venue spread big and PERSISTENT (one venue systematically pays more),
     or is it tightly arbitraged to zero?
  3. SIZE vs COST — is the typical |spread| meaningfully above realistic round-trip cost?

Run:  python crypto/backtest/xvenue_inspect.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")

B = pd.read_parquet(os.path.join(DATA, "binance_funding.parquet"))
H = pd.read_parquet(os.path.join(DATA, "hl_funding.parquet"))
B.index = pd.to_datetime(B.index).normalize()
H.index = pd.to_datetime(H.index).normalize()
common = sorted(set(B.columns) & set(H.columns))
idx = B.index.intersection(H.index)
B, H = B.loc[idx, common], H.loc[idx, common]

print("=" * 88)
print("CROSS-VENUE FUNDING — Binance vs Hyperliquid | %d coins | %s→%s" % (len(common), idx.min().date(), idx.max().date()))
print("=" * 88)

print("\n[1] UNITS CHECK — mean DAILY funding per venue (should be same order of magnitude, ann.%%):")
print(f"    {'coin':<8}{'Binance ann':>14}{'HL ann':>12}{'corr':>8}")
for c in common:
    b, h = B[c].dropna(), H[c].dropna()
    j = b.index.intersection(h.index)
    if len(j) < 60:
        continue
    corr = B[c].loc[j].corr(H[c].loc[j])
    print(f"    {c:<8}{B[c].mean()*365*100:>13.1f}%{H[c].mean()*365*100:>11.1f}%{corr:>8.2f}", flush=True)

print("\n[2] DISPERSION — cross-venue spread (Binance − HL) per coin, annualized:")
print(f"    {'coin':<8}{'mean spread':>13}{'std spread':>12}{'|spread|>10bp/d %':>18}{'persist(sign)':>15}")
spread = (B - H)
for c in common:
    s = spread[c].dropna()
    if len(s) < 60:
        continue
    persist = max((s > 0).mean(), (s < 0).mean()) * 100         # how one-sided is the spread
    big = (s.abs() > 0.0010).mean() * 100                        # days |spread|>10bp/day
    print(f"    {c:<8}{s.mean()*365*100:>12.1f}%{s.std()*np.sqrt(365)*100:>11.1f}%{big:>17.0f}%{persist:>14.0f}%", flush=True)

print("\n[3] SIZE vs COST — the harvestable question:")
allsp = spread.stack().dropna()
print("    median |daily spread|: %.1f bp/day  (%.1f%% annualized)" % (allsp.abs().median() * 1e4, allsp.abs().median() * 365 * 100))
print("    %% of coin-days with |spread| > 5bp/day:  %.0f%%" % ((allsp.abs() > 0.0005).mean() * 100))
print("    realistic round-trip cost to set up a x-venue pair (both legs, both ways): ~20-60 bp")
print("    → if you must hold the pair MANY days to earn back 20-60bp at a few bp/day, and the spread")
print("      flips sign before then, it's not capturable. Persistence column in [2] is the key tell.")
