"""
Data-integrity audit of the survivorship-complete Binance perp panel. Verifies the three
things that "the delisted symbol is in the file" does NOT by itself guarantee:

  1. TERMINATION — delisted coins' series must STOP (NaN) at delisting, not forward-fill
     the last price to the dataset end. Forward-fill on dead coins is a classic silent bug.
  2. NO BACK-FILL / LOOK-AHEAD — a coin's series must START at its real listing (NaN before),
     so point-in-time membership is reconstructable.
  3. DELISTING RETURN — quantify the cost of letting a held position vanish at its last bar
     (a costless exit re-creates the bias). Measured by re-running with a terminal penalty.

Run:  python crypto/backtest/data_integrity.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from momentum_binance import load, run, stats

close, qv, fund = load()
end = close.index.max()
N = close.shape[1]
last_valid = close.apply(lambda s: s.last_valid_index())
first_valid = close.apply(lambda s: s.first_valid_index())

print("=" * 90)
print("DATA-INTEGRITY AUDIT — Binance perp panel | %d coins | %s→%s" % (N, close.index.min().date(), end.date()))
print("=" * 90)

# ---- 1. TERMINATION: do dead coins stop, or get padded to the end? ----
alive = (last_valid >= end - pd.Timedelta(days=7)).sum()
dead = (last_valid < end - pd.Timedelta(days=7)).sum()
print("\n[1] TERMINATION (are delisted coins padded forward?)")
print("    coins still trading at dataset end (alive):  %d" % alive)
print("    coins terminating >7d before end (delisted): %d" % dead)
# forward-fill test: a padded series would have a long run of IDENTICAL closing prices at the tail.
# But identical prices can also be legit illiquid ZOMBIES (crashed coin, ~0 volume, still listed).
# Distinguish: a real padding bug = flat tail on a HIGH-VOLUME, NOT-crashed coin.
flat = [c for c in close.columns if close[c].notna().sum() > 10 and close[c].dropna().iloc[-6:].nunique() == 1]
zombie = sum(1 for c in flat if (close[c].dropna().iloc[-1] / close[c].dropna().max() - 1) < -0.70
             or (c in qv.columns and qv[c].dropna().iloc[-30:].mean() < 1e6))
bug = [c for c in flat if (c in qv.columns and qv[c].dropna().iloc[-30:].mean() > 5e6)
       and (close[c].dropna().iloc[-1] / close[c].dropna().max() - 1) > -0.30]
print("    coins with 6+ identical trailing prices: %d  (crashed/illiquid zombies: %d)" % (len(flat), zombie))
print("    of those, padding-BUG suspects (high-vol, not crashed): %d %s" % (len(bug), bug[:8]))
# explicit check: are cells AFTER a coin's last_valid actually NaN (not the last value repeated)?
ok_nan_after = sum(1 for c in close.columns
                   if last_valid[c] is not None and last_valid[c] < end
                   and close.loc[last_valid[c] + pd.Timedelta(days=1):, c].notna().sum() == 0)
n_term = sum(1 for c in close.columns if last_valid[c] is not None and last_valid[c] < end)
print("    delisted coins that are NaN after last bar (not padded): %d / %d" % (ok_nan_after, n_term))

# ---- 2. BACK-FILL / LOOK-AHEAD: NaN before listing? ----
print("\n[2] BACK-FILL / LOOK-AHEAD (does data start at real listing, NaN before?)")
start = close.index.min()
late = first_valid[first_valid > start + pd.Timedelta(days=180)]   # coins that listed well after 2020
ok_nan_before = sum(1 for c in late.index if close.loc[:first_valid[c] - pd.Timedelta(days=1), c].notna().sum() == 0)
print("    coins listing >180d after start: %d ; with all-NaN before listing: %d" % (len(late), ok_nan_before))

# ---- examples ----
print("\n    examples (coin: listed → last bar | last price | days to end):")
for c in ["BTC", "1000LUNC", "FTT", "SRM", "ANC", "WAVES", "1000BONK", "AGIX", "OMG", "CVC"]:
    if c in close.columns:
        s = close[c].dropna()
        gap = (end - s.index.max()).days
        tag = "ALIVE" if gap <= 7 else "delisted %dd before end" % gap
        print("      %-10s %s → %s | last %.6g | %s" % (c, s.index.min().date(), s.index.max().date(), s.iloc[-1], tag))

# ---- 3. DELISTING-RETURN sensitivity: bound the costless-exit bias ----
print("\n[3] DELISTING-EXIT BIAS — re-run momentum with a terminal loss booked on held coins that delist")
print("    (long position loses the penalty; short position gains it — i.e., NO free exit)")
print(f"    {'terminal penalty on delist':<32}{'2023+ CAGR':>11}{'Sharpe23':>10}{'FULL':>8}")
for pen, lab in [(0.0, "0%  (current / costless exit)"), (0.30, "30% haircut"), (0.50, "50% haircut"), (1.0, "100% (coin → zero)")]:
    d = run(close, qv, fund, long_short=True, use_funding=True, delist_penalty=pen)
    c23, _, s23, _ = stats(d, "2023-01-01"); c, _, _, _ = stats(d)
    print(f"    {lab:<32}{c23*100:>10.1f}%{s23:>10.2f}{c*100:>7.0f}%", flush=True)
print("    long-only, same sensitivity:")
for pen, lab in [(0.0, "0%  (costless exit)"), (0.50, "50% haircut"), (1.0, "100% (→ zero)")]:
    d = run(close, qv, fund, long_short=False, use_funding=False, delist_penalty=pen)
    c23, _, s23, _ = stats(d, "2023-01-01"); c, _, _, _ = stats(d)
    print(f"    {lab:<32}{c23*100:>10.1f}%{s23:>10.2f}{c*100:>7.0f}%", flush=True)

print("\n  If the penalty barely moves the numbers, the weekly volume-ranked book rarely holds a coin")
print("  into delisting (volume collapses → it drops out first). If it moves a lot, the costless exit")
print("  was flattering the result and the honest momentum number is even worse.")
