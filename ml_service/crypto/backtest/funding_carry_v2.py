"""
Funding carry v2 — broad universe + dynamic leverage, validated by regime.

Two upgrades over the basic version to fix the "all gains in 2020-21" problem:
  1. BROAD universe (≈58 perps) → harvest the cross-sectional funding DISPERSION,
     which is where the recent edge lives (majors compressed; alts still pay).
  2. DYNAMIC leverage = clamp(harvest_funding / margin_rate, 1, L_MAX): lever up
     only when the funding you're collecting beats your cost of borrow (bull manias),
     sit at 1x when compressed. Avoids the levered margin-drag that wrecked fixed-3x.

Honest constraints: alt fees set higher (0.25%/side), L_MAX capped for the alt
liquidation/squeeze tail. Survivorship caveat remains (delisted rug-coins missing →
flatters alt funding). Reported by regime, with a 2023+ holdout (the real question).

Run:  python crypto/backtest/funding_carry_v2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
F = pd.read_parquet(os.path.join(DATA, "binance_funding.parquet"))

FEE = 0.0025         # alt round-trip is dearer: 25bps/side on position changes
TRAIL, REBAL = 14, 7
MARGIN = 0.08
RISK_FREE = 0.045
L_MAX = 3.0          # capped for the alt liquidation/squeeze tail


def run(top_k=10, min_ann=0.05, lev_mode="fixed", leverage=1.0):
    sig = F.rolling(TRAIL).mean().shift(1) * 365
    cur, daily, levs = {}, [], []
    harvest = 0.0
    for i, dt in enumerate(F.index):
        if i % REBAL == 0:
            s = sig.loc[dt].dropna()
            s = s[s > min_ann]
            picks = s.sort_values(ascending=False).head(top_k)
            new = {c: 1.0 / len(picks) for c in picks.index} if len(picks) else {}
            turn = sum(abs(new.get(c, 0) - cur.get(c, 0)) for c in set(new) | set(cur))
            cur = new
            harvest = float(picks.mean()) if len(picks) else 0.0   # avg ann funding of held basket
        else:
            turn = 0.0
        L = (min(L_MAX, max(1.0, harvest / MARGIN)) if cur else 1.0) if lev_mode == "dynamic" else leverage
        levs.append(L if cur else 0)
        r = sum(w * F.loc[dt].get(c, 0.0) for c, w in cur.items() if pd.notna(F.loc[dt].get(c, np.nan))) if cur else 0.0
        g = L * r - (L - 1) * MARGIN / 365 * (1 if cur else 0) - turn * FEE * L
        daily.append(g)
    d = pd.Series(daily, index=F.index)
    return d, pd.Series(levs, index=F.index)


def stats(d, lo=None):
    x = d.loc[lo:] if lo else d
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            (x.mean() / x.std() * np.sqrt(365)) if x.std() else 0,
            ((nav - nav.cummax()) / nav.cummax()).min())


valid = [c for c in F.columns if F[c].notna().sum() > 200]
print("=" * 100)
print("FUNDING CARRY v2 — broad universe (%d coins) + dynamic leverage | Binance %s→%s"
      % (len(valid), F.index.min().date(), F.index.max().date()))
print("=" * 100)
print(f"  {'variant':<30}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")
configs = [("broad 1x fixed", dict(lev_mode="fixed", leverage=1.0)),
           ("broad 3x fixed", dict(lev_mode="fixed", leverage=3.0)),
           ("broad DYNAMIC (fund/margin)", dict(lev_mode="dynamic"))]
results = {}
for lab, kw in configs:
    d, lv = run(top_k=10, **kw)
    results[lab] = (d, lv)
    cg, vol, sh, md = stats(d)
    cg23, _, _, md23 = stats(d, "2023-01-01")
    print(f"  {lab:<30}{cg*100:>10.1f}%{cg23*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)

d, lv = results["broad DYNAMIC (fund/margin)"]
print("\n  DYNAMIC leverage — per-year return + avg leverage used (proves it de-risks when compressed):")
yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
lyr = lv[lv > 0].groupby(lv[lv > 0].index.year).mean()
ys = sorted(yr.index)
print("  %-10s" % "year" + "".join("%8d" % y for y in ys))
print("  %-10s" % "return" + "".join("%7.0f%%" % yr.get(y, 0) for y in ys))
print("  %-10s" % "avg lev" + "".join("%7.1fx" % lyr.get(y, 1) for y in ys))
print("\n  Goal: 2023+ CAGR materially > 1x-majors' ~4%, dynamic de-levers in 2022-26, and the")
print("  whole thing isn't just survivorship in the alt tail (next: validate vs Hyperliquid live).")
