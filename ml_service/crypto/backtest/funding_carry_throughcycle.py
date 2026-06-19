"""
Through-cycle funding-carry backtest (Binance funding 2020→2026).

The honest test the Hyperliquid (2023+) data couldn't give: does the delta-neutral
funding carry survive 2021's mania AND 2022's LUNA/FTX blowup? Compares:
  - always-on  : hold top-N by funding regardless of sign (bleeds when funding inverts)
  - timed      : "stand aside when funding turns negative" — only hold positive-carry coins

Return = funding collected on held coins (incl. the days it flips negative before we
rotate) minus turnover fees. Basis P&L is ~0 (perp tracks spot) and is left out here;
the dominant through-cycle risk is the FUNDING regime (negative in crises), which IS
in the data. Liquidation/squeeze tail is a separate leverage cap, noted below.

Run:  python crypto/backtest/funding_carry_throughcycle.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
F = pd.read_parquet(os.path.join(DATA, "binance_funding.parquet"))

FEE = 0.0015        # per position change (perp+spot ~15bps one-way)
TRAIL, REBAL = 7, 7
RISK_FREE = 0.045
LUNA_SQUEEZE = 0.03  # measured max intraday perp-vs-spot gap during LUNA (premiumIndex)


def run(top_k, timed=True, min_ann=0.0, leverage=1.0, margin=0.08):
    sig = F.rolling(TRAIL).mean().shift(1) * 365
    cur, daily, turn = {}, [], []
    for i, dt in enumerate(F.index):
        if i % REBAL == 0:
            s = sig.loc[dt].dropna()
            s = s[s > min_ann] if timed else s              # timed → positive carry only
            picks = s.sort_values(ascending=False).head(top_k)
            new = {c: 1.0 / len(picks) for c in picks.index} if len(picks) else {}
            turn.append(sum(abs(new.get(c, 0) - cur.get(c, 0)) for c in set(new) | set(cur)))
            cur = new
        else:
            turn.append(0.0)
        r = sum(w * F.loc[dt].get(c, 0.0) for c, w in cur.items() if pd.notna(F.loc[dt].get(c, np.nan))) if cur else 0.0
        g = leverage * r - (leverage - 1) * margin / 365 * (1 if cur else 0) - turn[-1] * FEE * leverage
        daily.append(g)
    d = pd.Series(daily, index=F.index)
    nav = (1 + d).cumprod()
    yrs = (F.index[-1] - F.index[0]).days / 365.25
    return dict(cagr=nav.iloc[-1] ** (1 / yrs) - 1, vol=d.std() * np.sqrt(365),
                sharpe=(d.mean() / d.std() * np.sqrt(365)) if d.std() else 0,
                mdd=((nav - nav.cummax()) / nav.cummax()).min(),
                yr=d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100, d=d)


print("=" * 96)
print("THROUGH-CYCLE FUNDING CARRY (Binance, %s→%s) | fees in | cash %.1f%%"
      % (F.index.min().date(), F.index.max().date(), RISK_FREE * 100))
print("=" * 96)
print(f"  {'variant':<30}{'CAGR':>8}{'excess':>9}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")
for lab, kw in [("always-on top-6", dict(top_k=6, timed=False)),
                ("TIMED top-6 (stand aside)", dict(top_k=6, timed=True)),
                ("TIMED top-6, only >10%/yr", dict(top_k=6, timed=True, min_ann=0.10))]:
    r = run(**kw)
    print(f"  {lab:<30}{r['cagr']*100:>7.1f}%{(r['cagr']-RISK_FREE)*100:>8.1f}%{r['vol']*100:>6.1f}%{r['sharpe']:>8.2f}{r['mdd']*100:>7.1f}%", flush=True)

print("\n  -- leverage frontier (TIMED top-6) — and the liquidation reality --")
print(f"  {'leverage':<12}{'CAGR':>8}{'MaxDD':>9}{'  LUNA-squeeze intraday hit':<28}")
for L in [1, 2, 3, 5]:
    r = run(top_k=6, timed=True, leverage=L)
    hit = LUNA_SQUEEZE * L * 100
    flag = "  ⚠️ LIQUIDATION" if hit > 25 else ""
    print(f"  {str(L)+'x':<12}{r['cagr']*100:>7.1f}%{r['mdd']*100:>8.1f}%   -{hit:.0f}% on the short leg{flag}", flush=True)

print("\n  Per-year return (the 2022 stress test):")
base = run(top_k=6, timed=False); tm = run(top_k=6, timed=True)
yrs = sorted(set(base["yr"].index) | set(tm["yr"].index))
print("  %-12s" % "year" + "".join("%7d" % y for y in yrs))
print("  %-12s" % "always-on" + "".join("%6.0f%%" % base["yr"].get(y, 0) for y in yrs))
print("  %-12s" % "timed" + "".join("%6.0f%%" % tm["yr"].get(y, 0) for y in yrs))
print("\n  Verdict if: timed survives 2022 with a tolerable drawdown AND safe leverage (1-2x)")
print("  keeps the LUNA-squeeze intraday hit well clear of liquidation.")
