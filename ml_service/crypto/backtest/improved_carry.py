"""
Improving the HL carry — the three levers, tested honestly: UNIVERSE, CONSTRUCTION, LEVERAGE.

Carry P&L per coin = funding (collected short) − Δpremium (convergence). Only positive-funding coins
(long spot / short HL perp). We test:
  1. UNIVERSE breadth — more coins = more dispersion to harvest + diversification (uncorrelated funding
     cycles → smoother aggregate), but lower-quality alts add basis-blowout tail. Sweep top_k.
  2. CONSTRUCTION — equal-weight vs funding-weighted (chase carry) vs inverse-premium-vol (risk parity).
  3. LEVERAGE — the CAGR dial on a market-neutral book. BUT the daily Sharpe is a carry illusion, so
     naive Kelly says "lever 50x" which is INSANE. We size by the TAIL, not daily vol, and show both.

US-deployable subset (Coinbase-spot majors) reported alongside the full-HL edge.
Run:  python crypto/backtest/improved_carry.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
US_SPOT = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "NEAR", "SUI", "AAVE", "XRP", "ADA", "LINK", "WLD",
           "LTC", "BCH", "UNI", "APT", "ARB", "OP", "INJ", "TIA", "SEI", "ATOM", "FIL", "ENA", "ONDO"]


def load():
    F = pd.read_parquet(os.path.join(DATA, "hl_funding.parquet"))
    P = pd.read_parquet(os.path.join(DATA, "hl_premium.parquet"))
    for d in (F, P):
        d.index = pd.to_datetime(d.index).normalize()
    return F, P.reindex(columns=F.columns)


def carry(F, P, top_k=10, rebal=7, weighting="equal", lev=1.0, rt_cost=0.0020, min_fund=0.0, us_only=False):
    cols = [c for c in F.columns if (not us_only or c in US_SPOT)]
    F, P = F[cols], P.reindex(columns=cols)
    dprem = P.diff()
    sig = F.rolling(14).mean().shift(1) * 365                  # trailing ann funding
    pvol = dprem.rolling(30).std().shift(1)                    # premium vol for risk-parity
    w = {}; daily = []; cost = []
    for i, dt in enumerate(F.index):
        tc = 0.0
        if i % rebal == 0:
            s = sig.loc[dt].dropna(); s = s[s > min_fund]
            picks = s.sort_values(ascending=False).head(top_k)
            if len(picks):
                if weighting == "funding":
                    raw = picks
                elif weighting == "invvol":
                    raw = 1.0 / pvol.loc[dt, picks.index].replace(0, np.nan)
                else:
                    raw = pd.Series(1.0, index=picks.index)
                raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                nw = (raw / raw.sum()).to_dict() if raw.sum() > 0 else {}
            else:
                nw = {}
            tc = sum(abs(nw.get(c, 0) - w.get(c, 0)) for c in set(nw) | set(w)) * rt_cost
            w = nw
        r = 0.0
        for c, wt in w.items():
            f = F.loc[dt, c] if pd.notna(F.loc[dt, c]) else 0.0
            dp = dprem.loc[dt, c] if pd.notna(dprem.loc[dt, c]) else 0.0
            r += wt * lev * (f - dp)
        daily.append(r); cost.append(tc * lev)
    return pd.Series(daily, index=F.index) - pd.Series(cost, index=F.index)


def stats(d, lo=None):
    x = d.loc[lo:].dropna() if lo else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    mo = x.resample("M").sum()
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min(), mo.min())


if __name__ == "__main__":
    F, P = load()
    print("=" * 96)
    print("IMPROVING HL CARRY — %d coins available | %s→%s" % (F.shape[1], F.index.min().date(), F.index.max().date()))
    print("=" * 96)

    print("\n[1] UNIVERSE BREADTH (equal-weight, 1x, 20bp r/t) — full HL universe vs US-deployable:")
    print(f"    {'top_k':<10}{'FULL CAGR':>11}{'2024+ CAGR':>12}{'vol':>7}{'worst-mo':>10}{'  | US-only CAGR':>18}")
    for k in [5, 10, 20, 40]:
        d = carry(F, P, top_k=k)
        du = carry(F, P, top_k=k, us_only=True)
        cg, vol, sh, md, wm = stats(d); cg24, _, _, _, _ = stats(d, "2024")
        cgu, _, _, _, _ = stats(du)
        print(f"    {k:<10}{cg*100:>10.1f}%{cg24*100:>11.1f}%{vol*100:>6.1f}%{wm*100:>9.1f}%{cgu*100:>16.1f}%", flush=True)

    print("\n[2] CONSTRUCTION (top_k=15, 1x) — weighting scheme:")
    print(f"    {'weighting':<14}{'CAGR':>9}{'vol':>7}{'Sharpe*':>9}{'worst-mo':>10}")
    for wt in ["equal", "funding", "invvol"]:
        d = carry(F, P, top_k=15, weighting=wt)
        cg, vol, sh, md, wm = stats(d)
        print(f"    {wt:<14}{cg*100:>8.1f}%{vol*100:>6.1f}%{sh:>9.1f}{wm*100:>9.1f}%", flush=True)

    print("\n[3] LEVERAGE (top_k=15, invvol) — the CAGR dial vs the TAIL. *Sharpe/Kelly from daily vol are")
    print("    ILLUSIONS (true tail not in daily vol) — shown to prove why you must NOT trust them:")
    print(f"    {'leverage':<10}{'CAGR':>9}{'worst-mo':>10}{'MaxDD(insample)':>17}{'HL-insolv tail':>16}")
    base = carry(F, P, top_k=15, weighting="invvol", lev=1.0)
    cg1, vol1, sh1, md1, wm1 = stats(base)
    kelly = base.mean() / base.var() if base.var() > 0 else 0
    for L in [1.0, 2.0, 3.0]:
        d = carry(F, P, top_k=15, weighting="invvol", lev=L)
        cg, vol, sh, md, wm = stats(d)
        insolv = -L * 0.6                                       # ~60% of notional on HL margin at this lev → insolvency loss
        print(f"    {L:.0f}x        {cg*100:>8.1f}%{wm*100:>9.1f}%{md*100:>16.1f}%{insolv*100:>15.0f}%", flush=True)
    print(f"\n    naive daily-vol Kelly = {kelly:.0f}x  ← ABSURD; the carry illusion. Real sizing: 1-2x max,")
    print("    bounded by the HL-insolvency/liquidation tail, NOT the fake low daily vol.")
    print("\n  Verdict printed above: see which lever actually raises CAGR without blowing the tail.")
