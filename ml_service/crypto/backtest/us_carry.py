"""
US-DEPLOYABLE carry — long spot (Coinbase/Kraken) + short perp on Hyperliquid, harvest HL's rich
funding. This is the version a US person can ACTUALLY trade (Binance/Bybit/OKX perps are off-limits).

Not a clean perp-perp hedge: you now eat the HL perp-PREMIUM convergence.
  net_daily = HL_funding (collected, short)  −  Δ(HL premium-to-spot)  −  costs
The funding is the reward; the premium blowout (perp spikes above spot while you're short) is the
risk. Honest reporting: NET CAGR, funding-flip %, and the TAIL (worst premium-spike days) — NOT the
illusory low-vol Sharpe (delta-neutral carry always looks like Sharpe 10; the risk is the tail).

Run:  python crypto/backtest/us_carry.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
# major alts a US person can hold as SPOT on Coinbase/Kraken AND short as perp on HL
US_SPOT = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "NEAR", "SUI", "AAVE", "XRP", "ADA", "LINK", "WLD"]


def load():
    F = pd.read_parquet(os.path.join(DATA, "hl_funding.parquet"))
    P = pd.read_parquet(os.path.join(DATA, "hl_premium.parquet"))
    for d in (F, P):
        d.index = pd.to_datetime(d.index).normalize()
    cols = [c for c in US_SPOT if c in F.columns and c in P.columns]
    idx = F.index.intersection(P.index)
    return F.loc[idx, cols], P.loc[idx, cols], cols


def stats(d, lo=None):
    x = d.loc[lo:].dropna() if lo else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min(), x.min())


def backtest(F, P, top_k=6, rebal=7, rt_cost=0.0020, lev=1.0):
    dprem = P.diff()                                           # premium convergence (loss if premium rises)
    sig = F.rolling(14).mean().shift(1)                        # trailing funding (collect where richest)
    w = {}; daily = []; cost = []
    for i, dt in enumerate(F.index):
        tc = 0.0
        if i % rebal == 0:
            s = sig.loc[dt].dropna(); s = s[s > 0]
            picks = s.sort_values(ascending=False).head(top_k)
            nw = {c: 1.0 / len(picks) for c in picks.index} if len(picks) else {}
            tc = sum(abs(nw.get(c, 0) - w.get(c, 0)) for c in set(nw) | set(w)) * rt_cost
            w = nw
        r = 0.0
        for c, wt in w.items():
            f = F.loc[dt, c] if pd.notna(F.loc[dt, c]) else 0.0
            dp = dprem.loc[dt, c] if pd.notna(dprem.loc[dt, c]) else 0.0
            r += wt * lev * (f - dp)
        daily.append(r); cost.append(tc * lev)
    return pd.Series(daily, index=F.index) - pd.Series(cost, index=F.index)


if __name__ == "__main__":
    F, P, cols = load()
    print("=" * 90)
    print("US-DEPLOYABLE CARRY — long spot / short HL perp | %d coins | %s→%s" % (len(cols), F.index.min().date(), F.index.max().date()))
    print("=" * 90)
    print("    coins:", cols)
    print("    avg HL funding collected: %.1f%%/yr | funding > 0 %% of days: %.0f%%"
          % (F.stack().mean() * 365 * 100, (F.stack() > 0).mean() * 100))

    print("\n[1] NET carry (funding − premium convergence − costs), 1x:")
    print(f"    {'config':<30}{'FULL CAGR':>11}{'2024+ CAGR':>12}{'vol':>7}{'Sharpe*':>9}{'MaxDD':>8}{'worstDay':>9}")
    for rt in [0.0010, 0.0020, 0.0040]:
        d = backtest(F, P, rt_cost=rt)
        cg, vol, sh, md, wd = stats(d); cg24, _, _, _, _ = stats(d, "2024")
        print(f"    top-6, {rt*1e4:.0f}bp round-trip        {cg*100:>10.1f}%{cg24*100:>11.1f}%{vol*100:>6.1f}%{sh:>9.1f}{md*100:>7.1f}%{wd*100:>8.1f}%", flush=True)
    print("    (*Sharpe is the carry ILLUSION — low daily vol. The real risk is the tail below.)")

    print("\n[2] THE TAIL — worst days (premium spikes: HL perp jumps above spot while you're short):")
    dprem = P.diff()
    worst = (-dprem).stack().nsmallest(8)                      # most negative net (premium rose most)
    for (dt, c), v in worst.items():
        print("      %s %-6s premium move %+.1f%% (one-day hit to the short)" % (dt.date(), c, -v * 100), flush=True)

    print("\n[3] per-year net (top-6, 20bp round-trip, 1x):")
    d = backtest(F, P, rt_cost=0.0020)
    yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("    " + "  ".join("%d:%+.0f%%" % (y, v) for y, v in yr.items()))
    print("\n  Honest read: net carry ~funding minus convergence. Deployable by a US person. The Sharpe is")
    print("  fake; size it by the TAIL (premium blowout = your short-leg loss) and keep leverage low.")
