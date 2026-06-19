"""
Cross-sectional alt-factor backtest on the GOLD-STANDARD survivorship-free universe
(crypto2 / full CoinMarketCap incl. delisted/dead coins).

Point-in-time: each rebalance, universe = top-N by market cap THAT DAY (dead coins
included while they had a market cap → momentum that bought LUNA/FTT then ate the
crash, which is the honest test). Factors tested:
  mom_30 / mom_90  trailing return (momentum)
  rev_7            short-term reversal (-trailing 7d)
Long-only top-quintile and market-neutral long-short, weekly rebalance, fees in.

Run (after crypto2_universe.R finishes):  python crypto/backtest/alt_factors.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
STABLES = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDD", "FRAX", "USDE", "FDUSD", "PYUSD",
           "USDP", "GUSD", "USTC", "UST", "LUSD", "USDS", "USD1", "WBTC", "WETH", "STETH", "WBETH", "WEETH"}
FEE = 0.002
REBAL = 7
RISK_FREE = 0.045


def load():
    df = pd.read_csv(os.path.join(DATA, "crypto2_history.csv"))
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = df[~df["symbol"].isin(STABLES)]
    close = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    mcap = df.pivot_table(index="date", columns="symbol", values="market_cap", aggfunc="last").sort_index()
    return close, mcap


def signal(close, factor):
    if factor == "mom_30":
        s = close.pct_change(30)
    elif factor == "mom_90":
        s = close.pct_change(90)
    elif factor == "rev_7":
        s = -close.pct_change(7)
    return s.shift(1)                       # lag → no look-ahead


def backtest(close, mcap, ret, factor, long_short=True, top_frac=0.2, top_uni=100):
    sig = signal(close, factor)
    w = {}
    daily = []
    for i, dt in enumerate(close.index):
        turn = 0.0
        if i % REBAL == 0:
            m = mcap.loc[dt].dropna()
            uni = m[m > 0].sort_values(ascending=False).head(top_uni).index
            s = sig.loc[dt, uni].dropna()
            if len(s) >= 15:
                k = max(int(len(s) * top_frac), 2)
                longs = s.sort_values(ascending=False).head(k).index
                nw = {c: 1.0 / len(longs) for c in longs}
                if long_short:
                    shorts = s.sort_values().head(k).index
                    for c in shorts:
                        nw[c] = nw.get(c, 0) - 1.0 / len(shorts)
                turn = sum(abs(nw.get(c, 0) - w.get(c, 0)) for c in set(nw) | set(w))
                w = nw
        r = sum(wt * ret.loc[dt].get(c, 0.0) for c, wt in w.items() if pd.notna(ret.loc[dt].get(c, np.nan)))
        daily.append(r - turn * FEE)
    return pd.Series(daily, index=close.index)


def stats(d, lo=None):
    x = d.loc[lo:] if lo else d
    x = x.dropna()
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            (x.mean() / x.std() * np.sqrt(365)) if x.std() else 0,
            ((nav - nav.cummax()) / nav.cummax()).min())


if __name__ == "__main__":
    close, mcap = load()
    ret = close.pct_change()
    print("=" * 96)
    print("CROSS-SECTIONAL ALT FACTORS — survivorship-free (crypto2) | %s→%s | %d coins"
          % (close.index.min().date(), close.index.max().date(), close.shape[1]))
    print("=" * 96)
    print(f"  {'factor / book':<28}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")
    for factor in ["mom_30", "mom_90", "rev_7"]:
        for ls, tag in [(False, "long-only Q5"), (True, "long-short")]:
            d = backtest(close, mcap, ret, factor, long_short=ls)
            cg, vol, sh, md = stats(d)
            cg23, _, _, _ = stats(d, "2023-01-01")
            print(f"  {factor + ' ' + tag:<28}{cg*100:>10.1f}%{cg23*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)
    print("\n  Honest read: survivorship-free, so momentum that bought coins which then died ate the loss.")
    print("  Edge is real if 2023+ Sharpe holds after fees — the post-mania, out-of-sample regime.")
