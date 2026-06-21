"""
Time-series TREND-FOLLOWING on the crypto majors — a different strategy class entirely.

Cross-sectional momentum (pick winners vs losers across 700 perps) is dead post-2021.
Trend-following is orthogonal: for each MAJOR coin independently, be long when it's in an
uptrend, FLAT (in cash) when it's not. No survivorship issue (BTC/ETH never died), no meme
blow-off traps. The thesis for "alpha + LESS risk": capture the big crypto up-moves while
stepping aside for the -77% bear crashes, so risk-adjusted return beats buy-and-hold.

Signal: price > SMA(n)  → long, else flat (cash earns risk-free). Equal-weight the majors
currently "on". Tested vs buy-and-hold BTC and an equal-weight majors basket. Fees on flips.

Run:  python crypto/backtest/trend_majors.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
# majors with through-cycle history (exist since ~2020-21); no survivorship concern
MAJORS = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "LTC", "LINK", "BCH",
          "XLM", "TRX", "ETC", "DOT", "AVAX", "ATOM", "UNI", "FIL", "NEAR", "AAVE"]
FEE = 0.0006        # taker round-trip-ish per flip on a major (cheap, liquid)
RF = 0.045


def load():
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    cols = [c for c in MAJORS if c in close.columns]
    return close[cols].sort_index()


def stats(d, lo=None):
    x = d.loc[lo:] if lo else d
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


def trend(close, n=100, allow_short=False):
    """Equal-weight long the majors above their SMA(n); flat (cash@RF) otherwise."""
    ret = close.pct_change(fill_method=None)
    sma = close.rolling(n, min_periods=n // 2).mean()
    pos = (close > sma).astype(float)                          # 1 long, 0 flat
    if allow_short:
        pos = pos * 2 - 1                                      # +1 / -1
    pos = pos.shift(1)                                         # trade on yesterday's signal
    active = pos.abs().sum(axis=1).replace(0, np.nan)
    w = pos.div(active, axis=0).fillna(0.0)                    # equal-weight across active names
    flips = (w - w.shift(1)).abs().sum(axis=1)
    cash_wt = (1 - w.abs().sum(axis=1)).clip(lower=0)          # uninvested → risk-free
    port = (w * ret).sum(axis=1) + cash_wt * RF / 365 - flips * FEE
    return port


def buyhold(close, basket=False):
    ret = close.pct_change(fill_method=None)
    if basket:
        return ret.mean(axis=1)                                # eq-weight majors B&H
    return ret["BTC"]


if __name__ == "__main__":
    close = load()
    print("=" * 92)
    print("TREND-FOLLOWING ON MAJORS — %d coins | %s→%s" % (close.shape[1], close.index.min().date(), close.index.max().date()))
    print("=" * 92)
    print(f"  {'strategy':<38}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")

    def show(label, d):
        cg, vol, sh, md = stats(d); cg23, _, sh23, _ = stats(d, "2023-01-01")
        print(f"  {label:<38}{cg*100:>10.1f}%{cg23*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)

    show("BTC buy & hold", buyhold(close))
    show("majors basket buy & hold", buyhold(close, basket=True))
    print("  -- trend long/flat (cash when below SMA) --")
    for n in [50, 100, 150, 200]:
        show(f"trend SMA{n} long/flat", trend(close, n=n))
    print("  -- trend long/short --")
    for n in [100, 200]:
        show(f"trend SMA{n} long/short", trend(close, n=n, allow_short=True))

    d = trend(close, n=100)
    yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    bh = buyhold(close); byr = bh.groupby(bh.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("\n  per-year — trend SMA100 long/flat vs BTC buy&hold:")
    print("  %-10s" % "year" + "".join("%8d" % y for y in yr.index))
    print("  %-10s" % "trend" + "".join("%7.0f%%" % v for v in yr.values))
    print("  %-10s" % "BTC B&H" + "".join("%7.0f%%" % byr.get(y, 0) for y in yr.index))
    print("\n  Win condition: similar/better CAGR than B&H at MUCH smaller drawdown (the 'less risk' goal).")
