"""
PILOT B — On-chain flows. Do exchange flows / network activity predict returns (genuinely
different data from price)?

Signals (Coin Metrics free, no look-ahead — all lagged):
  netflow   = (FlowInExUSD − FlowOutExUSD) / market cap. Coins flowing INTO exchanges = supply to
              sell (bearish); OUT = accumulation (bullish). Signal = −netflow.
  adr_mom   = active-address growth (network usage momentum).
  nvt       = market cap / tx count (valuation; high = rich → mean-revert).

First, INFORMATION COEFFICIENT (rank corr of signal vs forward return) — the honest existence test,
full and out-of-sample. Then a simple long/flat timing backtest on BTC & ETH vs buy-and-hold.

Run:  python crypto/backtest/onchain_pilot.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
RF = 0.045
FEE = 0.0006


def load():
    df = pd.read_parquet(os.path.join(DATA, "coinmetrics_onchain.parquet"))
    return df.sort_index()


def zscore(s, win=90):
    return (s - s.rolling(win).mean()) / s.rolling(win).std()


def signals(df, a):
    px = df[f"{a}_PriceUSD"]
    mc = df[f"{a}_CapMrktCurUSD"]
    netflow = (df[f"{a}_FlowInExUSD"] - df[f"{a}_FlowOutExUSD"]) / mc
    sig = {
        "neg_netflow": -zscore(netflow.rolling(7).mean()),       # outflow (accumulation) bullish
        "adr_mom":     zscore(df[f"{a}_AdrActCnt"].rolling(7).mean().pct_change(30)),
        "neg_nvt":     -zscore(mc / df[f"{a}_TxCnt"].rolling(7).mean()),
    }
    return px, pd.DataFrame(sig)


def ic(sig, fwd):
    d = pd.concat([sig.shift(1), fwd], axis=1).dropna()
    if len(d) < 50:
        return np.nan
    return d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman")


if __name__ == "__main__":
    df = load()
    print("=" * 88)
    print("PILOT B — ON-CHAIN FLOWS (Coin Metrics free) | %s→%s" % (df.index.min().date(), df.index.max().date()))
    print("=" * 88)

    print("\n[1] INFORMATION COEFFICIENT (Spearman, signal_{t-1} vs forward return) — full / 2023+ OOS")
    print(f"    {'asset/signal':<26}{'IC 7d':>9}{'IC 7d 23+':>11}{'IC 30d':>9}{'IC 30d 23+':>12}")
    books = {}
    for a in ["BTC", "ETH"]:
        px, sig = signals(df, a)
        books[a] = (px, sig)
        r7 = px.pct_change(7).shift(-7); r30 = px.pct_change(30).shift(-30)
        for c in sig.columns:
            i7 = ic(sig[c], r7); i7b = ic(sig[c].loc["2023":], r7.loc["2023":])
            i30 = ic(sig[c], r30); i30b = ic(sig[c].loc["2023":], r30.loc["2023":])
            print(f"    {a+' '+c:<26}{i7:>9.3f}{i7b:>11.3f}{i30:>9.3f}{i30b:>12.3f}", flush=True)

    print("\n[2] LONG/FLAT TIMING — long when combined on-chain signal > 0, else cash. vs buy&hold")
    print(f"    {'strategy':<26}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'Sharpe':>8}{'2023+ Sh':>10}{'MaxDD':>8}")

    def stats(x, lo=None):
        x = x.loc[lo:].dropna() if lo else x.dropna()
        if len(x) < 30 or x.std() == 0:
            return 0, 0, 0
        nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
        return (nav.iloc[-1] ** (1 / yrs) - 1, x.mean() / x.std() * np.sqrt(365),
                ((nav - nav.cummax()) / nav.cummax()).min())

    for a in ["BTC", "ETH"]:
        px, sig = books[a]
        ret = px.pct_change(fill_method=None)
        combo = sig.mean(axis=1)                                 # equal-weight z-score blend
        pos = (combo.shift(1) > 0).astype(float)
        turn = pos.diff().abs().fillna(0)
        strat = pos * ret + (1 - pos) * RF / 365 - turn * FEE
        cg, sh, md = stats(strat); cg23, sh23, _ = stats(strat, "2023")
        bg, bsh, bmd = stats(ret); bg23, bsh23, _ = stats(ret, "2023")
        print(f"    {a+' on-chain long/flat':<26}{cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>8.2f}{sh23:>10.2f}{md*100:>7.1f}%", flush=True)
        print(f"    {a+' buy & hold':<26}{bg*100:>10.1f}%{bg23*100:>11.1f}%{bsh:>8.2f}{bsh23:>10.2f}{bmd*100:>7.1f}%", flush=True)

    print("\n  Read: |IC| > ~0.03 that HOLDS out-of-sample = a real (if modest) signal worth deepening.")
    print("  Timing book must beat buy&hold on 2023+ Sharpe after fees to be worth anything.")
