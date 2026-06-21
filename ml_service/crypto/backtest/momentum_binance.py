"""
SURVIVORSHIP-COMPLETE market-neutral momentum — Binance perp universe (incl. delisted).

The definitive validation. Universe = every USD-M perp Binance ever listed (delisted coins
retained by the CDN → no survivorship bias), ranked point-in-time by trailing dollar-VOLUME
(capacity-aware, the real tradability constraint). All coins are perps → all shortable, so
the long-short book is genuinely executable. Funding tailwind applied where we have it.

This directly answers the question the crypto2 sweep couldn't: is small/mid-cap momentum
real, or a survivorship mirage? Here the dead coins ARE in the data, so the answer is honest.

Run (after binance_klines_fetcher finishes):  python crypto/backtest/momentum_binance.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
STABLES = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDD", "FDUSD", "USDE", "USDP", "GUSD",
           "USTC", "FRAX", "LUSD", "USD1", "AEUR", "EUR", "EURI"}
REBAL = 7
TOP_UNI = 100
TOP_FRAC = 0.2
VOLWIN = 30


def load():
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    qv = pd.read_parquet(os.path.join(DATA, "binance_qvol.parquet"))
    keep = [c for c in close.columns if c not in STABLES]
    close, qv = close[keep], qv[[c for c in keep if c in qv.columns]]
    fund = None
    fp = os.path.join(DATA, "binance_funding.parquet")
    if os.path.exists(fp):
        fund = pd.read_parquet(fp)
        fund.index = pd.to_datetime(fund.index).normalize()
    return close.sort_index(), qv.sort_index(), fund


def slip_bps(rank):
    if rank < 10:  return 0.0010
    if rank < 30:  return 0.0025
    if rank < 60:  return 0.0045
    return 0.0070


def run(close, qv, fund, lookback=30, rebal=REBAL, top_frac=TOP_FRAC, top_uni=TOP_UNI,
        long_short=True, use_funding=True, min_dvol=1e6, skip=0, min_age=0, delist_penalty=0.0):
    ret = close.pct_change(fill_method=None)
    advol = qv.rolling(VOLWIN, min_periods=10).mean()           # trailing $-volume for ranking
    if skip:                                                    # skip-window ("12-1") momentum: return over [t-lb-skip, t-skip]
        sig = (close.shift(skip) / close.shift(skip + lookback) - 1).shift(1)
    else:
        sig = close.pct_change(lookback, fill_method=None).shift(1)
    age = close.notna().cumsum()                                # trading days of history so far
    last_valid = {c: close[c].last_valid_index() for c in close.columns}  # delisting date per coin
    realized = set()                                            # coins whose terminal loss is booked
    fcols = set(fund.columns) if fund is not None else set()

    w, gross, costd, fundd = {}, [], [], []
    for i, dt in enumerate(close.index):
        turn_cost = 0.0
        # book the terminal P&L on any HELD coin that has delisted (no costless exit):
        # wt>0 (long) loses delist_penalty; wt<0 (short) GAINS it (the coin went to ~0).
        death_r = 0.0
        if delist_penalty:
            for c, wt in list(w.items()):
                lv = last_valid[c]
                if c not in realized and lv is not None and dt > lv:
                    death_r += wt * (-delist_penalty)
                    realized.add(c)
                    del w[c]
        if i % rebal == 0:
            v = advol.loc[dt].dropna()
            v = v[v > min_dvol]
            if min_age:                                         # drop fresh listings (pump traps)
                a = age.loc[dt]
                v = v[[c for c in v.index if a.get(c, 0) >= min_age]]
            v = v.sort_values(ascending=False)
            uni = v.head(top_uni).index
            rankmap = {c: r for r, c in enumerate(v.index)}
            s = sig.loc[dt, uni].dropna()
            if len(s) >= 15:
                k = max(int(len(s) * top_frac), 2)
                longs = s.sort_values(ascending=False).head(k).index
                nw = {c: 1.0 / len(longs) for c in longs}
                if long_short:
                    shorts = s.sort_values().head(k).index    # all are perps → all shortable
                    for c in shorts:
                        nw[c] = nw.get(c, 0) - 1.0 / len(shorts)
                for c in set(nw) | set(w):
                    turn_cost += abs(nw.get(c, 0) - w.get(c, 0)) * slip_bps(rankmap.get(c, 999))
                w = nw
        r = death_r + sum(wt * ret.loc[dt].get(c, 0.0) for c, wt in w.items() if pd.notna(ret.loc[dt].get(c, np.nan)))
        fr = 0.0
        if use_funding and fund is not None and dt in fund.index:
            row = fund.loc[dt]
            for c, wt in w.items():
                if wt < 0 and c in fcols and pd.notna(row.get(c, np.nan)):
                    fr += (-wt) * row[c]
        gross.append(r); fundd.append(fr); costd.append(turn_cost)
    return pd.Series(gross, index=close.index) + pd.Series(fundd, index=close.index) - pd.Series(costd, index=close.index)


def stats(d, lo=None):
    x = d.loc[lo:] if lo else d
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


if __name__ == "__main__":
    close, qv, fund = load()
    print("=" * 92)
    print("SURVIVORSHIP-COMPLETE MOMENTUM — Binance perps incl. delisted | %s→%s | %d coins"
          % (close.index.min().date(), close.index.max().date(), close.shape[1]))
    print("=" * 92)
    print(f"  {'configuration':<40}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")
    for lab, kw in [
        ("long-only Q5 (top-vol momentum)", dict(long_short=False, use_funding=False)),
        ("market-neutral long-short",        dict(long_short=True,  use_funding=False)),
        ("market-neutral + funding",         dict(long_short=True,  use_funding=True)),
    ]:
        d = run(close, qv, fund, **kw)
        cg, vol, sh, md = stats(d); cg23, _, sh23, md23 = stats(d, "2023-01-01")
        print(f"  {lab:<40}{cg*100:>10.1f}%{cg23*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)

    print("\n  robustness on the market-neutral + funding book (2023+ out-of-sample):")
    print(f"    {'axis':<20}{'2023+ CAGR':>11}{'Sharpe23':>10}{'MaxDD23':>10}")
    for lb in [15, 20, 30, 45, 60]:
        d = run(close, qv, fund, lookback=lb); c, _, s, m = stats(d, "2023-01-01")
        print(f"    lookback {lb:<11}{c*100:>10.1f}%{s:>10.2f}{m*100:>9.1f}%", flush=True)
    for tu in [50, 100, 150, 200, 300]:
        d = run(close, qv, fund, top_uni=tu); c, _, s, m = stats(d, "2023-01-01")
        print(f"    top_uni {tu:<12}{c*100:>10.1f}%{s:>10.2f}{m*100:>9.1f}%", flush=True)
    d = run(close, qv, fund)
    yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("\n  per-year (market-neutral + funding):")
    print("  " + "".join("%8d" % y for y in yr.index))
    print("  " + "".join("%7.0f%%" % v for v in yr.values))
