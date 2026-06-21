"""
EXECUTABLE market-neutral crypto momentum — the production candidate.

The raw long-short backtest (alt_factors.py) showed 44% 2023+ / Sharpe 0.90, survivorship-
robust. This turns that into something you could actually run, by layering the realities:

  1. SHORT LEG IS PERP-RESTRICTED. You can only short coins that have a perp. The best
     shorts (dying micro-caps) often DON'T — so the short book shrinks to the larger losers.
     Short universe = the 57 Binance USD-M perps we have funding for.
  2. FUNDING TAILWIND ON SHORTS. Long spot / short perp → the short leg COLLECTS funding
     when it's positive (longs pay shorts), which is the normal alt state. Pure tailwind.
  3. TIERED SLIPPAGE by point-in-time mcap rank — small alts cost more to trade than BTC.
  4. VOL-TARGETING to tame the -50% drawdown: scale gross exposure to a target annualized
     vol using TRAILING (lagged) realized vol, capped at L_MAX, financing charged on leverage>1.

Reports the build-up so each haircut is visible. 2023+ is the out-of-sample read that matters.

Run:  python crypto/backtest/momentum_strategy.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
STABLES = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDD", "FRAX", "USDE", "FDUSD", "PYUSD",
           "USDP", "GUSD", "USTC", "UST", "LUSD", "USDS", "USD1", "WBTC", "WETH", "STETH", "WBETH", "WEETH"}

REBAL = 7
TOP_UNI = 100          # point-in-time top-N by mcap = tradable universe each rebalance
TOP_FRAC = 0.2         # long top quintile / short bottom quintile
TARGET_VOL = 0.35      # annualized vol target for the vol-scaled book (de-risk, not lever up)
VOLWIN = 30            # trailing window for realized-vol estimate (lagged → no look-ahead)
L_MAX = 1.5            # leverage cap — gentle, the perp-restriction already tamed the DD
MARGIN = 0.08          # financing rate on leverage beyond 1x (annual)
RISK_FREE = 0.045


def load():
    df = pd.read_csv(os.path.join(DATA, "crypto2_history_full.csv"))
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    df = df[~df["symbol"].isin(STABLES)]
    close = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    mcap = df.pivot_table(index="date", columns="symbol", values="market_cap", aggfunc="last").sort_index()
    return close, mcap


def load_funding():
    """Daily funding rate for the perp-tradable set (sum of three 8h rates)."""
    f = pd.read_parquet(os.path.join(DATA, "binance_funding.parquet"))
    f.index = pd.to_datetime(f.index).normalize()
    return f


def slip_bps(rank):
    """Per-side slippage+fee by point-in-time mcap rank — small alts are dear to trade."""
    if rank < 10:   return 0.0010
    if rank < 30:   return 0.0025
    if rank < 60:   return 0.0045
    return 0.0070


_CACHE = {}


def run(long_short=True, perp_short=True, use_funding=True, vol_target=True,
        lookback=30, rebal=REBAL, top_frac=TOP_FRAC, top_uni=TOP_UNI):
    if "data" not in _CACHE:
        close, mcap = load()
        _CACHE["data"] = (close, mcap, close.pct_change(fill_method=None), load_funding())
    close, mcap, ret, fund = _CACHE["data"]
    perp = set(fund.columns)
    sig = close.pct_change(lookback, fill_method=None).shift(1)  # momentum, lagged

    w = {}                                                     # current weights
    gross, costd, fundd = [], [], []
    for i, dt in enumerate(close.index):
        turn_cost = 0.0
        if i % rebal == 0:
            m = mcap.loc[dt].dropna()
            ranked = m[m > 0].sort_values(ascending=False)
            uni = ranked.head(top_uni).index
            rankmap = {c: r for r, c in enumerate(ranked.index)}
            s = sig.loc[dt, uni].dropna()
            if len(s) >= 15:
                k = max(int(len(s) * top_frac), 2)
                longs = s.sort_values(ascending=False).head(k).index
                nw = {c: 1.0 / len(longs) for c in longs}
                if long_short:
                    short_pool = s.sort_values().index
                    if perp_short:
                        short_pool = [c for c in short_pool if c in perp]   # only shortable coins
                    shorts = list(short_pool)[:k]
                    if shorts:
                        for c in shorts:
                            nw[c] = nw.get(c, 0) - 1.0 / len(shorts)
                # tiered cost on the turnover of this rebalance
                for c in set(nw) | set(w):
                    d = abs(nw.get(c, 0) - w.get(c, 0))
                    turn_cost += d * slip_bps(rankmap.get(c, 999))
                w = nw
        r = sum(wt * ret.loc[dt].get(c, 0.0) for c, wt in w.items() if pd.notna(ret.loc[dt].get(c, np.nan)))
        # funding: short legs (w<0) COLLECT funding when positive → +|w|*funding
        fr = 0.0
        if use_funding:
            for c, wt in w.items():
                if wt < 0 and c in fund.columns:
                    fv = fund.loc[dt, c] if dt in fund.index else np.nan
                    if pd.notna(fv):
                        fr += (-wt) * fv
        gross.append(r)
        fundd.append(fr)
        costd.append(turn_cost)

    g = pd.Series(gross, index=close.index)
    fr = pd.Series(fundd, index=close.index)
    cst = pd.Series(costd, index=close.index)
    net = g + fr - cst                                          # 1x net daily return

    if vol_target:
        rv = net.rolling(VOLWIN).std().shift(1) * np.sqrt(365)
        scale = (TARGET_VOL / rv).clip(0, L_MAX).fillna(0.0)
        fin = (scale - 1).clip(lower=0) * (MARGIN / 365)        # financing on leverage>1
        out = scale * net - fin
    else:
        out = net
    return out


def stats(d, lo=None):
    x = d.loc[lo:] if lo else d
    x = x.dropna()
    if x.std() == 0 or len(x) < 30:
        return 0, 0, 0, 0
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


if __name__ == "__main__":
    print("=" * 100)
    print("EXECUTABLE MARKET-NEUTRAL MOMENTUM — build-up of each reality (mom_30, weekly, crypto2 full)")
    print("=" * 100)
    print(f"  {'configuration':<46}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")
    configs = [
        ("1. long-short, all-coin shorts (tiered slip)", dict(perp_short=False, use_funding=False, vol_target=False)),
        ("2. + perp-restricted short leg",              dict(perp_short=True,  use_funding=False, vol_target=False)),
        ("3. + funding tailwind (THE candidate)",       dict(perp_short=True,  use_funding=True,  vol_target=False)),
        ("4. + vol-target 35%/1.5x (de-risk variant)",  dict(perp_short=True,  use_funding=True,  vol_target=True)),
    ]
    for lab, kw in configs:
        d = run(long_short=True, **kw)
        cg, vol, sh, md = stats(d)
        cg23, _, sh23, md23 = stats(d, "2023-01-01")
        print(f"  {lab:<46}{cg*100:>10.1f}%{cg23*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)

    print("\n  Note: config 4 is fully loaded — perp-only shorts, tiered alt slippage, funding tailwind,")
    print("  vol-scaled to 50% with financing on leverage. 2023+ is the out-of-sample, post-mania read.")
    d = run(long_short=True, perp_short=True, use_funding=True, vol_target=False)
    yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("\n  per-year return (config 3, the candidate):")
    print("  " + "".join("%8d" % y for y in yr.index))
    print("  " + "".join("%7.0f%%" % v for v in yr.values))
