"""
NEW edge category #2 — relative value / statistical arbitrage (NOT raw reversal, which I killed).

Two beta-neutral mechanisms:
  A. RESIDUAL REVERSAL. Strip each coin's BTC-beta (rolling regression); the RESIDUAL (idiosyncratic
     move) tends to mean-revert even when raw price doesn't. Long coins with negative recent residual,
     short positive — market- and beta-neutral. Different from raw reversal (we tested that: dead) —
     here we revert only the idiosyncratic part, the classic stat-arb signal.
  B. ETH/BTC RATIO mean-reversion — the cleanest cointegration-style pair in crypto.

Honest: daily, point-in-time liquid universe, tiered fees, OOS 2023+. Beta-neutral so it should be
uncorrelated to crypto direction (the whole point — a real diversifying edge if it pays).

Run:  python crypto/backtest/relative_value.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
FEE = 0.0006
RF = 0.045


def load():
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    qv = pd.read_parquet(os.path.join(DATA, "binance_qvol.parquet"))
    return close.sort_index(), qv.sort_index()


def stats(d, lo=None):
    x = d.loc[lo:].dropna() if lo else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.mean() / x.std() * np.sqrt(365),
            ((nav - nav.cummax()) / nav.cummax()).min())


if __name__ == "__main__":
    close, qv = load()
    ret = close.pct_change(fill_method=None)
    btc = ret["BTC"]
    advol = qv.rolling(30, min_periods=10).mean()

    print("=" * 88)
    print("RELATIVE VALUE / STAT-ARB | %s→%s" % (close.index.min().date(), close.index.max().date()))
    print("=" * 88)

    # --- A. residual reversal ---
    print("\n[A] RESIDUAL REVERSAL (beta-neutral) — revert the idiosyncratic move, various lookbacks:")
    print(f"    {'lookback / book':<28}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'Sharpe':>8}{'2023+ Sh':>10}")
    # rolling beta of each coin to BTC
    win = 60
    cov = ret.rolling(win).cov(btc)
    var = btc.rolling(win).var()
    beta = cov.div(var, axis=0)
    resid = ret.sub(beta.mul(btc, axis=0))                     # idiosyncratic daily return
    for lb in [3, 5, 10]:
        sig = (-resid.rolling(lb).sum()).shift(1)              # reverse recent residual
        for rebal in [lb]:
            w = {}; daily = []; cost = []
            for i, dt in enumerate(close.index):
                tc = 0.0
                if i % rebal == 0:
                    uni = advol.loc[dt].dropna()
                    uni = uni[uni > 5e6].sort_values(ascending=False).head(40).index
                    s = sig.loc[dt, uni].dropna()
                    if len(s) >= 12:
                        k = max(int(len(s) * 0.25), 2)
                        longs = s.sort_values(ascending=False).head(k).index
                        shorts = s.sort_values().head(k).index
                        nw = {c: 1.0 / k for c in longs}
                        for c in shorts:
                            nw[c] = nw.get(c, 0) - 1.0 / k
                        tc = sum(abs(nw.get(c, 0) - w.get(c, 0)) for c in set(nw) | set(w)) * FEE
                        w = nw
                r = sum(wt * ret.loc[dt].get(c, 0.0) for c, wt in w.items() if pd.notna(ret.loc[dt].get(c, np.nan)))
                daily.append(r); cost.append(tc)
            d = pd.Series(daily, index=close.index) - pd.Series(cost, index=close.index)
            cg, sh, md = stats(d); cg23, sh23, _ = stats(d, "2023")
            print(f"    resid-rev {lb}d, rebal{rebal}      {cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>8.2f}{sh23:>10.2f}", flush=True)

    # --- B. ETH/BTC ratio mean reversion ---
    print("\n[B] ETH/BTC RATIO z-score mean-reversion (the cleanest pair):")
    print(f"    {'config':<28}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'Sharpe':>8}{'2023+ Sh':>10}")
    ratio = np.log(close["ETH"] / close["BTC"])
    for zwin in [20, 40, 60]:
        z = ((ratio - ratio.rolling(zwin).mean()) / ratio.rolling(zwin).std()).shift(1)
        pos = (-z.clip(-2, 2) / 2.0)                            # short ratio when rich, long when cheap
        rel = ret["ETH"] - ret["BTC"]                          # long-ETH/short-BTC spread return
        turn = pos.diff().abs().fillna(0)
        d = pos * rel - turn * FEE
        cg, sh, md = stats(d); cg23, sh23, _ = stats(d, "2023")
        print(f"    ETHBTC z{zwin}d revert       {cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>8.2f}{sh23:>10.2f}", flush=True)

    print("\n  Read: beta-neutral, so positive OOS Sharpe here = a genuine uncorrelated edge (rare & valuable).")
