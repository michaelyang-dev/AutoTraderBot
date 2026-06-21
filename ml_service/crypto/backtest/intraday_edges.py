"""
Intraday/microstructure edges — the last untested frequency. These can exist intraday even
when nothing works daily, because they're structural (positioning, settlement mechanics):

  A. INTRADAY MOMENTUM. The first part of the UTC day predicting the rest (a documented equity
     effect; tests whether crypto trends within the day).
  B. FUNDING-SETTLEMENT DRIFT. Perp funding stamps at 00/08/16 UTC. Does price drift predictably
     into/out of settlement (people position to dodge/collect funding)? Pure microstructure.
  C. SESSION / TIME-OF-DAY. Are some UTC hours systematically up/down (Asia vs US session)?
  D. HOURLY CROSS-SECTIONAL MOMENTUM / REVERSAL on majors (short-horizon continuation vs revert).

Honest costs: per-side taker fee on every entry/exit. At this frequency, fees usually decide it.
Hourly close only (no order book) → these are SIGNAL-existence tests, not execution-proof.

Run (after binance_hourly_fetcher):  python crypto/backtest/intraday_edges.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
TAKER = 0.0004
RF = 0.045


def load():
    close = pd.read_parquet(os.path.join(DATA, "binance_close_1h.parquet"))
    qv = pd.read_parquet(os.path.join(DATA, "binance_qvol_1h.parquet"))
    return close.sort_index(), qv.sort_index()


def sharpe(x, periods=24 * 365):
    x = x.dropna()
    if len(x) < 100 or x.std() == 0:
        return 0.0, 0.0
    cg = (1 + x).prod() ** (periods / len(x)) - 1
    return cg, x.mean() / x.std() * np.sqrt(periods)


def oos(x, split="2023-01-01"):
    return sharpe(x.loc[split:])


if __name__ == "__main__":
    close, qv = load()
    print("=" * 88)
    print("INTRADAY EDGES — hourly majors | %d coins | %s→%s UTC" % (close.shape[1], close.index.min(), close.index.max()))
    print("=" * 88)
    btc = close["BTC"]
    hr = close.pct_change()
    hour = close.index.hour

    # ---- C. SESSION / TIME-OF-DAY (BTC mean return by UTC hour) ----
    print("\n[C] BTC mean hourly return by UTC hour (bps) — is any session systematically directional?")
    by_hour = (hr["BTC"] * 1e4).groupby(hour).mean()
    line = "  " + " ".join("%02d:%+5.1f" % (h, by_hour.get(h, 0)) for h in range(0, 12))
    line2 = "  " + " ".join("%02d:%+5.1f" % (h, by_hour.get(h, 0)) for h in range(12, 24))
    print(line); print(line2)
    print("  (00/08/16 UTC are funding stamps — watch those hours)")

    # ---- B. FUNDING-SETTLEMENT DRIFT (return in the hour BEFORE vs AFTER each 8h stamp) ----
    print("\n[B] Funding-settlement drift (BTC, bps): hour before stamp vs hour after")
    pre = hr["BTC"][np.isin(hour, [7, 15, 23])].mean() * 1e4
    post = hr["BTC"][np.isin(hour, [0, 8, 16])].mean() * 1e4
    other = hr["BTC"][~np.isin(hour, [0, 7, 8, 15, 16, 23])].mean() * 1e4
    print("  pre-stamp hours (07/15/23): %+.2f | post-stamp (00/08/16): %+.2f | other: %+.2f" % (pre, post, other))

    # ---- A. INTRADAY MOMENTUM (first-6h return predicts next-6h, BTC) ----
    print("\n[A] Intraday momentum — does the day's first 6h predict the next 6h? (BTC, daily obs)")
    c = close["BTC"]
    cd = pd.DataFrame({"c": c.values}, index=c.index)
    cd["date"] = cd.index.date; cd["hour"] = cd.index.hour
    op = cd[cd.hour == 0].set_index("date")["c"]
    h6 = cd[cd.hour == 6].set_index("date")["c"]
    cl = cd[cd.hour == 23].set_index("date")["c"]
    first6 = (h6 / op - 1); rest = (cl / h6 - 1)
    both = pd.DataFrame({"first6": first6, "rest": rest}).dropna()
    both.index = pd.to_datetime(both.index)
    strat = np.sign(both["first6"]) * both["rest"] - 2 * TAKER         # trade direction of first 6h into the rest
    cg, sh = sharpe(strat, periods=365); cg23, sh23 = oos(strat)
    corr = both["first6"].corr(both["rest"])
    print("  corr(first6, rest) = %+.3f | strat CAGR %.1f%% Sharpe %.2f | 2023+ Sharpe %.2f" % (corr, cg * 100, sh, sh23))

    # ---- D. HOURLY CROSS-SECTIONAL momentum vs reversal on majors ----
    print("\n[D] Hourly cross-sectional (long-short top/bottom 25%%) — continuation vs reversal:")
    print(f"    {'signal':<28}{'CAGR':>10}{'Sharpe':>9}{'2023+ Sh':>10}")
    liquid = qv.rolling(24).mean() > 2e6
    for lb in [1, 3, 6, 12, 24]:
        sig = close.pct_change(lb).shift(1)
        for direction, nm in [(1, "mom"), (-1, "rev")]:
            s = (sig * direction).where(liquid)
            rk = s.rank(axis=1, pct=True)
            k = 0.25
            w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
            w[rk >= 1 - k] = 1.0; w[rk <= k] = -1.0
            w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            # rebalance every lb hours to match signal horizon
            mask = (np.arange(len(w)) % lb == 0)
            hold = w.copy(); hold.iloc[~mask] = np.nan; hold = hold.ffill().fillna(0.0)
            turn = (hold - hold.shift(1)).abs().sum(axis=1).fillna(0)
            pnl = (hold * hr).sum(axis=1) - turn * TAKER
            cg, sh = sharpe(pnl); _, sh23 = oos(pnl)
            print(f"    {nm+' '+str(lb)+'h, rebal'+str(lb)+'h':<28}{cg*100:>9.1f}%{sh:>9.2f}{sh23:>10.2f}", flush=True)

    print("\n  Read: a positive 2023+ Sharpe that SURVIVES the taker fee is a candidate. Anything needing")
    print("  sub-hour holding or tiny per-trade edges is not honestly tradable without order-book data.")
