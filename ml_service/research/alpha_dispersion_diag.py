"""
Idea 2 — Cross-sectional dispersion regime: DIAGNOSTIC (does momentum's edge
depend on dispersion?)
=============================================================================
Thesis: momentum only pays when there is cross-sectional SPREAD between winners
and the pack. When dispersion collapses (macro-driven tape, everything moves
together), the top-5 has no raw material and momentum bleeds / whipsaws. This is
distinct from the breadth filter (SPY<SMA200 / % above 50d) the system already
runs — dispersion is about spread, not direction.

Measure, at each 20d rebalance over the sample:
  - DISPERSION proxies (cross-sectional, universe-wide):
      xstd60  = cross-sectional std of 60d returns
      xstd252 = cross-sectional std of 252d returns (momentum dispersion)
  - FORWARD 20d: top-5 momentum book mean return, universe mean return, and
    EXCESS = top5 - universe (the momentum edge that period).
Then bucket forward EXCESS by dispersion tercile. If excess is large in
high-dispersion regimes and ~0/negative in low-dispersion regimes -> scaling the
momentum sleeve weight by dispersion is a real timing lever.

Also reports forward excess by BREADTH tercile to check dispersion adds info
beyond what the system already uses.

Run: cd ml_service && ./venv/bin/python research/alpha_dispersion_diag.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_production_backtest import FastBacktester
from strategies.multi_strategy_engine import strategy1_momentum_reversal


def _fwd(prices, sym, d0, d1):
    s = prices.get(sym)
    if s is None:
        return None
    p0 = s.loc[:d0].dropna()
    p1 = s.loc[:d1].dropna()
    if len(p0) == 0 or len(p1) == 0:
        return None
    a, b = p0.iloc[-1], p1.iloc[-1]
    if a <= 0 or np.isnan(a) or np.isnan(b):
        return None
    return b / a - 1.0


def run_dispersion_diagnostic(bt, start="2001-01-01", end="2025-12-31",
                              top_n=5, rebal_days=20):
    prices = {c: bt.prices[c].dropna() for c in bt.prices.columns}
    bt.uni.get_sp500 = bt._get_sp1500
    dates = [d for d in sorted(bt.prices.index)
             if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    rebal_idx = list(range(0, len(dates), rebal_days))

    rows = []
    for k, di in enumerate(rebal_idx[:-1]):
        date = dates[di]
        d_next = dates[rebal_idx[k + 1]]
        members = bt._get_sp1500(date)

        r60 = bt.uni.get_feature_map(date, "ret_60d", members)
        r252 = bt.uni.get_feature_map(date, "ret_252d", members)
        d50 = bt.uni.get_feature_map(date, "dist_sma50", members)
        if len(r60) < 50 or len(r252) < 50:
            continue

        xstd60 = float(np.std(list(r60.values())))
        xstd252 = float(np.std(list(r252.values())))
        breadth = np.mean([1.0 if v > 0 else 0.0 for v in d50.values()]) if d50 else 0.5

        w = strategy1_momentum_reversal(date, bt.uni, di, top_n=top_n,
                                        rebal_days=rebal_days)
        if not w:
            continue
        top5_fwd = [f for f in (_fwd(prices, s, date, d_next) for s in w) if f is not None]
        uni_fwd = [f for f in (_fwd(prices, s, date, d_next) for s in members) if f is not None]
        if not top5_fwd or not uni_fwd:
            continue
        top5 = float(np.mean(top5_fwd))
        univ = float(np.mean(uni_fwd))
        rows.append({"date": date, "xstd60": xstd60, "xstd252": xstd252,
                     "breadth": breadth, "top5_fwd": top5, "uni_fwd": univ,
                     "excess": top5 - univ})

    df = pd.DataFrame(rows)
    print(f"\n[{len(df)} rebalances, {start[:4]}-{end[:4]}]\n")
    return df


def _terc_report(df, col, label):
    df = df.dropna(subset=[col, "excess"]).copy()
    df["bkt"] = pd.qcut(df[col], 3, labels=["low", "mid", "high"])
    print(f"--- forward 20d by {label} tercile ---")
    g = df.groupby("bkt")
    for idx, sub in g:
        print(f"  {label}={str(idx):<6} top5 {sub['top5_fwd'].mean()*100:+6.2f}%  "
              f"uni {sub['uni_fwd'].mean()*100:+6.2f}%  EXCESS {sub['excess'].mean()*100:+6.2f}%  "
              f"excess_hit {(sub['excess']>0).mean()*100:4.0f}%  n={len(sub)}")
    # correlation
    print(f"    corr({label}, forward excess) = {df[col].corr(df['excess']):+.3f}")
    print(f"    corr({label}, forward top5)   = {df[col].corr(df['top5_fwd']):+.3f}\n")


def report(df):
    _terc_report(df, "xstd60", "XSTD60")
    _terc_report(df, "xstd252", "XSTD252")
    _terc_report(df, "breadth", "BREADTH")


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    print(f"\n[loaded in {time.time()-t0:.0f}s]")
    df = run_dispersion_diagnostic(bt)
    df.to_parquet("research/_dispersion_diag.parquet")
    report(df)
