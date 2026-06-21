"""
Idea 3 — Book-crowding / stretch crash predictor: DIAGNOSTIC
===========================================================
Thesis: the system de-risks on a MARKET signal (SPY<SMA200) + NAV vol-scaling.
But its drawdowns come from its OWN concentrated high-flyer book, which can crack
while SPY is still above its 200d SMA (COVID, 2021Q4 growth unwind, 2022). A
signal built from the BOOK's own state — how stretched the holdings are vs their
own history, their realized vol, and their internal correlation (fragility when
the 5 names move as one) — might see book-specific crashes the index filter
misses.

Cross-sectional vol/stretch FILTERS are already known-dead (vol=momentum=alpha,
roadmap). So this is strictly a TIME-SERIES exposure test: is the book unusually
stretched/vol/correlated vs ITS OWN trailing norm, and does that predict worse
forward book returns / deeper forward drawdowns — INCREMENTAL to SPY<SMA200?

Measure per 20d rebalance for the top-5 book:
  - fwd20 return, and fwd worst-path drawdown over next 20d (crash capture)
  - spy_bull (SPY>SMA200)            [what the system already uses]
  - book_vol   = mean vol_20d of holdings
  - book_strch = mean dist_sma200 of holdings
  - book_corr  = mean pairwise 60d-return corr of holdings (fragility)
  - *_z = z-score of each vs trailing 24-rebalance window (time-series anomaly)
Report forward stats split by SPY regime AND by book-state terciles WITHIN regime.

Run: cd ml_service && ./venv/bin/python research/alpha_crowding_diag.py
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


def _path(prices_df, syms, weights, d0, d1):
    """Equal-weighted (by given weights) forward path return + worst drawdown
    between trading dates d0..d1."""
    sub = prices_df.loc[d0:d1, [s for s in syms if s in prices_df.columns]]
    sub = sub.dropna(axis=1, how="any")
    if sub.shape[0] < 2 or sub.shape[1] == 0:
        return None, None
    w = np.array([weights.get(s, 0) for s in sub.columns])
    if w.sum() <= 0:
        w = np.ones(len(sub.columns))
    w = w / w.sum()
    norm = sub / sub.iloc[0]
    port = (norm * w).sum(axis=1)
    fwd = port.iloc[-1] - 1.0
    dd = (port / port.cummax() - 1.0).min()
    return float(fwd), float(dd)


def run_crowding_diagnostic(bt, start="2001-01-01", end="2025-12-31",
                            top_n=5, rebal_days=20):
    bt.uni.get_sp500 = bt._get_sp1500
    prices_df = bt.prices
    dr = bt.uni._daily_returns
    dates = [d for d in sorted(prices_df.index)
             if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    rebal_idx = list(range(0, len(dates), rebal_days))

    rows = []
    for k, di in enumerate(rebal_idx[:-1]):
        date = dates[di]
        d_next = dates[rebal_idx[k + 1]]
        w = strategy1_momentum_reversal(date, bt.uni, di, top_n=top_n,
                                        rebal_days=rebal_days)
        if not w:
            continue
        syms = list(w.keys())

        vol20 = bt.uni.get_feature_map(date, "vol_20d")
        strch = bt.uni.get_feature_map(date, "dist_sma200")
        book_vol = np.nanmean([vol20.get(s, np.nan) for s in syms])
        book_strch = np.nanmean([strch.get(s, np.nan) for s in syms])

        # internal correlation (fragility): mean off-diagonal of 60d daily-ret corr
        cols = [s for s in syms if s in dr.columns]
        book_corr = np.nan
        if len(cols) >= 2:
            rr = dr[cols].loc[:date].iloc[-60:].dropna(axis=1, how="any")
            if rr.shape[1] >= 2 and rr.shape[0] >= 30:
                c = rr.corr().values
                book_corr = float((c.sum() - len(c)) / (len(c) * (len(c) - 1)))

        # market regime
        spy = prices_df["SPY"].loc[:date].dropna()
        spy_bull = bool(spy.iloc[-1] > spy.iloc[-200:].mean()) if len(spy) >= 200 else True

        fwd, dd = _path(prices_df, syms, w, date, d_next)
        if fwd is None:
            continue
        rows.append({"date": date, "spy_bull": spy_bull,
                     "book_vol": book_vol, "book_strch": book_strch,
                     "book_corr": book_corr, "fwd20": fwd, "fwd_dd": dd})

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # time-series z-scores vs trailing 24-rebalance window
    for c in ["book_vol", "book_strch", "book_corr"]:
        m = df[c].rolling(24, min_periods=8).mean()
        s = df[c].rolling(24, min_periods=8).std()
        df[c + "_z"] = (df[c] - m) / s
    print(f"\n[{len(df)} rebalances, {start[:4]}-{end[:4]}]\n")
    return df


def report(df):
    print(f"=== unconditional ===  fwd20 mean {df['fwd20'].mean()*100:+.2f}%  "
          f"worst-path DD mean {df['fwd_dd'].mean()*100:+.2f}%  "
          f"crash freq(fwd<-10%) {(df['fwd20']<-0.10).mean()*100:.0f}%\n")

    print("=== by SPY regime (what system already uses) ===")
    for b, sub in df.groupby("spy_bull"):
        print(f"  spy_bull={b!s:<5} fwd20 {sub['fwd20'].mean()*100:+6.2f}%  "
              f"DD {sub['fwd_dd'].mean()*100:+6.2f}%  crash% {(sub['fwd20']<-0.10).mean()*100:4.0f}  n={len(sub)}")
    print()

    print("=== book-state terciles, INCREMENTAL within SPY-bull only ===")
    bull = df[df["spy_bull"]].dropna(subset=["book_vol_z", "book_corr_z", "book_strch_z"])
    for col in ["book_vol_z", "book_strch_z", "book_corr_z"]:
        sub = bull.dropna(subset=[col]).copy()
        if len(sub) < 12:
            continue
        sub["bkt"] = pd.qcut(sub[col], 3, labels=["low", "mid", "high"], duplicates="drop")
        print(f"  -- {col} (within SPY-bull) --")
        for idx, g in sub.groupby("bkt"):
            print(f"     {col}={str(idx):<5} fwd20 {g['fwd20'].mean()*100:+6.2f}%  "
                  f"DD {g['fwd_dd'].mean()*100:+6.2f}%  crash% {(g['fwd20']<-0.10).mean()*100:4.0f}  n={len(g)}")
        print(f"     corr({col}, fwd20)={sub[col].corr(sub['fwd20']):+.3f}  "
              f"corr({col}, fwd_dd)={sub[col].corr(sub['fwd_dd']):+.3f}")
    print()


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    print(f"\n[loaded in {time.time()-t0:.0f}s]")
    df = run_crowding_diagnostic(bt)
    df.to_parquet("research/_crowding_diag.parquet")
    report(df)
