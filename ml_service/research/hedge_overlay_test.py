"""
HEDGE-OVERLAY TEST (answers "does shorting index futures to hedge our long book help?").

A short S&P-futures hedge of ratio a is modeled as: hedged_daily_return = book_r - a * SPY_r.
(Holding a short SPY-futures position sized at a x the book, marked daily — standard overlay.)
Also tests a REGIME-CONDITIONAL hedge: short only on days SPY closed < its SMA200 the PRIOR
day (no lookahead) — i.e. "insurance only in bear markets".

Read-only use of FastBacktester (deployed condition: enhanced OFF, SI OFF, exact V12). One run,
then pure overlay math for every variant. No source files modified. 1x leverage (live ~1.49x
scales all figures ~proportionally). NOTE: this is the FUTURES hedge (linear, no premium). The
PUT-option hedge (capped loss, pays premium) needs option prices — not available until the July
WRDS/OptionMetrics refresh — so it is explicitly NOT tested here.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
START, END = "2018-01-02", "2025-12-31"


def stats(r):
    r = r.dropna()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    curve = (1 + r).cumprod()
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    mdd = ((curve - curve.cummax()) / curve.cummax()).min()
    return cagr, vol, sharpe, mdd


def main():
    bt = FastBacktester()
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

    m = bt.run(START, END, V12)
    book = m["daily_values"].pct_change().dropna()

    spy_px = bt.prices["SPY"].reindex(book.index.union(bt.prices["SPY"].index)).ffill()
    spy_r = spy_px.pct_change().reindex(book.index)
    # bear mask: SPY < its 200d SMA as of the PRIOR trading day (shift(1) = no lookahead)
    sma200 = spy_px.rolling(200).mean()
    bear_prior = (spy_px < sma200).shift(1).reindex(book.index).fillna(False)

    # book beta to SPY (for context / a "beta-neutral" hedge ratio)
    beta = np.cov(book.fillna(0), spy_r.fillna(0))[0, 1] / np.var(spy_r.fillna(0))

    print("=" * 80, flush=True)
    print(f"HEDGE-OVERLAY TEST — v12 long book vs short-SPY-futures overlay ({START}->{END}, 1x)", flush=True)
    print(f"book beta to SPY = {beta:.2f} | bear days (SPY<SMA200 prior) = {bear_prior.mean():.0%}", flush=True)
    print("=" * 80, flush=True)
    hdr = f"{'variant':<34}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}"
    print(hdr); print("-" * len(hdr), flush=True)

    def row(name, r):
        c, v, s, d = stats(r)
        print(f"{name:<34}{c:>+7.1%}{v:>7.1%}{s:>8.2f}{d:>+8.1%}", flush=True)

    row("NO HEDGE (book as-is)", book)
    print("- continuous short-SPY hedge (always on) -", flush=True)
    for a in [0.25, 0.50, beta, 1.0]:
        row(f"  hedge {a:.2f}x SPY (continuous)", book - a * spy_r.fillna(0))
    print("- regime hedge (short only when SPY<SMA200) -", flush=True)
    for a in [0.50, 1.0]:
        row(f"  hedge {a:.2f}x SPY (bear-only)", book - a * spy_r.fillna(0) * bear_prior)

    print("\nReads: continuous hedge trades CAGR for DD ~linearly (book is mostly beta);", flush=True)
    print("bear-only hedge = the SPY<SMA200 de-risk we already run, in overlay form.", flush=True)


if __name__ == "__main__":
    main()
