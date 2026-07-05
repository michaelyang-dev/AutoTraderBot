"""
Tranche staggering test — v12 book split into 4 offset tranches.
================================================================
HYPOTHESIS: live v12 rebalances 100% of the book every 20 trading days.
Split into 4 equal tranches on offset schedules (0/5/10/15 trading days),
each tranche itself rebalancing every 20d => same per-name holding period
and same per-tranche turnover/costs, but portfolio-level signal age drops
from ~10d avg to ~2.5d and the start-date lottery collapses toward its mean.

METHOD (honest):
  - FastBacktester instantiated ONCE (read-only use; no source files modified).
  - Honest deployed data condition (matches locate_v12_number.py /
    v12_ground_truth.py condition A = DEPLOYED): enhanced snapshots OFF,
    short interest OFF. V12 config exactly as the honest scripts use it
    (no vol_scaling / trend_scale flags — locate_v12_number.py sets none).
  - run() called 4x with starts offset ~5 trading days:
      2018-01-02 (0d), 2018-01-09 (+5td), 2018-01-17 (+10td), 2018-01-24 (+15td)
  - Daily NAV comes from run()'s returned "daily_values" Series (no patching).
  - Tranche book = equal-weight average of the 4 tranches' DAILY RETURNS on
    the common (inner-join) date index. Averaging RETURNS, never NAVs.
    Costs are already inside each tranche's own NAV path.
  - All comparison stats computed on the SAME common date index so single
    starts vs tranche book is apples-to-apples.

Run: cd ml_service && OMP_NUM_THREADS=1 ./venv/bin/python research/tranche_stagger_test.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from main_production_backtest import FastBacktester

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

# Offsets 0 / 5 / 10 / 15 trading days (verified against 2018 trading calendar)
STARTS = ["2018-01-02", "2018-01-09", "2018-01-17", "2018-01-24"]
END = "2025-12-31"


def clear_enhanced(bt):
    """Deployed condition: enhanced snapshot data OFF (as in locate_v12_number.py)."""
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}


def clear_si(bt):
    """Deployed condition: short interest OFF (hurts -3pp, disabled live)."""
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats_from_returns(r):
    """CAGR / ann.vol / Sharpe / MaxDD from a daily-return Series."""
    r = r.dropna()
    years = (r.index[-1] - r.index[0]).days / 365.25
    curve = (1 + r).cumprod()
    cagr = curve.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    max_dd = ((curve - curve.cummax()) / curve.cummax()).min()
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}


def main():
    print("=" * 78, flush=True)
    print("TRANCHE STAGGER TEST — v12, 4 tranches at 0/5/10/15 td offsets, 20d rebal",
          flush=True)
    print("Data condition: DEPLOYED (enhanced OFF, SI OFF) — honest baseline",
          flush=True)
    print("=" * 78, flush=True)

    bt = FastBacktester()
    clear_enhanced(bt)
    clear_si(bt)

    rets = {}          # start -> daily return series (from run's own NAV)
    full_run_stats = {}  # start -> run()'s own full-period metrics (transparency)
    for st in STARTS:
        t0 = time.time()
        m = bt.run(st, END, V12)
        dt = time.time() - t0
        if m is None:
            print(f"  RUN FAILED for start {st}", flush=True)
            sys.exit(1)
        dv = m["daily_values"]
        rets[st] = dv.pct_change().dropna()
        full_run_stats[st] = m
        print(f"  ran start {st}: {len(dv)} days, {dt:.0f}s "
              f"(full-run CAGR {m['cagr']:+.1%}, Sharpe {m['sharpe']:.2f}, "
              f"MaxDD {m['max_dd']:.1%})", flush=True)

    # Common (inner-join) date index across all 4 tranches
    common = rets[STARTS[0]].index
    for st in STARTS[1:]:
        common = common.intersection(rets[st].index)
    print(f"\nCommon index: {common[0].date()} -> {common[-1].date()} "
          f"({len(common)} days)", flush=True)

    aligned = pd.DataFrame({st: rets[st].reindex(common) for st in STARTS})
    assert not aligned.isna().any().any(), "NaNs after inner join alignment"

    # Tranche book: equal-weight average of DAILY RETURNS
    book = aligned.mean(axis=1)

    rows = []
    for st in STARTS:
        rows.append((f"single start {st}", stats_from_returns(aligned[st])))
    singles = [r[1] for r in rows]
    mean_of_singles = {k: float(np.mean([s[k] for s in singles]))
                       for k in ("cagr", "vol", "sharpe", "max_dd")}
    book_stats = stats_from_returns(book)

    hdr = f"{'Variant':<38} {'CAGR':>7} {'Vol':>6} {'Sharpe':>7} {'MaxDD':>7}"
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for name, s in rows:
        print(f"{name:<38} {s['cagr']:>+6.1%} {s['vol']:>5.1%} "
              f"{s['sharpe']:>7.2f} {s['max_dd']:>6.1%}", flush=True)
    print("-" * len(hdr), flush=True)
    print(f"{'MEAN of single-start stats':<38} {mean_of_singles['cagr']:>+6.1%} "
          f"{mean_of_singles['vol']:>5.1%} {mean_of_singles['sharpe']:>7.2f} "
          f"{mean_of_singles['max_dd']:>6.1%}", flush=True)
    print(f"{'TRANCHE BOOK (avg of returns)':<38} {book_stats['cagr']:>+6.1%} "
          f"{book_stats['vol']:>5.1%} {book_stats['sharpe']:>7.2f} "
          f"{book_stats['max_dd']:>6.1%}", flush=True)

    cagrs = [s["cagr"] for s in singles]
    spread = max(cagrs) - min(cagrs)
    print(f"\nStart-date lottery width (best - worst single CAGR, common index): "
          f"{spread:+.1%} ({max(cagrs):+.1%} vs {min(cagrs):+.1%})", flush=True)

    # Average pairwise correlation of tranche daily returns (context)
    corr = aligned.corr()
    off_diag = corr.values[np.triu_indices(len(STARTS), k=1)]
    print(f"Avg pairwise daily-return correlation between tranches: "
          f"{off_diag.mean():.3f}", flush=True)

    print("\nNOTE: all stats above are computed on the common inner-join index "
          f"({common[0].date()} onward). run()'s own full-period metrics per "
          "start are printed at the top for transparency.", flush=True)


if __name__ == "__main__":
    main()
