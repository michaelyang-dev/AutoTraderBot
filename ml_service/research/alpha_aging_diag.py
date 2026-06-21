"""
Idea 1 — Momentum life-cycle / position aging: DIAGNOSTIC (does the effect exist?)
=================================================================================
Thesis: a held momentum name's forward return depends on its life-cycle state
(how long it's been a top pick = AGE, and whether its momentum is still
ACCELERATING or rolling over), not just its current rank. The existing sleeve
re-ranks fresh every 20d with NO memory of holding duration and NO acceleration
gate beyond a 40% trailing stop.

This script does NOT build a strategy yet. It measures the conditional forward
20d/60d return of the top-5 momentum book, bucketed by:
  - AGE: consecutive prior 20d-rebalances the name has been in the top-5
  - ACCEL: recent 1-month return sign (rolling over vs still pushing)
  - DECEL: recent 3m leg annualized minus older 9m leg annualized
  - DD52: distance below trailing 252d high (late-stage proxy)

If forward returns are flat across buckets -> idea is dead, move on (honest).
If "fresh + accelerating" >> "stale + rolling-over" -> there is exploitable
path structure and a trading variant is warranted.

Run: cd ml_service && ./venv/bin/python research/alpha_aging_diag.py
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


def _fwd_return(prices, sym, d0, d1):
    """Total return of sym between trading dates d0 and d1 (None if missing)."""
    s = prices.get(sym)
    if s is None:
        return None
    try:
        p0 = s.loc[:d0].dropna()
        p1 = s.loc[:d1].dropna()
    except Exception:
        return None
    if len(p0) == 0 or len(p1) == 0:
        return None
    a, b = p0.iloc[-1], p1.iloc[-1]
    if a is None or b is None or a <= 0 or np.isnan(a) or np.isnan(b):
        return None
    return b / a - 1.0


def run_aging_diagnostic(bt, start="2001-01-01", end="2025-12-31",
                         top_n=5, rebal_days=20, track_pool=30):
    prices = {c: bt.prices[c].dropna() for c in bt.prices.columns}
    bt.uni.get_sp500 = bt._get_sp1500
    dates = [d for d in sorted(bt.prices.index)
             if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    rebal_idx = list(range(0, len(dates), rebal_days))

    age = {}          # sym -> consecutive rebalances in top_n
    records = []      # one row per (rebal, held name)

    for k, di in enumerate(rebal_idx):
        date = dates[di]
        # momentum weights (top_n) and a wider candidate pool for context
        w = strategy1_momentum_reversal(date, bt.uni, di, top_n=top_n,
                                        rebal_days=rebal_days)
        if not w:
            age = {}
            continue
        held = list(w.keys())

        # next rebalance date for forward returns
        d_next = dates[rebal_idx[k + 1]] if k + 1 < len(rebal_idx) else None
        di60 = min(di + 60, len(dates) - 1)
        d_60 = dates[di60]

        ret20 = bt.uni.get_feature_map(date, "ret_20d")
        ret60 = bt.uni.get_feature_map(date, "ret_60d")
        ret252 = bt.uni.get_feature_map(date, "ret_252d")
        d52 = bt.uni.get_feature_map(date, "dist_52w_high")

        new_age = {}
        for sym in held:
            a = age.get(sym, 0) + 1
            new_age[sym] = a

            r20 = ret20.get(sym)
            r60 = ret60.get(sym)
            r252 = ret252.get(sym)
            dd = d52.get(sym)

            # acceleration: recent 3m leg vs older (12m-3m) leg, annualized
            accel = np.nan
            if r252 is not None and r60 is not None and (1 + r60) > 0:
                older = (1 + r252) / (1 + r60) - 1.0   # ~9-month older leg
                recent_ann = (1 + r60) ** (252 / 60) - 1.0
                older_ann = (1 + older) ** (252 / 192) - 1.0
                accel = recent_ann - older_ann

            fwd = _fwd_return(prices, sym, date, d_next) if d_next is not None else None
            fwd60 = _fwd_return(prices, sym, date, d_60)

            records.append({
                "date": date, "sym": sym, "age": a,
                "rollover": (r20 is not None and r20 < 0),   # 1m return negative
                "accel": accel,
                "dd52": dd,
                "fwd20": fwd, "fwd60": fwd60,
            })
        age = new_age

    df = pd.DataFrame(records)
    print(f"\n[{len(df)} held-name observations, {df['date'].nunique()} rebalances, "
          f"{start[:4]}-{end[:4]}]\n")
    return df


def _bucket_report(df, col, label, fwd="fwd20"):
    sub = df.dropna(subset=[fwd])
    print(f"--- forward {fwd} by {label} ---")
    g = sub.groupby(col)[fwd]
    out = g.agg(["mean", "median", "count"])
    out["hit"] = sub.groupby(col)[fwd].apply(lambda x: (x > 0).mean())
    for idx, row in out.iterrows():
        print(f"  {str(idx):<18} mean {row['mean']*100:+6.2f}%  med {row['median']*100:+6.2f}%  "
              f"hit {row['hit']*100:4.1f}%  n={int(row['count'])}")
    print()


def report(df):
    # AGE buckets
    df["age_bkt"] = pd.cut(df["age"], [0, 1, 3, 6, 100],
                           labels=["fresh(1)", "young(2-3)", "mature(4-6)", "stale(7+)"])
    for fwd in ("fwd20", "fwd60"):
        _bucket_report(df, "age_bkt", "AGE", fwd)
    # ROLLOVER (1m return negative while held)
    _bucket_report(df, "rollover", "ROLLOVER(1m<0)", "fwd20")
    _bucket_report(df, "rollover", "ROLLOVER(1m<0)", "fwd60")
    # ACCEL terciles
    df["accel_bkt"] = pd.qcut(df["accel"], 3, labels=["decel", "mid", "accel"])
    _bucket_report(df, "accel_bkt", "ACCEL tercile", "fwd20")
    _bucket_report(df, "accel_bkt", "ACCEL tercile", "fwd60")
    # DD52 terciles (distance below 52w high; less negative = nearer high)
    df["dd_bkt"] = pd.qcut(df["dd52"], 3, labels=["far(deep)", "mid", "near-high"])
    _bucket_report(df, "dd_bkt", "DIST-52W-HIGH tercile", "fwd20")


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    print(f"\n[loaded in {time.time()-t0:.0f}s]")
    df = run_aging_diagnostic(bt)
    df.to_parquet("research/_aging_diag.parquet")
    report(df)
