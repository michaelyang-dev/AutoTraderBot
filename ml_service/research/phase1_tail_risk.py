"""
Phase 1.5 + 1.2 — n=5 vs n=8/10 tail metrics, and crash stress at 1.49x.
"Fatter tails is the one thing leverage can't afford." Compare CVaR and MaxDD
(not just Sharpe) across position counts, at 1x and at 1.49x leverage.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

LEV = 1.49
BORROW = 0.055  # annual margin interest on borrowed portion


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def lever_series(vals, L):
    """Apply L leverage to a daily NAV series (with borrow cost)."""
    r = vals.pct_change().fillna(0.0)
    daily_borrow = (L - 1) * BORROW / 252
    rl = L * r - daily_borrow
    return (1 + rl).cumprod()


def metrics(vals):
    r = vals.pct_change().dropna()
    peak = vals.cummax(); dd = (vals - peak) / peak
    maxdd = dd.min()
    # CVaR 5% on 20-day (monthly) returns
    m20 = vals.iloc[::20].pct_change().dropna()
    cvar = m20[m20 <= m20.quantile(0.05)].mean() if len(m20) > 20 else np.nan
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (252 / len(vals)) - 1
    return cagr, maxdd, cvar


bt = FastBacktester()
clear(bt)
base = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
        "sec_w": 0.0, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

print("=" * 70, flush=True)
print("TAIL RISK by position count (momentum top_n), 2018-2025")
print("=" * 70)
print(f"\n{'n':>4} {'CAGR 1x':>9} {'MaxDD 1x':>9} {'CVaR 1x':>9} | {'CAGR 1.49x':>11} {'MaxDD 1.49x':>12} {'CVaR 1.49x':>11}")
print("-" * 78)

worst_periods = {}
for n in [5, 8, 10]:
    cfg = {**base, "top_n": n}
    m = bt.run("2018-01-01", "2025-12-31", cfg)
    vals = m["daily_values"]
    c1, dd1, cv1 = metrics(vals)
    vlev = lever_series(vals, LEV)
    cL, ddL, cvL = metrics(vlev)
    print(f"{n:>4} {c1*100:>8.1f}% {dd1*100:>8.1f}% {cv1*100:>8.1f}% | "
          f"{cL*100:>10.1f}% {ddL*100:>11.1f}% {cvL*100:>10.1f}%")
    # worst drawdown date for n=5
    if n == 5:
        peak = vlev.cummax(); ddseries = (vlev - peak) / peak
        trough = ddseries.idxmin()
        worst_periods[n] = (trough, ddseries.min())

print(f"\nWorst 1.49x drawdown (n=5): {worst_periods[5][1]*100:.0f}% bottoming {pd.Timestamp(worst_periods[5][0]).date()}")
print("\nReads: CVaR = average of the worst 5% of monthly returns (tail severity).")
print("At 1.49x, MaxDD and CVaR are what leverage amplifies — compare n=5 vs n=8/10.")
