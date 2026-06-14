"""
Phase 1.4 — Volatility scaling test.
Vol is more forecastable than returns, so scaling exposure inverse to realized
vol should cut crash risk more robustly than the backward-looking regime switch.
Compare: baseline v12 vs v12 + vol-scaling (targets 20% and 15% annualized).
Metric that matters for a leveraged book: MaxDD and CVaR, not just CAGR.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np

LEV = 1.49


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(vals, L=1.0):
    r = vals.pct_change().fillna(0.0)
    if L != 1.0:
        r = L * r - (L - 1) * 0.055 / 252
        vals = (1 + r).cumprod()
    peak = vals.cummax(); maxdd = ((vals - peak) / peak).min()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (252 / len(vals)) - 1
    m20 = vals.iloc[::20].pct_change().dropna()
    cvar = m20[m20 <= m20.quantile(0.05)].mean()
    return cagr, maxdd, cvar


bt = FastBacktester()
clear(bt)
base = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
        "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

configs = [
    ("baseline v12 (regime overlay only)", base),
    ("+ vol-scaling target 20%", {**base, "vol_scaling": True, "vol_target": 0.20}),
    ("+ vol-scaling target 15%", {**base, "vol_scaling": True, "vol_target": 0.15}),
]

print("=" * 76, flush=True)
print("VOL-SCALING TEST (v12 n=5, 2018-2025) — does it cut the tail better?")
print("=" * 76)
print(f"\n{'config':<36}{'CAGR':>7}{'MaxDD':>8}{'CVaR':>7} | {'CAGR@1.49':>10}{'MaxDD@1.49':>11}")
print("-" * 76)
for label, cfg in configs:
    m = bt.run("2018-01-01", "2025-12-31", cfg)
    v = m["daily_values"]
    c1, dd1, cv1 = stats(v, 1.0)
    cL, ddL, cvL = stats(v, LEV)
    print(f"{label:<36}{c1*100:>6.1f}%{dd1*100:>7.1f}%{cv1*100:>6.1f}% | {cL*100:>9.1f}%{ddL*100:>10.1f}%")
print("\nVERDICT: vol-scaling wins if it cuts MaxDD/CVaR at 1.49x for less CAGR")
print("cost than the regime overlay did (which gave up ~1.6pp CAGR for ~4.7pp MaxDD).")
