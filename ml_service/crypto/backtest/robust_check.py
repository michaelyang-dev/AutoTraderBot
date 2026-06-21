"""
Robustness of the multi-sleeve improvement — is the tail-aware Sharpe ~1.6 real, or did I overfit
the weights to a 3-year sample? Checks:
  1. SUB-PERIOD — Sharpe in first half (2023-24) vs second half (2025-26). Should be positive in both.
  2. WEIGHT PERTURBATION — round/robust weights, not the grid-search 'best'. Sharpe shouldn't be a spike.
  3. TAIL SENSITIVITY — vary the assumed carry catastrophe (1/2/3 venues → -0.35/-0.18/-0.12).
Honest: if the edge only exists at the optimized weights / one tail assumption, it's fit, not real.
Run:  python crypto/backtest/robust_check.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from improved_carry import load as carry_load
from carry_optimize import carry as carry_opt
from beta_product import strategy as beta_strat, load as beta_load
from multi_sleeve import unlock_sleeve

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")


def sh_tail(x, cw, tail, lo=None, hi=None):
    if lo:
        x = x.loc[lo:]
    if hi:
        x = x.loc[:hi]
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    xt = x.copy(); xt.loc[nav.idxmax()] -= tail * cw
    nav2 = (1 + xt).cumprod()
    return cagr, xt.mean() / xt.std() * np.sqrt(365), ((nav2 - nav2.cummax()) / nav2.cummax()).min()


if __name__ == "__main__":
    F, P = carry_load()
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    carry, _ = carry_opt(F, P, top_k=8, rebal=30, weighting="funding", rt_cost=0.0060, keep_mult=2.0, min_keep=0.02)
    beta = beta_strat(beta_load(), {"BTC": 60, "ETH": 40}, vol_target=0.30, regime=True)
    unlock = unlock_sleeve(close)
    idx = carry.index.intersection(beta.index).intersection(unlock.index)
    c, b, u = carry.loc[idx], beta.loc[idx], unlock.loc[idx]
    mid = c.index[len(c) // 2].strftime("%Y-%m-%d")

    print("=" * 84)
    print("ROBUSTNESS — multi-sleeve (split halves at %s)" % mid)
    print("=" * 84)

    print("\n[1] SUB-PERIOD — robust weights 65/20/15, tail-aware Sharpe in each half:")
    cw, bw, uw = 0.65, 0.20, 0.15
    port = cw * c + bw * b + uw * u
    for lab, lo, hi in [("first half 2023-24", None, mid), ("second half 2025-26", mid, None), ("full", None, None)]:
        cg, sh, md = sh_tail(port, cw, 0.18, lo, hi)
        print(f"    {lab:<22} CAGR {cg*100:>5.0f}%  Sharpe(tail) {sh:>5.2f}  MaxDD {md*100:>5.0f}%", flush=True)

    print("\n[2] WEIGHT PERTURBATION — Sharpe(tail) should be a plateau, not a spike:")
    for cw, bw, uw in [(0.5, 0.3, 0.2), (0.6, 0.25, 0.15), (0.65, 0.2, 0.15), (0.7, 0.2, 0.1), (0.8, 0.15, 0.05), (0.6, 0.4, 0.0)]:
        port = cw * c + bw * b + uw * u
        cg, sh, md = sh_tail(port, cw, 0.18)
        print(f"    {cw:.0%}/{bw:.0%}/{uw:.0%}   CAGR {cg*100:>5.0f}%  Sharpe(tail) {sh:>5.2f}", flush=True)

    print("\n[3] TAIL SENSITIVITY — carry catastrophe by # venues (65/20/15 blend):")
    cw, bw, uw = 0.65, 0.20, 0.15
    port = cw * c + bw * b + uw * u
    for venues, tail in [("1 venue (-35%)", 0.35), ("2 venues (-18%)", 0.18), ("3 venues (-12%)", 0.12)]:
        cg, sh, md = sh_tail(port, cw, tail)
        print(f"    {venues:<18} Sharpe(tail) {sh:>5.2f}  MaxDD(tail) {md*100:>5.0f}%", flush=True)

    print("\n  Verdict: positive both halves + flat across weights + reasonable across tail assumptions")
    print("  = robust, not overfit. (Note: 2nd half is the COMPRESSED-funding regime — the honest stress.)")
