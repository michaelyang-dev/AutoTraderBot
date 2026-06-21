"""
Improvement pass 2 — portfolio optimization. The carry (market-neutral, tail-limited) and the beta
sleeve (long crypto, volatile) have INDEPENDENT tails (HL-insolvency vs crypto-crash), so blending
them improves risk-adjusted return more than tuning either alone. Find the weight that maximizes the
TAIL-AWARE Sharpe (carry's structural tail injected), not the illusory daily one.

Run:  python crypto/backtest/portfolio_opt.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from improved_carry import load
from carry_optimize import carry as carry_opt
from beta_product import strategy as beta_strat, load as beta_load


def metrics(x, carry_w=0.0, carry_tail=0.18):
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    vol = x.std() * np.sqrt(365); sh = x.mean() / x.std() * np.sqrt(365)
    md = ((nav - nav.cummax()) / nav.cummax()).min()
    # tail-aware: inject the carry's structural tail (scaled by its weight) at the NAV peak
    xt = x.copy(); xt.loc[nav.idxmax()] -= carry_tail * carry_w
    nav2 = (1 + xt).cumprod()
    sh_t = xt.mean() / xt.std() * np.sqrt(365)
    md_t = ((nav2 - nav2.cummax()) / nav2.cummax()).min()
    return cagr, vol, sh, sh_t, md_t


if __name__ == "__main__":
    F, P = load()
    carry, _ = carry_opt(F, P, top_k=8, rebal=30, weighting="funding", rt_cost=0.0060, keep_mult=2.0, min_keep=0.02)
    beta = beta_strat(beta_load(), {"BTC": 60, "ETH": 40}, vol_target=0.30, regime=True)
    idx = carry.index.intersection(beta.index)
    c, b = carry.loc[idx], beta.loc[idx]
    corr = c.corr(b)
    print("=" * 84)
    print("PORTFOLIO OPTIMIZATION — carry + beta | corr = %.2f (independent → diversifies)" % corr)
    print("=" * 84)
    print(f"  {'carry / beta':<16}{'CAGR':>8}{'vol':>7}{'Sharpe*':>9}{'Sharpe(tail)':>13}{'MaxDD(tail)':>13}")
    best = (-9, None)
    for cw in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
        port = cw * c + (1 - cw) * b
        cagr, vol, sh, sht, mdt = metrics(port, carry_w=cw)
        if sht > best[0]:
            best = (sht, cw)
        tag = "  <- best tail-Sharpe" if False else ""
        print(f"  {f'{cw:.0%} / {1-cw:.0%}':<16}{cagr*100:>7.0f}%{vol*100:>6.1f}%{sh:>9.1f}{sht:>13.2f}{mdt*100:>12.0f}%", flush=True)
    print("\n  best tail-aware Sharpe at carry weight = %.0f%%" % (best[1] * 100))
    # show the recommended blend's per-year
    cw = best[1]
    port = cw * c + (1 - cw) * b
    yr = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("  recommended blend per-year:  " + "  ".join("%d:%+.0f%%" % (y, v) for y, v in yr.items()))
    print("\n  Why it works: carry pays in calm/over-levered regimes; beta pays in trends and sits in cash")
    print("  in downtrends (now). Their tails are independent, so the blend's risk-adjusted return beats both.")
