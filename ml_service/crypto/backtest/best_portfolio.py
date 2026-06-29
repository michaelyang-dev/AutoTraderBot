"""
FULL backtest of the recommended book: BTC/ETH funding carry (1.5x, liquidation-modeled, costs)
+ risk-managed beta sleeve. Same rigor as the carry audit — nothing inflated.

Both sleeves on 6yr survivorship-complete Binance data (BTC/ETH never delisted → clean):
  CARRY : armored_backtest.carry — funding − liquidation losses − costs (the HONEST carry, not
          funding-only). 1.5x. BTC/ETH have 0 squeeze liquidations historically.
  BETA  : beta_product.strategy — 60/40 BTC/ETH, vol-targeted, 200d regime de-risk, costs.

Reports, for full / 2023+ (OOS) / each half:
  CAGR, vol, Sharpe (DAILY = flattered by carry's tiny vol) AND TAIL-ADJUSTED (inject the carry's
  counterparty tail), MaxDD. Plus a weight sweep (is 60/40 a plateau or a knife-edge?) and the
  beta vol-target tradeoff. Look-ahead: both sleeves use lagged signals (verified in their audits).
Run:  python crypto/backtest/best_portfolio.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from armored_backtest import load as aload, carry as acarry
from beta_product import strategy as bstrat, load as bload

CARRY_COUNTERPARTY_TAIL = 0.34 * 1.5 / 1.5     # HL insolvency at 1.5x w/ venue-split ≈ -34% (per counterparty_risk)


def metrics(x, carry_w=0.0, lo=None, hi=None):
    x = x.loc[lo:hi].dropna() if (lo or hi) else x.dropna()
    if len(x) < 30 or x.std() == 0:
        return dict(cagr=0, vol=0, sharpe=0, sh_tail=0, maxdd=0)
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; vol = x.std() * np.sqrt(365); sh = x.mean() / x.std() * np.sqrt(365)
    md = ((nav - nav.cummax()) / nav.cummax()).min()
    xt = x.copy(); xt.loc[nav.idxmax()] -= CARRY_COUNTERPARTY_TAIL * carry_w     # inject carry tail
    nav2 = (1 + xt).cumprod()
    return dict(cagr=cagr, vol=vol, sharpe=sh, sh_tail=xt.mean() / xt.std() * np.sqrt(365),
                maxdd=md, maxdd_tail=((nav2 - nav2.cummax()) / nav2.cummax()).min())


if __name__ == "__main__":
    F, C = aload()
    Fe, Ce = F[["BTC", "ETH"]], C[["BTC", "ETH"]]
    carry15, _ = acarry(Fe, Ce, top_k=2, lev=1.5, quality=False, liq_pct=0.40)   # honest carry, costs+liq in
    beta40 = bstrat(bload(), {"BTC": 60, "ETH": 40}, vol_target=0.40, regime=True)
    beta30 = bstrat(bload(), {"BTC": 60, "ETH": 40}, vol_target=0.30, regime=True)
    idx = carry15.index.intersection(beta40.index)
    cr, b40, b30 = carry15.loc[idx], beta40.loc[idx], beta30.loc[idx]
    mid = idx[len(idx) // 2].strftime("%Y-%m-%d")
    print("=" * 100)
    print("FULL PORTFOLIO BACKTEST — BTC/ETH carry(1.5x,liq+costs) + beta | corr=%.2f | %s→%s"
          % (cr.corr(b40), idx.min().date(), idx.max().date()))
    print("=" * 100)

    print("\n[1] 60/40 carry/beta(vt40) across periods (Sharpe* = daily illusion; sh_tail = honest):")
    print(f"    {'period':<16}{'CAGR':>7}{'vol':>6}{'Sharpe*':>9}{'sh_tail':>9}{'MaxDD':>8}{'DD_tail':>9}")
    port = 0.6 * cr + 0.4 * b40
    for lab, lo, hi in [("full 6yr", None, None), ("2023+ (OOS)", "2023", None),
                        ("first half", None, mid), ("second half", mid, None)]:
        m = metrics(port, carry_w=0.6, lo=lo, hi=hi)
        print(f"    {lab:<16}{m['cagr']*100:>6.0f}%{m['vol']*100:>5.0f}%{m['sharpe']:>9.2f}{m['sh_tail']:>9.2f}{m['maxdd']*100:>7.0f}%{m['maxdd_tail']*100:>8.0f}%", flush=True)

    print("\n[2] WEIGHT SWEEP (full) — is 60/40 robust or a knife-edge? (carry/beta)")
    print(f"    {'weights':<14}{'CAGR':>7}{'Sharpe*':>9}{'sh_tail':>9}{'MaxDD':>8}")
    for cw in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0]:
        p = cw * cr + (1 - cw) * b40; m = metrics(p, carry_w=cw)
        print(f"    {f'{cw:.0%}/{1-cw:.0%}':<14}{m['cagr']*100:>6.0f}%{m['sharpe']:>9.2f}{m['sh_tail']:>9.2f}{m['maxdd']*100:>7.0f}%", flush=True)

    print("\n[3] BETA vol-target tradeoff (60/40, full):")
    for nm, bx in [("beta vt30 (calm)", b30), ("beta vt40 (aggressive)", b40)]:
        p = 0.6 * cr + 0.4 * bx; m = metrics(p, carry_w=0.6)
        print(f"    {nm:<24} CAGR {m['cagr']*100:>3.0f}%  sh_tail {m['sh_tail']:.2f}  MaxDD {m['maxdd']*100:>3.0f}%", flush=True)

    print("\n[4] per-year (60/40, vt40):")
    yr = port.groupby(port.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("    " + "  ".join("%d:%+.0f%%" % (y, v) for y, v in yr.items()))
    print("\n  Honest read: trust the 2023+ (OOS) row + sh_tail/DD_tail columns. The full-period CAGR is")
    print("  lifted by the 2020-21 mania (beta). Counterparty tail injected at 0.6*-34%.")
