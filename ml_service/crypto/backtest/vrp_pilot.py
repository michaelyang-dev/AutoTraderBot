"""
PILOT A — Vol-Risk-Premium. Is selling crypto vol a real, persistent edge?

Thesis (structural, not price-prediction): implied vol (DVOL) systematically exceeds realized vol
because hedgers/speculators overpay for crypto optionality. Selling vol harvests IV − RV.

Tests:
  1. PREMIUM: is IV > subsequently-realized vol, on average and out-of-sample?
  2. SHORT-VARIANCE CARRY: daily P&L of being short variance struck at IV = IV²/365 − r².
     Reported as a vol-targeted NAV so the TAIL (the steamroller) is visible, not hidden.
  3. TIMED: only sell vol when IV is rich vs trailing realized (avoid selling cheap vol).

The honest question for short-vol isn't "is the average positive" (it usually is) — it's whether
the tail-adjusted, out-of-sample Sharpe justifies the crash risk. Run: python crypto/backtest/vrp_pilot.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")


def load():
    iv = pd.read_parquet(os.path.join(DATA, "deribit_dvol.parquet"))
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))[["BTC", "ETH"]]
    idx = iv.index.intersection(close.index)
    return iv.loc[idx], close.loc[idx]


def navstats(x, lo=None):
    x = x.loc[lo:].dropna() if lo else x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0, 0
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min(), x.min())


if __name__ == "__main__":
    iv, close = load()
    ret = close.pct_change(fill_method=None)
    print("=" * 92)
    print("PILOT A — VOL-RISK-PREMIUM (Deribit DVOL vs realized) | %s→%s" % (iv.index.min().date(), iv.index.max().date()))
    print("=" * 92)

    print("\n[1] PREMIUM — is implied vol > realized vol? (the raw edge)")
    print(f"    {'asset':<8}{'mean IV':>9}{'mean RV':>9}{'IV-RV':>8}{'% IV>RV':>9}{'2023+ IV-RV':>13}")
    for a in ["BTC", "ETH"]:
        rv = ret[a].rolling(30).std() * np.sqrt(365)
        rv_fwd = rv.shift(-30)                                   # vol realized over the NEXT 30d
        prem = (iv[a] - rv_fwd).dropna()
        prem23 = prem.loc["2023":]
        pct = (iv[a] > rv_fwd).mean() * 100
        print(f"    {a:<8}{iv[a].mean()*100:>8.1f}%{rv_fwd.mean()*100:>8.1f}%{prem.mean()*100:>7.1f}%{pct:>8.0f}%{prem23.mean()*100:>12.1f}%", flush=True)

    print("\n[2] SHORT-VARIANCE CARRY — vol-targeted NAV (tail visible). daily P&L = IV²/365 − r²")
    print(f"    {'book':<26}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'Sharpe':>8}{'2023+ Sh':>10}{'MaxDD':>8}{'worstDay':>9}")
    for a in ["BTC", "ETH"]:
        carry = (iv[a].shift(1) ** 2) / 365 - ret[a] ** 2        # short variance, no look-ahead
        # vol-target the carry to 10% annual so NAV/tail are comparable
        sc = (0.10 / (carry.rolling(30).std() * np.sqrt(365)).shift(1)).clip(0, 5).fillna(0)
        nav_r = sc * carry
        cg, vol, sh, md, wd = navstats(nav_r); cg23, _, sh23, _, _ = navstats(nav_r, "2023")
        print(f"    {a+' short-vol (always)':<26}{cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>8.2f}{sh23:>10.2f}{md*100:>7.1f}%{wd*100:>8.1f}%", flush=True)
        # TIMED: only sell when IV rich vs trailing 20d realized
        rv_tr = (ret[a].rolling(20).std() * np.sqrt(365)).shift(1)
        rich = (iv[a].shift(1) > rv_tr * 1.0)
        carry_t = carry.where(rich, 0.0)
        sc2 = (0.10 / (carry_t.rolling(30).std() * np.sqrt(365)).shift(1)).clip(0, 5).fillna(0)
        navt = sc2 * carry_t
        cg, vol, sh, md, wd = navstats(navt); cg23, _, sh23, _, _ = navstats(navt, "2023")
        print(f"    {a+' short-vol (timed)':<26}{cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>8.2f}{sh23:>10.2f}{md*100:>7.1f}%{wd*100:>8.1f}%", flush=True)

    print("\n  Read: positive IV−RV confirms the premium exists. The verdict is the 2023+ Sharpe AND the")
    print("  worst-day/MaxDD — short-vol that needs to survive a -X% day is only real if the tail is bearable.")
