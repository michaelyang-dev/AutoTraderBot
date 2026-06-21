"""
Wick-capture HARDENED — is the deep-liquidation-bid edge real, or a hidden-tail knife-catch?

Fixes the inflations in wick_capture.py:
  1. DAILY aggregation. Annualizing a mostly-zero hourly series with √(24·365) fakes a huge Sharpe.
     Aggregate fills to a daily P&L on deployed capital, annualize honestly with √365.
  2. POSITION CAP. Can't put 100% on one wick; cap per-coin weight, rest in cash. In a market-wide
     cascade many fill together → correlated, capped exposure (the real risk).
  3. REALISTIC EXIT. Fill at the limit but exit at close minus slippage; deep cascades cost more.
  4. TAIL TRUTH. Report MaxDD, worst day, and the 8 worst days WITH DATES — do they line up with
     LUNA (May 2022) / FTX (Nov 2022) / Aug-2024 unwind? A knife-catch blows up exactly there.

Run:  python crypto/backtest/wick_hardened.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")


def load():
    return (pd.read_parquet(os.path.join(DATA, "binance_1h_l.parquet")).sort_index(),
            pd.read_parquet(os.path.join(DATA, "binance_1h_c.parquet")).sort_index(),
            pd.read_parquet(os.path.join(DATA, "binance_1h_qv.parquet")).sort_index())


def run(x=0.05, exit_slip=0.0015, fee=0.0004, per_cap=0.20, exit_next=False):
    low, c, qv = load()
    prev_c = c.shift(1)
    bid = prev_c * (1 - x)
    liquid = qv.rolling(24).mean() > 3e6
    fill = (low <= bid) & liquid & prev_c.notna()
    exitpx = c.shift(-1) if exit_next else c
    pnl = (exitpx / bid - 1) - 2 * fee - exit_slip              # fill at bid, exit at close − slippage
    pnl = pnl.where(fill)
    # capital allocation: each filled coin gets min(per_cap, 1/n_fills); rest idle (cash, 0 return)
    nfill = fill.sum(axis=1)
    wgt = np.minimum(per_cap, 1.0 / nfill.replace(0, np.nan))
    hourly = (pnl.mul(wgt, axis=0)).sum(axis=1).fillna(0.0)     # capital-weighted hourly P&L
    daily = hourly.resample("1D").sum()                        # HONEST: aggregate to daily
    return daily


def stats(d, lo=None):
    x = d.loc[lo:].dropna() if lo else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.mean() / x.std() * np.sqrt(365),
            ((nav - nav.cummax()) / nav.cummax()).min())


if __name__ == "__main__":
    print("=" * 90)
    print("WICK-CAPTURE HARDENED — daily P&L, capped sizing, realistic exit, TAIL examined")
    print("=" * 90)
    print(f"  {'config (exit-at-close)':<34}{'CAGR':>9}{'Sharpe':>8}{'2023+ Sh':>10}{'2024+ Sh':>10}{'MaxDD':>8}")
    for x in [0.03, 0.05, 0.08]:
        d = run(x=x)
        cg, sh, md = stats(d); _, sh23, _ = stats(d, "2023"); _, sh24, _ = stats(d, "2024")
        print(f"    -{x*100:.0f}% bid, cap20%, slip15bps     {cg*100:>8.0f}%{sh:>8.2f}{sh23:>10.2f}{sh24:>10.2f}{md*100:>7.1f}%", flush=True)

    print("\n  TAIL TRUTH — the -5% book's worst days (do they line up with the big cascades?):")
    d = run(x=0.05)
    worst = d.nsmallest(8)
    for dt, v in worst.items():
        print("    %s   %+.1f%%" % (dt.date(), v * 100), flush=True)
    cg, sh, md = stats(d)
    print("\n    per-year Sharpe (-5% book):")
    for y in range(2020, 2027):
        ys = d[d.index.year == y]
        _, s, _ = stats(ys) if len(ys) > 30 else (0, 0, 0)
        print("      %d: Sharpe %5.2f  ret %+6.0f%%" % (y, s, ((1 + ys).prod() - 1) * 100), flush=True)
    print("\n  Read: if it survives LUNA/FTX with a bearable MaxDD AND holds OOS Sharpe after honest")
    print("  daily annualization, it's a real (if capacity-limited) liquidity edge. If the tail is")
    print("  catastrophic or the daily Sharpe collapses, it was a knife-catch dressed as alpha.")
