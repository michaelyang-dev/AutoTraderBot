"""
Crypto cash-and-carry backtest — the honest version.

Models the delta-neutral trade (long spot + short CME front-month future) from the
ACTUAL daily basis path, not the noisy annualized figure:

  daily carry P&L = -Δ(basis_ratio)        # basis narrows → you earn; widens → mark-to-market loss
                                            # excluded on roll days (that jump is a roll, not P&L)
  net = carry - financing - roll fees
  levered:  L·carry - (L-1)·margin_rate - L·roll_fees

Risk comes from the real basis fluctuations (so Sharpe is honest, not infinite).
Financing + fees baked in. Benchmark is cash (risk-free), since this is market-neutral.

Run:  python crypto/backtest/basis_backtest.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

BASIS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "data", "crypto", "cme_basis.parquet")

MARGIN_RATE = 0.055     # IBKR margin (cost of borrowed capital when levered)
RISK_FREE = 0.045       # cash benchmark (what your equity would earn idle)
ROLL_FEE = 0.0010       # ~10 bps round-trip to roll both legs (monthly)
DAY = 1 / 365.0


MAX_BASIS = 0.05      # real CME basis never exceeds ~5% raw; beyond = bad tick (fut/spot misalign)
MAX_DAILY = 0.015     # real delta-neutral daily move; beyond = glitch


def clean_basis(df):
    """Winsorize the raw basis to a sane band (kills misaligned fut/spot ticks)."""
    return df["basis"].clip(-MAX_BASIS, MAX_BASIS)


def carry_pnl(df):
    """Daily delta-neutral carry return from the cleaned basis path (roll jumps excluded)."""
    dbasis = clean_basis(df).diff()
    pnl = (-dbasis).clip(-MAX_DAILY, MAX_DAILY)   # basis narrows → positive; clip residual glitches
    pnl[df["roll"].values] = 0.0                  # the roll-day jump is not P&L
    return pnl.fillna(0.0)


def basis_ann_robust(df):
    """Annualized basis with dte floored at 5d (avoids the near-expiry blow-up) for timing only."""
    return clean_basis(df) * (365.0 / df["dte"].clip(lower=5))


def run(df, asset, leverage=1.0, timed=False, thresh=0.0):
    carry = carry_pnl(df)
    # 'timed': only hold the carry when the annualized basis clears a threshold (e.g., financing)
    on = pd.Series(True, index=df.index)
    if timed:
        on = (basis_ann_robust(df) > thresh).shift(1).fillna(False).astype(bool)   # prior close, no look-ahead
    daily = leverage * carry * on
    daily = daily - (leverage - 1) * MARGIN_RATE * DAY * on   # margin only while in the trade
    daily = daily - df["roll"] * ROLL_FEE * leverage * on     # roll cost (scaled by leverage)
    nav = (1 + daily).cumprod()
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    cg = nav.iloc[-1] ** (1 / yrs) - 1
    vol = daily.std() * np.sqrt(365)
    sh = daily.mean() / daily.std() * np.sqrt(365) if daily.std() else 0
    md = ((nav - nav.cummax()) / nav.cummax()).min()
    excess = cg - RISK_FREE
    return dict(cagr=cg, vol=vol, sharpe=sh, mdd=md, excess=excess, nav=nav)


panel = pd.read_parquet(BASIS)
print("=" * 92)
print("CRYPTO CASH-AND-CARRY (CME basis) — financing + fees baked in | benchmark: cash @ %.1f%%" % (RISK_FREE*100))
print("=" * 92)

for asset in ["BTC", "ETH"]:
    df = panel[asset].dropna(subset=["basis", "dte"]).copy()
    print(f"\n### {asset}  ({df.index.min().date()} → {df.index.max().date()}) ###")
    print(f"  {'variant':<28}{'CAGR':>8}{'excess':>9}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")
    base = run(df, asset, 1.0)
    print(f"  {'always-on  1x':<28}{base['cagr']*100:>7.1f}%{base['excess']*100:>8.1f}%{base['vol']*100:>6.1f}%{base['sharpe']:>8.2f}{base['mdd']*100:>7.1f}%", flush=True)
    timed = run(df, asset, 1.0, timed=True, thresh=MARGIN_RATE)
    print(f"  {'timed(basis>margin) 1x':<28}{timed['cagr']*100:>7.1f}%{timed['excess']*100:>8.1f}%{timed['vol']*100:>6.1f}%{timed['sharpe']:>8.2f}{timed['mdd']*100:>7.1f}%", flush=True)
    print(f"  -- leverage frontier (timed) --")
    for L in [2, 3, 5]:
        r = run(df, asset, L, timed=True, thresh=MARGIN_RATE)
        print(f"  {'timed  '+str(L)+'x':<28}{r['cagr']*100:>7.1f}%{r['excess']*100:>8.1f}%{r['vol']*100:>6.1f}%{r['sharpe']:>8.2f}{r['mdd']*100:>7.1f}%", flush=True)
    # per-year gross carry (regime dependence)
    cyr = carry_pnl(df).groupby(df.index.year).sum() * 100
    print("  gross carry by year: " + "  ".join(f"{y}:{v:+.0f}%" for y, v in cyr.items()))

print("\nRead: 'excess' = CAGR over cash. If excess ~0 the trade isn't beating T-bills for its risk.")
print("Leverage only helps when basis > margin; it amplifies a thin/negative edge otherwise.")
