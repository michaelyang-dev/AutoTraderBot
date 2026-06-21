"""
Shared library for the exposure-overlay threads (#3 market de-risk, #4 ML crash
trigger, #5 meta book-fragility). All answer: does signal X time de-risking BETTER
than the incumbent SPY<SMA200 + vol-scaling?

Honest method: apply each de-risk signal as a daily exposure e_t in [floor,1] on
the SAME base (deployed book daily returns, exposure overlays OFF). Compare at
MATCHED AVERAGE EXPOSURE (tune threshold to a target avg) so we measure TIMING,
not just cash drag. Overlay is at NAV-return level: r_overlaid = e_t * r_t (cash
~0). Approximation suitable for ranking signal quality; winner re-checked in the
full backtest.
"""
import numpy as np
import pandas as pd


def metrics(r, idx):
    s = pd.Series(r, index=idx).dropna()
    eq = (1 + s).cumprod()
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    sd = s[s < 0].std()
    sortino = s.mean() / sd * np.sqrt(252) if sd and sd > 0 else 0
    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino, mdd=mdd,
                vol=s.std() * np.sqrt(252))


def exposure_from_signal(sig, base_idx, floor=0.40, target_avg=None,
                         high_is_risk=True, expanding_min=252):
    """Map a raw risk signal -> daily exposure in [floor,1].
    Risk fires when sig is in the upper (high_is_risk) tail vs its EXPANDING
    history (no look-ahead). If target_avg given, binary-search the percentile
    cutoff so mean exposure ~= target_avg (matched-exposure comparison)."""
    s = sig.reindex(base_idx).ffill()
    if not high_is_risk:
        s = -s
    # expanding percentile rank (no look-ahead)
    pr = s.expanding(min_periods=expanding_min).apply(
        lambda x: (x[-1] >= x).mean(), raw=True)
    pr = pr.fillna(0.5)

    def expo(cut):
        return np.where(pr > cut, floor, 1.0)

    if target_avg is None:
        e = expo(0.80)
    else:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            avg = expo(mid).mean()
            if avg < target_avg:   # too much de-risk -> raise cutoff
                lo = mid
            else:
                hi = mid
        e = expo((lo + hi) / 2)
    return pd.Series(e, index=base_idx)


def spy_trend_exposure(prices, base_idx, bear=0.40, caution=0.75):
    """Incumbent: SPY<SMA200 -> bear, SPY<SMA50 -> caution, else 1.0."""
    spy = prices["SPY"].reindex(base_idx).ffill()
    sma200 = spy.rolling(200).mean()
    sma50 = spy.rolling(50).mean()
    e = pd.Series(1.0, index=base_idx)
    e[spy < sma50] = caution
    e[spy < sma200] = bear
    return e


def vol_scale_exposure(base_r, base_idx, target=0.20, lookback=40, floor=0.3, cap=1.5):
    """Incumbent vol-scaling on the book's own realized vol (capped at 1.0 here
    since we study DE-RISK only)."""
    rv = pd.Series(base_r, index=base_idx).rolling(lookback).std() * np.sqrt(252)
    e = (target / rv).clip(floor, cap).clip(upper=1.0)
    return e.fillna(1.0)


def apply_overlay(base_r, base_idx, exposure):
    e = exposure.reindex(base_idx).ffill().fillna(1.0).shift(1).fillna(1.0)  # act next day
    return (np.asarray(base_r) * e.values), e.mean()
