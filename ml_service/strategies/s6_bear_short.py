"""
Strategy 6: Bear Market Short (activated only in bear markets)
==============================================================
When SPY is below 200-day SMA, short the weakest stocks.

Signal: composite of worst momentum + worst quality + worst sector
Position: short top-10 weakest names, equal-weight
Only active when bear=True. Returns empty portfolio in bull.

This is designed to OFFSET losses from long strategies during bear markets.
"""

import numpy as np
import pandas as pd


def strategy6_bear_short(date, uni, day_idx, top_n=10, rebal_days=10):
    """Bear-market short strategy. Short worst 10 stocks when bear."""
    if day_idx % rebal_days != 0:
        return None

    regime = uni.get_regime(date)
    spy_bull = regime.get("spy_above_sma200", True)

    # Only active in bear markets
    if spy_bull:
        return {}

    members = uni.get_sp500(date)
    if len(members) < 50:
        return {}

    # Signals (INVERTED — we want the worst stocks)
    ret_252 = uni.get_feature_map(date, "ret_252d", members)
    ret_20 = uni.get_feature_map(date, "ret_20d", members)
    gross_m = uni.get_feature_map(date, "gross_margin", members)
    vol_60 = uni.get_feature_map(date, "vol_60d", members)

    # Worst momentum (lowest 12-1 = most likely to keep falling)
    mom = {s: -(ret_252.get(s, 0) - ret_20.get(s, 0)) for s in members
           if s in ret_252 and s in ret_20}

    # Worst quality (lowest margins)
    bad_qual = {s: -gross_m.get(s, 0) for s in members if s in gross_m}

    # Highest volatility (most likely to crash further)
    high_vol = {s: vol_60.get(s, 0) for s in members if s in vol_60 and vol_60[s] > 0}

    def zscore(d):
        if len(d) < 20:
            return {}
        vals = np.array(list(d.values()))
        mu, sig = vals.mean(), vals.std()
        return {s: (v - mu) / sig for s, v in d.items()} if sig > 1e-10 else {}

    z_mom = zscore(mom)
    z_qual = zscore(bad_qual)
    z_vol = zscore(high_vol)

    composite = {}
    for sym in members:
        zs = [z for z in [z_mom.get(sym), z_qual.get(sym), z_vol.get(sym)] if z is not None]
        if len(zs) >= 2:
            composite[sym] = np.mean(zs)

    if not composite:
        return {}

    # Top N by "worst" composite = stocks to short
    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:top_n]
    w = 1.0 / len(sorted_syms)
    # Return negative weights to indicate short positions
    return {s: -w for s in sorted_syms}
