"""
Strategy 2: Drift-Conditional Value-Reversal (20% of capital)
=============================================================
Applies value and reversal signals ONLY to stocks in persistent uptrends.

Drift filter: stock must have >60% positive-return days over trailing 63 days.
Signals (within qualifiers only):
  1. Value proxy: inverse of price level (1/price), cross-sectionally ranked
  2. Short-term reversal: negative of past 10-day return

Top 30 by combined signal, equal-weight, 15-day hold.
Self-regulating: fewer stocks qualify in bear markets -> natural exposure reduction.
"""

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class DriftValueReversal(BaseStrategy):

    DRIFT_THRESHOLD = 0.60     # 60% positive days
    DRIFT_LOOKBACK = 63        # trading days
    TOP_N = 30
    MIN_QUALIFIERS = 5

    @property
    def name(self):
        return "drift_value_reversal"

    @property
    def capital_pct(self):
        return 0.20

    @property
    def rebalance_days(self):
        return 15

    def compute_targets(self, date, universe):
        members = universe.sp500_members
        if len(members) < 50:
            return {}

        # --- Drift filter: >60% positive return days over 63 trading days ---
        qualifiers = []
        for sym in members:
            close = universe.get_close(sym, lookback=self.DRIFT_LOOKBACK + 5)
            if len(close) < self.DRIFT_LOOKBACK:
                continue
            recent = close.iloc[-self.DRIFT_LOOKBACK:]
            daily_rets = recent.pct_change().dropna()
            if len(daily_rets) < 30:
                continue
            pct_positive = (daily_rets > 0).mean()
            if pct_positive >= self.DRIFT_THRESHOLD:
                qualifiers.append(sym)

        if len(qualifiers) < self.MIN_QUALIFIERS:
            return {}

        # --- Signal 1: Value proxy (inverse of price, cross-sectionally ranked) ---
        prices_today = {}
        for sym in qualifiers:
            close = universe.get_close(sym, lookback=1)
            if len(close) > 0:
                px = close.iloc[-1]
                if px > 0 and not np.isnan(px):
                    prices_today[sym] = px

        if len(prices_today) < self.MIN_QUALIFIERS:
            return {}

        # Z-score of 1/price
        inv_prices = {sym: 1.0 / px for sym, px in prices_today.items()}
        vals = np.array(list(inv_prices.values()))
        mu, sigma = vals.mean(), vals.std()
        z_value = {}
        if sigma > 1e-10:
            z_value = {sym: (v - mu) / sigma for sym, v in inv_prices.items()}

        # --- Signal 2: Short-term reversal (negative of 10d return) ---
        ret_10d = {}
        for sym in qualifiers:
            r = universe.get_feature(sym, "ret_10d")
            if r is not None and not np.isnan(r):
                ret_10d[sym] = -r  # negative = reversal signal

        vals2 = np.array(list(ret_10d.values())) if ret_10d else np.array([])
        z_reversal = {}
        if len(vals2) > 5:
            mu2, sigma2 = vals2.mean(), vals2.std()
            if sigma2 > 1e-10:
                z_reversal = {sym: (v - mu2) / sigma2 for sym, v in ret_10d.items()}

        # --- Composite: equal-weight average of available signals ---
        composite = {}
        for sym in qualifiers:
            zs = []
            if sym in z_value:
                zs.append(z_value[sym])
            if sym in z_reversal:
                zs.append(z_reversal[sym])
            if zs:
                composite[sym] = np.mean(zs)

        if not composite:
            return {}

        # --- Select top N ---
        sorted_syms = sorted(composite.keys(), key=lambda s: composite[s], reverse=True)
        selected = sorted_syms[:self.TOP_N]

        weight = 1.0 / len(selected)
        return {sym: weight for sym in selected}
