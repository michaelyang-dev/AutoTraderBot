"""
Strategy 3: Sector Momentum Rotation (15% of capital)
=====================================================
Rotate among sector ETFs based on 6-month relative strength.
Hold top 3 sectors equal-weighted.

Regime overlay: if SPY below 200-day SMA, exit all sectors and hold cash
(simulate T-bill return of ~0% daily since we don't hold BIL in features).

Monthly rebalance on first trading day of each month.
"""

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC",
]

LOOKBACK = 126  # ~6 months


class SectorRotation(BaseStrategy):

    @property
    def name(self):
        return "sector_rotation"

    @property
    def capital_pct(self):
        return 0.15

    @property
    def rebalance_days(self):
        return 21  # ~monthly

    def should_rebalance(self, day_index):
        return day_index % self.rebalance_days == 0

    def compute_targets(self, date, universe):
        # --- Regime check: SPY above 200-day SMA? ---
        spy_above = universe.regime.get("spy_above_sma200", True)
        if not spy_above:
            # Bear market: hold cash (return empty targets = 100% cash)
            return {}

        # --- Compute 6-month returns for each sector ETF ---
        sector_rets = {}
        for etf in SECTOR_ETFS:
            close = universe.get_close(etf, lookback=LOOKBACK + 5)
            if len(close) < LOOKBACK:
                continue
            ret_6m = (close.iloc[-1] / close.iloc[-LOOKBACK]) - 1.0
            if not np.isnan(ret_6m):
                sector_rets[etf] = ret_6m

        if len(sector_rets) < 3:
            return {}

        # --- Top 3 by 6-month return ---
        sorted_etfs = sorted(sector_rets.keys(),
                              key=lambda s: sector_rets[s], reverse=True)
        top3 = sorted_etfs[:3]

        weight = 1.0 / 3.0
        return {etf: weight for etf in top3}
