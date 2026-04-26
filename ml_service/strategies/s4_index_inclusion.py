"""
Strategy 4: Index Inclusion Arbitrage (10% of capital)
=====================================================
Buy stocks added to S&P 500 on announcement, hold 20 trading days.
Max 5 concurrent positions, 20% of strategy capital each.

Uses sp500_changes.json from FMP to identify additions.
"""

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class IndexInclusionArbitrage(BaseStrategy):

    MAX_POSITIONS = 5
    HOLD_DAYS = 20

    @property
    def name(self):
        return "index_inclusion"

    @property
    def capital_pct(self):
        return 0.10

    @property
    def rebalance_days(self):
        return 1  # check daily

    def __init__(self, sp500_changes):
        """
        Args:
            sp500_changes: list of dicts from sp500_changes.json
        """
        # Parse additions into a lookup: {date: [symbols added]}
        self._additions_by_date = {}
        for change in sp500_changes:
            sym = change.get("symbol", "")
            date_str = change.get("date") or change.get("dateAdded", "")
            if not sym or not date_str:
                continue
            try:
                d = pd.Timestamp(date_str).normalize()
                if d not in self._additions_by_date:
                    self._additions_by_date[d] = []
                self._additions_by_date[d].append(sym)
            except (ValueError, TypeError):
                continue

        # Active trades: {symbol: entry_date}
        self._active_trades = {}

    def compute_targets(self, date, universe):
        # --- Expire old trades ---
        expired = [sym for sym, entry in self._active_trades.items()
                    if (date - entry).days >= self.HOLD_DAYS * 1.5]  # ~20 trading days
        for sym in expired:
            del self._active_trades[sym]

        # --- Check for new additions ---
        # Look for additions in the last 3 calendar days (account for weekends)
        for lookback_days in range(4):
            check_date = date - pd.Timedelta(days=lookback_days)
            new_additions = self._additions_by_date.get(check_date, [])
            for sym in new_additions:
                if sym in self._active_trades:
                    continue
                if len(self._active_trades) >= self.MAX_POSITIONS:
                    break
                # Verify the symbol has price data
                close = universe.get_close(sym, lookback=5)
                if len(close) > 0:
                    self._active_trades[sym] = date

        if not self._active_trades:
            return {}

        # --- Equal weight among active trades ---
        n = len(self._active_trades)
        weight = 1.0 / self.MAX_POSITIONS  # fixed 20% per slot
        return {sym: weight for sym in self._active_trades}
