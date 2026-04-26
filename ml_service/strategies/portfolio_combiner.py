"""
Portfolio Combiner
==================
Takes target positions from all 5 strategies and produces a single
combined portfolio with position-level and portfolio-level constraints.

Constraints:
  - Max 10% in any single name
  - Max 30% in any single GICS sector
  - Max 100% gross exposure
  - Min 1% per position (drop smaller)

Circuit breaker:
  - Drawdown > 15%: reduce all allocations by 50% for 30 days
  - Drawdown > 25%: halt new entries until equity recovers to 95% of peak

Master regime overlay:
  - VIX > 40: pause strategies 1, 2, 4 (keep 3 and 5)
"""

import numpy as np
import pandas as pd

MAX_SINGLE_NAME = 0.10
MAX_SECTOR = 0.30
MAX_GROSS = 1.00
MIN_POSITION = 0.01
DD_REDUCE_THRESHOLD = -0.15
DD_HALT_THRESHOLD = -0.25
DD_RECOVERY_PCT = 0.95
VIX_EXTREME = 40


class PortfolioCombiner:

    def __init__(self, strategies, sector_map):
        """
        Args:
            strategies: list of BaseStrategy instances
            sector_map: {symbol: sector_name}
        """
        self.strategies = strategies
        self.sector_map = sector_map
        self.peak_equity = 0
        self.current_equity = 0
        self._dd_reduce_until = None  # date when reduction ends
        self._halted = False

    def update_equity(self, equity, date):
        """Update equity tracking for circuit breaker."""
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity <= 0:
            return

        dd = (equity - self.peak_equity) / self.peak_equity

        # Circuit breaker: halt
        if dd < DD_HALT_THRESHOLD:
            self._halted = True
        elif self._halted and equity >= self.peak_equity * DD_RECOVERY_PCT:
            self._halted = False

        # Circuit breaker: reduce for 30 days
        if dd < DD_REDUCE_THRESHOLD and self._dd_reduce_until is None:
            self._dd_reduce_until = date + pd.Timedelta(days=45)  # ~30 trading days

        if self._dd_reduce_until is not None and date > self._dd_reduce_until:
            self._dd_reduce_until = None

    def combine(self, strategy_targets, date, regime):
        """
        Combine targets from all strategies into a single portfolio.

        Args:
            strategy_targets: {strategy_name: {symbol: weight}}
            date: current date
            regime: dict with vix, vix_term_structure, etc.

        Returns:
            {symbol: weight} final portfolio, weights sum to <= 1.0
        """
        vix = regime.get("vix", 20)

        # --- Master regime overlay: VIX > 40 pauses strategies 1, 2, 4 ---
        paused = set()
        if vix > VIX_EXTREME:
            paused = {"momentum_reversal_ensemble", "drift_value_reversal",
                       "index_inclusion"}

        # --- Combine: scale each strategy's targets by its capital_pct ---
        combined = {}
        for strat in self.strategies:
            if strat.name in paused:
                continue
            targets = strategy_targets.get(strat.name, {})
            for sym, w in targets.items():
                scaled_w = w * strat.capital_pct
                combined[sym] = combined.get(sym, 0) + scaled_w

        if not combined:
            return {}

        # --- Circuit breaker ---
        if self._halted:
            return {}  # halt all entries

        dd_scale = 0.5 if self._dd_reduce_until is not None else 1.0

        # --- Apply position constraints ---
        for _ in range(10):
            changed = False

            # Single-name cap
            for sym in list(combined.keys()):
                if combined[sym] * dd_scale > MAX_SINGLE_NAME:
                    combined[sym] = MAX_SINGLE_NAME / dd_scale
                    changed = True

            # Sector cap
            sector_totals = {}
            for sym, w in combined.items():
                sec = self.sector_map.get(sym, "Unknown")
                sector_totals[sec] = sector_totals.get(sec, 0) + w * dd_scale

            for sec, total in sector_totals.items():
                if total > MAX_SECTOR:
                    scale = MAX_SECTOR / total
                    for sym in list(combined.keys()):
                        if self.sector_map.get(sym, "Unknown") == sec:
                            combined[sym] *= scale
                    changed = True

            # Gross exposure cap
            gross = sum(combined.values()) * dd_scale
            if gross > MAX_GROSS:
                scale = MAX_GROSS / gross
                for sym in combined:
                    combined[sym] *= scale
                changed = True

            if not changed:
                break

        # Apply drawdown scale
        if dd_scale < 1.0:
            combined = {sym: w * dd_scale for sym, w in combined.items()}

        # Drop tiny positions
        combined = {sym: w for sym, w in combined.items() if w >= MIN_POSITION}

        # Renormalize if over 100%
        gross = sum(combined.values())
        if gross > MAX_GROSS:
            for sym in combined:
                combined[sym] /= gross

        return combined
