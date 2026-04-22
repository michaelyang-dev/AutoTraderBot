"""
Strategy Base Classes
=====================
Contains the Strategy ABC and Signal/Position dataclasses used by all
strategy implementations and the PortfolioManager.

Extracted to avoid circular imports between unified_backtester.py
and individual strategy modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Signal:
    """A buy signal produced by a strategy."""
    symbol: str
    confidence: float          # 0.0–1.0
    strategy_name: str
    fwd_ret: float = 0.0      # backtesting only — actual forward return
    price_based: bool = False  # True for strategies that use actual prices (not fwd_ret)


@dataclass
class Position:
    """An open position managed by the portfolio."""
    symbol: str
    strategy_name: str
    cost: float
    entry_idx: int
    exit_idx: int
    fwd_ret: float
    confidence: float
    price_based: bool = False  # True for price-based strategies (momentum, etc.)
    entry_price: float = 0.0   # actual entry price (price-based only)
    peak_price: float = 0.0    # highest price since entry (for trailing stop)


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Every strategy must implement four members:

        name                        → unique string identifier
        generate_signals(date, ud)  → list[Signal] of BUY candidates
        check_exit(pos, cur)        → (should_exit, reason)
        get_position_size(sig, pv)  → target $ amount
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_signals(self, date, universe_data):
        """Return a list of Signal objects for BUY candidates, sorted by
        confidence descending.  *universe_data* is strategy-specific
        context (may be None)."""
        ...

    @abstractmethod
    def check_exit(self, position, current_data):
        """Return *(should_exit, reason)*.  *current_data* is a dict with
        keys ``idx``, ``date``, ``n_dates``."""
        ...

    @abstractmethod
    def get_position_size(self, signal, portfolio_value, date=None):
        """Return the target dollar amount to invest.  The PortfolioManager
        will cap it at ``available_cash * 0.95``."""
        ...
