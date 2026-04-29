"""
BaseStrategy — Common interface for all v8 strategies.

Each strategy implements:
  - name: human-readable name
  - capital_pct: fraction of total portfolio allocated
  - rebalance_days: how often to rebalance (trading days)
  - should_rebalance(date, day_index): whether to rebalance on this date
  - compute_targets(date, universe_data): returns {symbol: weight} target portfolio
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Common interface for all portfolio strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def capital_pct(self) -> float:
        """Fraction of total portfolio capital allocated to this strategy."""
        pass

    @property
    @abstractmethod
    def rebalance_days(self) -> int:
        """Rebalance every N trading days."""
        pass

    def should_rebalance(self, day_index: int) -> bool:
        """Whether to rebalance on this day index."""
        return day_index % self.rebalance_days == 0

    @abstractmethod
    def compute_targets(self, date: pd.Timestamp,
                         universe: "UniverseData") -> dict:
        """
        Compute target portfolio weights for this strategy.

        Args:
            date: current trading date
            universe: UniverseData object with prices, features, fundamentals

        Returns:
            {symbol: weight} where weights sum to <= 1.0
            (relative to this strategy's capital allocation)
        """
        pass


class UniverseData:
    """
    Container for all data needed by strategies on a given date.
    Pre-computed once per date, shared across all strategies.
    """

    def __init__(self, date, prices_df, features_df, fundamentals,
                 sp500_members, sector_map, regime):
        self.date = date
        self.prices = prices_df       # DataFrame: index=dates, columns=symbols, values=close
        self.features = features_df   # DataFrame: full features.parquet
        self.fundamentals = fundamentals  # dict of DataFrames: income, ratios, etc.
        self.sp500_members = sp500_members  # set of symbols in SP500 on this date
        self.sector_map = sector_map  # {symbol: sector}
        self.regime = regime          # dict: {vix, vix_term_structure, spy_above_sma200, drawdown_pct}

    def get_close(self, symbol, lookback=None):
        """Get close price series for a symbol, optionally trimmed to lookback days."""
        if symbol not in self.prices.columns:
            return pd.Series(dtype=float)
        s = self.prices[symbol].loc[:self.date].dropna()
        if lookback:
            s = s.iloc[-lookback:]
        return s

    def get_feature(self, symbol, feature_name):
        """Get a single feature value for a symbol on current date."""
        mask = (self.features["date"] == self.date) & (self.features["symbol"] == symbol)
        rows = self.features.loc[mask, feature_name]
        return rows.iloc[0] if len(rows) > 0 else None

    def get_sp500_features(self, feature_name):
        """Get a feature for all SP500 members on current date as {symbol: value}."""
        mask = (self.features["date"] == self.date) & (
            self.features["symbol"].isin(self.sp500_members))
        sub = self.features.loc[mask, ["symbol", feature_name]].dropna()
        return dict(zip(sub["symbol"], sub[feature_name]))
