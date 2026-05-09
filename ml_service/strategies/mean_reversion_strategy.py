"""
Mean Reversion Strategy (Strategy 3)
=====================================
Buys stocks that have dropped significantly from recent highs, betting on
a reversion to the mean. Uses price-based entries/exits with actual prices.

Signal generation:
  - BUY when a stock has dropped > 15% over the past 30 trading days
  - Must be above 200-day SMA (not in structural downtrend)
  - Must NOT be at 30-day low (want early bounce, not falling knife)

Confidence scoring:
  - 0.50 at -15% drop, 0.95 at -30% drop (linear interpolation)

Filters:
  - NEVER_BUY blacklist (leveraged/inverse/volatility products)
  - 20-day avg volume > 500K shares
  - Symbol not already held

Exit rules (checked in order):
  - Take-profit: price recovers to 20-day SMA
  - Stop-loss: price drops 10% from entry
  - Max hold: 10 trading days

Position sizing:
  - Base: 12% of portfolio
  - Reduced to 8% if ATR(14)/price > 4%
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path so we can import from strategy_base
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_base import Strategy, Signal


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy: buy oversold stocks expecting a bounce.
    """

    NEVER_BUY = {
        "VIXY", "UVXY", "VXX", "SVXY",
        "TQQQ", "SQQQ", "QQQ3",
        "SPXU", "SPXS", "SDS", "UPRO",
        "QID", "SDOW",
        "LABU", "LABD", "JNUG", "JDST", "NUGT", "DUST",
        "FNGU", "FNGD", "SOXL", "SOXS", "YANG", "YINN",
    }

    # Tunable parameters
    DROP_PERIOD     = 30      # trading days to measure drop
    DROP_THRESHOLD  = -0.15   # minimum drop to trigger signal
    DROP_MAX        = -0.30   # drop level for maximum confidence
    SMA_PERIOD      = 200     # trend filter
    SMA_SHORT       = 20      # take-profit target (20-day SMA)
    VOL_PERIOD      = 20      # avg volume window
    VOL_MIN         = 500_000
    ATR_PERIOD      = 14
    STOP_LOSS       = -0.10   # -10% from entry
    MAX_HOLD        = 10      # trading days
    BASE_PCT        = 0.12
    HIGH_ATR_PCT    = 0.08
    HIGH_ATR_THRESH = 0.04    # ATR/price threshold (higher than momentum's 3%)

    def __init__(self, price_data, volume_data=None):
        """
        Args
        ----
        price_data : DataFrame — close prices, columns=symbols, index=dates
        volume_data : DataFrame — daily volume, same shape (optional)
        """
        self._price_data = price_data
        self._volume_data = volume_data
        self._symbols = [s for s in price_data.columns
                         if s not in self.NEVER_BUY and s != "SPY"]

        # Pre-computed indicators keyed by date
        self._drop_30d = {}     # {date → {sym → 30d return}}
        self._low_30d = {}      # {date → {sym → 30d low price}}
        self._sma200 = {}       # {date → {sym → sma200}}
        self._sma20 = {}        # {date → {sym → sma20}}
        self._atr_pct = {}      # {date → {sym → atr/price}}
        self._avg_vol = {}      # {date → {sym → 20d avg volume}}
        self._precompute()

    @property
    def name(self):
        return "mean_reversion"

    def _precompute(self):
        """Pre-compute all indicators needed for signal generation and exits.

        Uses vectorized pandas rolling operations for all indicators,
        then fast row-wise .dropna().to_dict() for materialization."""
        px = self._price_data
        sym_cols = [s for s in self._symbols if s in px.columns]
        px_sub = px[sym_cols]

        # Vectorized rolling computations (whole-DataFrame, no per-symbol loop)
        ret30 = px_sub.pct_change(self.DROP_PERIOD)
        low30 = px_sub.rolling(self.DROP_PERIOD, min_periods=self.DROP_PERIOD).min()
        sma200 = px_sub.rolling(self.SMA_PERIOD, min_periods=self.SMA_PERIOD).mean()
        sma20 = px_sub.rolling(self.SMA_SHORT, min_periods=self.SMA_SHORT).mean()
        daily_ret_abs = px_sub.pct_change().abs()
        atr = daily_ret_abs.rolling(self.ATR_PERIOD, min_periods=self.ATR_PERIOD).mean()

        avg_vol_df = None
        if self._volume_data is not None:
            vol_cols = [s for s in sym_cols if s in self._volume_data.columns]
            if vol_cols:
                avg_vol_df = self._volume_data[vol_cols].rolling(
                    self.VOL_PERIOD, min_periods=self.VOL_PERIOD).mean()

        # Materialize per-date dicts using fast .dropna().to_dict()
        for date in px_sub.index:
            ret_row = ret30.loc[date].dropna()
            self._drop_30d[date] = ret_row.to_dict() if not ret_row.empty else {}

            low_row = low30.loc[date].dropna()
            self._low_30d[date] = low_row.to_dict() if not low_row.empty else {}

            sma200_row = sma200.loc[date].dropna()
            self._sma200[date] = sma200_row.to_dict() if not sma200_row.empty else {}

            sma20_row = sma20.loc[date].dropna()
            self._sma20[date] = sma20_row.to_dict() if not sma20_row.empty else {}

            atr_row = atr.loc[date].dropna()
            self._atr_pct[date] = atr_row.to_dict() if not atr_row.empty else {}

            if avg_vol_df is not None:
                vol_row = avg_vol_df.loc[date].dropna() if date in avg_vol_df.index else pd.Series(dtype=float)
                self._avg_vol[date] = vol_row.to_dict() if not vol_row.empty else {}

    def generate_signals(self, date, universe_data):
        drops = self._drop_30d.get(date, {})
        if not drops:
            return []

        sma_day = self._sma200.get(date, {})
        low_day = self._low_30d.get(date, {})
        vol_day = self._avg_vol.get(date, {})
        prices = (self._price_data.loc[date]
                  if date in self._price_data.index else {})

        signals = []
        for sym in self._symbols:
            drop = drops.get(sym)
            if drop is None or drop > self.DROP_THRESHOLD:
                # Not dropped enough (drop is negative, threshold is -0.15)
                continue

            # Get current price
            if isinstance(prices, dict):
                px = prices.get(sym, np.nan)
            else:
                px = prices.get(sym, np.nan) if sym in prices.index else np.nan
            if np.isnan(px) or px <= 0:
                continue

            # Filter: above 200-day SMA (not in structural downtrend)
            sma = sma_day.get(sym)
            if sma is None or px < sma:
                continue

            # Filter: NOT at 30-day low (want early bounce, not falling knife)
            low = low_day.get(sym)
            if low is not None and px <= low * 1.001:  # within 0.1% of low
                continue

            # Filter: volume
            if vol_day:
                vol = vol_day.get(sym)
                if vol is not None and vol < self.VOL_MIN:
                    continue

            # Confidence: linear scale from 0.50 at -15% to 0.95 at -30%
            # drop ranges from -0.15 to -0.30 (more negative = bigger drop)
            drop_range = self.DROP_MAX - self.DROP_THRESHOLD  # -0.30 - (-0.15) = -0.15
            drop_pct = (drop - self.DROP_THRESHOLD) / drop_range  # 0.0 at -15%, 1.0 at -30%
            drop_pct = min(1.0, max(0.0, drop_pct))
            confidence = 0.50 + drop_pct * 0.45  # 0.50 to 0.95

            signals.append(Signal(
                symbol=sym,
                confidence=confidence,
                strategy_name=self.name,
                fwd_ret=0.0,
                price_based=True,
            ))

        # Sort by confidence descending (biggest drops first)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def check_exit(self, position, current_data):
        idx = current_data["idx"]
        date = current_data["date"]
        prices = current_data.get("prices", {})

        # Max hold
        days_held = idx - position.entry_idx
        if days_held >= self.MAX_HOLD:
            return True, "max_hold"

        # Need current price
        cur_px = prices.get(position.symbol, np.nan)
        if np.isnan(cur_px) or cur_px <= 0:
            return False, ""

        entry_px = position.entry_price
        if entry_px <= 0:
            return False, ""

        # Stop-loss: -10% from entry
        ret_from_entry = (cur_px / entry_px) - 1.0
        if ret_from_entry <= self.STOP_LOSS:
            return True, "stop_loss"

        # Take-profit: price recovers to 20-day SMA
        sma20_day = self._sma20.get(date, {})
        sma20 = sma20_day.get(position.symbol)
        if sma20 is not None and cur_px >= sma20:
            return True, "take_profit_sma20"

        return False, ""

    def get_position_size(self, signal, portfolio_value, date=None):
        from unified_backtester import USE_VOL_SIZING, _vol_targeted_size_pct
        if USE_VOL_SIZING:
            size_pct = _vol_targeted_size_pct(signal.symbol, date)
            if size_pct is not None:
                return portfolio_value * size_pct
        # Check ATR for high-volatility reduction
        atr_pct = 0.0
        for d in sorted(self._atr_pct.keys(), reverse=True):
            atr_day = self._atr_pct[d]
            if signal.symbol in atr_day:
                atr_pct = atr_day[signal.symbol]
                break

        if atr_pct > self.HIGH_ATR_THRESH:
            return portfolio_value * self.HIGH_ATR_PCT
        return portfolio_value * self.BASE_PCT
