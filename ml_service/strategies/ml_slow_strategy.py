"""
ML Slow Strategy (Strategy 4)
==============================
Uses a LightGBM model trained to predict 30-day forward returns > 5%.
Complements the ML Medium model (10-day horizon) with longer-term picks.

Signal generation:
  - BUY when calibrated probability > 0.55

Confidence scaling:
  - Position size scales with confidence:
    25% of base size at 0.55, 100% at 0.80+

Exit rules (checked in order):
  - Stop-loss: price drops 10% from entry
  - Take-profit: price rises 10% from entry
  - Max hold: 45 trading days

Position sizing:
  - Base: 12% of portfolio
  - Reduced to 8% if ATR(14)/price > 3%
  - Scaled by confidence (0.25 at 0.55, 1.0 at 0.80+)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_base import Strategy, Signal


class MLSlowStrategy(Strategy):
    """
    ML Slow strategy: 30-day predictions from a calibrated LightGBM model.
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
    PROB_THRESHOLD = 0.55
    STOP_LOSS      = -0.10   # -10% from entry
    TAKE_PROFIT    = 0.10    # +10% from entry
    MAX_HOLD       = 45      # trading days
    BASE_PCT       = 0.12
    HIGH_ATR_PCT   = 0.08
    HIGH_ATR_THRESH = 0.03   # ATR/price > 3%
    ATR_PERIOD     = 14

    # Confidence scaling: 25% size at 0.55, 100% at 0.80+
    CONF_MIN       = 0.55
    CONF_MAX       = 0.80
    SIZE_MIN       = 0.25
    SIZE_MAX       = 1.00

    def __init__(self, predictions_df, price_data=None):
        """
        Args
        ----
        predictions_df : DataFrame with columns: date, symbol, prob
        price_data     : DataFrame — close prices, columns=symbols, index=dates
                         (used for ATR-based position sizing)
        """
        self._price_data = price_data
        self._signals_by_date = {}
        self._atr_pct = {}
        self._build_lookup(predictions_df)
        if price_data is not None:
            self._precompute_atr()

    @property
    def name(self):
        return "ml_slow"

    def _build_lookup(self, df):
        """Pre-build per-date signal lists, sorted by prob descending."""
        for date, grp in df[["date", "symbol", "prob"]].groupby("date"):
            g = grp.sort_values("prob", ascending=False)
            sigs = []
            for row in g.itertuples(index=False):
                if row.symbol in self.NEVER_BUY or row.symbol == "SPY":
                    continue
                if row.prob > self.PROB_THRESHOLD:
                    sigs.append((row.symbol, row.prob))
            if sigs:
                self._signals_by_date[date] = sigs

    def _precompute_atr(self):
        """Pre-compute ATR as percentage of price for position sizing."""
        px = self._price_data
        daily_ret_abs = px.pct_change().abs()
        atr = daily_ret_abs.rolling(self.ATR_PERIOD, min_periods=self.ATR_PERIOD).mean()

        symbols = [s for s in px.columns if s not in self.NEVER_BUY and s != "SPY"]
        for date in px.index:
            atr_day = {}
            for sym in symbols:
                if sym in atr.columns:
                    v = atr.at[date, sym]
                    if not np.isnan(v):
                        atr_day[sym] = v
            self._atr_pct[date] = atr_day

    def generate_signals(self, date, universe_data):
        raw = self._signals_by_date.get(date, [])
        signals = []
        for sym, prob in raw:
            signals.append(Signal(
                symbol=sym,
                confidence=prob,
                strategy_name=self.name,
                fwd_ret=0.0,
                price_based=True,
            ))
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

        ret_from_entry = (cur_px / entry_px) - 1.0

        # Stop-loss: -10% from entry
        if ret_from_entry <= self.STOP_LOSS:
            return True, "stop_loss"

        # Take-profit: +10% from entry
        if ret_from_entry >= self.TAKE_PROFIT:
            return True, "take_profit"

        return False, ""

    def get_position_size(self, signal, portfolio_value):
        # ATR-based base sizing
        atr_pct = 0.0
        if self._atr_pct:
            for date in sorted(self._atr_pct.keys(), reverse=True):
                atr_day = self._atr_pct[date]
                if signal.symbol in atr_day:
                    atr_pct = atr_day[signal.symbol]
                    break

        base = self.HIGH_ATR_PCT if atr_pct > self.HIGH_ATR_THRESH else self.BASE_PCT

        # Confidence scaling: 25% at 0.55, 100% at 0.80+
        conf = signal.confidence
        if conf >= self.CONF_MAX:
            scale = self.SIZE_MAX
        elif conf <= self.CONF_MIN:
            scale = self.SIZE_MIN
        else:
            t = (conf - self.CONF_MIN) / (self.CONF_MAX - self.CONF_MIN)
            scale = self.SIZE_MIN + t * (self.SIZE_MAX - self.SIZE_MIN)

        return portfolio_value * base * scale
