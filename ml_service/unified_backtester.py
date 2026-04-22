"""
Unified Multi-Strategy Backtest Engine
======================================
Infrastructure for coordinating multiple trading strategies in a single
backtest. Each strategy implements a standard interface; the PortfolioManager
handles slot allocation, overlap/conflict resolution, and SPY idle cash.

Validation
----------
Run this file directly to validate against the original backtest_ml.py:

    python3 unified_backtester.py

The validation runs MLMediumStrategy (wrapping the existing LightGBM model)
as the sole active strategy, and compares results against the original
backtest_ml.py simulation. Results must match exactly.

Multi-Strategy Slot System
--------------------------
  ML Medium:      2 primary slots
  Momentum:       1 primary slot  (placeholder)
  Mean Reversion: 1 primary slot  (placeholder)
  ML Fast:        1 primary slot  (placeholder)
  ML Slow:        1 primary slot  (placeholder)
  Flex pool:      2 shared slots
  Total max:      8 positions
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import time
import warnings

import numpy as np
import pandas as pd

# Import base classes from strategy_base (shared with strategy modules)
from strategy_base import Strategy, Signal, Position

warnings.filterwarnings("ignore")

DATA_DIR  = Path(__file__).resolve().parent / "data"
CACHE_DIR = Path(__file__).resolve().parent / "cache"


# ══════════════════════════════════════════════════════════════════════════════
#  Caching helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    """Return path to a cache file inside ml_service/cache/."""
    return CACHE_DIR / f"{name}.parquet"


def _file_mtime(path: Path) -> float:
    """Return mtime of *path*, or 0.0 if it doesn't exist."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _cache_meta_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.meta"


def _write_meta(name: str, key: str):
    _ensure_cache_dir()
    _cache_meta_path(name).write_text(key)


def _read_meta(name: str) -> str:
    p = _cache_meta_path(name)
    return p.read_text().strip() if p.exists() else ""


def _bars_cache_key(symbols: list[str], start: str, end: str) -> str:
    sym_hash = hashlib.md5(",".join(sorted(symbols)).encode()).hexdigest()[:12]
    return f"{start}_{end}_{sym_hash}"


def load_bars_cached(symbols: list[str], start: str, end: str,
                     no_cache: bool = False) -> pd.DataFrame:
    """
    Download close prices via yfinance, with 24-hour disk cache.

    Returns a DataFrame with columns=symbols, index=dates (close prices).
    """
    import yfinance as yf

    name = "bars"
    key = _bars_cache_key(symbols, start, end)
    cache_file = _cache_path(name)

    if not no_cache and cache_file.exists():
        cached_key = _read_meta(name)
        age_hours = (time.time() - _file_mtime(cache_file)) / 3600
        if cached_key == key and age_hours < 24:
            return pd.read_parquet(cache_file)

    syms = list(set(["SPY"] + symbols))
    raw = yf.download(syms, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    _ensure_cache_dir()
    close.to_parquet(cache_file)
    _write_meta(name, key)
    return close


def load_predictions_cached(pred_file: Path = None,
                            no_cache: bool = False,
                            validation: bool = False) -> pd.DataFrame:
    """
    Load predictions.parquet with cache keyed on model file mtime.

    When *validation=True*, loads predictions_validation.parquet instead
    (produced by train_validation_model.py with 2024+ holdout).
    """
    if pred_file is None:
        if validation:
            pred_file = DATA_DIR / "predictions_validation.parquet"
        else:
            pred_file = DATA_DIR / "predictions.parquet"

    name = "predictions_validation" if validation else "predictions"
    model_file = DATA_DIR / ("model_validation.lgb" if validation else "model.lgb")
    model_mt = str(_file_mtime(model_file))
    pred_mt = str(_file_mtime(pred_file))
    key = f"{model_mt}_{pred_mt}"

    cache_file = _cache_path(name)
    if not no_cache and cache_file.exists() and _read_meta(name) == key:
        df = pd.read_parquet(cache_file)
        df["date"] = pd.to_datetime(df["date"])
        # Normalize column name for strategy compatibility
        if "prob_ensemble" in df.columns and "prob" not in df.columns:
            df = df.rename(columns={"prob_ensemble": "prob"})
        return df

    df = pd.read_parquet(pred_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    # Normalize column name for strategy compatibility
    if "prob_ensemble" in df.columns and "prob" not in df.columns:
        df = df.rename(columns={"prob_ensemble": "prob"})

    _ensure_cache_dir()
    df.to_parquet(cache_file, index=False)
    _write_meta(name, key)
    return df


def load_features_cached(features_file: Path = None,
                         no_cache: bool = False,
                         symbols: list[str] = None,
                         start: str = None,
                         end: str = None) -> pd.DataFrame:
    """
    Load features.parquet with cache keyed on file mtime.

    When *symbols*, *start*, or *end* are provided, uses pyarrow filter
    pushdown so only matching rows are loaded from disk — drastically
    reduces peak memory for targeted backtests.
    """
    import pyarrow.parquet as pq

    if features_file is None:
        features_file = DATA_DIR / "features.parquet"

    # Build pyarrow filters for pushdown
    filters = []
    if symbols:
        filters.append(("symbol", "in", symbols))
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))

    has_filters = len(filters) > 0

    # Cache is only used for unfiltered reads (full file)
    name = "features"
    feat_mt = str(_file_mtime(features_file))
    key = feat_mt

    if not has_filters:
        cache_file = _cache_path(name)
        if not no_cache and cache_file.exists() and _read_meta(name) == key:
            df = pd.read_parquet(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            return df

    # Read with optional filter pushdown
    if has_filters:
        df = pd.read_parquet(features_file, filters=filters)
    else:
        df = pd.read_parquet(features_file)

    df["date"] = pd.to_datetime(df["date"])

    # Only cache unfiltered reads
    if not has_filters:
        _ensure_cache_dir()
        df.to_parquet(cache_file, index=False)
        _write_meta(name, key)

    return df

# ── Constants ────────────────────────────────────────────────────────────────

INITIAL_CASH    = 100_000.0
SLIPPAGE        = 0.0005       # 0.05% per leg (one-way)
HOLD_DAYS       = 10           # default hold period (trading days)
POSITION_PCT    = 0.12         # 12% of portfolio per position at full confidence

# SPY idle-cash parking
SPY_RESERVE_PCT   = 0.30      # always keep 30% as cash reserve
SPY_THRESHOLD_PCT = 0.20      # park idle cash when it exceeds 20% of portfolio
SPY_INVEST_PCT    = 0.85      # invest 85% of idle cash into SPY

# Multi-strategy coordination
OVERLAP_2X      = 1.25        # size multiplier when 2 strategies agree
OVERLAP_3X      = 1.50        # size multiplier when 3+ agree
COOLDOWN_DAYS   = 5           # days a symbol is blocked after a sell

# ── Volatility-targeted position sizing ─────────────────────────────
VOL_TARGET_RISK_PCT = 0.01      # Target 1% of portfolio at 1-ATR move
VOL_MAX_POSITION_PCT = 0.20     # Cap at 20% of portfolio
VOL_MIN_POSITION_PCT = 0.03     # Floor at 3% of portfolio
USE_VOL_SIZING = False          # Global flag; set True by --vol-sizing CLI flag

# Module-level ATR lookup (set by PortfolioManager before run)
_global_atr_df = None


def _get_atr_pct(symbol, date):
    """Lookup atr_pct for (symbol, date). Returns None if unavailable."""
    global _global_atr_df
    if _global_atr_df is None:
        return None
    try:
        v = _global_atr_df.at[date, symbol]
        return float(v) if not np.isnan(v) else None
    except (KeyError, IndexError):
        return None


def _vol_targeted_size_pct(symbol, date):
    """Return position size as % of portfolio using volatility targeting.
    Falls back to None if ATR unavailable — caller should use strategy BASE_PCT."""
    atr_pct = _get_atr_pct(symbol, date)
    if atr_pct is None or atr_pct <= 0:
        return None
    size_pct = VOL_TARGET_RISK_PCT / atr_pct
    return max(VOL_MIN_POSITION_PCT, min(size_pct, VOL_MAX_POSITION_PCT))


# ══════════════════════════════════════════════════════════════════════════════
#  Slot Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SlotConfig:
    """Controls how many positions each strategy may hold."""
    strategy_slots: dict       # strategy_name → primary slot count
    flex_slots: int = 2        # shared pool available to any strategy
    max_positions: int = 8     # hard cap across all strategies

    def available_for(self, strategy_name, strategy_counts, total_open):
        """Return how many additional positions *strategy_name* can open."""
        primary = self.strategy_slots.get(strategy_name, 0)
        held = strategy_counts.get(strategy_name, 0)
        primary_avail = max(0, primary - held)

        # Flex: count positions beyond each strategy's primary allocation
        total_in_primary = sum(
            min(strategy_counts.get(s, 0), self.strategy_slots.get(s, 0))
            for s in self.strategy_slots
        )
        flex_used = max(0, total_open - total_in_primary)
        flex_avail = max(0, self.flex_slots - flex_used)

        return min(primary_avail + flex_avail, self.max_positions - total_open)


# ══════════════════════════════════════════════════════════════════════════════
#  ML Medium Strategy — wraps the existing LightGBM model
# ══════════════════════════════════════════════════════════════════════════════

class MLMediumStrategy(Strategy):
    """
    Produces BUY signals from the LightGBM model's predicted probabilities.
    Uses a fixed 10-trading-day hold period and ML confidence scaling for
    position sizing.

    Two selection modes:
      - ``"top_n"``     (default): cross-sectional top-N picks per day,
        matching the live signal_server.py behaviour.
      - ``"threshold"``: absolute probability cutoff (legacy backtest mode).
    """

    def __init__(self, predictions_df, threshold=0.55, top_n=5,
                 selection_mode="top_n", position_pct=POSITION_PCT):
        self._threshold = threshold
        self._top_n = top_n
        self._selection_mode = selection_mode  # "top_n" or "threshold"
        self._position_pct = position_pct
        self._signals_by_date = {}
        self._build_lookup(predictions_df)

    @property
    def name(self):
        return "ml_medium"

    # -- private ----------------------------------------------------------

    def _build_lookup(self, df):
        """Pre-build per-date signal lists, sorted by prob descending."""
        for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
            g = grp.sort_values("prob", ascending=False)
            self._signals_by_date[date] = [
                (row.symbol, row.prob, row.fwd_ret)
                for row in g.itertuples(index=False)
            ]

    # -- interface --------------------------------------------------------

    def generate_signals(self, date, universe_data):
        raw = self._signals_by_date.get(date, [])
        if self._selection_mode == "top_n":
            # Cross-sectional: top-N stocks by probability, regardless of
            # absolute value.  Matches live signal_server.py behaviour.
            selected = raw[:self._top_n]
        else:
            # Legacy threshold mode
            selected = [(s, p, r) for s, p, r in raw if p > self._threshold]
        return [
            Signal(symbol=sym, confidence=prob,
                   strategy_name=self.name, fwd_ret=fwd_ret)
            for sym, prob, fwd_ret in selected
        ]

    def check_exit(self, position, current_data):
        if current_data["idx"] >= position.exit_idx:
            return True, "hold_complete"
        return False, ""

    def get_position_size(self, signal, portfolio_value, date=None):
        if USE_VOL_SIZING:
            size_pct = _vol_targeted_size_pct(signal.symbol, date)
            if size_pct is not None:
                return portfolio_value * size_pct
        ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
        return portfolio_value * self._position_pct * ml_mult


# ══════════════════════════════════════════════════════════════════════════════
#  Placeholder Strategies (not yet implemented — generate no signals)
# ══════════════════════════════════════════════════════════════════════════════

class MomentumStrategy(Strategy):
    """
    3-month momentum strategy: buy top-ranked stocks by 63-day return.

    Signal generation:
      - Rank all universe symbols by 63-trading-day total return
      - Generate BUY for top 5 that pass filters
      - Confidence = (100 - rank) / 100  (rank 1 → 0.99, rank 5 → 0.95)

    Filters:
      - NEVER_BUY blacklist (leveraged/inverse/volatility products)
      - Price above 200-day SMA
      - 20-day avg volume > 500K shares

    Exit rules (checked in order):
      - Stop-loss: price drops 8% from entry
      - Take-profit: price rises 20% from entry
      - Trailing stop: price drops 8% from peak since entry
      - Momentum break: symbol's rank drops below top 20
      - Max hold: 60 trading days

    Position sizing:
      - Base: 12% of portfolio
      - Reduced to 8% if ATR(14) / price > 3%
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
    LOOKBACK       = 63     # trading days for momentum ranking
    TOP_N          = 5      # signals per day
    SMA_PERIOD     = 200    # trend filter
    VOL_PERIOD     = 20     # avg volume window
    VOL_MIN        = 500_000
    ATR_PERIOD     = 14
    STOP_LOSS      = -0.08
    TAKE_PROFIT    = 0.20
    TRAIL_STOP     = -0.08  # from peak
    RANK_BREAK     = 20     # exit if rank drops below top N
    MAX_HOLD       = 60     # trading days
    BASE_PCT       = 0.12
    HIGH_ATR_PCT   = 0.08
    HIGH_ATR_THRESH = 0.03  # ATR/price threshold

    def __init__(self, price_data, volume_data=None):
        """
        Args
        ----
        price_data : DataFrame — close prices, columns=symbols, index=dates
        volume_data : DataFrame — daily volume, same shape (optional, for volume filter)
        """
        self._price_data = price_data
        self._volume_data = volume_data
        self._symbols = [s for s in price_data.columns if s not in self.NEVER_BUY and s != "SPY"]

        # Pre-compute all indicators
        self._mom_ret = {}      # {date → {sym → 63d return}}
        self._rankings = {}     # {date → {sym → rank (1-based)}}
        self._sma200 = {}       # {date → {sym → sma200}}
        self._atr_pct = {}      # {date → {sym → atr/price}}
        self._avg_vol = {}      # {date → {sym → 20d avg volume}}
        self._precompute()

    @property
    def name(self):
        return "momentum"

    def _precompute(self):
        """Pre-compute momentum returns, rankings, SMA, ATR, volume."""
        dates = self._price_data.index.tolist()
        px = self._price_data

        # 63-day returns
        ret63 = px.pct_change(self.LOOKBACK)

        # 200-day SMA
        sma200 = px.rolling(self.SMA_PERIOD, min_periods=self.SMA_PERIOD).mean()

        # ATR(14) as percentage of price
        # True Range = max(H-L, |H-Cp|, |L-Cp|) — with close-only data, use |C-Cp|
        daily_ret_abs = px.pct_change().abs()
        atr = daily_ret_abs.rolling(self.ATR_PERIOD, min_periods=self.ATR_PERIOD).mean()

        # 20-day avg volume
        avg_vol = None
        if self._volume_data is not None:
            avg_vol = self._volume_data.rolling(self.VOL_PERIOD, min_periods=self.VOL_PERIOD).mean()

        for date in dates:
            if date not in ret63.index:
                continue

            # Momentum returns for this date
            rets = {}
            for sym in self._symbols:
                if sym in ret63.columns:
                    r = ret63.at[date, sym]
                    if not np.isnan(r):
                        rets[sym] = r
            self._mom_ret[date] = rets

            # Rankings (1 = best momentum)
            if rets:
                sorted_syms = sorted(rets.keys(), key=lambda s: rets[s], reverse=True)
                self._rankings[date] = {s: rank + 1 for rank, s in enumerate(sorted_syms)}

            # SMA200
            sma_day = {}
            for sym in self._symbols:
                if sym in sma200.columns:
                    v = sma200.at[date, sym]
                    if not np.isnan(v):
                        sma_day[sym] = v
            self._sma200[date] = sma_day

            # ATR percentage
            atr_day = {}
            for sym in self._symbols:
                if sym in atr.columns:
                    v = atr.at[date, sym]
                    if not np.isnan(v):
                        atr_day[sym] = v
            self._atr_pct[date] = atr_day

            # Volume
            if avg_vol is not None:
                vol_day = {}
                for sym in self._symbols:
                    if sym in avg_vol.columns:
                        v = avg_vol.at[date, sym]
                        if not np.isnan(v):
                            vol_day[sym] = v
                self._avg_vol[date] = vol_day

    def generate_signals(self, date, universe_data):
        rankings = self._rankings.get(date, {})
        if not rankings:
            return []

        sma_day = self._sma200.get(date, {})
        vol_day = self._avg_vol.get(date, {})
        prices = self._price_data.loc[date] if date in self._price_data.index else {}

        signals = []
        # Iterate by rank order
        sorted_syms = sorted(rankings.keys(), key=lambda s: rankings[s])

        for sym in sorted_syms:
            if len(signals) >= self.TOP_N:
                break

            rank = rankings[sym]

            # Filter: above 200-SMA
            if sym in sma_day and sym in prices:
                px = prices[sym] if not isinstance(prices, dict) else prices.get(sym, np.nan)
                if isinstance(px, (int, float)) and not np.isnan(px):
                    if px < sma_day[sym]:
                        continue
                else:
                    continue
            else:
                continue  # skip if no SMA data

            # Filter: volume
            if vol_day and sym in vol_day:
                if vol_day[sym] < self.VOL_MIN:
                    continue

            confidence = (100 - rank) / 100.0

            signals.append(Signal(
                symbol=sym,
                confidence=confidence,
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

        # Need current price for other exit rules
        cur_px = prices.get(position.symbol, np.nan)
        if np.isnan(cur_px) or cur_px <= 0:
            return False, ""

        entry_px = position.entry_price
        if entry_px <= 0:
            return False, ""

        ret_from_entry = (cur_px / entry_px) - 1.0

        # Stop-loss
        if ret_from_entry <= self.STOP_LOSS:
            return True, "stop_loss"

        # Take-profit
        if ret_from_entry >= self.TAKE_PROFIT:
            return True, "take_profit"

        # Trailing stop (from peak)
        if position.peak_price > 0:
            ret_from_peak = (cur_px / position.peak_price) - 1.0
            if ret_from_peak <= self.TRAIL_STOP:
                return True, "trailing_stop"

        # Momentum break — rank dropped below top 20
        rankings = self._rankings.get(date, {})
        rank = rankings.get(position.symbol, 999)
        if rank > self.RANK_BREAK:
            return True, "momentum_break"

        return False, ""

    def get_position_size(self, signal, portfolio_value, date=None):
        if USE_VOL_SIZING:
            size_pct = _vol_targeted_size_pct(signal.symbol, date)
            if size_pct is not None:
                return portfolio_value * size_pct
        # Check ATR for the signal date — use reduced size for high-vol stocks
        # (ATR data keyed by most recent date is close enough)
        atr_pct = 0.0
        for d in sorted(self._atr_pct.keys(), reverse=True):
            atr_day = self._atr_pct[d]
            if signal.symbol in atr_day:
                atr_pct = atr_day[signal.symbol]
                break

        if atr_pct > self.HIGH_ATR_THRESH:
            return portfolio_value * self.HIGH_ATR_PCT
        return portfolio_value * self.BASE_PCT


# MeanReversionStrategy — imported from strategies module
from strategies.mean_reversion_strategy import MeanReversionStrategy  # noqa: E402
from strategies.ml_slow_strategy import MLSlowStrategy  # noqa: E402


class MLFastStrategy(Strategy):
    @property
    def name(self):
        return "ml_fast"
    def generate_signals(self, date, universe_data):
        return []
    def check_exit(self, position, current_data):
        return False, ""
    def get_position_size(self, signal, portfolio_value, date=None):
        return 0.0


# MLSlowStrategy imported from strategies/ml_slow_strategy.py above


# ══════════════════════════════════════════════════════════════════════════════
#  Mega-Cap Momentum Strategy
# ══════════════════════════════════════════════════════════════════════════════

MEGACAP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "V", "MA", "COST",
]


# ── TSMOM Strategy Configuration ──────────────────────────────────────────────
TSMOM_UNIVERSE = [
    # US Equity Broad
    "SPY", "QQQ", "IWM",
    # US Sectors
    "XLK", "XLF", "XLV", "XLE", "XLY",
    # International Equity
    "EFA", "EEM", "FXI",
    # Commodities
    "GLD", "DBC", "USO",
    # Bonds
    "TLT", "IEF", "HYG",
    # Currency
    "UUP",
]
TSMOM_LOOKBACK_LONG = 252     # 12-month return check
TSMOM_LOOKBACK_SHORT = 63     # 3-month return check
TSMOM_TOP_N = 2               # Max 2 ETFs held at once
TSMOM_TARGET_VOL = 0.10       # 10% annualized target per position
TSMOM_VOL_LOOKBACK = 20       # Days for realized vol
TSMOM_MAX_POS_PCT = 0.15      # Cap at 15% of portfolio per ETF
TSMOM_STOP_LOSS = -0.08       # -8% stop loss
TSMOM_MAX_HOLD = 63           # Force re-eval after 63 trading days


class MegaCapStrategy(Strategy):
    """
    12-month-minus-1-month momentum on mega-cap stocks.

    Signal generation:
      - Universe: top 15 mega-caps by market cap (fixed list)
      - Rank by 12-1 momentum (252d return skipping recent 21d)
      - Filter: above 200-day SMA, within 5% of 52-week high
      - Generate top-N signals

    Exit rules:
      - Stop-loss: -5% from entry
      - Take-profit: +20% from entry
      - Below 50-day SMA
      - Max hold: 90 trading days
    """

    LOOKBACK    = 252
    SKIP        = 21
    HIGH_52WK   = 0.95   # within 5% of 52-week high
    TOP_N       = 2
    SMA_LONG    = 200
    SMA_EXIT    = 50
    STOP_LOSS   = -0.05
    TAKE_PROFIT = 0.20
    MAX_HOLD    = 90
    BASE_PCT    = 0.12

    def __init__(self, price_data):
        self._price_data = price_data
        self._symbols = [s for s in MEGACAP_UNIVERSE if s in price_data.columns]
        self._mom = {}       # date → [(sym, mom_12_1)]
        self._sma200 = {}    # date → {sym → sma200}
        self._sma50 = {}     # date → {sym → sma50}
        self._precompute()

    @property
    def name(self):
        return "mega_cap"

    def _precompute(self):
        px = self._price_data
        dates = px.index.tolist()
        for date in dates:
            loc = px.index.get_loc(date)
            if loc < self.LOOKBACK:
                continue

            candidates = []
            sma200_day = {}
            sma50_day = {}

            for sym in self._symbols:
                col = px[sym]
                curr = col.iloc[loc]
                if np.isnan(curr) or curr <= 0:
                    continue

                # SMA200
                if loc >= self.SMA_LONG:
                    sma200_day[sym] = col.iloc[loc - self.SMA_LONG + 1:loc + 1].mean()
                    if curr <= sma200_day[sym]:
                        continue
                else:
                    continue

                # SMA50 (for exit check)
                if loc >= self.SMA_EXIT:
                    sma50_day[sym] = col.iloc[loc - self.SMA_EXIT + 1:loc + 1].mean()

                # 12-1 momentum
                price_skip = col.iloc[loc - self.SKIP]
                price_12m = col.iloc[loc - self.LOOKBACK]
                if np.isnan(price_12m) or price_12m <= 0:
                    continue
                if np.isnan(price_skip) or price_skip <= 0:
                    continue
                mom_12_1 = (price_skip / price_12m) - 1.0

                # 52-week high filter
                if loc >= 252:
                    high_52wk = col.iloc[loc - 251:loc + 1].max()
                    if high_52wk > 0 and (curr / high_52wk) < self.HIGH_52WK:
                        continue

                candidates.append((sym, mom_12_1))

            candidates.sort(key=lambda x: x[1], reverse=True)
            self._mom[date] = candidates
            self._sma200[date] = sma200_day
            self._sma50[date] = sma50_day

    def generate_signals(self, date, universe_data):
        candidates = self._mom.get(date, [])
        return [
            Signal(symbol=sym, confidence=0.7,
                   strategy_name=self.name, fwd_ret=0.0,
                   price_based=True)
            for sym, _ in candidates[:self.TOP_N]
        ]

    def check_exit(self, position, current_data):
        idx = current_data["idx"]
        date = current_data["date"]
        prices = current_data.get("prices", {})

        days_held = idx - position.entry_idx
        if days_held >= self.MAX_HOLD:
            return True, "max_hold"

        cur_px = prices.get(position.symbol, np.nan)
        if np.isnan(cur_px) or cur_px <= 0:
            return False, ""

        entry_px = position.entry_price
        if entry_px <= 0:
            return False, ""

        ret = (cur_px / entry_px) - 1.0

        if ret <= self.STOP_LOSS:
            return True, "stop_loss"
        if ret >= self.TAKE_PROFIT:
            return True, "take_profit"

        # Below 50-SMA exit
        sma50_day = self._sma50.get(date, {})
        sma50 = sma50_day.get(position.symbol)
        if sma50 is not None and cur_px < sma50:
            return True, "below_sma50"

        return False, ""

    def get_position_size(self, signal, portfolio_value, date=None):
        if USE_VOL_SIZING:
            size_pct = _vol_targeted_size_pct(signal.symbol, date)
            if size_pct is not None:
                return portfolio_value * size_pct
        return portfolio_value * self.BASE_PCT


class TSMOMStrategy(Strategy):
    """
    Time-Series Momentum across a multi-asset ETF universe (17 ETFs).

    Each ETF evaluated INDEPENDENTLY on its own history. Eligible ONLY if its
    12-month AND 3-month returns are both positive. In bear markets, TSMOM exits
    to cash rather than forcing long positions.

    Provides bear-market protection and cross-asset diversification that the
    cross-sectional MomentumStrategy (always long top-5) cannot offer.
    """

    def __init__(self, price_data):
        """price_data: DataFrame -- close prices, columns=symbols, index=dates."""
        self._price_data = price_data
        self._symbols = [s for s in TSMOM_UNIVERSE if s in price_data.columns]
        self._build_lookup()

    @property
    def name(self):
        return "tsmom"

    def _build_lookup(self):
        """Pre-compute per-date, per-symbol: ret_252d, ret_63d, vol_20d."""
        self._stats_by_date = {}
        px = self._price_data

        for sym in self._symbols:
            if sym not in px.columns:
                continue
            prices = px[sym]
            returns = prices.pct_change()

            ret_252d = prices.pct_change(periods=TSMOM_LOOKBACK_LONG)
            ret_63d = prices.pct_change(periods=TSMOM_LOOKBACK_SHORT)
            vol_20d = returns.rolling(TSMOM_VOL_LOOKBACK).std() * np.sqrt(252)

            for date in px.index:
                r252 = ret_252d.get(date)
                r63 = ret_63d.get(date)
                v20 = vol_20d.get(date)

                if pd.isna(r252) or pd.isna(r63) or pd.isna(v20) or v20 <= 0:
                    continue

                if date not in self._stats_by_date:
                    self._stats_by_date[date] = {}
                self._stats_by_date[date][sym] = {
                    "ret_252d": float(r252),
                    "ret_63d": float(r63),
                    "vol_20d": float(v20),
                }

    def generate_signals(self, date, universe_data):
        """Top 2 ETFs by 3m return, only if both 12m AND 3m returns are positive."""
        stats_today = self._stats_by_date.get(date, {})
        if not stats_today:
            return []

        candidates = []
        for sym, stats in stats_today.items():
            if stats["ret_252d"] > 0 and stats["ret_63d"] > 0:
                candidates.append((sym, stats["ret_63d"], stats["vol_20d"]))

        candidates.sort(key=lambda x: -x[1])
        top_n = candidates[:TSMOM_TOP_N]

        signals = []
        for sym, mom_score, vol in top_n:
            sig = Signal(
                symbol=sym,
                confidence=1.0,
                strategy_name=self.name,
                fwd_ret=0.0,
                price_based=True,
            )
            sig._vol = vol
            signals.append(sig)
        return signals

    def get_position_size(self, signal, portfolio_value, date=None):
        """Inverse-volatility sizing, capped at TSMOM_MAX_POS_PCT."""
        if USE_VOL_SIZING:
            size_pct = _vol_targeted_size_pct(signal.symbol, date)
            if size_pct is not None:
                return portfolio_value * size_pct
        vol = getattr(signal, "_vol", 0.20)
        scale = TSMOM_TARGET_VOL / max(vol, 0.05)
        return portfolio_value * TSMOM_MAX_POS_PCT * min(scale, 1.0)

    def check_exit(self, position, current_data):
        """Exit if: max hold hit, stop loss, or either time horizon flips negative."""
        sym = position.symbol
        idx = current_data["idx"]
        date = current_data["date"]
        prices = current_data.get("prices", {})

        days_held = idx - position.entry_idx
        if days_held >= TSMOM_MAX_HOLD:
            return True, "tsmom_max_hold"

        cur_px = prices.get(sym, np.nan)
        if not np.isnan(cur_px) and position.entry_price > 0:
            ret = (cur_px / position.entry_price) - 1.0
            if ret <= TSMOM_STOP_LOSS:
                return True, "tsmom_stop_loss"

        stats = self._stats_by_date.get(date, {}).get(sym)
        if stats:
            if stats["ret_252d"] <= 0:
                return True, "tsmom_trend_break_12m"
            if stats["ret_63d"] <= 0:
                return True, "tsmom_trend_break_3m"

        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
#  Portfolio Manager
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioManager:
    """
    Coordinates multiple strategies in a single simulation.

    Day-by-day loop:
      1. Close expiring positions (via ``strategy.check_exit``)
      2. Gather signals from every active strategy
      3. Resolve overlaps (size boost) and conflicts (cooldowns)
      4. Allocate slots, size positions, execute buys
      5. Park idle cash in SPY
      6. Mark-to-market
    """

    def __init__(self, strategies, slot_config,
                 initial_cash=INITIAL_CASH, slippage=SLIPPAGE,
                 hold_days=HOLD_DAYS):
        self.strategies = {s.name: s for s in strategies}
        self.slot_config = slot_config
        self.initial_cash = initial_cash
        self.slippage = slippage
        self.hold_days = hold_days

    # ── public API ───────────────────────────────────────────────────────

    def run(self, all_dates, spy_prices=None, price_data=None):
        """
        Execute the backtest.

        Args
        ----
        all_dates : sorted list of trading-day timestamps
        spy_prices : ``{date → float}`` for idle-cash parking (None to disable)
        price_data : DataFrame with close prices (columns=symbols, index=dates).
                     Required for price-based strategies (momentum, etc.).

        Returns
        -------
        (portfolio_series, trades_list)
        """
        n_dates = len(all_dates)
        multi = len(self.strategies) > 1

        # Compute global ATR for vol-targeted sizing (if enabled)
        global _global_atr_df
        if USE_VOL_SIZING and price_data is not None:
            daily_ret_abs = price_data.pct_change().abs()
            _global_atr_df = daily_ret_abs.rolling(14, min_periods=14).mean()
        else:
            _global_atr_df = None

        # Pre-build price lookup for price-based strategies
        # {date → {symbol → price}}
        px_lookup = {}
        if price_data is not None:
            for date in all_dates:
                if date in price_data.index:
                    px_lookup[date] = price_data.loc[date].to_dict()

        # -- per-run state (local, so PortfolioManager is reusable) --------
        cash            = float(self.initial_cash)
        positions       = {}        # symbol → Position
        idle_spy_shares = 0.0
        trades          = []        # list of per-trade returns
        cooldowns       = {}        # (symbol, strategy_name) → expiry index
        port_vals       = []

        for i, date in enumerate(all_dates):
            spy_px = spy_prices.get(date) if spy_prices else None
            if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
                spy_px = None

            day_prices = px_lookup.get(date, {})
            cur = {"idx": i, "date": date, "n_dates": n_dates,
                   "prices": day_prices}

            # ── 1. Update peak prices for price-based positions ──────────
            for sym, pos in positions.items():
                if pos.price_based and sym in day_prices:
                    px = day_prices[sym]
                    if not np.isnan(px) and px > pos.peak_price:
                        pos.peak_price = px

            # ── 2. Close expiring positions ──────────────────────────────
            to_close = []
            for sym, pos in positions.items():
                strat = self.strategies[pos.strategy_name]
                should_exit, _reason = strat.check_exit(pos, cur)
                if should_exit:
                    to_close.append(sym)

            for sym in to_close:
                pos = positions.pop(sym)
                if pos.price_based:
                    # Use actual closing price
                    close_px = day_prices.get(sym, pos.entry_price)
                    if np.isnan(close_px):
                        close_px = pos.entry_price
                    actual_ret = (close_px / pos.entry_price) - 1.0
                    gross = pos.cost * (1.0 + actual_ret)
                else:
                    gross = pos.cost * (1.0 + pos.fwd_ret)
                net   = gross * (1.0 - self.slippage)
                cash += net
                trades.append((net - pos.cost) / pos.cost)
                if multi:
                    cooldowns[(sym, pos.strategy_name)] = i + COOLDOWN_DAYS

            # ── 3. Gather signals from all strategies ────────────────────
            all_signals = []
            for strat in self.strategies.values():
                all_signals.extend(strat.generate_signals(date, None))

            # Exclude held symbols
            held = set(positions.keys())
            # For non-price-based signals, also exclude NaN fwd_ret
            all_signals = [s for s in all_signals
                           if s.symbol not in held
                           and (s.price_based or not np.isnan(s.fwd_ret))]

            # Cooldowns (multi-strategy only — per-strategy cooldown)
            if multi:
                all_signals = [s for s in all_signals
                               if cooldowns.get((s.symbol, s.strategy_name), -1) <= i]

            # ── 4. Resolve overlaps (first-strategy-wins) ────────────────
            # When multiple strategies signal the same symbol, keep the first
            # signal (by generation order). No size amplification.
            seen_syms = set()
            resolved = []
            for sig in all_signals:
                if sig.symbol not in seen_syms:
                    seen_syms.add(sig.symbol)
                    resolved.append(sig)

            resolved.sort(key=lambda s: s.confidence, reverse=True)

            # ── 5. Execute buys ──────────────────────────────────────────
            max_slots = self.slot_config.max_positions - len(positions)

            # Release idle SPY before buying picks
            if (spy_px and idle_spy_shares > 0
                    and resolved and max_slots > 0):
                proceeds = idle_spy_shares * spy_px * (1.0 - self.slippage)
                cash += proceeds
                idle_spy_shares = 0.0

            # Consider up to max_slots candidates (matches original [:slots])
            for sig in resolved[:max_slots]:
                # Per-strategy slot check
                counts = {}
                for p in positions.values():
                    counts[p.strategy_name] = counts.get(
                        p.strategy_name, 0) + 1
                if self.slot_config.available_for(
                        sig.strategy_name, counts, len(positions)) <= 0:
                    continue

                # For price-based signals, require a valid entry price
                entry_px = 0.0
                if sig.price_based:
                    entry_px = day_prices.get(sig.symbol, np.nan)
                    if np.isnan(entry_px) or entry_px <= 0:
                        continue

                # Position sizing (strategy provides target)
                strat = self.strategies[sig.strategy_name]
                port_est = cash
                for p in positions.values():
                    if p.price_based:
                        px = day_prices.get(p.symbol, p.entry_price)
                        port_est += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                    else:
                        port_est += p.cost
                target = strat.get_position_size(sig, port_est, date=date)

                cost = min(target, cash * 0.95)
                if cost < 50.0:
                    continue

                cash -= cost * (1.0 + self.slippage)
                exit_idx = min(i + self.hold_days, n_dates - 1)

                positions[sig.symbol] = Position(
                    symbol=sig.symbol,
                    strategy_name=sig.strategy_name,
                    cost=cost,
                    entry_idx=i,
                    exit_idx=exit_idx,
                    fwd_ret=sig.fwd_ret,
                    confidence=sig.confidence,
                    price_based=sig.price_based,
                    entry_price=entry_px,
                    peak_price=entry_px,
                )

            # ── 6. Park idle cash in SPY ─────────────────────────────────
            if spy_px:
                pos_val = 0.0
                for p in positions.values():
                    if p.price_based:
                        px = day_prices.get(p.symbol, p.entry_price)
                        pos_val += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                    else:
                        pos_val += p.cost * (1.0 + p.fwd_ret
                                             * (i - p.entry_idx) / self.hold_days)
                est_port  = cash + idle_spy_shares * spy_px + pos_val
                reserved  = est_port * SPY_RESERVE_PCT
                idle_cash = cash - reserved
                if idle_cash > est_port * SPY_THRESHOLD_PCT:
                    invest = min(idle_cash * SPY_INVEST_PCT, cash * 0.95)
                    new_shares       = invest / spy_px
                    cash            -= invest * (1.0 + self.slippage)
                    idle_spy_shares += new_shares

            # ── 7. Mark-to-market ────────────────────────────────────────
            port_val = cash + (idle_spy_shares * spy_px if spy_px else 0)
            for pos in positions.values():
                if pos.price_based:
                    px = day_prices.get(pos.symbol, pos.entry_price)
                    port_val += pos.cost * (px / pos.entry_price if pos.entry_price > 0 else 1.0)
                else:
                    days_held  = i - pos.entry_idx
                    interp_ret = pos.fwd_ret * days_held / self.hold_days
                    port_val  += pos.cost * (1.0 + interp_ret)
            port_vals.append(port_val)

        series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates))
        return series, [float(t) for t in trades]


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-built Slot Configurations
# ══════════════════════════════════════════════════════════════════════════════

# Validation mode: ML-only, 6 slots (matches original backtest_ml.py)
SLOT_ML_ONLY = SlotConfig(
    strategy_slots={"ml_medium": 6},
    flex_slots=0,
    max_positions=6,
)

# Momentum-only mode: 5 primary + 2 flex = 7 max positions
SLOT_MOM_ONLY = SlotConfig(
    strategy_slots={"momentum": 5},
    flex_slots=2,
    max_positions=7,
)

# ML + Momentum combined: ML 2 + Mom 4 + 2 flex = max 8
SLOT_ML_MOM = SlotConfig(
    strategy_slots={"ml_medium": 2, "momentum": 4},
    flex_slots=2,
    max_positions=8,
)

# Mean Reversion only: 5 primary, no flex
SLOT_MR_ONLY = SlotConfig(
    strategy_slots={"mean_reversion": 5},
    flex_slots=0,
    max_positions=5,
)

# ML + Mean Reversion: ML 2 + MR 2 + 2 flex = max 6
SLOT_ML_MR = SlotConfig(
    strategy_slots={"ml_medium": 2, "mean_reversion": 2},
    flex_slots=2,
    max_positions=6,
)

# ML + Momentum + Mean Reversion: ML 2 + Mom 4 + MR 2 + 2 flex = max 10
SLOT_ML_MOM_MR = SlotConfig(
    strategy_slots={"ml_medium": 2, "momentum": 4, "mean_reversion": 2},
    flex_slots=2,
    max_positions=10,
)

# ML Slow only: 5 primary, no flex
SLOT_ML_SLOW_ONLY = SlotConfig(
    strategy_slots={"ml_slow": 5},
    flex_slots=0,
    max_positions=5,
)

# ML Medium + Momentum + MR + ML Slow: 2+4+2+1+2 flex = max 11
SLOT_ML_MOM_MR_SLOW = SlotConfig(
    strategy_slots={"ml_medium": 2, "momentum": 4, "mean_reversion": 2, "ml_slow": 1},
    flex_slots=2,
    max_positions=11,
)

# Mega-cap only: 2 primary, no flex
SLOT_MCAP_ONLY = SlotConfig(
    strategy_slots={"mega_cap": 2},
    flex_slots=0,
    max_positions=2,
)

# ML + Momentum + Mean Reversion + Mega-cap: ML 2 + Mom 4 + MR 2 + MC 2 + 2 flex = max 12
SLOT_ML_MOM_MR_MCAP = SlotConfig(
    strategy_slots={"ml_medium": 2, "momentum": 4, "mean_reversion": 2, "mega_cap": 2},
    flex_slots=2,
    max_positions=12,
)

# TSMOM only: 2 primary, no flex
SLOT_TSMOM_ONLY = SlotConfig(
    strategy_slots={"tsmom": 2},
    flex_slots=0,
    max_positions=2,
)

# ML + Momentum + Mega-cap + TSMOM: ML 2 + Mom 3 + MC 2 + TSMOM 2 + 2 flex = max 11
SLOT_ML_MOM_MCAP_TSMOM = SlotConfig(
    strategy_slots={"ml_medium": 2, "momentum": 3, "mega_cap": 2, "tsmom": 2},
    flex_slots=2,
    max_positions=11,
)

# Multi-strategy production mode (full)
SLOT_MULTI = SlotConfig(
    strategy_slots={
        "ml_medium":      2,
        "momentum":       1,
        "mean_reversion": 1,
        "ml_fast":        1,
        "ml_slow":        1,
    },
    flex_slots=2,
    max_positions=8,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Validation — compare against original backtest_ml.py
# ══════════════════════════════════════════════════════════════════════════════

def validate():
    """Run both engines side-by-side and print a comparison table."""
    from backtest_utils import load_predictions, calc_metrics, calc_alpha_beta
    from backtest_ml import (
        fetch_benchmarks, run_simulation,
        make_ml_signal_fn, INITIAL_CASH as BT_CASH,
    )

    t0 = time.perf_counter()
    print("=" * 75)
    print("  UNIFIED BACKTESTER — VALIDATION")
    print("  Original backtest_ml.py  vs  unified_backtester.py")
    print("  Strategy: ML Medium only  |  Threshold >0.55  |  SPY idle ON")
    print("=" * 75)

    # ── Load data ────────────────────────────────────────────────────────
    df = load_predictions()
    all_dates     = sorted(df["date"].unique().tolist())
    universe_syms = sorted(df["symbol"].unique().tolist())
    years         = (all_dates[-1] - all_dates[0]).days / 365.25

    start = pd.Timestamp(all_dates[0]).strftime("%Y-%m-%d")
    end   = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close = fetch_benchmarks(start, end, universe_syms)
    close = close.reindex(
        pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates]), method="ffill")
    spy_px  = close["SPY"].dropna()
    spy_bh  = spy_px / spy_px.iloc[0] * BT_CASH
    spy_dict = close["SPY"].to_dict()

    # ── 1. Original backtest_ml.py ───────────────────────────────────────
    print("\n  Running ORIGINAL backtest_ml.py ...", flush=True)
    df_sigs = df[["date", "symbol", "prob", "fwd_ret"]].copy()
    sigs_by_date = {}
    for date, grp in df_sigs.groupby("date"):
        grp_s = grp.sort_values("prob", ascending=False)
        sigs_by_date[date] = list(
            grp_s[["symbol", "prob", "fwd_ret"]].itertuples(
                index=False, name=None))

    orig_vals, orig_trades = run_simulation(
        sigs_by_date, all_dates,
        signal_fn=make_ml_signal_fn(0.55),
        years=years, label="Original", verbose=False,
        spy_prices=spy_dict,
    )
    m_orig = calc_metrics(orig_vals, orig_trades, years, "Original")
    a_orig, _ = calc_alpha_beta(
        orig_vals, spy_bh.reindex(orig_vals.index, method="ffill"))
    m_orig["alpha"] = a_orig

    # ── 2. Unified backtester ────────────────────────────────────────────
    # NOTE: validate() uses threshold mode to match original backtest_ml.py.
    # Production and backtest.py CLI default to top_n mode.
    print("  Running UNIFIED backtester ...", flush=True)
    ml = MLMediumStrategy(df, threshold=0.55, selection_mode="threshold")
    pm = PortfolioManager(strategies=[ml], slot_config=SLOT_ML_ONLY)
    uni_vals, uni_trades = pm.run(all_dates, spy_prices=spy_dict)
    m_uni = calc_metrics(uni_vals, uni_trades, years, "Unified")
    a_uni, _ = calc_alpha_beta(
        uni_vals, spy_bh.reindex(uni_vals.index, method="ffill"))
    m_uni["alpha"] = a_uni

    # ── 3. Comparison table ──────────────────────────────────────────────
    rows = [
        ("CAGR",          "cagr",          lambda v: f"{v:+.4%}"),
        ("Sharpe",        "sharpe",        lambda v: f"{v:.6f}"),
        ("Sortino",       "sortino",       lambda v: f"{v:.6f}"),
        ("Max Drawdown",  "max_dd",        lambda v: f"{v:.6%}"),
        ("Final Value",   "final_value",   lambda v: f"${v:,.2f}"),
        ("Alpha vs SPY",  "alpha",         lambda v: f"{v:+.4%}"),
        ("Total Trades",  "n_trades",      lambda v: f"{v}"),
        ("Win Rate",      "win_rate",      lambda v: f"{v:.6%}"),
        ("Profit Factor", "profit_factor",
         lambda v: f"{v:.6f}" if np.isfinite(v) else "inf"),
        ("Avg Trade Ret", "avg_trade_ret", lambda v: f"{v:+.6%}"),
    ]

    c0, c1, c2, c3 = 16, 18, 18, 18
    print(f"\n{'=' * (c0+c1+c2+c3+4)}")
    print(f"  {'Metric':<{c0}} {'Original':<{c1}} {'Unified':<{c2}} {'Delta':<{c3}}")
    print(f"  {'─'*(c0-1)} {'─'*(c1-1)} {'─'*(c2-1)} {'─'*(c3-1)}")

    all_match = True
    for label, key, fmt in rows:
        vo, vu = m_orig[key], m_uni[key]
        if key == "n_trades":
            d = int(vu) - int(vo)
            ds = f"{d:+d}"
            ok = d == 0
        else:
            d = vu - vo
            ds = f"{d:+.10f}"
            ok = abs(d) < 1e-6
        if not ok:
            all_match = False
        tag = " ✓" if ok else " ✗"
        print(f"  {label:<{c0}} {fmt(vo):<{c1}} {fmt(vu):<{c2}} {ds:<{c3}}{tag}")

    # ── 4. Day-by-day & trade-by-trade diffs ─────────────────────────────
    day_diff = np.abs(orig_vals.values - uni_vals.values)
    print(f"\n  Portfolio value — day-by-day:")
    print(f"    Max  |Δ|: ${day_diff.max():,.6f}")
    print(f"    Mean |Δ|: ${day_diff.mean():,.6f}")

    if len(orig_trades) == len(uni_trades):
        t_diff = [abs(a - b) for a, b in zip(orig_trades, uni_trades)]
        print(f"\n  Trade returns — trade-by-trade:")
        print(f"    Max  |Δ|: {max(t_diff):.12f}")
        print(f"    Count:    {len(orig_trades)} == {len(uni_trades)} ✓")
    else:
        print(f"\n  Trade count MISMATCH: {len(orig_trades)} vs {len(uni_trades)} ✗")
        all_match = False

    # ── 5. Verdict ───────────────────────────────────────────────────────
    print(f"\n{'=' * (c0+c1+c2+c3+4)}")
    if all_match and day_diff.max() < 0.01:
        print("  VERDICT:  EXACT MATCH ✓")
        print("  The unified backtester reproduces the original results.")
    elif day_diff.max() < 1.0:
        print("  VERDICT:  NEAR MATCH (rounding diffs < $1)")
    else:
        print("  VERDICT:  MISMATCH ✗  — investigate the differences above.")
    print(f"{'=' * (c0+c1+c2+c3+4)}")

    # ── 6. Slot configuration summary ────────────────────────────────────
    print(f"\n  Validation slot config (matches backtest_ml.py):")
    print(f"    ml_medium: {SLOT_ML_ONLY.strategy_slots['ml_medium']} slots  "
          f"|  flex: {SLOT_ML_ONLY.flex_slots}  |  max: {SLOT_ML_ONLY.max_positions}")

    print(f"\n  Multi-strategy slot config (for future use):")
    for sname, n in SLOT_MULTI.strategy_slots.items():
        status = "active" if sname == "ml_medium" else "placeholder"
        print(f"    {sname:<20} {n} slot(s)  ({status})")
    print(f"    {'flex pool':<20} {SLOT_MULTI.flex_slots} slot(s)")
    print(f"    {'total max':<20} {SLOT_MULTI.max_positions}")

    print(f"\n  Runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    validate()
