"""
ML Signal Server (v6 — Dual-Ensemble 40/60 Blend)
=================================================
FastAPI service that loads the V6 dual LGBM+XGB ensemble (87 features)
and serves trading signals to the JS trading bot.

V4 changes from v2:
  - Model trained on cross-sectional rank target (top 20% of S&P 500)
  - Signal selection uses top-N ranking (top 5) instead of probability threshold
  - 4 new cross-sectional rank features: vol_rank_20d, momentum_rank_60d,
    rsi_rank, dist_sma50_rank

Features computed at runtime:
  - 18 per-symbol technical features (returns, vol, RSI, MACD, BB, SMA, OBV)
  - 8 long-timeframe technicals (126d/252d returns, 52w high/low, SMA200 slope, etc.)
  - 22 fundamental features (from FMP parquet cache)
  - 13 cross-asset features (SPY, TLT, VIXY returns + calendar)
  - 6 FRED macro features (yield curve, HY spread, DXY)
  - 4 cross-asset ratio features (HYG/LQD, CPER/GLD)
  - 9 cross-sectional rank features (return/vol/momentum/RSI/SMA ranks)
  - 2 fundamental cross-sectional (PE/PS vs universe median)
  - 2 always-NaN (eps_revision_30d, revenue_revision_30d)

Endpoints
---------
GET /health              — model status, last refresh time, staleness flag
GET /signals             — all universe signals sorted by probability desc
GET /signal/{symbol}     — signal for one symbol

Run with:
    python3 signal_server.py
"""

import asyncio
import json
import logging
import os
import ssl
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Paths & env ───────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent
DATA_DIR           = BASE_DIR / "data"
MODEL_FILE         = DATA_DIR / "model.lgb"
EARNINGS_CACHE_DIR = DATA_DIR / "earnings_cache"
FUND_CACHE_DIR     = DATA_DIR / "fundamentals_cache"
load_dotenv(BASE_DIR.parent / ".env")
FMP_API_KEY        = os.getenv("FMP_API_KEY", "")

# SSL context for FMP API (Mac Python sometimes lacks certs)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger("signal_server")

# ── Universe (loaded from sp500_universe module) ─────────────────────────────
from sp500_universe import get_stock_symbols, get_full_universe, get_cross_asset, get_all_symbols
from unified_backtester import SYMBOL_SECTOR

STOCK_SYMBOLS = get_stock_symbols()
UNIVERSE = get_full_universe()
CROSS_ASSET = get_cross_asset()
ALL_SYMBOLS = get_all_symbols()

# Feature columns in exact training order (83 features for V5c)
FEATURE_COLS = [
    "ret_5d","ret_10d","ret_20d","ret_60d","ret_120d",
    "vol_10d","vol_20d","vol_60d",
    "rsi_14","macd_line","macd_signal","bb_position",
    "dist_sma50","dist_sma200",
    "new_high_20d","new_high_50d",
    "vol_ratio_20d","obv_trend_20d",
    # Long-timeframe technicals
    "ret_126d","ret_252d","dist_52w_high","dist_52w_low",
    "sma200_slope","max_dd_6m","consec_up_months","consec_down_months",
    # Fundamental features (NaN for ETFs — LightGBM handles natively)
    "revenue_growth_yoy","eps_growth_yoy","revenue_growth_qoq",
    "gross_margin","operating_margin","net_margin","margin_trend_4q",
    "pe_ratio","ps_ratio","debt_to_equity","current_ratio","roe","roa",
    "days_since_earnings","eps_surprise_last",
    "insider_buy_ratio_90d","insider_net_shares_90d",
    # Cross-asset
    "spy_ret_5d","tlt_ret_5d","spy_ret_10d","tlt_ret_10d",
    "spy_ret_20d","tlt_ret_20d","spy_ret_60d","tlt_ret_60d",
    "spy_ret_120d","tlt_ret_120d",
    "vixy_level","vixy_ret_5d","vixy_ret_20d",
    "day_of_week","month","quarter",
    # FRED macro
    "yield_curve_10y2y","yield_curve_30d_change",
    "hy_spread","hy_spread_30d_change",
    "dxy_level","dxy_30d_change",
    # Cross-asset ratios
    "hyg_lqd_ratio","hyg_lqd_30d_change",
    "copper_gold_ratio","copper_gold_30d_change",
    # Cross-sectional ranks
    "return_rank_3m","return_rank_6m","return_rank_12m",
    "vol_rank_3m","vol_126d","vol_rank_6m",
    # Fundamental cross-sectional
    "pe_vs_universe_median","ps_vs_universe_median",
    # V4 cross-sectional rank features
    "vol_rank_20d","momentum_rank_60d","rsi_rank","dist_sma50_rank",
    # Sector-relative features (v5d)
    "ret_10d_vs_sector","ret_20d_vs_sector","rsi_14_vs_sector","vol_20d_vs_sector",
]

SECTOR_FEATURE_COLS = [
    "ret_10d_vs_sector","ret_20d_vs_sector","rsi_14_vs_sector","vol_20d_vs_sector",
]

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy","eps_growth_yoy","revenue_growth_qoq",
    "gross_margin","operating_margin","net_margin","margin_trend_4q",
    "pe_ratio","ps_ratio","debt_to_equity","current_ratio","roe","roa",
    "days_since_earnings","eps_surprise_last",
    "insider_buy_ratio_90d","insider_net_shares_90d",
]

# Warmup: 252d for 52w high/low + SMA200 + buffer → 550 calendar days
WARMUP_DAYS     = 550
REFRESH_MINUTES = 15
PROB_THRESHOLD  = 0.55   # kept for backwards compat, but V4 uses top-N ranking
TOP_N_PICKS     = 5      # V4: BUY signal for top 5 stocks by probability
ET              = ZoneInfo("America/New_York")


# ── FMP helpers ──────────────────────────────────────────────────────────────

def _fmp_fetch(endpoint: str, params: str = "") -> list:
    """Fetch from FMP /stable/ endpoint."""
    url = f"https://financialmodelingprep.com/stable/{endpoint}?{params}&apikey={FMP_API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list):
            return data
        return [data] if isinstance(data, dict) and "error" not in data else []
    except Exception as exc:
        log.warning("FMP fetch failed %s: %s", endpoint, exc)
        return []


def _parse_earnings(data: list) -> pd.DataFrame:
    rows = []
    for item in data:
        try:
            if item.get("epsActual") is None:
                continue
            date = pd.Timestamp(item["date"])
            actual = float(item["epsActual"])
            estimate = item.get("epsEstimated")
            if estimate is None:
                continue
            estimate = float(estimate)
            surprise = float(np.clip((actual - estimate) / abs(estimate), -2.0, 2.0)) \
                       if abs(estimate) > 1e-9 else 0.0
            rows.append({"date": date, "earnings_surprise": surprise,
                         "eps_actual": actual, "eps_estimated": estimate})
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame(columns=["date", "earnings_surprise", "eps_actual", "eps_estimated"])
    return pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _fetch_earnings(symbol: str) -> pd.DataFrame:
    EARNINGS_CACHE_DIR.mkdir(exist_ok=True)
    cache_file = EARNINGS_CACHE_DIR / f"{symbol}.json"
    if cache_file.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age < 7:
            try:
                with open(cache_file) as f:
                    return _parse_earnings(json.load(f))
            except Exception:
                pass
    if not FMP_API_KEY:
        return pd.DataFrame(columns=["date", "earnings_surprise", "eps_actual", "eps_estimated"])
    url = f"https://financialmodelingprep.com/stable/earnings?symbol={symbol}&apikey={FMP_API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list):
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return _parse_earnings(data)
    except Exception as exc:
        log.warning("Earnings fetch failed for %s: %s", symbol, exc)
    return pd.DataFrame(columns=["date", "earnings_surprise", "eps_actual", "eps_estimated"])


def _load_all_earnings() -> dict:
    earnings = {}
    if not FMP_API_KEY:
        log.warning("FMP_API_KEY not set — earnings features will be zero")
        return earnings
    for sym in STOCK_SYMBOLS:
        earnings[sym] = _fetch_earnings(sym)
    loaded = sum(1 for v in earnings.values() if not v.empty)
    log.info("Earnings loaded: %d/%d symbols", loaded, len(STOCK_SYMBOLS))
    return earnings


def _load_fundamental_parquets() -> dict:
    """Load fundamentals_*.parquet files into a dict of DataFrames (float32 for memory)."""
    files = {
        "income": DATA_DIR / "fundamentals_income.parquet",
        "ratios": DATA_DIR / "fundamentals_ratios.parquet",
        "metrics": DATA_DIR / "fundamentals_metrics.parquet",
        "insiders": DATA_DIR / "fundamentals_insiders.parquet",
    }
    data = {}
    for key, path in files.items():
        if path.exists():
            df = pd.read_parquet(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            if "filing_date" in df.columns:
                df["filing_date"] = pd.to_datetime(df["filing_date"])
            # Downcast numeric columns to float32 to save memory
            float_cols = df.select_dtypes(include=["float64"]).columns
            df[float_cols] = df[float_cols].astype(np.float32)
            data[key] = df
        else:
            log.warning("Missing %s — fundamental features will be NaN", path.name)
            data[key] = pd.DataFrame()
    return data


def _load_fred_macro() -> pd.DataFrame:
    """Load FRED macro data from parquet (float32 for memory)."""
    macro_file = DATA_DIR / "macro_fred.parquet"
    if macro_file.exists():
        macro = pd.read_parquet(macro_file)
        macro.index = pd.to_datetime(macro.index).tz_localize(None)
        float_cols = macro.select_dtypes(include=["float64"]).columns
        macro[float_cols] = macro[float_cols].astype(np.float32)
        return macro
    log.warning("macro_fred.parquet not found — macro features will be NaN")
    return pd.DataFrame()


# ── Server state ──────────────────────────────────────────────────────────────
class State:
    model:          object                 = None    # sector LGBM (or legacy LGBM)
    model_xgb:      object                 = None    # sector XGB (or legacy XGB/RF)
    imputer:        object                 = None    # sector imputer
    model_base_lgbm: object                = None    # base LGBM (no sector features)
    model_base_xgb:  object                = None    # base XGB (no sector features)
    imputer_base:    object                = None    # base imputer
    blend_mode:      str                   = "single"  # "dual" or "single"
    cache:          list                   = []
    last_update:    Optional[datetime]     = None
    is_stale:       bool                   = True
    refresh_task:   Optional[asyncio.Task] = None
    earnings_cache: dict                   = {}
    fund_data:      dict                   = {}
    macro_data:     pd.DataFrame           = None
    # Breadth & regime
    current_breadth: float                 = 50.0
    current_breadth_regime: str            = "BROAD"
    ml_mode:        str                    = "A"   # A=BROAD, B=NARROW, C=TRANSITION

state = State()


# ── Feature computation ──────────────────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(close: pd.Series):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    line   = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


def _compute_bb_position(close: pd.Series, window: int = 20) -> pd.Series:
    sma   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _compute_obv_trend(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    direction = np.sign(close.diff())
    obv       = (direction * volume).fillna(0).cumsum()
    return obv.rolling(window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )


def _consec_streak(returns: pd.Series, positive: bool = True) -> pd.Series:
    result = pd.Series(0, index=returns.index, dtype=float)
    streak = 0
    for i in range(len(returns)):
        r = returns.iloc[i]
        if np.isnan(r):
            streak = 0
        elif (positive and r > 0) or (not positive and r < 0):
            streak += 1
        else:
            streak = 0
        result.iloc[i] = streak
    return result


# ── Market breadth & regime ──────────────────────────────────────────────────

BREADTH_LOOKBACK = 60  # trading days

def compute_market_breadth(raw: dict, spy_close: pd.Series, lookback_days: int = BREADTH_LOOKBACK) -> float:
    """
    Returns % of S&P 500 stocks that beat SPY over lookback period.
    Uses close prices only through D-1 (no lookahead).
    """
    if len(spy_close) < lookback_days + 1:
        return 50.0  # default neutral

    spy_ret = (spy_close.iloc[-1] / spy_close.iloc[-lookback_days - 1]) - 1.0
    beat_count = 0
    total_count = 0

    for sym in STOCK_SYMBOLS:
        df = raw.get(sym, pd.DataFrame())
        if df.empty or "close" not in df.columns:
            continue
        c = df["close"].dropna()
        if len(c) < lookback_days + 1:
            continue
        stock_ret = (c.iloc[-1] / c.iloc[-lookback_days - 1]) - 1.0
        total_count += 1
        if stock_ret > spy_ret:
            beat_count += 1

    if total_count == 0:
        return 50.0
    return (beat_count / total_count) * 100.0


def classify_breadth_regime(breadth: float) -> str:
    """
    Classify market breadth into regime using historical tercile boundaries.
    breadth < 40: NARROW (concentrated market)
    breadth 40-55: TRANSITION
    breadth > 55: BROAD (diversified market)
    """
    if breadth < 40:
        return "NARROW"
    elif breadth <= 55:
        return "TRANSITION"
    return "BROAD"


def get_ml_mode(breadth_regime: str) -> str:
    """Map breadth regime to ML mode: A=BROAD, B=NARROW, C=TRANSITION."""
    return {"BROAD": "A", "NARROW": "B", "TRANSITION": "C"}.get(breadth_regime, "A")


def _symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-symbol technical features (original + long-timeframe)."""
    c = df["close"]
    v = df["volume"]
    feat = pd.DataFrame(index=df.index)

    # Original technical features
    for n in [5, 10, 20, 60, 120]:
        feat[f"ret_{n}d"] = c.pct_change(n)

    daily_ret = c.pct_change()
    for n in [10, 20, 60]:
        feat[f"vol_{n}d"] = daily_ret.rolling(n).std() * np.sqrt(252)

    feat["rsi_14"] = _compute_rsi(c, 14)
    feat["macd_line"], feat["macd_signal"] = _compute_macd(c)
    feat["bb_position"] = _compute_bb_position(c, 20)

    for n in [50, 200]:
        sma = c.rolling(n).mean()
        feat[f"dist_sma{n}"] = (c - sma) / sma

    feat["new_high_20d"] = (c == c.rolling(20).max()).astype(int)
    feat["new_high_50d"] = (c == c.rolling(50).max()).astype(int)
    feat["vol_ratio_20d"] = v / v.rolling(20).mean().replace(0, np.nan)
    feat["obv_trend_20d"] = _compute_obv_trend(c, v, 20)

    # Long-timeframe technicals
    feat["ret_126d"] = c.pct_change(126)
    feat["ret_252d"] = c.pct_change(252)

    high_252 = df["high"].rolling(252, min_periods=252).max()
    low_252  = df["low"].rolling(252, min_periods=252).min()
    feat["dist_52w_high"] = (c - high_252) / high_252.replace(0, np.nan)
    feat["dist_52w_low"]  = (c - low_252) / low_252.replace(0, np.nan)

    sma200 = c.rolling(200).mean()
    sma200_30ago = sma200.shift(30)
    feat["sma200_slope"] = (sma200 - sma200_30ago) / sma200_30ago.replace(0, np.nan)

    # Max drawdown 6 months
    arr = c.values
    max_dd = pd.Series(np.nan, index=c.index)
    for i in range(126, len(arr)):
        seg = arr[i-126:i+1]
        valid = seg[~np.isnan(seg)]
        if len(valid) >= 10:
            peak = np.maximum.accumulate(valid)
            dd = (valid - peak) / np.where(peak > 0, peak, np.nan)
            max_dd.iloc[i] = np.nanmin(dd)
    feat["max_dd_6m"] = max_dd

    monthly_ret = c.pct_change(21)
    feat["consec_up_months"] = _consec_streak(monthly_ret, positive=True)
    feat["consec_down_months"] = _consec_streak(monthly_ret, positive=False)

    # vol_126d (needed for cross-sectional ranking later)
    feat["vol_126d"] = daily_ret.rolling(126, min_periods=126).std() * np.sqrt(252)

    return feat


def _compute_fundamental_features_for_symbol(symbol: str, today: pd.Timestamp) -> dict:
    """
    Compute fundamental feature values for one symbol on the latest date.
    Uses the cached parquet data. Returns a dict of feature_name → value (or NaN).
    """
    fund = state.fund_data
    result = {col: np.nan for col in FUNDAMENTAL_FEATURE_COLS}

    if not fund or symbol not in STOCK_SYMBOLS:
        return result

    income = fund.get("income", pd.DataFrame())
    ratios = fund.get("ratios", pd.DataFrame())
    metrics = fund.get("metrics", pd.DataFrame())
    insiders = fund.get("insiders", pd.DataFrame())

    # Income statement: latest filed quarter
    sym_inc = income[income["symbol"] == symbol].copy() if len(income) > 0 else pd.DataFrame()
    if len(sym_inc) > 0 and "filing_date" in sym_inc.columns:
        sym_inc = sym_inc.dropna(subset=["filing_date"])
        sym_inc = sym_inc[sym_inc["filing_date"] <= today].sort_values("filing_date")
        if len(sym_inc) > 0:
            latest = sym_inc.iloc[-1]
            rev = latest.get("revenue")
            gp = latest.get("gross_profit")
            oi = latest.get("operating_income")
            ni = latest.get("net_income")

            if rev and abs(rev) > 0:
                result["gross_margin"] = gp / rev if gp is not None else np.nan
                result["operating_margin"] = oi / rev if oi is not None else np.nan
                result["net_margin"] = ni / rev if ni is not None else np.nan

            # YoY growth (4 quarters back)
            if len(sym_inc) >= 5:
                prev = sym_inc.iloc[-5]
                prev_rev = prev.get("revenue")
                if prev_rev and abs(prev_rev) > 1e-6 and rev is not None:
                    result["revenue_growth_yoy"] = (rev - prev_rev) / abs(prev_rev)
                prev_eps = prev.get("eps_diluted") or prev.get("eps")
                cur_eps = latest.get("eps_diluted") or latest.get("eps")
                if prev_eps and abs(prev_eps) > 1e-6 and cur_eps is not None:
                    result["eps_growth_yoy"] = (cur_eps - prev_eps) / abs(prev_eps)

            # QoQ growth
            if len(sym_inc) >= 2:
                prev = sym_inc.iloc[-2]
                prev_rev = prev.get("revenue")
                if prev_rev and abs(prev_rev) > 1e-6 and rev is not None:
                    result["revenue_growth_qoq"] = (rev - prev_rev) / abs(prev_rev)

            # Margin trend (slope of net_margin over last 4 quarters)
            if len(sym_inc) >= 4:
                margins = []
                for idx in range(-4, 0):
                    r = sym_inc.iloc[idx]
                    r_rev = r.get("revenue")
                    r_ni = r.get("net_income")
                    if r_rev and abs(r_rev) > 0 and r_ni is not None:
                        margins.append(r_ni / r_rev)
                if len(margins) >= 3:
                    slope = np.polyfit(range(len(margins)), margins, 1)[0]
                    result["margin_trend_4q"] = slope

    # Ratios
    sym_rat = ratios[ratios["symbol"] == symbol] if len(ratios) > 0 else pd.DataFrame()
    if len(sym_rat) > 0:
        sym_rat = sym_rat[sym_rat["date"] <= today].sort_values("date")
        if len(sym_rat) > 0:
            r = sym_rat.iloc[-1]
            result["pe_ratio"] = np.clip(r.get("pe_ratio", np.nan), -100, 500)
            result["ps_ratio"] = np.clip(r.get("ps_ratio", np.nan), 0, 100)
            result["debt_to_equity"] = np.clip(r.get("debt_to_equity", np.nan), 0, 50)
            result["current_ratio"] = np.clip(r.get("current_ratio", np.nan), 0, 50)

    # Key metrics
    sym_met = metrics[metrics["symbol"] == symbol] if len(metrics) > 0 else pd.DataFrame()
    if len(sym_met) > 0:
        sym_met = sym_met[sym_met["date"] <= today].sort_values("date")
        if len(sym_met) > 0:
            m = sym_met.iloc[-1]
            result["roe"] = np.clip(m.get("roe", np.nan), -5, 5)
            result["roa"] = np.clip(m.get("roa", np.nan), -5, 5)

    # Earnings timing
    earnings = state.earnings_cache.get(symbol, pd.DataFrame())
    if len(earnings) > 0:
        reported = earnings[earnings["date"] <= today].sort_values("date")
        if len(reported) > 0:
            last_earn = reported.iloc[-1]
            result["days_since_earnings"] = (today - last_earn["date"]).days

            # EPS surprise
            actual = last_earn.get("eps_actual")
            est = last_earn.get("eps_estimated")
            if actual is not None and est is not None and abs(est) > 1e-9:
                result["eps_surprise_last"] = np.clip((actual - est) / abs(est), -2.0, 2.0)

    # Always NaN (no daily estimate snapshots)
    result["eps_revision_30d"] = np.nan
    result["revenue_revision_30d"] = np.nan

    # Insider activity (90-day window)
    sym_ins = insiders[insiders["symbol"] == symbol] if len(insiders) > 0 else pd.DataFrame()
    if len(sym_ins) > 0:
        cutoff = today - pd.Timedelta(days=90)
        recent = sym_ins[(sym_ins["date"] >= cutoff) & (sym_ins["date"] <= today)]
        if len(recent) > 0:
            buys = recent["is_buy"].sum()
            total = len(recent)
            result["insider_buy_ratio_90d"] = buys / total if total > 0 else np.nan
            buy_shares = recent.loc[recent["is_buy"] == 1, "shares"].sum()
            sell_shares = recent.loc[recent["is_buy"] == 0, "shares"].sum()
            net = buy_shares - sell_shares
            result["insider_net_shares_90d"] = float(np.sign(net) * np.log1p(abs(net)))

    return result


def _fetch_bars_batch() -> dict[str, pd.DataFrame]:
    """Batch-download WARMUP_DAYS of history for all symbols.
    Splits into chunks of 100 to avoid yfinance/Yahoo timeouts with 500+ symbols."""
    start = (datetime.today() - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    CHUNK_SIZE = 100

    raw = {}
    for i in range(0, len(ALL_SYMBOLS), CHUNK_SIZE):
        chunk = ALL_SYMBOLS[i:i + CHUNK_SIZE]
        log.info("  Downloading bars chunk %d/%d (%d symbols) ...",
                 i // CHUNK_SIZE + 1,
                 (len(ALL_SYMBOLS) + CHUNK_SIZE - 1) // CHUNK_SIZE,
                 len(chunk))
        try:
            batch = yf.download(
                chunk, start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
            )
        except Exception as exc:
            log.error("  yfinance chunk %d failed: %s", i // CHUNK_SIZE + 1, exc)
            continue

        for sym in chunk:
            try:
                if isinstance(batch.columns, pd.MultiIndex):
                    df = batch.xs(sym, axis=1, level=1).copy()
                else:
                    df = batch.copy()
                df.columns = df.columns.str.lower()
                df = df[["open", "high", "low", "close", "volume"]].dropna(how="all")
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "date"
                # Use float32 to save memory with 500+ symbols
                for c in ["open", "high", "low", "close"]:
                    df[c] = df[c].astype(np.float32)
                raw[sym] = df.sort_index()
            except Exception:
                raw[sym] = pd.DataFrame()

    log.info("  Downloaded bars for %d/%d symbols", len([s for s in raw if len(raw[s]) > 0]), len(ALL_SYMBOLS))
    return raw


def build_signals(raw: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Compute all 80 features for the latest day, run the model,
    return signal dicts.
    """
    spy_df = raw.get("SPY", pd.DataFrame())
    if spy_df.empty:
        raise RuntimeError("SPY data unavailable — cannot build signals")

    date_index = spy_df.index
    today = date_index[-1]

    def aligned_close(sym):
        df = raw.get(sym, pd.DataFrame())
        return df["close"].reindex(date_index) if not df.empty else pd.Series(np.nan, index=date_index)

    spy_c  = aligned_close("SPY")
    vixy_c = aligned_close("VIXY")
    tlt_c  = aligned_close("TLT")
    hyg_c  = aligned_close("HYG")
    lqd_c  = aligned_close("LQD")
    cper_c = aligned_close("CPER")
    gld_c  = aligned_close("GLD")

    # ── Cross-asset + calendar features ────────────────────────────────────
    cross = pd.DataFrame(index=date_index)
    for n in [5, 10, 20, 60, 120]:
        cross[f"spy_ret_{n}d"] = spy_c.pct_change(n)
        cross[f"tlt_ret_{n}d"] = tlt_c.pct_change(n)
    cross["vixy_level"]  = vixy_c
    cross["vixy_ret_5d"] = vixy_c.pct_change(5)
    cross["vixy_ret_20d"]= vixy_c.pct_change(20)
    cross["day_of_week"] = date_index.dayofweek
    cross["month"]       = date_index.month
    cross["quarter"]     = date_index.quarter

    # ── FRED macro features ────────────────────────────────────────────────
    macro = state.macro_data
    if macro is not None and len(macro) > 0:
        macro_aligned = macro.reindex(date_index, method="ffill").bfill()
        yc = macro_aligned.get("T10Y2Y", pd.Series(np.nan, index=date_index))
        cross["yield_curve_10y2y"] = yc
        cross["yield_curve_30d_change"] = yc - yc.shift(30)

        hy = macro_aligned.get("BAMLH0A0HYM2", pd.Series(np.nan, index=date_index))
        cross["hy_spread"] = hy
        cross["hy_spread_30d_change"] = hy - hy.shift(30)

        dxy = macro_aligned.get("DTWEXBGS", pd.Series(np.nan, index=date_index))
        cross["dxy_level"] = dxy
        cross["dxy_30d_change"] = (dxy - dxy.shift(30)) / dxy.shift(30).replace(0, np.nan)
    else:
        for col in ["yield_curve_10y2y","yield_curve_30d_change",
                     "hy_spread","hy_spread_30d_change",
                     "dxy_level","dxy_30d_change"]:
            cross[col] = np.nan

    # ── Cross-asset ratio features ─────────────────────────────────────────
    hyg_lqd = hyg_c / lqd_c.replace(0, np.nan)
    cross["hyg_lqd_ratio"] = hyg_lqd
    cross["hyg_lqd_30d_change"] = (hyg_lqd - hyg_lqd.shift(30)) / hyg_lqd.shift(30).replace(0, np.nan)

    copper_gold = cper_c / gld_c.replace(0, np.nan)
    cross["copper_gold_ratio"] = copper_gold
    cross["copper_gold_30d_change"] = (copper_gold - copper_gold.shift(30)) / copper_gold.shift(30).replace(0, np.nan)

    # ── Compute per-symbol features ────────────────────────────────────────
    sym_features = {}  # sym → dict of features for today
    sym_ret_60d = {}   # for cross-sectional ranking
    sym_ret_126d = {}
    sym_ret_252d = {}
    sym_vol_60d = {}
    sym_vol_126d = {}
    sym_vol_20d = {}   # V4 new rank features
    sym_rsi_14 = {}
    sym_dist_sma50 = {}
    sym_ret_10d = {}   # V5d sector-relative features
    sym_ret_20d = {}
    sym_pe = {}
    sym_ps = {}

    for sym in UNIVERSE:
        df = raw.get(sym, pd.DataFrame())
        if df.empty:
            log.warning("No data for %s — skipping", sym)
            continue

        df_aligned = df.reindex(date_index)
        feat = _symbol_features(df_aligned)
        feat = feat.join(cross, how="left")

        # Fundamental features
        fund_feats = _compute_fundamental_features_for_symbol(sym, today)
        for col, val in fund_feats.items():
            feat[col] = val  # scalar broadcast to all rows

        # Get the latest row
        # For non-fundamental columns, require them to be non-NaN
        non_fund = [c for c in FEATURE_COLS if c not in FUNDAMENTAL_FEATURE_COLS
                    and c not in ("pe_vs_universe_median", "ps_vs_universe_median",
                                  "return_rank_3m","return_rank_6m","return_rank_12m",
                                  "vol_rank_3m","vol_rank_6m",
                                  "ret_10d_vs_sector","ret_20d_vs_sector",
                                  "rsi_14_vs_sector","vol_20d_vs_sector")]
        last_idx = feat.index[-1]
        row = feat.loc[last_idx]

        # Check that core features are not NaN
        core_missing = sum(1 for c in non_fund if c in row.index and pd.isna(row[c]))
        if core_missing > 10:
            log.warning("Too many NaN features for %s (%d missing) — skipping", sym, core_missing)
            continue

        sym_features[sym] = {c: row.get(c, np.nan) for c in FEATURE_COLS
                             if c not in ("return_rank_3m","return_rank_6m","return_rank_12m",
                                         "vol_rank_3m","vol_rank_6m",
                                         "pe_vs_universe_median","ps_vs_universe_median",
                                         "vol_rank_20d","momentum_rank_60d",
                                         "rsi_rank","dist_sma50_rank",
                                         "ret_10d_vs_sector","ret_20d_vs_sector",
                                         "rsi_14_vs_sector","vol_20d_vs_sector")}
        # Store values for cross-sectional ranking
        sym_ret_60d[sym] = row.get("ret_60d", np.nan)
        sym_ret_126d[sym] = row.get("ret_126d", np.nan)
        sym_ret_252d[sym] = row.get("ret_252d", np.nan)
        sym_vol_60d[sym] = row.get("vol_60d", np.nan)
        sym_vol_126d[sym] = row.get("vol_126d", np.nan)
        sym_vol_20d[sym] = row.get("vol_20d", np.nan)
        sym_rsi_14[sym] = row.get("rsi_14", np.nan)
        sym_ret_10d[sym] = row.get("ret_10d", np.nan)
        sym_ret_20d[sym] = row.get("ret_20d", np.nan)
        sym_dist_sma50[sym] = row.get("dist_sma50", np.nan)
        sym_pe[sym] = fund_feats.get("pe_ratio", np.nan)
        sym_ps[sym] = fund_feats.get("ps_ratio", np.nan)

    if not sym_features:
        raise RuntimeError("No valid feature rows produced")

    # ── Market breadth computation ────────────────────────────────────────
    breadth = compute_market_breadth(raw, spy_c)
    breadth_regime = classify_breadth_regime(breadth)
    ml_mode = get_ml_mode(breadth_regime)
    state.current_breadth = breadth
    state.current_breadth_regime = breadth_regime
    state.ml_mode = ml_mode
    log.info("Market breadth: %.1f%% → %s regime → ML mode %s",
             breadth, breadth_regime, ml_mode)

    # ── Cross-sectional rank features ──────────────────────────────────────
    symbols = list(sym_features.keys())

    def _rank_pct(values_dict):
        vals = pd.Series({s: values_dict.get(s, np.nan) for s in symbols})
        return vals.rank(pct=True)

    rank_3m = _rank_pct(sym_ret_60d)
    rank_6m = _rank_pct(sym_ret_126d)
    rank_12m = _rank_pct(sym_ret_252d)
    vrank_3m = _rank_pct(sym_vol_60d)
    vrank_6m = _rank_pct(sym_vol_126d)

    # PE/PS vs universe median
    pe_vals = pd.Series({s: sym_pe.get(s, np.nan) for s in symbols})
    ps_vals = pd.Series({s: sym_ps.get(s, np.nan) for s in symbols})
    pe_median = pe_vals.median()
    ps_median = ps_vals.median()

    # V4 new cross-sectional rank features
    vrank_20d = _rank_pct(sym_vol_20d)
    mom_rank_60d = _rank_pct(sym_ret_60d)  # same source as return_rank_3m but separate feature
    rsi_rank = _rank_pct(sym_rsi_14)
    dist_sma50_rank = _rank_pct(sym_dist_sma50)

    for sym in symbols:
        sym_features[sym]["return_rank_3m"] = rank_3m.get(sym, np.nan)
        sym_features[sym]["return_rank_6m"] = rank_6m.get(sym, np.nan)
        sym_features[sym]["return_rank_12m"] = rank_12m.get(sym, np.nan)
        sym_features[sym]["vol_rank_3m"] = vrank_3m.get(sym, np.nan)
        sym_features[sym]["vol_rank_6m"] = vrank_6m.get(sym, np.nan)

        pe = sym_pe.get(sym, np.nan)
        ps = sym_ps.get(sym, np.nan)
        sym_features[sym]["pe_vs_universe_median"] = (
            np.clip(pe / pe_median - 1.0, -10, 20) if pd.notna(pe) and pd.notna(pe_median) and abs(pe_median) > 0 else np.nan
        )
        sym_features[sym]["ps_vs_universe_median"] = (
            np.clip(ps / ps_median - 1.0, -1, 10) if pd.notna(ps) and pd.notna(ps_median) and abs(ps_median) > 0 else np.nan
        )

        # V4 rank features
        sym_features[sym]["vol_rank_20d"] = vrank_20d.get(sym, np.nan)
        sym_features[sym]["momentum_rank_60d"] = mom_rank_60d.get(sym, np.nan)
        sym_features[sym]["rsi_rank"] = rsi_rank.get(sym, np.nan)
        sym_features[sym]["dist_sma50_rank"] = dist_sma50_rank.get(sym, np.nan)

    # V5d sector-relative features: each stock's metric minus its sector median
    sector_groups = {}  # sector → list of symbols
    for sym in sym_features:
        sector = SYMBOL_SECTOR.get(sym, "Other")
        sector_groups.setdefault(sector, []).append(sym)

    for src, dst, src_dict in [
        ("ret_10d", "ret_10d_vs_sector", sym_ret_10d),
        ("ret_20d", "ret_20d_vs_sector", sym_ret_20d),
        ("rsi_14", "rsi_14_vs_sector", sym_rsi_14),
        ("vol_20d", "vol_20d_vs_sector", sym_vol_20d),
    ]:
        sector_medians = {}
        for sector, members in sector_groups.items():
            vals = [src_dict.get(s, np.nan) for s in members]
            vals = [v for v in vals if not np.isnan(v)]
            sector_medians[sector] = np.median(vals) if vals else 0.0
        for sym in sym_features:
            sector = SYMBOL_SECTOR.get(sym, "Other")
            val = src_dict.get(sym, np.nan)
            med = sector_medians.get(sector, 0.0)
            sym_features[sym][dst] = val - med if not np.isnan(val) else 0.0

    # ── Build feature matrix and run model ─────────────────────────────────
    rows = []
    sym_order = []
    for sym in symbols:
        row = [sym_features[sym].get(c, np.nan) for c in FEATURE_COLS]
        rows.append(row)
        sym_order.append(sym)

    X = np.array(rows, dtype=np.float64)
    log.info("Feature matrix: %d symbols × %d features", X.shape[0], X.shape[1])

    X_imp = state.imputer.transform(X) if state.imputer is not None else X

    if state.blend_mode == "dual" and state.model_base_lgbm is not None:
        # v6 dual-ensemble 40/60 blend
        # Sector ensemble (87 features) — uses full X
        sect_probs = 0.5 * state.model.predict_proba(X_imp)[:, 1] + \
                     0.5 * state.model_xgb.predict_proba(X_imp)[:, 1]
        # Base ensemble (83 features) — exclude sector columns
        base_cols_mask = [i for i, c in enumerate(FEATURE_COLS) if c not in SECTOR_FEATURE_COLS]
        X_base = X[:, base_cols_mask]
        X_base_imp = state.imputer_base.transform(X_base) if state.imputer_base is not None else X_base
        base_probs = 0.5 * state.model_base_lgbm.predict_proba(X_base_imp)[:, 1] + \
                     0.5 * state.model_base_xgb.predict_proba(X_base_imp)[:, 1]
        probs = 0.4 * base_probs + 0.6 * sect_probs
        log.info("Dual-ensemble 40/60 blend: base=%d feat, sector=%d feat", X_base.shape[1], X.shape[1])
    else:
        # Legacy single-ensemble 50/50 blend
        lgbm_probs = state.model.predict_proba(X)[:, 1]
        xgb_probs = state.model_xgb.predict_proba(X_imp)[:, 1]
        probs = 0.5 * lgbm_probs + 0.5 * xgb_probs

    # ── Regime-adaptive signal generation ─────────────────────────────────
    ml_mode = state.ml_mode

    # Mode A (BROAD): standard ML ensemble ranking over full universe
    ml_scores = {sym: float(prob) for sym, prob in zip(sym_order, probs)}

    # Mode B (NARROW): top 30 by market cap, rank by 3-month momentum
    # Use ret_60d as proxy for 3-month momentum (63 trading days ≈ 60d feature)
    mega_cap_30 = sorted(sym_order, key=lambda s: sym_ret_252d.get(s, 0), reverse=True)[:30]
    # Actually rank by market cap proxy: use 252d volume × price as rough cap proxy
    # Better: just use the top 30 by absolute price level × volume as rough cap
    # For live: we use ret_60d (3-month momentum) to rank within top 30
    mom_scores = {}
    for sym in mega_cap_30:
        mom_scores[sym] = sym_ret_60d.get(sym, 0.0) if pd.notna(sym_ret_60d.get(sym)) else 0.0

    if ml_mode == "A":
        # BROAD: pure ML ranking
        final_scores = ml_scores
        log.info("ML Mode A (BROAD): ranking %d stocks by ML probability", len(final_scores))
    elif ml_mode == "B":
        # NARROW: mega-cap momentum only (top 30 by cap, rank by 3m momentum)
        final_scores = {}
        # Normalize momentum scores to [0, 1]
        mom_vals = list(mom_scores.values())
        mom_min, mom_max = min(mom_vals) if mom_vals else 0, max(mom_vals) if mom_vals else 1
        mom_range = mom_max - mom_min if mom_max > mom_min else 1.0
        for sym in mega_cap_30:
            final_scores[sym] = (mom_scores[sym] - mom_min) / mom_range
        # Include rest of universe with 0 score so they appear in signal list
        for sym in sym_order:
            if sym not in final_scores:
                final_scores[sym] = 0.0
        log.info("ML Mode B (NARROW): ranking top 30 mega-caps by 3m momentum")
    else:
        # TRANSITION: 60% Mode A + 40% Mode B blend
        # Normalize both score sets to [0, 1]
        ml_vals = list(ml_scores.values())
        ml_min, ml_max = min(ml_vals), max(ml_vals)
        ml_range = ml_max - ml_min if ml_max > ml_min else 1.0

        mom_vals = list(mom_scores.values())
        mom_min, mom_max = min(mom_vals) if mom_vals else 0, max(mom_vals) if mom_vals else 1
        mom_range = mom_max - mom_min if mom_max > mom_min else 1.0

        final_scores = {}
        for sym in sym_order:
            ml_norm = (ml_scores.get(sym, 0) - ml_min) / ml_range
            if sym in mega_cap_30:
                mom_norm = (mom_scores.get(sym, 0) - mom_min) / mom_range
                final_scores[sym] = 0.6 * ml_norm + 0.4 * mom_norm
            else:
                final_scores[sym] = 0.6 * ml_norm
        log.info("ML Mode C (TRANSITION): 60%% ML + 40%% momentum blend")

    # Build signals with rank and top-N classification
    signals = []
    for sym in sym_order:
        score = final_scores.get(sym, 0.0)
        signals.append({
            "symbol":      sym,
            "probability": round(score, 4),
            "confidence":  round(float(ml_scores.get(sym, 0.0)), 4),
            "ml_mode":     ml_mode,
        })

    # Sort by final score descending and assign rank
    signals.sort(key=lambda x: x["probability"], reverse=True)
    for i, sig in enumerate(signals):
        rank = i + 1
        sig["rank"] = rank
        sig["is_top_5"] = rank <= TOP_N_PICKS
        sig["signal"] = "BUY" if sig["is_top_5"] else "HOLD"

    return signals


# ── Refresh logic ─────────────────────────────────────────────────────────────

def _is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return open_ <= now <= close_


async def _refresh() -> bool:
    try:
        log.info("Refreshing signals ...")
        t0 = time.perf_counter()
        raw = await asyncio.get_event_loop().run_in_executor(None, _fetch_bars_batch)
        new_signals = await asyncio.get_event_loop().run_in_executor(
            None, build_signals, raw
        )
        state.cache       = new_signals
        state.last_update = datetime.now(ET)
        state.is_stale    = False

        elapsed = time.perf_counter() - t0
        buys = [s for s in new_signals if s["signal"] == "BUY"]
        log.info(
            "Signals refreshed in %.1fs — %d BUY, %d HOLD (%d features) | "
            "Breadth: %.1f%% (%s) → Mode %s",
            elapsed, len(buys), len(new_signals) - len(buys), len(FEATURE_COLS),
            state.current_breadth, state.current_breadth_regime, state.ml_mode,
        )
        top5 = new_signals[:5]
        log.info(
            "Top 5: %s",
            " | ".join(f"{s['symbol']} {s['probability']:.2%}" for s in top5),
        )
        return True

    except Exception as exc:
        log.error("Refresh failed: %s", exc, exc_info=True)
        state.is_stale = True
        return False


async def _background_refresh_loop():
    while True:
        if _is_market_hours():
            await _refresh()
            await asyncio.sleep(REFRESH_MINUTES * 60)
        else:
            await asyncio.sleep(5 * 60)


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Load LGBM model
    lgbm_file = DATA_DIR / "model.lgb"
    if not lgbm_file.exists():
        log.error("LGBM model not found: %s", lgbm_file)
        sys.exit(1)
    log.info("Loading LGBM model from %s ...", lgbm_file)
    state.model = joblib.load(str(lgbm_file))
    n_lgbm = state.model.calibrated_classifiers_[0].estimator.n_features_in_
    log.info("LGBM loaded (calibrated) — %d features", n_lgbm)

    # Load XGBoost model (sector)
    xgb_file = DATA_DIR / "model_xgb.pkl"
    if xgb_file.exists():
        log.info("Loading XGB model from %s ...", xgb_file)
        state.model_xgb = joblib.load(str(xgb_file))
        n_xgb = state.model_xgb.calibrated_classifiers_[0].estimator.n_features_in_
        log.info("XGB loaded (calibrated) — %d features", n_xgb)
    else:
        # Fall back to RF model if XGB not yet trained
        rf_file = DATA_DIR / "model_rf.pkl"
        if rf_file.exists():
            log.info("XGB not found, falling back to RF model from %s ...", rf_file)
            state.model_xgb = joblib.load(str(rf_file))
            n_rf = state.model_xgb.calibrated_classifiers_[0].estimator.n_features_in_
            log.info("RF loaded as fallback — %d features", n_rf)
        else:
            log.warning("No XGB or RF model found — using LGBM only")

    # Load imputer (sector)
    imp_file = DATA_DIR / "imputer.pkl"
    if imp_file.exists():
        state.imputer = joblib.load(str(imp_file))
        log.info("Imputer loaded")

    # Load v6 dual-ensemble base models (40/60 blend)
    base_lgbm_file = DATA_DIR / "model_base_lgbm.pkl"
    base_xgb_file  = DATA_DIR / "model_base_xgb.pkl"
    base_imp_file  = DATA_DIR / "imputer_base.pkl"
    if base_lgbm_file.exists() and base_xgb_file.exists():
        state.model_base_lgbm = joblib.load(str(base_lgbm_file))
        state.model_base_xgb  = joblib.load(str(base_xgb_file))
        state.imputer_base    = joblib.load(str(base_imp_file)) if base_imp_file.exists() else None
        state.blend_mode = "dual"
        n_base = state.model_base_lgbm.calibrated_classifiers_[0].estimator.n_features_in_
        log.info("v6 dual-ensemble loaded — base=%d feat, blend=40/60", n_base)
    else:
        state.blend_mode = "single"
        log.info("Base models not found — using single-ensemble mode")

    # Load fundamental data from parquets
    log.info("Loading fundamental data ...")
    state.fund_data = _load_fundamental_parquets()
    for key, df in state.fund_data.items():
        log.info("  %s: %d rows", key, len(df))

    # Load FRED macro data
    log.info("Loading FRED macro data ...")
    state.macro_data = _load_fred_macro()
    if state.macro_data is not None and len(state.macro_data) > 0:
        log.info("  Macro data: %d rows, cols: %s",
                 len(state.macro_data), list(state.macro_data.columns))

    # Load earnings history
    state.earnings_cache = _load_all_earnings()

    # Verify feature count matches model
    expected = len(FEATURE_COLS)
    log.info("Expected features: %d  |  FEATURE_COLS: %d", expected, len(FEATURE_COLS))

    # Initial fetch
    await _refresh()

    # Start background loop
    state.refresh_task = asyncio.create_task(_background_refresh_loop())
    log.info("Background refresh task started (every %d min during market hours)", REFRESH_MINUTES)

    yield

    if state.refresh_task:
        state.refresh_task.cancel()
        try:
            await state.refresh_task
        except asyncio.CancelledError:
            pass
    log.info("Signal server shut down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ML Trading Signal Server",
    description = "Dual LGBM+XGB ensemble signals for the auto-trader bot (v6, 40/60 blend)",
    version     = "5.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET"],
    allow_headers  = ["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": state.model is not None,
        "model_version": "v6_dual_ensemble_40_60_blend",
        "feature_count": len(FEATURE_COLS),
        "last_update":  state.last_update.isoformat() if state.last_update else None,
        "is_stale":     state.is_stale,
        "cached_signals": len(state.cache),
        "market_open":  _is_market_hours(),
        "current_breadth": round(state.current_breadth, 1),
        "current_regime":  state.current_breadth_regime,
        "ml_mode":         state.ml_mode,
    }


@app.get("/signals")
def get_signals():
    if not state.cache:
        raise HTTPException(status_code=503, detail="Signals not yet available — try again shortly")
    return {
        "signals":     state.cache,
        "last_update": state.last_update.isoformat() if state.last_update else None,
        "is_stale":    state.is_stale,
        "count":       len(state.cache),
        "buy_count":   sum(1 for s in state.cache if s["signal"] == "BUY"),
        "top_5_count": sum(1 for s in state.cache if s.get("is_top_5")),
        "ml_mode":     state.ml_mode,
        "breadth":     round(state.current_breadth, 1),
        "regime":      state.current_breadth_regime,
    }


@app.get("/signal/{symbol}")
def get_signal(symbol: str):
    symbol = symbol.upper()
    if not state.cache:
        raise HTTPException(status_code=503, detail="Signals not yet available — try again shortly")
    match = next((s for s in state.cache if s["symbol"] == symbol), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"{symbol} not found in universe")
    return {
        **match,
        "last_update": state.last_update.isoformat() if state.last_update else None,
        "is_stale":    state.is_stale,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "signal_server:app",
        host      = "0.0.0.0",
        port      = 5001,
        log_level = "info",
        reload    = False,
    )
