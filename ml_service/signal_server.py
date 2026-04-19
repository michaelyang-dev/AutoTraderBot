"""
ML Signal Server (v2 — with Fundamental Features)
===================================================
FastAPI service that loads the trained calibrated LightGBM model (80 features)
and serves trading signals to the JS trading bot.

Features computed at runtime:
  - 18 per-symbol technical features (returns, vol, RSI, MACD, BB, SMA, OBV)
  - 8 long-timeframe technicals (126d/252d returns, 52w high/low, SMA200 slope, etc.)
  - 22 fundamental features (from FMP parquet cache)
  - 13 cross-asset features (SPY, TLT, VIXY returns + calendar)
  - 6 FRED macro features (yield curve, HY spread, DXY)
  - 4 cross-asset ratio features (HYG/LQD, CPER/GLD)
  - 5 cross-sectional rank features (return/vol ranks across universe)
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

# ── Universe ──────────────────────────────────────────────────────────────────
STOCK_SYMBOLS = [
    "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
    "CRM","ORCL","ADBE","CSCO","QCOM","COST","WMT","HD","LOW",
    "LLY","JNJ","ABBV","BAC","GS","MS","CVX","XOM","CAT","DE","BA",
]

UNIVERSE = [
    "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
    "CRM","ORCL","ADBE","CSCO","QCOM","COST","WMT","HD","LOW",
    "LLY","JNJ","ABBV","BAC","GS","MS","CVX","XOM","CAT","DE","BA",
    "XLE","XLF","XLV","XLI","XLK","XLY","XLP","XLU","XLRE","XLB","XLC",
    "EWZ","EWJ","FXI","INDA","EFA","EEM","VGK","VWO","IEFA",
    "GLD","SLV","USO","DBC",
    "TLT","HYG","LQD","IEF","SHY",
    "UUP","CPER",
    "VIXY",
]
CROSS_ASSET = ["SPY", "VIXY", "TLT"]
ALL_SYMBOLS = sorted(set(UNIVERSE + CROSS_ASSET))

# Feature columns in exact training order (80 features)
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
    "days_since_earnings","days_until_earnings","eps_surprise_last",
    "eps_revision_30d","revenue_revision_30d",
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
]

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy","eps_growth_yoy","revenue_growth_qoq",
    "gross_margin","operating_margin","net_margin","margin_trend_4q",
    "pe_ratio","ps_ratio","debt_to_equity","current_ratio","roe","roa",
    "days_since_earnings","days_until_earnings","eps_surprise_last",
    "eps_revision_30d","revenue_revision_30d",
    "insider_buy_ratio_90d","insider_net_shares_90d",
]

# Warmup: 252d for 52w high/low + SMA200 + buffer → 550 calendar days
WARMUP_DAYS     = 550
REFRESH_MINUTES = 15
PROB_THRESHOLD  = 0.55
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
    """Load fundamentals_*.parquet files into a dict of DataFrames."""
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
            data[key] = df
        else:
            log.warning("Missing %s — fundamental features will be NaN", path.name)
            data[key] = pd.DataFrame()
    return data


def _load_fred_macro() -> pd.DataFrame:
    """Load FRED macro data from parquet."""
    macro_file = DATA_DIR / "macro_fred.parquet"
    if macro_file.exists():
        macro = pd.read_parquet(macro_file)
        macro.index = pd.to_datetime(macro.index).tz_localize(None)
        return macro
    log.warning("macro_fred.parquet not found — macro features will be NaN")
    return pd.DataFrame()


# ── Server state ──────────────────────────────────────────────────────────────
class State:
    model:          object                 = None
    model_type:     str                    = "booster"
    cache:          list                   = []
    last_update:    Optional[datetime]     = None
    is_stale:       bool                   = True
    refresh_task:   Optional[asyncio.Task] = None
    earnings_cache: dict                   = {}
    fund_data:      dict                   = {}
    macro_data:     pd.DataFrame           = None

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

        # Days until next earnings
        future = earnings[earnings["date"] > today].sort_values("date")
        if len(future) > 0:
            result["days_until_earnings"] = (future.iloc[0]["date"] - today).days

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
    """Batch-download WARMUP_DAYS of history for all symbols."""
    start = (datetime.today() - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    batch = yf.download(
        ALL_SYMBOLS, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )

    raw = {}
    for sym in ALL_SYMBOLS:
        try:
            if isinstance(batch.columns, pd.MultiIndex):
                df = batch.xs(sym, axis=1, level=1).copy()
            else:
                df = batch.copy()
            df.columns = df.columns.str.lower()
            df = df[["open", "high", "low", "close", "volume"]].dropna(how="all")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "date"
            raw[sym] = df.sort_index()
        except Exception:
            raw[sym] = pd.DataFrame()

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
                                  "vol_rank_3m","vol_rank_6m")]
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
                                         "pe_vs_universe_median","ps_vs_universe_median")}
        # Store values for cross-sectional ranking
        sym_ret_60d[sym] = row.get("ret_60d", np.nan)
        sym_ret_126d[sym] = row.get("ret_126d", np.nan)
        sym_ret_252d[sym] = row.get("ret_252d", np.nan)
        sym_vol_60d[sym] = row.get("vol_60d", np.nan)
        sym_vol_126d[sym] = row.get("vol_126d", np.nan)
        sym_pe[sym] = fund_feats.get("pe_ratio", np.nan)
        sym_ps[sym] = fund_feats.get("ps_ratio", np.nan)

    if not sym_features:
        raise RuntimeError("No valid feature rows produced")

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

    # ── Build feature matrix and run model ─────────────────────────────────
    rows = []
    sym_order = []
    for sym in symbols:
        row = [sym_features[sym].get(c, np.nan) for c in FEATURE_COLS]
        rows.append(row)
        sym_order.append(sym)

    X = np.array(rows, dtype=np.float64)
    log.info("Feature matrix: %d symbols × %d features", X.shape[0], X.shape[1])

    if state.model_type == "calibrated":
        probs = state.model.predict_proba(X)[:, 1]
    else:
        probs = state.model.predict(X)

    signals = []
    for sym, prob in zip(sym_order, probs):
        signals.append({
            "symbol":      sym,
            "probability": round(float(prob), 4),
            "signal":      "BUY" if prob >= PROB_THRESHOLD else "HOLD",
            "confidence":  round(float(prob), 4),
        })

    signals.sort(key=lambda x: x["probability"], reverse=True)
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
            "Signals refreshed in %.1fs — %d BUY, %d HOLD (%d features)",
            elapsed, len(buys), len(new_signals) - len(buys), len(FEATURE_COLS),
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
    if not MODEL_FILE.exists():
        log.error("Model file not found: %s", MODEL_FILE)
        log.error("Run train_model.py first.")
        sys.exit(1)

    log.info("Loading model from %s ...", MODEL_FILE)
    try:
        state.model = joblib.load(str(MODEL_FILE))
        state.model_type = "calibrated"
        n_feat = state.model.calibrated_classifiers_[0].estimator.n_features_in_
        log.info("Model loaded (calibrated/joblib) — %d features", n_feat)
    except Exception:
        state.model = lgb.Booster(model_file=str(MODEL_FILE))
        state.model_type = "booster"
        log.info("Model loaded (native LightGBM) — %d features", state.model.num_feature())

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
    description = "LightGBM signals for the auto-trader bot (v2 with fundamentals)",
    version     = "2.0.0",
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
        "model_version": "v2_fundamentals",
        "feature_count": len(FEATURE_COLS),
        "last_update":  state.last_update.isoformat() if state.last_update else None,
        "is_stale":     state.is_stale,
        "cached_signals": len(state.cache),
        "market_open":  _is_market_hours(),
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
