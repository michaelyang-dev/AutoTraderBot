"""
ML Signal Server
================
FastAPI service that loads the trained LightGBM model and serves trading
signals to the JS trading bot.

Endpoints
---------
GET /health              — model status, last refresh time, staleness flag
GET /signals             — all 37 universe signals sorted by probability desc
GET /signal/{symbol}     — signal for one symbol

Run with:
    python3 signal_server.py
"""

import asyncio
import json
import logging
import os
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Paths & env ───────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent
MODEL_FILE        = BASE_DIR / "data" / "model.lgb"
EARNINGS_CACHE_DIR = BASE_DIR / "data" / "earnings_cache"
load_dotenv(BASE_DIR.parent / ".env")
FMP_API_KEY       = os.getenv("FMP_API_KEY", "")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger("signal_server")

# ── Universe ──────────────────────────────────────────────────────────────────
# Individual stocks that have quarterly earnings (ETFs get zeros)
STOCK_SYMBOLS = [
    "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
]

UNIVERSE = [
    "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
    "XLE","XLF","XLV","XLI","XLK","XLY","XLP","XLU","XLRE","XLB","XLC",
    "EWZ","EWJ","FXI","INDA","EFA","EEM",
    "GLD","SLV","USO","DBC",
    "TLT","HYG","LQD",
    "VIXY",
]
CROSS_ASSET  = ["SPY", "VIXY", "TLT"]
ALL_SYMBOLS  = sorted(set(UNIVERSE + CROSS_ASSET))

# Feature columns in exact training order (positional — model stored as Column_0..35)
FEATURE_COLS = [
    "ret_5d","ret_10d","ret_20d","ret_60d","ret_120d",
    "vol_10d","vol_20d","vol_60d",
    "rsi_14","macd_line","macd_signal","bb_position",
    "dist_sma50","dist_sma200",
    "new_high_20d","new_high_50d",
    "vol_ratio_20d","obv_trend_20d",
    "spy_ret_5d","tlt_ret_5d","spy_ret_10d","tlt_ret_10d",
    "spy_ret_20d","tlt_ret_20d","spy_ret_60d","tlt_ret_60d",
    "spy_ret_120d","tlt_ret_120d",
    "vixy_level","vixy_ret_5d","vixy_ret_20d",
    "day_of_week","month","quarter",
]

# Warmup: longest lookback is SMA200 + ret_120d buffer → 340 calendar days is safe
WARMUP_DAYS     = 340
REFRESH_MINUTES = 15
PROB_THRESHOLD  = 0.55
ET              = ZoneInfo("America/New_York")

# ── Earnings surprise helpers ─────────────────────────────────────────────────

def _parse_earnings(data: list) -> pd.DataFrame:
    """Parse FMP /stable/earnings JSON into DataFrame[date, earnings_surprise]."""
    rows = []
    for item in data:
        try:
            if item.get("epsActual") is None:
                continue
            date     = pd.Timestamp(item["date"])
            actual   = float(item["epsActual"])
            estimate = item.get("epsEstimated")
            if estimate is None:
                continue
            estimate = float(estimate)
            surprise = float(np.clip((actual - estimate) / abs(estimate), -2.0, 2.0)) \
                       if abs(estimate) > 1e-9 else 0.0
            rows.append({"date": date, "earnings_surprise": surprise})
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame(columns=["date", "earnings_surprise"])
    return pd.DataFrame(rows).dropna().sort_values("date").reset_index(drop=True)


def _fetch_earnings(symbol: str) -> pd.DataFrame:
    """Fetch earnings surprises with a 7-day file cache (FMP /stable/earnings)."""
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
        return pd.DataFrame(columns=["date", "earnings_surprise"])

    url = (
        f"https://financialmodelingprep.com/stable/earnings"
        f"?symbol={symbol}&apikey={FMP_API_KEY}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list):
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return _parse_earnings(data)
    except Exception as exc:
        log.warning("Earnings fetch failed for %s: %s", symbol, exc)
    return pd.DataFrame(columns=["date", "earnings_surprise"])


def _load_all_earnings() -> dict:
    """Load earnings history for all individual stock symbols."""
    earnings = {}
    if not FMP_API_KEY:
        log.warning("FMP_API_KEY not set — earnings features will be zero")
        return earnings
    for sym in STOCK_SYMBOLS:
        earnings[sym] = _fetch_earnings(sym)
    loaded = sum(1 for v in earnings.values() if not v.empty)
    log.info("Earnings loaded: %d/%d symbols", loaded, len(STOCK_SYMBOLS))
    return earnings


# ── Server state ──────────────────────────────────────────────────────────────
class State:
    model:          Optional[lgb.Booster]  = None
    cache:          list                   = []
    last_update:    Optional[datetime]     = None
    is_stale:       bool                   = True
    refresh_task:   Optional[asyncio.Task] = None
    earnings_cache: dict                   = {}   # {symbol: DataFrame}

state = State()

# ── Feature computation ───────────────────────────────────────────────────────

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


def _symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all per-symbol technical features. Returns full history."""
    c    = df["close"]
    v    = df["volume"]
    feat = pd.DataFrame(index=df.index)

    for n in [5, 10, 20, 60, 120]:
        feat[f"ret_{n}d"] = c.pct_change(n)

    daily_ret = c.pct_change()
    for n in [10, 20, 60]:
        feat[f"vol_{n}d"] = daily_ret.rolling(n).std() * np.sqrt(252)

    feat["rsi_14"]                  = _compute_rsi(c, 14)
    feat["macd_line"], feat["macd_signal"] = _compute_macd(c)
    feat["bb_position"]             = _compute_bb_position(c, 20)

    for n in [50, 200]:
        sma = c.rolling(n).mean()
        feat[f"dist_sma{n}"] = (c - sma) / sma

    feat["new_high_20d"]  = (c == c.rolling(20).max()).astype(int)
    feat["new_high_50d"]  = (c == c.rolling(50).max()).astype(int)
    feat["vol_ratio_20d"] = v / v.rolling(20).mean().replace(0, np.nan)
    feat["obv_trend_20d"] = _compute_obv_trend(c, v, 20)

    return feat


def _fetch_bars_batch() -> dict[str, pd.DataFrame]:
    """Batch-download WARMUP_DAYS of history for all symbols."""
    start = (datetime.today() - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")

    batch = yf.download(
        ALL_SYMBOLS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
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
    Given raw OHLCV bars, compute the latest-day feature row for every
    universe symbol, run the model, and return a list of signal dicts.
    """
    spy_df  = raw.get("SPY", pd.DataFrame())
    if spy_df.empty:
        raise RuntimeError("SPY data unavailable — cannot build signals")

    date_index = spy_df.index

    def aligned_close(sym):
        df = raw.get(sym, pd.DataFrame())
        return df["close"].reindex(date_index) if not df.empty else pd.Series(np.nan, index=date_index)

    spy_c  = aligned_close("SPY")
    vixy_c = aligned_close("VIXY")
    tlt_c  = aligned_close("TLT")

    # Cross-asset + calendar (same index as SPY)
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

    rows    = []   # feature rows for model
    symbols = []   # corresponding symbol names

    for sym in UNIVERSE:
        df = raw.get(sym, pd.DataFrame())
        if df.empty:
            log.warning("No data for %s — skipping", sym)
            continue

        df_aligned = df.reindex(date_index)
        feat = _symbol_features(df_aligned)
        feat = feat.join(cross, how="left")

        # Use most recent row with complete features
        last = feat[FEATURE_COLS].dropna().tail(1)
        if last.empty:
            log.warning("No complete feature row for %s — skipping", sym)
            continue

        rows.append(last.values[0])
        symbols.append(sym)

    if not rows:
        raise RuntimeError("No valid feature rows produced")

    X     = np.array(rows, dtype=np.float64)
    probs = state.model.predict(X)   # shape (n,) for binary booster

    signals = []
    for sym, prob in zip(symbols, probs):
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
    """Returns True if NYSE is currently open (approximate)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    open_  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return open_ <= now <= close_


async def _refresh() -> bool:
    """Fetch data and rebuild signals. Returns True on success."""
    try:
        log.info("Refreshing signals ...")
        t0  = time.perf_counter()
        raw = await asyncio.get_event_loop().run_in_executor(None, _fetch_bars_batch)
        new_signals = await asyncio.get_event_loop().run_in_executor(
            None, build_signals, raw
        )
        state.cache       = new_signals
        state.last_update = datetime.now(ET)
        state.is_stale    = False

        elapsed = time.perf_counter() - t0
        buys    = [s for s in new_signals if s["signal"] == "BUY"]
        log.info(
            "Signals refreshed in %.1fs — %d BUY, %d HOLD",
            elapsed, len(buys), len(new_signals) - len(buys),
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
    """Refresh every REFRESH_MINUTES during market hours; sleep otherwise."""
    while True:
        if _is_market_hours():
            await _refresh()
            await asyncio.sleep(REFRESH_MINUTES * 60)
        else:
            # Check again in 5 minutes — catches the market open transition
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
    state.model = lgb.Booster(model_file=str(MODEL_FILE))
    log.info("Model loaded — %d features", state.model.num_feature())

    # Load earnings history (uses file cache; only hits FMP API on first run)
    state.earnings_cache = _load_all_earnings()

    # Initial fetch (runs even outside market hours so the cache is warm)
    await _refresh()

    # Start background loop
    state.refresh_task = asyncio.create_task(_background_refresh_loop())
    log.info("Background refresh task started (every %d min during market hours)", REFRESH_MINUTES)

    yield  # ── server is running ──

    # Shutdown
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
    description = "LightGBM signals for the auto-trader bot",
    version     = "1.0.0",
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
    # Note: macOS reserves port 5000 for AirPlay Receiver (Control Center).
    # Using 5001. Disable AirPlay Receiver in System Preferences → General →
    # AirDrop & Handoff if you need port 5000 specifically.
    uvicorn.run(
        "signal_server:app",
        host      = "0.0.0.0",
        port      = 5001,
        log_level = "info",
        reload    = False,
    )
