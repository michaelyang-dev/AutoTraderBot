"""
ML Signal Server (v9.5 — Multi-Strategy Factor Framework)
=========================================================
FastAPI service that runs the v9.5 multi-strategy factor framework
and serves trading signals to the JS trading bot.

Strategies:
  - S1 Adaptive Momentum (80% bull / 10% bear): consistency-weighted multi-timeframe
  - S3 Sector Rotation (10% bull / 20% bear): relative-strength sector ETFs
  - S4 Index Inclusion (event-driven): SP500 add/remove events
  - S5 Low-Vol Quality (10% bull / 70% bear): defensive quality stocks
  - Dynamic regime blending via market breadth (% above 50d SMA)
  - -15% stop-loss, 15% single-name cap, 35% sector cap

Endpoints
---------
GET /health              — strategy status, last refresh time, staleness flag
GET /signals             — all universe signals sorted by score desc
GET /signal/{symbol}     — signal for one symbol

Run with:
    python3 signal_server.py
"""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import uvicorn
import yfinance as yf  # kept as fallback only
from massive_data_provider import fetch_bars_batch_massive, get_provider
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from signal_server_v9 import build_signals_v9

# ── Paths & env ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
load_dotenv(BASE_DIR.parent / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger("signal_server")

# ── Universe ──────────────────────────────────────────────────────────────────
from sp500_universe import get_all_symbols

ALL_SYMBOLS = get_all_symbols()

# Warmup: 252d for SMA200 + buffer → 550 calendar days
WARMUP_DAYS     = 550
REFRESH_MINUTES = 15
ET              = ZoneInfo("America/New_York")


# ── Server state ──────────────────────────────────────────────────────────────
class State:
    cache:          list                   = []
    last_update:    Optional[datetime]     = None
    is_stale:       bool                   = True
    refresh_task:   Optional[asyncio.Task] = None
    enhanced_data:  dict                   = {}
    # Strategy info
    strategy_version: str                  = "v9.5"
    top_n:          int                    = 8

state = State()


# ── Data fetching ────────────────────────────────────────────────────────────

def _fetch_bars_batch() -> dict[str, pd.DataFrame]:
    """Batch-download WARMUP_DAYS of history for all symbols.
    Uses Massive (Polygon) as primary source with yfinance fallback."""
    try:
        raw = fetch_bars_batch_massive(ALL_SYMBOLS, warmup_days=WARMUP_DAYS)
        loaded = sum(1 for s in raw if len(raw[s]) > 0)
        log.info("  Massive: %d/%d symbols loaded", loaded, len(ALL_SYMBOLS))
        return raw
    except Exception as exc:
        log.error("  Massive provider failed: %s — falling back to yfinance", exc)
        return _fetch_bars_yfinance(ALL_SYMBOLS)


def _fetch_bars_yfinance(symbols: list) -> dict[str, pd.DataFrame]:
    """Legacy yfinance fetcher (fallback only)."""
    start = (datetime.today() - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    CHUNK_SIZE = 100

    raw = {}
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i + CHUNK_SIZE]
        log.info("  yfinance chunk %d/%d (%d symbols) ...",
                 i // CHUNK_SIZE + 1,
                 (len(symbols) + CHUNK_SIZE - 1) // CHUNK_SIZE,
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
                for c in ["open", "high", "low", "close"]:
                    df[c] = df[c].astype(np.float32)
                raw[sym] = df.sort_index()
            except Exception:
                raw[sym] = pd.DataFrame()

    log.info("  Downloaded bars for %d/%d symbols", len([s for s in raw if len(raw[s]) > 0]), len(symbols))
    return raw


# ── Signal persistence (survive restarts) ────────────────────────────────────
import json as _json

_SIGNAL_CACHE_FILE = DATA_DIR / "signal_cache_v9.json"


def _save_signal_cache(signals):
    """Persist signals to disk so restarts don't serve stale data."""
    try:
        with open(_SIGNAL_CACHE_FILE, "w") as f:
            _json.dump({"signals": signals, "ts": datetime.now(ET).isoformat()}, f)
    except Exception:
        pass


def _load_signal_cache():
    """Load last known signals from disk (for fast startup)."""
    try:
        if _SIGNAL_CACHE_FILE.exists():
            with open(_SIGNAL_CACHE_FILE) as f:
                data = _json.load(f)
            return data.get("signals", []), data.get("ts")
    except Exception:
        pass
    return [], None


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
        log.info("Refreshing v9.5 signals ...")
        t0 = time.perf_counter()
        raw = await asyncio.get_event_loop().run_in_executor(None, _fetch_bars_batch)
        new_signals = await asyncio.get_event_loop().run_in_executor(
            None, build_signals_v9, raw, state.enhanced_data, state.top_n
        )
        state.cache       = new_signals
        state.last_update = datetime.now(ET)
        state.is_stale    = False
        _save_signal_cache(new_signals)

        elapsed = time.perf_counter() - t0
        buys = [s for s in new_signals if s["signal"] == "BUY"]
        log.info(
            "v9.5 signals refreshed in %.1fs — %d BUY, %d HOLD",
            elapsed, len(buys), len(new_signals) - len(buys),
        )
        top8 = new_signals[:8]
        log.info(
            "Top 8: %s",
            " | ".join(f"{s['symbol']} {s['probability']:.4f}" for s in top8),
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
    # Startup — load enhanced data for v9.5 strategy
    log.info("=" * 60)
    log.info("  Signal Server v9.5 — Multi-Strategy Factor Framework")
    log.info("=" * 60)

    # Load enhanced data (price targets, DCF, financial growth, etc.)
    enhanced_dir = DATA_DIR / "enhanced_data"
    if enhanced_dir.exists():
        for fname, key in [("price_targets.parquet", "price_targets"),
                            ("dcf_values.parquet", "dcf"),
                            ("financial_growth.parquet", "financial_growth"),
                            ("enterprise_values.parquet", "enterprise_values"),
                            ("company_profiles.parquet", "profiles"),
                            ("crypto_forex_extended.parquet", "crypto_forex"),
                            ("transcript_sentiment.parquet", "transcript_sentiment"),
                            ("options_snapshots.parquet", "options")]:
            fpath = enhanced_dir / fname
            if fpath.exists():
                state.enhanced_data[key] = pd.read_parquet(fpath)
                if "date" in state.enhanced_data[key].columns:
                    state.enhanced_data[key]["date"] = pd.to_datetime(
                        state.enhanced_data[key]["date"]
                    )
                log.info("  Loaded %s: %d rows", key, len(state.enhanced_data[key]))

    # Also load fundamentals-level data not in enhanced_data/
    for fname, key in [("fundamentals_insiders.parquet", "insiders"),
                        ("fundamentals_estimates.parquet", "estimates")]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            state.enhanced_data[key] = pd.read_parquet(fpath)
            if "date" in state.enhanced_data[key].columns:
                state.enhanced_data[key]["date"] = pd.to_datetime(
                    state.enhanced_data[key]["date"]
                )
            log.info("  Loaded %s: %d rows", key, len(state.enhanced_data[key]))

    if state.enhanced_data:
        log.info("Enhanced data ready: %s", ", ".join(state.enhanced_data.keys()))
    else:
        log.warning("No enhanced data found — strategy will run without boosts")

    # Load last known signals from disk (instant startup — serve immediately)
    cached_signals, cached_ts = _load_signal_cache()
    if cached_signals:
        state.cache = cached_signals
        state.is_stale = True  # mark stale until fresh refresh completes
        log.info("Loaded %d cached signals from disk (ts=%s) — serving while refreshing",
                 len(cached_signals), cached_ts)

    # Initial signal refresh
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
    description = "v9.5 Multi-Strategy Factor Framework signals for the auto-trader bot",
    version     = "9.5.0",
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
        "status":          "ok",
        "strategy_version": state.strategy_version,
        "top_n":           state.top_n,
        "enhanced_data":   list(state.enhanced_data.keys()),
        "last_update":     state.last_update.isoformat() if state.last_update else None,
        "is_stale":        state.is_stale,
        "cached_signals":  len(state.cache),
        "market_open":     _is_market_hours(),
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
        "strategy":    state.strategy_version,
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
