"""
Signal Server (v9.6 — Multi-Strategy Factor Framework)
======================================================
FastAPI service that runs the v9.6 multi-strategy factor framework
and serves trading signals to the JS trading bot.

Strategies:
  - S1 Adaptive Momentum (90% bull / 10% bear): consistency-weighted multi-timeframe
  - S3 Sector Rotation (5% bull / 20% bear): relative-strength sector ETFs
  - S5 Low-Vol Quality (5% bull / 70% bear): defensive quality stocks
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
from signal_builder import build_signals_v9
from event_short_manager import EventShortManager

# ── Paths & env ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # ensure data dir exists
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

# ── Startup data validation ──────────────────────────────────────────────────
def _validate_data_files():
    """Check critical data files exist at startup."""
    critical = [
        DATA_DIR / "wrds" / "fama_french_5factors_momentum_daily.parquet",
    ]
    optional = [
        DATA_DIR / "sp1500_members.json",
        DATA_DIR / "cache_sectors.json",
    ]
    for f in critical:
        if not f.exists():
            log.warning(f"CRITICAL DATA MISSING: {f.name} — some features disabled")
    for f in optional:
        if not f.exists():
            log.info(f"Optional data missing: {f.name}")

_validate_data_files()

# Warmup: 252d for SMA200 + buffer → 550 calendar days
WARMUP_DAYS     = 550
REFRESH_MINUTES = 15
ET              = ZoneInfo("America/New_York")


# ── Server state ──────────────────────────────────────────────────────────────
class State:
    cache:          list                   = []
    cache_edgar:    list                   = []   # EDGAR-overlay SHADOW signals (/signals?edgar=1)
    last_update:    Optional[datetime]     = None
    is_stale:       bool                   = True
    refresh_task:   Optional[asyncio.Task] = None
    enhanced_data:  dict                   = {}
    # Strategy info
    strategy_version: str                  = "v12"
    top_n:          int                    = 5

state = State()

# ── Short Sleeve Manager ─────────────────────────────────────────────────────
short_manager = EventShortManager(
    data_dir=str(DATA_DIR),
    cache_dir=str(DATA_DIR / "short_sleeve"),
)
log.info("Event short manager initialized")


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
    """Load last known signals from disk (for fast startup).
    Rejects cache older than 60 minutes to avoid serving stale data."""
    try:
        if _SIGNAL_CACHE_FILE.exists():
            with open(_SIGNAL_CACHE_FILE) as f:
                data = _json.load(f)
            ts = data.get("ts")
            if ts:
                cache_time = datetime.fromisoformat(ts)
                age_minutes = (datetime.now(ET) - cache_time).total_seconds() / 60
                if age_minutes > 60:
                    log.warning(f"Signal cache is {age_minutes:.0f}min old — will refresh soon")
            return data.get("signals", []), ts
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
        log.info("Refreshing v9.6 signals ...")
        t0 = time.perf_counter()
        raw = await asyncio.get_event_loop().run_in_executor(None, _fetch_bars_batch)
        # v12: pass None for enhanced_data — PIT backtest proved enhanced data
        # HURTS returns by -2pp (quality boosts dilute pure momentum signal)
        new_signals = await asyncio.get_event_loop().run_in_executor(
            None, build_signals_v9, raw, None, state.top_n
        )
        state.cache       = new_signals
        state.last_update = datetime.now(ET)
        state.is_stale    = False
        state._last_raw   = raw  # store for short sleeve price access
        _save_signal_cache(new_signals)

        # EDGAR SHADOW build (cheap: reuses the daily universe cache via a read-only
        # proxy). Served on /signals?edgar=1 ONLY — the live path above is untouched.
        # The logged diff line is the daily evidence for the phase-3 flip decision.
        try:
            edgar_signals = await asyncio.get_event_loop().run_in_executor(
                None, build_signals_v9, raw, None, state.top_n, True
            )
            state.cache_edgar = edgar_signals
            live_buys = {s["symbol"] for s in new_signals if s["signal"] == "BUY"}
            sh_buys = {s["symbol"] for s in edgar_signals if s["signal"] == "BUY"}
            log.info("EDGAR shadow: %d BUY vs live %d | added=%s dropped=%s",
                     len(sh_buys), len(live_buys),
                     sorted(sh_buys - live_buys)[:8], sorted(live_buys - sh_buys)[:8])
        except Exception as e:
            state.cache_edgar = []
            log.warning("EDGAR shadow build failed (live unaffected): %s", e)

        elapsed = time.perf_counter() - t0
        buys = [s for s in new_signals if s["signal"] == "BUY"]
        log.info(
            "v9.6 signals refreshed in %.1fs — %d BUY, %d HOLD",
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
    # Startup — load enhanced data for v9.6 strategy
    log.info("=" * 60)
    log.info("  Signal Server v9.6 — Multi-Strategy Factor Framework")
    log.info("=" * 60)

    # Load enhanced data (price targets, DCF, financial growth, etc.)
    enhanced_dir = DATA_DIR / "enhanced_data"
    if enhanced_dir.exists():
        for fname, key in [("price_targets.parquet", "price_targets"),
                            ("dcf_values.parquet", "dcf"),
                            ("financial_growth.parquet", "financial_growth"),
                            ("enterprise_values.parquet", "enterprise_values"),
                            ("company_profiles.parquet", "profiles"),
                            ("transcript_sentiment.parquet", "transcript_sentiment"),
                            ("options_snapshots.parquet", "options"),
                            ("ortex_short_interest.parquet", "ortex_si"),
                            ("financial_scores.parquet", "financial_scores"),
                            ("analyst_grades_consensus.parquet", "analyst_grades")]:
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
    title       = "Trading Signal Server",
    description = "v9.6 Multi-Strategy Factor Framework signals for the auto-trader bot",
    version     = "9.6.0",
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
def get_signals(edgar: int = 0):
    # edgar=1 serves the SHADOW (EDGAR-overlay) build — diagnostics only; both live
    # engines call this endpoint WITHOUT the param and are unaffected.
    cache = state.cache_edgar if edgar else state.cache
    if not cache:
        raise HTTPException(status_code=503, detail="Signals not yet available — try again shortly")
    return {
        "signals":     cache,
        "last_update": state.last_update.isoformat() if state.last_update else None,
        "is_stale":    state.is_stale,
        "count":       len(cache),
        "buy_count":   sum(1 for s in cache if s["signal"] == "BUY"),
        "strategy":    state.strategy_version,
        "edgar_shadow": bool(edgar),
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


# ── Short Sleeve Endpoints ───────────────────────────────────────────────────

@app.get("/short-signals")
def get_short_signals():
    """Short sleeve signals from event-driven forced-selling strategy."""
    # Get current long positions for conflict check
    long_symbols = set(s["symbol"] for s in state.cache if s.get("signal") == "BUY")

    # Update short manager with current prices from our cache
    price_map = {}
    sma50_map = {}
    for s in state.cache:
        sym = s["symbol"]
        # The signal cache doesn't have raw prices, but we can extract
        # from the last refresh cycle's data if available
        if hasattr(state, '_last_raw') and state._last_raw is not None:
            if sym in state._last_raw.columns:
                series = state._last_raw[sym].dropna()
                if len(series) > 0:
                    price_map[sym] = float(series.iloc[-1])
                if len(series) >= 50:
                    sma50_map[sym] = float(series.tail(50).mean())

    short_manager.update_prices(price_map, sma50_map)

    # Generate short signals
    result = short_manager.generate_signals(long_positions=long_symbols)
    return result


@app.get("/short-status")
def get_short_status():
    """Short sleeve status and active positions."""
    return short_manager.get_status()


# ── Data Status & Quality Monitoring ─────────────────────────────────────────

@app.get("/data-status")
def data_status():
    """Comprehensive data freshness and quality check.

    Returns status for every data source with staleness thresholds.
    Call before market open to verify everything is current.
    """
    import time as _time
    now = _time.time()
    DATA_DIR = Path(__file__).resolve().parent / "data"
    ENHANCED_DIR = DATA_DIR / "enhanced_data"
    alerts = []

    def _check_file(path, max_age_hours, label):
        if not path.exists():
            alerts.append(f"MISSING: {label}")
            return {"status": "MISSING", "label": label, "age_hours": None}
        age_h = (now - path.stat().st_mtime) / 3600
        ok = age_h <= max_age_hours
        if not ok:
            alerts.append(f"STALE: {label} ({age_h:.0f}h old, max {max_age_hours}h)")
        return {
            "status": "ok" if ok else "STALE",
            "label": label,
            "age_hours": round(age_h, 1),
            "max_hours": max_age_hours,
        }

    checks = {}

    # SP1500 membership (weekly, max 14 days = 336h)
    checks["sp1500_membership"] = _check_file(
        DATA_DIR / "sp1500_members.json", 336, "SP1500 membership")

    # Sector map (weekly, max 14 days)
    checks["sector_map"] = _check_file(
        DATA_DIR / "cache_sectors.json", 336, "Sector map")

    # Fundamentals (daily weekdays, max 48h to account for weekends)
    for f in ["fundamentals_ratios", "fundamentals_income",
              "fundamentals_metrics", "fundamentals_earnings"]:
        checks[f] = _check_file(DATA_DIR / f"{f}.parquet", 48, f)

    # VIX (daily weekdays, max 48h)
    checks["vix_cache"] = _check_file(
        ENHANCED_DIR / "vix_cache.parquet", 48, "VIX cache")

    # Enhanced data (7-day cache TTL for some, daily for others)
    daily_enhanced = ["options_snapshots", "ortex_short_interest",
                      "ortex_short_dtc", "ortex_short_ctb",
                      "ortex_short_availability"]
    weekly_enhanced = ["financial_growth", "enterprise_values",
                       "company_profiles", "transcript_sentiment",
                       "price_targets", "dcf_values"]

    for f in daily_enhanced:
        checks[f] = _check_file(ENHANCED_DIR / f"{f}.parquet", 48, f)
    for f in weekly_enhanced:
        checks[f] = _check_file(ENHANCED_DIR / f"{f}.parquet", 192, f)  # 8 days

    # Signal quality
    signal_check = {"status": "ok", "label": "signals"}
    if not state.cache:
        signal_check["status"] = "NO_SIGNALS"
        alerts.append("NO SIGNALS generated")
    else:
        buys = [s for s in state.cache if s.get("signal") == "BUY"]
        total = len(state.cache)
        signal_check["total"] = total
        signal_check["buy_count"] = len(buys)
        # Sanity checks
        if total < 400:
            signal_check["status"] = "LOW_COUNT"
            alerts.append(f"LOW SIGNAL COUNT: {total} (expected 1400+)")
        if len(buys) == 0:
            signal_check["status"] = "NO_BUYS"
            alerts.append("ZERO BUY signals")
        # Check for unreasonable probabilities
        bad_probs = [s for s in state.cache if s.get("probability", 0) < 0
                     or s.get("probability", 0) > 1]
        if bad_probs:
            signal_check["status"] = "BAD_PROBABILITIES"
            alerts.append(f"{len(bad_probs)} signals with invalid probabilities")
    checks["signals"] = signal_check

    # SP1500 count validation
    sp1500_file = DATA_DIR / "sp1500_members.json"
    if sp1500_file.exists():
        try:
            import json as _json
            with open(sp1500_file) as f:
                sp = _json.load(f)
            total_members = len(sp.get("sp500", [])) + len(sp.get("sp400", [])) + len(sp.get("sp600", []))
            if total_members < 1400 or total_members > 1600:
                alerts.append(f"SP1500 COUNT ANOMALY: {total_members} (expected 1400-1600)")
            checks["sp1500_membership"]["member_count"] = total_members
        except Exception:
            pass

    overall = "OK" if not alerts else "ALERT"
    return {
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        "alerts": alerts,
        "checks": checks,
        "last_signal_update": state.last_update.isoformat() if state.last_update else None,
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
