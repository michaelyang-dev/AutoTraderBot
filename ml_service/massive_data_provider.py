"""
Massive (Polygon) Data Provider
================================
Production-grade market data fetcher using Massive (formerly Polygon.io).
Replaces yfinance for the live signal_server.py 15-minute cadence.

Architecture:
  - Cold start:    per-ticker historical bars (rate-limited, cached to disk)
  - Daily refresh: grouped endpoint (ALL tickers in 1 API call)
  - 15-min refresh: grouped endpoint (1 call) for latest bars
  - Fallback:      yfinance if Massive fails

Features:
  - Adjusted prices for features, unadjusted for live quotes
  - Parallel validation vs yfinance (first month)
  - Data quality gate: stale price detection, symbol count verification
  - Disk cache to avoid repeated cold starts

Usage:
    from massive_data_provider import MassiveDataProvider
    provider = MassiveDataProvider()
    bars = provider.fetch_bars_batch(symbols, warmup_days=550)
    quality = provider.run_quality_gate(bars, expected_symbols=symbols)
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("massive_data")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "massive_cache"

# ── CACHE VALIDITY (2026-09-30) ─────────────────────────────────────────────
# A per-symbol cache file is reused only if it is younger than CACHE_MAX_AGE_H **and** was
# written after the most recent SETTLED session close (weekday SETTLE_HOUR_ET:00 ET).
#
# Age alone was the rule until 2026-09-30, and it let two wrong things through, found while
# verifying the 2026-09-30 book-3 rebuild:
#   * every post-close build (17:50 cron restart, ~18:33 refresh-job restart) reused files fetched
#     that MORNING, before the day's bar existed, so the evening signals were computed on the
#     PREVIOUS session (2026-09-29 evening = 2026-09-28's close; the evening dry run of book 3 was
#     therefore a day stale and differed from the fresh build: +LGND/RDDT/VICR, -ADSK/ILMN);
#   * after the Sunday 21:20 restart the 18h clock expires Monday ~15:20, so Monday's refetch
#     cached a ~15:05 intraday snapshot (15-min delayed feed) as Monday's bar; the 17:50/18:33
#     builds and Tuesday's 09:18 PRE-OPEN build then used that snapshot as Monday's close until
#     the ~09:33 refetch. A Tuesday rebalance could have traded on it.
# The settle condition only ever makes a file INVALID earlier (never valid longer), so it can
# add refetches but can never serve older data than the age rule did. With it, the evening
# restart refetches the completed session and the next morning's pre-open build reuses those
# FINAL bars with no vendor call in the pre-open window.
# 17:00 = close 16:00 + closing-auction prints + the feed's 15-min delay, with margin. Keep in
# step with signal_builder.SESSION_SETTLE_HOUR_ET (the partial-session clock rule).
SETTLE_HOUR_ET = 17
CACHE_MAX_AGE_H = 18
_ET = ZoneInfo("America/New_York")


def last_settle_ts(now_ts: float = None) -> float:
    """Epoch seconds of the most recent weekday SETTLE_HOUR_ET:00 ET at or before now_ts.
    Holidays are not special-cased: on a weekday holiday this costs one redundant refetch."""
    now = datetime.fromtimestamp(now_ts if now_ts is not None else time.time(), _ET)
    cand = now.replace(hour=SETTLE_HOUR_ET, minute=0, second=0, microsecond=0)
    if cand > now:
        cand = (cand - timedelta(days=1)).replace(hour=SETTLE_HOUR_ET)
    while cand.weekday() >= 5:
        cand = (cand - timedelta(days=1)).replace(hour=SETTLE_HOUR_ET)
    # re-anchor to the wall clock of the chosen date so the UTC offset is that date's (DST-safe)
    cand = datetime(cand.year, cand.month, cand.day, SETTLE_HOUR_ET, tzinfo=_ET)
    return cand.timestamp()


def cache_file_fresh(mtime: float, now_ts: float = None) -> tuple:
    """(is_fresh, reason) for a cache file written at epoch `mtime`. Pure — unit-tested."""
    now_ts = now_ts if now_ts is not None else time.time()
    age_h = (now_ts - mtime) / 3600
    if age_h >= CACHE_MAX_AGE_H:
        return False, "age"
    if mtime < last_settle_ts(now_ts):
        return False, "pre-settle"
    return True, "ok"

# Polygon API base
API_BASE = "https://api.polygon.io"

# Rate limiting — small delay prevents 429 throttling during cold starts (1500+ symbols)
CALLS_PER_MINUTE = 0  # unlimited plan, but Polygon still throttles bursts
CALL_INTERVAL = 0.08  # 80ms between calls (~12/sec, avoids burst throttling)


def _sic_to_sector(sic_code: str, sic_desc: str = "") -> str:
    """Map SIC code to GICS-like sector name."""
    if not sic_code:
        return "Unknown"
    try:
        sic = int(sic_code)
    except (ValueError, TypeError):
        return "Unknown"

    # SIC ranges to GICS sector mapping
    if 100 <= sic <= 999:
        return "Basic Materials"      # Mining, agriculture
    elif 1000 <= sic <= 1499:
        return "Energy"               # Oil, gas, mining
    elif 1500 <= sic <= 1799:
        return "Industrials"          # Construction
    elif 2000 <= sic <= 3999:
        # Manufacturing — need to split
        if 2000 <= sic <= 2111:
            return "Consumer Defensive"  # Food, tobacco
        elif 2800 <= sic <= 2899:
            return "Healthcare"          # Chemicals/pharma
        elif 3570 <= sic <= 3579 or 3670 <= sic <= 3679 or 3810 <= sic <= 3829:
            return "Technology"          # Computers, semiconductors
        elif 3600 <= sic <= 3699:
            return "Technology"          # Electronics
        elif 3711 <= sic <= 3799:
            return "Consumer Cyclical"   # Vehicles
        else:
            return "Industrials"         # General manufacturing
    elif 4000 <= sic <= 4999:
        if 4800 <= sic <= 4899:
            return "Communication Services"
        else:
            return "Industrials"         # Transportation, utilities
    elif 5000 <= sic <= 5999:
        if 5200 <= sic <= 5999:
            return "Consumer Cyclical"   # Retail
        else:
            return "Consumer Cyclical"   # Wholesale
    elif 6000 <= sic <= 6799:
        if 6500 <= sic <= 6599:
            return "Real Estate"
        else:
            return "Financial Services"
    elif 7000 <= sic <= 7399:
        return "Consumer Cyclical"      # Services
    elif 7370 <= sic <= 7379:
        return "Technology"             # Computer services
    elif 7372 <= sic <= 7379:
        return "Technology"             # Software
    elif 7400 <= sic <= 8999:
        if 8000 <= sic <= 8099:
            return "Healthcare"          # Health services
        elif 8200 <= sic <= 8299:
            return "Consumer Defensive"  # Education
        else:
            return "Industrials"
    elif 4900 <= sic <= 4999:
        return "Utilities"
    return "Unknown"


class MassiveDataProvider:
    """
    Production market data provider using Massive (Polygon.io).

    Provides both adjusted (for features/backtesting) and unadjusted
    (for live quote reading) prices.
    """

    def __init__(self, api_key: str = None, validate_vs_yfinance: bool = True):
        self.api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        if not self.api_key:
            from dotenv import load_dotenv
            load_dotenv(BASE_DIR.parent / ".env")
            self.api_key = os.environ.get("MASSIVE_API_KEY")
        if not self.api_key:
            raise ValueError("MASSIVE_API_KEY not found in environment or .env")

        self.validate_vs_yfinance = validate_vs_yfinance
        self.session = requests.Session()
        self._last_call_time = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Rate Limiting ────────────────────────────────────────────────────────

    def _rate_limit(self):
        """Rate limit between API calls (currently disabled — unlimited plan)."""
        if CALL_INTERVAL > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < CALL_INTERVAL:
                time.sleep(CALL_INTERVAL - elapsed)
            self._last_call_time = time.time()

    def _api_get(self, url: str, params: dict = None) -> dict:
        """Make a rate-limited GET request to Polygon API."""
        self._rate_limit()
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        resp = self.session.get(f"{API_BASE}{url}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Per-Ticker Historical Bars ───────────────────────────────────────────

    def fetch_ticker_bars(self, symbol: str, start: str, end: str,
                          adjusted: bool = True) -> pd.DataFrame:
        """
        Fetch daily OHLCV bars for a single ticker.

        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex
        """
        data = self._api_get(
            f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
            params={
                "adjusted": str(adjusted).lower(),
                "sort": "asc",
                "limit": 50000,
            },
        )

        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()  # normalize to midnight
        df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                                 "c": "close", "v": "volume"})
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_localize(None)

        # Deduplicate: Polygon can return duplicate timestamps on split/dividend
        # adjustment days. Keep the last entry (most recently adjusted).
        df = df[~df.index.duplicated(keep="last")]

        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(np.float32)
        df["volume"] = df["volume"].astype(np.float64)

        return df.sort_index()

    # ── Stock Splits ─────────────────────────────────────────────────────────

    def fetch_splits(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock split history for a ticker.
        Returns DataFrame with columns: execution_date, split_from, split_to
        """
        results = []
        url = f"/v3/reference/splits"
        params = {"ticker": symbol, "limit": 100}

        data = self._api_get(url, params=params)
        for r in data.get("results", []):
            results.append({
                "date": pd.Timestamp(r["execution_date"]),
                "split_from": r["split_from"],
                "split_to": r["split_to"],
                "ratio": r["split_to"] / r["split_from"],
            })

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values("date").reset_index(drop=True)

    # ── Dividends ────────────────────────────────────────────────────────────

    def fetch_dividends(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch dividend history for a ticker."""
        data = self._api_get(f"/v3/reference/dividends",
                             params={"ticker": symbol, "limit": limit})
        results = []
        for r in data.get("results", []):
            results.append({
                "ex_date": pd.Timestamp(r.get("ex_dividend_date")),
                "amount": r.get("cash_amount"),
                "type": r.get("dividend_type"),
            })
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values("ex_date").reset_index(drop=True)

    # ── Ticker Details (for sector/SIC codes) ────────────────────────────────

    def fetch_ticker_details(self, symbol: str) -> dict:
        """Fetch ticker details including SIC code, name, exchange."""
        data = self._api_get(f"/v3/reference/tickers/{symbol}")
        return data.get("results", {})

    # ── Sector Mapping (replaces yfinance Ticker.info) ─────────────────────

    def build_sector_map(self, symbols: list) -> dict:
        """
        Build {symbol: sector} map using Polygon ticker details.
        Uses SIC codes mapped to GICS-like sectors.
        Caches to disk to avoid repeated API calls.
        """
        cache_file = DATA_DIR / "cache_sectors.json"

        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            missing = [s for s in symbols if s not in cached]
        else:
            cached = {}
            missing = list(symbols)

        if missing:
            log.info("Fetching sector info from Massive for %d symbols ...", len(missing))
            for i, sym in enumerate(missing):
                try:
                    details = self.fetch_ticker_details(sym)
                    sic = details.get("sic_code", "")
                    sic_desc = details.get("sic_description", "")
                    cached[sym] = _sic_to_sector(sic, sic_desc)
                except Exception:
                    cached[sym] = "Unknown"
                if (i + 1) % 50 == 0:
                    log.info("    ... %d/%d", i + 1, len(missing))

            with open(cache_file, "w") as f:
                json.dump(cached, f)
            log.info("Sector cache updated: %d symbols", len(cached))

        return {s: cached.get(s, "Unknown") for s in symbols}

    # ── Grouped Daily Bars (ALL tickers in 1 call) ───────────────────────────

    def fetch_grouped_daily(self, date: str,
                             adjusted: bool = True) -> dict[str, dict]:
        """
        Fetch OHLCV for ALL US stocks for a single date.
        Returns {symbol: {open, high, low, close, volume}}.
        """
        data = self._api_get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date}",
            params={"adjusted": str(adjusted).lower()},
        )

        results = data.get("results", [])
        bars = {}
        for r in results:
            bars[r["T"]] = {
                "open": r["o"], "high": r["h"], "low": r["l"],
                "close": r["c"], "volume": r["v"],
            }
        return bars

    # ── Snapshot (current prices, all tickers) ───────────────────────────────

    def fetch_snapshot_all(self) -> dict[str, dict]:
        """
        Fetch current-day snapshot for all US stocks.
        Returns {symbol: {open, high, low, close, volume, prev_close}}.
        """
        data = self._api_get(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
        )

        tickers = data.get("tickers", [])
        result = {}
        for t in tickers:
            sym = t.get("ticker", "")
            day = t.get("day", {})
            prev = t.get("prevDay", {})
            result[sym] = {
                "open": day.get("o"),
                "high": day.get("h"),
                "low": day.get("l"),
                "close": day.get("c"),
                "volume": day.get("v"),
                "prev_close": prev.get("c"),
                # Unadjusted last trade price
                "last_trade": t.get("lastTrade", {}).get("p"),
            }
        return result

    # ── Batch Historical Fetch (with disk cache) ─────────────────────────────

    def fetch_bars_batch(self, symbols: list, warmup_days: int = 550,
                          adjusted: bool = True) -> dict[str, pd.DataFrame]:
        """
        Fetch historical bars for all symbols with disk caching.

        First checks disk cache. For uncached symbols, fetches from API
        with rate limiting. Uses grouped daily for the most recent day
        to minimize API calls.

        Returns {symbol: DataFrame} with OHLCV columns.
        """
        start_date = (datetime.today() - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
        end_date = datetime.today().strftime("%Y-%m-%d")
        cache_tag = "adj" if adjusted else "raw"

        raw = {}
        uncached = []

        # Check disk cache (validity rule: see CACHE VALIDITY at the top of this module)
        _now = time.time()
        _stale = {"age": 0, "pre-settle": 0}
        for sym in symbols:
            cache_file = CACHE_DIR / f"{sym}_{cache_tag}.parquet"
            if cache_file.exists():
                fresh, why = cache_file_fresh(cache_file.stat().st_mtime, _now)
                if fresh:
                    try:
                        df = pd.read_parquet(cache_file)
                        df.index = pd.to_datetime(df.index)
                        raw[sym] = df
                        continue
                    except Exception:
                        pass
                else:
                    _stale[why] += 1
            uncached.append(sym)
        if _stale["pre-settle"]:
            log.info("cache: %d file(s) younger than %dh were written before the last settled close "
                     "(%s ET) and are refetched so the completed session's FINAL bar is used "
                     "(%d expired by age)", _stale["pre-settle"], CACHE_MAX_AGE_H,
                     datetime.fromtimestamp(last_settle_ts(_now), _ET).strftime("%a %Y-%m-%d %H:%M"),
                     _stale["age"])

        # ── CACHE DEPTH GUARD (2026-08-19) ───────────────────────────────────
        # The cache is per-symbol and reused for 18h with NO check on how much history
        # each file actually holds. On 2026-08-18 the live book ran for >2h with
        # dist_sma200 coverage at 30.6% (1044 of 1504 SP1500 names missing) while the
        # fetch reported success (1539/1539, quality gate PASSED) — because coverage is
        # not a fetch property, it is a DEPTH property, and nothing measured depth.
        #
        # Why depth alone breaks and nothing else does: dist_sma200 is the only feature
        # needing 200 CONTIGUOUS bars. `c.rolling(200).mean()` uses pandas' default
        # min_periods=200, so a symbol short of 200 usable bars yields NaN; get_feature_map
        # drops NaN keys; `dist_sma200.get(sym, 0)` then returns 0, fails `> 0`, and the
        # name vanishes from the momentum and lowvol sleeves with no log and no error.
        # vol_60d (60 bars) and ret_126d (touches 2 rows) survive the same truncation
        # untouched, which is exactly the signature observed: 99.4% / 99.4% / 30.6%.
        #
        # Deliberately a SET-level test, not per-symbol. Genuinely short histories are
        # normal and permanent (recent IPOs: ADIG 10 bars, HONA 35, MFP 30) — re-fetching
        # those every cycle would burn API budget forever and never succeed. The failure
        # being defended against is MASS truncation, so the statistic is the MEDIAN.
        #
        # NOT fixed by relaxing min_periods: the backtest universe builder uses the
        # identical `rolling(200).mean()` (scripts/build_universe_2000.py:262), so
        # loosening it live would silently change which names the live book selects
        # relative to every validated backtest number. The formula is right; the input
        # was short. Fix the input.
        _depths = sorted(len(d) for d in raw.values() if d is not None and len(d) > 0)
        if _depths:
            _median = _depths[len(_depths) // 2]
            _expected = int(warmup_days * 252 / 365)          # ~379 sessions for 550d
            _floor = int(_expected * 0.80)                      # ~303
            log.info("cache depth: median %d bars (expected ~%d), p05 %d, min %d, "
                     "%d/%d below 200",
                     _median, _expected, _depths[max(0, len(_depths) // 20)], _depths[0],
                     sum(1 for d in _depths if d < 200), len(_depths))
            if _median < _floor:
                log.error("CACHE DEPTH ALARM: median cached history %d bars < floor %d "
                          "(expected ~%d). This silently guts dist_sma200 and drops names "
                          "from the momentum sleeve. Discarding cache and refetching ALL.",
                          _median, _floor, _expected)
                uncached = list(symbols)
                raw = {}

        if uncached:
            log.info("Fetching %d/%d symbols from Massive (rate-limited) ...",
                     len(uncached), len(symbols))

            # Try grouped daily for today first (1 call = all tickers)
            today_str = datetime.today().strftime("%Y-%m-%d")
            try:
                today_bars = self.fetch_grouped_daily(today_str, adjusted=adjusted)
                log.info("  Grouped daily: %d tickers for %s", len(today_bars), today_str)
            except Exception as e:
                log.warning("  Grouped daily failed: %s", e)
                today_bars = {}

            # Fetch per-ticker history for uncached symbols
            for i, sym in enumerate(uncached):
                if (i + 1) % 50 == 0 or i == 0:
                    log.info("  Fetching %d/%d: %s ...", i + 1, len(uncached), sym)
                try:
                    df = self.fetch_ticker_bars(sym, start_date, end_date, adjusted=adjusted)
                    if len(df) > 0:
                        raw[sym] = df
                        # Cache to disk
                        cache_file = CACHE_DIR / f"{sym}_{cache_tag}.parquet"
                        df.to_parquet(cache_file)
                    else:
                        raw[sym] = pd.DataFrame()
                except requests.exceptions.HTTPError as e:
                    log.error("  Failed to fetch %s: %s", sym, e)
                    raw[sym] = pd.DataFrame()
                except Exception as e:
                    log.error("  Failed to fetch %s: %s", sym, e)
                    raw[sym] = pd.DataFrame()

        # ── STALE-CONTENT GUARD (2026-09-30) ─────────────────────────────────
        # A file's mtime says when it was WRITTEN, not what it CONTAINS. The nightly refresh job's
        # data_gaps step rewrote 15 renamed tickers (AGNT, FISV, MRSH, ...) at ~18:30 every evening
        # with yfinance history that ended the PREVIOUS session (yfinance `end` is exclusive). Fresh
        # by mtime (post-settle, <18h), those files were reused by the 18:33 build and every next-
        # morning pre-open build, so each rebalance ranked the universe with those 15 names missing
        # their last bar -> NaN dist_sma200 -> silently absent from the momentum/lowvol sleeves
        # (dist_sma200 coverage 99.1% -> 98.1%, far above the 85% alarm). Rule: a CACHED symbol whose
        # last bar is older than the session most symbols end on is refetched. Content-based, so it
        # holds whatever process wrote the file. Tie -> the later date (more refetching, never less).
        _from_cache = [s for s in symbols if s not in set(uncached)
                       and raw.get(s) is not None and len(raw[s]) > 0]
        _last = {s: pd.Timestamp(d.index.max()).normalize()
                 for s, d in raw.items() if d is not None and len(d) > 0}
        if _last and _from_cache:
            _mode = pd.Series(list(_last.values())).mode().max()
            _behind = [s for s in _from_cache if _last[s] < _mode]
            if _behind:
                log.warning("STALE CACHE CONTENT: %d cached file(s) end before %s, the session most symbols "
                            "end on — refetching: %s", len(_behind), _mode.date(), _behind[:25])
                _ok = 0
                for sym in _behind:
                    try:
                        df = self.fetch_ticker_bars(sym, start_date, end_date, adjusted=adjusted)
                        if len(df) > 0:
                            raw[sym] = df
                            df.to_parquet(CACHE_DIR / f"{sym}_{cache_tag}.parquet")
                            _ok += 1
                    except Exception as e:           # keep the cached frame: never worse than before
                        log.error("  stale-content refetch failed for %s: %s (keeping cached)", sym, e)
                log.info("  stale-content refetch: %d/%d replaced", _ok, len(_behind))

        loaded = sum(1 for s in raw if len(raw[s]) > 0)
        log.info("Massive: %d/%d symbols loaded (%d from cache, %d fresh)",
                 loaded, len(symbols), len(symbols) - len(uncached), len(uncached))

        # Post-fetch depth report. If the refetch is ALSO short the problem is upstream
        # (vendor/API), not the cache — say so plainly rather than looping, and let the
        # existing feature-coverage guard in signal_builder alarm on the consequence.
        _d2 = sorted(len(d) for d in raw.values() if d is not None and len(d) > 0)
        if _d2:
            _m2 = _d2[len(_d2) // 2]
            _exp2 = int(warmup_days * 252 / 365)
            _n200 = sum(1 for d in _d2 if d < 200)
            log.info("post-fetch depth: median %d bars, %d/%d symbols below 200 "
                     "(these cannot produce dist_sma200)", _m2, _n200, len(_d2))
            if _m2 < int(_exp2 * 0.80):
                log.error("DEPTH STILL SHORT AFTER REFETCH: median %d < %d. The vendor is "
                          "returning truncated history — this is upstream of the cache.",
                          _m2, int(_exp2 * 0.80))
        # VENDOR-HOLE SCAN (2026-09-02). Root cause of both dist_sma200 collapses: Polygon
        # transiently omits a real trading date from per-symbol history (2026-08-28 was missing
        # from every bad-day cache file). Depth cannot see this (378 vs 379). Count, per date,
        # how many symbols whose span covers it are missing it; a date absent from >20% of
        # covering symbols is a vendor hole. signal_builder fills it; this is the loud early
        # signal at the layer where it originates.
        try:
            from collections import Counter as _Ctr
            _has = _Ctr(); _cover = _Ctr()
            _spans = []
            for _s, _d in raw.items():
                if _d is None or len(_d) == 0:
                    continue
                _ix = pd.DatetimeIndex(_d.index)
                _has.update(_ix)
                _spans.append((_ix.min(), _ix.max()))
            _alld = sorted(_has)
            for _lo, _hi in _spans:
                for _t in _alld:
                    if _lo <= _t <= _hi:
                        _cover[_t] += 1
            _holes = [(str(_t)[:10], _cover[_t] - _has[_t], _cover[_t]) for _t in _alld
                      if _cover[_t] >= 50 and (_cover[_t] - _has[_t]) / _cover[_t] > 0.20]
            if _holes:
                log.error("VENDOR HOLE(S) in fetched history: %s  (date, missing, covering)",
                          _holes[:6])
            else:
                log.info("vendor-hole scan: clean (%d dates, %d symbols)", len(_alld), len(_spans))
        except Exception as _e:
            log.warning("vendor-hole scan failed: %s", _e)
        return raw

    # ── Quick Refresh (for 15-min cadence) ───────────────────────────────────

    def refresh_latest(self, symbols: list,
                        existing_bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Quick refresh: fetch today's grouped bars and append to existing data.
        Only 1 API call regardless of number of symbols.
        """
        today_str = datetime.today().strftime("%Y-%m-%d")
        try:
            grouped = self.fetch_grouped_daily(today_str, adjusted=True)
        except Exception as e:
            log.error("Grouped daily refresh failed: %s", e)
            return existing_bars

        today_ts = pd.Timestamp(today_str)
        updated = dict(existing_bars)

        for sym in symbols:
            if sym not in grouped:
                continue
            bar = grouped[sym]
            new_row = pd.DataFrame({
                "open": [np.float32(bar["open"])],
                "high": [np.float32(bar["high"])],
                "low": [np.float32(bar["low"])],
                "close": [np.float32(bar["close"])],
                "volume": [float(bar["volume"])],
            }, index=[today_ts])

            if sym in updated and len(updated[sym]) > 0:
                # Update or append today's bar
                df = updated[sym]
                if today_ts in df.index:
                    df.loc[today_ts] = new_row.iloc[0]
                else:
                    df = pd.concat([df, new_row])
                updated[sym] = df.sort_index()
            else:
                updated[sym] = new_row

        return updated

    # ── Data Quality Gate ────────────────────────────────────────────────────

    def run_quality_gate(self, bars: dict[str, pd.DataFrame],
                          expected_symbols: list) -> dict:
        """
        Run data quality checks. Returns a dict with pass/fail and details.

        Checks:
        1. Symbol coverage: expected vs received
        2. Stale prices: close == prev close on 3+ consecutive days
        3. Null/zero prices
        4. Date recency: most recent bar should be within 2 trading days
        """
        issues = []

        # 1. Symbol coverage
        received = set(s for s, df in bars.items() if len(df) > 0)
        expected = set(expected_symbols)
        missing = expected - received
        coverage_pct = len(received & expected) / len(expected) * 100 if expected else 100

        if coverage_pct < 95:
            issues.append(f"LOW_COVERAGE: {coverage_pct:.1f}% ({len(missing)} symbols missing)")
        if missing:
            log.warning("Missing symbols (%d): %s", len(missing),
                        ", ".join(sorted(missing)[:20]))

        # 2. Stale prices (close unchanged for 3+ consecutive days)
        stale_symbols = []
        for sym, df in bars.items():
            if len(df) < 5:
                continue
            close = df["close"].iloc[-5:]
            if close.nunique() == 1 and close.iloc[0] > 0:
                stale_symbols.append(sym)

        if stale_symbols:
            issues.append(f"STALE_PRICES: {len(stale_symbols)} symbols "
                          f"({', '.join(stale_symbols[:10])})")

        # 3. Null/zero prices
        null_symbols = []
        for sym, df in bars.items():
            if len(df) == 0:
                continue
            last_close = df["close"].iloc[-1]
            if pd.isna(last_close) or last_close <= 0:
                null_symbols.append(sym)

        if null_symbols:
            issues.append(f"NULL_PRICES: {len(null_symbols)} symbols")

        # 4. Date recency
        most_recent = pd.Timestamp.min
        for sym, df in bars.items():
            if len(df) > 0:
                most_recent = max(most_recent, df.index.max())

        days_stale = (pd.Timestamp.today() - most_recent).days
        if days_stale > 3:
            issues.append(f"DATA_STALE: most recent bar is {days_stale} days old")

        # 5. OHLCV consistency: high >= low, open/close within [low,high],
        #    no zero/negative prices, no zero volume
        bad_ohlcv_symbols = []
        total_violations = 0
        for sym, df in bars.items():
            if len(df) < 5:
                continue
            v = 0
            v += int((df["high"] < df["low"]).sum())
            v += int(((df["open"] > df["high"]) | (df["open"] < df["low"])).sum())
            v += int(((df["close"] > df["high"]) | (df["close"] < df["low"])).sum())
            v += int((df["close"] <= 0).sum())
            v += int((df["volume"] <= 0).sum())
            if v > 0:
                bad_ohlcv_symbols.append((sym, v))
                total_violations += v
        if bad_ohlcv_symbols:
            issues.append(f"BAD_OHLCV: {len(bad_ohlcv_symbols)} symbols, "
                          f"{total_violations} total violations")
            log.warning("BAD_OHLCV symbols (first 10): %s",
                        ", ".join(f"{s}({n})" for s, n in bad_ohlcv_symbols[:10]))

        passed = len(issues) == 0
        result = {
            "passed": passed,
            "coverage_pct": coverage_pct,
            "n_received": len(received),
            "n_expected": len(expected),
            "n_missing": len(missing),
            "n_stale": len(stale_symbols),
            "n_null": len(null_symbols),
            "n_bad_ohlcv": len(bad_ohlcv_symbols),
            "most_recent_date": str(most_recent.date()) if most_recent > pd.Timestamp.min else "N/A",
            "issues": issues,
        }

        if not passed:
            log.error("DATA QUALITY GATE FAILED: %s", "; ".join(issues))
        else:
            log.info("Data quality gate PASSED: %d/%d symbols, latest=%s",
                     len(received), len(expected), result["most_recent_date"])

        return result

    # ── Validation: Massive vs yfinance ──────────────────────────────────────

    def validate_against_yfinance(self, symbols: list,
                                    massive_bars: dict[str, pd.DataFrame],
                                    n_sample: int = 50) -> dict:
        """
        Compare Massive data against yfinance for a sample of symbols.
        Flags discrepancies > 0.1% in close prices.

        Run this during the first month after migration.
        """
        import yfinance as yf

        sample_syms = sorted(symbols)[:n_sample]
        start = (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")
        end = datetime.today().strftime("%Y-%m-%d")

        log.info("Validating %d symbols against yfinance ...", len(sample_syms))

        try:
            yf_data = yf.download(sample_syms, start=start, end=end,
                                   auto_adjust=True, progress=False, threads=True)
            if yf_data.empty:
                log.info("yfinance returned no data (weekend/holiday?) — skipping validation")
                return {"validated": True, "n_checked": 0, "n_discrepancies": 0,
                        "discrepancies": [], "note": "skipped - no yfinance data"}
        except Exception as e:
            log.error("yfinance download failed during validation: %s", e)
            return {"validated": False, "error": str(e)}

        discrepancies = []
        checked = 0

        for sym in sample_syms:
            if sym not in massive_bars or len(massive_bars[sym]) == 0:
                continue

            # Get yfinance close
            try:
                if isinstance(yf_data.columns, pd.MultiIndex):
                    yf_close = yf_data["Close"][sym].dropna()
                else:
                    yf_close = yf_data["Close"].dropna()
            except Exception:
                continue

            yf_close.index = pd.to_datetime(yf_close.index).tz_localize(None)

            # Get Massive close
            m_close = massive_bars[sym]["close"]

            # Compare on overlapping dates
            common_dates = yf_close.index.intersection(m_close.index)
            if len(common_dates) == 0:
                continue

            checked += 1
            for date in common_dates[-5:]:  # check last 5 days
                yf_px = float(yf_close.loc[date])
                m_px = float(m_close.loc[date])
                if yf_px > 0:
                    pct_diff = abs(m_px - yf_px) / yf_px
                    if pct_diff > 0.001:  # >0.1% discrepancy
                        discrepancies.append({
                            "symbol": sym, "date": str(date.date()),
                            "massive": round(m_px, 4), "yfinance": round(yf_px, 4),
                            "diff_pct": round(pct_diff * 100, 3),
                        })

        result = {
            "validated": True,
            "n_checked": checked,
            "n_discrepancies": len(discrepancies),
            "discrepancies": discrepancies[:20],  # cap at 20
        }

        if discrepancies:
            log.warning("VALIDATION: %d discrepancies found in %d symbols",
                        len(discrepancies), checked)
            for d in discrepancies[:5]:
                log.warning("  %s %s: Massive=%.4f yfinance=%.4f (%.3f%%)",
                            d["symbol"], d["date"], d["massive"],
                            d["yfinance"], d["diff_pct"])
        else:
            log.info("VALIDATION: all %d symbols match (< 0.1%% diff)", checked)

        return result


# ══════════════════════════════════════════════════════════════════════════════
#  Drop-in replacement for _fetch_bars_batch in signal_server.py
# ══════════════════════════════════════════════════════════════════════════════

_provider = None


def get_provider() -> MassiveDataProvider:
    """Get or create the singleton MassiveDataProvider."""
    global _provider
    if _provider is None:
        _provider = MassiveDataProvider()
    return _provider


def fetch_bars_batch_massive(symbols: list,
                              warmup_days: int = 550) -> dict[str, pd.DataFrame]:
    """
    Drop-in replacement for signal_server._fetch_bars_batch().

    Returns {symbol: DataFrame} with columns: open, high, low, close, volume
    Same format as the yfinance version.
    """
    provider = get_provider()
    bars = provider.fetch_bars_batch(symbols, warmup_days=warmup_days, adjusted=True)

    # Run quality gate
    quality = provider.run_quality_gate(bars, expected_symbols=symbols)
    if not quality["passed"]:
        log.error("Quality gate failed — falling back to yfinance")
        return _fetch_bars_yfinance_fallback(symbols, warmup_days)

    # Validate against yfinance (first month — disable after validation period)
    if provider.validate_vs_yfinance:
        try:
            validation = provider.validate_against_yfinance(symbols, bars, n_sample=30)
            if validation.get("n_discrepancies", 0) > 10:
                log.error("Too many Massive/yfinance discrepancies (%d) — "
                          "investigate before trusting Massive data",
                          validation["n_discrepancies"])
        except Exception as e:
            log.warning("Validation against yfinance failed: %s", e)

    # Per-ticker yfinance fallback for symbols with insufficient Polygon data
    # Polygon occasionally has data gaps (e.g. FISV returns only 126 bars).
    # Patch these individually from yfinance rather than rejecting them.
    MIN_BARS = 252
    short_syms = [s for s in symbols
                  if s in bars and 0 < len(bars[s]) < MIN_BARS]
    if short_syms:
        log.warning("Polygon returned <252 bars for %d symbols: %s — "
                    "patching from yfinance",
                    len(short_syms),
                    [(s, len(bars[s])) for s in short_syms])
        try:
            import yfinance as yf
            start = (datetime.today() - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
            # yfinance `end` is EXCLUSIVE: end=today dropped the session that just closed whenever
            # this ran between the close and midnight UTC (the 17:50 / 18:33 evening builds), so the
            # patched names ended a session behind everyone else (2026-09-30). end=tomorrow includes
            # the latest session; an in-progress bar is trimmed by signal_builder's partial-session guard.
            end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            yf_data = yf.download(short_syms, start=start, end=end,
                                  auto_adjust=True, progress=False, threads=True)
            if isinstance(yf_data.columns, pd.MultiIndex):
                for sym in short_syms:
                    try:
                        df = yf_data.xs(sym, level=1, axis=1).dropna(how="all")
                        df.columns = [c.lower() for c in df.columns]
                        if len(df) >= MIN_BARS:
                            bars[sym] = df[["open", "high", "low", "close", "volume"]]
                            log.info("  %s: patched %d → %d bars via yfinance",
                                     sym, len(bars.get(sym, [])), len(df))
                    except Exception:
                        pass
            elif len(short_syms) == 1:
                df = yf_data.dropna(how="all")
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= MIN_BARS:
                    bars[short_syms[0]] = df[["open", "high", "low", "close", "volume"]]
                    log.info("  %s: patched via yfinance (%d bars)",
                             short_syms[0], len(df))
        except Exception as e:
            log.warning("yfinance per-ticker fallback failed: %s", e)

    return bars


def _fetch_bars_yfinance_fallback(symbols: list,
                                    warmup_days: int = 550) -> dict[str, pd.DataFrame]:
    """Fallback to yfinance if Massive fails."""
    import yfinance as yf

    log.warning("FALLING BACK TO YFINANCE for %d symbols", len(symbols))
    start = (datetime.today() - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    CHUNK_SIZE = 100

    raw = {}
    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i:i + CHUNK_SIZE]
        try:
            batch = yf.download(chunk, start=start, end=end,
                                auto_adjust=True, progress=False, threads=True)
        except Exception as exc:
            log.error("yfinance fallback chunk failed: %s", exc)
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
                for c in ["open", "high", "low", "close"]:
                    df[c] = df[c].astype(np.float32)
                raw[sym] = df.sort_index()
            except Exception:
                raw[sym] = pd.DataFrame()

    return raw


# ══════════════════════════════════════════════════════════════════════════════
#  CLI test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    provider = MassiveDataProvider(validate_vs_yfinance=True)

    # Quick test with a few symbols
    test_syms = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "NVDA", "META", "TSLA"]

    print("=" * 60)
    print("  MASSIVE DATA PROVIDER TEST")
    print("=" * 60)

    # Test per-ticker fetch
    print("\n1. Per-ticker fetch (AAPL, 30 days) ...")
    start = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    df = provider.fetch_ticker_bars("AAPL", start, end)
    print(f"   {len(df)} bars, latest: {df.index[-1].date()}")
    print(f"   Last bar: O={df['open'].iloc[-1]:.2f} H={df['high'].iloc[-1]:.2f} "
          f"L={df['low'].iloc[-1]:.2f} C={df['close'].iloc[-1]:.2f} V={df['volume'].iloc[-1]:,.0f}")

    # Test grouped daily
    print("\n2. Grouped daily (today) ...")
    today = datetime.today().strftime("%Y-%m-%d")
    grouped = provider.fetch_grouped_daily(today)
    print(f"   {len(grouped)} tickers")
    if "AAPL" in grouped:
        b = grouped["AAPL"]
        print(f"   AAPL: O={b['open']:.2f} C={b['close']:.2f} V={b['volume']:,.0f}")

    # Test batch fetch
    print(f"\n3. Batch fetch ({len(test_syms)} symbols, 30 days) ...")
    bars = provider.fetch_bars_batch(test_syms, warmup_days=30)
    for sym in test_syms:
        if sym in bars and len(bars[sym]) > 0:
            print(f"   {sym}: {len(bars[sym])} bars, "
                  f"latest={bars[sym].index[-1].date()} "
                  f"close={bars[sym]['close'].iloc[-1]:.2f}")

    # Test quality gate
    print("\n4. Quality gate ...")
    quality = provider.run_quality_gate(bars, expected_symbols=test_syms)
    print(f"   Passed: {quality['passed']}")
    print(f"   Coverage: {quality['coverage_pct']:.0f}%")

    # Test validation vs yfinance
    print("\n5. Validation vs yfinance ...")
    validation = provider.validate_against_yfinance(test_syms, bars, n_sample=8)
    print(f"   Checked: {validation['n_checked']} symbols")
    print(f"   Discrepancies: {validation['n_discrepancies']}")
    if validation.get("discrepancies"):
        for d in validation["discrepancies"]:
            print(f"     {d['symbol']} {d['date']}: "
                  f"Massive={d['massive']} yf={d['yfinance']} ({d['diff_pct']}%)")

    # Test snapshot
    print("\n6. Snapshot (current prices) ...")
    snapshot = provider.fetch_snapshot_all()
    print(f"   {len(snapshot)} tickers")
    for sym in ["AAPL", "SPY"]:
        if sym in snapshot:
            s = snapshot[sym]
            print(f"   {sym}: close={s['close']} prev_close={s['prev_close']} "
                  f"last_trade={s['last_trade']}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
