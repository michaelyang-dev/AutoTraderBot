"""
v12 Signal Adapter
==================
Live wrapper around the shared v12 selection core (strategies.multi_strategy_engine,
also imported by main_production_backtest — so selection logic can't diverge). Blends
the sleeves, applies the breadth/UMD regime, loads fundamentals, and outputs signals
in the JSON format the engines expect.

This module is imported by signal_server.py.
"""

import json as _json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sp500_history import get_sp500_on_date
from sp1500_membership import get_sp1500_on_date
from strategies.multi_strategy_engine import (
    FastUniverse, strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, strategy4_index_inclusion,
    strategy_value, SECTOR_ETFS,
    PROD_WEIGHTS_BULL, PROD_WEIGHTS_BEAR, PROD_WEIGHTS_CRASH,
    compute_price_umd_20d, UMD_CRASH_THRESHOLD,
)

# v12 strategy config: 50% momentum, 35% value, 15% low-vol quality
# Honest backtest (deployed config, no look-ahead, no SI/enhanced, start-day
# averaged, 2018-2025): 22.0% CAGR, 0.90 Sharpe, -28.3% MaxDD at 1x.
# Changes from v11: weights 35/25/40→50/35/15, top_n 8→5, rebal 15→20d,
#   stop 35%→40%, cap 25%→15%, SI DISABLED (hurts -3pp).
# Sleeve weights are now the SINGLE SOURCE in multi_strategy_engine.py
# (PROD_WEIGHTS_*) — shared with the backtest so they can't drift apart.
STRATEGY_CONFIG_BULL = list(PROD_WEIGHTS_BULL.items())
STRATEGY_CONFIG_BEAR = list(PROD_WEIGHTS_BEAR.items())

log = logging.getLogger("signal_server")

# Module-level cache for the FastUniverse (expensive to build)
_uni_cache = None
_uni_cache_date = None


def _load_si_change_data(uni):
    """DEAD in v12 — NOT called (the only call site, ~line 556, is commented out; the
    live path sets _si_change_rank = {}). The SI-change sleeve was DISABLED because it
    HURTS returns by ~3pp (verified May 2026) — the opposite of the "+3.5% edge" this
    once claimed. Kept only for research/backtest experimentation. Do NOT re-enable
    without re-validating; see the v11->v12 note above (SI DISABLED).
    """
    DATA_DIR = Path(__file__).resolve().parent / "data"
    si_file = DATA_DIR / "wrds" / "compustat_short_interest.parquet"
    if not si_file.exists():
        uni._si_change_rank = {}
        return
    try:
        si = pd.read_parquet(si_file, columns=["tic", "datadate", "shortintadj"])
        si["datadate"] = pd.to_datetime(si["datadate"])
        si = si.dropna(subset=["shortintadj"])
        si = si[si["shortintadj"] > 0].sort_values(["tic", "datadate"])
        si["si_prev"] = si.groupby("tic")["shortintadj"].shift(2)
        si["si_change"] = (si["shortintadj"] - si["si_prev"]) / si["si_prev"]
        si = si.dropna(subset=["si_change"])
        latest = si.sort_values("datadate").groupby("tic")["si_change"].last()
        ranks = (-latest).rank(pct=True)
        uni._si_change_rank = ranks.to_dict()
        log.info(f"SI change data loaded: {len(uni._si_change_rank)} tickers")
    except Exception as e:
        log.warning(f"Could not load SI change data: {e}")
        uni._si_change_rank = {}


# ── COVERAGE STATE, PUBLISHED (2026-09-02) ───────────────────────────────────
# The coverage guard has fired twice (2026-08-18 dist_sma200 30.6%, 2026-09-01 19.7%) and BOTH
# times the cause had self-healed before anyone could look: the per-symbol disk cache turns over
# within ~18h, so post-hoc forensics arrive after the evidence is gone. The 2026-08-19 depth guard
# addressed a hypothesis (mass truncation) that the telemetry has since REFUTED -- during the
# second incident depth was healthy (median 378 bars, expected ~379) while coverage was 19.7%.
#
# So: stop guessing. Capture the state AT ALARM TIME, and publish coverage so the trading engine
# can refuse to act on a degraded book.
LAST_COVERAGE: dict = {}          # {feature: fraction}, read by signal_server /health
LAST_COVERAGE_OK: bool = True     # False while any feature is under its floor


def _coverage_forensics(raw, today, pool, missing, path):
    """Dump WHY names are missing, at the moment it happens. Read-only, best-effort.

    Discriminates the three live hypotheses in one shot:
      STALE    -- symbol's series ends before the union index does (NaN on recent rows)
      SHORT    -- symbol has fewer than 200 bars at all
      GAP      -- symbol spans the window but has interior NaNs inside the trailing 200
    Whichever bucket dominates names the cause; that is exactly what could not be recovered
    after the fact on 2026-08-18 or 2026-09-01.
    """
    import json
    idx = None
    for _d in raw.values():
        if _d is not None and len(_d):
            idx = _d.index if idx is None else idx.union(_d.index)
    if idx is None:
        return
    idx = pd.DatetimeIndex(sorted(idx))
    tail = idx[-200:]
    union_last = idx.max()
    buckets = {"STALE": 0, "SHORT": 0, "GAP": 0, "ABSENT": 0}
    detail = []
    for sym in list(missing)[:400]:
        d = raw.get(sym)
        if d is None or not len(d):
            buckets["ABSENT"] += 1
            continue
        c = d["close"].reindex(tail) if "close" in d.columns else None
        n_nan = int(c.isna().sum()) if c is not None else -1
        last = d.index.max()
        if len(d) < 200:
            b = "SHORT"
        elif last < union_last:
            b = "STALE"
        elif n_nan > 0:
            b = "GAP"
        else:
            b = "GAP"
        buckets[b] += 1
        if len(detail) < 60:
            detail.append({"sym": sym, "bucket": b, "bars": int(len(d)),
                           "first": str(d.index.min())[:10], "last": str(last)[:10],
                           "nan_in_trailing_200": n_nan})
    out = {"ts": datetime.now(ZoneInfo("US/Eastern")).isoformat(),
           "today": str(today)[:10],
           "union_index": {"rows": int(len(idx)), "first": str(idx.min())[:10],
                           "last": str(union_last)[:10]},
           "pool_size": int(len(pool)), "n_missing": int(len(missing)),
           "buckets": buckets, "sample": detail}
    try:
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        log.error("COVERAGE FORENSICS written to %s — buckets %s", path, buckets)
    except Exception as _e:
        log.error("could not write coverage forensics: %s", _e)


def _is_partial_session(idx, coverage=None):
    """True when idx[-1] belongs to a session that has NOT closed yet, OR whose bar is not
    yet published for most names (see the DATA RULE below).

    SINGLE SOURCE OF TRUTH for the partial-session guard — used by BOTH the price-matrix
    trim in _build_universe AND the `today` selection in build_signals_v9. Those two
    decided "what is today" independently once, and the mismatch produced an EMPTY book
    (features ended on the completed session while the lookup asked for the in-progress
    one), so they must never diverge again. Returns False for len < 2 so the caller can
    never be left with nothing.
    """
    if len(idx) < 2:
        return False
    now_et = datetime.now(ZoneInfo("US/Eastern"))
    # CLOCK RULE: today's bar is not final until SESSION_SETTLE_HOUR_ET (17:00), not the 16:00 bell
    # (2026-09-30). The price cache now refetches only after the session SETTLES (see
    # massive_data_provider CACHE VALIDITY); a file refetched mid-session by the 18h age rule holds
    # a ~15-min-delayed intraday snapshot for EVERY name, so row coverage is ~100% and the data rule
    # below cannot catch it. With `hour < 16` the 16:00-16:30 refreshes would have promoted that
    # snapshot to "today's close". No trading happens 16:00-17:00; the evening builds (17:50/18:33)
    # are unaffected.
    if pd.Timestamp(idx[-1]).date() == now_et.date() and now_et.hour < SESSION_SETTLE_HOUR_ET:
        return True
    # DATA RULE (2026-09-08, third coverage alarm): the clock alone is not enough. At 16:13 ET
    # the session had closed, so the clock rule said "final", but the vendor had published the
    # day's bar for only 421 of 1,504 names; the other 1,083 were NaN on the last row, so
    # rolling(200) went NaN and dist_sma200 coverage read 28%. A last row that is materially
    # emptier than the sessions before it is NOT a completed session, whatever the clock says.
    # `coverage` is the caller's measure of the last row (see _last_row_coverage): the fraction
    # of names that printed on idx[-1] relative to the recent sessions. Below 0.90 -> partial.
    if coverage is not None and coverage < PARTIAL_ROW_MIN_COVERAGE:
        return True
    return False


PARTIAL_ROW_MIN_COVERAGE = 0.90   # last row must carry >= 90% of the recent per-session print count
SESSION_SETTLE_HOUR_ET = 17       # today's bar counts as final from 17:00 ET; == massive_data_provider.SETTLE_HOUR_ET
SPLICE_MAX_RATIO = 4.0            # one-day close ratio above this = two securities under one ticker (BNY 12.6x, SOLS 487,000x; GME's record day was 2.35x)


def _splice_guard(raw):
    """Truncate each symbol's bars at the last one-day close ratio > SPLICE_MAX_RATIO (see the
    guard's comment in _build_universe). Returns (new_raw, [(sym, jump_date, ratio), ...]).
    Pure: never mutates the input frames."""
    out, spliced = {}, []
    for sym, df in raw.items():
        try:
            if df is None or len(df) < 3 or "close" not in df.columns:
                out[sym] = df
                continue
            c = pd.to_numeric(df["close"], errors="coerce")
            ratio = (c / c.shift(1))
            jumps = ratio[ratio > SPLICE_MAX_RATIO]
            if len(jumps) == 0:
                out[sym] = df
                continue
            cut = jumps.index[-1]                      # the jump bar is the NEW security's first close: keep it and what follows
            pos = df.index.get_loc(cut)
            out[sym] = df.iloc[pos:]
            spliced.append((sym, str(pd.Timestamp(cut).date()), float(jumps.iloc[-1])))
        except Exception:
            out[sym] = df
    return out, spliced


def _last_row_coverage(prices):
    """Fraction of names printed on the last row of a close-price matrix, relative to the
    largest per-row count over the previous 5 rows. 1.0 on a normal day; ~0.28 on 2026-09-08
    at 16:13 ET. Returns None when there is no history to compare against."""
    if prices is None or len(prices.index) < 2:
        return None
    counts = prices.notna().sum(axis=1)
    ref = float(counts.iloc[-6:-1].max())
    return float(counts.iloc[-1]) / ref if ref > 0 else None


def _raw_last_row_coverage(raw, last_date):
    """Same measure computed from the raw per-symbol bar dict before the matrix exists: the
    share of symbols whose last bar is `last_date`, relative to the share that have the
    session before it. Used by the `today` selection so it agrees with the matrix trim."""
    try:
        last = pd.Timestamp(last_date).normalize()
        have_last = have_prev = 0
        for df in raw.values():
            if df is None or len(df) == 0:
                continue
            ix = pd.to_datetime(df.index).normalize()
            if ix[-1] == last:
                have_last += 1
                if len(ix) > 1:
                    have_prev += 1
            elif ix[-1] < last:
                have_prev += 1
        return have_last / have_prev if have_prev > 0 else None
    except Exception:
        return None


def _build_universe(raw, enhanced_data=None):
    """Build FastUniverse from raw bar data."""
    DATA_DIR = Path(__file__).resolve().parent / "data"

    # ── TICKER-REUSE SPLICE GUARD (2026-09-08) ──────────────────────────────────
    # The vendor keys history by TICKER. When a company adopts a ticker that another security
    # used before, the per-ticker history is a splice of two securities. Found live 2026-09-08:
    # "BNY" (BNY Mellon since 2026-05-21) carried a ~$10.5 closed-end muni fund before that
    # date -> a +1,263% one-day jump -> fake +1,608% 12-month momentum -> ranked #2 in the
    # momentum sleeve and was the largest planned buy. The backtest is PERMNO-keyed and ranks
    # the real BNY at +53%. Rule: a one-day close/close ratio above SPLICE_MAX_RATIO cannot be a
    # single security (the largest genuine S&P 1500 day in 8 years of data is GME +134.8%), so
    # everything up to and including that bar is discarded; the name is then simply too short
    # for the 200/252-bar features until real history accrues, which is also what the
    # backtest sees for a fresh listing.
    raw, _spliced = _splice_guard(raw)
    if _spliced:
        log.warning("SPLICE GUARD: %d symbol(s) truncated at a >%.0fx one-day jump: %s", len(_spliced),
                    SPLICE_MAX_RATIO, ", ".join(f"{s}@{d} x{r:.0f}" for s, d, r in _spliced[:10]))

    # Build prices DataFrame
    close_frames = {}
    for sym, df in raw.items():
        if len(df) > 0 and "close" in df.columns:
            close_frames[sym] = df["close"]
    prices = pd.DataFrame(close_frames)
    prices.index = pd.to_datetime(prices.index)

    # ── PARTIAL-SESSION GUARD (2026-08-11) ────────────────────────────────────
    # v12 is a daily-CLOSE strategy: it must never compute features on a session
    # that has not closed yet. The signal server re-runs this every REFRESH_MINUTES
    # (15) *during market hours*, so without this guard the in-progress bar enters
    # the matrix. Any symbol whose bar has not printed yet is NaN, and because
    # pandas defaults rolling()'s min_periods to the window, ONE missing bar makes
    # c.rolling(200).mean() NaN for that name. get_feature_map then DROPS NaN keys
    # entirely, so `dist_sma200.get(sym, 0)` returns 0, fails `> 0`, and the name is
    # silently excluded from the momentum and lowvol sleeves — no log, no alarm.
    # Measured live 2026-08-04 10:11 ET: SNDK held 199/200 bars (the one gap being
    # that morning) and vanished from the book despite a 12-1 momentum score of
    # 30.37, over 4x the #1 name. 21 symbols were affected that morning.
    # Note ret_252d survives the same gap (pct_change touches only two rows), which
    # is why a name can post a huge score and still be invisible.
    # The backtest CANNOT express this failure — it only ever sees completed bars —
    # so dropping the in-progress row restores parity rather than creating a new
    # divergence, and makes the book deterministic through the trading day instead
    # of shifting every 15 minutes.
    # Only drops while the session is still open; after 16:00 ET the bar is final
    # (and the 17:30 refresh cron + 17:50 restart rebuild it properly). The 15-min
    # refresh loop is gated on _is_market_hours(), so nothing runs in between.
    _cov = _last_row_coverage(prices)
    if _is_partial_session(prices.index, _cov):
        _partial = prices.index[-1]
        _missing = int(prices.loc[_partial].isna().sum())
        prices = prices.iloc[:-1]
        log.info("Partial-session guard: dropped in-progress/incomplete bar %s (%d/%d symbols had "
                 "no print yet, row coverage %s); computing on last completed session %s",
                 str(_partial)[:10], _missing, prices.shape[1],
                 f"{_cov:.0%}" if _cov is not None else "n/a", str(prices.index[-1])[:10])

    # Load sector map
    sector_file = DATA_DIR / "cache_sectors.json"
    sector_map = {}
    if sector_file.exists():
        with open(sector_file) as f:
            sector_map = _json.load(f)

    # Load SP500 changes
    changes_file = DATA_DIR / "sp500_changes.json"
    sp500_changes = []
    if changes_file.exists():
        with open(changes_file) as f:
            sp500_changes = _json.load(f)

    # VIX data (cached)
    vix_data = None
    vix_cache = DATA_DIR / "enhanced_data" / "vix_cache.parquet"
    if vix_cache.exists():
        vix_data = pd.read_parquet(vix_cache)
        vix_data.index = pd.to_datetime(vix_data.index)

    # Build features DataFrame from raw bars (simplified — just what strategies need)
    features = _compute_features_from_raw(raw, prices)

    uni = FastUniverse(features, prices, sector_map, sp500_changes, vix_data,
                       enhanced_data=enhanced_data)
    return uni


# ── VENDOR-HOLE STATS (2026-09-02) — filled in by _compute_features_from_raw, logged by the
# coverage guard. ROOT CAUSE of the 2026-08-18 (30.6%) and 2026-09-01 (19.7%) dist_sma200
# collapses, proven from 16 bad-day cache files still on disk: every one was missing
# 2026-08-28 — a real trading day — that the good files have. Polygon transiently omits a
# date from per-symbol history (backfill lag); the union index keeps the date because SOME
# symbols have it; everyone else gets a NaN row inside the 200-bar window.
#
# WHY ONLY dist_sma200 BROKE: pandas pct_change() PADS NaNs by default, so ret_*/vol_* walk
# straight through the hole and look healthy (99.4% both incidents). rolling(200).mean() on
# the raw close does not pad, so ONE missing day kills it for 200 sessions.
_HOLE_STATS = {"symbols": 0, "cells": 0, "unfilled": 0, "dates": {}}


def _compute_features_from_raw(raw, prices):
    """Compute the feature columns that v9.6 strategies need from raw bars."""
    date_index = prices.index
    all_features = []
    _HOLE_STATS.update({"symbols": 0, "cells": 0, "unfilled": 0, "dates": {}})

    for sym in prices.columns:
        if sym not in raw or len(raw[sym]) == 0:
            continue
        df = raw[sym]
        # Fix: require minimum 252 bars of non-null close data
        # Stocks with insufficient history (e.g. FISV with 126 bars) produce
        # NaN for ret_252d and other long-lookback features, corrupting signals
        valid_bars = df["close"].dropna().shape[0]
        if valid_bars < 252:
            log.warning(f"Skipping {sym}: only {valid_bars} bars (need 252+)")
            continue
        c = df["close"].reindex(date_index)
        v = df["volume"].reindex(date_index) if "volume" in df.columns else pd.Series(1, index=date_index)

        # ── VENDOR-HOLE FILL (2026-09-02) ─────────────────────────────────────────
        # Forward-fill INTERIOR gaps only (between the first and last real print), at most
        # 3 sessions. This is parity-RESTORING, not a formula change: the backtest's CRSP
        # panel carries a row for every trading day per security, so its rolling(200) never
        # meets an interior NaN. Leading NaNs (pre-listing) are untouched; a gap longer than 3
        # sessions (a real halt/delisting) stays NaN and the name stays excluded. The formula
        # (rolling(200).mean(), default min_periods) is deliberately NOT relaxed.
        _fv, _lv = c.first_valid_index(), c.last_valid_index()
        if _fv is not None and _lv is not None and _fv != _lv:
            _inner = c.loc[_fv:_lv]
            _miss = _inner.isna()
            if _miss.any():
                _filled = _inner.ffill(limit=3)
                c = c.copy(); c.loc[_fv:_lv] = _filled
                _HOLE_STATS["symbols"] += 1
                _HOLE_STATS["cells"] += int(_miss.sum())
                _HOLE_STATS["unfilled"] += int(_filled.isna().sum())
                for _d in _inner.index[_miss.values]:
                    _HOLE_STATS["dates"][_d] = _HOLE_STATS["dates"].get(_d, 0) + 1

        feat = pd.DataFrame(index=date_index)
        feat["symbol"] = sym
        feat["date"] = date_index

        # Returns
        for n, name in [(5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d"),
                         (60, "ret_60d"), (120, "ret_120d"), (126, "ret_126d"), (252, "ret_252d")]:
            feat[name] = c.pct_change(n)

        # Volatility (annualized — must match backtest wrds_universe.py)
        daily_r = c.pct_change()
        for n, name in [(10, "vol_10d"), (20, "vol_20d"), (60, "vol_60d")]:
            feat[name] = daily_r.rolling(n).std() * np.sqrt(252)

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        feat["macd_line"] = ema12 - ema26
        feat["macd_signal"] = feat["macd_line"].ewm(span=9).mean()

        # SMA distances
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        feat["dist_sma50"] = (c - sma50) / sma50.replace(0, np.nan)
        feat["dist_sma200"] = (c - sma200) / sma200.replace(0, np.nan)

        # Bollinger
        sma20 = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        feat["bb_position"] = (c - sma20) / (2 * bb_std).replace(0, np.nan)

        # 52-week high/low
        h252 = c.rolling(252, min_periods=252).max()
        l252 = c.rolling(252, min_periods=252).min()
        feat["dist_52w_high"] = (c - h252) / h252.replace(0, np.nan)
        feat["dist_52w_low"] = (c - l252) / l252.replace(0, np.nan)

        # New highs
        feat["new_high_20d"] = (c >= c.rolling(20).max()).astype(float)
        feat["new_high_50d"] = (c >= c.rolling(50).max()).astype(float)

        # Volume ratio
        avg_vol = v.rolling(20).mean()
        feat["vol_ratio_20d"] = v / avg_vol.replace(0, np.nan)

        # OBV trend
        obv = (np.sign(c.diff()) * v).cumsum()
        obv_sma = obv.rolling(20).mean()
        feat["obv_trend_20d"] = (obv - obv_sma) / obv_sma.abs().replace(0, np.nan)

        # Max DD 6m
        peak_6m = c.rolling(126, min_periods=20).max()
        feat["max_dd_6m"] = (c - peak_6m) / peak_6m.replace(0, np.nan)

        # SMA200 slope
        sma200_30ago = sma200.shift(30)
        feat["sma200_slope"] = (sma200 - sma200_30ago) / sma200_30ago.replace(0, np.nan)

        # Fundamental placeholders — filled below from FMP parquets
        for fcol in ["gross_margin", "operating_margin", "net_margin", "pe_ratio",
                      "ps_ratio", "debt_to_equity", "current_ratio", "roe", "roa",
                      "eps_surprise_last", "days_since_earnings",
                      "revenue_growth_yoy", "eps_growth_yoy",
                      "insider_buy_ratio_90d", "insider_net_shares_90d"]:
            feat[fcol] = np.nan

        # Sector-relative return (needed by strategy1 for sector-strength boost)
        feat["ret_10d_vs_sector"] = np.nan  # filled below after all symbols computed

        feat["in_sp500"] = True  # will be filtered per-date by strategy

        # Only keep the canonical "today" row — use SPY's last date as reference
        # This prevents date mismatches when a few stocks have newer bars than SPY
        spy_last = prices["SPY"].dropna().index[-1] if "SPY" in prices.columns else date_index[-1]
        if spy_last in feat.index:
            today_feat = feat.loc[[spy_last]]
            if not today_feat["ret_20d"].isna().all():
                all_features.append(today_feat)
        else:
            # Fallback: use last valid row
            today_feat = feat.dropna(subset=["ret_20d"]).tail(1)
            if len(today_feat) > 0:
                all_features.append(today_feat)

    if not all_features:
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])

    # ── Fill fundamentals from FMP parquet caches ──
    DATA_DIR = Path(__file__).resolve().parent / "data"
    result = _fill_fundamentals(result, DATA_DIR)

    # ── Compute sector-relative returns ──
    result = _fill_sector_relative(result, DATA_DIR)

    return result


def _fill_fundamentals(features, data_dir):
    """Fill fundamental columns. Primary: WRDS Compustat/IBES. Fallback: FMP.

    WRDS Compustat provides standardized GAAP financials (gold standard).
    WRDS IBES provides institutional-grade earnings consensus.
    FMP is used as fallback if WRDS files are missing.
    """
    wrds_dir = data_dir / "wrds"
    wrds_fund = wrds_dir / "compustat_fundamentals_quarterly.parquet"
    wrds_ibes = wrds_dir / "ibes_summary_latest.parquet"

    wrds_loaded = False

    # ── PRIMARY: WRDS Compustat fundamentals ──
    if wrds_fund.exists():
        try:
            fund = pd.read_parquet(wrds_fund)
            fund["datadate"] = pd.to_datetime(fund["datadate"])
            # Use rdq (report date) for point-in-time if available, else datadate
            date_col = "rdq" if "rdq" in fund.columns else "datadate"
            fund[date_col] = pd.to_datetime(fund[date_col], errors="coerce")
            fund = fund.dropna(subset=[date_col])
            # WHOLE latest row per ticker (2026-09-30 parity fix). GroupBy.last() returns the last
            # NON-NULL value of EACH COLUMN separately: where Compustat left a field blank in the
            # newest quarter (dlcq for 208 of 1,496 pool names on 2026-09-29; seqq for VSXY/MDT;
            # cogsq for CPB) live silently took that field from an OLDER quarter, while the
            # backtest (build_universe_v2: one record per quarter, NaN stays NaN, missing debt
            # counts as 0) does not. A/B on 2026-09-29 data: identical BUY list, PAYX/NTNX weights
            # move in the 4th decimal — parity, not a pick change.
            latest = fund.sort_values(date_col).groupby("tic").tail(1).set_index("tic")

            # Compute fundamentals from Compustat fields
            # gross_margin = (saleq - cogsq) / saleq
            if "saleq" in latest.columns and "cogsq" in latest.columns:
                gm = (latest["saleq"] - latest["cogsq"]) / latest["saleq"].replace(0, np.nan)
                gm_map = gm.to_dict()
                mask = features["symbol"].isin(gm_map)
                features.loc[mask, "gross_margin"] = features.loc[mask, "symbol"].map(gm_map)
            elif "gpq" in latest.columns and "saleq" in latest.columns:
                gm = latest["gpq"] / latest["saleq"].replace(0, np.nan)
                gm_map = gm.to_dict()
                mask = features["symbol"].isin(gm_map)
                features.loc[mask, "gross_margin"] = features.loc[mask, "symbol"].map(gm_map)

            # operating_margin = oiadpq / saleq
            if "oiadpq" in latest.columns and "saleq" in latest.columns:
                om = latest["oiadpq"] / latest["saleq"].replace(0, np.nan)
                om_map = om.to_dict()
                mask = features["symbol"].isin(om_map)
                features.loc[mask, "operating_margin"] = features.loc[mask, "symbol"].map(om_map)

            # net_margin = niq / saleq
            if "niq" in latest.columns and "saleq" in latest.columns:
                nm = latest["niq"] / latest["saleq"].replace(0, np.nan)
                nm_map = nm.to_dict()
                mask = features["symbol"].isin(nm_map)
                features.loc[mask, "net_margin"] = features.loc[mask, "symbol"].map(nm_map)

            # roe = niq / seqq (annualized: *4 for quarterly) — matches backtest exactly
            # Note: ceqq (common equity) != seqq (stockholders' equity) for ~10% of stocks
            if "niq" in latest.columns and "seqq" in latest.columns:
                # F3 PARITY FIX (2026-07-25): require seqq > 0 like the backtest
                # (wrds_universe.py) — negative-equity + negative-income firms otherwise
                # compute spurious POSITIVE roe and slip into the value sleeve (measured:
                # 10 live-eligible names the backtest excludes, incl. SABR which was picked).
                roe = (latest["niq"] * 4) / latest["seqq"].where(latest["seqq"] > 0)
                roe_map = roe.to_dict()
                mask = features["symbol"].isin(roe_map)
                features.loc[mask, "roe"] = features.loc[mask, "symbol"].map(roe_map)

            # roa = niq / atq (annualized)
            if "niq" in latest.columns and "atq" in latest.columns:
                roa = (latest["niq"] * 4) / latest["atq"].replace(0, np.nan)
                roa_map = roa.to_dict()
                mask = features["symbol"].isin(roa_map)
                features.loc[mask, "roa"] = features.loc[mask, "symbol"].map(roa_map)

            # debt_to_equity = (dlttq + dlcq) / seqq — matches backtest exactly
            if "dlttq" in latest.columns and "seqq" in latest.columns:
                dlc = latest.get("dlcq", 0)
                if isinstance(dlc, (int, float)):
                    dlc = pd.Series(dlc, index=latest.index)
                de = (latest["dlttq"].fillna(0) + dlc.fillna(0)) / latest["seqq"].replace(0, np.nan)
                de_map = de.to_dict()
                mask = features["symbol"].isin(de_map)
                features.loc[mask, "debt_to_equity"] = features.loc[mask, "symbol"].map(de_map)

            # current_ratio = actq / lctq
            if "actq" in latest.columns and "lctq" in latest.columns:
                cr = latest["actq"] / latest["lctq"].replace(0, np.nan)
                cr_map = cr.to_dict()
                mask = features["symbol"].isin(cr_map)
                features.loc[mask, "current_ratio"] = features.loc[mask, "symbol"].map(cr_map)

            wrds_loaded = True
            log.info(f"Fundamentals loaded from WRDS Compustat ({len(latest)} tickers)")
        except Exception as e:
            log.warning(f"WRDS Compustat load failed: {e}, falling back to FMP")

    # ── EPS-SURPRISE BOOST REMOVED 2026-07-12 (divergence #61) ──
    # The backtest feature store NEVER populated eps_surprise_last, so the x1.15
    # momentum boost it controls never fired in ANY validated number. Live DID load
    # it from IBES (this block) => live ran an unvalidated live-only feature. PIT A/B
    # (research/eps_boost_test.py, 3 starts, 2018-25): boost ON = -0.5pp CAGR,
    # -0.02 Sharpe, worse on 3/3 starts. Removed for parity + performance. Re-enable
    # ONLY with fresh evidence.
    EPS_BOOST_ENABLED = False
    if EPS_BOOST_ENABLED and wrds_ibes.exists():
        try:
            ibes = pd.read_parquet(wrds_ibes)
            ibes["ANNDATS_ACT"] = pd.to_datetime(ibes["ANNDATS_ACT"], errors="coerce")
            # Filter to stocks with actual earnings reported
            ibes = ibes.dropna(subset=["ACTUAL", "MEANEST", "ANNDATS_ACT"])
            # EPS surprise = (actual - estimate) / |estimate|
            ibes["surprise"] = (ibes["ACTUAL"] - ibes["MEANEST"]) / ibes["MEANEST"].abs().replace(0, np.nan)
            # IBES uses TICKER (not tic), and OFTIC is the "official ticker"
            ticker_col = "OFTIC" if "OFTIC" in ibes.columns else "TICKER"
            latest_eps = ibes.sort_values("ANNDATS_ACT").groupby(ticker_col).last()
            surp_map = latest_eps["surprise"].to_dict()
            mask = features["symbol"].isin(surp_map)
            features.loc[mask, "eps_surprise_last"] = features.loc[mask, "symbol"].map(surp_map)
            log.info(f"Earnings loaded from WRDS IBES ({len(latest_eps)} tickers)")
        except Exception as e:
            log.warning(f"WRDS IBES load failed: {e}, falling back to FMP")

    # ── FALLBACK: FMP data (if WRDS not available) ──
    if not wrds_loaded:
        log.info("Using FMP fallback for fundamentals")
        ratios_file = data_dir / "fundamentals_ratios.parquet"
        income_file = data_dir / "fundamentals_income.parquet"
        metrics_file = data_dir / "fundamentals_metrics.parquet"
        earnings_file = data_dir / "fundamentals_earnings.parquet"

        if ratios_file.exists():
            ratios = pd.read_parquet(ratios_file)
            rat_date_col = "filing_date" if "filing_date" in ratios.columns else "date"
            if rat_date_col in ratios.columns:
                ratios[rat_date_col] = pd.to_datetime(ratios[rat_date_col])
            ratios_latest = ratios.sort_values(rat_date_col).groupby("symbol").last()
            for col in ["pe_ratio", "ps_ratio", "debt_to_equity", "current_ratio"]:
                if col in ratios_latest.columns:
                    sym_vals = ratios_latest[col].to_dict()
                    mask = features["symbol"].isin(sym_vals)
                    features.loc[mask, col] = features.loc[mask, "symbol"].map(sym_vals)

        if income_file.exists():
            income = pd.read_parquet(income_file)
            filing_col = "filing_date" if "filing_date" in income.columns else "date"
            if filing_col in income.columns:
                income[filing_col] = pd.to_datetime(income[filing_col])
            latest = income.sort_values(filing_col).groupby("symbol").last()
            for margin_name, num_col, den_col in [
                ("gross_margin", "gross_profit", "revenue"),
                ("operating_margin", "operating_income", "revenue"),
                ("net_margin", "net_income", "revenue"),
            ]:
                if num_col in latest.columns and den_col in latest.columns:
                    vals = (latest[num_col] / latest[den_col].replace(0, np.nan)).to_dict()
                    mask = features["symbol"].isin(vals)
                    features.loc[mask, margin_name] = features.loc[mask, "symbol"].map(vals)

        if metrics_file.exists():
            metrics = pd.read_parquet(metrics_file)
            met_date_col = "filing_date" if "filing_date" in metrics.columns else "date"
            if met_date_col in metrics.columns:
                metrics[met_date_col] = pd.to_datetime(metrics[met_date_col])
            latest = metrics.sort_values(met_date_col).groupby("symbol").last()
            for col in ["roe", "roa"]:
                if col in latest.columns:
                    sym_vals = latest[col].to_dict()
                    mask = features["symbol"].isin(sym_vals)
                    features.loc[mask, col] = features.loc[mask, "symbol"].map(sym_vals)

        if earnings_file.exists():
            earnings = pd.read_parquet(earnings_file)
            if "date" in earnings.columns:
                earnings["date"] = pd.to_datetime(earnings["date"])
            if "earnings_surprise" in earnings.columns:
                latest = earnings.sort_values("date").groupby("symbol").last()
                if "earnings_surprise" in latest.columns:
                    sym_vals = latest["earnings_surprise"].to_dict()
                    mask = features["symbol"].isin(sym_vals)
                    features.loc[mask, "eps_surprise_last"] = features.loc[mask, "symbol"].map(sym_vals)

    return features


def _fill_sector_relative(features, data_dir):
    """Compute ret_10d_vs_sector: each stock's 10d return minus its sector median."""
    sector_file = data_dir / "cache_sectors.json"
    if not sector_file.exists():
        return features

    with open(sector_file) as f:
        sector_map = _json.load(f)

    # Vectorized: map symbol → sector, then groupby (date, sector) → median
    features["_sector"] = features["symbol"].map(sector_map).fillna("Unknown")
    sector_medians = features.groupby(["date", "_sector"])["ret_10d"].transform("median")
    mask = features["ret_10d"].notna()
    features.loc[mask, "ret_10d_vs_sector"] = features.loc[mask, "ret_10d"] - sector_medians[mask]
    features.drop(columns=["_sector"], inplace=True)

    return features


# NOTE: the value sleeve now lives in multi_strategy_engine.strategy_value()
# (shared with the backtest — single source of truth). Imported at top.


_edgar_cache = {"mtime": None, "data": {}}


def _load_edgar_overlay(data_dir):
    """{feature: {sym: fresh_value}} from the EDGAR overlay, vintage-guarded: an entry
    applies ONLY if its period_end is NEWER than the symbol's current Compustat quarter
    (a fresh WRDS upload automatically retires stale overlay entries). Never raises —
    any problem returns {} and behavior is byte-identical to no-overlay."""
    try:
        f = data_dir / "edgar_feature_overlay.json"
        if not f.exists():
            return {}
        llm_f = data_dir / "edgar_llm_overlay.json"
        mtime = (f.stat().st_mtime, llm_f.stat().st_mtime if llm_f.exists() else 0)
        if _edgar_cache["mtime"] == mtime:
            return _edgar_cache["data"]
        ov = _json.load(open(f))
        # merge LLM-extracted entries (same proof gate upstream); XBRL wins conflicts
        if llm_f.exists():
            try:
                llm_feats = _json.load(open(llm_f)).get("features", {})
                for sym, feats in llm_feats.items():
                    tgt = ov.setdefault("features", {}).setdefault(sym, {})
                    for feat, e in feats.items():
                        tgt.setdefault(feat, e)
            except Exception:
                pass
        fund = pd.read_parquet(data_dir / "wrds" / "compustat_fundamentals_quarterly.parquet",
                               columns=["tic", "datadate"])
        fund["datadate"] = pd.to_datetime(fund["datadate"])
        last_dd = fund.groupby("tic")["datadate"].max()
        # LIVE-ELIGIBLE FEATURES ONLY — calibrated 2026-07-12: roe extractions test at
        # 99.2% next-quarter accuracy (deploy); gross_margin (~92%) and debt_to_equity
        # (~93%) are NOT provable to the 99% bar at any gate depth -> they stay
        # stale-clean in live signals (still generated in the overlay for research).
        out = {"roe": {}}  # eps removed 2026-07-12 (boost hurts; divergence #61)
        for sym, feats in ov.get("features", {}).items():
            dd = last_dd.get(sym)
            for feat, e in feats.items():
                if feat not in out:
                    continue
                # eps entries are vintage-guarded by the patcher (vs IBES, regenerated
                # daily by cron); the Compustat-datadate guard applies to the other three
                # LLM-sourced entries are RESEARCH-ONLY: holdout failed the 99% bar
                # (roe n=1; cogsq 17%, gm 33%) — only tag-extracted (XBRL) values,
                # certified at 98.9% over ~6k events, are live-eligible.
                if str(e.get("source", "")).startswith("llm"):
                    continue
                if feat == "eps_surprise_last" or (dd is None or pd.Timestamp(e["period_end"]) > dd):
                    v = float(e["value"])
                    # plausibility bounds (any source): absurd values never reach signals
                    if feat == "gross_margin" and not (-1.0 <= v <= 1.0):
                        continue
                    if feat in ("roe", "debt_to_equity", "eps_surprise_last") and abs(v) > 25:
                        continue
                    out[feat][sym] = v
        _edgar_cache["mtime"], _edgar_cache["data"] = mtime, out
        log.info("EDGAR overlay loaded: %s fresh symbols",
                 {k: len(v) for k, v in out.items()})
        return out
    except Exception as e:
        log.warning("EDGAR overlay load failed (%s) — proceeding without", e)
        return {}


class _EdgarOverlayUniverse:
    """Read-only proxy over the cached FastUniverse: overrides the live-eligible
    overlay feature(s) in get_feature_map with gate-validated EDGAR values (currently
    ROE ONLY — gm/d2e are generated for research but EXCLUDED from live, proven to
    hurt via partial-coverage rank inconsistency; see _load_edgar_overlay). Everything
    else delegates; the cached universe itself is never mutated."""

    def __init__(self, uni, overlay):
        object.__setattr__(self, "_uni", uni)
        object.__setattr__(self, "_ov", overlay)

    def __getattr__(self, name):
        return getattr(self._uni, name)

    def get_feature_map(self, date, feature, members=None):
        base = self._uni.get_feature_map(date, feature, members)
        fresh = self._ov.get(feature)
        if fresh:
            for sym, v in fresh.items():
                if members is None or sym in members:
                    base[sym] = v
        return base


def build_signals_v9(raw, enhanced_data=None, top_n=5, edgar_overlay=False):
    """
    Build signals using the v12 multi-sleeve strategy.

    edgar_overlay=True applies the gate-validated EDGAR freshness overlay to the
    live-eligible fundamental (ROE only; gm/d2e are research-only, excluded — proven
    to hurt) via a non-mutating universe proxy. Used by the signal server's SHADOW
    build; the live default stays False until the shadow diff is reviewed and flipped.

    Returns list of signal dicts in the same format as build_signals():
    [{symbol, probability, confidence, ml_mode, rank, is_top_5, signal}, ...]
    """
    global _uni_cache, _uni_cache_date

    # Use the last available trading date from raw data (not today — may be weekend/holiday)
    spy_df = raw.get("SPY", pd.DataFrame())
    if spy_df.empty or len(spy_df) == 0:
        log.error("SPY data unavailable — cannot build v9.6 signals")
        return []
    # PARTIAL-SESSION GUARD: must use the SAME rule as the price-matrix trim in
    # _build_universe, or `today` points at a session whose features were dropped and
    # every lookup returns {} -> empty book. Hence the shared _is_partial_session().
    _spy_idx = spy_df.index
    # same clock rule AND the same data rule (row coverage) as the matrix trim, computed here
    # from the raw bars so the two decisions cannot diverge
    _row_cov = _raw_last_row_coverage(raw, _spy_idx[-1])
    today = pd.Timestamp(_spy_idx[-2] if _is_partial_session(_spy_idx, _row_cov)
                         else _spy_idx[-1]).normalize()
    if _row_cov is not None and _row_cov < PARTIAL_ROW_MIN_COVERAGE:
        log.warning("Partial-session guard (data rule): only %.0f%% of names have printed for %s "
                    "-- computing on %s", _row_cov * 100, str(_spy_idx[-1])[:10], str(today)[:10])

    # Rebuild universe if needed (once per day)
    if _uni_cache is None or _uni_cache_date != today:
        log.info("Building v9.6 FastUniverse ...")
        _uni_cache = _build_universe(raw, enhanced_data)
        # RECONCILE: SPY's bars and the price matrix are assembled separately, so a
        # disagreement about the last completed session is possible (SPY printing while
        # a symbol has not, DST, a provider gap). Trust the universe — asking it for a
        # date it lacks yields an empty book with no error. Loud, because silent
        # degradation is the failure mode this whole guard exists to prevent.
        _feat_dates = getattr(_uni_cache, "_feat_by_date", {})
        if _feat_dates and today not in _feat_dates:
            _actual = max(_feat_dates)
            log.warning("Session mismatch: signal date %s absent from universe features "
                        "(last available %s) — using the universe's date instead.",
                        str(today)[:10], str(_actual)[:10])
            today = pd.Timestamp(_actual).normalize()
        _uni_cache_date = today
        # v12: SI DISABLED — hurts returns by -3pp (verified May 2026)
        # _load_si_change_data(_uni_cache)
        _uni_cache._si_change_rank = {}
        # F1 PARITY FIX (2026-07-25): momentum + lowvol sleeves must select from the FULL
        # SP1500, like every validated backtest (which overrides get_sp500 to SP1500).
        # Live never did — FastUniverse.get_sp500 served SP500-only (503 names), so 65% of
        # the book picked from 1/3 of the validated universe. Live-mirror A/B (clean PIT
        # membership, 2018-25 3-start): SP500-pool −15.7pp CAGR / −0.38 Sharpe vs SP1500.
        # Value sleeve already used SP1500 (below); this aligns mom/s5 with it.
        _uni_cache.get_sp500 = lambda d: get_sp1500_on_date(d)
        # F2 PARITY FIX (2026-07-25): rev-surprise ×1.10 / beat-streak ×1.05 momentum
        # boosts were ACTIVE live but zeroed in every validated number (FastUniverse loads
        # fundamentals_earnings.parquet unconditionally; validated runs clear_deployed()).
        # Unvalidated live-only behavior — same class as the removed eps boost (−0.5pp).
        _uni_cache._earnings_signals = {}
        _uni_cache._revenue_surprise = {}
        _uni_cache._beat_streak = {}
        # PARITY GUARD (2026-07-26): the F1 bug ran undetected for 2 months because nothing
        # ever asserted the sleeve pool size. Fail LOUD (Telegram + log) if the momentum
        # pool is ever not SP1500-sized again — silent degradation is the real enemy here.
        try:
            _pool = _uni_cache.get_sp500(today)
            _pool_n = len(_pool)
            # Count names the sleeves can ACTUALLY rank (membership alone is not enough:
            # get_sp1500_on_date falls back to FMP-SP500-only if sp1500_members.json is
            # missing, and price coverage can collapse independently of membership —
            # either failure silently shrinks the effective universe. Hardened 2026-07-26
            # after the adversarial audit flagged that a membership-only count would read
            # ~1500 even if S&P600 price data vanished).
            _r252 = _uni_cache.get_feature_map(today, "ret_252d")
            _usable_n = sum(1 for s in _pool if _r252.get(s) is not None)
            _PARITY_MIN_POOL = 1200          # SP1500 minus normal membership churn/coverage
            if _pool_n < _PARITY_MIN_POOL or _usable_n < _PARITY_MIN_POOL:
                msg = (f"PARITY ALARM: momentum/lowvol pool is {_pool_n} names "
                       f"({_usable_n} with usable ret_252d; expected ~1500, floor "
                       f"{_PARITY_MIN_POOL}). Sleeves may be running "
                       f"an UNVALIDATED universe — this is the 2026-07-25 F1 regression.")
                log.error(msg)
                try:   # same direct-API pattern as data_freshness_check._send_telegram
                    import os as _os, requests
                    _tok = (_os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
                    _chat = (_os.getenv("TELEGRAM_CHAT_ID") or "").strip()
                    if _tok and _chat:
                        requests.post(f"https://api.telegram.org/bot{_tok}/sendMessage",
                                      json={"chat_id": _chat, "text": f"🚨 {msg}"}, timeout=10)
                except Exception:
                    pass
            else:
                log.info(f"parity guard OK: sleeve pool {_pool_n} names "
                         f"({_usable_n} rankable) (SP1500)")
        except Exception as _e:
            log.error(f"PARITY GUARD could not verify pool size: {_e}")

        # FEATURE-COVERAGE GUARD (2026-08-11). The pool guard above only checks ret_252d.
        # But get_feature_map DROPS NaN keys entirely, so a name missing ANY filter feature
        # is silently removed from that sleeve — `dist_sma200.get(sym, 0)` returns 0, fails
        # `> 0`, and the name is gone with no log, no count, no alarm. That is the exact
        # mechanism behind the SNDK incident (2026-08-04); the partial-session guard fixes
        # the unclosed-bar CAUSE, but genuinely missing vendor/fundamentals data produces the
        # same silent exclusion. Observed 2026-08-11: 18 names had no dist_sma200, 108 no roe,
        # 44 no gross_margin — invisible until someone went looking.
        # So: log coverage EVERY build (observability), and alarm if a source collapses.
        # Floors are set well under normal coverage (roe ~93%, gm ~97%, sma200 ~99%) — they
        # are here to catch a source VANISHING (a parquet failing to load takes roe to ~0),
        # not to police ordinary churn.
        try:
            _cov_floor = {"dist_sma200": 0.85, "roe": 0.70, "gross_margin": 0.80,
                          "vol_60d": 0.85, "ret_126d": 0.85}
            _pool2 = _uni_cache.get_sp500(today)
            _n = max(len(_pool2), 1)
            _cov_msgs, _cov_bad = [], []
            _cov_frac, _cov_missing = {}, {}
            for _f, _floor in _cov_floor.items():
                _m = _uni_cache.get_feature_map(today, _f)
                _have = sum(1 for s in _pool2 if _m.get(s) is not None)
                _frac = _have / _n
                _cov_frac[_f] = round(_frac, 4)
                _cov_msgs.append(f"{_f} {_frac:.1%} ({_n - _have} missing)")
                if _frac < _floor:
                    _cov_bad.append(f"{_f} {_frac:.1%} < floor {_floor:.0%}")
                    _cov_missing[_f] = [s for s in _pool2 if _m.get(s) is None]
            # VENDOR-HOLE REPORT — the root-cause signal, logged every universe build.
            _hs = _HOLE_STATS
            if _hs["symbols"]:
                _top = sorted(_hs["dates"].items(), key=lambda kv: -kv[1])[:4]
                _worst_frac = (_top[0][1] / _n) if _top else 0.0
                log.warning("VENDOR HOLES filled: %d symbols, %d cells (%d unfillable >3d); "
                            "worst dates %s",
                            _hs["symbols"], _hs["cells"], _hs["unfilled"],
                            [(str(d)[:10], k) for d, k in _top])
                if _worst_frac > 0.20:
                    _hmsg = (f"VENDOR HOLE: {str(_top[0][0])[:10]} missing from "
                             f"{_top[0][1]}/{_n} symbols ({_worst_frac:.0%}). Filled by carry; "
                             f"without the fill dist_sma200 would have collapsed. Check Polygon.")
                    log.error(_hmsg)
                    try:
                        import os as _os2, requests as _rq
                        _t = (_os2.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
                        _c = (_os2.getenv("TELEGRAM_CHAT_ID") or "").strip()
                        if _t and _c:
                            _rq.post(f"https://api.telegram.org/bot{_t}/sendMessage",
                                     json={"chat_id": _c, "text": f"⚠️ {_hmsg}"}, timeout=10)
                    except Exception:
                        pass
            log.info("feature coverage: %s", " | ".join(_cov_msgs))
            # PUBLISH so ibkr_engine can refuse to rebalance on a degraded book
            globals()["LAST_COVERAGE"] = _cov_frac
            globals()["LAST_COVERAGE_OK"] = not _cov_bad
            if _cov_bad:
                try:
                    _worst = min(_cov_bad, key=lambda x: x)
                    _ts = datetime.now(ZoneInfo("US/Eastern")).strftime("%Y%m%d-%H%M%S")
                    _fp = (Path(__file__).resolve().parent / "data" /
                           f"coverage_forensics_{_ts}.json")
                    _feat = sorted(_cov_missing, key=lambda k: _cov_frac[k])[0]
                    _coverage_forensics(raw, today, _pool2, _cov_missing[_feat], str(_fp))
                except Exception as _e:
                    log.error("coverage forensics failed: %s", _e)
            if _cov_bad:
                msg = ("COVERAGE ALARM: " + "; ".join(_cov_bad) +
                       ". Names missing a filter feature are SILENTLY dropped from that "
                       "sleeve — check the fundamentals/price feeds before trusting signals.")
                log.error(msg)
                try:
                    import os as _os, requests
                    _tok = (_os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
                    _chat = (_os.getenv("TELEGRAM_CHAT_ID") or "").strip()
                    if _tok and _chat:
                        requests.post(f"https://api.telegram.org/bot{_tok}/sendMessage",
                                      json={"chat_id": _chat, "text": f"🚨 {msg}"}, timeout=10)
                except Exception:
                    pass
        except Exception as _e:
            log.error(f"COVERAGE GUARD could not verify feature coverage: {_e}")

        log.info("v9.6 FastUniverse ready (SP1500 pool, earnings boosts OFF — parity)")

    uni = _uni_cache
    if edgar_overlay:
        # SHADOW/overlay path only — wraps (never mutates) the cached universe.
        overlay = _load_edgar_overlay(Path(__file__).resolve().parent / "data")
        if any(overlay.get(k) for k in ("roe", "gross_margin", "debt_to_equity")):
            uni = _EdgarOverlayUniverse(_uni_cache, overlay)

    # Run all strategies (day_idx=0 forces rebalance)
    s4_active = {}
    t1 = strategy1_momentum_reversal(today, uni, 0, top_n=top_n, rebal_days=1)
    t3 = strategy3_sector_rotation(today, uni, 0, rebal_days=1)
    t4 = strategy4_index_inclusion(today, uni, 0, s4_active)
    t5 = strategy5_lowvol_quality(today, uni, 0, top_n=10, rebal_days=1)

    # Value strategy: uses SP1500 membership for the value stock universe
    members = get_sp1500_on_date(today)
    t7 = strategy_value(uni, today, members, top_n=10)

    targets = {
        "s1_momentum": t1 or {},
        "s3_sector": t3 or {},
        "s4_inclusion": t4 or {},
        "s5_lowvol": t5 or {},
        "s7_value": t7 or {},
    }

    # Breadth blend
    dist_sma50 = uni.get_feature_map(today, "dist_sma50")
    breadth = sum(1 for v in dist_sma50.values() if v > 0) / max(len(dist_sma50), 1) if dist_sma50 else 0.5
    blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))

    # Legacy regime tilts removed — not in backtest, negligible impact
    paused = set()

    # UMD crash regime (matches backtest: shift to value-heavy when momentum crashes).
    # Price-based, real-time UMD from our own price matrix — replaces the ~46-day-lagged
    # Fama-French file (which made this detector blind in live). Shared code path with the
    # backtest via compute_price_umd_20d (validated behavior-neutral, research/phase11*).
    umd_crash = False
    try:
        # Live universe carries only ~380 trading days, so compute over all of it (cheap,
        # ~18 monthly reforms) for maximum momentum warmup before today's 20-day sum.
        umd_20d = compute_price_umd_20d(
            uni.prices, uni._daily_returns, get_sp1500_on_date)
        recent = umd_20d.loc[:today] if len(umd_20d) else umd_20d
        if len(recent) > 0 and pd.notna(recent.iloc[-1]):
            val = float(recent.iloc[-1])
            if val < UMD_CRASH_THRESHOLD:
                umd_crash = True
                log.info(f"UMD CRASH detected: 20d sum = {val:.4f} — shifting to value-heavy weights")
            else:
                log.info(f"UMD ok: 20d sum = {val:.4f} (crash threshold {UMD_CRASH_THRESHOLD})")
        else:
            log.warning("UMD: insufficient price history for crash detection — defaulting to no-crash")
    except Exception as e:
        log.warning(f"UMD check failed: {e}")

    # Override bull weights during momentum crash (shared constant — matches backtest)
    if umd_crash:
        crash_weights = PROD_WEIGHTS_CRASH
    else:
        crash_weights = None

    # Combine with dynamic allocation
    combined = {}
    for name, bull_pct in STRATEGY_CONFIG_BULL:
        if name in paused:
            continue
        if crash_weights:
            bull_pct = crash_weights.get(name, 0)
        bear_pct = dict(STRATEGY_CONFIG_BEAR).get(name, 0)
        cap = bull_pct * blend + bear_pct * (1 - blend)
        for sym, w in targets.get(name, {}).items():
            if w > 0:
                combined[sym] = combined.get(sym, 0) + w * cap

    # Apply constraints (must match backtest exactly)
    # 1) Single-name cap: 15% (v12: matches backtest cap=0.15)
    for sym in list(combined):
        if combined[sym] > 0.15:
            combined[sym] = 0.15
    # 2) Sector cap: removed (backtest has no sector cap)
    # 3) Gross exposure cap: 100%
    gross = sum(combined.values())
    if gross > 1.0:
        for sym in combined:
            combined[sym] /= gross
    combined = {s: w for s, w in combined.items() if w >= 0.005}

    # Convert to signal format
    # Normalize scores to 0-1 range for trading bot compatibility
    # (bot uses probability thresholds like 0.65 for CAUTIOUS mode)
    max_score = max(combined.values()) if combined else 1.0
    if max_score <= 0:
        max_score = 1.0

    # All SP1500 symbols should appear in the output (falls back to SP500 if Norgate unavailable)
    members = get_sp1500_on_date(today)
    signals = []
    for sym in sorted(members):
        raw_score = combined.get(sym, 0.0)
        # Normalize: top pick → ~0.95, others scale proportionally
        norm_prob = min(0.95, (raw_score / max_score) * 0.95) if raw_score > 0 else 0.0
        signals.append({
            "symbol": sym,
            "probability": round(norm_prob, 4),
            "confidence": round(norm_prob, 4),
            "ml_mode": "V9",
        })

    # Sort by score and assign ranks
    signals.sort(key=lambda x: x["probability"], reverse=True)
    for i, sig in enumerate(signals):
        rank = i + 1
        sig["rank"] = rank
        sig["is_top_5"] = rank <= top_n
        # BUY all stocks with positive signal weight (matches backtest behavior)
        # Backtest holds ALL stocks from combined sleeves (~25-28 positions)
        sig["signal"] = "BUY" if sig["probability"] > 0 else "HOLD"

    buy_count = sum(1 for s in signals if s["signal"] == "BUY")
    log.info("v9.6 signals: %d total, %d BUY (breadth=%.1f%%, blend=%.2f)",
             len(signals), buy_count, breadth * 100, blend)

    return signals
