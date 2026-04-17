"""
ML Data Pipeline — fetches 15 years of daily OHLCV from Yahoo Finance,
computes technical + cross-asset + calendar features, and saves
a Parquet feature matrix ready for model training.
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# ── FMP credentials ───────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

# Individual stocks that have quarterly earnings (ETFs get zeros)
STOCK_SYMBOLS = [
    "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
]

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    # Individual stocks (12)
    "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
    # Sector ETFs (11)
    "XLE","XLF","XLV","XLI","XLK","XLY","XLP","XLU","XLRE","XLB","XLC",
    # International ETFs (6)
    "EWZ","EWJ","FXI","INDA","EFA","EEM",
    # Commodities (4)
    "GLD","SLV","USO","DBC",
    # Bonds (3)
    "TLT","HYG","LQD",
    # Volatility (1)
    "VIXY",
]


# Cross-asset reference symbols (may overlap with UNIVERSE)
CROSS_ASSET = ["SPY", "VIXY", "TLT"]

ALL_SYMBOLS = sorted(set(UNIVERSE + CROSS_ASSET))

# ── Date range: 15 years + 220-day warmup for long SMAs ──────────────────────
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 15 + 220)).strftime("%Y-%m-%d")

OUTPUT_DIR           = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE          = OUTPUT_DIR / "features.parquet"
EARNINGS_CACHE_DIR   = OUTPUT_DIR / "earnings_cache"
EARNINGS_CACHE_DIR.mkdir(exist_ok=True)
EARNINGS_CACHE_TTL   = 7   # days before re-fetching from FMP

def fetch_bars(symbol: str) -> pd.DataFrame:
    """Fetch daily OHLCV bars for one symbol from Yahoo Finance.

    yfinance returns split- and dividend-adjusted closes automatically
    when auto_adjust=True (the default). Downloads are batched internally
    so no manual rate-limit sleep is needed for individual calls.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=START_DATE, end=END_DATE, auto_adjust=True)
    except Exception as exc:
        print(f"  WARNING: could not fetch {symbol}: {exc}")
        return pd.DataFrame()

    if df.empty:
        print(f"  WARNING: no bars returned for {symbol}")
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index).tz_localize(None)   # strip tz
    df.index.name = "date"
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df.sort_index()


# ── Earnings surprise helpers ─────────────────────────────────────────────────

def _parse_fmp_earnings(data: list) -> pd.DataFrame:
    """Parse FMP /stable/earnings JSON into DataFrame[date, earnings_surprise]."""
    rows = []
    for item in data:
        try:
            # Skip future entries where actual EPS hasn't been reported yet
            if item.get("epsActual") is None:
                continue
            date     = pd.Timestamp(item["date"])
            actual   = float(item["epsActual"])
            estimate = item.get("epsEstimated")
            if estimate is None:
                continue
            estimate = float(estimate)
            if abs(estimate) > 1e-9:
                surprise = (actual - estimate) / abs(estimate)
            else:
                surprise = 0.0
            rows.append({"date": date, "earnings_surprise": float(np.clip(surprise, -2.0, 2.0))})
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame(columns=["date", "earnings_surprise"])
    return pd.DataFrame(rows).dropna().sort_values("date").reset_index(drop=True)


def fetch_earnings_surprises(symbol: str) -> pd.DataFrame:
    """
    Fetch historical earnings surprise history for one stock from FMP.
    Uses the /stable/earnings endpoint (epsActual / epsEstimated fields).
    Caches results to EARNINGS_CACHE_DIR for EARNINGS_CACHE_TTL days.
    ETFs should not be passed here; use an empty DataFrame for those.
    """
    if not FMP_API_KEY:
        return pd.DataFrame(columns=["date", "earnings_surprise"])

    cache_file = EARNINGS_CACHE_DIR / f"{symbol}.json"

    # Serve from cache if fresh enough
    if cache_file.exists():
        age = (datetime.today() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age < EARNINGS_CACHE_TTL:
            try:
                with open(cache_file) as f:
                    return _parse_fmp_earnings(json.load(f))
            except Exception:
                pass   # corrupt cache — fall through to re-fetch

    url = (
        f"https://financialmodelingprep.com/stable/earnings"
        f"?symbol={symbol}&apikey={FMP_API_KEY}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, list):
            raise ValueError(f"Unexpected FMP response type: {type(data)}")
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return _parse_fmp_earnings(data)
    except Exception as exc:
        print(f"  WARNING: FMP earnings fetch failed for {symbol}: {exc}")
        return pd.DataFrame(columns=["date", "earnings_surprise"])


# ── Technical feature helpers ─────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series):
    ema12   = close.ewm(span=12, adjust=False).mean()
    ema26   = close.ewm(span=26, adjust=False).mean()
    line    = ema12 - ema26
    signal  = line.ewm(span=9, adjust=False).mean()
    return line, signal


def compute_bb_position(close: pd.Series, window: int = 20) -> pd.Series:
    """Percent-B: where price sits within the Bollinger Bands (0 = lower, 1 = upper)."""
    sma    = close.rolling(window).mean()
    std    = close.rolling(window).std()
    upper  = sma + 2 * std
    lower  = sma - 2 * std
    band_w = upper - lower
    return (close - lower) / band_w.replace(0, np.nan)


def compute_obv_trend(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    direction = np.sign(close.diff())
    obv       = (direction * volume).fillna(0).cumsum()
    return obv.rolling(window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )


def compute_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    v = df["volume"]
    feat = pd.DataFrame(index=df.index)

    # Returns
    for n in [5, 10, 20, 60, 120]:
        feat[f"ret_{n}d"] = c.pct_change(n)

    # Rolling volatility (annualised)
    daily_ret = c.pct_change()
    for n in [10, 20, 60]:
        feat[f"vol_{n}d"] = daily_ret.rolling(n).std() * np.sqrt(252)

    # RSI
    feat["rsi_14"] = compute_rsi(c, 14)

    # MACD
    feat["macd_line"], feat["macd_signal"] = compute_macd(c)

    # Bollinger Band position
    feat["bb_position"] = compute_bb_position(c, 20)

    # Distance from SMAs
    for n in [50, 200]:
        sma = c.rolling(n).mean()
        feat[f"dist_sma{n}"] = (c - sma) / sma

    # New high flags
    feat["new_high_20d"] = (c == c.rolling(20).max()).astype(int)
    feat["new_high_50d"] = (c == c.rolling(50).max()).astype(int)

    # Volume ratio vs 20-day average
    avg_vol = v.rolling(20).mean()
    feat["vol_ratio_20d"] = v / avg_vol.replace(0, np.nan)

    # OBV 20-day trend (slope)
    feat["obv_trend_20d"] = compute_obv_trend(c, v, 20)

    return feat


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ML Data Pipeline — Yahoo Finance 15-Year Feature Matrix")
    print(f"Date range : {START_DATE} → {END_DATE}")
    print(f"Symbols    : {len(ALL_SYMBOLS)} total ({len(UNIVERSE)} universe + cross-asset)")
    print("=" * 60)

    # 1. Fetch all raw bars — batch download first for speed, then fall back
    #    to per-symbol for any that are missing from the batch.
    print("Batch-downloading all symbols via yfinance ...")
    batch = yf.download(
        ALL_SYMBOLS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    raw: dict[str, pd.DataFrame] = {}
    for sym in ALL_SYMBOLS:
        try:
            if isinstance(batch.columns, pd.MultiIndex):
                sym_df = batch.xs(sym, axis=1, level=1).copy()
            else:
                # Single-symbol download returns flat columns
                sym_df = batch.copy()

            sym_df.columns = sym_df.columns.str.lower()
            sym_df = sym_df[["open", "high", "low", "close", "volume"]].dropna(how="all")
            sym_df.index = pd.to_datetime(sym_df.index).tz_localize(None)
            sym_df.index.name = "date"
            raw[sym] = sym_df.sort_index()
        except (KeyError, Exception):
            raw[sym] = pd.DataFrame()

    # Fall back: re-fetch individually any symbol that came back empty
    missing = [s for s in ALL_SYMBOLS if raw[s].empty]
    if missing:
        print(f"\nFetching {len(missing)} missing symbol(s) individually ...")
        for i, sym in enumerate(missing, 1):
            print(f"  [{i}/{len(missing)}] {sym} ...", end=" ", flush=True)
            raw[sym] = fetch_bars(sym)
            print(f"{len(raw[sym])} bars")
            time.sleep(0.5)

    # Summary
    print()
    for sym in ALL_SYMBOLS:
        n = len(raw[sym])
        print(f"  {sym:<8} {n:,} bars" + ("  ⚠ empty" if n == 0 else ""))

    # 2. Build cross-asset feature frame aligned to a common date index
    #    Use SPY as the reference calendar
    spy_close = raw.get("SPY", pd.DataFrame())
    if spy_close.empty:
        sys.exit("ERROR: SPY data is required as the date calendar.")

    date_index = spy_close.index  # all trading days

    # Trading-day ordinal map: {Timestamp → int}, used for days_since_earnings
    td_map = {ts: i for i, ts in enumerate(date_index)}

    # 2b. Fetch earnings surprise history (stock symbols only; ETFs get zeros)
    print("\nFetching earnings surprise data from FMP API ...")
    earnings_data: dict = {}
    if FMP_API_KEY:
        for idx_s, sym in enumerate(STOCK_SYMBOLS, 1):
            cache_file = EARNINGS_CACHE_DIR / f"{sym}.json"
            source = "cache" if (cache_file.exists() and
                                  (datetime.today() - datetime.fromtimestamp(
                                      cache_file.stat().st_mtime)).days < EARNINGS_CACHE_TTL) else "API"
            print(f"  [{idx_s}/{len(STOCK_SYMBOLS)}] {sym:<6} ({source}) ...", end=" ", flush=True)
            df_earn = fetch_earnings_surprises(sym)
            earnings_data[sym] = df_earn
            print(f"{len(df_earn)} reports" if not df_earn.empty else "no data")
            if source == "API":
                time.sleep(0.3)   # gentle rate-limit when hitting FMP
    else:
        print("  FMP_API_KEY not set — earnings features will be zero for all symbols")

    def aligned_close(sym):
        df = raw.get(sym, pd.DataFrame())
        if df.empty:
            return pd.Series(np.nan, index=date_index, name=sym)
        return df["close"].reindex(date_index)

    spy_c  = aligned_close("SPY")
    vixy_c = aligned_close("VIXY")
    tlt_c  = aligned_close("TLT")

    # Cross-asset features
    cross = pd.DataFrame(index=date_index)
    for n in [5, 10, 20, 60, 120]:
        cross[f"spy_ret_{n}d"]  = spy_c.pct_change(n)
        cross[f"tlt_ret_{n}d"]  = tlt_c.pct_change(n)
    cross["vixy_level"]         = vixy_c
    cross["vixy_ret_5d"]        = vixy_c.pct_change(5)
    cross["vixy_ret_20d"]       = vixy_c.pct_change(20)

    # Calendar features
    cross["day_of_week"] = date_index.dayofweek          # 0=Mon … 4=Fri
    cross["month"]       = date_index.month
    cross["quarter"]     = date_index.quarter

    # 3. Compute per-symbol features + target, then concatenate
    print("\nComputing features ...")
    all_frames = []

    for i, sym in enumerate(UNIVERSE, 1):
        df = raw.get(sym, pd.DataFrame())
        if df.empty:
            print(f"  [{i:2d}/{len(UNIVERSE)}] {sym}: skipped (no data)")
            continue

        # Align to common date index
        df = df.reindex(date_index)

        # Per-symbol technical features
        feat = compute_symbol_features(df)

        # Target: forward 10-day return > 2% (computed BEFORE dropping NaNs)
        fwd_ret = df["close"].pct_change(10).shift(-10)
        feat["target"] = (fwd_ret > 0.02).astype("Int8")  # nullable int → NaN for last rows

        # Merge cross-asset features
        feat = feat.join(cross, how="left")

        # Add symbol identifier
        feat.insert(0, "symbol", sym)
        feat.index.name = "date"

        # Drop rows where the target is NaN (last 10 days — look-ahead unavailable)
        feat = feat[feat["target"].notna()]

        # Drop rows with any NaN feature (warmup period + genuine gaps)
        # Keep target separate so we don't accidentally forward-fill it
        feature_cols = [c for c in feat.columns if c not in ("symbol", "target")]
        feat = feat.dropna(subset=feature_cols)

        n_rows = len(feat)
        pos_pct = feat["target"].mean() * 100
        print(f"  [{i:2d}/{len(UNIVERSE)}] {sym}: {n_rows} rows  |  target+ {pos_pct:.1f}%")

        all_frames.append(feat.reset_index())

    if not all_frames:
        sys.exit("ERROR: no feature data produced — check yfinance and symbol list.")

    # 4. Combine and save
    print("\nCombining all symbols ...")
    master = pd.concat(all_frames, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    master = master.sort_values(["date", "symbol"]).reset_index(drop=True)

    print(f"Final dataset: {len(master):,} rows × {len(master.columns)} columns")
    print(f"Date range in data: {master['date'].min().date()} → {master['date'].max().date()}")
    print(f"Target balance: {master['target'].mean()*100:.1f}% positive")

    print(f"\nSaving to {OUTPUT_FILE} ...")
    master.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow", compression="snappy")
    print(f"Done.  File size: {OUTPUT_FILE.stat().st_size / 1_048_576:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
