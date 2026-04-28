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
import yfinance as yf  # kept as fallback only
from massive_data_provider import MassiveDataProvider
from dotenv import load_dotenv

# ── FMP credentials ───────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

# ── Universe (loaded from sp500_universe module) ─────────────────────────────
from sp500_universe import get_stock_symbols, get_etf_symbols, get_full_universe, get_cross_asset, get_all_symbols
from sp500_history import get_sp500_on_date, load_sp500_changes
from unified_backtester import SYMBOL_SECTOR

STOCK_SYMBOLS = get_stock_symbols()
UNIVERSE = get_full_universe()
CROSS_ASSET = get_cross_asset()
_BASE_SYMBOLS = get_all_symbols()

# ── Date range: 10 years + 220-day warmup for long SMAs ──────────────────────
# Massive (Polygon) Stocks Starter plan provides 10 years of history
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 10 + 220)).strftime("%Y-%m-%d")


# ── Ticker format normalization ──────────────────────────────────────────────
# sp500_history uses dashes (BRK-B), Massive uses dots (BRK.B),
# features.parquet uses dashes (matching sp500_history).

def _ticker_to_massive(sym: str) -> str:
    """Convert sp500_history format (BRK-B) to Massive/Polygon format (BRK.B)."""
    return sym.replace("-", ".")


def _ticker_from_massive(sym: str) -> str:
    """Convert Massive/Polygon format (BRK.B) to sp500_history format (BRK-B)."""
    # Only convert class-share dots (single letter after dot)
    parts = sym.rsplit(".", 1)
    if len(parts) == 2 and len(parts[1]) == 1:
        return f"{parts[0]}-{parts[1]}"
    return sym


# ── Historical S&P 500 universe (survivorship bias fix) ──────────────────────
# Build the union of ALL tickers that were ever in the S&P 500 during the
# training window. This ensures delisted/acquired/bankrupt stocks are fetched
# and included in training data.

def _build_historical_universe() -> list:
    """
    Build the complete set of tickers that were in the S&P 500 at any point
    during the training window. Samples every 30 days for efficiency.
    """
    print("Building historical S&P 500 universe (survivorship bias fix) ...")
    load_sp500_changes()

    historical = set()
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    # Sample every 30 days (constituent changes happen monthly at most)
    d = start
    n_dates = 0
    while d <= end:
        members = get_sp500_on_date(d)
        historical |= members
        d += pd.Timedelta(days=30)
        n_dates += 1

    # Also include current symbols and cross-asset
    base = set(_BASE_SYMBOLS)
    new_from_history = historical - base
    all_combined = sorted(base | historical)

    print(f"  Current universe: {len(base)} symbols")
    print(f"  Historical S&P 500 members found: {len(historical)} unique tickers")
    print(f"  New tickers from history: {len(new_from_history)}")
    if new_from_history:
        sample = sorted(new_from_history)[:20]
        print(f"  Examples: {', '.join(sample)}")
    print(f"  Total download universe: {len(all_combined)} symbols")

    return all_combined


ALL_SYMBOLS = _build_historical_universe()

OUTPUT_DIR           = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE          = OUTPUT_DIR / "features.parquet"
EARNINGS_CACHE_DIR   = OUTPUT_DIR / "earnings_cache"
EARNINGS_CACHE_DIR.mkdir(exist_ok=True)
EARNINGS_CACHE_TTL   = 7   # days before re-fetching from FMP

_massive_provider = None

def _get_massive():
    global _massive_provider
    if _massive_provider is None:
        _massive_provider = MassiveDataProvider(validate_vs_yfinance=False)
    return _massive_provider


def fetch_bars(symbol: str) -> pd.DataFrame:
    """Fetch daily OHLCV bars for one symbol from Massive (Polygon).
    Handles ticker format normalization (BRK-B -> BRK.B for Massive).
    Falls back to yfinance if Massive fails."""
    massive_sym = _ticker_to_massive(symbol)
    try:
        provider = _get_massive()
        df = provider.fetch_ticker_bars(massive_sym, START_DATE, END_DATE, adjusted=True)
        if len(df) > 0:
            return df
    except Exception as exc:
        print(f"  WARNING: Massive fetch failed for {symbol} (as {massive_sym}): {exc}")

    # Fallback to yfinance (uses original symbol format)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=START_DATE, end=END_DATE, auto_adjust=True)
    except Exception as exc:
        print(f"  WARNING: yfinance fallback failed for {symbol}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index).tz_localize(None)
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


# ── Fundamental feature columns (allowed to be NaN for ETFs) ─────────────────
FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]


def _load_fundamental_data(date_index):
    """Load all fundamental parquet files into a dict."""
    files = {
        "income": OUTPUT_DIR / "fundamentals_income.parquet",
        "ratios": OUTPUT_DIR / "fundamentals_ratios.parquet",
        "metrics": OUTPUT_DIR / "fundamentals_metrics.parquet",
        "estimates": OUTPUT_DIR / "fundamentals_estimates.parquet",
        "earnings": OUTPUT_DIR / "fundamentals_earnings.parquet",
        "insiders": OUTPUT_DIR / "fundamentals_insiders.parquet",
    }

    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        print(f"\n  WARNING: Missing fundamental files: {missing}")
        print("  Run fmp_fundamentals_pipeline.py first. Skipping fundamental features.")
        return None

    print("\n  Loading fundamental data ...")
    data = {}
    for key, path in files.items():
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "filing_date" in df.columns:
            df["filing_date"] = pd.to_datetime(df["filing_date"])
        data[key] = df
        print(f"    {key}: {len(df):,} rows")

    return data


def _compute_fundamental_features(symbol, date_index, fund_data):
    """
    Compute fundamental features for one symbol, using filingDate
    for timing to avoid look-ahead bias. Returns a DataFrame indexed
    by date, or None if no data.
    """
    income = fund_data["income"]
    ratios = fund_data["ratios"]
    metrics = fund_data["metrics"]
    earnings = fund_data["earnings"]
    estimates = fund_data["estimates"]
    insiders = fund_data["insiders"]

    sym_income = income[income["symbol"] == symbol].sort_values("date")
    sym_ratios = ratios[ratios["symbol"] == symbol].sort_values("date")
    sym_metrics = metrics[metrics["symbol"] == symbol].sort_values("date")
    sym_earnings = earnings[earnings["symbol"] == symbol].sort_values("date")
    sym_estimates = estimates[estimates["symbol"] == symbol].sort_values("date")
    sym_insiders = insiders[insiders["symbol"] == symbol].sort_values("date") if len(insiders) > 0 else pd.DataFrame()

    if len(sym_income) == 0:
        return None

    feat = pd.DataFrame(index=date_index)

    # ── Map quarterly data to daily dates using filingDate ────────────
    # For each trading day, the "latest known" quarter is the most recent
    # one with filingDate <= that trading day.

    # Build lookup: for each filing_date, store the quarter's data
    inc_by_filing = sym_income.dropna(subset=["filing_date"]).sort_values("filing_date")
    rat_by_date = sym_ratios.sort_values("date")
    met_by_date = sym_metrics.sort_values("date")

    if len(inc_by_filing) == 0:
        return None

    # ── Growth rates ─────────────────────────────────────────────────
    # Revenue YoY: compare to same quarter one year ago (4 quarters back)
    inc_sorted = inc_by_filing.copy()
    inc_sorted["revenue_yoy"] = np.nan
    inc_sorted["eps_yoy"] = np.nan
    inc_sorted["revenue_qoq"] = np.nan

    for idx in range(len(inc_sorted)):
        row = inc_sorted.iloc[idx]
        rev = row.get("revenue")
        eps = row.get("eps_diluted") or row.get("eps")

        # YoY: find same period 4 quarters ago
        if idx + 4 < len(inc_sorted):
            prev = inc_sorted.iloc[idx + 4]  # sorted descending by date? No, ascending
        # Actually inc_by_filing is sorted ascending, so idx-4 would be 4 quarters earlier
        if idx >= 4:
            prev = inc_sorted.iloc[idx - 4]
            prev_rev = prev.get("revenue")
            prev_eps = prev.get("eps_diluted") or prev.get("eps")
            if prev_rev and abs(prev_rev) > 1e-6 and rev is not None:
                inc_sorted.iloc[idx, inc_sorted.columns.get_loc("revenue_yoy")] = (rev - prev_rev) / abs(prev_rev)
            if prev_eps and abs(prev_eps) > 1e-6 and eps is not None:
                inc_sorted.iloc[idx, inc_sorted.columns.get_loc("eps_yoy")] = (eps - prev_eps) / abs(prev_eps)

        # QoQ: compare to previous quarter
        if idx >= 1:
            prev = inc_sorted.iloc[idx - 1]
            prev_rev = prev.get("revenue")
            if prev_rev and abs(prev_rev) > 1e-6 and rev is not None:
                inc_sorted.iloc[idx, inc_sorted.columns.get_loc("revenue_qoq")] = (rev - prev_rev) / abs(prev_rev)

    # ── Margins ──────────────────────────────────────────────────────
    inc_sorted["_gross_margin"] = np.where(
        inc_sorted["revenue"].notna() & (inc_sorted["revenue"].abs() > 0),
        inc_sorted["gross_profit"] / inc_sorted["revenue"], np.nan)
    inc_sorted["_operating_margin"] = np.where(
        inc_sorted["revenue"].notna() & (inc_sorted["revenue"].abs() > 0),
        inc_sorted["operating_income"] / inc_sorted["revenue"], np.nan)
    inc_sorted["_net_margin"] = np.where(
        inc_sorted["revenue"].notna() & (inc_sorted["revenue"].abs() > 0),
        inc_sorted["net_income"] / inc_sorted["revenue"], np.nan)

    # Margin trend (slope of net_margin over last 4 quarters)
    inc_sorted["_margin_trend"] = np.nan
    nm_vals = inc_sorted["_net_margin"].values
    for idx in range(3, len(nm_vals)):
        window = nm_vals[idx-3:idx+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 3:
            slope = np.polyfit(range(len(valid)), valid, 1)[0]
            inc_sorted.iloc[idx, inc_sorted.columns.get_loc("_margin_trend")] = slope

    # ── EPS surprise ─────────────────────────────────────────────────
    # Most recent earnings where actual is reported
    reported = sym_earnings[sym_earnings["eps_actual"].notna()].sort_values("date")

    # ── Map quarterly values to daily dates using filingDate ──────────
    # For each trading day, find the latest quarter with filingDate <= that day
    filing_dates = inc_sorted["filing_date"].values
    n_quarters = len(inc_sorted)

    # Pre-build arrays for fast lookup
    rev_yoy_arr = inc_sorted["revenue_yoy"].values
    eps_yoy_arr = inc_sorted["eps_yoy"].values
    rev_qoq_arr = inc_sorted["revenue_qoq"].values
    gm_arr = inc_sorted["_gross_margin"].values
    om_arr = inc_sorted["_operating_margin"].values
    nm_arr = inc_sorted["_net_margin"].values
    mt_arr = inc_sorted["_margin_trend"].values

    # Ratios arrays — BUG FIX: use filing_date from income, not period end date
    # Join ratios to income by (symbol, date) to get filing_date for each quarter
    rat_filing_dates = np.array([])
    pe_arr = np.array([])
    ps_arr = np.array([])
    dte_arr = np.array([])
    cr_arr = np.array([])
    if len(rat_by_date) > 0:
        # Build filing_date lookup from income: period_end_date → filing_date
        # Use pandas Series (not .values) so map() gets matching Timestamp types
        inc_filing_lookup = dict(zip(inc_by_filing["date"], inc_by_filing["filing_date"]))
        rat_with_filing = rat_by_date.copy()
        rat_with_filing["_filing_date"] = rat_with_filing["date"].map(inc_filing_lookup)
        # Drop rows without a matching filing date, sort by filing_date
        rat_with_filing = rat_with_filing.dropna(subset=["_filing_date"]).sort_values("_filing_date")
        rat_filing_dates = rat_with_filing["_filing_date"].values
        pe_arr = rat_with_filing["pe_ratio"].values
        ps_arr = rat_with_filing["ps_ratio"].values
        dte_arr = rat_with_filing["debt_to_equity"].values if "debt_to_equity" in rat_with_filing.columns else np.full(len(rat_with_filing), np.nan)
        cr_arr = rat_with_filing["current_ratio"].values if "current_ratio" in rat_with_filing.columns else np.full(len(rat_with_filing), np.nan)

    # Metrics arrays — same fix: use filing_date from income
    met_filing_dates = np.array([])
    roe_arr = np.array([])
    roa_arr = np.array([])
    if len(met_by_date) > 0:
        inc_filing_lookup2 = dict(zip(inc_by_filing["date"], inc_by_filing["filing_date"]))
        met_with_filing = met_by_date.copy()
        met_with_filing["_filing_date"] = met_with_filing["date"].map(inc_filing_lookup2)
        met_with_filing = met_with_filing.dropna(subset=["_filing_date"]).sort_values("_filing_date")
        met_filing_dates = met_with_filing["_filing_date"].values
        roe_arr = met_with_filing["roe"].values
        roa_arr = met_with_filing["roa"].values

    # Earnings dates
    earn_dates = reported["date"].values if len(reported) > 0 else np.array([])
    eps_actual_arr = reported["eps_actual"].values if len(reported) > 0 else np.array([])
    eps_est_arr = reported["eps_estimated"].values if len(reported) > 0 else np.array([])

    # Build daily feature arrays
    n_days = len(date_index)
    rev_growth_yoy = np.full(n_days, np.nan)
    eps_growth_yoy = np.full(n_days, np.nan)
    rev_growth_qoq = np.full(n_days, np.nan)
    gross_margin = np.full(n_days, np.nan)
    operating_margin = np.full(n_days, np.nan)
    net_margin = np.full(n_days, np.nan)
    margin_trend = np.full(n_days, np.nan)
    pe_ratio_daily = np.full(n_days, np.nan)
    ps_ratio_daily = np.full(n_days, np.nan)
    dte_daily = np.full(n_days, np.nan)
    cr_daily = np.full(n_days, np.nan)
    roe_daily = np.full(n_days, np.nan)
    roa_daily = np.full(n_days, np.nan)
    days_since = np.full(n_days, np.nan)
    eps_surprise = np.full(n_days, np.nan)

    date_vals = date_index.values

    for d in range(n_days):
        dt = date_vals[d]

        # Find latest quarter with filing_date <= dt
        q_idx = np.searchsorted(filing_dates, dt, side="right") - 1
        if q_idx >= 0 and q_idx < n_quarters:
            rev_growth_yoy[d] = rev_yoy_arr[q_idx]
            eps_growth_yoy[d] = eps_yoy_arr[q_idx]
            rev_growth_qoq[d] = rev_qoq_arr[q_idx]
            gross_margin[d] = gm_arr[q_idx]
            operating_margin[d] = om_arr[q_idx]
            net_margin[d] = nm_arr[q_idx]
            margin_trend[d] = mt_arr[q_idx]

        # Ratios: latest quarter with filing_date <= dt (BUG 1 FIX)
        if len(rat_filing_dates) > 0:
            r_idx = np.searchsorted(rat_filing_dates, dt, side="right") - 1
            if r_idx >= 0:
                pe_ratio_daily[d] = pe_arr[r_idx]
                ps_ratio_daily[d] = ps_arr[r_idx]
                dte_daily[d] = dte_arr[r_idx]
                cr_daily[d] = cr_arr[r_idx]

        # Metrics: latest quarter with filing_date <= dt (BUG 1 FIX)
        if len(met_filing_dates) > 0:
            m_idx = np.searchsorted(met_filing_dates, dt, side="right") - 1
            if m_idx >= 0:
                roe_daily[d] = roe_arr[m_idx]
                roa_daily[d] = roa_arr[m_idx]

        # Days since last earnings
        if len(earn_dates) > 0:
            e_idx = np.searchsorted(earn_dates, dt, side="right") - 1
            if e_idx >= 0:
                diff = (dt - earn_dates[e_idx]) / np.timedelta64(1, "D")
                days_since[d] = diff

                # EPS surprise from most recent reported
                actual = eps_actual_arr[e_idx]
                est = eps_est_arr[e_idx]
                if not np.isnan(actual) and est is not None and not np.isnan(est) and abs(est) > 1e-9:
                    eps_surprise[d] = (actual - est) / abs(est)

        # BUG 2 FIX: days_until_earnings REMOVED (used retroactive data)

    feat["revenue_growth_yoy"] = rev_growth_yoy
    feat["eps_growth_yoy"] = eps_growth_yoy
    feat["revenue_growth_qoq"] = rev_growth_qoq
    feat["gross_margin"] = gross_margin
    feat["operating_margin"] = operating_margin
    feat["net_margin"] = net_margin
    feat["margin_trend_4q"] = margin_trend
    feat["pe_ratio"] = pe_ratio_daily
    feat["ps_ratio"] = ps_ratio_daily
    feat["debt_to_equity"] = dte_daily
    feat["current_ratio"] = cr_daily
    feat["roe"] = roe_daily
    feat["roa"] = roa_daily
    feat["days_since_earnings"] = days_since
    feat["eps_surprise_last"] = np.clip(eps_surprise, -2.0, 2.0)

    # ── Insider activity ─────────────────────────────────────────────
    buy_ratio = np.full(n_days, np.nan)
    net_shares = np.full(n_days, np.nan)

    if len(sym_insiders) > 0:
        ins_dates = sym_insiders["date"].values
        ins_is_buy = sym_insiders["is_buy"].values
        ins_shares = sym_insiders["shares"].values

        for d in range(n_days):
            dt = date_vals[d]
            # Look back 90 calendar days
            cutoff = dt - np.timedelta64(90, "D")
            mask = (ins_dates >= cutoff) & (ins_dates <= dt)
            if mask.any():
                buys = ins_is_buy[mask].sum()
                total = mask.sum()
                buy_ratio[d] = buys / total if total > 0 else np.nan

                # Net shares: positive = net buying
                buy_shares = ins_shares[mask & (ins_is_buy == 1)].sum() if (mask & (ins_is_buy == 1)).any() else 0
                sell_shares = ins_shares[mask & (ins_is_buy == 0)].sum() if (mask & (ins_is_buy == 0)).any() else 0
                net_shares[d] = buy_shares - sell_shares

    feat["insider_buy_ratio_90d"] = buy_ratio
    feat["insider_net_shares_90d"] = net_shares

    return feat


def _consec_streak(returns: pd.Series, positive: bool = True) -> pd.Series:
    """Count consecutive positive (or negative) monthly returns."""
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

    # ── Long-timeframe technical features (for ML Slow v2) ──────────────
    feat["ret_126d"] = c.pct_change(126)
    feat["ret_252d"] = c.pct_change(252)

    # Distance from 52-week high/low
    high_252 = df["high"].rolling(252, min_periods=252).max()
    low_252  = df["low"].rolling(252, min_periods=252).min()
    feat["dist_52w_high"] = (c - high_252) / high_252.replace(0, np.nan)
    feat["dist_52w_low"]  = (c - low_252)  / low_252.replace(0, np.nan)

    # SMA200 slope (30-day change in SMA200)
    sma200 = c.rolling(200).mean()
    sma200_30ago = sma200.shift(30)
    feat["sma200_slope"] = (sma200 - sma200_30ago) / sma200_30ago.replace(0, np.nan)

    # Max drawdown over past 6 months (126 trading days)
    def _rolling_max_dd(prices, window=126):
        result = pd.Series(np.nan, index=prices.index)
        arr = prices.values
        for i in range(window, len(arr)):
            segment = arr[i - window : i + 1]
            valid = segment[~np.isnan(segment)]
            if len(valid) < 10:
                continue
            peak = np.maximum.accumulate(valid)
            dd = (valid - peak) / np.where(peak > 0, peak, np.nan)
            result.iloc[i] = np.nanmin(dd)
        return result

    feat["max_dd_6m"] = _rolling_max_dd(c, 126)

    # Consecutive up/down months
    monthly_ret = c.pct_change(21)  # ~1 month in trading days
    feat["consec_up_months"] = _consec_streak(monthly_ret, positive=True)
    feat["consec_down_months"] = _consec_streak(monthly_ret, positive=False)

    return feat


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ML Data Pipeline — Yahoo Finance 15-Year Feature Matrix")
    print(f"Date range : {START_DATE} → {END_DATE}")
    print(f"Symbols    : {len(ALL_SYMBOLS)} total ({len(UNIVERSE)} universe + cross-asset)")
    print("=" * 60)

    # 1. Fetch all raw bars from Massive (Polygon) with disk caching
    # Convert ticker formats: sp500_history uses BRK-B, Massive uses BRK.B
    massive_syms = [_ticker_to_massive(s) for s in ALL_SYMBOLS]
    massive_to_orig = {_ticker_to_massive(s): s for s in ALL_SYMBOLS}

    print(f"Fetching {len(massive_syms)} symbols from Massive (Polygon) ...")
    provider = _get_massive()
    warmup_cal_days = (datetime.today() - datetime.strptime(START_DATE, "%Y-%m-%d")).days + 30
    raw_massive = provider.fetch_bars_batch(massive_syms, warmup_days=warmup_cal_days, adjusted=True)

    # Map back to original ticker format (BRK.B -> BRK-B)
    raw = {}
    for massive_sym, df in raw_massive.items():
        orig_sym = massive_to_orig.get(massive_sym, _ticker_from_massive(massive_sym))
        raw[orig_sym] = df

    # Quality gate (using original ticker names)
    quality = provider.run_quality_gate(raw, expected_symbols=ALL_SYMBOLS)
    if not quality["passed"]:
        print(f"  WARNING: data quality issues: {quality['issues']}")

    # Fall back: re-fetch individually any symbol that came back empty
    missing = [s for s in ALL_SYMBOLS if s not in raw or raw[s].empty]
    if missing:
        print(f"\nFetching {len(missing)} missing symbol(s) individually ...")
        for i, sym in enumerate(missing, 1):
            if i <= 10 or i == len(missing):
                print(f"  [{i}/{len(missing)}] {sym} ...", end=" ", flush=True)
            raw[sym] = fetch_bars(sym)
            if i <= 10 or i == len(missing):
                print(f"{len(raw[sym])} bars")
            time.sleep(0.3)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Summary (counts, not per-symbol)
    loaded = sum(1 for s in ALL_SYMBOLS if s in raw and len(raw[s]) > 0)
    empty = len(ALL_SYMBOLS) - loaded
    print(f"\n  Bars loaded: {loaded}/{len(ALL_SYMBOLS)} symbols"
          + (f" ({empty} empty)" if empty > 0 else ""))

    # 2. Build cross-asset feature frame aligned to a common date index
    #    Use SPY as the reference calendar
    spy_close = raw.get("SPY", pd.DataFrame())
    if spy_close.empty:
        sys.exit("ERROR: SPY data is required as the date calendar.")

    date_index = spy_close.index  # all trading days

    # Trading-day ordinal map: {Timestamp → int}, used for days_since_earnings
    td_map = {ts: i for i, ts in enumerate(date_index)}

    # Build feature universe: all symbols with data (used for feature computation + FMP fetch)
    feature_universe = [s for s in ALL_SYMBOLS if s in raw and not raw[s].empty]

    # 2b. Fetch earnings surprise history (all stocks including historical; ETFs get zeros)
    etf_set = set(get_etf_symbols())
    earnings_symbols = [s for s in feature_universe if s not in etf_set]
    print(f"\nFetching earnings surprise data from FMP API ({len(earnings_symbols)} stocks) ...")
    earnings_data: dict = {}
    if FMP_API_KEY:
        cached_count = 0
        api_count = 0
        for idx_s, sym in enumerate(earnings_symbols, 1):
            cache_file = EARNINGS_CACHE_DIR / f"{sym}.json"
            source = "cache" if (cache_file.exists() and
                                  (datetime.today() - datetime.fromtimestamp(
                                      cache_file.stat().st_mtime)).days < EARNINGS_CACHE_TTL) else "API"
            df_earn = fetch_earnings_surprises(sym)
            earnings_data[sym] = df_earn
            if source == "cache":
                cached_count += 1
            else:
                api_count += 1
                time.sleep(0.05)  # FMP Premium: 750 calls/min
            # Progress every 50 symbols
            if idx_s % 50 == 0 or idx_s == len(earnings_symbols):
                print(f"  [{idx_s}/{len(earnings_symbols)}] processed ({cached_count} cached, {api_count} API)")
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

    # ── Forward-adjust VIXY for reverse splits ───────────────────────────
    # yfinance backward-adjusts prices (multiplying old prices UP for
    # reverse splits), which inflates historical VIXY by 80x+.  Since we
    # use vixy_level as an absolute feature, we need all prices on the
    # CURRENT share basis.  Compute cumulative forward-adjustment factor
    # from the split history and apply it.
    print("\n  Adjusting VIXY for reverse splits ...")
    try:
        vixy_splits_df = _get_massive().fetch_splits("VIXY")
        if len(vixy_splits_df) > 0:
            vixy_splits = pd.Series(
                vixy_splits_df["ratio"].values,
                index=vixy_splits_df["date"].values)
        else:
            vixy_splits = pd.Series(dtype=float)
    except Exception:
        # Fallback to yfinance
        vixy_ticker = yf.Ticker("VIXY")
        vixy_splits = vixy_ticker.splits
    if len(vixy_splits) > 0:
        # Build a cumulative forward-adjustment factor for each date:
        # For date d, factor = product of all split ratios AFTER d
        # (reverse split ratio < 1 reduces old prices to current basis)
        split_dates = pd.to_datetime(vixy_splits.index).tz_localize(None).sort_values()
        split_ratios = vixy_splits.reindex(vixy_splits.index.sort_values()).values

        # Cumulative factor from the END (most recent splits first)
        cum_factor = np.ones(len(split_dates) + 1)
        for i in range(len(split_ratios) - 1, -1, -1):
            cum_factor[i] = cum_factor[i + 1] * split_ratios[i]
        # cum_factor[0] = product of ALL ratios (for dates before first split)
        # cum_factor[-1] = 1.0 (for dates after last split)

        # For each date in vixy_c, find the adjustment factor
        adjustment = pd.Series(1.0, index=vixy_c.index)
        for i, sd in enumerate(split_dates):
            # Dates strictly before this split get multiplied by this split's ratio
            mask = adjustment.index < sd
            adjustment[mask] *= split_ratios[i]

        old_sample = vixy_c.loc[vixy_c.index == "2020-06-15"]
        vixy_c = vixy_c * adjustment
        new_sample = vixy_c.loc[vixy_c.index == "2020-06-15"]
        print(f"    Splits found: {len(vixy_splits)}")
        print(f"    Total adjustment factor for oldest data: {cum_factor[0]:.6f}")
        if len(old_sample) > 0 and len(new_sample) > 0:
            print(f"    2020-06-15 BEFORE: {old_sample.values[0]:.2f}")
            print(f"    2020-06-15 AFTER:  {new_sample.values[0]:.2f}")
        print(f"    VIXY range after adjustment: {vixy_c.min():.2f} - {vixy_c.max():.2f}")
    else:
        print("    No splits found for VIXY")

    # Cross-asset features
    # Use backward-adjusted (continuous) VIXY for returns, but
    # forward-adjusted VIXY for the absolute level feature.
    vixy_c_backward = aligned_close("VIXY")  # original backward-adjusted (continuous)
    cross = pd.DataFrame(index=date_index)
    for n in [5, 10, 20, 60, 120]:
        cross[f"spy_ret_{n}d"]  = spy_c.pct_change(n)
        cross[f"tlt_ret_{n}d"]  = tlt_c.pct_change(n)
    cross["vixy_level"]         = vixy_c       # forward-adjusted (current share basis)
    cross["vixy_ret_5d"]        = vixy_c_backward.pct_change(5)   # from continuous series
    cross["vixy_ret_20d"]       = vixy_c_backward.pct_change(20)  # from continuous series

    # Calendar features
    cross["day_of_week"] = date_index.dayofweek          # 0=Mon … 4=Fri
    cross["month"]       = date_index.month
    cross["quarter"]     = date_index.quarter

    # ── Macro regime features (from FRED + cross-asset ETFs) ─────────────
    macro_file = OUTPUT_DIR / "macro_fred.parquet"
    if macro_file.exists():
        print("\n  Loading FRED macro data ...")
        macro = pd.read_parquet(macro_file)
        macro.index = pd.to_datetime(macro.index).tz_localize(None)
        macro = macro.reindex(date_index, method="ffill")
        # DO NOT bfill — that would leak future data into early dates.
        # Instead, leave NaN for dates before each series' first observation.
        # LightGBM handles NaN natively.

        # Yield curve
        cross["yield_curve_10y2y"] = macro.get("T10Y2Y", pd.Series(np.nan, index=date_index))
        yc = cross["yield_curve_10y2y"]
        cross["yield_curve_30d_change"] = yc - yc.shift(30)

        # High yield spread
        hy = macro.get("BAMLH0A0HYM2", pd.Series(np.nan, index=date_index))
        cross["hy_spread"] = hy
        cross["hy_spread_30d_change"] = hy - hy.shift(30)

        # Dollar index
        dxy = macro.get("DTWEXBGS", pd.Series(np.nan, index=date_index))
        cross["dxy_level"] = dxy
        cross["dxy_30d_change"] = (dxy - dxy.shift(30)) / dxy.shift(30).replace(0, np.nan)

        print(f"    FRED features added: yield_curve, hy_spread, dxy")
    else:
        print("  WARNING: macro_fred.parquet not found — run fred_data_pipeline.py first")
        for col in ["yield_curve_10y2y", "yield_curve_30d_change",
                     "hy_spread", "hy_spread_30d_change",
                     "dxy_level", "dxy_30d_change"]:
            cross[col] = np.nan

    # Cross-asset ratio features (from ETF prices)
    hyg_c = aligned_close("HYG")
    lqd_c = aligned_close("LQD")
    cper_c = aligned_close("CPER")
    gld_c = aligned_close("GLD")

    hyg_lqd = hyg_c / lqd_c.replace(0, np.nan)
    cross["hyg_lqd_ratio"] = hyg_lqd
    cross["hyg_lqd_30d_change"] = (hyg_lqd - hyg_lqd.shift(30)) / hyg_lqd.shift(30).replace(0, np.nan)

    copper_gold = cper_c / gld_c.replace(0, np.nan)
    cross["copper_gold_ratio"] = copper_gold
    cross["copper_gold_30d_change"] = (copper_gold - copper_gold.shift(30)) / copper_gold.shift(30).replace(0, np.nan)

    print(f"    Cross-asset ratios added: HYG/LQD, CPER/GLD")

    # ── Load fundamental data (if available) ────────────────────────────
    fund_data = _load_fundamental_data(date_index)

    # 3. Compute per-symbol features + target, then concatenate
    print("\nComputing features ...")
    all_frames = []

    skipped = 0
    for i, sym in enumerate(feature_universe, 1):
        df = raw.get(sym, pd.DataFrame())
        if df.empty:
            skipped += 1
            continue

        # Align to common date index
        df = df.reindex(date_index)

        # Per-symbol technical features
        feat = compute_symbol_features(df)

        # Fundamental features (stocks only; ETFs get NaN which LightGBM handles)
        if fund_data is not None:
            fund_feat = _compute_fundamental_features(sym, date_index, fund_data)
            if fund_feat is not None:
                feat = feat.join(fund_feat, how="left")

        # Target: forward 10-day return > 2% (computed BEFORE dropping NaNs)
        fwd_ret = df["close"].pct_change(10).shift(-10)
        feat["target"] = (fwd_ret > 0.02).astype("Int8")  # nullable int → NaN for last rows

        # ── Delisting return fix ────────────────────────────────────────
        # If a stock's data ends well before END_DATE and it's no longer in
        # the current S&P 500, assign a large negative forward return for
        # its last rows. This ensures the model sees that delisted/bankrupt
        # stocks don't just "disappear" — they lose value.
        if sym not in etf_set:
            last_valid = df["close"].last_valid_index()
            if last_valid is not None:
                days_until_end = (pd.Timestamp(END_DATE) - last_valid).days
                if days_until_end > 30 and sym not in set(STOCK_SYMBOLS):
                    # Stock data ended early AND it's not in current S&P 500
                    # → likely delisted/acquired/bankrupt. Set forward return
                    # to -100% for the last 10 rows (where target would be NaN).
                    tail_mask = feat.index > (last_valid - pd.Timedelta(days=15))
                    feat.loc[tail_mask & feat["target"].isna(), "target"] = 0  # fwd_ret < 2% → 0

        # Merge cross-asset features
        feat = feat.join(cross, how="left")

        # Add symbol identifier
        feat.insert(0, "symbol", sym)
        feat.index.name = "date"

        # Drop rows where the target is NaN (last 10 days — look-ahead unavailable)
        feat = feat[feat["target"].notna()]

        # Drop rows with any NaN in non-fundamental features (warmup period)
        # Fundamental AND macro features are allowed to be NaN
        # because LightGBM handles NaN natively
        NAN_SAFE_COLS = set(FUNDAMENTAL_FEATURE_COLS) | {
            "symbol", "target",
            # FRED macro features (may have gaps in historical coverage)
            "yield_curve_10y2y", "yield_curve_30d_change",
            "hy_spread", "hy_spread_30d_change",
            "dxy_level", "dxy_30d_change",
            # Cross-sectional ranks (only computed for in_sp500 stocks)
            "return_rank_3m", "return_rank_6m", "return_rank_12m",
            "vol_rank_3m", "vol_rank_6m", "vol_126d",
            # Sector-relative features (only computed for in_sp500 stocks)
            "ret_10d_vs_sector", "ret_20d_vs_sector",
            "rsi_14_vs_sector", "vol_20d_vs_sector",
        }
        non_fund_cols = [c for c in feat.columns if c not in NAN_SAFE_COLS]
        feat = feat.dropna(subset=non_fund_cols)

        all_frames.append(feat.reset_index())

        # Progress every 50 symbols
        if i % 50 == 0 or i == len(UNIVERSE):
            print(f"  [{i}/{len(UNIVERSE)}] processed ({len(all_frames)} with data, {skipped} skipped)")

    if not all_frames:
        sys.exit("ERROR: no feature data produced — check yfinance and symbol list.")

    # 4. Combine and save
    print("\nCombining all symbols ...")
    master = pd.concat(all_frames, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    master = master.sort_values(["date", "symbol"]).reset_index(drop=True)

    # ── Point-in-time S&P 500 membership ──────────────────────────────────
    print("Computing point-in-time S&P 500 membership (survivorship bias) ...")
    load_sp500_changes()  # ensure cache is loaded
    etf_set = set(get_etf_symbols())
    unique_dates = sorted(master["date"].unique())
    # Build a dict: date -> set of S&P 500 members on that date
    sp500_by_date = {}
    for di, d in enumerate(unique_dates):
        sp500_by_date[d] = get_sp500_on_date(pd.Timestamp(d))
        if (di + 1) % 500 == 0 or (di + 1) == len(unique_dates):
            print(f"    [{di+1}/{len(unique_dates)}] dates processed")
    # Efficient lookup: group by date, mark membership for each group
    master["in_sp500"] = False
    # ETFs are always in
    master.loc[master["symbol"].isin(etf_set), "in_sp500"] = True
    # For each date, mark stocks that were in S&P 500
    non_etf_mask = ~master["symbol"].isin(etf_set)
    for d, members in sp500_by_date.items():
        date_mask = (master["date"] == d) & non_etf_mask
        master.loc[date_mask, "in_sp500"] = master.loc[date_mask, "symbol"].isin(members)
    n_in = master["in_sp500"].sum()
    n_out = len(master) - n_in
    print(f"  in_sp500=True: {n_in:,} rows  |  in_sp500=False: {n_out:,} rows")

    # ── Survivorship coverage quality gate ───────────────────────────────
    # For sampled dates, verify that the number of tickers with data matches
    # the point-in-time S&P 500 count within 5%
    print("Running survivorship coverage check ...")
    sample_dates = unique_dates[::60]  # every ~3 months
    coverage_issues = 0
    for d in sample_dates:
        expected = len(sp500_by_date[d])
        actual = master[(master["date"] == d) & (master["in_sp500"] == True)]["symbol"].nunique()
        coverage = actual / expected if expected > 0 else 1.0
        if coverage < 0.95:
            coverage_issues += 1
            if coverage_issues <= 3:
                print(f"  WARNING: {pd.Timestamp(d).date()}: {actual}/{expected} "
                      f"SP500 tickers have data ({coverage:.1%})")
    if coverage_issues == 0:
        print(f"  Survivorship coverage PASSED: all {len(sample_dates)} sampled dates "
              f"have >95% SP500 ticker coverage")
    else:
        print(f"  Survivorship coverage: {coverage_issues}/{len(sample_dates)} dates "
              f"below 95% threshold")

    # ── Cross-sectional rank features ────────────────────────────────────
    # Compute percentile ranks across only in_sp500 stocks on each date
    # to avoid cross-sectional leakage from non-tradeable stocks.
    print("Computing cross-sectional rank features (in_sp500 only) ...")
    sp500_mask = master["in_sp500"] == True
    for ret_col, rank_col, period in [
        ("ret_60d",  "return_rank_3m",  None),   # 60d ~ 3 months
        ("ret_126d", "return_rank_6m",  None),
        ("ret_252d", "return_rank_12m", None),
    ]:
        master[rank_col] = np.nan
        if ret_col in master.columns:
            master.loc[sp500_mask, rank_col] = (
                master.loc[sp500_mask].groupby("date")[ret_col].rank(pct=True)
            )

    # Volatility ranks (use existing vol columns)
    for vol_col, rank_col in [
        ("vol_60d",  "vol_rank_3m"),
    ]:
        master[rank_col] = np.nan
        if vol_col in master.columns:
            master.loc[sp500_mask, rank_col] = (
                master.loc[sp500_mask].groupby("date")[vol_col].rank(pct=True)
            )

    # 6-month volatility (126-day rolling std) — compute and rank
    # We need per-symbol close for this; use the return columns
    # vol_126d = daily_ret.rolling(126).std() * sqrt(252) — compute from ret_5d
    # Actually compute from the existing data: use ret_126d dispersion as proxy
    # Better: compute directly from close prices stored in raw
    vol_126d_vals = []
    for _, row in master.iterrows():
        vol_126d_vals.append(np.nan)  # placeholder

    # More efficient: compute vol_126d per symbol then merge
    print("  Computing 126-day volatility ...")
    vol_126d_series = {}
    for sym in feature_universe:
        df_sym = raw.get(sym, pd.DataFrame())
        if df_sym.empty:
            continue
        c = df_sym["close"].reindex(date_index)
        daily_r = c.pct_change()
        vol_126d_series[sym] = daily_r.rolling(126, min_periods=126).std() * np.sqrt(252)

    # Map back to master
    master["vol_126d"] = np.nan
    for sym, vol_s in vol_126d_series.items():
        mask = master["symbol"] == sym
        dates = master.loc[mask, "date"]
        master.loc[mask, "vol_126d"] = vol_s.reindex(dates.values).values

    master["vol_rank_6m"] = np.nan
    master.loc[sp500_mask, "vol_rank_6m"] = (
        master.loc[sp500_mask].groupby("date")["vol_126d"].rank(pct=True)
    )

    # ── Fundamental cross-sectional features ────────────────────────────
    # PE and PS relative to universe median on each date (in_sp500 only)
    if "pe_ratio" in master.columns:
        master["pe_vs_universe_median"] = np.nan
        sp500_pe = master.loc[sp500_mask].copy()
        pe_median = sp500_pe.groupby("date")["pe_ratio"].transform("median")
        master.loc[sp500_mask, "pe_vs_universe_median"] = np.where(
            pe_median.notna() & (pe_median.abs() > 0),
            sp500_pe["pe_ratio"] / pe_median - 1.0,
            np.nan,
        )
    if "ps_ratio" in master.columns:
        master["ps_vs_universe_median"] = np.nan
        sp500_ps = master.loc[sp500_mask].copy()
        ps_median = sp500_ps.groupby("date")["ps_ratio"].transform("median")
        master.loc[sp500_mask, "ps_vs_universe_median"] = np.where(
            ps_median.notna() & (ps_median.abs() > 0),
            sp500_ps["ps_ratio"] / ps_median - 1.0,
            np.nan,
        )

    n_rank_features = 5
    n_fund_cross = sum(1 for c in ["pe_vs_universe_median", "ps_vs_universe_median"] if c in master.columns)
    print(f"  Added {n_rank_features} cross-sectional rank features + {n_fund_cross} fundamental cross-sectional features")

    # ── Sector-relative features (in_sp500 only) ─────────────────────────
    print("Computing sector-relative features (in_sp500 only) ...")
    master["sector"] = master["symbol"].map(SYMBOL_SECTOR).fillna("Other")
    sector_relative_cols = [
        ("ret_10d",  "ret_10d_vs_sector"),
        ("ret_20d",  "ret_20d_vs_sector"),
        ("rsi_14",   "rsi_14_vs_sector"),
        ("vol_20d",  "vol_20d_vs_sector"),
    ]
    n_sector_feats = 0
    for src_col, dst_col in sector_relative_cols:
        master[dst_col] = np.nan
        if src_col in master.columns:
            sp500_sub = master.loc[sp500_mask]
            sector_date_median = sp500_sub.groupby(["date", "sector"])[src_col].transform("median")
            master.loc[sp500_mask, dst_col] = sp500_sub[src_col] - sector_date_median
            n_sector_feats += 1
    # Drop sector column — it's categorical and only used for computing relative features
    master.drop(columns=["sector"], inplace=True)
    print(f"  Added {n_sector_feats} sector-relative features")

    # ── Clip fundamental outliers ───────────────────────────────────────
    print("Clipping fundamental feature outliers ...")
    clip_rules = {
        "pe_ratio": (-100, 500),
        "ps_ratio": (0, 100),
        "debt_to_equity": (0, 50),
        "current_ratio": (0, 50),
        "roe": (-5, 5),
        "roa": (-5, 5),
    }
    for col, (lo, hi) in clip_rules.items():
        if col in master.columns:
            before_min, before_max = master[col].min(), master[col].max()
            master[col] = master[col].clip(lo, hi)
            print(f"  {col}: [{before_min:.2f}, {before_max:.2f}] → [{lo}, {hi}]")

    # Log-scale insider_net_shares_90d
    if "insider_net_shares_90d" in master.columns:
        before_min, before_max = master["insider_net_shares_90d"].min(), master["insider_net_shares_90d"].max()
        master["insider_net_shares_90d"] = np.sign(master["insider_net_shares_90d"]) * np.log1p(master["insider_net_shares_90d"].abs())
        print(f"  insider_net_shares_90d: [{before_min:.0f}, {before_max:.0f}] → log-scaled [{master['insider_net_shares_90d'].min():.2f}, {master['insider_net_shares_90d'].max():.2f}]")

    # Also clip cross-sectional valuation features derived from clipped PE/PS
    if "pe_vs_universe_median" in master.columns:
        master["pe_vs_universe_median"] = master["pe_vs_universe_median"].clip(-10, 20)
    if "ps_vs_universe_median" in master.columns:
        master["ps_vs_universe_median"] = master["ps_vs_universe_median"].clip(-1, 10)

    print(f"Final dataset: {len(master):,} rows × {len(master.columns)} columns")
    print(f"Date range in data: {master['date'].min().date()} → {master['date'].max().date()}")
    print(f"Target balance: {master['target'].mean()*100:.1f}% positive")

    print(f"\nSaving to {OUTPUT_FILE} ...")
    master.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow", compression="snappy")
    print(f"Done.  File size: {OUTPUT_FILE.stat().st_size / 1_048_576:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
