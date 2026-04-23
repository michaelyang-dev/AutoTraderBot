"""
FMP Alt-Data Client
===================
Fetches insider trades and analyst estimates from FMP Premium API
for feature engineering in the ML pipeline.

Endpoints used:
  - /stable/insider-trading/search   (Form 4 filings)
  - /stable/analyst-estimates        (quarterly EPS/revenue estimates)

Caches each symbol's data to parquet in data/altdata_cache/.
Resumable: skips symbols that already have fresh cache files.

Usage:
    python3 fmp_altdata_client.py              # test with AAPL
    python3 fmp_altdata_client.py --bulk       # fetch full universe
"""

import json
import os
import ssl
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "altdata_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_DAYS = 1  # re-fetch after 1 day for live; historical is permanent

# SSL context (Mac Python sometimes lacks certs)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# Rate limiting: max 10 req/sec (conservative; FMP allows 750/min = 12.5/sec)
_MIN_REQUEST_INTERVAL = 0.10  # 100ms between requests
_last_request_time = 0.0


def _rate_limit():
    """Enforce minimum interval between API requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _fmp_fetch(endpoint: str, params: str = "", retries: int = 3) -> list:
    """Fetch from FMP /stable/ endpoint with retries and exponential backoff."""
    import urllib.request

    url = (
        f"https://financialmodelingprep.com/stable/{endpoint}"
        f"?{params}&apikey={FMP_API_KEY}"
    )
    for attempt in range(retries):
        _rate_limit()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "error" in data:
                return []
            return [data] if isinstance(data, dict) else []
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            print(f"    FAILED {endpoint} after {retries} attempts: {e}")
            return []
    return []


def _cache_path(symbol: str, data_type: str) -> Path:
    """Cache file path for a symbol/data_type."""
    return CACHE_DIR / f"{symbol}_{data_type}.parquet"


def _cache_is_fresh(path: Path, historical: bool = False) -> bool:
    """Check if cache file exists and is fresh enough."""
    if not path.exists():
        return False
    if historical:
        return True  # historical data never expires
    age_days = (datetime.today() - datetime.fromtimestamp(path.stat().st_mtime)).days
    return age_days < CACHE_TTL_DAYS


# ══════════════════════════════════════════════════════════════════════════════
#  Insider Trades
# ══════════════════════════════════════════════════════════════════════════════

def get_insider_trades(symbol: str, from_date: str = None, to_date: str = None,
                       use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch insider trades (Form 4) for a symbol.

    Uses filingDate as the point-in-time availability date.
    Filters to open-market P-Purchase and S-Sale only.

    Returns DataFrame with columns:
        symbol, filingDate, transactionDate, reportingName, typeOfOwner,
        transactionType, securitiesTransacted, price, acquisitionOrDisposition
    """
    cache = _cache_path(symbol, "insider")
    if use_cache and _cache_is_fresh(cache, historical=(from_date is not None)):
        try:
            df = pd.read_parquet(cache)
            if from_date:
                df = df[df["filingDate"] >= pd.Timestamp(from_date)]
            if to_date:
                df = df[df["filingDate"] <= pd.Timestamp(to_date)]
            return df
        except Exception:
            pass

    # Fetch multiple pages to get full history
    all_rows = []
    for page in range(10):  # up to 1000 transactions
        data = _fmp_fetch("insider-trading/search", f"symbol={symbol}&page={page}&limit=100")
        if not data:
            break
        all_rows.extend(data)
        if len(data) < 100:
            break

    if not all_rows:
        empty = pd.DataFrame(columns=[
            "symbol", "filingDate", "transactionDate", "reportingName",
            "typeOfOwner", "transactionType", "securitiesTransacted",
            "price", "acquisitionOrDisposition",
        ])
        return empty

    df = pd.DataFrame(all_rows)

    # Normalize columns
    col_map = {
        "filingDate": "filingDate",
        "transactionDate": "transactionDate",
        "reportingName": "reportingName",
        "typeOfOwner": "typeOfOwner",
        "transactionType": "transactionType",
        "securitiesTransacted": "securitiesTransacted",
        "price": "price",
        "acquisitionOrDisposition": "acquisitionOrDisposition",
    }
    keep = [c for c in col_map if c in df.columns]
    df = df[keep].copy()
    df.insert(0, "symbol", symbol)

    # Parse dates
    for col in ["filingDate", "transactionDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric columns
    for col in ["securitiesTransacted", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to open-market purchases and sales only
    # P-Purchase and S-Sale are open-market; exclude M-Exempt, A-Award, etc.
    if "transactionType" in df.columns:
        open_market = df["transactionType"].isin(["P-Purchase", "S-Sale"])
        df = df[open_market].copy()

    df = df.dropna(subset=["filingDate"]).sort_values("filingDate", ascending=True)

    # Cache
    try:
        df.to_parquet(cache, index=False, engine="pyarrow", compression="snappy")
    except Exception:
        pass

    # Date filter
    if from_date:
        df = df[df["filingDate"] >= pd.Timestamp(from_date)]
    if to_date:
        df = df[df["filingDate"] <= pd.Timestamp(to_date)]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Analyst Estimates
# ══════════════════════════════════════════════════════════════════════════════

def get_analyst_estimates(symbol: str, from_date: str = None, to_date: str = None,
                          use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch quarterly analyst EPS/revenue estimates for a symbol.

    Returns DataFrame with columns:
        symbol, date, epsAvg, epsHigh, epsLow, revenueAvg,
        numAnalystsEps, numAnalystsRevenue
    """
    cache = _cache_path(symbol, "estimates")
    if use_cache and _cache_is_fresh(cache, historical=(from_date is not None)):
        try:
            df = pd.read_parquet(cache)
            if from_date:
                df = df[df["date"] >= pd.Timestamp(from_date)]
            if to_date:
                df = df[df["date"] <= pd.Timestamp(to_date)]
            return df
        except Exception:
            pass

    # Fetch quarterly estimates (up to 60 quarters = 15 years)
    data = _fmp_fetch("analyst-estimates", f"symbol={symbol}&period=quarter&limit=60")
    if not data:
        return pd.DataFrame(columns=[
            "symbol", "date", "epsAvg", "epsHigh", "epsLow",
            "revenueAvg", "numAnalystsEps", "numAnalystsRevenue",
        ])

    rows = []
    for item in data:
        try:
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(item["date"]),
                "epsAvg": item.get("epsAvg"),
                "epsHigh": item.get("epsHigh"),
                "epsLow": item.get("epsLow"),
                "revenueAvg": item.get("revenueAvg"),
                "numAnalystsEps": item.get("numAnalystsEps"),
                "numAnalystsRevenue": item.get("numAnalystsRevenue"),
            })
        except (KeyError, ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "date", "epsAvg", "epsHigh", "epsLow",
            "revenueAvg", "numAnalystsEps", "numAnalystsRevenue",
        ])

    df = pd.DataFrame(rows).sort_values("date", ascending=True)

    # Numeric columns
    for col in ["epsAvg", "epsHigh", "epsLow", "revenueAvg", "numAnalystsEps", "numAnalystsRevenue"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cache
    try:
        df.to_parquet(cache, index=False, engine="pyarrow", compression="snappy")
    except Exception:
        pass

    if from_date:
        df = df[df["date"] >= pd.Timestamp(from_date)]
    if to_date:
        df = df[df["date"] <= pd.Timestamp(to_date)]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Bulk Fetch
# ══════════════════════════════════════════════════════════════════════════════

def bulk_fetch_universe(symbols: list, from_date: str, to_date: str,
                        data_type: str):
    """
    Fetch data for all symbols in universe. Resumable via cache.

    data_type: 'insider' or 'estimates'
    """
    fetch_fn = {
        "insider": get_insider_trades,
        "estimates": get_analyst_estimates,
    }

    if data_type not in fetch_fn:
        print(f"ERROR: Unknown data_type '{data_type}'. Use: {list(fetch_fn.keys())}")
        return

    fn = fetch_fn[data_type]
    total = len(symbols)
    cached = 0
    fetched = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"Bulk fetch: {data_type} for {total} symbols")
    print(f"Date range: {from_date} → {to_date}")
    print(f"{'='*60}")

    for i, sym in enumerate(symbols, 1):
        cache = _cache_path(sym, data_type if data_type != "estimates" else "estimates")
        if _cache_is_fresh(cache, historical=True):
            cached += 1
        else:
            try:
                fn(sym, from_date, to_date, use_cache=False)
                fetched += 1
            except Exception as e:
                print(f"  FAILED {sym}: {e}")
                failed += 1

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] {cached} cached, {fetched} fetched, {failed} failed")

    print(f"\nDone: {cached} cached, {fetched} fetched, {failed} failed")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if not FMP_API_KEY:
        print("ERROR: FMP_API_KEY not set in .env")
        sys.exit(1)

    if "--bulk" in sys.argv:
        from sp500_universe import get_full_universe
        syms = get_full_universe()
        for dtype in ["insider", "estimates"]:
            bulk_fetch_universe(syms, "2020-01-01", "2026-04-22", dtype)
    else:
        # Quick test with AAPL
        print("=" * 60)
        print("FMP Alt-Data Client — AAPL Test")
        print("=" * 60)

        print("\n── Insider Trades ──")
        ins = get_insider_trades("AAPL", "2024-01-01", "2026-04-22", use_cache=False)
        print(f"  Shape: {ins.shape}")
        print(f"  Columns: {list(ins.columns)}")
        if len(ins) > 0:
            print(f"  Date range: {ins['filingDate'].min().date()} → {ins['filingDate'].max().date()}")
            print(f"  Transaction types: {ins['transactionType'].value_counts().to_dict()}")
            print(f"  Owner types: {ins['typeOfOwner'].value_counts().to_dict()}")
            print(f"\n  Sample rows:")
            for _, row in ins.head(5).iterrows():
                print(f"    {row['filingDate'].date()} | {row['transactionType']:12s} | "
                      f"{row['reportingName']:25s} | {row['typeOfOwner']:30s} | "
                      f"{row['securitiesTransacted']:>10,.0f} shares @ ${row['price']:.2f}")

        print("\n── Analyst Estimates ──")
        est = get_analyst_estimates("AAPL", "2020-01-01", "2026-12-31", use_cache=False)
        print(f"  Shape: {est.shape}")
        print(f"  Columns: {list(est.columns)}")
        if len(est) > 0:
            print(f"  Date range: {est['date'].min().date()} → {est['date'].max().date()}")
            print(f"\n  Recent quarters:")
            for _, row in est.tail(5).iterrows():
                print(f"    {row['date'].date()} | EPS avg={row['epsAvg']:.3f} "
                      f"[{row['epsLow']:.3f}–{row['epsHigh']:.3f}] | "
                      f"#analysts={row['numAnalystsEps']:.0f}")

        print("\n" + "=" * 60)
        print("Test complete.")
