"""
Fetch All Available Data Sources
=================================
Downloads and caches every useful data source from FMP and Massive
that we're currently NOT using. Builds a comprehensive feature set.

Data sources:
  FMP:
    - Price target consensus (analyst upside %)
    - DCF intrinsic value (model-based fair value)
    - Financial growth (pre-computed growth rates)
    - Enterprise value (EV/EBITDA)
    - Company profile (beta, market cap, sector)
    - Revenue segmentation (product + geo mix)
    - Economic calendar (macro events)
    - Earning call transcripts (for future NLP)

  Massive:
    - Historical option bars (for put/call volume ratio)
    - Crypto/Forex daily (BTC, ETH, EUR/USD, GBP/USD)

Run:
    cd ml_service && python3 -m strategies.fetch_all_data
"""

import json
import os
import ssl
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
FMP_KEY = os.environ.get("FMP_API_KEY", "")
MASSIVE_KEY = os.environ.get("MASSIVE_API_KEY", "")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "enhanced_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def log(msg):
    print(msg, flush=True)


def fmp_get(endpoint, params=""):
    url = "https://financialmodelingprep.com/stable/%s?%s&apikey=%s" % (endpoint, params, FMP_KEY)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return []


def massive_get(path, params=None):
    if params is None:
        params = {}
    params["apiKey"] = MASSIVE_KEY
    try:
        resp = requests.get("https://api.polygon.io" + path, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  FMP Data Fetchers
# ══════════════════════════════════════════════════════════════════════════════

def fetch_price_targets(symbols):
    """Fetch analyst price target consensus for each symbol."""
    cache = CACHE_DIR / "price_targets.parquet"
    if cache.exists() and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days < 3:
        return pd.read_parquet(cache)

    log("  Fetching price target consensus (%d symbols) ..." % len(symbols))
    rows = []
    for i, sym in enumerate(symbols):
        data = fmp_get("price-target-consensus", "symbol=%s" % sym)
        if isinstance(data, list) and data:
            d = data[0]
        elif isinstance(data, dict):
            d = data
        else:
            continue
        rows.append({
            "symbol": sym,
            "target_high": d.get("targetHigh"),
            "target_low": d.get("targetLow"),
            "target_consensus": d.get("targetConsensus"),
            "target_median": d.get("targetMedian"),
        })
        if (i + 1) % 100 == 0:
            log("    %d/%d" % (i + 1, len(symbols)))
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_parquet(cache, index=False)
    log("    Price targets: %d symbols" % len(df))
    return df


def fetch_dcf_values(symbols):
    """Fetch DCF intrinsic value for each symbol."""
    cache = CACHE_DIR / "dcf_values.parquet"
    if cache.exists() and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days < 3:
        return pd.read_parquet(cache)

    log("  Fetching DCF values (%d symbols) ..." % len(symbols))
    rows = []
    for i, sym in enumerate(symbols):
        data = fmp_get("discounted-cash-flow", "symbol=%s" % sym)
        if isinstance(data, list) and data:
            d = data[0]
        elif isinstance(data, dict):
            d = data
        else:
            continue
        rows.append({
            "symbol": sym,
            "dcf": d.get("dcf"),
            "stock_price": d.get("Stock Price") or d.get("stockPrice"),
        })
        if (i + 1) % 100 == 0:
            log("    %d/%d" % (i + 1, len(symbols)))
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        # Compute DCF upside
        df["dcf_upside"] = (df["dcf"] - df["stock_price"]) / df["stock_price"].replace(0, np.nan)
        df.to_parquet(cache, index=False)
    log("    DCF values: %d symbols" % len(df))
    return df


def fetch_financial_growth(symbols):
    """Fetch pre-computed financial growth rates from FMP."""
    cache = CACHE_DIR / "financial_growth.parquet"
    if cache.exists() and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days < 7:
        return pd.read_parquet(cache)

    log("  Fetching financial growth (%d symbols) ..." % len(symbols))
    rows = []
    for i, sym in enumerate(symbols):
        data = fmp_get("financial-growth", "symbol=%s&period=quarterly&limit=20" % sym)
        if not isinstance(data, list):
            continue
        for d in data:
            rows.append({
                "symbol": sym,
                "date": d.get("date"),
                "revenue_growth": d.get("revenueGrowth"),
                "net_income_growth": d.get("netIncomeGrowth"),
                "eps_growth": d.get("epsgrowth") or d.get("epsGrowth"),
                "operating_income_growth": d.get("operatingIncomeGrowth"),
                "gross_profit_growth": d.get("grossProfitGrowth"),
                "rd_growth": d.get("rdexpenseGrowth"),
                "free_cf_growth": d.get("freeCashFlowGrowth"),
            })
        if (i + 1) % 100 == 0:
            log("    %d/%d" % (i + 1, len(symbols)))
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(cache, index=False)
    log("    Financial growth: %d rows, %d symbols" % (len(df), df["symbol"].nunique() if len(df) > 0 else 0))
    return df


def fetch_enterprise_values(symbols):
    """Fetch enterprise value and EV/EBITDA from FMP."""
    cache = CACHE_DIR / "enterprise_values.parquet"
    if cache.exists() and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days < 7:
        return pd.read_parquet(cache)

    log("  Fetching enterprise values (%d symbols) ..." % len(symbols))
    rows = []
    for i, sym in enumerate(symbols):
        data = fmp_get("enterprise-values", "symbol=%s&period=quarterly&limit=20" % sym)
        if not isinstance(data, list):
            continue
        for d in data:
            rows.append({
                "symbol": sym,
                "date": d.get("date"),
                "market_cap": d.get("marketCapitalization"),
                "enterprise_value": d.get("enterpriseValue"),
                "ev_to_revenue": d.get("enterpriseValueToRevenue"),
                "shares_outstanding": d.get("numberOfShares"),
            })
        if (i + 1) % 100 == 0:
            log("    %d/%d" % (i + 1, len(symbols)))
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(cache, index=False)
    log("    Enterprise values: %d rows" % len(df))
    return df


def fetch_company_profiles(symbols):
    """Fetch company profiles (beta, market cap, sector, industry)."""
    cache = CACHE_DIR / "company_profiles.parquet"
    if cache.exists() and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days < 7:
        return pd.read_parquet(cache)

    log("  Fetching company profiles (%d symbols) ..." % len(symbols))
    rows = []
    for i, sym in enumerate(symbols):
        data = fmp_get("profile", "symbol=%s" % sym)
        if isinstance(data, list) and data:
            d = data[0]
        elif isinstance(data, dict):
            d = data
        else:
            continue
        rows.append({
            "symbol": sym,
            "beta": d.get("beta"),
            "market_cap": d.get("marketCap") or d.get("mktCap"),
            "sector": d.get("sector"),
            "industry": d.get("industry"),
            "country": d.get("country"),
            "is_etf": d.get("isEtf", False),
            "is_fund": d.get("isFund", False),
            "last_dividend": d.get("lastDividend") or d.get("lastDiv"),
            "exchange": d.get("exchange") or d.get("exchangeShortName"),
        })
        if (i + 1) % 100 == 0:
            log("    %d/%d" % (i + 1, len(symbols)))
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_parquet(cache, index=False)
    log("    Profiles: %d symbols" % len(df))
    return df


def fetch_economic_calendar():
    """Fetch economic calendar events."""
    cache = CACHE_DIR / "economic_calendar.parquet"
    if cache.exists() and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days < 1:
        return pd.read_parquet(cache)

    log("  Fetching economic calendar ...")
    rows = []
    # Fetch in chunks
    start = datetime(2022, 1, 1)
    end = datetime.now()
    while start < end:
        chunk_end = min(start + timedelta(days=90), end)
        data = fmp_get("economic-calendar",
                       "from=%s&to=%s" % (start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        if isinstance(data, list):
            for d in data:
                rows.append({
                    "date": d.get("date"),
                    "country": d.get("country"),
                    "event": d.get("event"),
                    "actual": d.get("actual"),
                    "estimate": d.get("estimate"),
                    "previous": d.get("previous"),
                    "impact": d.get("impact"),
                })
        start = chunk_end + timedelta(days=1)
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        # Filter to US high-impact events
        df = df[(df["country"] == "US") | (df["impact"] == "High")]
        df.to_parquet(cache, index=False)
    log("    Economic calendar: %d events" % len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Massive Data Fetchers
# ══════════════════════════════════════════════════════════════════════════════

def fetch_crypto_forex_extended():
    """Fetch BTC, ETH, EUR/USD, GBP/USD daily bars."""
    cache = CACHE_DIR / "crypto_forex_extended.parquet"
    if cache.exists() and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days < 1:
        return pd.read_parquet(cache)

    log("  Fetching crypto/forex extended ...")
    result = pd.DataFrame()

    tickers = [
        ("X:BTCUSD", "btc"),
        ("X:ETHUSD", "eth"),
        ("C:EURUSD", "eurusd"),
        ("C:GBPUSD", "gbpusd"),
        ("C:USDJPY", "usdjpy"),
    ]

    for ticker, prefix in tickers:
        data = massive_get("/v2/aggs/ticker/%s/range/1/day/2018-01-01/2027-01-01" % ticker,
                            {"limit": 50000, "sort": "asc"})
        bars = data.get("results", [])
        if bars:
            df = pd.DataFrame(bars)
            df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
            df = df.set_index("date")
            result[prefix + "_close"] = df["c"]
            result[prefix + "_volume"] = df["v"]
            log("    %s: %d bars" % (ticker, len(df)))
        time.sleep(0.3)

    # Compute derived features
    for prefix in ["btc", "eth"]:
        if prefix + "_close" in result:
            result[prefix + "_ret_5d"] = result[prefix + "_close"].pct_change(5)
            result[prefix + "_ret_20d"] = result[prefix + "_close"].pct_change(20)
            result[prefix + "_ret_60d"] = result[prefix + "_close"].pct_change(60)
            result[prefix + "_vol_20d"] = result[prefix + "_close"].pct_change().rolling(20).std()
            result[prefix + "_sma50_dist"] = (
                result[prefix + "_close"] / result[prefix + "_close"].rolling(50).mean() - 1
            )

    for prefix in ["eurusd", "gbpusd", "usdjpy"]:
        if prefix + "_close" in result:
            result[prefix + "_ret_5d"] = result[prefix + "_close"].pct_change(5)
            result[prefix + "_ret_20d"] = result[prefix + "_close"].pct_change(20)

    if len(result) > 0:
        result.to_parquet(cache)
    log("    Total crypto/forex features: %d columns" % len(result.columns))
    return result


def fetch_options_volume_history(symbols, lookback_months=6):
    """
    Fetch historical put/call volume for top stocks by downloading
    ATM option bars. Limited to top 50 stocks for speed.
    """
    cache = CACHE_DIR / "options_pcr.parquet"
    if cache.exists() and (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).days < 3:
        return pd.read_parquet(cache)

    log("  Fetching options volume history (top %d stocks, %d months) ..." % (
        min(len(symbols), 50), lookback_months))

    # Get current prices to know ATM strikes
    rows = []
    top_syms = symbols[:50]  # limit for speed

    for si, sym in enumerate(top_syms):
        # Get stock price history to know approximate ATM strike per month
        stock_data = massive_get("/v2/aggs/ticker/%s/range/1/month/2022-01-01/2025-12-31" % sym,
                                  {"limit": 50})
        monthly_prices = stock_data.get("results", [])
        if not monthly_prices:
            continue

        for mp in monthly_prices[-lookback_months * 2:]:
            month_date = pd.Timestamp(mp["t"], unit="ms").normalize()
            price = mp["c"]
            # Round to nearest $5 for strike
            atm_strike = round(price / 5) * 5
            if atm_strike <= 0:
                continue

            # Find expiration ~30 days out (3rd Friday of next month approx)
            exp_month = month_date + pd.DateOffset(months=1)
            exp_str = exp_month.strftime("%y%m")

            # Build option tickers for ATM call and put
            call_ticker = "O:%s%s%02dC%08d" % (sym, exp_str, 21, atm_strike * 1000)
            put_ticker = "O:%s%s%02dP%08d" % (sym, exp_str, 21, atm_strike * 1000)

            # Get volume for last trading day of the month
            start = month_date.strftime("%Y-%m-%d")
            end = (month_date + pd.DateOffset(months=1)).strftime("%Y-%m-%d")

            call_vol = 0
            put_vol = 0

            for ticker, is_call in [(call_ticker, True), (put_ticker, False)]:
                data = massive_get("/v2/aggs/ticker/%s/range/1/day/%s/%s" % (ticker, start, end),
                                    {"limit": 30})
                bars = data.get("results", [])
                vol = sum(b.get("v", 0) for b in bars)
                if is_call:
                    call_vol = vol
                else:
                    put_vol = vol

            if call_vol > 0 or put_vol > 0:
                rows.append({
                    "symbol": sym,
                    "date": month_date,
                    "call_volume": call_vol,
                    "put_volume": put_vol,
                    "pc_ratio": put_vol / max(call_vol, 1),
                    "total_option_vol": call_vol + put_vol,
                })

        if (si + 1) % 10 == 0:
            log("    %d/%d symbols" % (si + 1, len(top_syms)))
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        df.to_parquet(cache, index=False)
    log("    Options P/C history: %d rows, %d symbols" % (len(df), df["symbol"].nunique() if len(df) > 0 else 0))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 80)
    log("  FETCHING ALL AVAILABLE DATA SOURCES")
    log("=" * 80)

    # Get symbol list from SP500 constituents (no dependency on features.parquet)
    from sp500_universe import get_stock_symbols, get_etf_symbols
    all_symbols = sorted(get_stock_symbols())
    etfs = set(get_etf_symbols())
    stock_symbols = [s for s in all_symbols if s not in etfs]
    log("  Symbols: %d stocks, %d ETFs" % (len(stock_symbols), len(etfs)))

    # ── FMP Data ──
    log("\n── FMP Data ──")
    pt = fetch_price_targets(stock_symbols)
    dcf = fetch_dcf_values(stock_symbols)
    growth = fetch_financial_growth(stock_symbols)
    ev = fetch_enterprise_values(stock_symbols)
    profiles = fetch_company_profiles(stock_symbols)
    econ = fetch_economic_calendar()

    # ── Massive Data ──
    log("\n── Massive Data ──")
    crypto_fx = fetch_crypto_forex_extended()
    # options_pcr = fetch_options_volume_history(stock_symbols)  # slow, enable if needed

    elapsed = time.time() - t0
    log("\nTotal fetch time: %.0fs (%.1f min)" % (elapsed, elapsed / 60))

    # Summary
    log("\n── DATA SUMMARY ──")
    log("  Price targets:    %d symbols" % (len(pt) if pt is not None else 0))
    log("  DCF values:       %d symbols" % (len(dcf) if dcf is not None else 0))
    log("  Financial growth: %d rows" % (len(growth) if growth is not None else 0))
    log("  Enterprise value: %d rows" % (len(ev) if ev is not None else 0))
    log("  Company profiles: %d symbols" % (len(profiles) if profiles is not None else 0))
    log("  Economic calendar: %d events" % (len(econ) if econ is not None else 0))
    log("  Crypto/Forex:     %d cols x %d rows" % (
        len(crypto_fx.columns) if crypto_fx is not None else 0,
        len(crypto_fx) if crypto_fx is not None else 0))


if __name__ == "__main__":
    main()
