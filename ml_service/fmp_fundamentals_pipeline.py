"""
FMP Fundamentals Pipeline
=========================
Fetches fundamental data from FMP Premium /stable/ API endpoints
and caches to data/fundamentals.parquet for feature engineering.

Endpoints fetched per symbol:
  - Income statements (quarterly, 16 quarters)
  - Balance sheets (quarterly, 16 quarters)
  - Cash flow (quarterly, 16 quarters)
  - Key metrics (quarterly, 16 quarters)
  - Financial ratios (quarterly, 16 quarters)
  - Analyst estimates (annual, 4 years)
  - Earnings calendar
  - Insider trading (last 100 transactions)

Run with:
    python3 fmp_fundamentals_pipeline.py
"""

import os
import json
import ssl
import time
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "fundamentals_cache"
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "fundamentals.parquet"
CACHE_TTL = 3  # days before re-fetching

# Individual stocks with fundamental data (ETFs excluded)
# Use historical S&P 500 union to include delisted/acquired stocks
from sp500_universe import get_stock_symbols, get_etf_symbols
from sp500_history import get_sp500_on_date, load_sp500_changes

def _build_stock_universe():
    """Build union of all historical S&P 500 stock members (excludes ETFs)."""
    load_sp500_changes()
    etf_set = set(get_etf_symbols())
    historical = set()
    d = pd.Timestamp.today() - pd.Timedelta(days=365 * 10 + 220)
    end = pd.Timestamp.today()
    while d <= end:
        members = get_sp500_on_date(d)
        historical |= members
        d += pd.Timedelta(days=30)
    # Exclude ETFs — they don't have fundamentals
    stocks_only = sorted((historical | set(get_stock_symbols())) - etf_set)
    print(f"  FMP universe: {len(stocks_only)} stocks "
          f"(current SP500 + {len(stocks_only) - len(get_stock_symbols())} historical)")
    return stocks_only

STOCK_SYMBOLS = _build_stock_universe()

# SSL context (Mac Python sometimes lacks certs)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _fmp_fetch(endpoint: str, params: str = "", retries: int = 2) -> list:
    """Fetch from FMP /stable/ endpoint with retries and SSL handling."""
    url = (
        f"https://financialmodelingprep.com/stable/{endpoint}"
        f"?{params}&apikey={FMP_API_KEY}"
    )
    for attempt in range(retries + 1):
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
            if attempt < retries:
                time.sleep(1)
                continue
            raise
    return []


def _cache_path(symbol: str, endpoint: str) -> Path:
    """Cache file path for a symbol/endpoint combination."""
    safe = endpoint.replace("/", "_")
    return CACHE_DIR / f"{symbol}_{safe}.json"


def _fetch_cached(symbol: str, endpoint: str, params: str = "") -> list:
    """Fetch with disk cache."""
    cache = _cache_path(symbol, endpoint)

    if cache.exists():
        age = (datetime.today() - datetime.fromtimestamp(cache.stat().st_mtime)).days
        if age < CACHE_TTL:
            try:
                with open(cache) as f:
                    return json.load(f)
            except Exception:
                pass

    try:
        data = _fmp_fetch(endpoint, f"symbol={symbol}&{params}")
        with open(cache, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"    FAILED {endpoint}: {e}")
        return []


def fetch_symbol_data(symbol: str) -> dict:
    """Fetch all fundamental data for one symbol."""
    result = {}

    # Income statements (60 quarters = 15 years)
    result["income"] = _fetch_cached(symbol, "income-statement", "period=quarter&limit=60")

    # Balance sheets
    result["balance"] = _fetch_cached(symbol, "balance-sheet-statement", "period=quarter&limit=60")

    # Cash flow
    result["cashflow"] = _fetch_cached(symbol, "cash-flow-statement", "period=quarter&limit=60")

    # Key metrics
    result["metrics"] = _fetch_cached(symbol, "key-metrics", "period=quarter&limit=60")

    # Financial ratios
    result["ratios"] = _fetch_cached(symbol, "ratios", "period=quarter&limit=60")

    # Analyst estimates
    result["estimates"] = _fetch_cached(symbol, "analyst-estimates", "period=annual&limit=4")

    # Earnings calendar
    result["earnings"] = _fetch_cached(symbol, "earnings", "")

    # Insider trading
    result["insiders"] = _fetch_cached(symbol, "insider-trading/search", "page=0&limit=100")

    return result


def build_income_df(data: list, symbol: str) -> pd.DataFrame:
    """Parse income statement data into a DataFrame."""
    rows = []
    for item in data:
        try:
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(item["date"]),
                "filing_date": pd.Timestamp(item.get("filingDate")) if item.get("filingDate") else pd.NaT,
                "fiscal_year": item.get("fiscalYear"),
                "period": item.get("period"),
                "revenue": item.get("revenue"),
                "gross_profit": item.get("grossProfit"),
                "operating_income": item.get("operatingIncome"),
                "net_income": item.get("netIncome"),
                "eps": item.get("eps"),
                "eps_diluted": item.get("epsDiluted"),
                "rd_expense": item.get("researchAndDevelopmentExpenses"),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def build_ratios_df(data: list, symbol: str) -> pd.DataFrame:
    """Parse ratios data."""
    rows = []
    for item in data:
        try:
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(item["date"]),
                "gross_profit_margin": item.get("grossProfitMargin"),
                "operating_profit_margin": item.get("operatingProfitMargin"),
                "net_profit_margin": item.get("netProfitMargin"),
                "current_ratio": item.get("currentRatio"),
                "debt_to_equity": item.get("debtToEquityRatio"),
                "pe_ratio": item.get("priceToEarningsRatio"),
                "ps_ratio": item.get("priceToSalesRatio"),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def build_metrics_df(data: list, symbol: str) -> pd.DataFrame:
    """Parse key metrics data."""
    rows = []
    for item in data:
        try:
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(item["date"]),
                "roe": item.get("returnOnEquity"),
                "roa": item.get("returnOnAssets"),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def build_estimates_df(data: list, symbol: str) -> pd.DataFrame:
    """Parse analyst estimates."""
    rows = []
    for item in data:
        try:
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(item["date"]),
                "eps_avg": item.get("epsAvg"),
                "eps_high": item.get("epsHigh"),
                "eps_low": item.get("epsLow"),
                "revenue_avg": item.get("revenueAvg"),
                "revenue_high": item.get("revenueHigh"),
                "revenue_low": item.get("revenueLow"),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def build_earnings_df(data: list, symbol: str) -> pd.DataFrame:
    """Parse earnings calendar."""
    rows = []
    for item in data:
        try:
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(item["date"]),
                "eps_actual": item.get("epsActual"),
                "eps_estimated": item.get("epsEstimated"),
                "revenue_actual": item.get("revenueActual"),
                "revenue_estimated": item.get("revenueEstimated"),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def build_insiders_df(data: list, symbol: str) -> pd.DataFrame:
    """Parse insider trading data."""
    rows = []
    for item in data:
        try:
            tx_type = item.get("acquisitionOrDisposition", "")
            shares = item.get("securitiesTransacted", 0) or 0
            tx_date = item.get("transactionDate")
            if not tx_date:
                continue
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(tx_date),
                "is_buy": 1 if tx_type == "A" else 0,
                "is_sell": 1 if tx_type == "D" else 0,
                "shares": float(shares),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def main():
    if not FMP_API_KEY:
        print("ERROR: FMP_API_KEY not set in .env")
        return

    print("=" * 70)
    print("FMP Fundamentals Pipeline")
    print(f"Symbols: {len(STOCK_SYMBOLS)} stocks")
    print(f"Cache TTL: {CACHE_TTL} days")
    print("=" * 70)

    all_income = []
    all_ratios = []
    all_metrics = []
    all_estimates = []
    all_earnings = []
    all_insiders = []

    cached_count = 0
    api_count = 0
    failed_symbols = []

    for i, sym in enumerate(STOCK_SYMBOLS, 1):
        cache_file = _cache_path(sym, "income-statement")
        source = "cache" if (cache_file.exists() and
                             (datetime.today() - datetime.fromtimestamp(
                                 cache_file.stat().st_mtime)).days < CACHE_TTL) else "API"

        raw = fetch_symbol_data(sym)

        inc = build_income_df(raw["income"], sym)
        rat = build_ratios_df(raw["ratios"], sym)
        met = build_metrics_df(raw["metrics"], sym)
        est = build_estimates_df(raw["estimates"], sym)
        ear = build_earnings_df(raw["earnings"], sym)
        ins = build_insiders_df(raw["insiders"], sym)

        if len(inc) > 0: all_income.append(inc)
        if len(rat) > 0: all_ratios.append(rat)
        if len(met) > 0: all_metrics.append(met)
        if len(est) > 0: all_estimates.append(est)
        if len(ear) > 0: all_earnings.append(ear)
        if len(ins) > 0: all_insiders.append(ins)

        if len(inc) == 0 and len(rat) == 0:
            failed_symbols.append(sym)

        if source == "cache":
            cached_count += 1
        else:
            api_count += 1
            time.sleep(0.05)  # FMP Premium: 750 calls/min

        # Progress every 50 symbols
        if i % 50 == 0 or i == len(STOCK_SYMBOLS):
            print(f"  [{i}/{len(STOCK_SYMBOLS)}] processed ({cached_count} cached, {api_count} API, {len(failed_symbols)} no data)")

    if failed_symbols and len(failed_symbols) <= 20:
        print(f"  Symbols with no data: {', '.join(failed_symbols)}")

    # Combine all data
    print(f"\n{'─'*70}")
    print("Combining data ...")

    combined = {}
    if all_income:
        combined["income"] = pd.concat(all_income, ignore_index=True)
        print(f"  Income statements: {len(combined['income']):,} rows")
    if all_ratios:
        combined["ratios"] = pd.concat(all_ratios, ignore_index=True)
        print(f"  Ratios:            {len(combined['ratios']):,} rows")
    if all_metrics:
        combined["metrics"] = pd.concat(all_metrics, ignore_index=True)
        print(f"  Key metrics:       {len(combined['metrics']):,} rows")
    if all_estimates:
        combined["estimates"] = pd.concat(all_estimates, ignore_index=True)
        print(f"  Estimates:         {len(combined['estimates']):,} rows")
    if all_earnings:
        combined["earnings"] = pd.concat(all_earnings, ignore_index=True)
        print(f"  Earnings:          {len(combined['earnings']):,} rows")
    if all_insiders:
        combined["insiders"] = pd.concat(all_insiders, ignore_index=True)
        print(f"  Insiders:          {len(combined['insiders']):,} rows")

    # Save as parquet (multi-table in one file via pickle-compatible approach)
    # Actually, save each table separately for cleaner access
    for key, df in combined.items():
        outfile = DATA_DIR / f"fundamentals_{key}.parquet"
        df.to_parquet(outfile, index=False, engine="pyarrow", compression="snappy")
        print(f"  Saved {outfile.name}: {len(df):,} rows, {outfile.stat().st_size / 1024:.1f} KB")

    # ── Verification ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")

    if "income" in combined:
        inc = combined["income"]
        # AAPL latest quarter
        aapl = inc[(inc["symbol"] == "AAPL")].sort_values("date", ascending=False)
        if len(aapl) > 0:
            latest = aapl.iloc[0]
            print(f"\n  AAPL latest quarter ({latest['date'].date()}, period {latest['period']}):")
            print(f"    Revenue:    ${latest['revenue']/1e9:.1f}B")
            print(f"    Gross profit: ${latest['gross_profit']/1e9:.1f}B")
            rev = latest['revenue']
            gp = latest['gross_profit']
            if rev and rev > 0:
                print(f"    Gross margin: {gp/rev*100:.1f}%")
            print(f"    EPS diluted: ${latest['eps_diluted']:.2f}")
            print(f"    Filing date: {latest['filing_date'].date() if pd.notna(latest['filing_date']) else 'N/A'}")

        # Coverage summary
        syms_with_data = inc["symbol"].nunique()
        avg_quarters = inc.groupby("symbol").size().mean()
        print(f"\n  Income statement coverage:")
        print(f"    Symbols with data: {syms_with_data}/{len(STOCK_SYMBOLS)}")
        print(f"    Avg quarters per symbol: {avg_quarters:.0f}")
        print(f"    Date range: {inc['date'].min().date()} → {inc['date'].max().date()}")

    if "ratios" in combined:
        rat = combined["ratios"]
        aapl_r = rat[rat["symbol"] == "AAPL"].sort_values("date", ascending=False)
        if len(aapl_r) > 0:
            r = aapl_r.iloc[0]
            print(f"\n  AAPL ratios ({r['date'].date()}):")
            print(f"    PE ratio:      {r['pe_ratio']:.1f}")
            print(f"    PS ratio:      {r['ps_ratio']:.1f}")
            print(f"    Current ratio: {r['current_ratio']:.2f}")
            print(f"    Debt/Equity:   {r['debt_to_equity']:.2f}")

    if "metrics" in combined:
        met = combined["metrics"]
        aapl_m = met[met["symbol"] == "AAPL"].sort_values("date", ascending=False)
        if len(aapl_m) > 0:
            m = aapl_m.iloc[0]
            print(f"\n  AAPL key metrics ({m['date'].date()}):")
            print(f"    ROE: {m['roe']*100:.1f}%")
            print(f"    ROA: {m['roa']*100:.1f}%")

    print(f"\n{'='*70}")
    print("FMP Fundamentals Pipeline complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
