"""
Enhanced Features Builder
=========================
Computes additional features from FMP balance sheet and cash flow
for use in the v8 strategy framework.

Features added:
  1. gross_profitability: gross_profit / total_assets (Novy-Marx quality factor)
  2. fcf_yield: free_cash_flow / market_cap proxy
  3. accruals: (net_income - operating_cf) / total_assets (earnings quality)
  4. revenue_acceleration: this Q rev growth vs last Q rev growth
  5. debt_change: QoQ change in total debt / total assets

Usage:
    from strategies.enhanced_features import build_enhanced_features
    enhanced = build_enhanced_features()
    # Returns DataFrame indexed by (date, symbol) with new feature columns
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

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
CACHE_DIR = DATA_DIR / "enhanced_features_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _fmp_get(endpoint, params=""):
    url = "https://financialmodelingprep.com/stable/%s?%s&apikey=%s" % (endpoint, params, FMP_KEY)
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def _massive_get(endpoint, params=None):
    if params is None:
        params = {}
    params["apiKey"] = MASSIVE_KEY
    resp = requests.get("https://api.polygon.io" + endpoint, params=params, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    return {}


def fetch_balance_sheets(symbols):
    """Fetch quarterly balance sheets from FMP. Cache to disk."""
    cache_file = CACHE_DIR / "balance_sheets.parquet"
    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < 7:
            return pd.read_parquet(cache_file)

    print("  Fetching balance sheets from FMP (%d symbols) ..." % len(symbols))
    rows = []
    for i, sym in enumerate(symbols):
        data = _fmp_get("balance-sheet-statement", "symbol=%s&period=quarterly&limit=40" % sym)
        for d in (data if isinstance(data, list) else []):
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp(d.get("date")),
                "filing_date": pd.Timestamp(d.get("filingDate")) if d.get("filingDate") else None,
                "total_assets": d.get("totalAssets"),
                "total_equity": d.get("totalStockholdersEquity"),
                "total_debt": d.get("totalDebt"),
                "shares_outstanding": d.get("commonStockSharesOutstanding"),
            })
        if (i + 1) % 100 == 0:
            print("    %d/%d" % (i + 1, len(symbols)))
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_parquet(cache_file, index=False)
    print("  Balance sheets: %d rows" % len(df))
    return df


def fetch_cash_flows(symbols):
    """Fetch quarterly cash flow statements from FMP. Cache to disk."""
    cache_file = CACHE_DIR / "cash_flows.parquet"
    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < 7:
            return pd.read_parquet(cache_file)

    print("  Fetching cash flows from FMP (%d symbols) ..." % len(symbols))
    rows = []
    for i, sym in enumerate(symbols):
        data = _fmp_get("cash-flow-statement", "symbol=%s&period=quarterly&limit=40" % sym)
        for d in (data if isinstance(data, list) else []):
            rows.append({
                "symbol": sym,
                "date": pd.Timestamp(d.get("date")),
                "filing_date": pd.Timestamp(d.get("filingDate")) if d.get("filingDate") else None,
                "operating_cf": d.get("operatingCashFlow"),
                "capex": d.get("capitalExpenditure"),
                "free_cf": d.get("freeCashFlow"),
            })
        if (i + 1) % 100 == 0:
            print("    %d/%d" % (i + 1, len(symbols)))
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_parquet(cache_file, index=False)
    print("  Cash flows: %d rows" % len(df))
    return df


def build_enhanced_features(symbols=None):
    """
    Build enhanced features and return as a dict of {feature_name: {date: {symbol: value}}}.
    This format allows O(1) lookup in the strategy engine.
    """
    if symbols is None:
        feat = pd.read_parquet(DATA_DIR / "features.parquet", columns=["symbol"])
        symbols = sorted(feat["symbol"].unique().tolist())
        # Filter to stocks only (not ETFs)
        from sp500_universe import get_etf_symbols
        etfs = set(get_etf_symbols())
        symbols = [s for s in symbols if s not in etfs]

    print("Building enhanced features for %d symbols ..." % len(symbols))

    # Fetch data
    balance = fetch_balance_sheets(symbols)
    cashflow = fetch_cash_flows(symbols)
    income = pd.read_parquet(DATA_DIR / "fundamentals_income.parquet")

    # Build features
    enhanced = {}

    if len(balance) > 0 and len(income) > 0:
        # Gross profitability = gross_profit / total_assets
        merged = income.merge(balance[["symbol", "date", "total_assets"]],
                               on=["symbol", "date"], how="inner")
        merged["gross_profitability"] = merged["gross_profit"] / merged["total_assets"].replace(0, np.nan)

        # Accruals = (net_income - operating_cf) / total_assets
        if len(cashflow) > 0:
            merged2 = merged.merge(cashflow[["symbol", "date", "operating_cf"]],
                                    on=["symbol", "date"], how="left")
            merged2["accruals"] = (merged2["net_income"] - merged2["operating_cf"].fillna(0)) / \
                                   merged2["total_assets"].replace(0, np.nan)
        else:
            merged2 = merged
            merged2["accruals"] = np.nan

        # Revenue acceleration
        merged2 = merged2.sort_values(["symbol", "date"])
        merged2["rev_growth"] = merged2.groupby("symbol")["revenue"].pct_change()
        merged2["rev_accel"] = merged2.groupby("symbol")["rev_growth"].diff()

        # Debt change
        if "total_debt" in balance.columns:
            debt = balance[["symbol", "date", "total_debt", "total_assets"]].sort_values(["symbol", "date"])
            debt["debt_change"] = debt.groupby("symbol")["total_debt"].pct_change()
            merged2 = merged2.merge(debt[["symbol", "date", "debt_change"]],
                                     on=["symbol", "date"], how="left")

        # Use filing_date for point-in-time alignment
        date_col = "filing_date" if "filing_date" in merged2.columns and merged2["filing_date"].notna().any() else "date"

        for feat_name in ["gross_profitability", "accruals", "rev_accel", "debt_change"]:
            if feat_name in merged2.columns:
                enhanced[feat_name] = merged2[["symbol", date_col, feat_name]].dropna()
                enhanced[feat_name] = enhanced[feat_name].rename(columns={date_col: "date"})

    # FCF yield
    if len(cashflow) > 0 and len(balance) > 0:
        fcf = cashflow.merge(balance[["symbol", "date", "shares_outstanding"]],
                              on=["symbol", "date"], how="inner")
        # We don't have market cap directly, but FCF per share is still useful
        fcf["fcf_per_share"] = fcf["free_cf"] / fcf["shares_outstanding"].replace(0, np.nan)
        enhanced["fcf_per_share"] = fcf[["symbol", "date", "fcf_per_share"]].dropna()

    print("  Enhanced features built: %s" % ", ".join(enhanced.keys()))
    return enhanced


if __name__ == "__main__":
    enhanced = build_enhanced_features()
    for name, df in enhanced.items():
        if isinstance(df, pd.DataFrame) and "symbol" in df.columns:
            print("%s: %d rows, %d symbols" % (name, len(df), df["symbol"].nunique()))
        elif isinstance(df, pd.DataFrame):
            print("%s: %d rows (market-level)" % (name, len(df)))
