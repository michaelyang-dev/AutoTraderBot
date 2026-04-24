#!/usr/bin/env python3
"""
Short-Specific Feature Engineering
====================================
Computes features designed to predict underperformers (bottom 20%)
from fundamentals cache (cash flow, balance sheet, income, insiders, earnings).

These features focus on DETERIORATION signals:
  - Accruals quality (earnings vs cash flow divergence)
  - Debt growth and leverage changes
  - Margin compression
  - Insider selling acceleration
  - Negative earnings surprise patterns
  - Price breakdown signals (computed from existing price features)

Output: data/features_short_enriched.parquet
  = original features.parquet + ~25 new short-specific columns

Usage:
    python3 short_features.py
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "fundamentals_cache"
INPUT_FILE = DATA_DIR / "features.parquet"
OUTPUT_FILE = DATA_DIR / "features_short_enriched.parquet"


def log(msg: str):
    print(msg, flush=True)


# ── Load & parse fundamentals cache ──────────────────────────────────────────

def load_cash_flow(symbol: str) -> pd.DataFrame:
    """Load cash flow statements from cache."""
    path = CACHE_DIR / f"{symbol}_cash-flow-statement.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


def load_balance_sheet(symbol: str) -> pd.DataFrame:
    """Load balance sheet statements from cache."""
    path = CACHE_DIR / f"{symbol}_balance-sheet-statement.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


def load_income(symbol: str) -> pd.DataFrame:
    """Load income statements from cache."""
    path = CACHE_DIR / f"{symbol}_income-statement.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["filingDate"] = pd.to_datetime(df["filingDate"])
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


# ── Compute short-specific quarterly features ────────────────────────────────

def compute_quarterly_short_features(symbol: str) :
    """
    Compute short-specific features from quarterly fundamentals.
    Returns DataFrame with columns: filing_date, feature1, feature2, ...
    Uses filingDate for point-in-time correctness.
    """
    cf = load_cash_flow(symbol)
    bs = load_balance_sheet(symbol)
    inc = load_income(symbol)

    if cf.empty or bs.empty or inc.empty:
        return None

    # Merge on (date, period) to align quarters
    cf = cf.rename(columns={"filingDate": "filing_date"})
    bs = bs.rename(columns={"filingDate": "filing_date"})
    inc = inc.rename(columns={"filingDate": "filing_date"})

    # Use income filing_date as the anchor (point-in-time)
    merged = inc[["date", "filing_date", "revenue", "grossProfit", "operatingIncome",
                   "netIncome", "costOfRevenue", "sellingGeneralAndAdministrativeExpenses",
                   "interestExpense", "ebitda"]].copy()

    # Merge cash flow
    cf_cols = ["date", "operatingCashFlow", "freeCashFlow", "netIncome",
               "capitalExpenditure", "depreciationAndAmortization",
               "changeInWorkingCapital", "accountsReceivables", "inventory",
               "commonStockRepurchased", "netDebtIssuance"]
    cf_sub = cf[cf_cols].rename(columns={
        "netIncome": "cf_netIncome",
        "operatingCashFlow": "ocf",
        "freeCashFlow": "fcf",
        "capitalExpenditure": "capex",
        "depreciationAndAmortization": "dna",
        "changeInWorkingCapital": "delta_wc",
        "accountsReceivables": "delta_ar",
        "inventory": "delta_inv",
        "commonStockRepurchased": "buybacks",
        "netDebtIssuance": "net_debt_issuance",
    })
    merged = merged.merge(cf_sub, on="date", how="left")

    # Merge balance sheet
    bs_cols = ["date", "totalAssets", "totalLiabilities", "totalDebt",
               "totalCurrentAssets", "totalCurrentLiabilities",
               "totalStockholdersEquity", "netDebt", "inventory",
               "cashAndCashEquivalents", "longTermDebt", "shortTermDebt"]
    bs_sub = bs[bs_cols].rename(columns={
        "totalAssets": "total_assets",
        "totalLiabilities": "total_liabilities",
        "totalDebt": "total_debt",
        "totalCurrentAssets": "current_assets",
        "totalCurrentLiabilities": "current_liabilities",
        "totalStockholdersEquity": "equity",
        "netDebt": "net_debt",
        "inventory": "bs_inventory",
        "cashAndCashEquivalents": "cash",
        "longTermDebt": "lt_debt",
        "shortTermDebt": "st_debt",
    })
    merged = merged.merge(bs_sub, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)
    if len(merged) < 2:
        return None

    result = pd.DataFrame()
    result["filing_date"] = merged["filing_date"]

    # ── 1. ACCRUALS QUALITY ──────────────────────────────────────────
    # High accruals = earnings not backed by cash = red flag
    ni = merged["netIncome"].values.astype(float)
    ocf = merged["ocf"].values.astype(float)
    ta = merged["total_assets"].values.astype(float)

    # Accruals ratio: (NI - OCF) / Total Assets
    # High positive = aggressive accounting
    accruals = ni - ocf
    result["accruals_ratio"] = np.where(
        np.abs(ta) > 1e6, accruals / ta, np.nan)

    # Accruals ratio change (QoQ) — deterioration signal
    ar = result["accruals_ratio"].values
    result["accruals_ratio_change"] = np.concatenate([[np.nan], np.diff(ar)])

    # FCF / Net Income — low = cash flow doesn't support earnings
    fcf = merged["fcf"].values.astype(float)
    result["fcf_to_ni"] = np.where(
        np.abs(ni) > 1e4, fcf / ni, np.nan)

    # OCF / Net Income — operating quality
    result["ocf_to_ni"] = np.where(
        np.abs(ni) > 1e4, ocf / ni, np.nan)

    # ── 2. WORKING CAPITAL DETERIORATION ─────────────────────────────
    # Rising AR + rising inventory = stuffing channels
    rev = merged["revenue"].values.astype(float)
    delta_ar = merged["delta_ar"].values.astype(float)
    delta_inv = merged["delta_inv"].values.astype(float)

    # Days Sales in Receivables change (proxy)
    result["ar_to_revenue"] = np.where(
        np.abs(rev) > 1e4, delta_ar / rev, np.nan)

    # Inventory build relative to revenue
    result["inv_to_revenue"] = np.where(
        np.abs(rev) > 1e4, delta_inv / rev, np.nan)

    # ── 3. DEBT & LEVERAGE DETERIORATION ─────────────────────────────
    td = merged["total_debt"].values.astype(float)

    # Debt growth QoQ
    td_prev = np.concatenate([[np.nan], td[:-1]])
    result["debt_growth_qoq"] = np.where(
        np.abs(td_prev) > 1e4, (td - td_prev) / np.abs(td_prev), np.nan)

    # Debt growth YoY (4 quarters back)
    if len(td) >= 5:
        td_yoy = np.concatenate([np.full(4, np.nan), td[:-4]])
        result["debt_growth_yoy"] = np.where(
            np.abs(td_yoy) > 1e4, (td - td_yoy) / np.abs(td_yoy), np.nan)
    else:
        result["debt_growth_yoy"] = np.nan

    # Leverage ratio: total_debt / total_assets
    result["leverage_ratio"] = np.where(
        np.abs(ta) > 1e6, td / ta, np.nan)

    # Leverage change QoQ
    lr = result["leverage_ratio"].values
    result["leverage_change_qoq"] = np.concatenate([[np.nan], np.diff(lr)])

    # Net debt / EBITDA
    nd = merged["net_debt"].values.astype(float)
    ebitda = merged["ebitda"].values.astype(float)
    result["net_debt_to_ebitda"] = np.where(
        np.abs(ebitda) > 1e4, nd / ebitda, np.nan)

    # Interest coverage: EBITDA / interest expense
    int_exp = merged["interestExpense"].values.astype(float)
    result["interest_coverage"] = np.where(
        np.abs(int_exp) > 1e3, ebitda / np.abs(int_exp), np.nan)

    # Interest coverage deterioration (QoQ change)
    ic = result["interest_coverage"].values
    result["interest_coverage_change"] = np.concatenate([[np.nan], np.diff(ic)])

    # ── 4. MARGIN DETERIORATION (more granular than existing) ────────
    gp = merged["grossProfit"].values.astype(float)
    oi = merged["operatingIncome"].values.astype(float)

    gm = np.where(np.abs(rev) > 1e4, gp / rev, np.nan)
    om = np.where(np.abs(rev) > 1e4, oi / rev, np.nan)
    nm = np.where(np.abs(rev) > 1e4, ni / rev, np.nan)

    # QoQ margin changes (negative = deterioration)
    result["gross_margin_qoq_chg"] = np.concatenate([[np.nan], np.diff(gm)])
    result["operating_margin_qoq_chg"] = np.concatenate([[np.nan], np.diff(om)])
    result["net_margin_qoq_chg"] = np.concatenate([[np.nan], np.diff(nm)])

    # YoY margin changes (4 quarters back)
    if len(gm) >= 5:
        result["gross_margin_yoy_chg"] = np.concatenate([
            np.full(4, np.nan), gm[4:] - gm[:-4]])
        result["operating_margin_yoy_chg"] = np.concatenate([
            np.full(4, np.nan), om[4:] - om[:-4]])
    else:
        result["gross_margin_yoy_chg"] = np.nan
        result["operating_margin_yoy_chg"] = np.nan

    # SGA burden: SGA / Revenue (rising = cost discipline failing)
    sga = merged["sellingGeneralAndAdministrativeExpenses"].values.astype(float)
    result["sga_to_revenue"] = np.where(
        np.abs(rev) > 1e4, sga / rev, np.nan)
    sga_ratio = result["sga_to_revenue"].values
    result["sga_burden_change"] = np.concatenate([[np.nan], np.diff(sga_ratio)])

    # ── 5. REVENUE QUALITY ──────────────────────────────────────────
    # Revenue deceleration
    rev_qoq = np.where(
        np.abs(np.concatenate([[np.nan], rev[:-1]])) > 1e4,
        (rev - np.concatenate([[np.nan], rev[:-1]])) /
        np.abs(np.concatenate([[1], rev[:-1]])),
        np.nan)
    result["revenue_growth_qoq_chg"] = np.concatenate([[np.nan], np.diff(rev_qoq)])

    # ── 6. CAPEX & INVESTMENT BURDEN ────────────────────────────────
    capex = np.abs(merged["capex"].values.astype(float))
    result["capex_to_revenue"] = np.where(
        np.abs(rev) > 1e4, capex / rev, np.nan)
    result["capex_to_ocf"] = np.where(
        np.abs(ocf) > 1e4, capex / ocf, np.nan)

    # ── 7. SHAREHOLDER DILUTION / BUYBACK SIGNAL ────────────────────
    buybacks = merged["buybacks"].values.astype(float)  # negative = repurchase
    result["buyback_yield"] = np.where(
        np.abs(ta) > 1e6, buybacks / ta, np.nan)  # negative = buying back

    # Net debt issuance relative to assets (positive = borrowing more)
    ndi = merged["net_debt_issuance"].values.astype(float)
    result["net_debt_issuance_to_assets"] = np.where(
        np.abs(ta) > 1e6, ndi / ta, np.nan)

    return result


def compute_earnings_short_features(symbol: str) :
    """
    Compute earnings-based short features: miss streaks, surprise trends.
    Returns DataFrame with columns: date, feature1, feature2, ...
    """
    path = CACHE_DIR / f"{symbol}_earnings.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return None
    except Exception:
        return None

    df = pd.DataFrame(data)
    if "date" not in df.columns or "epsActual" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.dropna(subset=["epsActual"])

    if len(df) < 2:
        return None

    actual = df["epsActual"].values.astype(float)
    estimated = df["epsEstimated"].values.astype(float)
    rev_actual = df.get("revenueActual", pd.Series(dtype=float)).values.astype(float)
    rev_estimated = df.get("revenueEstimated", pd.Series(dtype=float)).values.astype(float)

    result = pd.DataFrame()
    result["date"] = df["date"].values

    # EPS surprise (normalized)
    eps_surprise = np.where(
        np.abs(estimated) > 1e-4,
        (actual - estimated) / np.abs(estimated),
        np.nan)
    result["eps_surprise_pct"] = eps_surprise

    # Revenue surprise (normalized)
    rev_surprise = np.where(
        np.abs(rev_estimated) > 1e4,
        (rev_actual - rev_estimated) / np.abs(rev_estimated),
        np.nan)
    result["revenue_surprise_pct"] = rev_surprise

    # Consecutive miss streak (negative surprise)
    eps_miss_streak = np.zeros(len(df))
    streak = 0
    for i in range(len(eps_surprise)):
        if np.isnan(eps_surprise[i]):
            streak = 0
        elif eps_surprise[i] < 0:
            streak += 1
        else:
            streak = 0
        eps_miss_streak[i] = streak
    result["eps_miss_streak"] = eps_miss_streak

    # Revenue miss streak
    rev_miss_streak = np.zeros(len(df))
    streak = 0
    for i in range(len(rev_surprise)):
        if np.isnan(rev_surprise[i]):
            streak = 0
        elif rev_surprise[i] < 0:
            streak += 1
        else:
            streak = 0
        rev_miss_streak[i] = streak
    result["revenue_miss_streak"] = rev_miss_streak

    # Surprise trend: slope of last 4 surprises
    surprise_trend = np.full(len(df), np.nan)
    for i in range(3, len(eps_surprise)):
        window = eps_surprise[i-3:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 3:
            slope = np.polyfit(range(len(valid)), valid, 1)[0]
            surprise_trend[i] = slope
    result["eps_surprise_trend"] = surprise_trend

    return result


def compute_insider_short_features(symbol: str) :
    """
    Compute insider selling acceleration features.
    Returns DataFrame with columns: date, feature1, feature2, ...
    """
    path = CACHE_DIR / f"{symbol}_insider-trading_search.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return None
    except Exception:
        return None

    df = pd.DataFrame(data)
    if "filingDate" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df.get("filingDate", df.get("transactionDate")))
    df = df.dropna(subset=["date"])

    if len(df) == 0:
        return None

    # Classify transactions
    if "transactionType" in df.columns:
        df["is_sell"] = df["transactionType"].str.contains("S-", case=False, na=False) | \
                        df["transactionType"].str.contains("Sale", case=False, na=False)
        df["is_buy"] = df["transactionType"].str.contains("P-", case=False, na=False) | \
                       df["transactionType"].str.contains("Purchase", case=False, na=False)
    elif "acquisitionOrDisposition" in df.columns:
        df["is_sell"] = df["acquisitionOrDisposition"] == "D"
        df["is_buy"] = df["acquisitionOrDisposition"] == "A"
    else:
        return None

    shares = df.get("securitiesTransacted", pd.Series(0, index=df.index)).fillna(0).values.astype(float)
    df["shares"] = shares

    return df[["date", "is_buy", "is_sell", "shares"]].sort_values("date")


# ── Map quarterly features to daily dates ────────────────────────────────────

def map_quarterly_to_daily(quarterly_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Map quarterly features to daily dates using filingDate for point-in-time.
    For each trading day, uses the most recent quarter with filing_date <= that day.
    """
    if quarterly_df is None or quarterly_df.empty:
        return None

    filing_dates = quarterly_df["filing_date"].values
    feature_cols = [c for c in quarterly_df.columns if c != "filing_date"]

    result = pd.DataFrame(index=date_index)
    date_vals = date_index.values

    # Pre-compute arrays
    arrays = {col: quarterly_df[col].values.astype(float) for col in feature_cols}

    daily_arrays = {col: np.full(len(date_index), np.nan) for col in feature_cols}

    for d in range(len(date_vals)):
        dt = date_vals[d]
        q_idx = np.searchsorted(filing_dates, dt, side="right") - 1
        if q_idx >= 0 and q_idx < len(filing_dates):
            for col in feature_cols:
                daily_arrays[col][d] = arrays[col][q_idx]

    for col in feature_cols:
        result[col] = daily_arrays[col]

    return result


def map_earnings_to_daily(earnings_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map earnings features to daily dates. Each earnings date updates features."""
    if earnings_df is None or earnings_df.empty:
        return None

    earn_dates = earnings_df["date"].values
    feature_cols = [c for c in earnings_df.columns if c != "date"]

    result = pd.DataFrame(index=date_index)
    date_vals = date_index.values

    arrays = {col: earnings_df[col].values.astype(float) for col in feature_cols}
    daily_arrays = {col: np.full(len(date_index), np.nan) for col in feature_cols}

    for d in range(len(date_vals)):
        dt = date_vals[d]
        e_idx = np.searchsorted(earn_dates, dt, side="right") - 1
        if e_idx >= 0 and e_idx < len(earn_dates):
            for col in feature_cols:
                daily_arrays[col][d] = arrays[col][e_idx]

    for col in feature_cols:
        result[col] = daily_arrays[col]

    return result


def map_insider_to_daily(insider_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute insider selling features at daily granularity using rolling windows."""
    if insider_df is None or insider_df.empty:
        return None

    result = pd.DataFrame(index=date_index)
    date_vals = date_index.values

    ins_dates = insider_df["date"].values
    is_sell = insider_df["is_sell"].values.astype(bool)
    is_buy = insider_df["is_buy"].values.astype(bool)
    shares = insider_df["shares"].values.astype(float)

    sell_ratio_30d = np.full(len(date_index), np.nan)
    sell_ratio_90d = np.full(len(date_index), np.nan)
    sell_acceleration = np.full(len(date_index), np.nan)
    net_sell_shares_30d = np.full(len(date_index), np.nan)
    large_sell_flag = np.full(len(date_index), 0.0)

    # Pre-compute median transaction size for "large" threshold
    sell_shares = shares[is_sell]
    large_threshold = np.median(sell_shares) * 3 if len(sell_shares) > 5 else 1e9

    for d in range(len(date_vals)):
        dt = date_vals[d]

        # 30-day window
        cutoff_30 = dt - np.timedelta64(30, "D")
        mask_30 = (ins_dates >= cutoff_30) & (ins_dates <= dt)
        if mask_30.any():
            sells_30 = is_sell[mask_30].sum()
            total_30 = mask_30.sum()
            sell_ratio_30d[d] = sells_30 / total_30 if total_30 > 0 else np.nan

            sell_shares_30 = shares[mask_30 & is_sell].sum() if (mask_30 & is_sell).any() else 0
            buy_shares_30 = shares[mask_30 & is_buy].sum() if (mask_30 & is_buy).any() else 0
            net_sell_shares_30d[d] = sell_shares_30 - buy_shares_30  # positive = net selling

            # Large sell flag
            if (mask_30 & is_sell).any():
                max_sell = shares[mask_30 & is_sell].max()
                if max_sell > large_threshold:
                    large_sell_flag[d] = 1.0

        # 90-day window
        cutoff_90 = dt - np.timedelta64(90, "D")
        mask_90 = (ins_dates >= cutoff_90) & (ins_dates <= dt)
        if mask_90.any():
            sells_90 = is_sell[mask_90].sum()
            total_90 = mask_90.sum()
            sell_ratio_90d[d] = sells_90 / total_90 if total_90 > 0 else np.nan

        # Sell acceleration: 30d sell ratio vs 90d sell ratio
        if not np.isnan(sell_ratio_30d[d]) and not np.isnan(sell_ratio_90d[d]):
            sell_acceleration[d] = sell_ratio_30d[d] - sell_ratio_90d[d]

    result["insider_sell_ratio_30d"] = sell_ratio_30d
    result["insider_sell_ratio_90d"] = sell_ratio_90d
    result["insider_sell_acceleration"] = sell_acceleration
    result["insider_net_sell_shares_30d"] = net_sell_shares_30d
    result["insider_large_sell_flag"] = large_sell_flag

    return result


# ── Price-based short features (from existing columns) ───────────────────────

def compute_price_short_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-based short signals from existing features in the dataframe."""
    result = pd.DataFrame(index=df.index)

    # Below SMA200 flag
    if "dist_sma200" in df.columns:
        result["below_sma200"] = (df["dist_sma200"] < 0).astype(float)

    # Below SMA50 flag
    if "dist_sma50" in df.columns:
        result["below_sma50"] = (df["dist_sma50"] < 0).astype(float)

    # Death cross proxy: dist_sma50 < dist_sma200 (short MA below long MA)
    if "dist_sma50" in df.columns and "dist_sma200" in df.columns:
        result["death_cross"] = (df["dist_sma50"] < df["dist_sma200"]).astype(float)

    # Distance from 52-week high (already exists, but make a "near low" flag)
    if "dist_52w_high" in df.columns:
        result["near_52w_low"] = (df["dist_52w_high"] < -0.30).astype(float)

    # RSI oversold (but we want to identify TREND down, not mean reversion)
    # Persistently low RSI = weak stock, not a buy signal
    if "rsi_14" in df.columns:
        result["rsi_weak"] = (df["rsi_14"] < 40).astype(float)

    # Volume-price divergence: price falling but volume increasing = distribution
    if "ret_20d" in df.columns and "vol_ratio_20d" in df.columns:
        result["vol_price_divergence"] = np.where(
            (df["ret_20d"] < -0.02) & (df["vol_ratio_20d"] > 1.2),
            1.0, 0.0)

    # Consecutive down months (already exists)
    # Momentum collapse: 60d return negative AND 20d return negative
    if "ret_60d" in df.columns and "ret_20d" in df.columns:
        result["momentum_collapse"] = np.where(
            (df["ret_60d"] < -0.05) & (df["ret_20d"] < -0.02),
            1.0, 0.0)

    # Volatility expansion (high vol = instability)
    if "vol_20d" in df.columns and "vol_60d" in df.columns:
        result["vol_expansion"] = np.where(
            df["vol_60d"] > 0,
            df["vol_20d"] / df["vol_60d"],
            np.nan)

    return result


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  SHORT-SPECIFIC FEATURE ENGINEERING")
    log("=" * 70)

    # Load existing features
    log(f"\nLoading {INPUT_FILE} ...")
    feat = pd.read_parquet(INPUT_FILE)
    feat["date"] = pd.to_datetime(feat["date"])
    log(f"  Shape: {feat.shape}, date range: {feat['date'].min()} to {feat['date'].max()}")

    symbols = feat["symbol"].unique()
    log(f"  Symbols: {len(symbols)}")

    # Get unique date index (from SPY or similar)
    date_index = pd.DatetimeIndex(sorted(feat["date"].unique()))
    log(f"  Trading days: {len(date_index)}")

    # ── 1. Compute quarterly fundamentals features per symbol ────────
    log("\n  Computing quarterly short features from fundamentals cache ...")
    quarterly_features = {}
    earnings_features = {}
    insider_features = {}

    n_q = 0
    n_e = 0
    n_i = 0

    for i, sym in enumerate(symbols):
        if (i + 1) % 100 == 0 or i + 1 == len(symbols):
            log(f"    [{i+1}/{len(symbols)}] {sym} (q={n_q}, e={n_e}, i={n_i})")

        # Quarterly fundamentals
        qf = compute_quarterly_short_features(sym)
        if qf is not None:
            daily_qf = map_quarterly_to_daily(qf, date_index)
            if daily_qf is not None:
                quarterly_features[sym] = daily_qf
                n_q += 1

        # Earnings surprise features
        ef = compute_earnings_short_features(sym)
        if ef is not None:
            daily_ef = map_earnings_to_daily(ef, date_index)
            if daily_ef is not None:
                earnings_features[sym] = daily_ef
                n_e += 1

        # Insider selling features
        ins_df = compute_insider_short_features(sym)
        if ins_df is not None:
            daily_ins = map_insider_to_daily(ins_df, date_index)
            if daily_ins is not None:
                insider_features[sym] = daily_ins
                n_i += 1

    log(f"\n  Coverage: quarterly={n_q}, earnings={n_e}, insider={n_i} out of {len(symbols)} symbols")

    # ── 2. Merge into features dataframe ─────────────────────────────
    log("\n  Merging short features into feature matrix ...")

    # Collect all new columns
    q_cols = set()
    e_cols = set()
    i_cols = set()

    for df in quarterly_features.values():
        q_cols.update(df.columns)
    for df in earnings_features.values():
        e_cols.update(df.columns)
    for df in insider_features.values():
        i_cols.update(df.columns)

    all_new_cols = sorted(q_cols | e_cols | i_cols)
    log(f"  New feature columns: {len(all_new_cols)}")
    for col in all_new_cols:
        log(f"    - {col}")

    # Initialize new columns with NaN
    for col in all_new_cols:
        feat[col] = np.nan

    # Fill per-symbol
    for sym in symbols:
        sym_mask = feat["symbol"] == sym
        sym_dates = feat.loc[sym_mask, "date"].values

        if sym in quarterly_features:
            qf = quarterly_features[sym]
            for col in qf.columns:
                # Map by date
                vals = qf[col].reindex(pd.DatetimeIndex(sym_dates)).values
                feat.loc[sym_mask, col] = vals

        if sym in earnings_features:
            ef = earnings_features[sym]
            for col in ef.columns:
                vals = ef[col].reindex(pd.DatetimeIndex(sym_dates)).values
                feat.loc[sym_mask, col] = vals

        if sym in insider_features:
            ins = insider_features[sym]
            for col in ins.columns:
                vals = ins[col].reindex(pd.DatetimeIndex(sym_dates)).values
                feat.loc[sym_mask, col] = vals

    # ── 3. Add price-based short features ────────────────────────────
    log("\n  Computing price-based short features ...")
    price_feats = compute_price_short_features(feat)
    for col in price_feats.columns:
        feat[col] = price_feats[col].values

    # ── 4. Report coverage ───────────────────────────────────────────
    log("\n  Feature coverage (non-null %):")
    new_cols = all_new_cols + list(price_feats.columns)
    for col in sorted(new_cols):
        if col in feat.columns:
            pct = feat[col].notna().mean() * 100
            log(f"    {col:40s} {pct:6.1f}%")

    # ── 5. Save ──────────────────────────────────────────────────────
    log(f"\n  Saving to {OUTPUT_FILE} ...")
    feat.to_parquet(OUTPUT_FILE, index=False)

    elapsed = time.perf_counter() - t0
    log(f"\n  Final shape: {feat.shape}")
    log(f"  Runtime: {elapsed:.0f}s")
    log("  Done!")


if __name__ == "__main__":
    main()
