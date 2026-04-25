#!/usr/bin/env python3
"""
Small-Cap Short Pipeline
=========================
Builds an expanded universe beyond S&P 500 to find real short signals.

Universe: Current S&P 500 + former S&P 500 members (removed since 2010)
         = ~800 stocks including actual failures/demotions

Thesis: S&P 500 stocks rarely go negative because the index removes weak
companies. But FORMER members — the ones that got removed — DO decline.
Training on this broader universe teaches the model what decline looks like.

Steps:
  1. Build expanded universe (current + former S&P 500)
  2. Fetch price data from yfinance (15 years)
  3. Fetch fundamentals from FMP cache + API
  4. Compute features (reuse existing + short-specific)
  5. Walk-forward train short model
  6. Validate: do short candidates actually decline?

Output: data/smallcap/features_smallcap.parquet
        data/smallcap/predictions_short_YYYY.parquet
        data/longshort/PHASE2_SMALLCAP_SHORT.md
"""

import json
import os
import ssl
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from scipy import stats
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

DATA_DIR = Path(__file__).resolve().parent / "data"
SC_DIR = DATA_DIR / "smallcap"
CACHE_DIR = DATA_DIR / "fundamentals_cache"
OUT_DIR = DATA_DIR / "longshort"
YEARS = list(range(2015, 2026))

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def log(msg: str):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════════
# 1. BUILD EXPANDED UNIVERSE
# ═══════════════════════════════════════════════════════════════════

def _fmp_fetch(endpoint, params=""):
    url = (f"https://financialmodelingprep.com/stable/{endpoint}"
           f"?{params}&apikey={FMP_API_KEY}")
    try:
        req = __import__("urllib.request", fromlist=["Request"]).Request(
            url, headers={"User-Agent": "auto-trader/1.0"})
        with __import__("urllib.request", fromlist=["urlopen"]).urlopen(
                req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def build_universe():
    """Build expanded universe: current S&P 500 + former members."""
    log("  Building expanded universe ...")

    # Current S&P 500
    current = _fmp_fetch("sp500-constituent")
    current_syms = {s["symbol"] for s in current}
    log(f"    Current S&P 500: {len(current_syms)}")

    # Historical changes — get removed tickers
    history = _fmp_fetch("historical-sp500-constituent")
    hist_df = pd.DataFrame(history)
    hist_df["dateAdded"] = pd.to_datetime(hist_df["dateAdded"])

    removed = hist_df[hist_df["removedTicker"].notna() &
                       (hist_df["dateAdded"] >= "2010-01-01")]
    former_syms = set(removed["removedTicker"].unique()) - current_syms - {""}

    # Filter out tickers that are clearly dead (acquired/merged — ticker changed)
    # We'll keep them and let yfinance tell us if they have data
    log(f"    Former S&P 500 (removed since 2010): {len(former_syms)}")

    # Build removal date lookup: when was each stock removed?
    removal_dates = {}
    for _, row in removed.iterrows():
        sym = row["removedTicker"]
        if sym and sym not in removal_dates:
            removal_dates[sym] = row["dateAdded"]

    all_syms = sorted(current_syms | former_syms)
    # Remove ETFs
    from sp500_universe import get_etf_symbols
    etf_set = set(get_etf_symbols())
    all_syms = [s for s in all_syms if s not in etf_set and s]
    log(f"    Total universe: {len(all_syms)} stocks")

    return all_syms, current_syms, former_syms, removal_dates


# ═══════════════════════════════════════════════════════════════════
# 2. FETCH PRICE DATA
# ═══════════════════════════════════════════════════════════════════

def fetch_price_data(symbols):
    """Batch download 15 years of price data from yfinance."""
    log("\n  Fetching price data from yfinance ...")
    START_DATE = "2010-01-01"
    END_DATE = "2026-04-18"
    CHUNK_SIZE = 100

    raw = {}
    n_chunks = (len(symbols) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for ci in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[ci:ci + CHUNK_SIZE]
        chunk_num = ci // CHUNK_SIZE + 1
        log(f"    Chunk {chunk_num}/{n_chunks} ({len(chunk)} symbols) ...")
        try:
            batch = yf.download(
                chunk, start=START_DATE, end=END_DATE,
                auto_adjust=True, progress=False, threads=True,
            )
            for sym in chunk:
                try:
                    if isinstance(batch.columns, pd.MultiIndex):
                        sym_df = batch.xs(sym, axis=1, level=1).copy()
                    else:
                        sym_df = batch.copy()
                    sym_df.columns = sym_df.columns.str.lower()
                    sym_df = sym_df[["open", "high", "low", "close", "volume"]].dropna(how="all")
                    sym_df.index = pd.to_datetime(sym_df.index).tz_localize(None)
                    sym_df.index.name = "date"
                    if len(sym_df) > 100:  # need at least 100 days
                        raw[sym] = sym_df.sort_index()
                except (KeyError, Exception):
                    pass
        except Exception as exc:
            log(f"      FAILED: {exc}")

        time.sleep(0.5)

    log(f"    Price data: {len(raw)}/{len(symbols)} symbols have data")
    return raw


# ═══════════════════════════════════════════════════════════════════
# 3. FETCH FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════

def fetch_fundamentals(symbols):
    """Fetch fundamentals from FMP cache + API for expanded universe."""
    log("\n  Loading fundamentals ...")

    endpoints = [
        ("income-statement", "period=quarter&limit=60"),
        ("cash-flow-statement", "period=quarter&limit=60"),
        ("balance-sheet-statement", "period=quarter&limit=60"),
        ("earnings", ""),
        ("insider-trading/search", "page=0&limit=100"),
    ]

    fund_data = {}
    cached = 0
    fetched = 0
    missing = 0

    for i, sym in enumerate(symbols):
        sym_data = {}
        for endpoint, params in endpoints:
            safe = endpoint.replace("/", "_")
            cache_file = CACHE_DIR / f"{sym}_{safe}.json"

            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        sym_data[endpoint] = json.load(f)
                    cached += 1
                    continue
                except Exception:
                    pass

            # Fetch from API
            data = _fmp_fetch(endpoint, f"symbol={sym}&{params}")
            sym_data[endpoint] = data

            # Cache it
            try:
                with open(cache_file, "w") as f:
                    json.dump(data, f)
                fetched += 1
            except Exception:
                pass

            time.sleep(0.15)  # rate limit

        fund_data[sym] = sym_data

        if (i + 1) % 100 == 0 or (i + 1) == len(symbols):
            log(f"    [{i+1}/{len(symbols)}] cached={cached}, fetched={fetched}")

    log(f"    Total: {cached} cached, {fetched} fetched from API")
    return fund_data


# ═══════════════════════════════════════════════════════════════════
# 4. COMPUTE FEATURES
# ═══════════════════════════════════════════════════════════════════

def compute_technical_features(df):
    """Compute technical features for one symbol's price data."""
    c = df["close"]
    v = df["volume"]
    feat = pd.DataFrame(index=df.index)

    # Returns
    for n in [5, 10, 20, 60, 120]:
        feat[f"ret_{n}d"] = c.pct_change(n)

    # Volatility
    daily_ret = c.pct_change()
    for n in [10, 20, 60]:
        feat[f"vol_{n}d"] = daily_ret.rolling(n).std() * np.sqrt(252)

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = 100 - 100 / (1 + rs)

    # Bollinger Band position
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feat["bb_position"] = (c - sma20) / std20.replace(0, np.nan)

    # SMA distances
    for n in [50, 200]:
        sma = c.rolling(n).mean()
        feat[f"dist_sma{n}"] = (c - sma) / sma.replace(0, np.nan)

    # 52-week high/low
    high_252 = df["high"].rolling(252, min_periods=252).max()
    low_252 = df["low"].rolling(252, min_periods=252).min()
    feat["dist_52w_high"] = (c - high_252) / high_252.replace(0, np.nan)
    feat["dist_52w_low"] = (c - low_252) / low_252.replace(0, np.nan)

    # SMA200 slope
    sma200 = c.rolling(200).mean()
    sma200_30ago = sma200.shift(30)
    feat["sma200_slope"] = (sma200 - sma200_30ago) / sma200_30ago.replace(0, np.nan)

    # Volume ratio
    avg_vol = v.rolling(20).mean()
    feat["vol_ratio_20d"] = v / avg_vol.replace(0, np.nan)

    # Max drawdown 6m
    def _max_dd(prices, window=126):
        result = pd.Series(np.nan, index=prices.index)
        arr = prices.values
        for i in range(window, len(arr)):
            seg = arr[i - window:i + 1]
            valid = seg[~np.isnan(seg)]
            if len(valid) < 10:
                continue
            peak = np.maximum.accumulate(valid)
            dd = (valid - peak) / np.where(peak > 0, peak, np.nan)
            result.iloc[i] = np.nanmin(dd)
        return result

    feat["max_dd_6m"] = _max_dd(c, 126)

    # Consecutive down months
    monthly_ret = c.pct_change(21)
    streak = pd.Series(0, index=c.index, dtype=float)
    s = 0
    for i in range(len(monthly_ret)):
        r = monthly_ret.iloc[i]
        if np.isnan(r):
            s = 0
        elif r < 0:
            s += 1
        else:
            s = 0
        streak.iloc[i] = s
    feat["consec_down_months"] = streak

    # Short-specific price signals
    feat["below_sma200"] = (feat["dist_sma200"] < 0).astype(float)
    feat["below_sma50"] = (feat["dist_sma50"] < 0).astype(float)
    feat["death_cross"] = (feat["dist_sma50"] < feat["dist_sma200"]).astype(float)
    feat["near_52w_low"] = (feat["dist_52w_high"] < -0.30).astype(float)
    feat["rsi_weak"] = (feat["rsi_14"] < 40).astype(float)

    feat["momentum_collapse"] = np.where(
        (feat["ret_60d"] < -0.05) & (feat["ret_20d"] < -0.02), 1.0, 0.0)

    feat["vol_expansion"] = np.where(
        feat["vol_60d"] > 0, feat["vol_20d"] / feat["vol_60d"], np.nan)

    return feat


def compute_fundamental_features_from_cache(sym_data, date_index):
    """Compute fundamental features from FMP cache for one symbol."""
    result = pd.DataFrame(index=date_index)
    n_days = len(date_index)
    date_vals = date_index.values

    # Income statement
    inc_raw = sym_data.get("income-statement", [])
    if not inc_raw:
        return None

    inc = pd.DataFrame(inc_raw)
    if "filingDate" not in inc.columns or "revenue" not in inc.columns:
        return None

    inc["filing_date"] = pd.to_datetime(inc["filingDate"])
    inc["date"] = pd.to_datetime(inc["date"])
    inc = inc.sort_values("date")

    if len(inc) < 2:
        return None

    filing_dates = inc["filing_date"].values
    n_q = len(inc)

    # Revenue/earnings
    rev = inc["revenue"].values.astype(float)
    ni = inc.get("netIncome", pd.Series(dtype=float)).values.astype(float)
    gp = inc.get("grossProfit", pd.Series(dtype=float)).values.astype(float)
    oi = inc.get("operatingIncome", pd.Series(dtype=float)).values.astype(float)
    sga = inc.get("sellingGeneralAndAdministrativeExpenses",
                   pd.Series(dtype=float)).values.astype(float)

    # Margins
    gm = np.where(np.abs(rev) > 1e4, gp / rev, np.nan)
    om = np.where(np.abs(rev) > 1e4, oi / rev, np.nan)
    nm = np.where(np.abs(rev) > 1e4, ni / rev, np.nan)

    # Margin QoQ changes
    gm_chg = np.concatenate([[np.nan], np.diff(gm)])
    om_chg = np.concatenate([[np.nan], np.diff(om)])
    nm_chg = np.concatenate([[np.nan], np.diff(nm)])

    # Revenue YoY growth
    rev_yoy = np.full(n_q, np.nan)
    for idx in range(4, n_q):
        prev_rev = rev[idx - 4]
        if abs(prev_rev) > 1e4:
            rev_yoy[idx] = (rev[idx] - prev_rev) / abs(prev_rev)

    # SGA burden
    sga_ratio = np.where(np.abs(rev) > 1e4, sga / rev, np.nan)
    sga_chg = np.concatenate([[np.nan], np.diff(sga_ratio)])

    # Margin trend (slope over 4 quarters)
    margin_trend = np.full(n_q, np.nan)
    for idx in range(3, n_q):
        window = nm[idx-3:idx+1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 3:
            margin_trend[idx] = np.polyfit(range(len(valid)), valid, 1)[0]

    # Cash flow
    cf_raw = sym_data.get("cash-flow-statement", [])
    ocf_arr = np.full(n_q, np.nan)
    fcf_arr = np.full(n_q, np.nan)
    capex_arr = np.full(n_q, np.nan)
    buyback_arr = np.full(n_q, np.nan)
    ndi_arr = np.full(n_q, np.nan)

    if cf_raw:
        cf = pd.DataFrame(cf_raw)
        cf["date"] = pd.to_datetime(cf["date"])
        # Merge by date
        cf_lookup = dict(zip(cf["date"], range(len(cf))))
        for qi in range(n_q):
            q_date = inc.iloc[qi]["date"]
            ci = cf_lookup.get(q_date)
            if ci is not None:
                row = cf.iloc[ci]
                ocf_arr[qi] = float(row.get("operatingCashFlow", np.nan) or np.nan)
                fcf_arr[qi] = float(row.get("freeCashFlow", np.nan) or np.nan)
                capex_arr[qi] = abs(float(row.get("capitalExpenditure", np.nan) or np.nan))
                buyback_arr[qi] = float(row.get("commonStockRepurchased", np.nan) or np.nan)
                ndi_arr[qi] = float(row.get("netDebtIssuance", np.nan) or np.nan)

    # Accruals
    accruals = ni - ocf_arr

    # Balance sheet
    bs_raw = sym_data.get("balance-sheet-statement", [])
    ta_arr = np.full(n_q, np.nan)
    td_arr = np.full(n_q, np.nan)
    eq_arr = np.full(n_q, np.nan)
    nd_arr = np.full(n_q, np.nan)
    ebitda_arr = np.full(n_q, np.nan)

    if bs_raw:
        bs = pd.DataFrame(bs_raw)
        bs["date"] = pd.to_datetime(bs["date"])
        bs_lookup = dict(zip(bs["date"], range(len(bs))))
        for qi in range(n_q):
            q_date = inc.iloc[qi]["date"]
            bi = bs_lookup.get(q_date)
            if bi is not None:
                row = bs.iloc[bi]
                ta_arr[qi] = float(row.get("totalAssets", np.nan) or np.nan)
                td_arr[qi] = float(row.get("totalDebt", np.nan) or np.nan)
                eq_arr[qi] = float(row.get("totalStockholdersEquity", np.nan) or np.nan)
                nd_arr[qi] = float(row.get("netDebt", np.nan) or np.nan)

    # EBITDA from income
    ebitda_arr = inc.get("ebitda", pd.Series(dtype=float)).values.astype(float)
    int_exp = inc.get("interestExpense", pd.Series(dtype=float)).values.astype(float)

    # Derived quarterly features
    accruals_ratio = np.where(np.abs(ta_arr) > 1e6, accruals / ta_arr, np.nan)
    fcf_to_ni = np.where(np.abs(ni) > 1e4, fcf_arr / ni, np.nan)
    ocf_to_ni = np.where(np.abs(ni) > 1e4, ocf_arr / ni, np.nan)
    leverage = np.where(np.abs(ta_arr) > 1e6, td_arr / ta_arr, np.nan)
    lev_chg = np.concatenate([[np.nan], np.diff(leverage)])
    dte = np.where(np.abs(eq_arr) > 1e4, td_arr / np.abs(eq_arr), np.nan)
    ic = np.where(np.abs(int_exp) > 1e3, ebitda_arr / np.abs(int_exp), np.nan)
    ic_chg = np.concatenate([[np.nan], np.diff(ic)])
    capex_rev = np.where(np.abs(rev) > 1e4, capex_arr / rev, np.nan)
    nd_ebitda = np.where(np.abs(ebitda_arr) > 1e4, nd_arr / ebitda_arr, np.nan)

    td_prev = np.concatenate([[np.nan], td_arr[:-1]])
    debt_chg = np.where(np.abs(td_prev) > 1e4, (td_arr - td_prev) / np.abs(td_prev), np.nan)

    # Map all quarterly features to daily using filing_date
    quarterly_features = {
        "revenue_growth_yoy": rev_yoy,
        "gross_margin": gm, "operating_margin": om, "net_margin": nm,
        "gross_margin_qoq_chg": gm_chg, "operating_margin_qoq_chg": om_chg,
        "net_margin_qoq_chg": nm_chg, "margin_trend_4q": margin_trend,
        "sga_to_revenue": sga_ratio, "sga_burden_change": sga_chg,
        "accruals_ratio": accruals_ratio,
        "fcf_to_ni": fcf_to_ni, "ocf_to_ni": ocf_to_ni,
        "leverage_ratio": leverage, "leverage_change_qoq": lev_chg,
        "debt_to_equity": dte, "debt_growth_qoq": debt_chg,
        "interest_coverage": ic, "interest_coverage_change": ic_chg,
        "capex_to_revenue": capex_rev, "net_debt_to_ebitda": nd_ebitda,
    }

    for feat_name, arr in quarterly_features.items():
        daily = np.full(n_days, np.nan)
        for d in range(n_days):
            q_idx = np.searchsorted(filing_dates, date_vals[d], side="right") - 1
            if 0 <= q_idx < n_q:
                daily[d] = arr[q_idx]
        result[feat_name] = daily

    # Earnings surprise features
    earn_raw = sym_data.get("earnings", [])
    if earn_raw:
        earn = pd.DataFrame(earn_raw)
        if "date" in earn.columns and "epsActual" in earn.columns:
            earn["date"] = pd.to_datetime(earn["date"])
            earn = earn.sort_values("date").dropna(subset=["epsActual"])

            if len(earn) > 0:
                actual = earn["epsActual"].values.astype(float)
                est = earn["epsEstimated"].values.astype(float)
                earn_dates = earn["date"].values

                surprise = np.where(np.abs(est) > 1e-4,
                                     (actual - est) / np.abs(est), np.nan)

                # Miss streak
                miss_streak = np.zeros(len(surprise))
                s = 0
                for i in range(len(surprise)):
                    if np.isnan(surprise[i]):
                        s = 0
                    elif surprise[i] < 0:
                        s += 1
                    else:
                        s = 0
                    miss_streak[i] = s

                # Map to daily
                eps_surp = np.full(n_days, np.nan)
                eps_miss = np.full(n_days, np.nan)
                for d in range(n_days):
                    e_idx = np.searchsorted(earn_dates, date_vals[d], side="right") - 1
                    if 0 <= e_idx < len(earn_dates):
                        eps_surp[d] = np.clip(surprise[e_idx], -2, 2)
                        eps_miss[d] = miss_streak[e_idx]

                result["eps_surprise_last"] = eps_surp
                result["eps_miss_streak"] = eps_miss

    return result


# ═══════════════════════════════════════════════════════════════════
# 5. BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════

def build_feature_matrix(raw_prices, fund_data, current_sp500, former_sp500):
    """Build complete feature matrix for expanded universe."""
    log("\n  Building feature matrix ...")

    # Use SPY as date calendar
    spy_close = raw_prices.get("SPY")
    if spy_close is None:
        log("    ERROR: SPY not in price data, using first symbol")
        first_sym = list(raw_prices.keys())[0]
        spy_close = raw_prices[first_sym]
    date_index = spy_close.index

    # SPY features for cross-asset context
    spy_tech = compute_technical_features(spy_close)

    all_rows = []
    syms_processed = 0

    for sym, price_df in raw_prices.items():
        if sym in ("SPY", "VIXY", "TLT"):  # skip cross-asset ETFs
            continue

        # Technical features
        try:
            tech = compute_technical_features(price_df)
        except Exception:
            continue

        # Fundamental features
        sym_fund = fund_data.get(sym)
        fund_feat = None
        if sym_fund:
            try:
                fund_feat = compute_fundamental_features_from_cache(
                    sym_fund, price_df.index)
            except Exception:
                pass

        # Build row DataFrame
        feat = tech.copy()
        feat["symbol"] = sym
        feat["date"] = feat.index

        # Add fundamentals
        if fund_feat is not None:
            for col in fund_feat.columns:
                feat[col] = fund_feat[col].values

        # Market cap bucket (use as feature)
        feat["is_former_sp500"] = 1.0 if sym in former_sp500 else 0.0
        feat["in_sp500"] = 1.0 if sym in current_sp500 else 0.0

        # Add SPY context
        spy_aligned = spy_tech.reindex(feat.index)
        feat["spy_ret_10d"] = spy_aligned.get("ret_10d")
        feat["spy_ret_20d"] = spy_aligned.get("ret_20d")

        # Forward return (target)
        feat["fwd_10d_ret"] = feat["ret_10d"].shift(-10)

        all_rows.append(feat)
        syms_processed += 1

        if syms_processed % 100 == 0:
            log(f"    [{syms_processed}] symbols processed ...")

    log(f"    Total: {syms_processed} symbols")

    master = pd.concat(all_rows, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])

    # Target: bottom 10% by forward 10-day return (among all stocks with data that day)
    has_fwd = master["fwd_10d_ret"].notna()
    valid = master[has_fwd].copy()
    valid["pct_rank"] = valid.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid["target_short"] = (valid["pct_rank"] <= 0.10).astype(int)
    master.loc[valid.index, "target_short"] = valid["target_short"]
    master.loc[valid.index, "pct_rank"] = valid["pct_rank"]
    master["fwd_ret"] = master["fwd_10d_ret"]

    # Cross-sectional ranks
    master["accruals_rank"] = master.groupby("date")["accruals_ratio"].rank(pct=True)
    master["leverage_rank"] = master.groupby("date")["leverage_ratio"].rank(pct=True)
    master["vol_rank_20d"] = master.groupby("date")["vol_20d"].rank(pct=True)
    master["momentum_rank_60d"] = master.groupby("date")["ret_60d"].rank(pct=True)

    log(f"    Feature matrix: {master.shape}")
    log(f"    Date range: {master['date'].min()} to {master['date'].max()}")
    log(f"    target_short pos rate: {master['target_short'].mean():.1%}")

    return master


# ═══════════════════════════════════════════════════════════════════
# 6. WALK-FORWARD SHORT MODEL
# ═══════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Technical
    "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "vol_10d", "vol_20d", "vol_60d",
    "rsi_14", "bb_position",
    "dist_sma50", "dist_sma200", "sma200_slope",
    "dist_52w_high", "dist_52w_low",
    "max_dd_6m", "consec_down_months",
    "vol_ratio_20d",
    "below_sma200", "below_sma50", "death_cross", "near_52w_low",
    "rsi_weak", "momentum_collapse", "vol_expansion",
    # Fundamentals
    "revenue_growth_yoy",
    "gross_margin", "operating_margin", "net_margin",
    "gross_margin_qoq_chg", "operating_margin_qoq_chg", "net_margin_qoq_chg",
    "margin_trend_4q", "sga_to_revenue", "sga_burden_change",
    "accruals_ratio", "fcf_to_ni", "ocf_to_ni",
    "leverage_ratio", "leverage_change_qoq",
    "debt_to_equity", "debt_growth_qoq",
    "interest_coverage", "interest_coverage_change",
    "capex_to_revenue", "net_debt_to_ebitda",
    "eps_surprise_last", "eps_miss_streak",
    # Cross-sectional ranks
    "accruals_rank", "leverage_rank", "vol_rank_20d", "momentum_rank_60d",
    # Context
    "spy_ret_10d", "spy_ret_20d",
    "is_former_sp500",
]

# Fundamental cols that may be NaN
FUND_COLS = [
    "revenue_growth_yoy", "gross_margin", "operating_margin", "net_margin",
    "gross_margin_qoq_chg", "operating_margin_qoq_chg", "net_margin_qoq_chg",
    "margin_trend_4q", "sga_to_revenue", "sga_burden_change",
    "accruals_ratio", "fcf_to_ni", "ocf_to_ni",
    "leverage_ratio", "leverage_change_qoq",
    "debt_to_equity", "debt_growth_qoq",
    "interest_coverage", "interest_coverage_change",
    "capex_to_revenue", "net_debt_to_ebitda",
    "eps_surprise_last", "eps_miss_streak",
    "accruals_rank", "leverage_rank",
]

LGB_PARAMS = dict(
    n_estimators=800, learning_rate=0.03, max_depth=5, num_leaves=24,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=1.0, objective="binary", metric="auc",
    random_state=42, n_jobs=-1, verbose=-1,
)

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    min_child_weight=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=1.0, tree_method="hist",
    n_jobs=1, random_state=42, eval_metric="auc", use_label_encoder=False,
)


def train_and_validate(df):
    """Walk-forward train + validate short model on expanded universe."""
    log("\n  Walk-forward training ...")

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    non_fund = [c for c in feature_cols if c not in FUND_COLS]
    log(f"    Features: {len(feature_cols)} ({len(non_fund)} required non-null)")

    # Drop rows missing technical features or target
    df = df.dropna(subset=non_fund + ["target_short"]).copy()
    log(f"    Rows after dropping NaN: {len(df):,}")

    all_preds = {}
    auc_by_year = {}

    for year in YEARS:
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        calib_start = pd.Timestamp(f"{year - 2}-01-01")
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")

        train_all = df[df["date"] <= train_end]
        calib_mask = train_all["date"] >= calib_start
        pure_train_mask = train_all["date"] < calib_start

        year_mask = (df["date"] >= year_start) & (df["date"] <= year_end)
        df_year = df[year_mask].copy()

        if pure_train_mask.sum() < 1000 or calib_mask.sum() < 100 or len(df_year) < 50:
            log(f"    {year}: SKIP (insufficient data)")
            continue

        X_train = train_all.loc[pure_train_mask, feature_cols].values
        y_train = train_all.loc[pure_train_mask, "target_short"].values
        X_calib = train_all.loc[calib_mask, feature_cols].values
        y_calib = train_all.loc[calib_mask, "target_short"].values

        scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

        # LGBM
        model_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_calib, y_calib)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

        calib_lgb = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
        calib_lgb.fit(X_calib, y_calib)
        lgbm_auc = roc_auc_score(y_calib, calib_lgb.predict_proba(X_calib)[:, 1])

        # XGBoost
        try:
            model_xgb = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale)
            model_xgb.fit(X_train, y_train)
            calib_xgb = CalibratedClassifierCV(model_xgb, method="isotonic", cv="prefit")
            calib_xgb.fit(X_calib, y_calib)
            xgb_auc = roc_auc_score(y_calib, calib_xgb.predict_proba(X_calib)[:, 1])
            xgb_ok = True
        except Exception as e:
            log(f"    XGB failed: {e}")
            xgb_ok = False
            xgb_auc = float("nan")

        # Predict year
        X_year = df_year[feature_cols].values
        lgbm_probs = calib_lgb.predict_proba(X_year)[:, 1]
        if xgb_ok:
            xgb_probs = calib_xgb.predict_proba(X_year)[:, 1]
            ensemble = 0.5 * lgbm_probs + 0.5 * xgb_probs
        else:
            ensemble = lgbm_probs

        df_year = df_year.copy()
        df_year["prob_short"] = ensemble

        log(f"    {year}: LGBM AUC={lgbm_auc:.4f}, XGB AUC={xgb_auc:.4f}, "
            f"n={len(df_year):,}, prob range=[{ensemble.min():.3f}, {ensemble.max():.3f}]")

        preds = df_year[["date", "symbol", "target_short", "prob_short",
                          "fwd_ret", "in_sp500", "is_former_sp500"]].copy()

        # Save
        pred_file = SC_DIR / f"predictions_short_{year}.parquet"
        preds.to_parquet(pred_file, index=False)

        all_preds[year] = preds
        auc_by_year[year] = {"lgbm_auc": lgbm_auc, "xgb_auc": xgb_auc}

    return all_preds, auc_by_year


# ═══════════════════════════════════════════════════════════════════
# 7. SIGNAL VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_signal(all_preds):
    """Check short signal: all stocks, SP500 only, and former-SP500 only."""
    log("\n  Validating short signal ...")

    universes = {
        "all_stocks": lambda df: df,
        "sp500_only": lambda df: df[df["in_sp500"] == 1.0],
        "former_sp500": lambda df: df[df["is_former_sp500"] == 1.0],
        "non_sp500": lambda df: df[df["in_sp500"] != 1.0],
    }

    results = {}
    for univ_name, filter_fn in universes.items():
        yearly_results = []
        all_rets = []
        all_spreads = []

        for year in YEARS:
            df = all_preds.get(year)
            if df is None:
                continue

            df_filt = filter_fn(df).dropna(subset=["fwd_ret", "prob_short"])
            if len(df_filt) == 0:
                continue

            day_rets = []
            day_spreads = []
            for date in sorted(df_filt["date"].unique()):
                day = df_filt[df_filt["date"] == date].sort_values(
                    "prob_short", ascending=False)
                if len(day) < 10:
                    continue

                top5 = day.head(5)
                bot5 = day.tail(5)
                rand5 = day.sample(min(5, len(day)), random_state=42 + year)

                sc_ret = top5["fwd_ret"].mean()
                ns_ret = bot5["fwd_ret"].mean()
                day_rets.append(sc_ret)
                day_spreads.append(ns_ret - sc_ret)

            if day_rets:
                yearly_results.append({
                    "year": year,
                    "sc_mean": np.mean(day_rets),
                    "spread_mean": np.mean(day_spreads),
                    "n_days": len(day_rets),
                })
                all_rets.extend(day_rets)
                all_spreads.extend(day_spreads)

        if not all_rets:
            results[univ_name] = None
            continue

        t_ret, p_ret = stats.ttest_1samp(all_rets, 0)
        t_spr, p_spr = stats.ttest_1samp(all_spreads, 0)
        neg_years = sum(1 for r in yearly_results if r["sc_mean"] < 0)
        avg_ret = np.mean([r["sc_mean"] for r in yearly_results])
        avg_spread = np.mean([r["spread_mean"] for r in yearly_results])

        results[univ_name] = {
            "avg_ret": avg_ret,
            "avg_spread": avg_spread,
            "t_ret": t_ret, "p_ret": p_ret,
            "t_spr": t_spr, "p_spr": p_spr,
            "neg_years": neg_years,
            "total_years": len(yearly_results),
            "yearly": yearly_results,
        }

        log(f"    {univ_name:20s}: sc_avg={avg_ret:+.3%}, spread={avg_spread:+.3%}, "
            f"neg_years={neg_years}/{len(yearly_results)}, "
            f"t={t_ret:.2f}, p={p_ret:.4f}")

    return results


def generate_report(results, auc_by_year):
    lines = []
    lines.append("# Phase 2: Small-Cap Short Model — Expanded Universe")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Model Design\n")
    lines.append("- **Universe**: Current S&P 500 + former S&P 500 members (removed since 2010)")
    lines.append("- **Target**: bottom 10% by forward 10-day return (across full universe)")
    lines.append("- **Key insight**: Former S&P 500 members include actual failures/demotions")
    lines.append("  — these provide real negative return training examples")
    lines.append("- **ETFs excluded**: No VIXY, USO, PSKY contamination")

    # AUC
    lines.append("\n## Model Quality (AUC per year)\n")
    lines.append("| Year | LGBM AUC | XGB AUC |")
    lines.append("|------|----------|---------|")
    for year in YEARS:
        a = auc_by_year.get(year, {})
        lines.append(f"| {year} | {a.get('lgbm_auc', 0):.4f} | {a.get('xgb_auc', 0):.4f} |")

    # Results by universe
    for univ_name, r in results.items():
        if r is None:
            continue

        lines.append(f"\n## {univ_name.replace('_', ' ').title()}\n")
        lines.append(f"| Year | Days | Short Cands Fwd | Spread | "
                     f"SC<0 p |")
        lines.append(f"|------|------|-----------------|--------|--------|")

        for yr in r["yearly"]:
            sig = "***" if abs(yr["sc_mean"]) > 0 else ""
            lines.append(f"| {yr['year']} | {yr['n_days']} | "
                         f"{yr['sc_mean']:+.3%} | {yr['spread_mean']:+.3%} | |")

        lines.append(f"\n- **Short candidates avg**: {r['avg_ret']:+.3%}")
        lines.append(f"- **Spread avg**: {r['avg_spread']:+.3%}")
        lines.append(f"- **t-stat (SC < 0)**: {r['t_ret']:.2f}, p={r['p_ret']:.6f}")
        lines.append(f"- **Negative years**: {r['neg_years']}/{r['total_years']}")

    # Decision
    lines.append("\n## Decision\n")

    best = None
    for name, r in results.items():
        if r and (best is None or r["avg_ret"] < results.get(best, {}).get("avg_ret", 999)):
            best = name

    if best and results[best]["avg_ret"] < -0.003:
        lines.append(f"**STRONG SHORT SIGNAL in `{best}`** "
                     f"(avg = {results[best]['avg_ret']:+.3%})")
        lines.append(f"\n→ Proceed to build B4 long/short using `{best}` universe")
    elif best and results[best]["avg_ret"] < 0:
        lines.append(f"**WEAK SHORT SIGNAL in `{best}`** "
                     f"(avg = {results[best]['avg_ret']:+.3%})")
    else:
        lines.append(f"**NO SHORT SIGNAL** even with expanded universe")
        if best:
            lines.append(f"Best: `{best}` = {results[best]['avg_ret']:+.3%}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  SMALL-CAP SHORT PIPELINE — EXPANDED UNIVERSE")
    log("=" * 70)

    SC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build universe
    all_syms, current_sp500, former_sp500, removal_dates = build_universe()

    # 2. Fetch prices — we need SPY too
    from sp500_universe import get_cross_asset
    price_syms = all_syms + ["SPY"]
    raw_prices = fetch_price_data(price_syms)

    # 3. Fetch fundamentals
    fund_data = fetch_fundamentals([s for s in all_syms if s in raw_prices])

    # 4. Build feature matrix
    df = build_feature_matrix(raw_prices, fund_data, current_sp500, former_sp500)

    # Save features
    feat_file = SC_DIR / "features_smallcap.parquet"
    df.to_parquet(feat_file, index=False)
    log(f"\n  Saved features: {feat_file}")

    # 5. Train & validate
    all_preds, auc_by_year = train_and_validate(df)

    # 6. Validate signal
    results = validate_signal(all_preds)

    # 7. Report
    report = generate_report(results, auc_by_year)
    report_file = OUT_DIR / "PHASE2_SMALLCAP_SHORT.md"
    report_file.write_text(report)

    elapsed = time.perf_counter() - t0
    log(f"\n  Report: {report_file}")
    log(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
