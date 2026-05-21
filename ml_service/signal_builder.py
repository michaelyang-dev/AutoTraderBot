"""
V9.6 Signal Adapter
===================
Bridges the v9.6 multi-strategy factor framework with signal_server.
Outputs signals in the same JSON format the trading bot expects.

This module is imported by signal_server.py.
"""

import json as _json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sp500_history import get_sp500_on_date
from sp1500_membership import get_sp1500_on_date
from strategies.multi_strategy_engine import (
    FastUniverse, strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, strategy4_index_inclusion,
    SECTOR_ETFS,
)

# v11 strategy config: 35% momentum, 25% value, 40% low-vol quality
# Improvement over v10.2: +30.9% CAGR (was +22.6%), 1.20 Sharpe (was 1.01)
# Max DD: -32.6% (was -38.5%), Calmar: 0.95 (was 0.59)
# Wins 8/8 individual years, 5/6 walk-forward windows
# Bootstrap 95% CI: [+15.2%, +50.4%], conservative CAGR: +20.8%
# Changes: rebal 10d→15d, RP OFF, weights 40/20/40→35/25/40, cap→25%
# With 1.5x leverage target
STRATEGY_CONFIG_BULL = [
    ("s1_momentum", 0.35),
    ("s7_value",    0.25),
    ("s3_sector",   0.00),
    ("s4_inclusion", 0.00),
    ("s5_lowvol",   0.40),
]

STRATEGY_CONFIG_BEAR = [
    ("s1_momentum", 0.10),
    ("s7_value",    0.20),
    ("s3_sector",   0.10),
    ("s4_inclusion", 0.00),
    ("s5_lowvol",   0.60),
]

log = logging.getLogger("signal_server")

# Module-level cache for the FastUniverse (expensive to build)
_uni_cache = None
_uni_cache_date = None


def _load_si_change_data(uni):
    """Load short interest change data and attach to universe for momentum scoring.
    SI change (shorts covering) adds +3.5% OOS CAGR — structural mechanical edge.

    Uses WRDS Compustat short interest (22K+ tickers, full SP1500 coverage).
    Re-upload from WRDS quarterly. Data stays valid for months because the signal
    uses cross-sectional RANKING which changes slowly.
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


def _build_universe(raw, enhanced_data=None):
    """Build FastUniverse from raw bar data."""
    DATA_DIR = Path(__file__).resolve().parent / "data"

    # Build prices DataFrame
    close_frames = {}
    for sym, df in raw.items():
        if len(df) > 0 and "close" in df.columns:
            close_frames[sym] = df["close"]
    prices = pd.DataFrame(close_frames)
    prices.index = pd.to_datetime(prices.index)

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


def _compute_features_from_raw(raw, prices):
    """Compute the feature columns that v9.6 strategies need from raw bars."""
    date_index = prices.index
    all_features = []

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

        feat = pd.DataFrame(index=date_index)
        feat["symbol"] = sym
        feat["date"] = date_index

        # Returns
        for n, name in [(5, "ret_5d"), (10, "ret_10d"), (20, "ret_20d"),
                         (60, "ret_60d"), (120, "ret_120d"), (126, "ret_126d"), (252, "ret_252d")]:
            feat[name] = c.pct_change(n)

        # Volatility
        daily_r = c.pct_change()
        for n, name in [(10, "vol_10d"), (20, "vol_20d"), (60, "vol_60d")]:
            feat[name] = daily_r.rolling(n).std()

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
            latest = fund.sort_values(date_col).groupby("tic").last()

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

            # roe = niq / ceqq (annualized: *4 for quarterly)
            if "niq" in latest.columns and "ceqq" in latest.columns:
                roe = (latest["niq"] * 4) / latest["ceqq"].replace(0, np.nan)
                roe_map = roe.to_dict()
                mask = features["symbol"].isin(roe_map)
                features.loc[mask, "roe"] = features.loc[mask, "symbol"].map(roe_map)

            # roa = niq / atq (annualized)
            if "niq" in latest.columns and "atq" in latest.columns:
                roa = (latest["niq"] * 4) / latest["atq"].replace(0, np.nan)
                roa_map = roa.to_dict()
                mask = features["symbol"].isin(roa_map)
                features.loc[mask, "roa"] = features.loc[mask, "symbol"].map(roa_map)

            # debt_to_equity = (dlttq + dlcq) / ceqq
            if "dlttq" in latest.columns and "ceqq" in latest.columns:
                dlc = latest.get("dlcq", 0)
                if isinstance(dlc, (int, float)):
                    dlc = pd.Series(dlc, index=latest.index)
                de = (latest["dlttq"].fillna(0) + dlc.fillna(0)) / latest["ceqq"].replace(0, np.nan)
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

    # ── PRIMARY: WRDS IBES for earnings surprise ──
    if wrds_ibes.exists():
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


def _strategy_value(uni, date, members, top_n=10):
    """Value strategy: high-quality stocks with strong fundamentals.

    Matches fast_backtest._strategy_value exactly:
    - Filters: ROE > 5%, gross margin > 15%, dist_sma200 > -15%, D/E < 3
    - Scores: -ret_252d * 0.30 + gross_margin * 0.25 + min(ROE, 0.5) * 0.25
    - Equal-weight top N picks
    """
    roe = uni.get_feature_map(date, "roe", members)
    gm = uni.get_feature_map(date, "gross_margin", members)
    r252 = uni.get_feature_map(date, "ret_252d", members)
    d200 = uni.get_feature_map(date, "dist_sma200", members)
    de = uni.get_feature_map(date, "debt_to_equity", members)

    scores = {}
    for sym in members:
        r = roe.get(sym)
        g = gm.get(sym)
        rv = r252.get(sym)
        dv = d200.get(sym)
        debt = de.get(sym)
        if r is None or g is None or rv is None:
            continue
        if np.isnan(r) or np.isnan(g) or np.isnan(rv):
            continue
        if r < 0.05 or g < 0.15:
            continue
        if dv is None or np.isnan(dv) or dv < -0.15:
            continue
        if debt is not None and not np.isnan(debt) and debt > 3.0:
            continue
        scores[sym] = -rv * 0.30 + g * 0.25 + min(r, 0.5) * 0.25

    if not scores:
        return {}
    ss = sorted(scores, key=scores.get, reverse=True)[:top_n]
    # Score-proportional weights for proper differentiation in combined ranking
    sc_vals = [max(scores[s], 0.001) for s in ss]
    total = sum(sc_vals)
    if total > 0:
        weights = {s: min(v / total, 2.0 / len(ss)) for s, v in zip(ss, sc_vals)}
        wt = sum(weights.values())
        if wt > 0:
            weights = {s: w / wt for s, w in weights.items()}
        return weights
    return {s: 1.0 / len(ss) for s in ss}


def build_signals_v9(raw, enhanced_data=None, top_n=8):
    """
    Build signals using v9.6 multi-strategy framework.

    Returns list of signal dicts in the same format as build_signals():
    [{symbol, probability, confidence, ml_mode, rank, is_top_5, signal}, ...]
    """
    global _uni_cache, _uni_cache_date

    # Use the last available trading date from raw data (not today — may be weekend/holiday)
    spy_df = raw.get("SPY", pd.DataFrame())
    if spy_df.empty or len(spy_df) == 0:
        log.error("SPY data unavailable — cannot build v9.6 signals")
        return []
    today = pd.Timestamp(spy_df.index[-1]).normalize()

    # Rebuild universe if needed (once per day)
    if _uni_cache is None or _uni_cache_date != today:
        log.info("Building v9.6 FastUniverse ...")
        _uni_cache = _build_universe(raw, enhanced_data)
        _uni_cache_date = today
        # Load SI change data for the enhanced momentum scoring
        _load_si_change_data(_uni_cache)
        log.info("v9.6 FastUniverse ready")

    uni = _uni_cache

    # Run all strategies (day_idx=0 forces rebalance)
    s4_active = {}
    t1 = strategy1_momentum_reversal(today, uni, 0, top_n=top_n, rebal_days=1)
    t3 = strategy3_sector_rotation(today, uni, 0, rebal_days=1)
    t4 = strategy4_index_inclusion(today, uni, 0, s4_active)
    t5 = strategy5_lowvol_quality(today, uni, 0, top_n=10, rebal_days=1)

    # Value strategy: uses SP1500 membership for the value stock universe
    members = get_sp1500_on_date(today)
    t7 = _strategy_value(uni, today, members, top_n=10)

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

    # Crypto tilt and VIX pause REMOVED — not in backtest, negligible impact
    paused = set()

    # UMD crash regime (matches backtest: shift to value-heavy when momentum crashes)
    umd_crash = False
    try:
        _data_dir = Path(__file__).resolve().parent / "data"
        ff_file = _data_dir / "wrds" / "fama_french_5factors_momentum_daily.parquet"
        if ff_file.exists():
            ff = pd.read_parquet(ff_file)
            ff["date"] = pd.to_datetime(ff["date"])
            ff = ff.set_index("date").sort_index()
            umd_20d = ff["umd"].rolling(20).sum()
            recent = umd_20d.loc[:today]
            if len(recent) > 0 and pd.notna(recent.iloc[-1]) and recent.iloc[-1] < -0.05:
                umd_crash = True
                log.info(f"UMD CRASH detected: 20d sum = {recent.iloc[-1]:.4f} — shifting to value-heavy weights")
    except Exception as e:
        log.warning(f"UMD check failed: {e}")

    # Override bull weights during momentum crash (matches backtest exactly)
    if umd_crash:
        crash_weights = {
            "s1_momentum": 0.15, "s7_value": 0.45,
            "s5_lowvol": 0.30, "s3_sector": 0.10, "s4_inclusion": 0.00,
        }
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
    # 1) Single-name cap: 25% (v11: matches backtest cap=0.25)
    for sym in list(combined):
        if combined[sym] > 0.25:
            combined[sym] = 0.25
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
