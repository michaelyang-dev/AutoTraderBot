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

# v10.2 strategy config: 40% momentum, 20% value, 40% low-vol quality
# Validated improvement over v10.1: +23.5% CAGR (was +22.8%), 1.14 Sharpe (was 1.00)
# WF GeoMean +27.8% (was +24.6%), wins 7/9 two-year windows, 6/10 years
# 2022 bear: only -5.8% (was -18.8%). Lower vol: 20.4% (was 23.2%)
# With 1.25x leverage: +27.4% CAGR, 1.41 Sharpe, worst year -8.6%
# Bear regime shifts to more defensive (same as before)
STRATEGY_CONFIG_BULL = [
    ("s1_momentum", 0.40),
    ("s7_value",    0.20),
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

        # Only keep last row (today) — live server doesn't need historical lookbacks
        # This saves ~1.35 GB of memory (was storing 380 days × 1540 symbols)
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
    """Fill fundamental columns from FMP parquet caches (same source as backtest)."""
    ratios_file = data_dir / "fundamentals_ratios.parquet"
    income_file = data_dir / "fundamentals_income.parquet"
    metrics_file = data_dir / "fundamentals_metrics.parquet"
    earnings_file = data_dir / "fundamentals_earnings.parquet"

    # Ratios: pe_ratio, ps_ratio, debt_to_equity, current_ratio
    if ratios_file.exists():
        ratios = pd.read_parquet(ratios_file)
        # Use filing_date if available (point-in-time safe), else fall back to date
        rat_date_col = "filing_date" if "filing_date" in ratios.columns else "date"
        if rat_date_col in ratios.columns:
            ratios[rat_date_col] = pd.to_datetime(ratios[rat_date_col])
        ratios_latest = ratios.sort_values(rat_date_col).groupby("symbol").last()
        for col in ["pe_ratio", "ps_ratio", "debt_to_equity", "current_ratio"]:
            if col in ratios_latest.columns:
                sym_vals = ratios_latest[col].to_dict()
                mask = features["symbol"].isin(sym_vals)
                features.loc[mask, col] = features.loc[mask, "symbol"].map(sym_vals)

    # Income: gross_margin, operating_margin, net_margin
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

    # Metrics: roe, roa
    if metrics_file.exists():
        metrics = pd.read_parquet(metrics_file)
        # Use filing_date if available (point-in-time safe)
        met_date_col = "filing_date" if "filing_date" in metrics.columns else "date"
        if met_date_col in metrics.columns:
            metrics[met_date_col] = pd.to_datetime(metrics[met_date_col])
        latest = metrics.sort_values(met_date_col).groupby("symbol").last()
        for col in ["roe", "roa"]:
            if col in latest.columns:
                sym_vals = latest[col].to_dict()
                mask = features["symbol"].isin(sym_vals)
                features.loc[mask, col] = features.loc[mask, "symbol"].map(sym_vals)

    # Earnings: eps_surprise_last
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
        if dv is not None and dv < -0.15:
            continue
        if debt is not None and not np.isnan(debt) and debt > 3.0:
            continue
        scores[sym] = -rv * 0.30 + g * 0.25 + min(r, 0.5) * 0.25

    if not scores:
        return {}
    ss = sorted(scores, key=scores.get, reverse=True)[:top_n]
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

    # Crypto risk-on tilt (matches backtester lines 664-667)
    cfx = uni._crypto_fx.get(today, {})
    btc_ret = cfx.get("btc_ret_20d", 0)
    if btc_ret and not np.isnan(btc_ret) and btc_ret > 0.15:
        blend = min(1.0, blend + 0.10)

    # VIX pause (matches backtest logic)
    paused = set()
    regime = uni.get_regime(today)
    vix = regime.get("vix", 20)
    if vix > 40:
        paused = {"s1_momentum", "s4_inclusion"}

    # Combine with dynamic allocation
    combined = {}
    for name, bull_pct in STRATEGY_CONFIG_BULL:
        if name in paused:
            continue
        bear_pct = dict(STRATEGY_CONFIG_BEAR).get(name, 0)
        cap = bull_pct * blend + bear_pct * (1 - blend)
        for sym, w in targets.get(name, {}).items():
            if w > 0:
                combined[sym] = combined.get(sym, 0) + w * cap

    # Apply constraints (must match backtest exactly)
    # 1) Single-name cap: 15%
    for sym in list(combined):
        if combined[sym] > 0.15:
            combined[sym] = 0.15
    # 2) Sector cap: 35%
    sec_tot = {}
    for sym, w in combined.items():
        sec = uni.sector_map.get(sym, "X")
        sec_tot[sec] = sec_tot.get(sec, 0) + w
    for sec, tot in sec_tot.items():
        if tot > 0.35:
            scale = 0.35 / tot
            for sym in list(combined):
                if uni.sector_map.get(sym, "X") == sec:
                    combined[sym] *= scale
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
        sig["signal"] = "BUY" if sig["is_top_5"] else "HOLD"

    buy_count = sum(1 for s in signals if s["signal"] == "BUY")
    log.info("v9.6 signals: %d total, %d BUY (breadth=%.1f%%, blend=%.2f)",
             len(signals), buy_count, breadth * 100, blend)

    return signals
