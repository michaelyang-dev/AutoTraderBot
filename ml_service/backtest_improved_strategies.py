#!/usr/bin/env python3
"""
Research-Backed Strategy Improvements Backtest
===============================================
Tests 3 variants across 6 periods:

Original Variant C: baseline (50/50 ML + 3mo mom + simple MR + 60d megacap)
Improved Variant C: ALL research improvements
  - Momentum: 12-1 momentum + 52wk high filter + consistency score
  - Mean Reversion: trend filter + dual confirm + volume cap + ATR stops
  - Mega-cap: 12-1 momentum + 52wk high filter + 50-SMA exit
Hybrid: Only momentum improvements (test if MR changes help)

ML strategy unchanged: bug-fixed 50/50 LGBM+RF ensemble.
"""

import sys
import time
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES_FILE = DATA_DIR / "features.parquet"

TOP_PERCENTILE = 0.20
CALIB_FRAC = 0.20

# ── Shared backtest parameters ──
INITIAL_CASH = 100_000.0
SLIPPAGE = 0.0005
POSITION_PCT = 0.12
TARGET_VOL = 0.15
MAX_LEVERAGE = 1.5
MIN_LEVERAGE = 0.3
VOL_LOOKBACK = 20
SPY_RESERVE_PCT = 0.30
SPY_THRESHOLD_PCT = 0.20
SPY_INVEST_PCT = 0.85
COOLDOWN_DAYS = 5
SPY_SMA_FAST = 50
SPY_SMA_SLOW = 200

# ML signal picks
ML_TOP_N = 2
HOLD_DAYS = 10
MAX_POSITIONS = 11

# ── ORIGINAL strategy params ──
ORIG_MOM_LOOKBACK = 63       # 3 months
ORIG_MOM_TOP_N = 3
ORIG_MOM_STOP = -0.08
ORIG_MOM_TP = 0.20
ORIG_MOM_MAX_HOLD = 60

ORIG_MR_TOP_N = 3
ORIG_MR_DROP_PERIOD = 30
ORIG_MR_DROP_THRESH = -0.15
ORIG_MR_STOP = -0.10
ORIG_MR_MAX_HOLD = 10

ORIG_MCAP_LOOKBACK = 60
ORIG_MCAP_TOP_N = 2
ORIG_MCAP_STOP = -0.05
ORIG_MCAP_TP = 0.20
ORIG_MCAP_MAX_HOLD = 90

# ── IMPROVED strategy params ──
# Momentum: 12-1 (Wes Gray / Quantitative Momentum)
IMP_MOM_LOOKBACK = 252       # 12 months
IMP_MOM_SKIP = 21            # skip most recent month
IMP_MOM_52WK_THRESHOLD = 0.92  # within 8% of 52-week high
IMP_MOM_CONSISTENCY_THRESHOLD = 0.55  # >55% positive days
IMP_MOM_CONSISTENCY_BOOST = 0.10
IMP_MOM_TOP_N = 3
IMP_MOM_STOP = -0.08
IMP_MOM_TP = 0.20
IMP_MOM_MAX_HOLD = 60

# Mean Reversion: dual confirmation + ATR stops
IMP_MR_TOP_N = 3
IMP_MR_RSI_PERIOD = 14
IMP_MR_RSI_ENTRY = 30        # RSI < 30
IMP_MR_RSI_EXIT = 50         # exit when RSI > 50
IMP_MR_BB_PERIOD = 20
IMP_MR_BB_STD = 2.0
IMP_MR_VOL_SPIKE = 1.5       # 1.5x avg volume
IMP_MR_ATR_STOP_MULT = 2.0   # stop = entry - 2*ATR
IMP_MR_ATR_TP_MULT = 3.0     # TP = entry + 3*ATR
IMP_MR_MAX_HOLD = 10

# Mega-cap: 12-1 momentum + 52wk high
IMP_MCAP_LOOKBACK = 252
IMP_MCAP_SKIP = 21
IMP_MCAP_52WK_THRESHOLD = 0.95  # within 5% of 52-week high
IMP_MCAP_TOP_N = 2           # was 3 in original, keep 2 for slot allocation
IMP_MCAP_STOP = -0.05
IMP_MCAP_TP = 0.20
IMP_MCAP_MAX_HOLD = 90

MEGACAP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "V", "MA", "COST",
]

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

PERIODS = [
    ("2011-01-01", "2013-12-31", "P1: 2011-2013"),
    ("2014-01-01", "2016-12-31", "P2: 2014-2016"),
    ("2017-01-01", "2019-12-31", "P3: 2017-2019"),
    ("2020-01-01", "2022-12-31", "P4: 2020-2022"),
    ("2023-01-01", "2026-04-17", "P5: 2023-2026"),
    ("2011-01-01", "2026-04-17", "Full: 2011-2026"),
]


def log(msg=""):
    print(msg, flush=True)


def get_feature_cols(df):
    exclude = {"date", "symbol", "target", "target_v4", "in_sp500"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


class ManualCalibratedModel:
    def __init__(self, base_model, X_calib, y_calib):
        self.base_model = base_model
        raw_probs = base_model.predict_proba(X_calib)[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.iso.fit(raw_probs, y_calib)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.iso.predict(raw)
        return np.column_stack([1 - calibrated, calibrated])


def compute_spy_regime(spy_series, date):
    prices = spy_series[:date].dropna()
    if len(prices) < SPY_SMA_SLOW:
        return "BULLISH"
    price = prices.iloc[-1]
    sma50 = prices.iloc[-SPY_SMA_FAST:].mean()
    sma200 = prices.iloc[-SPY_SMA_SLOW:].mean()
    if price > sma50 and sma50 > sma200:
        return "BULLISH"
    elif price > sma200:
        return "CAUTIOUS"
    return "BEARISH"


# ══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS (for improved strategies)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rsi(prices, period=14):
    """Compute RSI for an array of prices. Returns last RSI value."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger(prices, period=20, n_std=2.0):
    """Compute Bollinger Bands. Returns (upper, middle, lower)."""
    if len(prices) < period:
        return None, None, None
    window = prices[-period:]
    middle = window.mean()
    std = window.std()
    return middle + n_std * std, middle, middle - n_std * std


def compute_atr(highs, lows, closes, period=14):
    """Compute ATR from high/low/close arrays."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(-period, 0):
        h = highs[i]
        l = lows[i]
        pc = closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return np.mean(trs)


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-STRATEGY BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    preds_df, all_dates, spy_close_series,
    open_data, close_data, high_data, low_data, volume_data,
    label,
    mom_mode="original",      # "original" or "improved"
    mr_mode="original",       # "original" or "improved"
    mcap_mode="original",     # "original" or "improved"
):
    """
    Run multi-strategy backtest.
    mom_mode/mr_mode/mcap_mode control which strategy version is used.
    """
    # Pre-compute ML signals by date (unchanged across all variants)
    ml_signals = {}
    for date, grp in preds_df.groupby("date"):
        sp500 = grp[grp["in_sp500"] == True]
        if sp500.empty:
            continue
        ranked = sp500.nlargest(ML_TOP_N, "prob_ensemble")
        ml_signals[date] = [(row.symbol, getattr(row, "prob_ensemble", 0.5))
                            for row in ranked.itertuples(index=False)]

    # Pre-build lookups
    open_lookup = {}
    close_lookup = {}
    high_lookup = {}
    low_lookup = {}
    vol_lookup = {}
    for date in all_dates:
        if date in open_data.index:
            open_lookup[date] = open_data.loc[date].to_dict()
        if date in close_data.index:
            close_lookup[date] = close_data.loc[date].to_dict()
        if date in high_data.index:
            high_lookup[date] = high_data.loc[date].to_dict()
        if date in low_data.index:
            low_lookup[date] = low_data.loc[date].to_dict()
        if date in volume_data.index:
            vol_lookup[date] = volume_data.loc[date].to_dict()

    # State
    cash = float(INITIAL_CASH)
    positions = {}
    idle_spy_shares = 0.0
    trades = []
    cooldowns = {}
    port_vals = []
    daily_returns = []

    # Strategy-specific top-N
    mom_top_n = IMP_MOM_TOP_N if mom_mode == "improved" else ORIG_MOM_TOP_N
    mr_top_n = IMP_MR_TOP_N if mr_mode == "improved" else ORIG_MR_TOP_N
    mcap_top_n = IMP_MCAP_TOP_N if mcap_mode == "improved" else ORIG_MCAP_TOP_N

    for i, date in enumerate(all_dates):
        spy_close = spy_close_series.get(date)
        if spy_close is not None and (np.isnan(spy_close) or spy_close <= 0):
            spy_close = None

        day_close = close_lookup.get(date, {})
        day_open = open_lookup.get(date, {})
        day_high = high_lookup.get(date, {})
        day_low = low_lookup.get(date, {})
        day_vol = vol_lookup.get(date, {})

        # Vol targeting
        if len(daily_returns) >= VOL_LOOKBACK:
            recent = np.array(daily_returns[-VOL_LOOKBACK:])
            rv = np.std(recent) * np.sqrt(252)
            vol_scale = max(MIN_LEVERAGE, min(MAX_LEVERAGE, TARGET_VOL / rv if rv > 0 else 1.0))
        else:
            vol_scale = 1.0

        spy_regime = compute_spy_regime(spy_close_series, date)

        # Update peaks
        for sym, pos in positions.items():
            if sym in day_close and not np.isnan(day_close[sym]):
                if day_close[sym] > pos["peak_price"]:
                    pos["peak_price"] = day_close[sym]

        # Pre-compute D+1 open for both exits and entries
        next_idx = i + 1
        next_open = open_lookup.get(all_dates[next_idx], {}) if next_idx < len(all_dates) else {}

        # ── EXITS ── (decisions use D's close, execution at D+1 open)
        to_close = []
        for sym, pos in list(positions.items()):
            curr = day_close.get(sym, pos["entry_price"])
            if np.isnan(curr) or curr <= 0:
                curr = pos["entry_price"]
            entry_ret = (curr / pos["entry_price"]) - 1.0
            days_held = i - pos["entry_idx"]
            strategy = pos["strategy"]

            exit_triggered = False

            if strategy == "ml":
                if i >= pos["exit_idx"]:
                    exit_triggered = True

            elif strategy == "momentum":
                mom_stop = IMP_MOM_STOP if mom_mode == "improved" else ORIG_MOM_STOP
                mom_tp = IMP_MOM_TP if mom_mode == "improved" else ORIG_MOM_TP
                mom_max = IMP_MOM_MAX_HOLD if mom_mode == "improved" else ORIG_MOM_MAX_HOLD
                if entry_ret <= mom_stop or entry_ret >= mom_tp or days_held >= mom_max:
                    exit_triggered = True
                drop_from_peak = (curr / pos["peak_price"]) - 1.0 if pos["peak_price"] > 0 else 0
                if drop_from_peak <= -0.08:
                    exit_triggered = True

            elif strategy == "mean_reversion":
                if mr_mode == "improved":
                    # ATR-based stops
                    atr_stop = pos.get("atr_stop")
                    atr_tp = pos.get("atr_tp")
                    if atr_stop is not None and curr <= atr_stop:
                        exit_triggered = True
                    if atr_tp is not None and curr >= atr_tp:
                        exit_triggered = True
                    if days_held >= IMP_MR_MAX_HOLD:
                        exit_triggered = True
                    # RSI > 50 exit (mean reversion complete)
                    prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                    idx_loc = prices_list.index.get_indexer([date], method="pad")
                    if len(idx_loc) > 0 and idx_loc[0] >= IMP_MR_RSI_PERIOD:
                        loc = idx_loc[0]
                        rsi_val = compute_rsi(prices_list.iloc[max(0, loc-IMP_MR_RSI_PERIOD):loc+1].values)
                        if rsi_val > IMP_MR_RSI_EXIT:
                            exit_triggered = True
                else:
                    if entry_ret <= ORIG_MR_STOP or days_held >= ORIG_MR_MAX_HOLD:
                        exit_triggered = True
                    if entry_ret >= 0.05:
                        exit_triggered = True

            elif strategy == "mega_cap":
                mcap_stop = IMP_MCAP_STOP if mcap_mode == "improved" else ORIG_MCAP_STOP
                mcap_tp = IMP_MCAP_TP if mcap_mode == "improved" else ORIG_MCAP_TP
                mcap_max = IMP_MCAP_MAX_HOLD if mcap_mode == "improved" else ORIG_MCAP_MAX_HOLD
                if entry_ret <= mcap_stop or entry_ret >= mcap_tp or days_held >= mcap_max:
                    exit_triggered = True
                # Improved: exit if falls below 50-day SMA
                if mcap_mode == "improved":
                    prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                    idx_loc = prices_list.index.get_indexer([date], method="pad")
                    if len(idx_loc) > 0 and idx_loc[0] >= 50:
                        loc = idx_loc[0]
                        sma50 = prices_list.iloc[loc-49:loc+1].mean()
                        if curr < sma50:
                            exit_triggered = True

            if exit_triggered:
                to_close.append(sym)

        for sym in to_close:
            pos = positions.pop(sym)
            exit_px = next_open.get(sym, np.nan)
            if np.isnan(exit_px) or exit_px <= 0:
                exit_px = day_close.get(sym, pos["entry_price"])  # fallback if no D+1 open
            ret = (exit_px / pos["entry_price"]) - 1.0
            gross = pos["cost"] * (1.0 + ret)
            net = gross * (1.0 - SLIPPAGE)
            cash += net
            trade_ret = (net - pos["cost"]) / pos["cost"]
            trades.append({"ret": trade_ret, "strategy": pos["strategy"]})
            cooldowns[f"{sym}:{pos['strategy']}"] = i + COOLDOWN_DAYS

        # ── ENTRIES ──
        if spy_regime == "BEARISH":
            pass
        else:
            strat_counts = Counter(p["strategy"] for p in positions.values())
            total_pos = len(positions)

            candidates_exist = bool(ml_signals.get(date))
            if spy_close and idle_spy_shares > 0 and candidates_exist and total_pos < MAX_POSITIONS:
                cash += idle_spy_shares * spy_close * (1.0 - SLIPPAGE)
                idle_spy_shares = 0.0

            held_syms = set(positions.keys())

            def can_enter(sym, strategy):
                if sym in held_syms:
                    return False
                cd_key = f"{sym}:{strategy}"
                if cd_key in cooldowns and cooldowns[cd_key] > i:
                    return False
                return True

            def try_enter(sym, prob, strategy, slot_limit, slot_key, extra=None):
                nonlocal cash, total_pos
                if total_pos >= MAX_POSITIONS:
                    return
                if strat_counts.get(slot_key, 0) >= slot_limit:
                    return
                if not can_enter(sym, strategy):
                    return
                entry_px = next_open.get(sym, np.nan)
                if np.isnan(entry_px) or entry_px <= 0:
                    return

                port_est = cash + sum(
                    p["cost"] * (day_close.get(s, p["entry_price"]) / p["entry_price"]
                                 if p["entry_price"] > 0 else 1.0) for s, p in positions.items())
                ml_mult = min(1.0, max(0.60, prob * 1.6 - 0.28)) if strategy == "ml" else 1.0
                target = port_est * POSITION_PCT * ml_mult * vol_scale
                cost = min(target, cash * 0.95)
                if cost < 50.0:
                    return

                cash -= cost * (1.0 + SLIPPAGE)
                exit_idx = min(next_idx + HOLD_DAYS, len(all_dates) - 1) if strategy == "ml" else len(all_dates) - 1
                pos_data = {
                    "strategy": strategy, "cost": cost, "entry_price": entry_px,
                    "peak_price": entry_px, "entry_idx": next_idx, "exit_idx": exit_idx,
                }
                if extra:
                    pos_data.update(extra)
                positions[sym] = pos_data
                strat_counts[slot_key] = strat_counts.get(slot_key, 0) + 1
                total_pos += 1
                held_syms.add(sym)

            # 1. ML signals (2 slots, unchanged)
            for sym, prob in ml_signals.get(date, []):
                if spy_regime == "CAUTIOUS" and strat_counts.get("ml", 0) >= 1:
                    continue
                try_enter(sym, prob, "ml", ML_TOP_N, "ml")

            # 2. Momentum signals
            if spy_regime != "BEARISH":
                if mom_mode == "improved":
                    # 12-1 Momentum (Wes Gray)
                    mom_candidates = []
                    for sym in day_close:
                        if sym == "SPY" or sym in held_syms:
                            continue
                        prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                        idx_loc = prices_list.index.get_indexer([date], method="pad")
                        if len(idx_loc) == 0 or idx_loc[0] < IMP_MOM_LOOKBACK:
                            continue
                        loc = idx_loc[0]

                        # 12-1 momentum: skip recent month
                        price_skip = prices_list.iloc[loc - IMP_MOM_SKIP]  # price 21 days ago
                        price_12m = prices_list.iloc[loc - IMP_MOM_LOOKBACK]  # price 252 days ago
                        if price_12m <= 0:
                            continue
                        mom_12_1 = (price_skip / price_12m) - 1.0

                        curr_px = prices_list.iloc[loc]

                        # SMA200 filter
                        if loc >= 200:
                            sma200 = prices_list.iloc[loc-199:loc+1].mean()
                            if curr_px <= sma200:
                                continue

                        # 52-week high filter
                        if loc >= 252:
                            high_52wk = prices_list.iloc[loc-251:loc+1].max()
                            ratio = curr_px / high_52wk
                            if ratio < IMP_MOM_52WK_THRESHOLD:
                                continue
                        else:
                            continue

                        # Consistency score: % positive days in last 252 days
                        daily_rets = prices_list.iloc[loc-251:loc+1].pct_change().dropna()
                        pct_positive = (daily_rets > 0).mean()
                        consistency_boost = IMP_MOM_CONSISTENCY_BOOST if pct_positive > IMP_MOM_CONSISTENCY_THRESHOLD else 0.0

                        score = mom_12_1 + consistency_boost
                        mom_candidates.append((sym, score, mom_12_1))

                    mom_candidates.sort(key=lambda x: x[1], reverse=True)
                    for sym, score, mom_ret in mom_candidates[:mom_top_n * 2]:
                        try_enter(sym, 0.7, "momentum", mom_top_n, "momentum")

                else:
                    # Original: 3-month return ranking
                    mom_rets = {}
                    for sym in day_close:
                        if sym == "SPY" or sym in held_syms:
                            continue
                        prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                        idx_loc = prices_list.index.get_indexer([date], method="pad")
                        if len(idx_loc) == 0 or idx_loc[0] < ORIG_MOM_LOOKBACK:
                            continue
                        loc = idx_loc[0]
                        curr_px = prices_list.iloc[loc]
                        past_px = prices_list.iloc[loc - ORIG_MOM_LOOKBACK]
                        if past_px > 0 and curr_px > 0:
                            if loc >= 200:
                                sma200 = prices_list.iloc[loc-199:loc+1].mean()
                                if curr_px <= sma200:
                                    continue
                            mom_rets[sym] = (curr_px / past_px) - 1.0

                    mom_sorted = sorted(mom_rets.items(), key=lambda x: x[1], reverse=True)
                    for sym, ret in mom_sorted[:mom_top_n * 2]:
                        try_enter(sym, 0.7, "momentum", mom_top_n, "momentum")

            # 3. Mean reversion signals
            if spy_regime != "BEARISH":
                if mr_mode == "improved":
                    # Improved: dual confirmation + volume + ATR
                    mr_candidates = []
                    for sym in day_close:
                        if sym == "SPY" or sym in held_syms:
                            continue
                        prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                        idx_loc = prices_list.index.get_indexer([date], method="pad")
                        if len(idx_loc) == 0 or idx_loc[0] < max(200, IMP_MR_BB_PERIOD, IMP_MR_RSI_PERIOD + 1):
                            continue
                        loc = idx_loc[0]
                        curr_px = prices_list.iloc[loc]

                        # 1. Above 200-day SMA (CRITICAL trend filter)
                        sma200 = prices_list.iloc[loc-199:loc+1].mean()
                        if curr_px < sma200:
                            continue

                        # 2. RSI < 30
                        rsi_val = compute_rsi(prices_list.iloc[max(0, loc-IMP_MR_RSI_PERIOD):loc+1].values)
                        if rsi_val >= IMP_MR_RSI_ENTRY:
                            continue

                        # 3. Below lower Bollinger Band
                        bb_prices = prices_list.iloc[loc-IMP_MR_BB_PERIOD+1:loc+1].values
                        bb_upper, bb_mid, bb_lower = compute_bollinger(bb_prices, IMP_MR_BB_PERIOD, IMP_MR_BB_STD)
                        if bb_lower is None or curr_px >= bb_lower:
                            continue

                        # 4. Volume spike >= 1.5x 20-day avg
                        if sym in volume_data.columns:
                            vol_list = volume_data[sym].dropna()
                            v_idx = vol_list.index.get_indexer([date], method="pad")
                            if len(v_idx) > 0 and v_idx[0] >= 20:
                                v_loc = v_idx[0]
                                curr_vol = vol_list.iloc[v_loc]
                                avg_vol = vol_list.iloc[v_loc-19:v_loc].mean()
                                if avg_vol > 0 and curr_vol < avg_vol * IMP_MR_VOL_SPIKE:
                                    continue
                            else:
                                continue
                        else:
                            continue

                        # Compute ATR for dynamic stops
                        if sym in high_data.columns and sym in low_data.columns:
                            h_list = high_data[sym].dropna()
                            l_list = low_data[sym].dropna()
                            h_idx = h_list.index.get_indexer([date], method="pad")
                            l_idx = l_list.index.get_indexer([date], method="pad")
                            if len(h_idx) > 0 and len(l_idx) > 0 and h_idx[0] >= 15 and l_idx[0] >= 15:
                                h_loc = h_idx[0]
                                l_loc = l_idx[0]
                                atr_val = compute_atr(
                                    h_list.iloc[h_loc-14:h_loc+1].values,
                                    l_list.iloc[l_loc-14:l_loc+1].values,
                                    prices_list.iloc[loc-14:loc+1].values,
                                    14
                                )
                            else:
                                atr_val = curr_px * 0.02  # fallback
                        else:
                            atr_val = curr_px * 0.02

                        atr_stop = curr_px - IMP_MR_ATR_STOP_MULT * atr_val
                        atr_tp = curr_px + IMP_MR_ATR_TP_MULT * atr_val

                        confidence = (IMP_MR_RSI_ENTRY - rsi_val) / IMP_MR_RSI_ENTRY
                        mr_candidates.append((sym, confidence, atr_stop, atr_tp))

                    mr_candidates.sort(key=lambda x: x[1], reverse=True)
                    for sym, conf, atr_stop, atr_tp in mr_candidates[:mr_top_n * 2]:
                        try_enter(sym, 0.6, "mean_reversion", mr_top_n, "mean_reversion",
                                  extra={"atr_stop": atr_stop, "atr_tp": atr_tp})

                else:
                    # Original: simple drop detection
                    mr_candidates = []
                    for sym in day_close:
                        if sym == "SPY" or sym in held_syms:
                            continue
                        prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                        idx_loc = prices_list.index.get_indexer([date], method="pad")
                        if len(idx_loc) == 0 or idx_loc[0] < max(ORIG_MR_DROP_PERIOD, 200):
                            continue
                        loc = idx_loc[0]
                        curr_px = prices_list.iloc[loc]
                        past_px = prices_list.iloc[loc - ORIG_MR_DROP_PERIOD]
                        if past_px > 0 and curr_px > 0:
                            drop = (curr_px / past_px) - 1.0
                            if drop > ORIG_MR_DROP_THRESH:
                                continue
                            sma200 = prices_list.iloc[loc-199:loc+1].mean()
                            if curr_px < sma200:
                                continue
                            recent_low = prices_list.iloc[loc-ORIG_MR_DROP_PERIOD:loc+1].min()
                            if curr_px <= recent_low * 1.001:
                                continue
                            mr_candidates.append((sym, abs(drop)))

                    mr_candidates.sort(key=lambda x: x[1], reverse=True)
                    for sym, drop_mag in mr_candidates[:mr_top_n * 2]:
                        try_enter(sym, 0.6, "mean_reversion", mr_top_n, "mean_reversion")

            # 4. Mega-cap overlay
            if mcap_mode == "improved":
                # 12-1 Momentum + 52-week high filter
                mcap_candidates = []
                for sym in MEGACAP_UNIVERSE:
                    if sym in held_syms:
                        continue
                    prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                    idx_loc = prices_list.index.get_indexer([date], method="pad")
                    if len(idx_loc) == 0 or idx_loc[0] < IMP_MCAP_LOOKBACK:
                        continue
                    loc = idx_loc[0]
                    curr_px = prices_list.iloc[loc]

                    # SMA200 filter
                    if loc >= 200:
                        sma200 = prices_list.iloc[loc-199:loc+1].mean()
                        if curr_px <= sma200:
                            continue

                    # 12-1 momentum
                    price_skip = prices_list.iloc[loc - IMP_MCAP_SKIP]
                    price_12m = prices_list.iloc[loc - IMP_MCAP_LOOKBACK]
                    if price_12m <= 0:
                        continue
                    mom_12_1 = (price_skip / price_12m) - 1.0

                    # 52-week high filter (within 5%)
                    if loc >= 252:
                        high_52wk = prices_list.iloc[loc-251:loc+1].max()
                        ratio = curr_px / high_52wk
                        if ratio < IMP_MCAP_52WK_THRESHOLD:
                            continue

                    mcap_candidates.append((sym, mom_12_1))

                mcap_candidates.sort(key=lambda x: x[1], reverse=True)
                for sym, mom in mcap_candidates[:mcap_top_n * 2]:
                    try_enter(sym, 0.7, "mega_cap", mcap_top_n, "mega_cap")

            else:
                # Original: 60-day momentum
                mcap_rets = {}
                for sym in MEGACAP_UNIVERSE:
                    if sym in held_syms:
                        continue
                    prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                    idx_loc = prices_list.index.get_indexer([date], method="pad")
                    if len(idx_loc) == 0 or idx_loc[0] < max(ORIG_MCAP_LOOKBACK, 200):
                        continue
                    loc = idx_loc[0]
                    curr_px = prices_list.iloc[loc]
                    past_px = prices_list.iloc[loc - ORIG_MCAP_LOOKBACK]
                    if past_px > 0 and curr_px > 0:
                        sma200 = prices_list.iloc[loc-199:loc+1].mean()
                        if curr_px <= sma200:
                            continue
                        mcap_rets[sym] = (curr_px / past_px) - 1.0

                mcap_sorted = sorted(mcap_rets.items(), key=lambda x: x[1], reverse=True)
                for sym, ret in mcap_sorted[:mcap_top_n * 2]:
                    try_enter(sym, 0.7, "mega_cap", mcap_top_n, "mega_cap")

        # SPY parking
        if spy_close:
            pos_val = sum(p["cost"] * (day_close.get(s, p["entry_price"]) / p["entry_price"]
                          if p["entry_price"] > 0 else 1.0) for s, p in positions.items())
            est_port = cash + idle_spy_shares * spy_close + pos_val
            idle_cash = cash - est_port * SPY_RESERVE_PCT
            if idle_cash > est_port * SPY_THRESHOLD_PCT:
                invest = min(idle_cash * SPY_INVEST_PCT, cash * 0.95)
                cash -= invest * (1.0 + SLIPPAGE)
                idle_spy_shares += invest / spy_close

        # Mark to market
        port_val = cash + (idle_spy_shares * spy_close if spy_close else 0)
        for s, pos in positions.items():
            px = day_close.get(s, pos["entry_price"])
            port_val += pos["cost"] * (px / pos["entry_price"] if pos["entry_price"] > 0 else 1.0)
        port_vals.append(port_val)
        daily_returns.append((port_vals[-1] / port_vals[-2] - 1.0) if len(port_vals) >= 2 else 0.0)

    return pd.Series(port_vals, index=pd.DatetimeIndex(all_dates)), trades


def calc_metrics(vals, trades, label):
    if len(vals) < 2:
        return {"label": label, "cagr": 0, "sharpe": 0, "max_dd": 0,
                "n_trades": 0, "win_rate": 0}
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0:
        return {"label": label, "cagr": 0, "sharpe": 0, "max_dd": 0,
                "n_trades": 0, "win_rate": 0}
    final = vals.iloc[-1]
    init = vals.iloc[0]
    cagr = (final / init) ** (1.0 / years) - 1.0
    dr = vals.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    wins = sum(1 for t in trades if t["ret"] > 0)

    # Per-strategy trade stats
    strat_stats = {}
    for strat in ["ml", "momentum", "mean_reversion", "mega_cap"]:
        strat_trades = [t for t in trades if t["strategy"] == strat]
        if strat_trades:
            strat_wins = sum(1 for t in strat_trades if t["ret"] > 0)
            strat_stats[strat] = {
                "count": len(strat_trades),
                "win_rate": strat_wins / len(strat_trades),
                "avg_ret": np.mean([t["ret"] for t in strat_trades]),
            }

    return {"label": label, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
            "n_trades": len(trades), "win_rate": wins / len(trades) if trades else 0,
            "strat_stats": strat_stats}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  RESEARCH-BACKED STRATEGY IMPROVEMENTS BACKTEST")
    log("  3 variants × 6 periods")
    log("=" * 70)

    # ── Stage 1: Load features & train models ─────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 1: LOAD FEATURES & TRAIN MODELS")
    log("=" * 70)

    df = pd.read_parquet(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).copy()
    assert "days_until_earnings" not in df.columns

    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_v4"] = np.nan
    valid_df = df[sp500_mask & has_fwd].copy()
    valid_df["pct_rank"] = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_v4"] = (valid_df["pct_rank"] >= (1.0 - TOP_PERCENTILE)).astype(int)
    df.loc[valid_df.index, "target_v4"] = valid_df["target_v4"]

    for base_col, rank_col in [("vol_20d", "vol_rank_20d"), ("ret_60d", "momentum_rank_60d"),
                                ("rsi_14", "rsi_rank"), ("dist_sma50", "dist_sma50_rank")]:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    feature_cols = get_feature_cols(df)
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund_cols + ["target_v4"])
    train_df = df[df["in_sp500"] == True].copy()

    log(f"  Total rows: {len(train_df):,} | Features: {len(feature_cols)}")

    X_all = df[feature_cols].values
    X_filtered = train_df[feature_cols].values
    y_filtered = train_df["target_v4"].values

    all_dates_train = np.sort(train_df["date"].unique())
    split_idx = int(len(all_dates_train) * (1 - CALIB_FRAC))
    calib_start = pd.Timestamp(all_dates_train[split_idx])
    train_mask = train_df["date"] < calib_start
    calib_mask = train_df["date"] >= calib_start

    X_train, y_train = X_filtered[train_mask], y_filtered[train_mask]
    X_calib, y_calib = X_filtered[calib_mask], y_filtered[calib_mask]
    log(f"  Train: {train_mask.sum():,} | Calib: {calib_mask.sum():,}")

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_calib_imp = imp.transform(X_calib)
    X_all_imp = imp.transform(X_all)

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    log(f"\n  Training LightGBM ...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, objective="binary", metric="auc",
        random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=scale)
    lgb_model.fit(X_train, y_train, eval_set=[(X_calib, y_calib)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    calib_lgbm = CalibratedClassifierCV(lgb_model, method="isotonic", cv="prefit")
    calib_lgbm.fit(X_calib, y_calib)
    auc_lgbm = roc_auc_score(y_calib, calib_lgbm.predict_proba(X_calib)[:, 1])
    prob_lgbm = calib_lgbm.predict_proba(X_all)[:, 1]
    log(f"    LGBM AUC={auc_lgbm:.4f}")

    log(f"  Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=100,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train_imp, y_train)
    calib_rf = ManualCalibratedModel(rf, X_calib_imp, y_calib)
    auc_rf = roc_auc_score(y_calib, calib_rf.predict_proba(X_calib_imp)[:, 1])
    prob_rf = calib_rf.predict_proba(X_all_imp)[:, 1]
    log(f"    RF AUC={auc_rf:.4f}")

    df["prob_ensemble"] = 0.5 * prob_lgbm + 0.5 * prob_rf
    df["fwd_ret"] = df["fwd_10d_ret"]

    # ── Stage 2: Fetch OHLCV ──────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 2: FETCH OHLCV DATA")
    log("=" * 70)

    preds = df[["date", "symbol", "target_v4", "prob_ensemble",
                "fwd_ret", "in_sp500"]].copy()
    preds = preds.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds["date"].unique().tolist())
    universe_syms = sorted(preds["symbol"].unique().tolist())
    all_syms = list(set(["SPY"] + universe_syms + MEGACAP_UNIVERSE))

    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)

    log(f"  Fetching OHLCV for {len(all_syms)} symbols ...")
    raw = yf.download(all_syms, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    open_px = raw["Open"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Open"]]
    open_px.index = pd.to_datetime(open_px.index).tz_localize(None)
    high_px = raw["High"] if isinstance(raw.columns, pd.MultiIndex) else raw[["High"]]
    high_px.index = pd.to_datetime(high_px.index).tz_localize(None)
    low_px = raw["Low"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Low"]]
    low_px.index = pd.to_datetime(low_px.index).tz_localize(None)
    vol_px = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]
    vol_px.index = pd.to_datetime(vol_px.index).tz_localize(None)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    open_aligned = open_px.reindex(sim_index, method="ffill")
    high_aligned = high_px.reindex(sim_index, method="ffill")
    low_aligned = low_px.reindex(sim_index, method="ffill")
    vol_aligned = vol_px.reindex(sim_index, method="ffill")
    spy_close_series = close_aligned["SPY"].dropna()

    log(f"  OHLCV: {len(close_aligned)} dates, {len(close_aligned.columns)} symbols")

    # ── Stage 3: Run backtests ────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 3: RUN BACKTESTS (6 periods × 3 variants)")
    log("=" * 70)

    variants = [
        ("Original C",  "original",  "original",  "original"),
        ("Improved C",  "improved",  "improved",  "improved"),
        ("Hybrid",      "improved",  "original",  "original"),
    ]

    all_results = []

    for p_start, p_end, p_label in PERIODS:
        p_start_ts = pd.Timestamp(p_start)
        p_end_ts = pd.Timestamp(p_end)
        period_dates = [d for d in all_dates
                        if pd.Timestamp(d) >= p_start_ts and pd.Timestamp(d) <= p_end_ts]
        if len(period_dates) < 20:
            log(f"\n  {p_label}: too few dates — skipping")
            continue

        period_preds = preds[(preds["date"] >= p_start_ts) & (preds["date"] <= p_end_ts)].copy()

        log(f"\n  ── {p_label} ({len(period_dates)} trading days) ──")

        # SPY benchmark
        spy_period = spy_close_series.reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in period_dates])).dropna()
        if len(spy_period) > 1:
            spy_years = (spy_period.index[-1] - spy_period.index[0]).days / 365.25
            spy_cagr = (spy_period.iloc[-1] / spy_period.iloc[0]) ** (1.0 / spy_years) - 1.0 if spy_years > 0 else 0
            spy_dr = spy_period.pct_change().dropna()
            spy_sharpe = spy_dr.mean() / spy_dr.std() * np.sqrt(252) if spy_dr.std() > 0 else 0
            spy_peak = spy_period.cummax()
            spy_dd = ((spy_period - spy_peak) / spy_peak).min()
        else:
            spy_cagr = spy_sharpe = spy_dd = 0

        period_results = {}
        for v_label, mom_m, mr_m, mcap_m in variants:
            vals, trades = run_backtest(
                period_preds, period_dates, spy_close_series,
                open_aligned, close_aligned, high_aligned, low_aligned, vol_aligned,
                v_label, mom_mode=mom_m, mr_mode=mr_m, mcap_mode=mcap_m)
            m = calc_metrics(vals, trades, v_label)
            m["period"] = p_label
            m["spy_cagr"] = spy_cagr
            m["spy_sharpe"] = spy_sharpe
            m["spy_dd"] = spy_dd
            all_results.append(m)
            period_results[v_label] = m

        log(f"\n  {'Variant':<16s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>8s} {'Win%':>7s}")
        log(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
        for v_label, _, _, _ in variants:
            m = period_results[v_label]
            log(f"  {m['label']:<16s} {m['cagr']*100:>7.2f}% {m['sharpe']:>7.3f} "
                f"{m['max_dd']*100:>7.1f}% {m['n_trades']:>7d} {m['win_rate']*100:>5.1f}%")
        log(f"  {'SPY':<16s} {spy_cagr*100:>7.2f}% {spy_sharpe:>7.3f} {spy_dd*100:>7.1f}%")

        # Per-strategy breakdown for this period
        if p_label in ("P5: 2023-2026", "Full: 2011-2026"):
            for v_label, _, _, _ in variants:
                m = period_results[v_label]
                ss = m.get("strat_stats", {})
                if ss:
                    parts = []
                    for strat in ["ml", "momentum", "mean_reversion", "mega_cap"]:
                        if strat in ss:
                            s = ss[strat]
                            parts.append(f"{strat[:4]}:{s['count']}t/{s['win_rate']*100:.0f}%w/{s['avg_ret']*100:.2f}%r")
                    log(f"    {v_label}: {' | '.join(parts)}")

    # ── Stage 4: Summary & pass criteria ──────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 4: FULL COMPARISON TABLE")
    log("=" * 70)

    log(f"\n  {'Period':<20s} {'Variant':<16s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>7s} {'Win%':>6s} {'SPY':>8s}")
    log(f"  {'─'*20} {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*6} {'─'*8}")

    for p_start, p_end, p_label in PERIODS:
        period_res = [r for r in all_results if r["period"] == p_label]
        for m in period_res:
            log(f"  {m['period']:<20s} {m['label']:<16s} {m['cagr']*100:>7.2f}% {m['sharpe']:>7.3f} "
                f"{m['max_dd']*100:>7.1f}% {m['n_trades']:>6d} {m['win_rate']*100:>5.1f}% "
                f"{m.get('spy_cagr', 0)*100:>7.2f}%")

    # Winner by period
    log(f"\n  Winner by period (highest Sharpe):")
    for p_start, p_end, p_label in PERIODS:
        period_res = [r for r in all_results if r["period"] == p_label]
        if period_res:
            best = max(period_res, key=lambda x: x["sharpe"])
            log(f"    {p_label}: {best['label']} (Sharpe={best['sharpe']:.3f}, CAGR={best['cagr']*100:.1f}%)")

    # ── Contribution analysis ─────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 5: CONTRIBUTION ANALYSIS")
    log("=" * 70)

    # Compare Original vs Improved on full period
    orig_full = next((r for r in all_results if r["period"] == "Full: 2011-2026" and r["label"] == "Original C"), None)
    imp_full = next((r for r in all_results if r["period"] == "Full: 2011-2026" and r["label"] == "Improved C"), None)
    hyb_full = next((r for r in all_results if r["period"] == "Full: 2011-2026" and r["label"] == "Hybrid"), None)

    if orig_full and imp_full and hyb_full:
        log(f"\n  Full period deltas vs Original C:")
        log(f"  {'Variant':<16s} {'ΔCAGR':>8s} {'ΔSharpe':>9s} {'ΔDD':>8s}")
        log(f"  {'─'*16} {'─'*8} {'─'*9} {'─'*8}")
        for m in [imp_full, hyb_full]:
            dcagr = (m["cagr"] - orig_full["cagr"]) * 100
            dsharpe = m["sharpe"] - orig_full["sharpe"]
            ddd = (m["max_dd"] - orig_full["max_dd"]) * 100
            log(f"  {m['label']:<16s} {dcagr:>+7.2f}% {dsharpe:>+8.3f} {ddd:>+7.1f}%")

        # Which improvement contributed most?
        # Hybrid = improved mom + original MR + original mcap
        # Improved = improved mom + improved MR + improved mcap
        # So: MR+mcap contribution = Improved - Hybrid
        # Mom contribution = Hybrid - Original
        mom_delta_cagr = (hyb_full["cagr"] - orig_full["cagr"]) * 100
        mom_delta_sharpe = hyb_full["sharpe"] - orig_full["sharpe"]
        mr_mcap_delta_cagr = (imp_full["cagr"] - hyb_full["cagr"]) * 100
        mr_mcap_delta_sharpe = imp_full["sharpe"] - hyb_full["sharpe"]

        log(f"\n  Improvement attribution (full period):")
        log(f"    Momentum improvements: ΔCAGR={mom_delta_cagr:+.2f}%, ΔSharpe={mom_delta_sharpe:+.3f}")
        log(f"    MR + Mega-cap improvements: ΔCAGR={mr_mcap_delta_cagr:+.2f}%, ΔSharpe={mr_mcap_delta_sharpe:+.3f}")

    # ── Pass criteria ─────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 6: PASS CRITERIA")
    log("=" * 70)

    holdout = next((r for r in all_results if r["period"] == "P5: 2023-2026" and r["label"] == "Improved C"), None)
    full = next((r for r in all_results if r["period"] == "Full: 2011-2026" and r["label"] == "Improved C"), None)

    pass_criteria = []
    if holdout:
        pass_criteria.append(("Holdout CAGR >= 45%", holdout["cagr"] >= 0.45, f"{holdout['cagr']*100:.2f}%"))
    if full:
        pass_criteria.append(("Full period CAGR >= 35%", full["cagr"] >= 0.35, f"{full['cagr']*100:.2f}%"))
        pass_criteria.append(("Full period Sharpe >= 1.80", full["sharpe"] >= 1.80, f"{full['sharpe']:.3f}"))

    worst_dd = 0
    for p_start, p_end, p_label in PERIODS[:5]:
        imp_period = next((r for r in all_results if r["period"] == p_label and r["label"] == "Improved C"), None)
        if imp_period:
            worst_dd = min(worst_dd, imp_period["max_dd"])
    pass_criteria.append(("Max DD <= -22% worst period", worst_dd >= -0.22, f"{worst_dd*100:.1f}%"))

    log(f"\n  Improved Variant C pass criteria:")
    all_pass = True
    for criterion, passed, value in pass_criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        log(f"    [{status}] {criterion}: {value}")

    # ── Recommendation ────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("RECOMMENDATION")
    log("=" * 70)

    if all_pass:
        log("\n  All criteria PASS.")
        log("  Recommend deploying Improved Variant C with all research-backed improvements.")
    else:
        log("\n  NOT all criteria passed for Improved Variant C.")

        # Check if any variant beats Original C on holdout
        orig_holdout = next((r for r in all_results if r["period"] == "P5: 2023-2026" and r["label"] == "Original C"), None)
        imp_holdout = holdout
        hyb_holdout = next((r for r in all_results if r["period"] == "P5: 2023-2026" and r["label"] == "Hybrid"), None)

        log(f"\n  Holdout comparison:")
        for m in [orig_holdout, imp_holdout, hyb_holdout]:
            if m:
                log(f"    {m['label']:<16s}: CAGR={m['cagr']*100:.2f}%, Sharpe={m['sharpe']:.3f}, DD={m['max_dd']*100:.1f}%")

        # Identify which improvements hurt
        if orig_full and imp_full and hyb_full:
            if hyb_full["sharpe"] > orig_full["sharpe"] and imp_full["sharpe"] < hyb_full["sharpe"]:
                log(f"\n  Analysis: Momentum improvements HELPED (+{mom_delta_sharpe:+.3f} Sharpe)")
                log(f"           MR+Mega-cap improvements HURT ({mr_mcap_delta_sharpe:+.3f} Sharpe)")
                log(f"  Recommendation: Deploy Hybrid (momentum improvements only)")
            elif hyb_full["sharpe"] < orig_full["sharpe"]:
                log(f"\n  Analysis: Momentum improvements HURT ({mom_delta_sharpe:+.3f} Sharpe)")
                log(f"  Recommendation: Deploy Original Variant C as previously planned")
            else:
                log(f"\n  Analysis: All improvements contributed positively on full period")
                log(f"  But criteria not met — deploy Original Variant C for safety")

        holdout_results = [r for r in all_results if r["period"] == "P5: 2023-2026"]
        full_results = [r for r in all_results if r["period"] == "Full: 2011-2026"]

        if holdout_results:
            best_holdout = max(holdout_results, key=lambda x: x["sharpe"])
            log(f"\n  Best on holdout: {best_holdout['label']} (Sharpe={best_holdout['sharpe']:.3f})")
        if full_results:
            best_full = max(full_results, key=lambda x: x["sharpe"])
            log(f"  Best on full:    {best_full['label']} (Sharpe={best_full['sharpe']:.3f})")

    elapsed = time.perf_counter() - t0
    log(f"\n{'='*70}")
    log(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
