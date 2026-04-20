#!/usr/bin/env python3
"""
Regime-Adaptive Trading System Backtest
========================================
Tests 4 variants across 5 time periods + full period:

Variants:
  A: Baseline (bug-fixed 50/50 ML ensemble + momentum + mean reversion)
  B: Regime-adaptive ML only (same 3 strategies, but ML switches modes)
  C: Baseline ML + mega-cap overlay (4 strategies, no regime switch)
  D: Full system (regime-adaptive ML + mega-cap overlay = 4 strategies)

Periods:
  1: 2011-2013 (broad, post-2008 recovery)
  2: 2014-2016 (broad with mid-cycle)
  3: 2017-2019 (moderate concentration)
  4: 2020-2022 (COVID + bear)
  5: 2023-2026 (narrow market holdout)
  Full: 2011-2026

Also runs breadth backtesting (Task 7) and pass criteria evaluation (Task 8).
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

# ── Backtest parameters ──
INITIAL_CASH = 100_000.0
SLIPPAGE = 0.0005
HOLD_DAYS = 10
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
ML_TOP_N = 2           # ML gets 2 slots
MOM_TOP_N = 3          # Momentum gets 3 slots (reduced from 4)
MR_TOP_N = 3           # Mean reversion gets 3 slots
MCAP_TOP_N = 2         # Mega-cap overlay gets 2 slots
MAX_POSITIONS = 11     # Total slots

# Momentum parameters
MOM_LOOKBACK = 63
MOM_STOP = -0.08
MOM_TP = 0.20
MOM_MAX_HOLD = 60

# Mean reversion parameters
MR_DROP_PERIOD = 30
MR_DROP_THRESH = -0.15
MR_STOP = -0.10
MR_MAX_HOLD = 10

# Mega-cap overlay parameters
MCAP_LOOKBACK = 60
MCAP_STOP = -0.05
MCAP_TP = 0.20
MCAP_MAX_HOLD = 90
MCAP_COOLDOWN = 2

# Breadth parameters
BREADTH_LOOKBACK = 60
BREADTH_NARROW = 40
BREADTH_BROAD = 55

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

# Mega-cap universe (top 15 by market cap, stable over time)
# Using modern top-15 for simplicity; historical composition would require
# point-in-time data which we approximate
MEGACAP_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "V", "MA", "COST",
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


# ══════════════════════════════════════════════════════════════════════════════
#  BREADTH COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_breadth_series(close_df, spy_close, lookback=BREADTH_LOOKBACK):
    """
    Compute market breadth for each trading date.
    Returns Series: date → % of stocks beating SPY over lookback period.
    """
    dates = close_df.index
    breadth = pd.Series(np.nan, index=dates)

    spy_vals = spy_close.reindex(dates)

    for i in range(lookback, len(dates)):
        date = dates[i]
        spy_ret = (spy_vals.iloc[i] / spy_vals.iloc[i - lookback]) - 1.0
        if np.isnan(spy_ret):
            continue

        beat = 0
        total = 0
        for col in close_df.columns:
            if col == "SPY":
                continue
            curr = close_df[col].iloc[i]
            past = close_df[col].iloc[i - lookback]
            if np.isnan(curr) or np.isnan(past) or past <= 0:
                continue
            stock_ret = (curr / past) - 1.0
            total += 1
            if stock_ret > spy_ret:
                beat += 1

        if total > 0:
            breadth.iloc[i] = (beat / total) * 100.0

    return breadth


def classify_regime(breadth_val):
    if np.isnan(breadth_val):
        return "BROAD"
    if breadth_val < BREADTH_NARROW:
        return "NARROW"
    elif breadth_val <= BREADTH_BROAD:
        return "TRANSITION"
    return "BROAD"


# ══════════════════════════════════════════════════════════════════════════════
#  SPY REGIME (SMA-based, same as production)
# ══════════════════════════════════════════════════════════════════════════════

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
#  MULTI-STRATEGY BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

def run_multi_strategy_backtest(
    preds_df, all_dates, spy_close_series, open_data, close_data,
    breadth_series, label,
    use_regime_adaptive_ml=False,
    use_megacap_overlay=False,
):
    """
    Run multi-strategy backtest with:
    - ML signals (2 slots) — optionally regime-adaptive
    - Momentum signals (3 slots)
    - Mean reversion signals (3 slots)
    - Mega-cap overlay signals (2 slots) — optional
    """
    # Pre-compute ML signals by date
    ml_signals = {}
    for date, grp in preds_df.groupby("date"):
        sp500 = grp[grp["in_sp500"] == True]
        if sp500.empty:
            continue

        # Breadth regime for this date
        breadth_val = breadth_series.get(date, 50.0) if breadth_series is not None else 50.0
        breadth_regime = classify_regime(breadth_val)

        if use_regime_adaptive_ml and breadth_regime == "NARROW":
            # Mode B: rank by 3-month momentum among top 30 by market cap proxy
            # Use ret_60d as momentum proxy
            if "ret_60d" in sp500.columns:
                top30 = sp500.nlargest(30, "ret_60d")  # rough cap proxy via momentum
                ranked = top30.nlargest(ML_TOP_N, "ret_60d")
            else:
                ranked = sp500.nlargest(ML_TOP_N, "prob_ensemble")
        elif use_regime_adaptive_ml and breadth_regime == "TRANSITION":
            # Mode C: 60% ML + 40% momentum blend
            sp500 = sp500.copy()
            ml_min = sp500["prob_ensemble"].min()
            ml_max = sp500["prob_ensemble"].max()
            ml_range = ml_max - ml_min if ml_max > ml_min else 1.0
            sp500["ml_norm"] = (sp500["prob_ensemble"] - ml_min) / ml_range

            if "ret_60d" in sp500.columns:
                mom_min = sp500["ret_60d"].min()
                mom_max = sp500["ret_60d"].max()
                mom_range = mom_max - mom_min if mom_max > mom_min else 1.0
                sp500["mom_norm"] = (sp500["ret_60d"] - mom_min) / mom_range
                sp500["blend_score"] = 0.6 * sp500["ml_norm"] + 0.4 * sp500["mom_norm"]
            else:
                sp500["blend_score"] = sp500["ml_norm"]
            ranked = sp500.nlargest(ML_TOP_N, "blend_score")
        else:
            # Mode A: standard ML ranking
            ranked = sp500.nlargest(ML_TOP_N, "prob_ensemble")

        ml_signals[date] = [(row.symbol, getattr(row, "prob_ensemble", 0.5))
                            for row in ranked.itertuples(index=False)]

    # Pre-build lookups
    open_lookup = {}
    close_lookup = {}
    for date in all_dates:
        if date in open_data.index:
            open_lookup[date] = open_data.loc[date].to_dict()
        if date in close_data.index:
            close_lookup[date] = close_data.loc[date].to_dict()

    # State
    cash = float(INITIAL_CASH)
    positions = {}     # sym → {strategy, cost, entry_price, peak_price, entry_idx, exit_idx}
    idle_spy_shares = 0.0
    trades = []
    cooldowns = {}     # sym:strategy → date_idx
    port_vals = []
    daily_returns = []
    regime_log = []    # (date, breadth_regime)

    for i, date in enumerate(all_dates):
        spy_close = spy_close_series.get(date)
        if spy_close is not None and (np.isnan(spy_close) or spy_close <= 0):
            spy_close = None

        day_close = close_lookup.get(date, {})
        day_open = open_lookup.get(date, {})

        # Vol targeting
        if len(daily_returns) >= VOL_LOOKBACK:
            recent = np.array(daily_returns[-VOL_LOOKBACK:])
            rv = np.std(recent) * np.sqrt(252)
            vol_scale = max(MIN_LEVERAGE, min(MAX_LEVERAGE, TARGET_VOL / rv if rv > 0 else 1.0))
        else:
            vol_scale = 1.0

        spy_regime = compute_spy_regime(spy_close_series, date)
        breadth_val = breadth_series.get(date, 50.0) if breadth_series is not None else 50.0
        breadth_regime = classify_regime(breadth_val)
        regime_log.append((date, breadth_regime))

        # Update peaks
        for sym, pos in positions.items():
            if sym in day_close and not np.isnan(day_close[sym]):
                if day_close[sym] > pos["peak_price"]:
                    pos["peak_price"] = day_close[sym]

        # ── EXITS ──
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
                if entry_ret <= MOM_STOP or entry_ret >= MOM_TP or days_held >= MOM_MAX_HOLD:
                    exit_triggered = True
                # Trailing stop: -8% from peak
                drop_from_peak = (curr / pos["peak_price"]) - 1.0 if pos["peak_price"] > 0 else 0
                if drop_from_peak <= -0.08:
                    exit_triggered = True
            elif strategy == "mean_reversion":
                if entry_ret <= MR_STOP or days_held >= MR_MAX_HOLD:
                    exit_triggered = True
                # Take profit: price recovers (simplified: +5% from entry)
                if entry_ret >= 0.05:
                    exit_triggered = True
            elif strategy == "mega_cap":
                if entry_ret <= MCAP_STOP or entry_ret >= MCAP_TP or days_held >= MCAP_MAX_HOLD:
                    exit_triggered = True

            if exit_triggered:
                to_close.append(sym)

        for sym in to_close:
            pos = positions.pop(sym)
            exit_px = day_open.get(sym, pos["entry_price"])
            if np.isnan(exit_px) or exit_px <= 0:
                exit_px = day_close.get(sym, pos["entry_price"])
            ret = (exit_px / pos["entry_price"]) - 1.0
            gross = pos["cost"] * (1.0 + ret)
            net = gross * (1.0 - SLIPPAGE)
            cash += net
            trade_ret = (net - pos["cost"]) / pos["cost"]
            trades.append({"ret": trade_ret, "strategy": pos["strategy"]})
            cooldown_days = MCAP_COOLDOWN if pos["strategy"] == "mega_cap" else COOLDOWN_DAYS
            cooldowns[f"{sym}:{pos['strategy']}"] = i + cooldown_days

        # ── ENTRIES ──
        if spy_regime == "BEARISH":
            # No new entries in bearish
            pass
        else:
            next_idx = i + 1
            next_open = open_lookup.get(all_dates[next_idx], {}) if next_idx < len(all_dates) else {}

            # Count current positions by strategy
            strat_counts = Counter(p["strategy"] for p in positions.values())
            total_pos = len(positions)

            # Sell idle SPY if needed
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

            def try_enter(sym, prob, strategy, slot_limit, slot_key):
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
                positions[sym] = {
                    "strategy": strategy, "cost": cost, "entry_price": entry_px,
                    "peak_price": entry_px, "entry_idx": next_idx, "exit_idx": exit_idx,
                }
                strat_counts[slot_key] = strat_counts.get(slot_key, 0) + 1
                total_pos += 1
                held_syms.add(sym)

            # 1. ML signals (2 slots)
            for sym, prob in ml_signals.get(date, []):
                if spy_regime == "CAUTIOUS" and strat_counts.get("ml", 0) >= 1:
                    continue
                try_enter(sym, prob, "ml", ML_TOP_N, "ml")

            # 2. Momentum signals (3 slots)
            # Compute momentum: top stocks by 63-day return, above SMA200
            if spy_regime != "BEARISH":
                mom_rets = {}
                for sym in day_close:
                    if sym == "SPY" or sym in held_syms:
                        continue
                    prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                    idx_loc = prices_list.index.get_indexer([date], method="pad")
                    if len(idx_loc) == 0 or idx_loc[0] < MOM_LOOKBACK:
                        continue
                    loc = idx_loc[0]
                    curr_px = prices_list.iloc[loc]
                    past_px = prices_list.iloc[loc - MOM_LOOKBACK]
                    if past_px > 0 and curr_px > 0:
                        # SMA200 filter
                        if loc >= 200:
                            sma200 = prices_list.iloc[loc-199:loc+1].mean()
                            if curr_px <= sma200:
                                continue
                        mom_rets[sym] = (curr_px / past_px) - 1.0

                mom_sorted = sorted(mom_rets.items(), key=lambda x: x[1], reverse=True)
                for sym, ret in mom_sorted[:MOM_TOP_N * 2]:  # try more to fill 3 slots
                    try_enter(sym, 0.7, "momentum", MOM_TOP_N, "momentum")

            # 3. Mean reversion signals (3 slots)
            if spy_regime != "BEARISH":
                mr_candidates = []
                for sym in day_close:
                    if sym == "SPY" or sym in held_syms:
                        continue
                    prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                    idx_loc = prices_list.index.get_indexer([date], method="pad")
                    if len(idx_loc) == 0 or idx_loc[0] < max(MR_DROP_PERIOD, 200):
                        continue
                    loc = idx_loc[0]
                    curr_px = prices_list.iloc[loc]
                    past_px = prices_list.iloc[loc - MR_DROP_PERIOD]
                    if past_px > 0 and curr_px > 0:
                        drop = (curr_px / past_px) - 1.0
                        if drop > MR_DROP_THRESH:
                            continue
                        # Above SMA200
                        sma200 = prices_list.iloc[loc-199:loc+1].mean()
                        if curr_px < sma200:
                            continue
                        # Not at 30-day low
                        recent_low = prices_list.iloc[loc-MR_DROP_PERIOD:loc+1].min()
                        if curr_px <= recent_low * 1.001:
                            continue
                        mr_candidates.append((sym, abs(drop)))

                mr_candidates.sort(key=lambda x: x[1], reverse=True)
                for sym, drop_mag in mr_candidates[:MR_TOP_N * 2]:
                    try_enter(sym, 0.6, "mean_reversion", MR_TOP_N, "mean_reversion")

            # 4. Mega-cap overlay (2 slots) — ALWAYS active
            if use_megacap_overlay:
                mcap_rets = {}
                for sym in MEGACAP_UNIVERSE:
                    if sym in held_syms:
                        continue
                    prices_list = close_data[sym].dropna() if sym in close_data.columns else pd.Series()
                    idx_loc = prices_list.index.get_indexer([date], method="pad")
                    if len(idx_loc) == 0 or idx_loc[0] < max(MCAP_LOOKBACK, 200):
                        continue
                    loc = idx_loc[0]
                    curr_px = prices_list.iloc[loc]
                    past_px = prices_list.iloc[loc - MCAP_LOOKBACK]
                    if past_px > 0 and curr_px > 0:
                        sma200 = prices_list.iloc[loc-199:loc+1].mean()
                        if curr_px <= sma200:
                            continue
                        mcap_rets[sym] = (curr_px / past_px) - 1.0

                mcap_sorted = sorted(mcap_rets.items(), key=lambda x: x[1], reverse=True)
                for sym, ret in mcap_sorted[:MCAP_TOP_N * 2]:
                    try_enter(sym, 0.7, "mega_cap", MCAP_TOP_N, "mega_cap")

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

    return pd.Series(port_vals, index=pd.DatetimeIndex(all_dates)), trades, regime_log


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
    return {"label": label, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
            "n_trades": len(trades), "win_rate": wins / len(trades) if trades else 0}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  REGIME-ADAPTIVE TRADING SYSTEM BACKTEST")
    log("  4 variants × 6 periods + breadth analysis")
    log("=" * 70)

    # ── Stage 1: Load features & train models ─────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 1: LOAD FEATURES & TRAIN MODELS")
    log("=" * 70)

    df = pd.read_parquet(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).copy()
    assert "days_until_earnings" not in df.columns

    # Rank target
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_v4"] = np.nan
    valid_df = df[sp500_mask & has_fwd].copy()
    valid_df["pct_rank"] = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_v4"] = (valid_df["pct_rank"] >= (1.0 - TOP_PERCENTILE)).astype(int)
    df.loc[valid_df.index, "target_v4"] = valid_df["target_v4"]

    # Cross-sectional features
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

    # LGBM
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

    # RF
    log(f"  Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=100,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train_imp, y_train)
    calib_rf = ManualCalibratedModel(rf, X_calib_imp, y_calib)
    auc_rf = roc_auc_score(y_calib, calib_rf.predict_proba(X_calib_imp)[:, 1])
    prob_rf = calib_rf.predict_proba(X_all_imp)[:, 1]
    log(f"    RF AUC={auc_rf:.4f}")

    # 50/50 ensemble
    df["prob_lgbm"] = prob_lgbm
    df["prob_rf"] = prob_rf
    df["prob_ensemble"] = 0.5 * prob_lgbm + 0.5 * prob_rf
    df["fwd_ret"] = df["fwd_10d_ret"]

    # ── Stage 2: Fetch OHLCV ──────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 2: FETCH OHLCV DATA")
    log("=" * 70)

    preds = df[["date", "symbol", "target_v4", "prob_ensemble", "prob_lgbm",
                "prob_rf", "fwd_ret", "in_sp500", "ret_60d"]].copy()
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

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    open_aligned = open_px.reindex(sim_index, method="ffill")
    spy_close_series = close_aligned["SPY"].dropna()

    log(f"  OHLCV: {len(close_aligned)} dates, {len(close_aligned.columns)} symbols")

    # ── Stage 3: Compute breadth series ───────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 3: COMPUTE MARKET BREADTH")
    log("=" * 70)

    # Use only S&P 500 stocks for breadth
    sp500_syms = [s for s in universe_syms if s in close_aligned.columns]
    breadth_close = close_aligned[sp500_syms].copy()
    breadth_series_full = compute_breadth_series(breadth_close, spy_close_series)

    valid_breadth = breadth_series_full.dropna()
    log(f"  Breadth computed for {len(valid_breadth)} dates")
    log(f"  Mean: {valid_breadth.mean():.1f}%  Median: {valid_breadth.median():.1f}%")
    log(f"  Min: {valid_breadth.min():.1f}%  Max: {valid_breadth.max():.1f}%")

    regimes = valid_breadth.apply(classify_regime)
    regime_counts = regimes.value_counts()
    log(f"  Regime distribution: {dict(regime_counts)}")

    # ── Stage 4: Run backtests for all periods × variants ─────────────────
    log("\n" + "=" * 70)
    log("STAGE 4: RUN BACKTESTS (6 periods × 4 variants)")
    log("=" * 70)

    all_results = []

    for p_start, p_end, p_label in PERIODS:
        p_start_ts = pd.Timestamp(p_start)
        p_end_ts = pd.Timestamp(p_end)
        period_dates = [d for d in all_dates
                        if pd.Timestamp(d) >= p_start_ts and pd.Timestamp(d) <= p_end_ts]
        if len(period_dates) < 20:
            log(f"\n  {p_label}: too few dates ({len(period_dates)}) — skipping")
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

        variants = [
            ("A: Baseline", False, False),
            ("B: Regime ML", True, False),
            ("C: +Mega-cap", False, True),
            ("D: Full", True, True),
        ]

        period_results = {}
        for v_label, use_regime, use_mcap in variants:
            vals, trades, regime_log = run_multi_strategy_backtest(
                period_preds, period_dates, spy_close_series,
                open_aligned, close_aligned, breadth_series_full,
                v_label, use_regime_adaptive_ml=use_regime, use_megacap_overlay=use_mcap)
            m = calc_metrics(vals, trades, v_label)
            m["period"] = p_label
            m["spy_cagr"] = spy_cagr
            m["spy_sharpe"] = spy_sharpe
            m["spy_dd"] = spy_dd
            all_results.append(m)
            period_results[v_label] = m

            # Count regime distribution for this period
            regime_dist = Counter(r for _, r in regime_log)

        # Print period summary
        log(f"\n  {'Variant':<20s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>8s} {'Win%':>7s}")
        log(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
        for v_label, _, _ in variants:
            m = period_results[v_label]
            log(f"  {m['label']:<20s} {m['cagr']*100:>7.2f}% {m['sharpe']:>7.3f} "
                f"{m['max_dd']*100:>7.1f}% {m['n_trades']:>7d} {m['win_rate']*100:>5.1f}%")
        log(f"  {'SPY':<20s} {spy_cagr*100:>7.2f}% {spy_sharpe:>7.3f} {spy_dd*100:>7.1f}%")

        # Regime distribution for this period
        period_breadth = breadth_series_full.reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in period_dates])).dropna()
        if len(period_breadth) > 0:
            pd_regimes = period_breadth.apply(classify_regime).value_counts()
            total = pd_regimes.sum()
            log(f"  Breadth regimes: " + " | ".join(f"{k}: {v/total*100:.0f}%" for k, v in pd_regimes.items()))

    # ── Stage 5: Breadth predictiveness analysis ──────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 5: BREADTH SIGNAL ANALYSIS")
    log("=" * 70)

    log("\n  Is breadth regime predictive of forward SPY returns?")
    log("  For each date, classify regime and measure SPY return over next 60 days.\n")

    fwd_spy_returns = {}
    spy_arr = spy_close_series.values
    spy_idx = spy_close_series.index

    for i in range(len(spy_idx) - 60):
        date = spy_idx[i]
        fwd_ret = (spy_arr[i + 60] / spy_arr[i]) - 1.0
        breadth_val = breadth_series_full.get(date, np.nan)
        if np.isnan(breadth_val):
            continue
        regime = classify_regime(breadth_val)
        if regime not in fwd_spy_returns:
            fwd_spy_returns[regime] = []
        fwd_spy_returns[regime].append(fwd_ret)

    log(f"  {'Regime':<12s} {'Count':>7s} {'Mean 60d':>10s} {'Median 60d':>12s} {'% Positive':>12s}")
    log(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*12} {'─'*12}")
    for regime in ["NARROW", "TRANSITION", "BROAD"]:
        rets = fwd_spy_returns.get(regime, [])
        if rets:
            arr = np.array(rets)
            log(f"  {regime:<12s} {len(rets):>7d} {arr.mean()*100:>9.2f}% {np.median(arr)*100:>11.2f}% "
                f"{(arr > 0).mean()*100:>10.1f}%")

    log("\n  Breadth regime by year:")
    for year in range(2012, 2027):
        yr_breadth = valid_breadth[(valid_breadth.index.year == year)]
        if len(yr_breadth) > 0:
            yr_regimes = yr_breadth.apply(classify_regime).value_counts()
            total = yr_regimes.sum()
            parts = " | ".join(f"{k}: {v/total*100:.0f}%" for k, v in yr_regimes.items())
            log(f"    {year}: mean={yr_breadth.mean():.1f}%  {parts}")

    # ── Stage 6: Pass criteria evaluation ─────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 6: PASS CRITERIA EVALUATION")
    log("=" * 70)

    # Find Variant D results
    holdout = next((r for r in all_results if r["period"] == "P5: 2023-2026" and r["label"] == "D: Full"), None)
    full = next((r for r in all_results if r["period"] == "Full: 2011-2026" and r["label"] == "D: Full"), None)

    pass_criteria = []

    if holdout:
        c1 = holdout["cagr"] >= 0.14
        pass_criteria.append(("Holdout CAGR >= 14%", c1, f"{holdout['cagr']*100:.2f}%"))
    else:
        pass_criteria.append(("Holdout CAGR >= 14%", False, "N/A"))

    if full:
        c2 = full["cagr"] >= 0.28
        pass_criteria.append(("Full period CAGR >= 28%", c2, f"{full['cagr']*100:.2f}%"))
    else:
        pass_criteria.append(("Full period CAGR >= 28%", False, "N/A"))

    # Max DD <= -22% in all periods
    variant_d = [r for r in all_results if r["label"] == "D: Full" and "Full" not in r["period"]]
    worst_dd = min(r["max_dd"] for r in variant_d) if variant_d else -1.0
    c3 = worst_dd >= -0.22
    pass_criteria.append(("Max DD <= -22% all periods", c3, f"{worst_dd*100:.1f}%"))

    # Beats baseline in 4/5 periods
    wins = 0
    for p_start, p_end, p_label in PERIODS[:5]:
        baseline = next((r for r in all_results if r["period"] == p_label and r["label"] == "A: Baseline"), None)
        full_sys = next((r for r in all_results if r["period"] == p_label and r["label"] == "D: Full"), None)
        if baseline and full_sys and full_sys["sharpe"] > baseline["sharpe"]:
            wins += 1
    c4 = wins >= 4
    pass_criteria.append(("Beats baseline in >= 4/5 periods", c4, f"{wins}/5"))

    log("\n  Pass Criteria:")
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
        log("\n  All criteria PASS. Recommend deploying Variant D (full system).")
        log("  Regime-adaptive ML + mega-cap overlay.")
    else:
        log("\n  NOT all criteria passed.")

        # Check partial deployment options
        # Option 1: overlay only (Variant C)
        c_holdout = next((r for r in all_results if r["period"] == "P5: 2023-2026" and r["label"] == "C: +Mega-cap"), None)
        c_full = next((r for r in all_results if r["period"] == "Full: 2011-2026" and r["label"] == "C: +Mega-cap"), None)

        if c_holdout and c_full:
            log(f"\n  Partial option — Variant C (overlay only, no regime switch):")
            log(f"    Holdout CAGR: {c_holdout['cagr']*100:.2f}%  Sharpe: {c_holdout['sharpe']:.3f}")
            log(f"    Full CAGR: {c_full['cagr']*100:.2f}%  Sharpe: {c_full['sharpe']:.3f}")

        # Option 2: regime ML only (Variant B)
        b_holdout = next((r for r in all_results if r["period"] == "P5: 2023-2026" and r["label"] == "B: Regime ML"), None)
        b_full = next((r for r in all_results if r["period"] == "Full: 2011-2026" and r["label"] == "B: Regime ML"), None)

        if b_holdout and b_full:
            log(f"\n  Partial option — Variant B (regime ML only, no overlay):")
            log(f"    Holdout CAGR: {b_holdout['cagr']*100:.2f}%  Sharpe: {b_holdout['sharpe']:.3f}")
            log(f"    Full CAGR: {b_full['cagr']*100:.2f}%  Sharpe: {b_full['sharpe']:.3f}")

        # Best variant on holdout
        holdout_variants = [r for r in all_results if r["period"] == "P5: 2023-2026"]
        if holdout_variants:
            best = max(holdout_variants, key=lambda x: x["sharpe"])
            log(f"\n  Best on holdout: {best['label']} (Sharpe={best['sharpe']:.3f}, CAGR={best['cagr']*100:.2f}%)")

        log("\n  If no variant passes: honest conclusion that cross-sectional ML stock picking")
        log("  underperforms in concentration-driven markets regardless of regime adaptation.")

    # ── Comparison table ──────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("FULL COMPARISON TABLE")
    log("=" * 70)

    log(f"\n  {'Period':<20s} {'Variant':<20s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>7s} {'Win%':>6s} {'SPY':>8s}")
    log(f"  {'─'*20} {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*6} {'─'*8}")

    for p_start, p_end, p_label in PERIODS:
        period_res = [r for r in all_results if r["period"] == p_label]
        for m in period_res:
            log(f"  {m['period']:<20s} {m['label']:<20s} {m['cagr']*100:>7.2f}% {m['sharpe']:>7.3f} "
                f"{m['max_dd']*100:>7.1f}% {m['n_trades']:>6d} {m['win_rate']*100:>5.1f}% "
                f"{m.get('spy_cagr', 0)*100:>7.2f}%")

    # ── Which variant wins per period ─────────────────────────────────────
    log(f"\n  Winner by period (highest Sharpe):")
    for p_start, p_end, p_label in PERIODS:
        period_res = [r for r in all_results if r["period"] == p_label]
        if period_res:
            best = max(period_res, key=lambda x: x["sharpe"])
            log(f"    {p_label}: {best['label']} (Sharpe={best['sharpe']:.3f})")

    elapsed = time.perf_counter() - t0
    log(f"\n{'='*70}")
    log(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
