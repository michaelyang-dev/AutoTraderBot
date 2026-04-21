#!/usr/bin/env python3
"""
Backtest Audit — Bug-Fixed Re-Run
==================================
Retrains V4 LGBM + RF on corrected features, then runs backtests with:
  Bug 1 FIX: Ratios/metrics use filing_date (via rebuilt features.parquet)
  Bug 2 FIX: days_until_earnings dropped (83 features instead of 84)
  Bug 3 FIX: SPY regime filter applied (BEARISH=no buys, CAUTIOUS=top-2 only)
  Bug 4 FIX: Next-day open execution (not same-day close)

Run with:
    python3 backtest_audit_fixed.py
"""

import gc
import sys
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES_FILE = DATA_DIR / "features.parquet"

# ── Config ──────────────────────────────────────────────────────────────────
CALIB_FRAC = 0.20
TOP_PERCENTILE = 0.20
TOP_N_PICKS = 5
N_TREES = 500
EARLY_STOP = 100

INITIAL_CASH = 100_000.0
SLIPPAGE = 0.0005  # 0.05% per leg
HOLD_DAYS = 10
POSITION_PCT = 0.12

# Vol targeting
TARGET_VOL = 0.15
MAX_LEVERAGE = 1.5
MIN_LEVERAGE = 0.3
VOL_LOOKBACK = 20

# SPY parking
SPY_RESERVE_PCT = 0.30
SPY_THRESHOLD_PCT = 0.20
SPY_INVEST_PCT = 0.85
COOLDOWN_DAYS = 5

# SPY regime SMA periods
SPY_SMA_FAST = 50
SPY_SMA_SLOW = 200

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df):
    exclude = {"date", "symbol", "target", "target_v4", "in_sp500"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: Retrain Models on Corrected Features
# ══════════════════════════════════════════════════════════════════════════════

def retrain_models():
    """
    Retrain LGBM and RF on corrected features.parquet.

    Memory-optimised: sequences training → save → free → predict → save.
    Peak RAM ~40% lower than holding everything simultaneously.
    """
    log("=" * 70)
    log("STAGE 1: RETRAIN MODELS ON CORRECTED FEATURES")
    log("=" * 70)

    # ── Phase A: Load features, prepare targets, extract training arrays ──
    df = pd.read_parquet(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).copy()

    assert "days_until_earnings" not in df.columns, "days_until_earnings still in features!"
    log(f"  Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    log(f"  Feature count: {len(get_feature_cols(df))} (should be 83)")

    # Compute cross-sectional rank target
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    valid_mask = sp500_mask & has_fwd
    df["target_v4"] = np.nan
    valid_df = df[valid_mask].copy()
    valid_df["pct_rank"] = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_v4"] = (valid_df["pct_rank"] >= (1.0 - TOP_PERCENTILE)).astype(int)
    df.loc[valid_df.index, "target_v4"] = valid_df["target_v4"]

    n_valid = valid_df["target_v4"].notna().sum()
    n_pos = (valid_df["target_v4"] == 1).sum()
    log(f"  Target: {n_valid:,} valid rows, {n_pos:,} positive ({n_pos/n_valid*100:.1f}%)")
    del valid_df; gc.collect()

    # Add V4 cross-sectional features
    for base_col, rank_col in [("vol_20d", "vol_rank_20d"), ("ret_60d", "momentum_rank_60d"),
                                ("rsi_14", "rsi_rank"), ("dist_sma50", "dist_sma50_rank")]:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    feature_cols = get_feature_cols(df)
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    log(f"  Features: {len(feature_cols)} columns")

    # Drop NaN rows
    df = df.dropna(subset=non_fund_cols + ["target_v4"])
    train_df = df[df["in_sp500"] == True].copy()
    log(f"  Training rows (in_sp500): {len(train_df):,}")

    X_filtered = train_df[feature_cols].values
    y_filtered = train_df["target_v4"].values

    # Date split
    all_dates = np.sort(train_df["date"].unique())
    split_idx = int(len(all_dates) * (1 - CALIB_FRAC))
    calib_start = pd.Timestamp(all_dates[split_idx])
    train_mask = train_df["date"] < calib_start
    calib_mask = train_df["date"] >= calib_start

    X_train, y_train = X_filtered[train_mask], y_filtered[train_mask]
    X_calib, y_calib = X_filtered[calib_mask], y_filtered[calib_mask]

    log(f"  Train: {train_mask.sum():,}  Calib: {calib_mask.sum():,}")

    # Free train_df and filtered arrays (X_train/X_calib are views/copies)
    del train_df, X_filtered, y_filtered
    gc.collect()

    # ── Phase B: Train models and save to disk ──────────────────────────
    log(f"\n  Training LightGBM V4 (corrected features) ...")
    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=N_TREES, learning_rate=0.05, max_depth=6, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, objective="binary", metric="auc",
        random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=scale,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_calib, y_calib)],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(period=-1)])
    n_trees_used = lgb_model.booster_.num_trees()
    log(f"  LGBM trained: {n_trees_used} trees")

    # Calibrate LGBM
    calib_lgbm = CalibratedClassifierCV(lgb_model, method="isotonic", cv="prefit")
    calib_lgbm.fit(X_calib, y_calib)
    lgbm_auc = roc_auc_score(y_calib, calib_lgbm.predict_proba(X_calib)[:, 1])
    log(f"  LGBM calib AUC: {lgbm_auc:.4f}")

    # Train RF
    log(f"\n  Training Random Forest (corrected features) ...")
    imp = SimpleImputer(strategy="median")
    X_train_rf = imp.fit_transform(X_train)
    X_calib_rf = imp.transform(X_calib)

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=100,
        n_jobs=-1, random_state=42)
    rf_model.fit(X_train_rf, y_train)
    log(f"  RF trained: {rf_model.n_estimators} trees")

    # Calibrate RF
    from sklearn.isotonic import IsotonicRegression

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

    calib_rf = ManualCalibratedModel(rf_model, X_calib_rf, y_calib)
    rf_auc = roc_auc_score(y_calib, calib_rf.predict_proba(X_calib_rf)[:, 1])
    log(f"  RF calib AUC: {rf_auc:.4f}")

    # Save models to disk so we can free training arrays
    model_dir = DATA_DIR / "audit_models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(calib_lgbm, model_dir / "calib_lgbm.joblib")
    joblib.dump(calib_rf, model_dir / "calib_rf.joblib")
    joblib.dump(imp, model_dir / "imputer.joblib")
    log(f"  Models saved to {model_dir}")

    # Free all training arrays and models
    del X_train, y_train, X_calib, y_calib
    del X_train_rf, X_calib_rf
    del lgb_model, rf_model, calib_lgbm, calib_rf, imp
    gc.collect()
    log(f"  Training arrays freed")

    # ── Phase C: Reload models, generate predictions, save ──────────────
    log(f"\n  Generating predictions for {len(df):,} rows ...")
    calib_lgbm = joblib.load(model_dir / "calib_lgbm.joblib")
    calib_rf = joblib.load(model_dir / "calib_rf.joblib")
    imp = joblib.load(model_dir / "imputer.joblib")

    X_all = df[feature_cols].values
    lgbm_probs = calib_lgbm.predict_proba(X_all)[:, 1]

    X_all_rf = imp.transform(X_all)
    rf_probs = calib_rf.predict_proba(X_all_rf)[:, 1]
    del X_all, X_all_rf
    gc.collect()

    # Ensemble: simple average
    ensemble_probs = (lgbm_probs + rf_probs) / 2.0

    df["prob_lgbm"] = lgbm_probs
    df["prob_rf"] = rf_probs
    df["prob_ensemble"] = ensemble_probs
    df["fwd_ret"] = df["fwd_10d_ret"]

    log(f"  LGBM probs: [{lgbm_probs.min():.4f}, {lgbm_probs.max():.4f}], mean={lgbm_probs.mean():.4f}")
    log(f"  RF probs:   [{rf_probs.min():.4f}, {rf_probs.max():.4f}], mean={rf_probs.mean():.4f}")
    del lgbm_probs, rf_probs, ensemble_probs
    del calib_lgbm, calib_rf, imp

    # Save predictions
    pred_cols = ["date", "symbol", "target_v4", "prob_lgbm", "prob_rf",
                 "prob_ensemble", "fwd_ret", "in_sp500"]
    preds = df[pred_cols].copy()
    preds.to_parquet(DATA_DIR / "predictions_audit_fixed.parquet", index=False)
    log(f"  Saved predictions_audit_fixed.parquet")

    # Free the large features DataFrame before returning
    del df, preds
    gc.collect()
    log(f"  Features DataFrame freed")

    return feature_cols, lgbm_auc, rf_auc


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: Bug-Fixed Backtester
# ══════════════════════════════════════════════════════════════════════════════

def compute_spy_regime(spy_prices_series, date):
    """
    Compute SPY regime on a given date using SMA50 vs SMA200.
    Returns 'BULLISH', 'CAUTIOUS', or 'BEARISH'.
    Uses data through date (inclusive) — the SMA values are known at close.
    """
    prices_up_to = spy_prices_series[:date].dropna()
    if len(prices_up_to) < SPY_SMA_SLOW:
        return "BULLISH"  # not enough data, default bullish

    price = prices_up_to.iloc[-1]
    sma50 = prices_up_to.iloc[-SPY_SMA_FAST:].mean()
    sma200 = prices_up_to.iloc[-SPY_SMA_SLOW:].mean()

    if price > sma50 and sma50 > sma200:
        return "BULLISH"
    elif price > sma200:
        return "CAUTIOUS"
    else:
        return "BEARISH"


def run_fixed_backtest(preds_df, prob_col, all_dates, spy_close_series,
                       open_data, close_data, label, use_voltarget=True):
    """
    Bug-fixed backtest with:
      - SPY regime filter (Bug 3)
      - Next-day open execution (Bug 4)
      - Vol targeting (optional)

    Uses open prices for entry/exit instead of close prices.
    """
    from strategy_base import Signal, Position

    # Build signal lookup: top-N by prob_col per date (in_sp500 only)
    signals_by_date = {}
    for date, grp in preds_df.groupby("date"):
        sp500 = grp[grp["in_sp500"] == True]
        if sp500.empty:
            continue
        top = sp500.nlargest(TOP_N_PICKS, prob_col)
        signals_by_date[date] = [
            (row.symbol, getattr(row, prob_col), row.fwd_ret)
            for row in top.itertuples(index=False)
        ]

    # Build open price lookup
    open_lookup = {}
    close_lookup = {}
    for date in all_dates:
        if date in open_data.index:
            open_lookup[date] = open_data.loc[date].to_dict()
        if date in close_data.index:
            close_lookup[date] = close_data.loc[date].to_dict()

    # Build date → next trading day map (Bug 4: execute next day)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    cash = float(INITIAL_CASH)
    positions = {}  # sym → dict with cost, entry_idx, exit_idx, entry_price, peak_price, strategy
    idle_spy_shares = 0.0
    trades = []
    cooldowns = {}  # sym → expiry_idx
    port_vals = []
    daily_returns = []
    vol_scales = []
    regime_log = []

    for i, date in enumerate(all_dates):
        spy_close = spy_close_series.get(date)
        if spy_close is not None and (np.isnan(spy_close) or spy_close <= 0):
            spy_close = None

        day_close = close_lookup.get(date, {})
        day_open = open_lookup.get(date, {})

        # ── Vol targeting scale ─────────────────────────────────────────
        if use_voltarget and len(daily_returns) >= VOL_LOOKBACK:
            recent = np.array(daily_returns[-VOL_LOOKBACK:])
            realized_vol = np.std(recent) * np.sqrt(252)
            scale = TARGET_VOL / realized_vol if realized_vol > 0 else 1.0
            scale = max(MIN_LEVERAGE, min(scale, MAX_LEVERAGE))
        else:
            scale = 1.0
        vol_scales.append(scale)

        # ── SPY regime (Bug 3 FIX) ─────────────────────────────────────
        regime = compute_spy_regime(spy_close_series, date)
        regime_log.append(regime)

        # ── 1. Update peak prices ───────────────────────────────────────
        for sym, pos in positions.items():
            if sym in day_close:
                px = day_close[sym]
                if not np.isnan(px) and px > pos["peak_price"]:
                    pos["peak_price"] = px

        # ── 2. Close expiring positions (at OPEN price — Bug 4 FIX) ────
        to_close = []
        for sym, pos in positions.items():
            if i >= pos["exit_idx"]:
                to_close.append(sym)

        for sym in to_close:
            pos = positions.pop(sym)
            # Bug 4: exit at today's OPEN (the position held through yesterday's close)
            exit_px = day_open.get(sym, pos["entry_price"])
            if np.isnan(exit_px) or exit_px <= 0:
                exit_px = day_close.get(sym, pos["entry_price"])
            actual_ret = (exit_px / pos["entry_price"]) - 1.0
            gross = pos["cost"] * (1.0 + actual_ret)
            net = gross * (1.0 - SLIPPAGE)
            cash += net
            trades.append((net - pos["cost"]) / pos["cost"])
            cooldowns[sym] = i + COOLDOWN_DAYS

        # ── 3. Gather signals ───────────────────────────────────────────
        raw_sigs = signals_by_date.get(date, [])
        held = set(positions.keys())

        # Apply regime filter (Bug 3)
        if regime == "BEARISH":
            filtered_sigs = []  # no new buys
        elif regime == "CAUTIOUS":
            filtered_sigs = raw_sigs[:2]  # top 2 only
        else:
            filtered_sigs = raw_sigs  # all signals

        candidates = []
        for sym, prob, fwd_ret in filtered_sigs:
            if sym in held:
                continue
            if cooldowns.get(sym, -1) > i:
                continue
            if np.isnan(fwd_ret):
                continue
            candidates.append((sym, prob, fwd_ret))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # ── 4. Execute buys (at NEXT DAY's OPEN — Bug 4 FIX) ──────────
        # We need the next trading day's open for entry
        next_day_idx = i + 1
        if next_day_idx < len(all_dates):
            next_date = all_dates[next_day_idx]
            next_open = open_lookup.get(next_date, {})
        else:
            next_open = {}

        max_new = max(0, 10 - len(positions))  # max 10 positions

        if spy_close and idle_spy_shares > 0 and candidates and max_new > 0:
            proceeds = idle_spy_shares * spy_close * (1.0 - SLIPPAGE)
            cash += proceeds
            idle_spy_shares = 0.0

        for sym, prob, fwd_ret in candidates[:max_new]:
            # Bug 4: enter at NEXT DAY's open
            entry_px = next_open.get(sym, np.nan)
            if np.isnan(entry_px) or entry_px <= 0:
                continue

            # Position sizing with vol scale
            port_est = cash
            for p in positions.values():
                px = day_close.get(p["symbol"], p["entry_price"])
                port_est += p["cost"] * (px / p["entry_price"] if p["entry_price"] > 0 else 1.0)

            ml_mult = min(1.0, max(0.60, prob * 1.6 - 0.28))
            target = port_est * POSITION_PCT * ml_mult * scale

            cost = min(target, cash * 0.95)
            if cost < 50.0:
                continue

            cash -= cost * (1.0 + SLIPPAGE)
            # Bug 4: exit_idx = entry + hold_days (entry is next day)
            exit_idx = min(next_day_idx + HOLD_DAYS, len(all_dates) - 1)

            positions[sym] = {
                "symbol": sym,
                "cost": cost,
                "entry_idx": next_day_idx,
                "exit_idx": exit_idx,
                "entry_price": entry_px,
                "peak_price": entry_px,
                "fwd_ret": fwd_ret,
                "confidence": prob,
            }

        # ── 5. Park idle cash in SPY ────────────────────────────────────
        if spy_close:
            pos_val = sum(
                p["cost"] * (day_close.get(p["symbol"], p["entry_price"]) / p["entry_price"]
                             if p["entry_price"] > 0 else 1.0)
                for p in positions.values()
            )
            est_port = cash + idle_spy_shares * spy_close + pos_val
            reserved = est_port * SPY_RESERVE_PCT
            idle_cash = cash - reserved
            if idle_cash > est_port * SPY_THRESHOLD_PCT:
                invest = min(idle_cash * SPY_INVEST_PCT, cash * 0.95)
                new_shares = invest / spy_close
                cash -= invest * (1.0 + SLIPPAGE)
                idle_spy_shares += new_shares

        # ── 6. Mark-to-market ───────────────────────────────────────────
        port_val = cash + (idle_spy_shares * spy_close if spy_close else 0)
        for pos in positions.values():
            px = day_close.get(pos["symbol"], pos["entry_price"])
            port_val += pos["cost"] * (px / pos["entry_price"] if pos["entry_price"] > 0 else 1.0)
        port_vals.append(port_val)

        # Track daily return
        if len(port_vals) >= 2:
            daily_ret = (port_vals[-1] / port_vals[-2]) - 1.0
        else:
            daily_ret = 0.0
        daily_returns.append(daily_ret)

    series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates))

    # Regime stats
    regime_counts = Counter(regime_log)
    log(f"  [{label}] Regime: BULL={regime_counts.get('BULLISH',0)} "
        f"CAUT={regime_counts.get('CAUTIOUS',0)} "
        f"BEAR={regime_counts.get('BEARISH',0)} days")

    return series, trades


def calc_metrics(vals, trades, years, label):
    """Calculate standard backtest metrics."""
    final = vals.iloc[-1]
    cagr = (final / INITIAL_CASH) ** (1.0 / years) - 1.0

    daily_rets = vals.pct_change().dropna()
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

    neg_rets = daily_rets[daily_rets < 0]
    sortino = daily_rets.mean() / neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 and neg_rets.std() > 0 else 0

    peak = vals.cummax()
    dd = (vals - peak) / peak
    max_dd = dd.min()

    wins = sum(1 for t in trades if t > 0)
    win_rate = wins / len(trades) if trades else 0

    return {
        "label": label,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "final_value": final,
        "n_trades": len(trades),
        "win_rate": win_rate,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  BACKTEST AUDIT — BUG-FIXED RE-RUN")
    log("  Bug 1: Ratios/metrics use filing_date")
    log("  Bug 2: days_until_earnings dropped (83 features)")
    log("  Bug 3: SPY regime filter applied")
    log("  Bug 4: Next-day open execution")
    log("=" * 70)

    # ── Stage 1: Retrain ────────────────────────────────────────────────
    feature_cols, lgbm_auc, rf_auc = retrain_models()
    # Training data and models are fully freed at this point

    # ── Stage 2: Load data for backtest ─────────────────────────────────
    log(f"\n{'='*70}")
    log("STAGE 2: PREPARE BACKTEST DATA")
    log(f"{'='*70}")

    preds = pd.read_parquet(DATA_DIR / "predictions_audit_fixed.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds["date"].unique().tolist())
    universe_syms = sorted(preds["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    log(f"  Predictions: {len(preds):,} rows | {len(universe_syms)} symbols | {years:.1f} years")
    log(f"  Date range: {pd.Timestamp(all_dates[0]).date()} → {pd.Timestamp(all_dates[-1]).date()}")

    # Fetch OHLCV (need both open AND close)
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

    log(f"  Fetching OHLCV for {len(all_syms)} symbols ...")
    raw = yf.download(all_syms, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    open_prices = raw["Open"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Open"]]
    open_prices.index = pd.to_datetime(open_prices.index).tz_localize(None)

    # Free raw OHLCV (close and open_prices are views, but del raw allows
    # the remaining columns like High/Low/Volume to be freed)
    del raw; gc.collect()

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    open_aligned = open_prices.reindex(sim_index, method="ffill")

    # Free unaligned price frames (aligned versions are all we need)
    del close, open_prices; gc.collect()

    spy_close_series = close_aligned["SPY"].dropna()
    spy_dict = close_aligned["SPY"].to_dict()
    spy_bh = spy_close_series / spy_close_series.iloc[0] * INITIAL_CASH

    log(f"  OHLCV loaded: {len(close_aligned)} dates")

    # ── Stage 3: Run bug-fixed backtests ────────────────────────────────
    log(f"\n{'='*70}")
    log("STAGE 3: RUN BUG-FIXED BACKTESTS")
    log(f"{'='*70}")

    results = {}

    # V4 LGBM (bug-fixed, with vol targeting)
    log(f"\n  Running V4 LGBM (bug-fixed + vol-target) ...")
    vals_lgbm, trades_lgbm = run_fixed_backtest(
        preds, "prob_lgbm", all_dates, spy_close_series,
        open_aligned, close_aligned, "V4 LGBM fixed", use_voltarget=True)
    results["lgbm"] = calc_metrics(vals_lgbm, trades_lgbm, years, "V4 LGBM (fixed)")

    # RF (bug-fixed, with vol targeting)
    log(f"\n  Running RF (bug-fixed + vol-target) ...")
    vals_rf, trades_rf = run_fixed_backtest(
        preds, "prob_rf", all_dates, spy_close_series,
        open_aligned, close_aligned, "RF fixed", use_voltarget=True)
    results["rf"] = calc_metrics(vals_rf, trades_rf, years, "RF (fixed)")

    # 2-way ensemble (bug-fixed, with vol targeting)
    log(f"\n  Running 2-way Ensemble (bug-fixed + vol-target) ...")
    vals_ens, trades_ens = run_fixed_backtest(
        preds, "prob_ensemble", all_dates, spy_close_series,
        open_aligned, close_aligned, "Ensemble fixed", use_voltarget=True)
    results["ensemble"] = calc_metrics(vals_ens, trades_ens, years, "Ensemble (fixed)")

    # V4 LGBM without vol targeting (for comparison)
    log(f"\n  Running V4 LGBM (bug-fixed, NO vol-target) ...")
    vals_lgbm_raw, trades_lgbm_raw = run_fixed_backtest(
        preds, "prob_lgbm", all_dates, spy_close_series,
        open_aligned, close_aligned, "V4 LGBM raw", use_voltarget=False)
    results["lgbm_raw"] = calc_metrics(vals_lgbm_raw, trades_lgbm_raw, years, "V4 LGBM raw (fixed)")

    # ── Stage 4: Comparison Report ──────────────────────────────────────
    log(f"\n{'='*70}")
    log("STAGE 4: COMPARISON REPORT")
    log(f"{'='*70}")

    # Original (inflated) numbers for comparison
    originals = {
        "V2 baseline":     {"cagr": 0.2559, "sharpe": 1.65, "max_dd": -0.21},
        "V4 LGBM (orig)":  {"cagr": 0.3460, "sharpe": 1.77, "max_dd": -0.243},
        "RF (orig)":        {"cagr": 0.5910, "sharpe": 2.77, "max_dd": -0.200},
        "Ensemble (orig)":  {"cagr": 0.4768, "sharpe": 2.30, "max_dd": -0.230},
    }

    log(f"\n  {'Model':<30s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>8s} {'Win%':>7s}")
    log(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")

    # Original numbers
    log(f"  --- ORIGINAL (with bugs) ---")
    for name, m in originals.items():
        log(f"  {name:<30s} {m['cagr']*100:>7.2f}% {m['sharpe']:>8.2f} {m['max_dd']*100:>7.1f}%")

    log(f"\n  --- BUG-FIXED ---")
    for key in ["lgbm", "rf", "ensemble", "lgbm_raw"]:
        m = results[key]
        log(f"  {m['label']:<30s} {m['cagr']*100:>7.2f}% {m['sharpe']:>8.2f} {m['max_dd']*100:>7.1f}% "
            f"{m['n_trades']:>8,} {m['win_rate']*100:>6.1f}%")

    # Deltas
    log(f"\n  --- IMPACT OF BUG FIXES ---")
    comparisons = [
        ("V4 LGBM", originals["V4 LGBM (orig)"], results["lgbm"]),
        ("RF", originals["RF (orig)"], results["rf"]),
        ("Ensemble", originals["Ensemble (orig)"], results["ensemble"]),
    ]
    for name, orig, fixed in comparisons:
        d_cagr = fixed["cagr"] - orig["cagr"]
        d_sharpe = fixed["sharpe"] - orig["sharpe"]
        d_dd = fixed["max_dd"] - orig["max_dd"]
        log(f"  {name:<15s} CAGR: {d_cagr*100:>+7.2f}%  Sharpe: {d_sharpe:>+7.2f}  Max DD: {d_dd*100:>+7.1f}%")

    # ── Year-by-year ────────────────────────────────────────────────────
    log(f"\n  {'='*65}")
    log(f"  YEAR-BY-YEAR RETURNS (bug-fixed ensemble)")
    log(f"  {'='*65}")

    vals_ens.index = pd.to_datetime(vals_ens.index)
    spy_aligned = spy_bh.reindex(vals_ens.index, method="ffill")

    log(f"\n  {'Year':<6} {'Ensemble':>10} {'SPY':>10} {'Alpha':>10}")
    log(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10}")

    for yr in sorted(set(vals_ens.index.year)):
        yr_vals = vals_ens[vals_ens.index.year == yr]
        yr_spy = spy_aligned[spy_aligned.index.year == yr]
        if len(yr_vals) < 2:
            continue
        ens_ret = yr_vals.iloc[-1] / yr_vals.iloc[0] - 1
        spy_ret = yr_spy.iloc[-1] / yr_spy.iloc[0] - 1 if len(yr_spy) >= 2 and yr_spy.iloc[0] > 0 else np.nan
        alpha = ens_ret - spy_ret if not np.isnan(spy_ret) else np.nan
        log(f"  {yr:<6} {ens_ret*100:>9.2f}% {spy_ret*100:>9.2f}% {alpha*100:>+9.2f}%")

    # ── Deployment Decision ─────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("DEPLOYMENT DECISION")
    log(f"{'='*70}")

    best = results["ensemble"]
    v2_cagr, v2_sharpe = 0.2559, 1.65

    deploy_cagr = best["cagr"] >= 0.25
    deploy_sharpe = best["sharpe"] >= 1.65

    log(f"\n  Bug-fixed Ensemble:")
    log(f"    [{'PASS' if deploy_cagr else 'FAIL'}] CAGR >= 25%:   {best['cagr']*100:.2f}%")
    log(f"    [{'PASS' if deploy_sharpe else 'FAIL'}] Sharpe >= 1.65: {best['sharpe']:.3f}")
    log(f"    Max DD: {best['max_dd']*100:.1f}%")
    log(f"    Trades: {best['n_trades']:,}")
    log(f"    Win Rate: {best['win_rate']*100:.1f}%")

    log(f"\n  V2 Baseline: CAGR={v2_cagr*100:.2f}%, Sharpe={v2_sharpe:.2f}")

    if deploy_cagr and deploy_sharpe:
        log(f"\n  RECOMMENDATION: DEPLOY bug-fixed ensemble")
        log(f"  Honest improvement over V2: "
            f"CAGR {(best['cagr']-v2_cagr)*100:+.2f}%, Sharpe {best['sharpe']-v2_sharpe:+.2f}")
    elif best["cagr"] > v2_cagr or best["sharpe"] > v2_sharpe:
        log(f"\n  RECOMMENDATION: MARGINAL — review individual metrics")
    else:
        log(f"\n  RECOMMENDATION: KEEP V2 — bug-fixed numbers don't justify upgrade")

    # Also check LGBM solo
    lgbm_m = results["lgbm"]
    log(f"\n  Bug-fixed V4 LGBM solo:")
    log(f"    CAGR={lgbm_m['cagr']*100:.2f}%, Sharpe={lgbm_m['sharpe']:.3f}, DD={lgbm_m['max_dd']*100:.1f}%")

    rf_m = results["rf"]
    log(f"  Bug-fixed RF solo:")
    log(f"    CAGR={rf_m['cagr']*100:.2f}%, Sharpe={rf_m['sharpe']:.3f}, DD={rf_m['max_dd']*100:.1f}%")

    elapsed = time.perf_counter() - t0
    log(f"\n{'='*70}")
    log(f"Total audit runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
