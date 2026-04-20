#!/usr/bin/env python3
"""
V4 Cross-Sectional Ranking Model
=================================
NEW model (not a retrofit of V3). Replaces absolute return prediction with
cross-sectional ranking within the S&P 500 universe.

Key difference from V3:
  - V3 target: fwd_10d_ret > 2% (absolute) → lumpy trades, regime-dependent
  - V4 target: stock in top 20% of S&P 500 by fwd_10d_ret (relative) → stable

Backtest uses top-N selection (pick top 5 stocks by prediction) instead of
threshold-based selection. This guarantees consistent trade frequency.

Pipeline:
  1. Load features.parquet
  2. Compute cross-sectional rank target per date (in-S&P500 only)
  3. Add new cross-sectional rank features
  4. Train calibrated LightGBM (80/20 date split, isotonic calibration)
  5. Generate predictions for all rows → predictions_v4.parquet
  6. Verify model/predictions consistency
  7. Run Path B backtest with top-N selection
  8. Generate full comparison report

Run with:
    python3 train_v4_ranking.py
"""

import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).resolve().parent / "data"
INPUT_FILE   = DATA_DIR / "features.parquet"
MODEL_FILE   = DATA_DIR / "model_v4.lgb"
PRED_FILE    = DATA_DIR / "predictions_v4.parquet"

CALIB_FRAC   = 0.20   # last 20% of dates for calibration
TOP_PERCENTILE = 0.20  # top 20% = target 1

N_TREES = 500
EARLY_STOP_ROUNDS = 100

LGB_PARAMS = dict(
    n_estimators      = N_TREES,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 31,
    min_child_samples = 50,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    objective         = "binary",
    metric            = "auc",
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

# Fundamental feature columns (LightGBM handles NaN natively for ETFs)
FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

# V4 deployment criteria
V4_MIN_CAGR   = 0.27
V4_MIN_SHARPE = 1.60
V4_MAX_DD     = -0.25

# Prior results for comparison
PRIOR_RESULTS = {
    "V2 (60-stock)":              {"cagr": 0.2559, "sharpe": 1.65, "max_dd": -0.21},
    "V3 raw (500-stock thresh)":  {"cagr": 0.3280, "sharpe": 1.52, "max_dd": -0.34},
    "V3 + VolTgt":                {"cagr": 0.2865, "sharpe": 1.47, "max_dd": -0.26},
}

# Backtest top-N selection
TOP_N_PICKS = 5   # pick top 5 stocks per day by V4 prediction


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target", "target_v4", "in_sp500"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Compute cross-sectional rank target
# ══════════════════════════════════════════════════════════════════════════════

def compute_rank_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, among in-S&P500 stocks:
      1. Compute fwd_10d_ret = shift(ret_10d, -10) per symbol
      2. Rank stocks by fwd_10d_ret
      3. Target = 1 if stock in top 20%, else 0
    """
    log("Computing cross-sectional rank target ...")

    # Forward return: shift ret_10d by -10 within each symbol
    df = df.sort_values(["symbol", "date"]).copy()
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)

    # Only compute rank target for in-S&P500 rows with valid fwd returns
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    valid_mask = sp500_mask & has_fwd

    # Initialize target_v4 as NaN
    df["target_v4"] = np.nan

    # For each date, rank within S&P 500 stocks
    valid_df = df[valid_mask].copy()
    valid_df["pct_rank"] = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_v4"] = (valid_df["pct_rank"] >= (1.0 - TOP_PERCENTILE)).astype(int)

    # Write back
    df.loc[valid_df.index, "target_v4"] = valid_df["target_v4"]

    # Stats
    n_valid = valid_df["target_v4"].notna().sum()
    n_pos = (valid_df["target_v4"] == 1).sum()
    n_neg = (valid_df["target_v4"] == 0).sum()
    log(f"  Valid rows with rank target: {n_valid:,}")
    log(f"  Positive (top {TOP_PERCENTILE:.0%}): {n_pos:,} ({n_pos/n_valid*100:.1f}%)")
    log(f"  Negative: {n_neg:,} ({n_neg/n_valid*100:.1f}%)")

    # Check per-date distribution
    date_counts = valid_df.groupby("date")["target_v4"].agg(["sum", "count"])
    log(f"  Avg stocks per date: {date_counts['count'].mean():.0f}")
    log(f"  Avg positive per date: {date_counts['sum'].mean():.1f}")
    log(f"  Min stocks on a date: {date_counts['count'].min()}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Add new cross-sectional features
# ══════════════════════════════════════════════════════════════════════════════

def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vol_rank_20d, momentum_rank_60d, rsi_rank, dist_sma50_rank."""
    log("Adding new cross-sectional rank features ...")

    new_features = []

    # vol_rank_20d: 20-day volatility rank within universe per date
    if "vol_20d" in df.columns:
        df["vol_rank_20d"] = df.groupby("date")["vol_20d"].rank(pct=True)
        new_features.append("vol_rank_20d")
        log(f"  vol_rank_20d: mean={df['vol_rank_20d'].mean():.3f}")

    # momentum_rank_60d: 60-day return rank (already have return_rank_3m from ret_60d,
    # but this is explicit)
    if "ret_60d" in df.columns:
        df["momentum_rank_60d"] = df.groupby("date")["ret_60d"].rank(pct=True)
        new_features.append("momentum_rank_60d")
        log(f"  momentum_rank_60d: mean={df['momentum_rank_60d'].mean():.3f}")

    # rsi_rank: RSI rank within universe per date
    if "rsi_14" in df.columns:
        df["rsi_rank"] = df.groupby("date")["rsi_14"].rank(pct=True)
        new_features.append("rsi_rank")
        log(f"  rsi_rank: mean={df['rsi_rank'].mean():.3f}")

    # dist_sma50_rank: distance from SMA50 rank within universe per date
    if "dist_sma50" in df.columns:
        df["dist_sma50_rank"] = df.groupby("date")["dist_sma50"].rank(pct=True)
        new_features.append("dist_sma50_rank")
        log(f"  dist_sma50_rank: mean={df['dist_sma50_rank'].mean():.3f}")

    log(f"  Added {len(new_features)} new features: {new_features}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Train model
# ══════════════════════════════════════════════════════════════════════════════

def train_v4_model(df: pd.DataFrame):
    """Train calibrated LightGBM on cross-sectional rank target."""
    log(f"\n{'='*70}")
    log("STEP 3: TRAIN V4 MODEL")
    log(f"{'='*70}")

    feature_cols = get_feature_cols(df)
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]

    # Drop rows with NaN in non-fundamental features or target_v4
    before = len(df)
    df = df.dropna(subset=non_fund_cols + ["target_v4"])
    log(f"  Loaded {before:,} → {len(df):,} rows after dropping NaN (including target_v4)")

    # Survivorship filter: only train on in-S&P500 rows
    n_before = len(df)
    train_df = df[df["in_sp500"] == True].copy()
    log(f"  Survivorship filter: {n_before:,} → {len(train_df):,} rows")

    log(f"  {train_df['symbol'].nunique()} symbols  |  "
        f"{train_df['date'].min().date()} → {train_df['date'].max().date()}")
    log(f"  Features: {len(feature_cols)} columns")

    X_all = df[feature_cols].values
    y_v4_all = df["target_v4"].values

    X_filtered = train_df[feature_cols].values
    y_filtered = train_df["target_v4"].values

    # Date-based split
    all_dates = np.sort(train_df["date"].unique())
    split_idx = int(len(all_dates) * (1 - CALIB_FRAC))
    calib_start = pd.Timestamp(all_dates[split_idx])

    train_mask = train_df["date"] < calib_start
    calib_mask = train_df["date"] >= calib_start

    X_train, y_train = X_filtered[train_mask], y_filtered[train_mask]
    X_calib, y_calib = X_filtered[calib_mask], y_filtered[calib_mask]

    log(f"\n  Split by date:")
    log(f"    Train: {train_mask.sum():,} rows  ({train_df[train_mask]['date'].min().date()} → {train_df[train_mask]['date'].max().date()})")
    log(f"    Calib: {calib_mask.sum():,} rows  ({train_df[calib_mask]['date'].min().date()} → {train_df[calib_mask]['date'].max().date()})")

    # Class balance
    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    log(f"    Train pos/neg: {int(y_train.sum()):,} / {int(len(y_train) - y_train.sum()):,}  "
        f"(scale_pos_weight={scale:.2f})")

    # Train LightGBM
    log(f"\n  Training LGBMClassifier ({N_TREES} trees, early_stop={EARLY_STOP_ROUNDS}) ...")
    model_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_calib, y_calib)],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    n_trees = model_lgb.booster_.num_trees()
    log(f"  Trained with {n_trees} trees (early stopped from {N_TREES})")

    # Calibrate
    log(f"  Calibrating with isotonic regression on {calib_mask.sum():,} rows ...")
    calib_model = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
    calib_model.fit(X_calib, y_calib)

    # Calibration AUC
    calib_probs = calib_model.predict_proba(X_calib)[:, 1]
    calib_auc = roc_auc_score(y_calib, calib_probs)
    log(f"  Calibration AUC: {calib_auc:.4f}")

    # Generate predictions for ALL rows
    log(f"\n  Generating predictions for ALL {len(df):,} rows ...")
    all_probs = calib_model.predict_proba(X_all)[:, 1]

    df["prob_v4"] = all_probs

    # Summary stats
    log(f"  Prediction range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    log(f"  Mean: {all_probs.mean():.4f}  Std: {all_probs.std():.4f}  Median: {np.median(all_probs):.4f}")
    for thresh in [0.30, 0.40, 0.50, 0.55, 0.60]:
        n_above = int((all_probs >= thresh).sum())
        log(f"  Predictions >= {thresh}: {n_above:,} / {len(all_probs):,} ({n_above/len(all_probs)*100:.1f}%)")

    # Feature importance (top 20)
    importance = model_lgb.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    log(f"\n  Feature Importance (top 20):")
    for fname, fimp in feat_imp[:20]:
        log(f"    {fname:<30s} {fimp:>6d}")

    # Save model
    log(f"\n  Saving model → {MODEL_FILE.name}")
    joblib.dump(calib_model, str(MODEL_FILE))
    log(f"  Model size: {MODEL_FILE.stat().st_size / 1024:.1f} KB")

    # Save predictions
    # Include fwd_10d_ret for backtest and the original fwd_ret (shifted ret_10d)
    df["fwd_ret"] = df["fwd_10d_ret"]  # alias for backtest compatibility
    save_cols = ["date", "symbol", "target_v4", "prob_v4", "fwd_ret", "in_sp500"]
    df[save_cols].to_parquet(PRED_FILE, index=False, engine="pyarrow", compression="snappy")
    log(f"  Predictions → {PRED_FILE.name} ({len(df):,} rows)")

    return calib_model, df, feature_cols, calib_auc, feat_imp


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Verify predictions
# ══════════════════════════════════════════════════════════════════════════════

def verify_predictions(model, feature_cols):
    """Load model and predictions independently, re-predict random rows, verify match."""
    log(f"\n{'='*70}")
    log("STEP 4: VERIFICATION — model_v4.lgb vs predictions_v4.parquet")
    log(f"{'='*70}")

    loaded_model = joblib.load(str(MODEL_FILE))
    preds_df = pd.read_parquet(PRED_FILE)
    features_df = pd.read_parquet(INPUT_FILE)
    features_df["date"] = pd.to_datetime(features_df["date"])
    preds_df["date"] = pd.to_datetime(preds_df["date"])

    # Add the new cross-sectional features to features_df for re-prediction
    features_df = add_cross_sectional_features(features_df)

    np.random.seed(42)
    unique_dates = preds_df["date"].unique()
    sample_dates = np.random.choice(unique_dates, size=min(10, len(unique_dates)), replace=False)

    mismatches = 0
    log(f"\n  {'Date':<12} {'Symbol':<8} {'Pred.parquet':>13} {'Re-predicted':>13} {'Delta':>10} {'Match':>6}")
    log(f"  {'─'*12} {'─'*8} {'─'*13} {'─'*13} {'─'*10} {'─'*6}")

    for date in sorted(sample_dates):
        date_rows = preds_df[preds_df["date"] == date]
        if len(date_rows) == 0:
            continue
        # Pick a row that has a valid prob_v4
        valid_rows = date_rows[date_rows["prob_v4"].notna()]
        if len(valid_rows) == 0:
            continue
        row = valid_rows.sample(1, random_state=int(pd.Timestamp(date).timestamp()) % 10000).iloc[0]

        feat_row = features_df[
            (features_df["date"] == row["date"]) &
            (features_df["symbol"] == row["symbol"])
        ]
        if len(feat_row) == 0:
            continue

        X_single = feat_row[feature_cols].values
        re_prob = loaded_model.predict_proba(X_single)[:, 1][0]
        stored_prob = row["prob_v4"]
        delta = abs(re_prob - stored_prob)
        match = delta < 0.001

        status = "OK" if match else "FAIL"
        log(f"  {str(pd.Timestamp(date).date()):<12} {row['symbol']:<8} "
            f"{stored_prob:>13.6f} {re_prob:>13.6f} {delta:>10.6f} {status:>6}")

        if not match:
            mismatches += 1

    log(f"\n  Results: {10 - mismatches}/10 matched (tolerance < 0.001)")

    if mismatches > 0:
        log(f"\n  *** VERIFICATION FAILED: {mismatches} mismatches detected ***")
        sys.exit(1)
    else:
        log(f"  *** VERIFICATION PASSED: model_v4.lgb and predictions_v4.parquet are consistent ***")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: V4 Backtest — Top-N Selection
# ══════════════════════════════════════════════════════════════════════════════

def run_v4_backtest():
    """
    Run Path B backtest with V4 top-N selection.

    ML V4 strategy: instead of threshold, pick top 5 stocks by prediction
    from in-S&P500 universe each day. Allocate to ML slots (up to 2 active).
    """
    log(f"\n{'='*70}")
    log("STEP 5: V4 BACKTEST — Top-N Cross-Sectional Selection")
    log(f"{'='*70}")

    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS, SLIPPAGE, POSITION_PCT,
        SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
        COOLDOWN_DAYS,
        Signal, Position, SlotConfig,
        Strategy,
        MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from backtest_ml import calc_metrics, calc_alpha_beta
    from diagnose_combined import instrumented_run

    # ── Load V4 predictions ─────────────────────────────────────────────
    preds_df = pd.read_parquet(PRED_FILE)
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df = preds_df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    log(f"  Predictions: {len(preds_df):,} rows  |  {len(universe_syms)} symbols  |  {years:.1f} years")
    log(f"  Date range: {pd.Timestamp(all_dates[0]).date()} → {pd.Timestamp(all_dates[-1]).date()}")

    # ── V4 ML Strategy: top-N selection ─────────────────────────────────

    class MLV4Strategy(Strategy):
        """
        Cross-sectional ranking strategy: pick top N stocks by V4 prediction
        from the in-S&P500 universe each day, regardless of absolute prob value.
        """
        def __init__(self, predictions_df, top_n=TOP_N_PICKS, position_pct=POSITION_PCT):
            self._top_n = top_n
            self._position_pct = position_pct
            self._signals_by_date = {}
            self._build_lookup(predictions_df)

        @property
        def name(self):
            return "ml_medium"  # use same slot name for compatibility

        def _build_lookup(self, df):
            """Pre-build per-date signal lists: top N in-S&P500 stocks by prob_v4."""
            for date, grp in df.groupby("date"):
                # Filter to in-S&P500 only
                sp500_grp = grp[grp["in_sp500"] == True]
                if sp500_grp.empty:
                    continue
                # Sort by prob_v4 descending, take top N
                top = sp500_grp.nlargest(self._top_n, "prob_v4")
                self._signals_by_date[date] = [
                    (row.symbol, row.prob_v4, row.fwd_ret)
                    for row in top.itertuples(index=False)
                ]

        def generate_signals(self, date, universe_data):
            raw = self._signals_by_date.get(date, [])
            return [
                Signal(symbol=sym, confidence=prob,
                       strategy_name=self.name, fwd_ret=fwd_ret)
                for sym, prob, fwd_ret in raw
            ]

        def check_exit(self, position, current_data):
            if current_data["idx"] >= position.exit_idx:
                return True, "hold_complete"
            return False, ""

        def get_position_size(self, signal, portfolio_value):
            ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
            return portfolio_value * self._position_pct * ml_mult

    # ── Fetch OHLCV ─────────────────────────────────────────────────────
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

    log(f"  Fetching OHLCV for {len(all_syms)} symbols ...")
    raw = yf.download(all_syms, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume = None
    if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
        volume = raw["Volume"]
        volume.index = pd.to_datetime(volume.index).tz_localize(None)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")

    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close_aligned["SPY"].to_dict()

    # ── Build strategies ────────────────────────────────────────────────
    ml_v4 = MLV4Strategy(preds_df, top_n=TOP_N_PICKS)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ── Run backtest ────────────────────────────────────────────────────
    log(f"  Running Path B (ML V4 top-{TOP_N_PICKS} + Momentum + Mean Reversion) ...")
    vals, trades, diag = instrumented_run(
        [ml_v4, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="V4 Path B")

    m = calc_metrics(vals, trades, years, "V4 Path B")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    m["alpha"] = alpha
    m["beta"] = beta
    m["avg_pos"] = float(np.mean(diag["daily_pos_count"]))

    # ── Trade frequency analysis ────────────────────────────────────────
    log(f"\n  Trade Frequency Analysis:")

    # Analyze ML V4 signal availability
    ml_signal_dates = set(ml_v4._signals_by_date.keys())
    total_dates = len(all_dates)
    dates_with_signals = len(ml_signal_dates)
    log(f"    Dates with ML V4 signals: {dates_with_signals}/{total_dates} ({dates_with_signals/total_dates*100:.1f}%)")

    # Avg picks per signal day
    picks_per_day = [len(v) for v in ml_v4._signals_by_date.values()]
    log(f"    Avg picks per day: {np.mean(picks_per_day):.1f}")

    # Year-by-year trade count
    trade_dates = []
    for rec in diag.get("trade_records", []):
        # trade_records may not have date; use the trade index
        pass

    # Year-by-year from daily position counts
    log(f"\n  Year-by-Year Returns:")
    yearly_data = vals.resample("YE").last()
    prev_val = vals.iloc[0]
    yearly_returns = {}
    for yr_end in yearly_data.index:
        yr_val = yearly_data[yr_end]
        yr_ret = (yr_val / prev_val) - 1.0
        yearly_returns[yr_end.year] = yr_ret
        log(f"    {yr_end.year}: {yr_ret:+.2%}")
        prev_val = yr_val

    # 2026 Q1 analysis
    log(f"\n  2026 Q1 Pick Frequency:")
    q1_2026_dates = [d for d in all_dates if pd.Timestamp(d).year == 2026 and pd.Timestamp(d).month <= 3]
    q1_signals = sum(1 for d in q1_2026_dates if d in ml_signal_dates)
    log(f"    Trading days in 2026 Q1: {len(q1_2026_dates)}")
    log(f"    Days with ML V4 picks: {q1_signals}")
    if q1_2026_dates:
        log(f"    Coverage: {q1_signals/len(q1_2026_dates)*100:.1f}%")

    # Check all of 2026
    dates_2026 = [d for d in all_dates if pd.Timestamp(d).year == 2026]
    signals_2026 = sum(1 for d in dates_2026 if d in ml_signal_dates)
    log(f"    2026 total: {signals_2026} signal days out of {len(dates_2026)} trading days")

    return m, yearly_returns, diag, ml_v4


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Full Report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(metrics, calib_auc, feat_imp, yearly_returns, diag, ml_v4):
    """Print comprehensive comparison report."""
    log(f"\n{'='*70}")
    log("STEP 6: V4 CROSS-SECTIONAL RANKING — FULL REPORT")
    log(f"{'='*70}")

    # ── Model metrics ───────────────────────────────────────────────────
    log(f"\n  V4 Model Metrics:")
    log(f"    Calibration AUC: {calib_auc:.4f}")

    log(f"\n  Feature Importance (top 20):")
    for fname, fimp in feat_imp[:20]:
        log(f"    {fname:<30s} {fimp:>6d}")

    # ── Backtest comparison ─────────────────────────────────────────────
    log(f"\n  {'='*65}")
    log(f"  Backtest Comparison (all variants):")
    log(f"  {'='*65}")

    header = f"  {'Variant':<30s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s}"
    log(header)
    log(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8}")

    for name, r in PRIOR_RESULTS.items():
        log(f"  {name:<30s} {r['cagr']*100:>7.2f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    cagr = metrics["cagr"]
    sharpe = metrics["sharpe"]
    max_dd = metrics["max_dd"]
    log(f"  {'V4 (500-stock cross-sect.)':<30s} {cagr*100:>7.2f}% {sharpe:>8.2f} {max_dd*100:>7.1f}%")

    # Additional metrics
    log(f"\n  V4 Additional Metrics:")
    log(f"    Sortino:       {metrics['sortino']:.3f}")
    log(f"    Win Rate:      {metrics['win_rate']*100:.1f}%")
    log(f"    Profit Factor: {metrics['profit_factor']:.3f}" if np.isfinite(metrics['profit_factor']) else "    Profit Factor: inf")
    log(f"    Avg Trade Ret: {metrics['avg_trade_ret']*100:+.3f}%")
    log(f"    Total Trades:  {metrics['n_trades']:,}")
    log(f"    Alpha vs SPY:  {metrics['alpha']*100:+.2f}%")
    log(f"    Beta:          {metrics['beta']:.3f}")
    log(f"    Final Value:   ${metrics['final_value']:,.0f}")

    # ── Trade frequency ─────────────────────────────────────────────────
    log(f"\n  Trade Frequency (V4 fixes for V3 'no trades' problem):")
    ml_signal_dates = set(ml_v4._signals_by_date.keys())
    log(f"    Total signal days: {len(ml_signal_dates)}")
    picks_per_day = [len(v) for v in ml_v4._signals_by_date.values()]
    log(f"    Avg picks/day: {np.mean(picks_per_day):.1f}")
    log(f"    Min picks/day: {min(picks_per_day)}")
    log(f"    Max picks/day: {max(picks_per_day)}")

    # ── Deployment criteria ─────────────────────────────────────────────
    log(f"\n  {'='*65}")
    log(f"  DEPLOYMENT CRITERIA:")
    log(f"  {'='*65}")

    checks = [
        ("CAGR >= 27%",         cagr >= V4_MIN_CAGR,    f"{cagr*100:.2f}%"),
        ("Sharpe >= 1.60",      sharpe >= V4_MIN_SHARPE, f"{sharpe:.3f}"),
        ("Max DD <= -25%",      max_dd >= V4_MAX_DD,     f"{max_dd*100:.1f}%"),
    ]

    # Check trade consistency (no 0-trade years)
    zero_trade_years = [y for y, r in yearly_returns.items() if abs(r) < 0.0001]
    trade_consistent = len(zero_trade_years) == 0
    checks.append(("Consistent trades",  trade_consistent, f"{'Yes' if trade_consistent else 'No: ' + str(zero_trade_years)}"))

    all_pass = True
    for label, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        log(f"    [{status}] {label}: {value}")

    log(f"\n  {'='*50}")
    if all_pass:
        log(f"  VERDICT: ALL CRITERIA MET — ready for human review")
    else:
        log(f"  VERDICT: CRITERIA NOT MET — see failures above")
    log(f"  {'='*50}")
    log(f"\n  *** Do NOT deploy to AWS — V4 artifacts saved separately ***")
    log(f"  *** Current model.lgb (V2) remains in production ***")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  V4 CROSS-SECTIONAL RANKING MODEL")
    log("  Target: top 20% of S&P 500 by 10-day forward return")
    log("  Selection: top-5 stocks per day (no threshold)")
    log("=" * 70)

    # STEP 1: Load data
    log(f"\n{'='*70}")
    log("STEP 1: LOAD DATA & COMPUTE RANK TARGET")
    log(f"{'='*70}")

    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    log(f"  Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # Compute cross-sectional rank target
    df = compute_rank_target(df)

    # STEP 2: Add new features
    log(f"\n{'='*70}")
    log("STEP 2: ADD CROSS-SECTIONAL FEATURES")
    log(f"{'='*70}")
    df = add_cross_sectional_features(df)

    # STEP 3: Train
    calib_model, df, feature_cols, calib_auc, feat_imp = train_v4_model(df)

    # STEP 4: Verify
    verify_predictions(calib_model, feature_cols)

    # STEP 5: Backtest
    metrics, yearly_returns, diag, ml_v4 = run_v4_backtest()

    # STEP 6: Report
    print_report(metrics, calib_auc, feat_imp, yearly_returns, diag, ml_v4)

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
