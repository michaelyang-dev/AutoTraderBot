#!/usr/bin/env python3
"""
V5 Ensemble Research — 3-Way Diversity Experiments
===================================================
Tests architecturally diverse model combinations against the 2-way baseline.

Uses bug-fixed features (83 features, filing_date-based ratios).

Versions:
  2-Way Baseline: 0.5*LGBM + 0.5*RF
  Version A: (LGBM + RF + ExtraTrees) / 3
  Version B: (LGBM + RF + LogReg) / 3
  Version C: 0.3*LGBM + 0.7*RF (weighted)
"""

import sys
import time
import warnings
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES_FILE = DATA_DIR / "features.parquet"

CALIB_FRAC = 0.20
TOP_PERCENTILE = 0.20
TOP_N_PICKS = 5

# Backtest params
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

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
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


class ScaledLogRegModel:
    """Wraps scaler + logistic regression for predict_proba interface."""
    def __init__(self, scaler, lr_model):
        self.scaler = scaler
        self.lr_model = lr_model
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.lr_model.predict_proba(X_scaled)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: Train All Models
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models():
    log("=" * 70)
    log("STAGE 1: TRAIN ALL MODELS ON BUG-FIXED FEATURES")
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

    log(f"  Rows: {len(train_df):,} | Features: {len(feature_cols)}")

    X_all = df[feature_cols].values
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
    log(f"  Train: {train_mask.sum():,} | Calib: {calib_mask.sum():,}")

    # Imputer for tree models that can't handle NaN (RF, ET)
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_calib_imp = imp.transform(X_calib)
    X_all_imp = imp.transform(X_all)

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    models = {}
    probs = {}

    # ── 1. LGBM ─────────────────────────────────────────────────────────
    log(f"\n  [1/4] Training LightGBM ...")
    t1 = time.perf_counter()
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
    probs["lgbm"] = calib_lgbm.predict_proba(X_all)[:, 1]
    log(f"    AUC={auc_lgbm:.4f} | {lgb_model.booster_.num_trees()} trees | {time.perf_counter()-t1:.0f}s")

    # ── 2. Random Forest ────────────────────────────────────────────────
    log(f"  [2/4] Training Random Forest ...")
    t1 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=100,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train_imp, y_train)
    calib_rf = ManualCalibratedModel(rf, X_calib_imp, y_calib)
    auc_rf = roc_auc_score(y_calib, calib_rf.predict_proba(X_calib_imp)[:, 1])
    probs["rf"] = calib_rf.predict_proba(X_all_imp)[:, 1]
    log(f"    AUC={auc_rf:.4f} | {time.perf_counter()-t1:.0f}s")

    # ── 3. ExtraTrees ───────────────────────────────────────────────────
    log(f"  [3/4] Training ExtraTrees ...")
    t1 = time.perf_counter()
    et = ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=100,
                              n_jobs=-1, random_state=42)
    et.fit(X_train_imp, y_train)
    calib_et = ManualCalibratedModel(et, X_calib_imp, y_calib)
    auc_et = roc_auc_score(y_calib, calib_et.predict_proba(X_calib_imp)[:, 1])
    probs["et"] = calib_et.predict_proba(X_all_imp)[:, 1]
    joblib.dump(calib_et, str(DATA_DIR / "model_v5_extratrees.pkl"))
    log(f"    AUC={auc_et:.4f} | {time.perf_counter()-t1:.0f}s")

    # ── 4. Logistic Regression ──────────────────────────────────────────
    log(f"  [4/4] Training Logistic Regression ...")
    t1 = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_calib_scaled = scaler.transform(X_calib_imp)

    lr = LogisticRegression(C=1.0, max_iter=2000, solver="saga", random_state=42, n_jobs=-1)
    lr.fit(X_train_scaled, y_train)

    # Calibrate LR
    scaled_lr = ScaledLogRegModel(scaler, lr)
    calib_lr = ManualCalibratedModel(scaled_lr, X_calib_imp, y_calib)
    auc_lr = roc_auc_score(y_calib, calib_lr.predict_proba(X_calib_imp)[:, 1])
    probs["lr"] = calib_lr.predict_proba(X_all_imp)[:, 1]
    joblib.dump(calib_lr, str(DATA_DIR / "model_v5_logreg.pkl"))
    joblib.dump(scaler, str(DATA_DIR / "model_v5_scaler.pkl"))
    log(f"    AUC={auc_lr:.4f} | {time.perf_counter()-t1:.0f}s")

    # ── AUC Summary ─────────────────────────────────────────────────────
    log(f"\n  {'Model':<20s} {'Calib AUC':>10s}")
    log(f"  {'─'*20} {'─'*10}")
    for name, auc in [("LGBM", auc_lgbm), ("RF", auc_rf), ("ExtraTrees", auc_et), ("LogReg", auc_lr)]:
        log(f"  {name:<20s} {auc:>10.4f}")

    # ── Correlations ────────────────────────────────────────────────────
    log(f"\n  Prediction Correlations:")
    names = ["lgbm", "rf", "et", "lr"]
    labels = ["LGBM", "RF", "ExtraTrees", "LogReg"]
    log(f"  {'':>12s} {'LGBM':>8s} {'RF':>8s} {'ET':>8s} {'LR':>8s}")
    for i, n1 in enumerate(names):
        row = f"  {labels[i]:<12s}"
        for j, n2 in enumerate(names):
            corr = np.corrcoef(probs[n1], probs[n2])[0, 1]
            row += f" {corr:>7.4f}"
        log(row)

    # ── Build ensemble predictions ──────────────────────────────────────
    df["prob_lgbm"] = probs["lgbm"]
    df["prob_rf"] = probs["rf"]
    df["prob_et"] = probs["et"]
    df["prob_lr"] = probs["lr"]
    df["prob_2way"] = (probs["lgbm"] + probs["rf"]) / 2.0
    df["prob_vA"] = (probs["lgbm"] + probs["rf"] + probs["et"]) / 3.0
    df["prob_vB"] = (probs["lgbm"] + probs["rf"] + probs["lr"]) / 3.0
    df["prob_vC"] = 0.3 * probs["lgbm"] + 0.7 * probs["rf"]
    df["fwd_ret"] = df["fwd_10d_ret"]

    # Save
    pred_cols = ["date", "symbol", "target_v4", "prob_lgbm", "prob_rf", "prob_et",
                 "prob_lr", "prob_2way", "prob_vA", "prob_vB", "prob_vC",
                 "fwd_ret", "in_sp500"]
    df[pred_cols].to_parquet(DATA_DIR / "predictions_v5_ensemble.parquet", index=False)
    log(f"\n  Saved predictions_v5_ensemble.parquet")

    return df, probs


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: Bug-Fixed Backtester (same as backtest_audit_fixed.py)
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


def run_backtest(preds_df, prob_col, all_dates, spy_close_series,
                 open_data, close_data, label):
    """Bug-fixed backtest with regime filter, next-day open, vol targeting."""
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

    open_lookup = {}
    close_lookup = {}
    for date in all_dates:
        if date in open_data.index:
            open_lookup[date] = open_data.loc[date].to_dict()
        if date in close_data.index:
            close_lookup[date] = close_data.loc[date].to_dict()

    cash = float(INITIAL_CASH)
    positions = {}
    idle_spy_shares = 0.0
    trades = []
    cooldowns = {}
    port_vals = []
    daily_returns = []

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
            scale = max(MIN_LEVERAGE, min(MAX_LEVERAGE, TARGET_VOL / rv if rv > 0 else 1.0))
        else:
            scale = 1.0

        regime = compute_spy_regime(spy_close_series, date)

        # Update peaks
        for sym, pos in positions.items():
            if sym in day_close:
                px = day_close[sym]
                if not np.isnan(px) and px > pos["peak_price"]:
                    pos["peak_price"] = px

        # Close expiring (at open — Bug 4)
        to_close = [sym for sym, pos in positions.items() if i >= pos["exit_idx"]]
        for sym in to_close:
            pos = positions.pop(sym)
            exit_px = day_open.get(sym, pos["entry_price"])
            if np.isnan(exit_px) or exit_px <= 0:
                exit_px = day_close.get(sym, pos["entry_price"])
            ret = (exit_px / pos["entry_price"]) - 1.0
            gross = pos["cost"] * (1.0 + ret)
            net = gross * (1.0 - SLIPPAGE)
            cash += net
            trades.append((net - pos["cost"]) / pos["cost"])
            cooldowns[sym] = i + COOLDOWN_DAYS

        # Signals with regime filter (Bug 3)
        raw = signals_by_date.get(date, [])
        if regime == "BEARISH":
            filtered = []
        elif regime == "CAUTIOUS":
            filtered = raw[:2]
        else:
            filtered = raw

        held = set(positions.keys())
        candidates = [(s, p, f) for s, p, f in filtered
                      if s not in held and cooldowns.get(s, -1) <= i and not np.isnan(f)]
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Execute at next-day open (Bug 4)
        next_idx = i + 1
        next_open = open_lookup.get(all_dates[next_idx], {}) if next_idx < len(all_dates) else {}
        max_new = max(0, 10 - len(positions))

        if spy_close and idle_spy_shares > 0 and candidates and max_new > 0:
            cash += idle_spy_shares * spy_close * (1.0 - SLIPPAGE)
            idle_spy_shares = 0.0

        for sym, prob, fwd_ret in candidates[:max_new]:
            entry_px = next_open.get(sym, np.nan)
            if np.isnan(entry_px) or entry_px <= 0:
                continue
            port_est = cash + sum(
                p["cost"] * (day_close.get(p["symbol"], p["entry_price"]) / p["entry_price"]
                             if p["entry_price"] > 0 else 1.0) for p in positions.values())
            ml_mult = min(1.0, max(0.60, prob * 1.6 - 0.28))
            target = port_est * POSITION_PCT * ml_mult * scale
            cost = min(target, cash * 0.95)
            if cost < 50.0:
                continue
            cash -= cost * (1.0 + SLIPPAGE)
            positions[sym] = {
                "symbol": sym, "cost": cost, "entry_idx": next_idx,
                "exit_idx": min(next_idx + HOLD_DAYS, len(all_dates) - 1),
                "entry_price": entry_px, "peak_price": entry_px,
            }

        # SPY parking
        if spy_close:
            pos_val = sum(p["cost"] * (day_close.get(p["symbol"], p["entry_price"]) / p["entry_price"]
                          if p["entry_price"] > 0 else 1.0) for p in positions.values())
            est_port = cash + idle_spy_shares * spy_close + pos_val
            idle_cash = cash - est_port * SPY_RESERVE_PCT
            if idle_cash > est_port * SPY_THRESHOLD_PCT:
                invest = min(idle_cash * SPY_INVEST_PCT, cash * 0.95)
                cash -= invest * (1.0 + SLIPPAGE)
                idle_spy_shares += invest / spy_close

        # Mark to market
        port_val = cash + (idle_spy_shares * spy_close if spy_close else 0)
        for pos in positions.values():
            px = day_close.get(pos["symbol"], pos["entry_price"])
            port_val += pos["cost"] * (px / pos["entry_price"] if pos["entry_price"] > 0 else 1.0)
        port_vals.append(port_val)

        daily_returns.append((port_vals[-1] / port_vals[-2] - 1.0) if len(port_vals) >= 2 else 0.0)

    return pd.Series(port_vals, index=pd.DatetimeIndex(all_dates)), trades


def calc_metrics(vals, trades, years, label):
    final = vals.iloc[-1]
    cagr = (final / INITIAL_CASH) ** (1.0 / years) - 1.0
    dr = vals.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    neg = dr[dr < 0]
    sortino = dr.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    wins = sum(1 for t in trades if t > 0)
    return {"label": label, "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "final_value": final, "n_trades": len(trades),
            "win_rate": wins / len(trades) if trades else 0}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  V5 ENSEMBLE RESEARCH — DIVERSITY EXPERIMENTS")
    log("  Bug-fixed features (83 cols, filing_date ratios)")
    log("=" * 70)

    # ── Stage 1: Train ──────────────────────────────────────────────────
    df, probs = train_all_models()

    # ── Stage 2: Load OHLCV ─────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("STAGE 2: BACKTEST ALL VERSIONS")
    log(f"{'='*70}")

    preds = pd.read_parquet(DATA_DIR / "predictions_v5_ensemble.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds["date"].unique().tolist())
    universe_syms = sorted(preds["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    log(f"  {len(preds):,} rows | {len(universe_syms)} symbols | {years:.1f} years")

    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

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
    spy_bh = spy_close_series / spy_close_series.iloc[0] * INITIAL_CASH

    # ── Run all versions ────────────────────────────────────────────────
    versions = [
        ("prob_2way",  "2-Way Baseline (L+R)"),
        ("prob_vA",    "Version A (L+R+ET)"),
        ("prob_vB",    "Version B (L+R+LR)"),
        ("prob_vC",    "Version C (0.3L+0.7R)"),
        ("prob_rf",    "RF Alone (reference)"),
    ]

    results = {}
    all_vals = {}
    for prob_col, label in versions:
        log(f"\n  Running {label} ...")
        vals, trades = run_backtest(preds, prob_col, all_dates, spy_close_series,
                                    open_aligned, close_aligned, label)
        results[prob_col] = calc_metrics(vals, trades, years, label)
        all_vals[prob_col] = vals

    # ── Stage 3: Report ─────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("STAGE 3: RESULTS")
    log(f"{'='*70}")

    log(f"\n  {'Model':<30s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>8s} {'Win%':>7s}")
    log(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    for prob_col, label in versions:
        m = results[prob_col]
        log(f"  {m['label']:<30s} {m['cagr']*100:>7.2f}% {m['sharpe']:>8.2f} "
            f"{m['max_dd']*100:>7.1f}% {m['n_trades']:>8,} {m['win_rate']*100:>6.1f}%")

    # ── Deltas vs baseline ──────────────────────────────────────────────
    baseline = results["prob_2way"]
    log(f"\n  Delta vs 2-Way Baseline:")
    log(f"  {'Model':<30s} {'ΔCAGR':>8s} {'ΔSharpe':>9s} {'ΔDD':>8s}")
    log(f"  {'─'*30} {'─'*8} {'─'*9} {'─'*8}")
    for prob_col, label in versions:
        if prob_col == "prob_2way":
            continue
        m = results[prob_col]
        dc = (m["cagr"] - baseline["cagr"]) * 100
        ds = m["sharpe"] - baseline["sharpe"]
        dd = (m["max_dd"] - baseline["max_dd"]) * 100
        log(f"  {m['label']:<30s} {dc:>+7.2f}% {ds:>+8.2f} {dd:>+7.1f}%")

    # ── Correlations by version ─────────────────────────────────────────
    log(f"\n  Model Correlations (within each version):")
    names_all = ["lgbm", "rf", "et", "lr"]
    labels_all = ["LGBM", "RF", "ET", "LR"]
    log(f"\n  Full correlation matrix:")
    log(f"  {'':>8s} {'LGBM':>8s} {'RF':>8s} {'ET':>8s} {'LR':>8s}")
    for i, n1 in enumerate(names_all):
        row = f"  {labels_all[i]:<8s}"
        for j, n2 in enumerate(names_all):
            c = np.corrcoef(probs[n1], probs[n2])[0, 1]
            row += f" {c:>7.4f}"
        log(row)

    log(f"\n  Version A diversity: LGBM↔ET={np.corrcoef(probs['lgbm'], probs['et'])[0,1]:.4f}, "
        f"RF↔ET={np.corrcoef(probs['rf'], probs['et'])[0,1]:.4f}")
    log(f"  Version B diversity: LGBM↔LR={np.corrcoef(probs['lgbm'], probs['lr'])[0,1]:.4f}, "
        f"RF↔LR={np.corrcoef(probs['rf'], probs['lr'])[0,1]:.4f}")

    # ── Year-by-year for top 2 ──────────────────────────────────────────
    # Find top 2 by Sharpe
    sorted_results = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    top2 = sorted_results[:2]

    log(f"\n  {'='*65}")
    log(f"  YEAR-BY-YEAR — Top 2 by Sharpe")
    log(f"  {'='*65}")

    t2_labels = [t[1]["label"][:15] for t in top2]
    log(f"\n  {'Year':<6} {t2_labels[0]:>15s} {t2_labels[1]:>15s} {'SPY':>10s}")
    log(f"  {'─'*6} {'─'*15} {'─'*15} {'─'*10}")

    spy_aligned = spy_bh.reindex(all_vals[top2[0][0]].index, method="ffill")
    for yr in sorted(set(all_vals[top2[0][0]].index.year)):
        row = f"  {yr:<6}"
        for prob_col, _ in top2:
            v = all_vals[prob_col]
            yr_v = v[v.index.year == yr]
            if len(yr_v) >= 2:
                ret = yr_v.iloc[-1] / yr_v.iloc[0] - 1
                row += f" {ret*100:>14.2f}%"
            else:
                row += f" {'N/A':>15s}"
        yr_spy = spy_aligned[spy_aligned.index.year == yr]
        if len(yr_spy) >= 2 and yr_spy.iloc[0] > 0:
            spy_ret = yr_spy.iloc[-1] / yr_spy.iloc[0] - 1
            row += f" {spy_ret*100:>9.2f}%"
        log(row)

    # ── Deployment Decision ─────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("DEPLOYMENT DECISION")
    log(f"{'='*70}")

    # Find best non-RF version
    non_rf = [(k, v) for k, v in results.items() if k != "prob_rf"]
    best_key, best = max(non_rf, key=lambda x: x[1]["sharpe"])

    improve_sharpe = best["sharpe"] - baseline["sharpe"]
    improve_cagr = (best["cagr"] - baseline["cagr"]) * 100

    log(f"\n  Best ensemble: {best['label']}")
    log(f"    CAGR:   {best['cagr']*100:.2f}%")
    log(f"    Sharpe: {best['sharpe']:.3f}")
    log(f"    Max DD: {best['max_dd']*100:.1f}%")

    log(f"\n  vs 2-Way Baseline:")
    log(f"    Sharpe Δ: {improve_sharpe:+.3f} (need >= +0.05)")
    log(f"    CAGR Δ:   {improve_cagr:+.2f}% (need >= +2%)")

    pass_sharpe = improve_sharpe >= 0.05
    pass_cagr = improve_cagr >= 2.0
    dd_ok = best["max_dd"] >= baseline["max_dd"] - 0.05

    log(f"\n  Pass criteria:")
    log(f"    [{'PASS' if pass_sharpe else 'FAIL'}] Sharpe improvement >= 0.05: {improve_sharpe:+.3f}")
    log(f"    [{'PASS' if pass_cagr else 'FAIL'}] CAGR improvement >= 2%: {improve_cagr:+.2f}%")
    log(f"    [{'PASS' if dd_ok else 'FAIL'}] Max DD not significantly worse: "
        f"{best['max_dd']*100:.1f}% vs {baseline['max_dd']*100:.1f}%")

    if pass_sharpe and pass_cagr and dd_ok:
        log(f"\n  RECOMMENDATION: Deploy {best['label']} (replaces 2-way baseline)")
    else:
        log(f"\n  RECOMMENDATION: Deploy 2-way baseline as planned")
        log(f"  No 3-way version meaningfully improves on LGBM+RF")

    elapsed = time.perf_counter() - t0
    log(f"\n{'='*70}")
    log(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
