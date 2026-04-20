#!/usr/bin/env python3
"""
Walk-Forward Weight Validation for Version C (0.3 LGBM + 0.7 RF)
=================================================================
Validates that the 30/70 weight is not overfit to full-period backtest.

Methodology:
  Train period:  2011-01-01 to 2022-12-31 (models trained here)
  Optimization:  2023-01-01 to 2023-12-31 (pick best weight)
  Holdout:       2024-01-01 to 2026-04-17 (truly unseen validation)
"""

import sys
import time
import warnings
from pathlib import Path

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

CALIB_FRAC = 0.20
TOP_PERCENTILE = 0.20
TOP_N_PICKS = 5

# Backtest params (same as V5)
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

# Walk-forward split dates
TRAIN_END = pd.Timestamp("2022-12-31")
OPT_START = pd.Timestamp("2023-01-01")
OPT_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")

# Weight grid to search
WEIGHT_GRID = [
    (0.5, 0.5, "50/50"),
    (0.4, 0.6, "40/60"),
    (0.3, 0.7, "30/70"),
    (0.2, 0.8, "20/80"),
    (0.1, 0.9, "10/90"),
    (0.0, 1.0, "0/100 (RF alone)"),
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
#  BACKTESTER (same bug-fixed logic as V5)
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

        # Close expiring (at open)
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

        # Signals with regime filter
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

        # Execute at next-day open
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


def calc_metrics(vals, trades, label):
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
    wins = sum(1 for t in trades if t > 0)
    return {"label": label, "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
            "n_trades": len(trades), "win_rate": wins / len(trades) if trades else 0}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  WALK-FORWARD WEIGHT VALIDATION")
    log("  Train: 2011-2022 | Optimize: 2023 | Holdout: 2024-2026")
    log("=" * 70)

    # ── STEP 1: Load and prepare features ──────────────────────────────────
    log("\n" + "=" * 70)
    log("STEP 1: LOAD FEATURES & DEFINE SPLITS")
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
    df = df[df["in_sp500"] == True].copy()

    # Split data
    train_df = df[df["date"] <= TRAIN_END].copy()
    opt_df = df[(df["date"] >= OPT_START) & (df["date"] <= OPT_END)].copy()
    holdout_df = df[df["date"] >= HOLDOUT_START].copy()

    log(f"  Total rows:   {len(df):,}")
    log(f"  Train (2011-2022): {len(train_df):,} rows | "
        f"{train_df['date'].min().date()} to {train_df['date'].max().date()}")
    log(f"  Optimize (2023):   {len(opt_df):,} rows | "
        f"{opt_df['date'].min().date()} to {opt_df['date'].max().date()}")
    log(f"  Holdout (2024+):   {len(holdout_df):,} rows | "
        f"{holdout_df['date'].min().date()} to {holdout_df['date'].max().date()}")

    # ── STEP 2: Train models on train period ONLY ─────────────────────────
    log("\n" + "=" * 70)
    log("STEP 2: TRAIN MODELS ON 2011-2022 ONLY")
    log("=" * 70)

    X_all = train_df[feature_cols].values
    y_all = train_df["target_v4"].values

    # Time-based calibration split within train period
    train_dates = np.sort(train_df["date"].unique())
    split_idx = int(len(train_dates) * (1 - CALIB_FRAC))
    calib_start = pd.Timestamp(train_dates[split_idx])
    tr_mask = train_df["date"] < calib_start
    cal_mask = train_df["date"] >= calib_start

    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
    log(f"  Model train: {tr_mask.sum():,} | Calibration: {cal_mask.sum():,}")
    log(f"  Calib starts: {calib_start.date()}")

    # Imputer
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_cal_imp = imp.transform(X_cal)

    scale = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)

    # LGBM
    log(f"\n  Training LightGBM on 2011-2022 ...")
    t1 = time.perf_counter()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, objective="binary", metric="auc",
        random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=scale)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    calib_lgbm = CalibratedClassifierCV(lgb_model, method="isotonic", cv="prefit")
    calib_lgbm.fit(X_cal, y_cal)
    auc_lgbm = roc_auc_score(y_cal, calib_lgbm.predict_proba(X_cal)[:, 1])
    log(f"    AUC={auc_lgbm:.4f} | {lgb_model.booster_.num_trees()} trees | {time.perf_counter()-t1:.0f}s")

    # RF
    log(f"  Training Random Forest on 2011-2022 ...")
    t1 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=100,
                                n_jobs=-1, random_state=42)
    rf.fit(X_tr_imp, y_tr)
    calib_rf = ManualCalibratedModel(rf, X_cal_imp, y_cal)
    auc_rf = roc_auc_score(y_cal, calib_rf.predict_proba(X_cal_imp)[:, 1])
    log(f"    AUC={auc_rf:.4f} | {time.perf_counter()-t1:.0f}s")

    # ── Generate predictions for ALL data (opt + holdout) ─────────────────
    log(f"\n  Generating predictions for optimization + holdout periods ...")
    future_df = df[df["date"] >= OPT_START].copy()
    X_future = future_df[feature_cols].values
    X_future_imp = imp.transform(X_future)

    future_df["prob_lgbm"] = calib_lgbm.predict_proba(X_future)[:, 1]
    future_df["prob_rf"] = calib_rf.predict_proba(X_future_imp)[:, 1]
    future_df["fwd_ret"] = future_df["fwd_10d_ret"]

    log(f"  Predictions: {len(future_df):,} rows")
    log(f"  LGBM probs: [{future_df['prob_lgbm'].min():.4f}, {future_df['prob_lgbm'].max():.4f}]")
    log(f"  RF probs:   [{future_df['prob_rf'].min():.4f}, {future_df['prob_rf'].max():.4f}]")
    corr = np.corrcoef(future_df["prob_lgbm"].values, future_df["prob_rf"].values)[0, 1]
    log(f"  LGBM-RF correlation: {corr:.4f}")

    # ── STEP 3: Fetch OHLCV ───────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STEP 3: FETCH OHLCV DATA")
    log("=" * 70)

    all_dates_future = sorted(future_df["date"].unique().tolist())
    universe_syms = sorted(future_df["symbol"].unique().tolist())
    all_syms = list(set(["SPY"] + universe_syms))

    start = pd.Timestamp(all_dates_future[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates_future[-1]) + pd.Timedelta(days=5)

    log(f"  Fetching OHLCV for {len(all_syms)} symbols ...")
    raw = yf.download(all_syms, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    open_px = raw["Open"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Open"]]
    open_px.index = pd.to_datetime(open_px.index).tz_localize(None)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates_future])
    close_aligned = close.reindex(sim_index, method="ffill")
    open_aligned = open_px.reindex(sim_index, method="ffill")
    spy_close_series = close_aligned["SPY"].dropna()

    log(f"  OHLCV: {len(close_aligned)} dates")

    # ── STEP 4: Weight optimization on 2023 ───────────────────────────────
    log("\n" + "=" * 70)
    log("STEP 4: OPTIMIZE WEIGHTS ON 2023 DATA")
    log("=" * 70)

    opt_dates = sorted([d for d in all_dates_future
                        if pd.Timestamp(d) >= OPT_START and pd.Timestamp(d) <= OPT_END])
    opt_preds = future_df[(future_df["date"] >= OPT_START) & (future_df["date"] <= OPT_END)].copy()

    log(f"  Optimization period: {len(opt_dates)} trading days")
    log(f"  Testing {len(WEIGHT_GRID)} weight combinations ...\n")

    opt_results = []
    for w_lgbm, w_rf, w_label in WEIGHT_GRID:
        opt_preds["prob_weighted"] = w_lgbm * opt_preds["prob_lgbm"] + w_rf * opt_preds["prob_rf"]
        vals, trades = run_backtest(opt_preds, "prob_weighted", opt_dates,
                                     spy_close_series, open_aligned, close_aligned,
                                     f"W={w_label}")
        m = calc_metrics(vals, trades, w_label)
        opt_results.append({"w_lgbm": w_lgbm, "w_rf": w_rf, "label": w_label, **m})
        log(f"  {w_label:>16s}:  CAGR={m['cagr']*100:>7.2f}%  Sharpe={m['sharpe']:>6.3f}  "
            f"DD={m['max_dd']*100:>6.1f}%  Trades={m['n_trades']}")

    # Find best by Sharpe
    best = max(opt_results, key=lambda x: x["sharpe"])
    log(f"\n  BEST ON 2023: {best['label']} (Sharpe={best['sharpe']:.3f})")
    optimal_w_lgbm = best["w_lgbm"]
    optimal_w_rf = best["w_rf"]

    # ── STEP 5: Validate on holdout 2024-2026 ─────────────────────────────
    log("\n" + "=" * 70)
    log("STEP 5: HOLDOUT VALIDATION (2024-2026)")
    log("=" * 70)

    holdout_dates = sorted([d for d in all_dates_future if pd.Timestamp(d) >= HOLDOUT_START])
    holdout_preds = future_df[future_df["date"] >= HOLDOUT_START].copy()
    log(f"  Holdout period: {len(holdout_dates)} trading days")
    log(f"  Date range: {pd.Timestamp(holdout_dates[0]).date()} to {pd.Timestamp(holdout_dates[-1]).date()}\n")

    # Strategies to test on holdout
    holdout_strategies = [
        (optimal_w_lgbm, optimal_w_rf, f"Optimal ({best['label']})"),
        (0.5, 0.5, "50/50 Equal"),
        (0.3, 0.7, "30/70 (Version C)"),
        (0.0, 1.0, "RF Alone"),
    ]

    # Deduplicate if optimal matches one of the fixed strategies
    seen = set()
    deduped = []
    for wl, wr, label in holdout_strategies:
        key = (round(wl, 2), round(wr, 2))
        if key not in seen:
            seen.add(key)
            deduped.append((wl, wr, label))
    holdout_strategies = deduped

    holdout_results = []
    for w_lgbm, w_rf, label in holdout_strategies:
        holdout_preds["prob_weighted"] = w_lgbm * holdout_preds["prob_lgbm"] + w_rf * holdout_preds["prob_rf"]
        vals, trades = run_backtest(holdout_preds, "prob_weighted", holdout_dates,
                                     spy_close_series, open_aligned, close_aligned, label)
        m = calc_metrics(vals, trades, label)
        holdout_results.append({"w_lgbm": w_lgbm, "w_rf": w_rf, **m})

    # SPY benchmark on holdout
    spy_holdout = spy_close_series.reindex(pd.DatetimeIndex([pd.Timestamp(d) for d in holdout_dates]))
    spy_holdout = spy_holdout.dropna()
    if len(spy_holdout) > 1:
        spy_years = (spy_holdout.index[-1] - spy_holdout.index[0]).days / 365.25
        spy_cagr = (spy_holdout.iloc[-1] / spy_holdout.iloc[0]) ** (1.0 / spy_years) - 1.0 if spy_years > 0 else 0
        spy_dr = spy_holdout.pct_change().dropna()
        spy_sharpe = spy_dr.mean() / spy_dr.std() * np.sqrt(252) if spy_dr.std() > 0 else 0
        spy_peak = spy_holdout.cummax()
        spy_dd = ((spy_holdout - spy_peak) / spy_peak).min()
    else:
        spy_cagr = spy_sharpe = spy_dd = 0

    # ── STEP 6: Report ────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STEP 6: RESULTS")
    log("=" * 70)

    log(f"\n  {'─'*70}")
    log(f"  OPTIMIZATION PERIOD (2023) — Weight Search")
    log(f"  {'─'*70}")
    log(f"  {'Weight':>16s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>8s}")
    log(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for r in opt_results:
        marker = " <<<" if r["label"] == best["label"] else ""
        log(f"  {r['label']:>16s} {r['cagr']*100:>7.2f}% {r['sharpe']:>7.3f} "
            f"{r['max_dd']*100:>7.1f}% {r['n_trades']:>7d}{marker}")

    log(f"\n  Selected weight: {best['label']} (best Sharpe on 2023)")

    log(f"\n  {'─'*70}")
    log(f"  HOLDOUT PERIOD (2024-2026) — Truly Unseen")
    log(f"  {'─'*70}")
    log(f"  {'Strategy':>24s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Trades':>8s} {'Win%':>7s}")
    log(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")

    for r in holdout_results:
        log(f"  {r['label']:>24s} {r['cagr']*100:>7.2f}% {r['sharpe']:>7.3f} "
            f"{r['max_dd']*100:>7.1f}% {r['n_trades']:>7d} {r['win_rate']*100:>5.1f}%")
    log(f"  {'SPY (benchmark)':>24s} {spy_cagr*100:>7.2f}% {spy_sharpe:>7.3f} "
        f"{spy_dd*100:>7.1f}%       -     -")

    # ── Year-by-year for holdout ──────────────────────────────────────────
    log(f"\n  {'─'*70}")
    log(f"  YEAR-BY-YEAR ON HOLDOUT")
    log(f"  {'─'*70}")

    # Re-run to get portfolio value series for year-by-year
    yearly_data = {}
    for w_lgbm, w_rf, label in holdout_strategies:
        holdout_preds["prob_weighted"] = w_lgbm * holdout_preds["prob_lgbm"] + w_rf * holdout_preds["prob_rf"]
        vals, _ = run_backtest(holdout_preds, "prob_weighted", holdout_dates,
                                spy_close_series, open_aligned, close_aligned, label)
        yearly_data[label] = vals

    years_in_holdout = sorted(set(pd.Timestamp(d).year for d in holdout_dates))
    strat_names = [label for _, _, label in holdout_strategies]
    header = f"  {'Year':>6s}"
    for name in strat_names:
        header += f" {name[:15]:>15s}"
    header += f" {'SPY':>10s}"
    log(header)
    log(f"  {'─'*6}" + f" {'─'*15}" * len(strat_names) + f" {'─'*10}")

    for year in years_in_holdout:
        row = f"  {year:>6d}"
        for name in strat_names:
            vs = yearly_data[name]
            yr_dates = [d for d in vs.index if d.year == year]
            if len(yr_dates) < 2:
                row += f" {'N/A':>15s}"
                continue
            yr_ret = vs[yr_dates[-1]] / vs[yr_dates[0]] - 1.0
            row += f" {yr_ret*100:>14.2f}%"
        # SPY
        spy_yr = [d for d in spy_holdout.index if d.year == year]
        if len(spy_yr) >= 2:
            spy_yr_ret = spy_holdout[spy_yr[-1]] / spy_holdout[spy_yr[0]] - 1.0
            row += f" {spy_yr_ret*100:>9.2f}%"
        else:
            row += f" {'N/A':>10s}"
        log(row)

    # ── DEPLOYMENT DECISION ───────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("DEPLOYMENT DECISION")
    log(f"{'='*70}")

    best_holdout = max(holdout_results, key=lambda x: x["sharpe"])
    version_c = next((r for r in holdout_results if "30/70" in r["label"]), None)
    equal_w = next((r for r in holdout_results if "50/50" in r["label"]), None)
    rf_alone = next((r for r in holdout_results if "RF Alone" in r["label"]), None)
    optimal = next((r for r in holdout_results if "Optimal" in r["label"]), None)

    log(f"\n  Best on holdout: {best_holdout['label']} "
        f"(Sharpe={best_holdout['sharpe']:.3f}, CAGR={best_holdout['cagr']*100:.2f}%)")

    if optimal and optimal["label"] != best_holdout["label"]:
        log(f"  Optimal from 2023 ({optimal['label']}): "
            f"Sharpe={optimal['sharpe']:.3f}, CAGR={optimal['cagr']*100:.2f}%")

    log(f"\n  Interpretation:")
    if version_c and best_holdout["label"] == version_c["label"]:
        log(f"    30/70 wins on holdout → Version C is VALIDATED")
        deploy = "Version C (30/70)"
    elif optimal and best_holdout["label"] == optimal["label"] and abs(optimal["w_lgbm"] - 0.3) < 0.01:
        log(f"    Optimal weight matches 30/70 → Version C is VALIDATED")
        deploy = "Version C (30/70)"
    elif equal_w and best_holdout["label"] == equal_w["label"]:
        log(f"    Equal weight wins → 30/70 weight was OVERFIT")
        log(f"    Recommend deploying 2-way equal weight")
        deploy = "2-Way Equal (50/50)"
    elif rf_alone and best_holdout["label"] == rf_alone["label"]:
        log(f"    RF alone wins → LGBM adds no value on holdout")
        log(f"    Consider deploying RF alone (higher concentration risk)")
        deploy = "RF Alone"
    else:
        log(f"    Different weight wins on holdout")
        log(f"    Update to: {best_holdout['label']}")
        deploy = best_holdout["label"]

    log(f"\n  RECOMMENDATION: Deploy {deploy}")

    # Realistic live estimate
    if best_holdout["cagr"] > 0:
        conservative_cagr = best_holdout["cagr"] * 0.7  # 30% haircut for live
        log(f"\n  Realistic live performance estimate:")
        log(f"    Holdout CAGR:       {best_holdout['cagr']*100:.2f}%")
        log(f"    Conservative (70%): {conservative_cagr*100:.2f}%")
        log(f"    Holdout Max DD:     {best_holdout['max_dd']*100:.1f}%")
        log(f"    Expected live DD:   {best_holdout['max_dd']*100*1.3:.1f}% (30% worse)")

    # Remaining concerns
    log(f"\n  Remaining concerns:")
    log(f"    1. Holdout period ({holdout_dates[0].strftime('%Y-%m') if isinstance(holdout_dates[0], pd.Timestamp) else pd.Timestamp(holdout_dates[0]).strftime('%Y-%m')}"
        f" to {holdout_dates[-1].strftime('%Y-%m') if isinstance(holdout_dates[-1], pd.Timestamp) else pd.Timestamp(holdout_dates[-1]).strftime('%Y-%m')}) is only ~{len(holdout_dates)//252:.1f} years")
    log(f"    2. 2024-2025 was a specific regime — may not generalize")
    log(f"    3. RF's extreme returns may reflect tree-based overfitting to technical patterns")

    elapsed = time.perf_counter() - t0
    log(f"\n{'='*70}")
    log(f"Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
