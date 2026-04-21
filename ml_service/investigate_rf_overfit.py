#!/usr/bin/env python3
"""
RF Overfitting Investigation
=============================
4 rigorous tests to determine if RF's 59% CAGR / 2.77 Sharpe is real or artifact.

Test 1: Strict out-of-sample holdout (train ≤2022, test 2023-2026)
Test 2: Feature lookahead audit (same-day target correlation)
Test 3: Temporal stability (3-year chunks, RF vs LGBM)
Test 4: Prediction distribution analysis

Run: python3 investigate_rf_overfit.py
"""

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DATA_DIR     = Path(__file__).resolve().parent / "data"
INPUT_FILE   = DATA_DIR / "features.parquet"
LGBM_PREDS   = DATA_DIR / "predictions_v4.parquet"
RF_PREDS     = DATA_DIR / "predictions_v4_rf.parquet"

TOP_PERCENTILE = 0.20
TOP_N_PICKS    = 5

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "days_until_earnings", "eps_surprise_last",
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


def compute_rank_target(df):
    df = df.sort_values(["symbol", "date"]).copy()
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    valid_mask = sp500_mask & has_fwd
    df["target_v4"] = np.nan
    valid_df = df[valid_mask].copy()
    valid_df["pct_rank"] = valid_df.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid_df["target_v4"] = (valid_df["pct_rank"] >= (1.0 - TOP_PERCENTILE)).astype(int)
    df.loc[valid_df.index, "target_v4"] = valid_df["target_v4"]
    return df


def add_cross_sectional_features(df):
    for col, src in [("vol_rank_20d", "vol_20d"), ("momentum_rank_60d", "ret_60d"),
                     ("rsi_rank", "rsi_14"), ("dist_sma50_rank", "dist_sma50")]:
        if src in df.columns:
            df[col] = df.groupby("date")[src].rank(pct=True)
    return df


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


def run_backtest_on_preds(preds_df, prob_col, label, date_filter=None):
    """Run Path B + vol targeting backtest. Optionally filter to date range."""
    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS, POSITION_PCT,
        Signal, Strategy,
        MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from backtest_utils import calc_metrics, calc_alpha_beta
    from backtest_v4_voltarget import instrumented_run_voltarget

    preds_df = preds_df.copy()
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df = preds_df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    if date_filter:
        start_dt, end_dt = pd.Timestamp(date_filter[0]), pd.Timestamp(date_filter[1])
        preds_df = preds_df[(preds_df["date"] >= start_dt) & (preds_df["date"] <= end_dt)]

    if len(preds_df) == 0:
        return None, None

    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    if years < 0.5:
        return None, None

    class TopNStrategy(Strategy):
        def __init__(self, predictions_df, prob_column, top_n=TOP_N_PICKS, position_pct=POSITION_PCT):
            self._top_n = top_n
            self._position_pct = position_pct
            self._signals_by_date = {}
            for date, grp in predictions_df.groupby("date"):
                sp500_grp = grp[grp["in_sp500"] == True]
                if sp500_grp.empty:
                    continue
                top = sp500_grp.nlargest(self._top_n, prob_column)
                self._signals_by_date[date] = [
                    (row.symbol, getattr(row, prob_column), row.fwd_ret)
                    for row in top.itertuples(index=False)
                ]

        @property
        def name(self):
            return "ml_medium"

        def generate_signals(self, date, universe_data):
            raw = self._signals_by_date.get(date, [])
            return [Signal(symbol=s, confidence=p, strategy_name=self.name, fwd_ret=f)
                    for s, p, f in raw]

        def check_exit(self, position, current_data):
            if current_data["idx"] >= position.exit_idx:
                return True, "hold_complete"
            return False, ""

        def get_position_size(self, signal, portfolio_value):
            ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
            return portfolio_value * self._position_pct * ml_mult

    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

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

    ml_strat = TopNStrategy(preds_df, prob_col, top_n=TOP_N_PICKS)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    vals, trades, diag = instrumented_run_voltarget(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label=label)

    m = calc_metrics(vals, trades, years, label)
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    m["alpha"] = alpha
    m["beta"] = beta
    return m, vals


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1: Strict Out-of-Sample Holdout
# ══════════════════════════════════════════════════════════════════════════════

def test1_holdout():
    log(f"\n{'='*70}")
    log("TEST 1: STRICT OUT-OF-SAMPLE HOLDOUT")
    log(f"{'='*70}")
    log("  Train: 2011-2022  |  Holdout: 2023-01-01 to 2026-04-17")
    log("  RF retrained on training period only, predictions on holdout only")

    # Load and prepare
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = compute_rank_target(df)
    df = add_cross_sectional_features(df)

    feature_cols = get_feature_cols(df)
    non_fund = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund + ["target_v4"])
    df_sp500 = df[df["in_sp500"] == True].copy()

    # Strict temporal split
    holdout_start = pd.Timestamp("2023-01-01")
    train_data = df_sp500[df_sp500["date"] < holdout_start]
    holdout_data = df_sp500[df_sp500["date"] >= holdout_start]

    log(f"  Train rows:   {len(train_data):,}  ({train_data['date'].min().date()} → {train_data['date'].max().date()})")
    log(f"  Holdout rows: {len(holdout_data):,}  ({holdout_data['date'].min().date()} → {holdout_data['date'].max().date()})")

    # Calibration split: last 20% of TRAINING dates
    train_dates = np.sort(train_data["date"].unique())
    calib_split = int(len(train_dates) * 0.80)
    calib_start = pd.Timestamp(train_dates[calib_split])

    tr_mask = train_data["date"] < calib_start
    cal_mask = train_data["date"] >= calib_start

    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(train_data.loc[tr_mask, feature_cols].values)
    y_tr = train_data.loc[tr_mask, "target_v4"].values
    X_cal = imputer.transform(train_data.loc[cal_mask, feature_cols].values)
    y_cal = train_data.loc[cal_mask, "target_v4"].values

    log(f"  RF training split: {tr_mask.sum():,} train + {cal_mask.sum():,} calib")

    # Train RF
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=100,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    log(f"  RF trained in {elapsed:.1f}s")

    # Calibrate on training calib set
    calib_rf = ManualCalibratedModel(rf, X_cal, y_cal)

    # AUC on training calib
    raw_auc = roc_auc_score(y_cal, rf.predict_proba(X_cal)[:, 1])
    log(f"  Train calib AUC: {raw_auc:.4f}")

    # Predict on HOLDOUT only
    X_holdout = imputer.transform(holdout_data[feature_cols].values)
    holdout_probs = calib_rf.predict_proba(X_holdout)[:, 1]

    holdout_auc = roc_auc_score(holdout_data["target_v4"].values, holdout_probs)
    log(f"  Holdout AUC:     {holdout_auc:.4f}")
    log(f"  AUC degradation: {holdout_auc - raw_auc:+.4f}")

    # Build prediction dataframe for holdout backtest
    holdout_preds = holdout_data[["date", "symbol", "target_v4", "in_sp500"]].copy()
    holdout_preds["prob_rf_holdout"] = holdout_probs
    holdout_preds["fwd_ret"] = holdout_data["fwd_10d_ret"]

    # Backtest on holdout period
    log(f"\n  Running Path B backtest on holdout (2023-2026) ...")
    m_rf_ho, _ = run_backtest_on_preds(holdout_preds, "prob_rf_holdout", "RF Holdout")

    # Also run LGBM on same holdout period for comparison
    log(f"  Running LGBM backtest on same holdout period ...")
    lgbm_preds = pd.read_parquet(LGBM_PREDS)
    lgbm_preds["date"] = pd.to_datetime(lgbm_preds["date"])
    m_lgbm_ho, _ = run_backtest_on_preds(lgbm_preds, "prob_v4", "LGBM Holdout",
                                          date_filter=("2023-01-01", "2026-12-31"))

    log(f"\n  {'─'*60}")
    log(f"  HOLDOUT RESULTS (2023-01-01 → 2026-04-17)")
    log(f"  {'─'*60}")
    log(f"  {'Model':<25s} {'CAGR':>8s} {'Sharpe':>8s} {'Max DD':>8s} {'Win%':>6s}")
    log(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

    if m_rf_ho:
        log(f"  {'RF (retrained, holdout)':<25s} {m_rf_ho['cagr']*100:>7.2f}% {m_rf_ho['sharpe']:>8.2f} {m_rf_ho['max_dd']*100:>7.1f}% {m_rf_ho['win_rate']*100:>5.1f}%")
    if m_lgbm_ho:
        log(f"  {'LGBM (holdout period)':<25s} {m_lgbm_ho['cagr']*100:>7.2f}% {m_lgbm_ho['sharpe']:>8.2f} {m_lgbm_ho['max_dd']*100:>7.1f}% {m_lgbm_ho['win_rate']*100:>5.1f}%")

    log(f"\n  Full-period reference:")
    log(f"  {'RF (full data, orig)':<25s} {'59.10%':>8s} {'2.77':>8s} {'-20.0%':>8s}")
    log(f"  {'LGBM (full data, orig)':<25s} {'34.60%':>8s} {'1.77':>8s} {'-24.3%':>8s}")

    # Interpretation
    if m_rf_ho:
        cagr = m_rf_ho["cagr"]
        if cagr >= 0.40:
            verdict = "LIKELY REAL ALPHA — holdout CAGR 40%+"
        elif cagr >= 0.20:
            verdict = "MODERATE OVERFIT — holdout CAGR 20-40%, use ensemble"
        else:
            verdict = "SEVERE OVERFIT — holdout CAGR < 20%, do NOT deploy RF standalone"
        log(f"\n  VERDICT: {verdict}")

    return m_rf_ho, m_lgbm_ho


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2: Feature Lookahead Audit
# ══════════════════════════════════════════════════════════════════════════════

def test2_lookahead():
    log(f"\n{'='*70}")
    log("TEST 2: FEATURE LOOKAHEAD AUDIT")
    log(f"{'='*70}")
    log("  Checking each feature for correlation with same-day target")
    log("  High corr (>0.3) = potential forward-looking leak")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = compute_rank_target(df)
    df = add_cross_sectional_features(df)

    feature_cols = get_feature_cols(df)
    valid = df.dropna(subset=["target_v4"])

    log(f"  Analyzing {len(feature_cols)} features on {len(valid):,} rows")

    results = []
    for col in feature_cols:
        col_data = valid[col].dropna()
        if len(col_data) < 1000:
            continue
        common = valid.loc[col_data.index]
        corr = common[col].corr(common["target_v4"])
        abs_corr = abs(corr)
        results.append((col, corr, abs_corr))

    results.sort(key=lambda x: x[2], reverse=True)

    # Report suspicious features (>0.10 is worth noting, >0.30 is alarming)
    log(f"\n  {'Feature':<35s} {'Corr':>8s} {'|Corr|':>8s} {'Risk':<15s}")
    log(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*15}")

    n_suspicious = 0
    for col, corr, abs_corr in results[:30]:  # show top 30
        if abs_corr > 0.30:
            risk = "*** ALARM ***"
            n_suspicious += 1
        elif abs_corr > 0.15:
            risk = "** WARNING **"
            n_suspicious += 1
        elif abs_corr > 0.10:
            risk = "* ELEVATED *"
        else:
            risk = "OK"
        log(f"  {col:<35s} {corr:>+8.4f} {abs_corr:>8.4f} {risk:<15s}")

    # Specifically check the categories mentioned
    log(f"\n  Category Analysis:")
    categories = {
        "Cross-sectional ranks": [c for c in feature_cols if "rank" in c.lower()],
        "Fundamental": [c for c in feature_cols if c in FUNDAMENTAL_FEATURE_COLS],
        "Macro (FRED)": [c for c in feature_cols if any(x in c for x in ["yield", "hy_spread", "dxy", "spy_ret"])],
        "Volatility": [c for c in feature_cols if "vol" in c.lower()],
        "Forward-looking keywords": [c for c in feature_cols if any(kw in c.lower() for kw in ["fwd", "forward", "future"])],
    }

    for cat_name, cat_cols in categories.items():
        if not cat_cols:
            log(f"  {cat_name}: (none found)")
            continue
        cat_corrs = [(c, corr) for c, corr, _ in results if c in cat_cols]
        max_corr = max(abs(c) for _, c in cat_corrs) if cat_corrs else 0
        log(f"  {cat_name} ({len(cat_cols)} features): max |corr| = {max_corr:.4f} {'⚠' if max_corr > 0.15 else '✓'}")

    if n_suspicious == 0:
        log(f"\n  VERDICT: No features with suspicious target correlation (>0.15)")
    else:
        log(f"\n  VERDICT: {n_suspicious} features with elevated correlation — review needed")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3: Temporal Stability (3-year chunks)
# ══════════════════════════════════════════════════════════════════════════════

def test3_temporal():
    log(f"\n{'='*70}")
    log("TEST 3: TEMPORAL STABILITY — 3-Year Chunks")
    log(f"{'='*70}")

    rf_preds = pd.read_parquet(RF_PREDS)
    lgbm_preds = pd.read_parquet(LGBM_PREDS)
    rf_preds["date"] = pd.to_datetime(rf_preds["date"])
    lgbm_preds["date"] = pd.to_datetime(lgbm_preds["date"])

    periods = [
        ("2012-2013", "2012-01-01", "2013-12-31"),
        ("2014-2016", "2014-01-01", "2016-12-31"),
        ("2017-2019", "2017-01-01", "2019-12-31"),
        ("2020-2022", "2020-01-01", "2022-12-31"),
        ("2023-2026", "2023-01-01", "2026-12-31"),
    ]

    log(f"\n  Fetching OHLCV data (one download, reused across periods) ...")
    all_syms_rf = sorted(rf_preds["symbol"].unique().tolist())
    all_syms = list(set(["SPY"] + all_syms_rf))
    raw = yf.download(all_syms, start="2011-01-01", end="2026-12-31",
                      auto_adjust=True, progress=False, threads=True)
    ohlcv_close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    ohlcv_close.index = pd.to_datetime(ohlcv_close.index).tz_localize(None)
    ohlcv_volume = None
    if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
        ohlcv_volume = raw["Volume"]
        ohlcv_volume.index = pd.to_datetime(ohlcv_volume.index).tz_localize(None)

    rf_results = {}
    lgbm_results = {}

    for period_name, start, end in periods:
        log(f"\n  ── {period_name} ──")

        for model_name, preds, prob_col, results_dict in [
            ("RF", rf_preds, "prob_rf", rf_results),
            ("LGBM", lgbm_preds, "prob_v4", lgbm_results),
        ]:
            from unified_backtester import (
                INITIAL_CASH, HOLD_DAYS, POSITION_PCT,
                Signal, Strategy,
                MomentumStrategy, MeanReversionStrategy,
                SLOT_ML_MOM_MR,
            )
            from backtest_utils import calc_metrics, calc_alpha_beta
            from backtest_v4_voltarget import instrumented_run_voltarget

            start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
            period_preds = preds[(preds["date"] >= start_dt) & (preds["date"] <= end_dt)]
            period_preds = period_preds.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

            if len(period_preds) == 0:
                log(f"    {model_name}: No data")
                continue

            all_dates = sorted(period_preds["date"].unique().tolist())
            years = (all_dates[-1] - all_dates[0]).days / 365.25
            if years < 0.5:
                log(f"    {model_name}: Period too short ({years:.1f}y)")
                continue

            class TopNStrategy(Strategy):
                def __init__(self, predictions_df, prob_column, top_n=TOP_N_PICKS, position_pct=POSITION_PCT):
                    self._top_n = top_n
                    self._position_pct = position_pct
                    self._signals_by_date = {}
                    for date, grp in predictions_df.groupby("date"):
                        sp500_grp = grp[grp["in_sp500"] == True]
                        if sp500_grp.empty:
                            continue
                        top = sp500_grp.nlargest(self._top_n, prob_column)
                        self._signals_by_date[date] = [
                            (row.symbol, getattr(row, prob_column), row.fwd_ret)
                            for row in top.itertuples(index=False)
                        ]

                @property
                def name(self):
                    return "ml_medium"

                def generate_signals(self, date, universe_data):
                    raw = self._signals_by_date.get(date, [])
                    return [Signal(symbol=s, confidence=p, strategy_name=self.name, fwd_ret=f)
                            for s, p, f in raw]

                def check_exit(self, position, current_data):
                    if current_data["idx"] >= position.exit_idx:
                        return True, "hold_complete"
                    return False, ""

                def get_position_size(self, signal, portfolio_value):
                    ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
                    return portfolio_value * self._position_pct * ml_mult

            sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
            close_aligned = ohlcv_close.reindex(sim_index, method="ffill")
            spy_px = close_aligned["SPY"].dropna()
            spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
            spy_dict = close_aligned["SPY"].to_dict()

            ml_strat = TopNStrategy(period_preds, prob_col, top_n=TOP_N_PICKS)
            mom_strat = MomentumStrategy(ohlcv_close, volume_data=ohlcv_volume)
            mr_strat = MeanReversionStrategy(ohlcv_close, volume_data=ohlcv_volume)

            vals, trades, diag = instrumented_run_voltarget(
                [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
                close_aligned, hold_days=HOLD_DAYS, label=f"{model_name} {period_name}")

            m = calc_metrics(vals, trades, years, f"{model_name} {period_name}")
            results_dict[period_name] = m
            log(f"    {model_name}: CAGR={m['cagr']*100:+.1f}%  Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.1f}%")

    # Summary table
    log(f"\n  {'─'*70}")
    log(f"  TEMPORAL STABILITY SUMMARY")
    log(f"  {'─'*70}")
    log(f"  {'Period':<12s} {'RF CAGR':>10s} {'RF Sharpe':>10s} {'LGBM CAGR':>10s} {'LGBM Sharpe':>12s} {'RF-LGBM':>8s}")
    log(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")

    rf_sharpes = []
    lgbm_sharpes = []
    for period_name, _, _ in periods:
        rf_m = rf_results.get(period_name)
        lgbm_m = lgbm_results.get(period_name)
        if rf_m and lgbm_m:
            diff = rf_m["sharpe"] - lgbm_m["sharpe"]
            log(f"  {period_name:<12s} {rf_m['cagr']*100:>+9.1f}% {rf_m['sharpe']:>10.2f} "
                f"{lgbm_m['cagr']*100:>+9.1f}% {lgbm_m['sharpe']:>12.2f} {diff:>+7.2f}")
            rf_sharpes.append(rf_m["sharpe"])
            lgbm_sharpes.append(lgbm_m["sharpe"])

    if rf_sharpes:
        rf_std = np.std(rf_sharpes)
        lgbm_std = np.std(lgbm_sharpes)
        log(f"\n  Sharpe std across periods: RF={rf_std:.2f}  LGBM={lgbm_std:.2f}")
        if rf_std > lgbm_std * 1.5:
            log(f"  ⚠ RF Sharpe is more volatile across periods — regime-dependent")
        else:
            log(f"  ✓ RF Sharpe stability comparable to LGBM")

        # Check for declining performance
        if len(rf_sharpes) >= 3:
            recent_avg = np.mean(rf_sharpes[-2:])
            early_avg = np.mean(rf_sharpes[:2])
            if recent_avg < early_avg * 0.5:
                log(f"  ⚠ RF performance degrading over time (recent: {recent_avg:.2f} vs early: {early_avg:.2f})")
            else:
                log(f"  ✓ No clear degradation trend")

    return rf_results, lgbm_results


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4: Prediction Distribution
# ══════════════════════════════════════════════════════════════════════════════

def test4_distribution():
    log(f"\n{'='*70}")
    log("TEST 4: PREDICTION DISTRIBUTION ANALYSIS")
    log(f"{'='*70}")

    rf_preds = pd.read_parquet(RF_PREDS)
    lgbm_preds = pd.read_parquet(LGBM_PREDS)

    for name, df, col in [("RF", rf_preds, "prob_rf"), ("LightGBM", lgbm_preds, "prob_v4")]:
        probs = df[col].dropna()
        log(f"\n  {name} ({len(probs):,} predictions):")
        log(f"    Range:  [{probs.min():.4f}, {probs.max():.4f}]")
        log(f"    Mean:   {probs.mean():.4f}")
        log(f"    Std:    {probs.std():.4f}")
        log(f"    Median: {probs.median():.4f}")

        # Distribution buckets
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00]
        log(f"    Distribution:")
        for t in thresholds:
            if t < 1.0:
                n = (probs >= t).sum()
                log(f"      >= {t:.2f}: {n:>8,} ({n/len(probs)*100:.3f}%)")
            else:
                n = (probs >= t - 0.0001).sum()
                log(f"      ~= {t:.2f}: {n:>8,} ({n/len(probs)*100:.3f}%)")

    # Compare extreme predictions
    rf_extreme = (rf_preds["prob_rf"] >= 0.90).sum()
    lgbm_extreme = (lgbm_preds["prob_v4"] >= 0.90).sum()
    log(f"\n  Extreme Predictions (>= 0.90):")
    log(f"    RF:      {rf_extreme:,} ({rf_extreme/len(rf_preds)*100:.3f}%)")
    log(f"    LightGBM: {lgbm_extreme:,} ({lgbm_extreme/len(lgbm_preds)*100:.3f}%)")

    if rf_extreme > lgbm_extreme * 10:
        log(f"  ⚠ RF has {rf_extreme/max(lgbm_extreme,1):.0f}x more extreme predictions — suggests overconfidence")
    elif rf_extreme > lgbm_extreme * 3:
        log(f"  ⚠ RF has {rf_extreme/max(lgbm_extreme,1):.1f}x more extreme predictions — mild concern")
    else:
        log(f"  ✓ RF extreme prediction rate comparable to LGBM")

    # Check for exactly 1.000
    rf_exact_1 = (rf_preds["prob_rf"] >= 0.9999).sum()
    log(f"\n  RF predictions exactly ~1.000: {rf_exact_1:,}")
    if rf_exact_1 > 100:
        log(f"  ⚠ HIGH COUNT of 1.000 predictions — isotonic calibration mapping leaf to 100% positive")
        log(f"    This happens when some training leaves had 100% positive rate")
        log(f"    Not necessarily a bug, but indicates calibration extrapolation")
    elif rf_exact_1 > 0:
        log(f"  ⚠ Some 1.000 predictions exist — minor calibration artifact")
    else:
        log(f"  ✓ No exactly-1.000 predictions")

    # Actual accuracy of extreme predictions
    rf_full = rf_preds.dropna(subset=["prob_rf"])
    rf_full_valid = rf_full.dropna(subset=["target_v4" if "target_v4" in rf_full.columns else "prob_rf"])
    if "target_v4" in rf_full_valid.columns:
        extreme_mask = rf_full_valid["prob_rf"] >= 0.80
        if extreme_mask.sum() > 0:
            extreme_acc = rf_full_valid.loc[extreme_mask, "target_v4"].mean()
            baseline_rate = rf_full_valid["target_v4"].mean()
            log(f"\n  Actual hit rate of RF prob >= 0.80:")
            log(f"    RF extreme hit rate: {extreme_acc*100:.1f}% ({extreme_mask.sum():,} predictions)")
            log(f"    Baseline positive rate: {baseline_rate*100:.1f}%")
            log(f"    Lift: {extreme_acc/max(baseline_rate, 0.01):.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()

    m_rf_ho, m_lgbm_ho = test1_holdout()
    feature_audit = test2_lookahead()
    rf_temporal, lgbm_temporal = test3_temporal()
    test4_distribution()

    # ── FINAL REPORT ──
    log(f"\n{'='*70}")
    log("FINAL REPORT: RF OVERFITTING INVESTIGATION")
    log(f"{'='*70}")

    log(f"\n  Test 1 — Holdout Backtest:")
    if m_rf_ho:
        log(f"    RF holdout CAGR:   {m_rf_ho['cagr']*100:.1f}%")
        log(f"    RF holdout Sharpe: {m_rf_ho['sharpe']:.2f}")
        log(f"    RF holdout DD:     {m_rf_ho['max_dd']*100:.1f}%")
        if m_lgbm_ho:
            log(f"    LGBM holdout CAGR: {m_lgbm_ho['cagr']*100:.1f}%")
            log(f"    LGBM holdout Sharpe: {m_lgbm_ho['sharpe']:.2f}")

    log(f"\n  Test 2 — Feature Lookahead:")
    n_alarm = sum(1 for _, _, ac in feature_audit if ac > 0.15)
    log(f"    Features with |corr| > 0.15: {n_alarm}")

    log(f"\n  Test 3 — Temporal Stability:")
    if rf_temporal:
        sharpes = [m["sharpe"] for m in rf_temporal.values()]
        log(f"    RF Sharpe range: {min(sharpes):.2f} – {max(sharpes):.2f}")
        log(f"    RF Sharpe std:   {np.std(sharpes):.2f}")

    log(f"\n  Test 4 — Prediction Distribution:")
    rf_preds_check = pd.read_parquet(RF_PREDS)
    n_extreme = (rf_preds_check["prob_rf"] >= 0.90).sum()
    n_exact1 = (rf_preds_check["prob_rf"] >= 0.9999).sum()
    log(f"    RF >= 0.90: {n_extreme:,}")
    log(f"    RF ~= 1.00: {n_exact1:,}")

    # Final recommendation
    log(f"\n  {'─'*60}")
    log(f"  RECOMMENDATION:")
    log(f"  {'─'*60}")

    issues = []
    if m_rf_ho and m_rf_ho["cagr"] < 0.20:
        issues.append("Holdout CAGR < 20% (severe overfit)")
    if n_alarm > 3:
        issues.append(f"{n_alarm} features with suspicious correlation")
    if n_exact1 > 1000:
        issues.append(f"Too many 1.000 predictions ({n_exact1:,})")
    if rf_temporal:
        sharpes = [m["sharpe"] for m in rf_temporal.values()]
        if np.std(sharpes) > 1.0:
            issues.append("RF Sharpe highly variable across periods")

    if not issues:
        log(f"  ✓ ALL TESTS PASSED — RF alpha appears REAL")
        log(f"    Recommend: Deploy RF standalone or LGBM+RF ensemble")
    elif len(issues) <= 1:
        log(f"  ⚠ MINOR CONCERNS:")
        for i in issues:
            log(f"    - {i}")
        log(f"    Recommend: Deploy as LGBM+RF ensemble (dilutes risk)")
    else:
        log(f"  ✗ SIGNIFICANT CONCERNS:")
        for i in issues:
            log(f"    - {i}")
        log(f"    Recommend: Do NOT deploy RF. Stick with LGBM V4.")

    elapsed = time.perf_counter() - t_start
    log(f"\n  Total investigation time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
