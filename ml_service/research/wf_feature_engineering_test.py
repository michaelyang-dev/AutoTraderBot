#!/usr/bin/env python3
"""
Walk-Forward Feature Engineering A/B Test (LGBM + XGB Ensemble)
================================================================
Compare OLD features vs NEW features (sector-relative, market breadth,
dynamic beta) using full LGBM+XGB ensemble walk-forward.

New features (8 total):
  1. Sector-relative: ret_10d, ret_20d, rsi_14, vol_20d vs sector median
  2. Market breadth: pct_above_sma50, advance_decline_5d, breadth_thrust
  3. Dynamic beta: rolling 60-day beta vs SPY
"""

import os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    CautiousMLStrategy, MomentumStrategy, PortfolioManager, SlotConfig,
    SYMBOL_SECTOR, load_bars_cached, compute_regime_live,
    INITIAL_CASH, DATA_DIR,
)
from backtest_utils import calc_metrics, calc_alpha_beta

WF_DIR = DATA_DIR / "walkforward"
INPUT_FILE = DATA_DIR / "features.parquet"
YEARS = list(range(2015, 2026))

LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, objective="binary", metric="auc",
    random_state=42, n_jobs=-1, verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    tree_method="hist", n_jobs=1, random_state=42,
    eval_metric="auc",
)
EARLY_STOP_ROUNDS = 100

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

SLOT_CONFIG = SlotConfig(
    strategy_slots={"ml_medium": 5, "momentum": 3},
    flex_slots=0, max_positions=8,
)


def log(msg):
    print(msg, flush=True)


def get_feature_cols(df):
    exclude = {"date", "symbol", "target", "target_v5", "in_sp500", "pct_rank", "sector"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


def add_new_features(df):
    """Add sector-relative, market breadth, dynamic beta features."""
    log("  Adding new features ...")

    # ── 1. Sector-relative features ──
    df["sector"] = df["symbol"].map(SYMBOL_SECTOR).fillna("Other")
    for src, dst in [("ret_10d", "ret_10d_vs_sector"), ("ret_20d", "ret_20d_vs_sector"),
                     ("rsi_14", "rsi_14_vs_sector"), ("vol_20d", "vol_20d_vs_sector")]:
        if src in df.columns:
            median = df.groupby(["date", "sector"])[src].transform("median")
            df[dst] = df[src] - median
    log(f"    Sector-relative: 4 features")

    # ── 2. Market breadth ──
    if "dist_sma50" in df.columns:
        pct_above = df.groupby("date")["dist_sma50"].transform(
            lambda x: (x > 0).sum() / max(x.notna().sum(), 1)
        )
        df["pct_above_sma50"] = pct_above

    if "ret_5d" in df.columns:
        def _ad_ratio(x):
            adv = (x > 0).sum()
            dec = (x < 0).sum()
            total = adv + dec
            return (adv - dec) / total if total > 0 else 0.0
        df["advance_decline_5d"] = df.groupby("date")["ret_5d"].transform(_ad_ratio)

        unique_dates = sorted(df["date"].unique())
        date_adv_ratio = {}
        for d in unique_dates:
            r = df.loc[df["date"] == d, "ret_5d"]
            adv = (r > 0).sum()
            total = r.notna().sum()
            date_adv_ratio[d] = adv / total if total > 0 else 0.5
        adv_series = pd.Series(date_adv_ratio).sort_index()
        bt = adv_series.ewm(span=10, min_periods=5).mean()
        df["breadth_thrust"] = df["date"].map(bt)
    log(f"    Market breadth: 3 features")

    # ── 3. Dynamic beta ──
    log("    Computing dynamic beta ...")
    if "ret_5d" in df.columns and "spy_ret_5d" in df.columns:
        df = df.sort_values(["symbol", "date"])
        betas = []
        for sym, grp in df.groupby("symbol"):
            s_ret = grp["ret_5d"]
            spy_ret = grp["spy_ret_5d"]
            cov = s_ret.rolling(60, min_periods=60).cov(spy_ret)
            var = spy_ret.rolling(60, min_periods=60).var()
            beta = cov / var.replace(0, np.nan)
            betas.append(beta)
        df["beta_60d"] = pd.concat(betas)
        log(f"    Dynamic beta: {df['beta_60d'].notna().sum():,} valid values")
    else:
        df["beta_60d"] = np.nan

    df.drop(columns=["sector"], inplace=True, errors="ignore")

    # Fill NaN in new features with 0 (neutral)
    for c in ["ret_10d_vs_sector", "ret_20d_vs_sector", "rsi_14_vs_sector",
              "vol_20d_vs_sector", "pct_above_sma50", "advance_decline_5d",
              "breadth_thrust", "beta_60d"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


def prepare_features(df):
    """Prepare features with rank features and target_v5."""
    RANK_FEATURES = [
        ("vol_20d", "vol_rank_20d"),
        ("ret_60d", "momentum_rank_60d"),
        ("rsi_14", "rsi_rank"),
        ("dist_sma50", "dist_sma50_rank"),
    ]
    for src, dst in RANK_FEATURES:
        if src in df.columns:
            df[dst] = df.groupby("date")[src].rank(pct=True)

    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    valid = df[sp500_mask & has_fwd].copy()
    valid["pct_rank"] = valid.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid["target_v5"] = (valid["pct_rank"] >= 0.80).astype(int)
    df.loc[valid.index, "target_v5"] = valid["target_v5"]
    df["fwd_ret"] = df["fwd_10d_ret"]
    return df


def train_ensemble_year(df_full, feature_cols, year):
    """Train LGBM + XGB ensemble for a single walk-forward year."""
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    calib_start = pd.Timestamp(f"{year - 2}-01-01")
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    train_all = df_full[df_full["date"] <= train_end].copy()
    sp500_train = train_all[train_all["in_sp500"] == True].copy()

    calib_mask = sp500_train["date"] >= calib_start
    pure_train_mask = sp500_train["date"] < calib_start

    year_mask = (df_full["date"] >= year_start) & (df_full["date"] <= year_end)
    df_year = df_full[year_mask].copy()

    if pure_train_mask.sum() < 1000 or calib_mask.sum() < 100 or len(df_year) < 50:
        return None

    X_train = sp500_train.loc[pure_train_mask, feature_cols].values
    y_train = sp500_train.loc[pure_train_mask, "target_v5"].values
    X_calib = sp500_train.loc[calib_mask, feature_cols].values
    y_calib = sp500_train.loc[calib_mask, "target_v5"].values

    valid_train = ~np.isnan(y_train)
    valid_calib = ~np.isnan(y_calib)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_calib, y_calib = X_calib[valid_calib], y_calib[valid_calib]

    if len(y_train) < 500 or len(y_calib) < 50:
        return None

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    # ── LGBM ──
    model_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_calib, y_calib)],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    lgb_trees = model_lgb.booster_.num_trees()

    calib_lgb = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
    calib_lgb.fit(X_calib, y_calib)
    lgbm_auc = roc_auc_score(y_calib, calib_lgb.predict_proba(X_calib)[:, 1])

    # ── XGBoost ──
    xgb_ok = True
    try:
        model_xgb = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale)
        model_xgb.fit(X_train, y_train)
        calib_xgb = CalibratedClassifierCV(model_xgb, method="isotonic", cv="prefit")
        calib_xgb.fit(X_calib, y_calib)
        xgb_auc = roc_auc_score(y_calib, calib_xgb.predict_proba(X_calib)[:, 1])
    except Exception as e:
        log(f"    [WARN] XGBoost failed: {e}")
        xgb_ok = False
        xgb_auc = float("nan")

    # ── Ensemble predictions ──
    X_year = df_year[feature_cols].values
    lgbm_probs = calib_lgb.predict_proba(X_year)[:, 1]

    if xgb_ok:
        xgb_probs = calib_xgb.predict_proba(X_year)[:, 1]
        ensemble_probs = 0.5 * lgbm_probs + 0.5 * xgb_probs
    else:
        ensemble_probs = lgbm_probs

    df_year["prob_ensemble"] = ensemble_probs

    save_cols = ["date", "symbol", "prob_ensemble", "fwd_ret", "in_sp500"]
    for c in save_cols:
        if c not in df_year.columns:
            df_year[c] = np.nan
    preds_df = df_year[save_cols].copy()

    return preds_df, lgbm_auc, xgb_auc, lgb_trees, xgb_ok


def run_backtest_year(preds_df, year, close, regime_dict):
    """Run ML + Momentum backtest for one year."""
    if "prob" not in preds_df.columns and "prob_ensemble" in preds_df.columns:
        preds_df = preds_df.rename(columns={"prob_ensemble": "prob"})

    all_dates = sorted(preds_df["date"].unique().tolist())
    if len(all_dates) < 10:
        return None

    years_span = (all_dates[-1] - all_dates[0]).days / 365.25
    if years_span <= 0:
        years_span = len(all_dates) / 252.0

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        return None
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    ml = CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                            threshold=0.55, top_n=5, selection_mode="top_n")
    mom = MomentumStrategy(close, volume_data=None, regime_filter=True)
    pm = PortfolioManager(strategies=[ml, mom], slot_config=SLOT_CONFIG)
    vals, trades = pm.run(all_dates, spy_prices=None, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"y{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    return metrics


def main():
    t0 = time.perf_counter()

    log("=" * 80)
    log("  FEATURE ENGINEERING A/B TEST (LGBM + XGB Ensemble)")
    log("  A: Old features (83 cols)")
    log("  B: New features (+8 cols: sector-rel, breadth, beta)")
    log("=" * 80)

    log(f"\nLoading {INPUT_FILE} ...")
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    log(f"  {len(df):,} rows | {df['symbol'].nunique()} symbols")

    df = prepare_features(df)

    old_feature_cols = get_feature_cols(df)
    non_fund = [c for c in old_feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df_clean = df.dropna(subset=non_fund + ["target_v5"]).copy()
    log(f"  Old features: {len(old_feature_cols)} columns, {len(df_clean):,} clean rows")

    # Add new features
    df_new = add_new_features(df_clean.copy())
    new_feature_cols = get_feature_cols(df_new)
    new_only = [c for c in new_feature_cols if c not in old_feature_cols]
    log(f"  New features: {len(new_feature_cols)} columns")
    log(f"  New-only columns: {new_only}")

    # Load price data
    all_syms = sorted(df["symbol"].unique())
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    spy_full = close["SPY"].dropna()
    regime_series = compute_regime_live(spy_full)
    regime_dict = regime_series.to_dict()

    results_old = []
    results_new = []

    for year in YEARS:
        log(f"\n{'─'*70}")
        log(f"  YEAR {year}")
        log(f"{'─'*70}")

        # ── A: Old features ──
        t1 = time.perf_counter()
        result_a = train_ensemble_year(df_clean, old_feature_cols, year)
        if result_a is None:
            log(f"  [SKIP] insufficient data")
            continue
        preds_a, lgbm_auc_a, xgb_auc_a, trees_a, xgb_ok_a = result_a

        preds_year = preds_a[preds_a["date"].dt.year == year]
        all_dates = sorted(preds_year["date"].unique())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")
        metrics_a = run_backtest_year(preds_year, year, close_year, regime_dict)
        elapsed_a = time.perf_counter() - t1
        if metrics_a:
            results_old.append(metrics_a)
            log(f"  OLD:  LGBM={lgbm_auc_a:.4f} XGB={xgb_auc_a:.4f} "
                f"trees={trees_a}  CAGR={metrics_a['cagr']:+.1%}  "
                f"Sharpe={metrics_a['sharpe']:.2f}  DD={metrics_a['max_dd']:.1%}  "
                f"({elapsed_a:.1f}s)")

        # ── B: New features ──
        t1 = time.perf_counter()
        result_b = train_ensemble_year(df_new, new_feature_cols, year)
        if result_b is None:
            continue
        preds_b, lgbm_auc_b, xgb_auc_b, trees_b, xgb_ok_b = result_b

        preds_year_b = preds_b[preds_b["date"].dt.year == year]
        metrics_b = run_backtest_year(preds_year_b, year, close_year, regime_dict)
        elapsed_b = time.perf_counter() - t1
        if metrics_b:
            results_new.append(metrics_b)
            log(f"  NEW:  LGBM={lgbm_auc_b:.4f} XGB={xgb_auc_b:.4f} "
                f"trees={trees_b}  CAGR={metrics_b['cagr']:+.1%}  "
                f"Sharpe={metrics_b['sharpe']:.2f}  DD={metrics_b['max_dd']:.1%}  "
                f"({elapsed_b:.1f}s)")

    # ── Report ──
    print("\n" + "=" * 100)
    print("FEATURE ENGINEERING A/B TEST — RESULTS (LGBM + XGB Ensemble)")
    print("=" * 100)

    configs = [
        ("A: Old features (83 cols)", results_old),
        ("B: +8 new features (91 cols)", results_new),
    ]

    print(f"\n{'Config':<32s} {'MedCAGR':>8s} {'MnCAGR':>8s} {'MedShp':>7s} "
          f"{'WrstDD':>8s} {'MnDD':>8s} {'WinRate':>8s}")
    print("─" * 100)

    for label, results in configs:
        if not results:
            print(f"  {label:<30s}  (no results)")
            continue
        cagrs = [m["cagr"] for m in results]
        sharpes = [m["sharpe"] for m in results]
        dds = [m["max_dd"] for m in results]
        win_rate = sum(1 for c in cagrs if c > 0) / len(cagrs)

        print(f"  {label:<30s} {np.median(cagrs):>+7.1%} {np.mean(cagrs):>+7.1%} "
              f"{np.median(sharpes):>7.2f} {min(dds):>8.1%} {np.mean(dds):>8.1%} "
              f"{win_rate:>7.0%}")

    # Delta
    if results_old and results_new:
        d_cagr = np.median([m["cagr"] for m in results_new]) - np.median([m["cagr"] for m in results_old])
        d_shp = np.median([m["sharpe"] for m in results_new]) - np.median([m["sharpe"] for m in results_old])
        d_dd = min([m["max_dd"] for m in results_new]) - min([m["max_dd"] for m in results_old])
        print(f"\n  {'DELTA (B - A)':<30s} {d_cagr:>+7.1%} {'':>8s} "
              f"{d_shp:>+7.2f} {d_dd:>+7.1%}")

    # Per-year comparison
    if results_old:
        print(f"\n{'Year':>6s}   {'OLD_CAGR':>9s} {'OLD_DD':>8s} {'OLD_Shp':>8s}"
              f"   {'NEW_CAGR':>9s} {'NEW_DD':>8s} {'NEW_Shp':>8s}"
              f"   {'dCAGR':>7s} {'dDD':>7s}")
        print("─" * 100)

        n = min(len(results_old), len(results_new))
        for yi in range(n):
            yr = results_old[yi]["year"]
            a = results_old[yi]
            b = results_new[yi]
            dc = b["cagr"] - a["cagr"]
            dd = b["max_dd"] - a["max_dd"]
            print(f"{yr:>6d}   {a['cagr']:>+8.1%} {a['max_dd']:>8.1%} {a['sharpe']:>8.2f}"
                  f"   {b['cagr']:>+8.1%} {b['max_dd']:>8.1%} {b['sharpe']:>8.2f}"
                  f"   {dc:>+6.1%} {dd:>+6.1%}")

        # Win/loss summary
        wins = sum(1 for yi in range(n)
                   if results_new[yi]["cagr"] > results_old[yi]["cagr"])
        print(f"\n  New features win {wins}/{n} years on CAGR")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
