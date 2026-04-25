#!/usr/bin/env python3
"""
Feature Ablation Test (Full LGBM + XGB Ensemble)
==================================================
Test each new feature group individually to isolate which ones help:
  A: Baseline (83 features)
  B: +4 sector-relative features only
  C: +3 market breadth features only
  D: +1 dynamic beta only
  E: +7 sector-relative + breadth (no beta)
  F: +4 sector-relative + beta (no breadth)
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

# Feature groups
SECTOR_COLS = ["ret_10d_vs_sector", "ret_20d_vs_sector", "rsi_14_vs_sector", "vol_20d_vs_sector"]
BREADTH_COLS = ["pct_above_sma50", "advance_decline_5d", "breadth_thrust"]
BETA_COLS = ["beta_60d"]


def log(msg):
    print(msg, flush=True)


def get_feature_cols(df, extra_exclude=None):
    exclude = {"date", "symbol", "target", "target_v5", "in_sp500", "pct_rank", "sector"}
    if extra_exclude:
        exclude.update(extra_exclude)
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


def add_all_new_features(df):
    """Add all new features to df."""
    # Sector-relative
    df["sector"] = df["symbol"].map(SYMBOL_SECTOR).fillna("Other")
    for src, dst in [("ret_10d", "ret_10d_vs_sector"), ("ret_20d", "ret_20d_vs_sector"),
                     ("rsi_14", "rsi_14_vs_sector"), ("vol_20d", "vol_20d_vs_sector")]:
        if src in df.columns:
            median = df.groupby(["date", "sector"])[src].transform("median")
            df[dst] = df[src] - median
    df.drop(columns=["sector"], inplace=True, errors="ignore")

    # Market breadth
    if "dist_sma50" in df.columns:
        df["pct_above_sma50"] = df.groupby("date")["dist_sma50"].transform(
            lambda x: (x > 0).sum() / max(x.notna().sum(), 1)
        )
    if "ret_5d" in df.columns:
        def _ad_ratio(x):
            adv = (x > 0).sum(); dec = (x < 0).sum()
            total = adv + dec
            return (adv - dec) / total if total > 0 else 0.0
        df["advance_decline_5d"] = df.groupby("date")["ret_5d"].transform(_ad_ratio)

        unique_dates = sorted(df["date"].unique())
        date_adv = {}
        for d in unique_dates:
            r = df.loc[df["date"] == d, "ret_5d"]
            adv = (r > 0).sum(); total = r.notna().sum()
            date_adv[d] = adv / total if total > 0 else 0.5
        bt = pd.Series(date_adv).sort_index().ewm(span=10, min_periods=5).mean()
        df["breadth_thrust"] = df["date"].map(bt)

    # Dynamic beta
    if "ret_5d" in df.columns and "spy_ret_5d" in df.columns:
        df = df.sort_values(["symbol", "date"])
        betas = []
        for sym, grp in df.groupby("symbol"):
            cov = grp["ret_5d"].rolling(60, min_periods=60).cov(grp["spy_ret_5d"])
            var = grp["spy_ret_5d"].rolling(60, min_periods=60).var()
            betas.append(cov / var.replace(0, np.nan))
        df["beta_60d"] = pd.concat(betas)

    # Fill NaN
    for c in SECTOR_COLS + BREADTH_COLS + BETA_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


def prepare_features(df):
    """Same as walk_forward_validation.py."""
    for src, dst in [("vol_20d", "vol_rank_20d"), ("ret_60d", "momentum_rank_60d"),
                     ("rsi_14", "rsi_rank"), ("dist_sma50", "dist_sma50_rank")]:
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
    """Train LGBM + XGB ensemble for one year."""
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

    valid_train = ~np.isnan(y_train); valid_calib = ~np.isnan(y_calib)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_calib, y_calib = X_calib[valid_calib], y_calib[valid_calib]
    if len(y_train) < 500 or len(y_calib) < 50:
        return None

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    # LGBM
    model_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
    model_lgb.fit(X_train, y_train, eval_set=[(X_calib, y_calib)],
                  callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                             lgb.log_evaluation(period=-1)])
    calib_lgb = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
    calib_lgb.fit(X_calib, y_calib)

    # XGBoost
    model_xgb = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale)
    model_xgb.fit(X_train, y_train)
    calib_xgb = CalibratedClassifierCV(model_xgb, method="isotonic", cv="prefit")
    calib_xgb.fit(X_calib, y_calib)

    # Ensemble
    X_year = df_year[feature_cols].values
    lgbm_probs = calib_lgb.predict_proba(X_year)[:, 1]
    xgb_probs = calib_xgb.predict_proba(X_year)[:, 1]
    df_year["prob_ensemble"] = 0.5 * lgbm_probs + 0.5 * xgb_probs

    save_cols = ["date", "symbol", "prob_ensemble", "fwd_ret", "in_sp500"]
    for c in save_cols:
        if c not in df_year.columns:
            df_year[c] = np.nan

    # Feature importance (LGBM)
    imp = dict(zip(feature_cols, model_lgb.feature_importances_))

    return df_year[save_cols].copy(), imp


def run_backtest_year(preds_df, year, close, regime_dict):
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
    log("  FEATURE ABLATION TEST (LGBM + XGB Ensemble)")
    log("=" * 80)

    log(f"\nLoading {INPUT_FILE} ...")
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = prepare_features(df)

    base_cols = get_feature_cols(df)
    non_fund = [c for c in base_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund + ["target_v5"]).copy()
    log(f"  {len(df):,} rows | {len(base_cols)} base features")

    # Add all new features
    log("\nComputing new features ...")
    df = add_all_new_features(df)

    # Load price data + regime
    all_syms = sorted(df["symbol"].unique())
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    spy_full = close["SPY"].dropna()
    regime_dict = compute_regime_live(spy_full).to_dict()

    # Define configs: each is (label, short, columns_to_exclude_from_new)
    configs = [
        ("A: Baseline",          "Base",   SECTOR_COLS + BREADTH_COLS + BETA_COLS),
        ("B: +Sector-relative",  "+Sect",  BREADTH_COLS + BETA_COLS),
        ("C: +Market breadth",   "+Brdth", SECTOR_COLS + BETA_COLS),
        ("D: +Dynamic beta",     "+Beta",  SECTOR_COLS + BREADTH_COLS),
        ("E: +Sector+Breadth",   "+S+B",   BETA_COLS),
        ("F: +Sector+Beta",      "+S+Bt",  BREADTH_COLS),
        ("G: All 8 new",         "+All",   []),
    ]

    all_results = {label: [] for label, _, _ in configs}
    all_importances = {label: {} for label, _, _ in configs}

    for year in YEARS:
        log(f"\n{'─'*70}")
        log(f"  YEAR {year}")
        log(f"{'─'*70}")

        # Prep close for this year
        first_result = None
        close_year = None

        for label, short, exclude_cols in configs:
            feat_cols = [c for c in get_feature_cols(df) if c not in exclude_cols]

            t1 = time.perf_counter()
            result = train_ensemble_year(df, feat_cols, year)
            if result is None:
                log(f"  {short:6s}  [SKIP]")
                continue

            preds, imp = result
            preds_year = preds[preds["date"].dt.year == year]

            if close_year is None:
                all_dates = sorted(preds_year["date"].unique())
                sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
                close_year = close.reindex(close.index.union(sim_index), method="ffill")

            metrics = run_backtest_year(preds_year, year, close_year, regime_dict)
            elapsed = time.perf_counter() - t1

            if metrics:
                all_results[label].append(metrics)
                log(f"  {short:6s}  CAGR={metrics['cagr']:+.1%}  "
                    f"Sharpe={metrics['sharpe']:.2f}  DD={metrics['max_dd']:.1%}  "
                    f"({elapsed:.0f}s)")

                # Accumulate importance
                for k, v in imp.items():
                    all_importances[label][k] = all_importances[label].get(k, 0) + v

    # ── Report ──
    print("\n" + "=" * 110)
    print("FEATURE ABLATION — RESULTS (LGBM + XGB Ensemble)")
    print("=" * 110)

    base_results = all_results["A: Baseline"]
    if not base_results:
        print("No baseline results!")
        return

    base_med = np.median([m["cagr"] for m in base_results])
    base_shp = np.median([m["sharpe"] for m in base_results])
    base_dd = min([m["max_dd"] for m in base_results])

    print(f"\n{'Config':<25s} {'MedCAGR':>8s} {'MnCAGR':>8s} {'MedShp':>7s} "
          f"{'WrstDD':>8s} {'MnDD':>8s} {'Win%':>5s}"
          f"  {'dCAGR':>7s} {'dShp':>6s} {'dDD':>6s}")
    print("─" * 110)

    for label, short, _ in configs:
        results = all_results[label]
        if not results:
            continue
        cagrs = [m["cagr"] for m in results]
        sharpes = [m["sharpe"] for m in results]
        dds = [m["max_dd"] for m in results]
        win = sum(1 for c in cagrs if c > 0) / len(cagrs)

        d_cagr = np.median(cagrs) - base_med
        d_shp = np.median(sharpes) - base_shp
        d_dd = min(dds) - base_dd

        print(f"  {label:<23s} {np.median(cagrs):>+7.1%} {np.mean(cagrs):>+7.1%} "
              f"{np.median(sharpes):>7.2f} {min(dds):>8.1%} {np.mean(dds):>8.1%} "
              f"{win:>4.0%}"
              f"  {d_cagr:>+6.1%} {d_shp:>+5.2f} {d_dd:>+5.1%}")

    # Per-year table
    n = len(base_results)
    print(f"\n{'Year':>6s}", end="")
    for _, short, _ in configs:
        print(f" {short:>8s}", end="")
    print("  (CAGR)")
    print("─" * (6 + 9 * len(configs) + 8))

    for yi in range(n):
        yr = base_results[yi]["year"]
        print(f"{yr:>6d}", end="")
        for label, _, _ in configs:
            results = all_results[label]
            if yi < len(results):
                print(f" {results[yi]['cagr']:>+7.1%}", end="")
            else:
                print(f" {'N/A':>8s}", end="")
        print()

    # Feature importance for new features (from "All 8" config)
    all_imp = all_importances.get("G: All 8 new", {})
    if all_imp:
        new_feat_names = SECTOR_COLS + BREADTH_COLS + BETA_COLS
        print(f"\n{'─'*60}")
        print("New feature importance (summed across years, from G: All 8):")
        total_imp = sum(all_imp.values())
        for f in sorted(new_feat_names, key=lambda x: all_imp.get(x, 0), reverse=True):
            v = all_imp.get(f, 0)
            pct = v / total_imp * 100 if total_imp > 0 else 0
            print(f"  {f:<25s} {v:>6.0f}  ({pct:.1f}%)")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
