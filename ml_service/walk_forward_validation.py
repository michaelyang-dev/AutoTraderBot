#!/usr/bin/env python3
"""
Walk-Forward Validation
=======================
Train 11 annual models (2015-2025), each using ONLY data from prior years.
Generate truly out-of-sample predictions for each year, run backtests,
and produce a consistency report.

For year Y:
  - Train on data from 2012-01-01 through (Y-1)-12-31
  - Calibrate on last year of training data (Y-2 to Y-1)
  - Predict on year Y only
  - Backtest year Y with those predictions

Artifacts saved to ml_service/data/walkforward/:
  - predictions_YYYY.parquet   (per-year OOS predictions)
  - metrics.json               (all years' backtest metrics)
  - REPORT.md                  (consistency analysis)

Run:
    python3 walk_forward_validation.py
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path

# Prevent OpenMP deadlock on macOS ARM64
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    MegaCapStrategy, PortfolioManager, SlotConfig,
    SLOT_LIVE, load_bars_cached, load_predictions_cached,
    INITIAL_CASH, DATA_DIR,
)
from backtest_utils import calc_metrics, calc_alpha_beta

# ── Config ────────────────────────────────────────────────────────────────────
WF_DIR = DATA_DIR / "walkforward"
INPUT_FILE = DATA_DIR / "features.parquet"

YEARS = list(range(2015, 2026))  # 2015..2025

# Same model params as train_validation_model_fast.py
LGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="binary",
    metric="auc",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=1,  # single-threaded to avoid macOS deadlock
    random_state=42,
    eval_metric="auc",
    use_label_encoder=False,
)

EARLY_STOP_ROUNDS = 100

RANK_FEATURES = [
    ("vol_20d", "vol_rank_20d"),
    ("ret_60d", "momentum_rank_60d"),
    ("rsi_14", "rsi_rank"),
    ("dist_sma50", "dist_sma50_rank"),
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

# Backtest slot config — combined_live preset
STRAT_NAMES = ["ml", "momentum", "mean_reversion", "mega_cap"]


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target", "target_v5", "in_sp500", "pct_rank"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank features and target, drop NaN rows."""
    # Cross-sectional rank features
    for base_col, rank_col in RANK_FEATURES:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    # Cross-sectional target: top 20% by forward 10-day return
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_v5"] = np.nan
    valid = df[sp500_mask & has_fwd].copy()
    valid["pct_rank"] = valid.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid["target_v5"] = (valid["pct_rank"] >= 0.80).astype(int)
    df.loc[valid.index, "target_v5"] = valid["target_v5"]

    # Forward return for predictions
    df["fwd_ret"] = df["fwd_10d_ret"]

    return df


def train_year_model(df_full, feature_cols, year):
    """
    Train LGBM + XGB for a single walk-forward year.

    Training data: all dates < year
    Calibration: last year of training data (year-1)
    Prediction: year Y only

    Returns (predictions_df, auc_dict) or None if insufficient data.
    """
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    calib_start = pd.Timestamp(f"{year - 2}-01-01")
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    # Split
    train_all = df_full[df_full["date"] <= train_end].copy()
    sp500_train = train_all[train_all["in_sp500"] == True].copy()

    # Calibration = last year of training period
    calib_mask = sp500_train["date"] >= calib_start
    pure_train_mask = sp500_train["date"] < calib_start

    # Year to predict
    year_mask = (df_full["date"] >= year_start) & (df_full["date"] <= year_end)
    df_year = df_full[year_mask].copy()

    if pure_train_mask.sum() < 1000 or calib_mask.sum() < 100 or len(df_year) < 50:
        log(f"  [SKIP] Insufficient data for year {year}")
        return None

    X_train = sp500_train.loc[pure_train_mask, feature_cols].values
    y_train = sp500_train.loc[pure_train_mask, "target_v5"].values
    X_calib = sp500_train.loc[calib_mask, feature_cols].values
    y_calib = sp500_train.loc[calib_mask, "target_v5"].values

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    log(f"  Train: {pure_train_mask.sum():,} rows | Calib: {calib_mask.sum():,} rows | "
        f"Predict: {len(df_year):,} rows | scale_pos_weight={scale:.2f}")

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
    n_trees = model_lgb.booster_.num_trees()

    calib_lgbm = CalibratedClassifierCV(model_lgb, method="isotonic", cv="prefit")
    calib_lgbm.fit(X_calib, y_calib)

    lgbm_calib_probs = calib_lgbm.predict_proba(X_calib)[:, 1]
    lgbm_auc = roc_auc_score(y_calib, lgbm_calib_probs) if len(np.unique(y_calib)) > 1 else float("nan")

    # ── XGBoost ──
    xgb_ok = True
    try:
        model_xgb = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale)
        model_xgb.fit(X_train, y_train)
        calib_xgb = CalibratedClassifierCV(model_xgb, method="isotonic", cv="prefit")
        calib_xgb.fit(X_calib, y_calib)
        xgb_calib_probs = calib_xgb.predict_proba(X_calib)[:, 1]
        xgb_auc = roc_auc_score(y_calib, xgb_calib_probs) if len(np.unique(y_calib)) > 1 else float("nan")
    except Exception as e:
        log(f"  [WARN] XGBoost failed: {e} — using LGBM-only")
        xgb_ok = False
        xgb_auc = float("nan")

    # ── Generate predictions for year Y ──
    X_year = df_year[feature_cols].values
    lgbm_probs = calib_lgbm.predict_proba(X_year)[:, 1]

    if xgb_ok:
        xgb_probs = calib_xgb.predict_proba(X_year)[:, 1]
        ensemble_probs = 0.5 * lgbm_probs + 0.5 * xgb_probs
    else:
        xgb_probs = lgbm_probs  # fallback
        ensemble_probs = lgbm_probs

    df_year["prob_lgbm"] = lgbm_probs
    df_year["prob_rf"] = xgb_probs
    df_year["prob_ensemble"] = ensemble_probs

    log(f"  LGBM: {n_trees} trees, AUC={lgbm_auc:.4f} | "
        f"XGB: {'OK' if xgb_ok else 'SKIP'} AUC={xgb_auc:.4f} | "
        f"Ensemble range=[{ensemble_probs.min():.3f}, {ensemble_probs.max():.3f}]")

    save_cols = ["date", "symbol", "target_v5", "prob_lgbm", "prob_rf",
                 "prob_ensemble", "fwd_ret", "in_sp500"]
    preds_df = df_year[save_cols].copy()

    auc_dict = {"lgbm_calib_auc": lgbm_auc, "xgb_calib_auc": xgb_auc, "xgb_ok": xgb_ok}
    return preds_df, auc_dict


def run_backtest_for_year(preds_df, year, momentum_regime_filter=False):
    """Run combined_live backtest for a single year using provided predictions."""
    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())

    if len(all_dates) < 10:
        return None

    years_span = (all_dates[-1] - all_dates[0]).days / 365.25
    if years_span <= 0:
        years_span = len(all_dates) / 252.0

    # Fetch price bars
    start_str = (pd.Timestamp(all_dates[0]) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close = load_bars_cached(universe_syms, start_str, end_str)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close = close.reindex(sim_index, method="ffill")

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        close = load_bars_cached(universe_syms, start_str, end_str, no_cache=True)
        close = close.reindex(sim_index, method="ffill")
        spy_px = close["SPY"].dropna()
        if spy_px.empty:
            log(f"  [WARN] No SPY data for {year}")
            return None

    spy_dict = close["SPY"].to_dict()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    # Normalize prob column
    if "prob_ensemble" in preds_df.columns and "prob" not in preds_df.columns:
        preds_df = preds_df.rename(columns={"prob_ensemble": "prob"})

    # Build strategies
    strategies = []
    strategies.append(MLMediumStrategy(preds_df, threshold=0.55, top_n=5, selection_mode="top_n"))
    strategies.append(MomentumStrategy(close, volume_data=None, regime_filter=momentum_regime_filter))
    strategies.append(MeanReversionStrategy(close, volume_data=None))
    strategies.append(MegaCapStrategy(close))

    pm = PortfolioManager(strategies=strategies, slot_config=SLOT_LIVE)
    vals, trades = pm.run(all_dates, spy_prices=spy_dict, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"combined_live_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year

    return metrics


def generate_report(results_no_filter, results_with_filter, auc_by_year):
    """Generate REPORT.md with consistency analysis."""
    lines = []
    lines.append("# Walk-Forward Validation Report")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\nYears tested: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} independent periods)")
    lines.append("\n## Per-Year Results (combined_live, no momentum filter)\n")

    # Table header
    lines.append("| Year | CAGR | Sharpe | Sortino | Max DD | Trades | Win% | Alpha | LGBM AUC | XGB AUC |")
    lines.append("|------|------|--------|---------|--------|--------|------|-------|----------|---------|")

    for m in results_no_filter:
        y = m["year"]
        auc = auc_by_year.get(y, {})
        lgbm_auc = auc.get("lgbm_calib_auc", float("nan"))
        xgb_auc = auc.get("xgb_calib_auc", float("nan"))
        lines.append(
            f"| {y} "
            f"| {m['cagr']:+.1%} "
            f"| {m['sharpe']:.2f} "
            f"| {m['sortino']:.2f} "
            f"| {m['max_dd']:.1%} "
            f"| {m['n_trades']} "
            f"| {m['win_rate']:.1%} "
            f"| {'+' if m.get('alpha') and m['alpha'] > 0 else ''}{m.get('alpha', 0) or 0:.1%} "
            f"| {lgbm_auc:.3f} "
            f"| {xgb_auc:.3f} |"
        )

    # Summary stats
    cagrs = [m["cagr"] for m in results_no_filter]
    sharpes = [m["sharpe"] for m in results_no_filter]
    max_dds = [m["max_dd"] for m in results_no_filter]
    win_rates = [m["win_rate"] for m in results_no_filter if not np.isnan(m["win_rate"])]
    alphas = [m["alpha"] for m in results_no_filter if m.get("alpha") is not None]

    lines.append("")
    lines.append("### Summary Statistics (no filter)")
    lines.append(f"- **Median CAGR**: {np.median(cagrs):+.1%}")
    lines.append(f"- **Mean CAGR**: {np.mean(cagrs):+.1%}")
    lines.append(f"- **CAGR range**: [{min(cagrs):+.1%}, {max(cagrs):+.1%}]")
    lines.append(f"- **Years with positive CAGR**: {sum(1 for c in cagrs if c > 0)}/{len(cagrs)}")
    lines.append(f"- **Median Sharpe**: {np.median(sharpes):.2f}")
    lines.append(f"- **Mean Sharpe**: {np.mean(sharpes):.2f}")
    lines.append(f"- **Sharpe > 1.0**: {sum(1 for s in sharpes if s > 1.0)}/{len(sharpes)} years")
    lines.append(f"- **Worst max DD**: {min(max_dds):.1%}")
    if win_rates:
        lines.append(f"- **Mean win rate**: {np.mean(win_rates):.1%}")
    if alphas:
        lines.append(f"- **Years with positive alpha**: {sum(1 for a in alphas if a > 0)}/{len(alphas)}")

    # Momentum filter comparison
    if results_with_filter:
        lines.append("\n## Momentum Regime Filter Comparison\n")
        lines.append("| Year | CAGR (no) | CAGR (filter) | Sharpe (no) | Sharpe (filter) | DD (no) | DD (filter) |")
        lines.append("|------|-----------|---------------|-------------|-----------------|---------|-------------|")

        for m_no, m_yes in zip(results_no_filter, results_with_filter):
            y = m_no["year"]
            lines.append(
                f"| {y} "
                f"| {m_no['cagr']:+.1%} "
                f"| {m_yes['cagr']:+.1%} "
                f"| {m_no['sharpe']:.2f} "
                f"| {m_yes['sharpe']:.2f} "
                f"| {m_no['max_dd']:.1%} "
                f"| {m_yes['max_dd']:.1%} |"
            )

        cagrs_f = [m["cagr"] for m in results_with_filter]
        sharpes_f = [m["sharpe"] for m in results_with_filter]
        max_dds_f = [m["max_dd"] for m in results_with_filter]

        lines.append("")
        lines.append("### Filter Impact")
        lines.append(f"- **Median CAGR**: {np.median(cagrs):+.1%} -> {np.median(cagrs_f):+.1%}")
        lines.append(f"- **Median Sharpe**: {np.median(sharpes):.2f} -> {np.median(sharpes_f):.2f}")
        lines.append(f"- **Worst DD**: {min(max_dds):.1%} -> {min(max_dds_f):.1%}")
        improved_sharpe = sum(1 for s1, s2 in zip(sharpes, sharpes_f) if s2 > s1)
        lines.append(f"- **Filter improves Sharpe**: {improved_sharpe}/{len(sharpes)} years")

    # Consistency analysis
    lines.append("\n## Edge Consistency Analysis\n")
    positive_years = sum(1 for c in cagrs if c > 0)
    lines.append(f"- **Positive returns**: {positive_years}/{len(cagrs)} years ({positive_years/len(cagrs):.0%})")
    profitable_sharpe = sum(1 for s in sharpes if s > 0)
    lines.append(f"- **Positive Sharpe**: {profitable_sharpe}/{len(sharpes)} years ({profitable_sharpe/len(sharpes):.0%})")

    if len(cagrs) >= 3:
        # Check for trend in edge decay
        from scipy import stats
        x = np.arange(len(cagrs))
        slope, _, r, p, _ = stats.linregress(x, cagrs)
        lines.append(f"- **CAGR trend**: slope={slope:+.3f}/year, R²={r**2:.2f}, p={p:.3f}")
        if p < 0.05 and slope < 0:
            lines.append("  - ⚠️ Statistically significant edge decay detected")
        elif p < 0.05 and slope > 0:
            lines.append("  - Edge appears to be strengthening over time")
        else:
            lines.append("  - No statistically significant trend in edge")

    lines.append("\n## Conclusion\n")
    if positive_years >= len(cagrs) * 0.7:
        lines.append("The strategy shows **consistent positive returns** across multiple independent OOS periods.")
    elif positive_years >= len(cagrs) * 0.5:
        lines.append("The strategy shows **mixed results** — positive in most years but not consistently.")
    else:
        lines.append("The strategy shows **inconsistent edge** — loses money in most independent OOS periods.")

    return "\n".join(lines)


def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  WALK-FORWARD VALIDATION")
    log(f"  {len(YEARS)} annual models ({YEARS[0]}-{YEARS[-1]})")
    log(f"  LGBM + XGBoost ensemble, combined_live backtest")
    log("=" * 70)

    # Create output directory
    WF_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and prepare features ──
    log(f"\nLoading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: {INPUT_FILE} not found — run data_pipeline.py first.")

    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    assert "days_until_earnings" not in df.columns

    df = prepare_features(df)
    feature_cols = get_feature_cols(df)

    # Drop NaN on non-fundamental columns
    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund_cols + ["target_v5"])

    log(f"  {len(df):,} rows | {df['symbol'].nunique()} symbols | "
        f"{df['date'].min().date()} -> {df['date'].max().date()}")
    log(f"  Features: {len(feature_cols)} columns")

    # ── Train models and generate predictions per year ──
    all_preds = []
    auc_by_year = {}
    results_no_filter = []
    results_with_filter = []

    for year in YEARS:
        log(f"\n{'─'*70}")
        log(f"  YEAR {year}: train on <={year-1}, predict {year}")
        log(f"{'─'*70}")

        t_year = time.perf_counter()
        result = train_year_model(df, feature_cols, year)

        if result is None:
            log(f"  Skipped year {year}")
            continue

        preds_df, auc_dict = result
        auc_by_year[year] = auc_dict

        # Save per-year predictions
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        preds_df.to_parquet(pred_file, index=False, engine="pyarrow", compression="snappy")
        log(f"  Saved {pred_file.name} ({len(preds_df):,} rows)")

        all_preds.append(preds_df)

        # Run backtest without momentum filter
        log(f"  Running backtest (no filter) ...")
        metrics = run_backtest_for_year(preds_df.copy(), year, momentum_regime_filter=False)
        if metrics:
            results_no_filter.append(metrics)
            log(f"  -> CAGR={metrics['cagr']:+.1%}  Sharpe={metrics['sharpe']:.2f}  "
                f"DD={metrics['max_dd']:.1%}  Trades={metrics['n_trades']}")

        # Run backtest with momentum filter
        log(f"  Running backtest (momentum filter) ...")
        metrics_f = run_backtest_for_year(preds_df.copy(), year, momentum_regime_filter=True)
        if metrics_f:
            results_with_filter.append(metrics_f)
            log(f"  -> CAGR={metrics_f['cagr']:+.1%}  Sharpe={metrics_f['sharpe']:.2f}  "
                f"DD={metrics_f['max_dd']:.1%}  Trades={metrics_f['n_trades']}")

        elapsed_year = time.perf_counter() - t_year
        log(f"  Year {year} complete in {elapsed_year:.0f}s")

    # ── Save combined predictions ──
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        combined.to_parquet(WF_DIR / "predictions_walkforward_all.parquet",
                           index=False, engine="pyarrow", compression="snappy")
        log(f"\nCombined predictions: {len(combined):,} rows saved")

    # ── Save metrics JSON ──
    metrics_out = {
        "no_filter": results_no_filter,
        "with_filter": results_with_filter,
        "auc_by_year": {str(k): v for k, v in auc_by_year.items()},
    }
    metrics_file = WF_DIR / "metrics.json"
    metrics_file.write_text(json.dumps(metrics_out, indent=2, default=str))
    log(f"Metrics saved to {metrics_file}")

    # ── Generate report ──
    report = generate_report(results_no_filter, results_with_filter, auc_by_year)
    report_file = WF_DIR / "REPORT.md"
    report_file.write_text(report)
    log(f"Report saved to {report_file}")

    # ── Print summary table ──
    log(f"\n{'='*70}")
    log("  WALK-FORWARD RESULTS SUMMARY")
    log(f"{'='*70}")
    log(f"\n  {'Year':<6} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Win%':>7} {'Alpha':>8}")
    log(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*8}")
    for m in results_no_filter:
        alpha_str = f"{m['alpha']:+.1%}" if m.get("alpha") is not None else "N/A"
        log(f"  {m['year']:<6} {m['cagr']:>+7.1%} {m['sharpe']:>8.2f} "
            f"{m['max_dd']:>7.1%} {m['n_trades']:>8} {m['win_rate']:>6.1%} {alpha_str:>8}")

    if results_no_filter:
        cagrs = [m["cagr"] for m in results_no_filter]
        sharpes = [m["sharpe"] for m in results_no_filter]
        log(f"\n  Median CAGR: {np.median(cagrs):+.1%}  |  Median Sharpe: {np.median(sharpes):.2f}")
        log(f"  Positive CAGR years: {sum(1 for c in cagrs if c > 0)}/{len(cagrs)}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"\nArtifacts in {WF_DIR}/")


if __name__ == "__main__":
    main()
