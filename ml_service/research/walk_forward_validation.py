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
from sklearn.impute import SimpleImputer

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

YEARS = list(range(2022, 2026))  # 2022..2025 (ensures >4yr training depth per fold)
MIN_TRAIN_SAMPLES = 300_000     # halt if any fold has fewer training samples

# Same model params as train_validation_model_fast.py
N_DECILES = 10  # graded relevance labels 0..(N_DECILES-1)

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
    objective="lambdarank",
    metric="ndcg",
    eval_at=[5],
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
    objective="rank:ndcg",
    eval_metric="ndcg@5",
)

EARLY_STOP_ROUNDS = 100
PURGE_TRADING_DAYS = 10  # must match label horizon (10-day forward return)

# Dual-ensemble blend weights (must match train_production_model.py)
BLEND_WEIGHT_BASE   = 0.4
BLEND_WEIGHT_SECTOR = 0.6

SECTOR_FEATURE_COLS = [
    "ret_10d_vs_sector", "ret_20d_vs_sector",
    "rsi_14_vs_sector", "vol_20d_vs_sector",
]

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


def get_feature_cols(df: pd.DataFrame, exclude_cols: set = None) -> list:
    exclude = {"date", "symbol", "target", "target_v5", "target_rank",
               "in_sp500", "pct_rank"}
    if exclude_cols:
        exclude.update(exclude_cols)
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank features and target, drop NaN rows."""
    # Cross-sectional rank features
    for base_col, rank_col in RANK_FEATURES:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    # Cross-sectional targets
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_v5"] = np.nan
    df["target_rank"] = np.nan
    valid = df[sp500_mask & has_fwd].copy()
    pct = valid.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid["pct_rank"] = pct
    valid["target_v5"] = (pct >= 0.80).astype(int)
    valid["target_rank"] = np.clip((pct * N_DECILES).astype(int), 0, N_DECILES - 1)
    df.loc[valid.index, "target_v5"] = valid["target_v5"]
    df.loc[valid.index, "target_rank"] = valid["target_rank"]

    # Forward return for predictions
    df["fwd_ret"] = df["fwd_10d_ret"]

    return df


class RankerWrapper:
    """Wraps LGBMRanker/XGBRanker to provide predict_proba() for signal_server compatibility."""

    def __init__(self, ranker):
        self.ranker = ranker

    def predict_proba(self, X):
        scores = self.ranker.predict(X)
        probs = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return self.ranker.predict(X)


def train_year_model(df_full, base_feature_cols, all_feature_cols, year):
    """
    Train dual LGBMRanker+XGBRanker ensemble (40/60 base/sector blend) for a
    single walk-forward year.  Uses LambdaRank (NDCG@5) on decile labels.

    Training data: all dates < year
    Calibration: last year of training data (year-1), with purge gaps
    Prediction: year Y only

    Returns (predictions_df, metric_dict) or None if insufficient data.
    """
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    calib_start = pd.Timestamp(f"{year - 2}-01-01")
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    # Split
    train_all = df_full[df_full["date"] <= train_end].copy()
    sp500_train = train_all[train_all["in_sp500"] == True].copy()

    # Sort by date for proper ranking groups
    sp500_train = sp500_train.sort_values("date").reset_index(drop=True)

    # Purge gaps
    train_dates_arr = pd.DatetimeIndex(sp500_train["date"].unique()).sort_values()
    calib_boundary_idx = train_dates_arr.searchsorted(calib_start)
    purge_train_end = pd.Timestamp(train_dates_arr[max(0, calib_boundary_idx - PURGE_TRADING_DAYS)])
    purge_calib_start = pd.Timestamp(train_dates_arr[min(len(train_dates_arr) - 1, calib_boundary_idx + PURGE_TRADING_DAYS)])
    calib_end_idx = len(train_dates_arr) - 1
    purge_calib_end = pd.Timestamp(train_dates_arr[max(0, calib_end_idx - PURGE_TRADING_DAYS)])

    pure_train_mask = sp500_train["date"] < purge_train_end
    calib_mask = (sp500_train["date"] >= purge_calib_start) & (sp500_train["date"] <= purge_calib_end)

    # Year to predict
    year_mask = (df_full["date"] >= year_start) & (df_full["date"] <= year_end)
    df_year = df_full[year_mask].copy()

    n_train = pure_train_mask.sum()
    if n_train < 1000 or calib_mask.sum() < 100 or len(df_year) < 50:
        log(f"  [SKIP] Insufficient data for year {year}")
        return None
    if n_train < MIN_TRAIN_SAMPLES:
        log(f"  [HALT] Training fold has {n_train:,} samples (minimum: {MIN_TRAIN_SAMPLES:,})")
        log(f"         Increase training window or adjust YEARS range.")
        raise RuntimeError(f"Training fold too small: {n_train:,} < {MIN_TRAIN_SAMPLES:,}")

    y_train = sp500_train.loc[pure_train_mask, "target_rank"].values.astype(int)
    y_calib = sp500_train.loc[calib_mask, "target_rank"].values.astype(int)

    # Compute group/qid arrays for rankers (data is sorted by date)
    train_dates_s = sp500_train.loc[pure_train_mask, "date"]
    calib_dates_s = sp500_train.loc[calib_mask, "date"]
    train_groups = train_dates_s.groupby(train_dates_s).size().values
    calib_groups = calib_dates_s.groupby(calib_dates_s).size().values
    train_unique = train_dates_s.unique()
    calib_unique = calib_dates_s.unique()
    train_qids = train_dates_s.map({d: i for i, d in enumerate(train_unique)}).values
    calib_qids = calib_dates_s.map({d: i for i, d in enumerate(calib_unique)}).values

    log(f"  Train: {pure_train_mask.sum():,} rows [{len(train_groups)} groups] | "
        f"Calib: {calib_mask.sum():,} rows [{len(calib_groups)} groups] | "
        f"Predict: {len(df_year):,} rows")

    def _train_ensemble(feat_cols, label):
        """Train LGBMRanker + XGBRanker on given feature set."""
        X_tr = sp500_train.loc[pure_train_mask, feat_cols].values
        X_cal = sp500_train.loc[calib_mask, feat_cols].values

        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_cal_imp = imp.transform(X_cal)

        # LGBMRanker
        m_lgb = lgb.LGBMRanker(**LGB_PARAMS)
        m_lgb.fit(
            X_tr_imp, y_train, group=train_groups,
            eval_set=[(X_cal_imp, y_calib)], eval_group=[calib_groups],
            callbacks=[
                lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        n_trees = m_lgb.booster_.num_trees()
        w_lgbm = RankerWrapper(m_lgb)

        # XGBRanker
        m_xgb = xgb.XGBRanker(**XGB_PARAMS)
        m_xgb.fit(
            X_tr_imp, y_train, qid=train_qids,
            eval_set=[(X_cal_imp, y_calib)], eval_qid=[calib_qids],
            verbose=False,
        )
        w_xgb = RankerWrapper(m_xgb)

        # Report score ranges on calib
        lgb_scores = w_lgbm.predict_proba(X_cal_imp)[:, 1]
        xgb_scores = w_xgb.predict_proba(X_cal_imp)[:, 1]
        ens_scores = 0.5 * lgb_scores + 0.5 * xgb_scores

        log(f"    {label}: LGBM {n_trees} trees | "
            f"Ens score range=[{ens_scores.min():.3f}, {ens_scores.max():.3f}]")

        return w_lgbm, w_xgb, imp

    # ── Train BASE ensemble (no sector features) ──
    base_lgbm, base_xgb, imp_base = _train_ensemble(base_feature_cols, "Base")

    # ── Train SECTOR ensemble (all features) ──
    sect_lgbm, sect_xgb, imp_sect = _train_ensemble(all_feature_cols, "Sector")

    # ── Generate 40/60 blended predictions for year Y ──
    X_year_base = imp_base.transform(df_year[base_feature_cols].values)
    X_year_sect = imp_sect.transform(df_year[all_feature_cols].values)

    base_probs = 0.5 * base_lgbm.predict_proba(X_year_base)[:, 1] + \
                 0.5 * base_xgb.predict_proba(X_year_base)[:, 1]
    sect_probs = 0.5 * sect_lgbm.predict_proba(X_year_sect)[:, 1] + \
                 0.5 * sect_xgb.predict_proba(X_year_sect)[:, 1]
    ensemble_probs = BLEND_WEIGHT_BASE * base_probs + BLEND_WEIGHT_SECTOR * sect_probs

    df_year["prob_base"] = base_probs
    df_year["prob_sector"] = sect_probs
    df_year["prob_ensemble"] = ensemble_probs

    log(f"    Blend ({BLEND_WEIGHT_BASE:.0%}/{BLEND_WEIGHT_SECTOR:.0%}): "
        f"range=[{ensemble_probs.min():.3f}, {ensemble_probs.max():.3f}]")

    save_cols = ["date", "symbol", "target_v5", "prob_base", "prob_sector",
                 "prob_ensemble", "fwd_ret", "in_sp500"]
    preds_df = df_year[save_cols].copy()

    metric_dict = {"base": "ok", "sector": "ok"}
    return preds_df, metric_dict


def run_backtest_for_year(preds_df, year, momentum_regime_filter=False,
                          preloaded_prices=None):
    """Run combined_live backtest for a single year using provided predictions."""
    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())

    if len(all_dates) < 10:
        return None

    years_span = (all_dates[-1] - all_dates[0]).days / 365.25
    if years_span <= 0:
        years_span = len(all_dates) / 252.0

    # Use preloaded prices if available, otherwise fetch
    if preloaded_prices is not None:
        # Slice to relevant date range (with lookback for momentum strategies)
        lookback_start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=250)
        close = preloaded_prices[
            (preloaded_prices.index >= lookback_start) &
            (preloaded_prices.index <= pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5))
        ].copy()
    else:
        start_str = (pd.Timestamp(all_dates[0]) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
        end_str = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        close = load_bars_cached(universe_syms, start_str, end_str)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close = close.reindex(sim_index, method="ffill")

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        if preloaded_prices is None:
            start_str = (pd.Timestamp(all_dates[0]) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
            end_str = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
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


def generate_report(results_no_filter, results_with_filter, metrics_by_year):
    """Generate REPORT.md with consistency analysis."""
    lines = []
    lines.append("# Walk-Forward Validation Report")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\nYears tested: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} independent periods)")
    lines.append("\n## Per-Year Results (combined_live, no momentum filter)\n")

    # Table header
    lines.append("| Year | CAGR | Sharpe | Sortino | Max DD | Trades | Win% | Alpha |")
    lines.append("|------|------|--------|---------|--------|--------|------|-------|")

    for m in results_no_filter:
        lines.append(
            f"| {m['year']} "
            f"| {m['cagr']:+.1%} "
            f"| {m['sharpe']:.2f} "
            f"| {m['sortino']:.2f} "
            f"| {m['max_dd']:.1%} "
            f"| {m['n_trades']} "
            f"| {m['win_rate']:.1%} "
            f"| {'+' if m.get('alpha') and m['alpha'] > 0 else ''}{m.get('alpha', 0) or 0:.1%} |"
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
    log(f"  LambdaRank dual ensemble, 40/60 base/sector blend")
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
    all_feature_cols = get_feature_cols(df)
    base_feature_cols = get_feature_cols(df, exclude_cols=set(SECTOR_FEATURE_COLS))

    # Drop NaN on non-fundamental columns
    non_fund_cols = [c for c in all_feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund_cols + ["target_rank"])

    log(f"  {len(df):,} rows | {df['symbol'].nunique()} symbols | "
        f"{df['date'].min().date()} -> {df['date'].max().date()}")
    log(f"  Base features: {len(base_feature_cols)} | Sector features: {len(all_feature_cols)}")

    # ── Pre-download price data once (avoids 22 yfinance calls) ──
    all_symbols = sorted(df["symbol"].unique().tolist())
    price_start = (df["date"].min() - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    price_end = (df["date"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    log(f"\n  Pre-downloading prices for {len(all_symbols)} symbols ...")
    t_dl = time.perf_counter()
    preloaded_prices = load_bars_cached(all_symbols, price_start, price_end)
    log(f"  Prices loaded: {preloaded_prices.shape} in {time.perf_counter() - t_dl:.0f}s")

    # ── Train models and generate predictions per year ──
    all_preds = []
    metrics_by_year = {}
    results_no_filter = []
    results_with_filter = []

    for year in YEARS:
        log(f"\n{'─'*70}")
        log(f"  YEAR {year}: train on <={year-1}, predict {year}")
        log(f"{'─'*70}")

        t_year = time.perf_counter()
        result = train_year_model(df, base_feature_cols, all_feature_cols, year)

        if result is None:
            log(f"  Skipped year {year}")
            continue

        preds_df, year_metrics = result
        metrics_by_year[year] = year_metrics

        # Save per-year predictions
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        preds_df.to_parquet(pred_file, index=False, engine="pyarrow", compression="snappy")
        log(f"  Saved {pred_file.name} ({len(preds_df):,} rows)")

        all_preds.append(preds_df)

        # Run backtest without momentum filter
        log(f"  Running backtest (no filter) ...")
        metrics = run_backtest_for_year(preds_df.copy(), year,
                                        momentum_regime_filter=False,
                                        preloaded_prices=preloaded_prices)
        if metrics:
            results_no_filter.append(metrics)
            log(f"  -> CAGR={metrics['cagr']:+.1%}  Sharpe={metrics['sharpe']:.2f}  "
                f"DD={metrics['max_dd']:.1%}  Trades={metrics['n_trades']}")

        # Run backtest with momentum filter
        log(f"  Running backtest (momentum filter) ...")
        metrics_f = run_backtest_for_year(preds_df.copy(), year,
                                          momentum_regime_filter=True,
                                          preloaded_prices=preloaded_prices)
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
        "metrics_by_year": {str(k): v for k, v in metrics_by_year.items()},
    }
    metrics_file = WF_DIR / "metrics.json"
    metrics_file.write_text(json.dumps(metrics_out, indent=2, default=str))
    log(f"Metrics saved to {metrics_file}")

    # ── Generate report ──
    report = generate_report(results_no_filter, results_with_filter, metrics_by_year)
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
