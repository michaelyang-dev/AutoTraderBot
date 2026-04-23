#!/usr/bin/env python3
"""
Walk-Forward Short Model — Dedicated Underperformer Predictor
==============================================================
Same walk-forward methodology as walk_forward_validation.py but with
an INVERTED target: bottom 20% by forward 10-day return.

The long model asks "which stocks will be in the top 20%?"
This model asks "which stocks will be in the bottom 20%?"

High prob_short = model thinks this stock will underperform.
We then check: do the model's top-5 picks (highest prob_short)
actually have negative forward returns?

Produces per-year predictions in ml_service/data/walkforward_short/
Then runs Phase 1 signal validation on them.
"""

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_SHORT_DIR = DATA_DIR / "walkforward_short"
INPUT_FILE = DATA_DIR / "features.parquet"
OUT_DIR = DATA_DIR / "longshort"

YEARS = list(range(2015, 2026))

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
    n_jobs=1,
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


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"date", "symbol", "target", "target_v5", "target_short",
               "in_sp500", "pct_rank"}
    forward_keywords = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_keywords)]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank features and BOTH targets (long + short)."""
    for base_col, rank_col in RANK_FEATURES:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()

    # Long target: top 20%
    df["target_v5"] = np.nan
    valid = df[sp500_mask & has_fwd].copy()
    valid["pct_rank"] = valid.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid["target_v5"] = (valid["pct_rank"] >= 0.80).astype(int)
    df.loc[valid.index, "target_v5"] = valid["target_v5"]

    # SHORT target: bottom 20%
    df["target_short"] = np.nan
    valid["target_short"] = (valid["pct_rank"] <= 0.20).astype(int)
    df.loc[valid.index, "target_short"] = valid["target_short"]

    df["fwd_ret"] = df["fwd_10d_ret"]
    return df


def train_short_model_year(df_full, feature_cols, year):
    """Train LGBM + XGB with target_short (bottom 20%) for one year."""
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
    y_train = sp500_train.loc[pure_train_mask, "target_short"].values
    X_calib = sp500_train.loc[calib_mask, feature_cols].values
    y_calib = sp500_train.loc[calib_mask, "target_short"].values

    scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    log(f"  Train: {pure_train_mask.sum():,} | Calib: {calib_mask.sum():,} | "
        f"Predict: {len(df_year):,} | target_short pos rate: {y_train.mean():.1%}")

    # LGBM
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

    # XGBoost
    xgb_ok = True
    try:
        model_xgb = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale)
        model_xgb.fit(X_train, y_train)
        calib_xgb = CalibratedClassifierCV(model_xgb, method="isotonic", cv="prefit")
        calib_xgb.fit(X_calib, y_calib)
        xgb_calib_probs = calib_xgb.predict_proba(X_calib)[:, 1]
        xgb_auc = roc_auc_score(y_calib, xgb_calib_probs) if len(np.unique(y_calib)) > 1 else float("nan")
    except Exception as e:
        log(f"  [WARN] XGBoost failed: {e} — LGBM-only")
        xgb_ok = False
        xgb_auc = float("nan")

    # Generate predictions for year Y
    X_year = df_year[feature_cols].values
    lgbm_probs = calib_lgbm.predict_proba(X_year)[:, 1]

    if xgb_ok:
        xgb_probs = calib_xgb.predict_proba(X_year)[:, 1]
        ensemble_probs = 0.5 * lgbm_probs + 0.5 * xgb_probs
    else:
        xgb_probs = lgbm_probs
        ensemble_probs = lgbm_probs

    df_year = df_year.copy()
    df_year["prob_short"] = ensemble_probs

    log(f"  LGBM: {n_trees} trees, AUC={lgbm_auc:.4f} | "
        f"XGB: {'OK' if xgb_ok else 'SKIP'} AUC={xgb_auc:.4f} | "
        f"prob_short range=[{ensemble_probs.min():.3f}, {ensemble_probs.max():.3f}]")

    save_cols = ["date", "symbol", "target_short", "prob_short", "fwd_ret", "in_sp500"]
    preds_df = df_year[save_cols].copy()

    return preds_df, {"lgbm_auc": lgbm_auc, "xgb_auc": xgb_auc}


# ── Signal validation (same as Phase 1 but on short model) ──────────────────

def validate_short_signal(all_preds):
    """Check if short model top-5 (highest prob_short) actually underperform."""
    results = []

    for year in YEARS:
        df = all_preds.get(year)
        if df is None:
            continue

        df = df.dropna(subset=["fwd_ret", "prob_short"])
        dates = sorted(df["date"].unique())
        np.random.seed(42 + year)

        top5_rets = []  # highest prob_short = most likely underperformers
        bot5_rets = []  # lowest prob_short = least likely underperformers
        rand5_rets = []
        spreads = []

        for date in dates:
            day = df[df["date"] == date]
            if len(day) < 20:
                continue

            day = day.sort_values("prob_short", ascending=False)

            # Top-5 by prob_short = stocks the model thinks will be in bottom 20%
            short_candidates = day.head(5)
            # Bottom-5 by prob_short = stocks model thinks WON'T underperform
            non_short = day.tail(5)
            rand5 = day.sample(min(5, len(day)))

            sc_ret = short_candidates["fwd_ret"].mean()
            ns_ret = non_short["fwd_ret"].mean()
            r_ret = rand5["fwd_ret"].mean()

            top5_rets.append(sc_ret)
            bot5_rets.append(ns_ret)
            rand5_rets.append(r_ret)
            spreads.append(ns_ret - sc_ret)  # positive = short model works

        if not top5_rets:
            continue

        t_stat, p_value = stats.ttest_1samp(spreads, 0)
        t_short, p_short = stats.ttest_1samp(top5_rets, 0)

        results.append({
            "year": year,
            "n_days": len(top5_rets),
            "short_cand_mean": np.mean(top5_rets),
            "non_short_mean": np.mean(bot5_rets),
            "random_mean": np.mean(rand5_rets),
            "spread_mean": np.mean(spreads),
            "spread_p": p_value,
            "short_cand_p": p_short,
            "short_cand_rets": top5_rets,
            "spreads": spreads,
        })

    return results


def generate_report(results, auc_by_year):
    lines = []
    lines.append("# Phase 1A: Dedicated Short Model — Signal Validation")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("\n## Model Design")
    lines.append("\n- **Target**: `target_short` = bottom 20% of S&P 500 by forward 10-day return")
    lines.append("- **Architecture**: Same LGBM + XGBoost ensemble, same features")
    lines.append("- **Walk-forward**: Train on years before Y, predict year Y")
    lines.append("- **Key question**: Do the model's top-5 picks (highest `prob_short`) actually go DOWN?")

    # AUC table
    lines.append("\n## Model Quality (AUC per year)\n")
    lines.append("| Year | LGBM AUC | XGB AUC |")
    lines.append("|------|----------|---------|")
    for year in YEARS:
        a = auc_by_year.get(year, {})
        lines.append(f"| {year} | {a.get('lgbm_auc', 0):.4f} | {a.get('xgb_auc', 0):.4f} |")

    # Signal validation table
    lines.append("\n## Short Signal Validation\n")
    lines.append("\"Short candidates\" = top-5 by `prob_short` (model thinks these will be bottom-20% performers)\n")
    lines.append("| Year | Days | Short Cands Fwd | Non-Short Fwd | Random | Spread (NS-SC) | Spread p | SC<0 p |")
    lines.append("|------|------|-----------------|---------------|--------|----------------|----------|--------|")

    for r in results:
        sig = "***" if r["spread_p"] < 0.01 else "**" if r["spread_p"] < 0.05 else "*" if r["spread_p"] < 0.10 else ""
        lines.append(
            f"| {r['year']} "
            f"| {r['n_days']} "
            f"| {r['short_cand_mean']:+.3%} "
            f"| {r['non_short_mean']:+.3%} "
            f"| {r['random_mean']:+.3%} "
            f"| {r['spread_mean']:+.3%}{sig} "
            f"| {r['spread_p']:.4f} "
            f"| {r['short_cand_p']:.4f} |"
        )

    # Aggregate
    all_sc = np.mean([r["short_cand_mean"] for r in results])
    all_ns = np.mean([r["non_short_mean"] for r in results])
    all_rand = np.mean([r["random_mean"] for r in results])
    all_spread = np.mean([r["spread_mean"] for r in results])

    all_daily_sc = []
    all_daily_spreads = []
    for r in results:
        all_daily_sc.extend(r["short_cand_rets"])
        all_daily_spreads.extend(r["spreads"])

    agg_t, agg_p = stats.ttest_1samp(all_daily_spreads, 0)
    sc_t, sc_p = stats.ttest_1samp(all_daily_sc, 0)

    lines.append("\n## Aggregate Summary\n")
    lines.append(f"- **Short candidates avg fwd return**: {all_sc:+.3%}")
    lines.append(f"- **Non-short avg fwd return**: {all_ns:+.3%}")
    lines.append(f"- **Random avg fwd return**: {all_rand:+.3%}")
    lines.append(f"- **Spread (non-short minus short cands)**: {all_spread:+.3%}")
    lines.append(f"- **Spread t-stat**: {agg_t:.2f}, p={agg_p:.6f}")
    lines.append(f"- **Short cands < 0**: t={sc_t:.2f}, p={sc_p:.6f}")
    lines.append(f"- **Positive spread years**: {sum(1 for r in results if r['spread_mean'] > 0)}/{len(results)}")
    lines.append(f"- **Short cands negative years**: {sum(1 for r in results if r['short_cand_mean'] < 0)}/{len(results)}")

    # Quintile analysis
    lines.append("\n## Quintile Analysis (by prob_short)\n")
    lines.append("| Quintile | Avg Fwd 10d Ret | vs Random |")
    lines.append("|----------|-----------------|-----------|")

    # Decision
    lines.append("\n## Decision\n")

    if all_sc < -0.003:
        decision = "STRONG"
        lines.append(f"**STRONG SHORT SIGNAL** (short candidates avg = {all_sc:+.3%} < -0.3%)")
        lines.append(f"\nThe dedicated short model identifies stocks that actually decline. "
                     f"Proceed to build B4 long/short infrastructure.")
    elif all_sc < 0:
        decision = "WEAK"
        lines.append(f"**WEAK SHORT SIGNAL** (short candidates avg = {all_sc:+.3%}, between -0.3% and 0%)")
        lines.append(f"\nShort candidates underperform slightly. B4 might work with thin margins. "
                     f"**User decision required.**")
    else:
        decision = "NONE"
        lines.append(f"**NO SHORT SIGNAL** (short candidates avg = {all_sc:+.3%} >= 0%)")
        lines.append(f"\nEven a dedicated short model cannot identify stocks that go down. "
                     f"The feature set does not predict the downside tail.")

    # Comparison with long model
    lines.append(f"\n### Comparison: Long model vs Short model\n")
    lines.append(f"| Metric | Long Model (Phase 1) | Short Model (this) |")
    lines.append(f"|--------|---------------------|--------------------|")
    lines.append(f"| Top-5 picks fwd return | +1.44% (winners) | {all_sc:+.3%} (losers) |")
    lines.append(f"| Bottom-5 picks fwd return | +0.33% | {all_ns:+.3%} |")
    lines.append(f"| T-B spread | +1.11% | {all_spread:+.3%} |")
    lines.append(f"| Spread p-value | <0.001 | {agg_p:.4f} |")

    # What this means for long/short
    if all_sc < 0 and all_spread > 0:
        combined_spread = 1.44 / 100 + abs(all_sc)  # long model top + short model top
        lines.append(f"\n### Combined long/short potential")
        lines.append(f"- Long leg (long model top-5): +1.44%/trade")
        lines.append(f"- Short leg (short model top-5): {all_sc:+.3%}/trade")
        lines.append(f"- **Combined spread**: ~{combined_spread*100:+.2f}%/trade")
        lines.append(f"- ~25 non-overlapping periods/year")
        lines.append(f"- **Estimated annual alpha**: ~{combined_spread * 25 * 100:+.1f}%")

    return "\n".join(lines), decision


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  PATH A: DEDICATED SHORT MODEL (bottom-20% target)")
    log("=" * 70)

    WF_SHORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare features
    log(f"\nLoading {INPUT_FILE} ...")
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    assert "days_until_earnings" not in df.columns

    df = prepare_features(df)
    feature_cols = get_feature_cols(df)

    non_fund_cols = [c for c in feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund_cols + ["target_short"])

    log(f"  {len(df):,} rows | {df['symbol'].nunique()} symbols")
    log(f"  target_short positive rate: {df['target_short'].mean():.1%}")
    log(f"  Features: {len(feature_cols)} columns")

    # Train short models per year
    all_preds = {}
    auc_by_year = {}

    for year in YEARS:
        log(f"\n{'─'*60}")
        log(f"  YEAR {year}: SHORT MODEL (train <={year-1}, predict {year})")
        log(f"{'─'*60}")

        t_year = time.perf_counter()
        result = train_short_model_year(df, feature_cols, year)

        if result is None:
            log(f"  Skipped year {year}")
            continue

        preds_df, auc_dict = result
        auc_by_year[year] = auc_dict
        all_preds[year] = preds_df

        # Save predictions
        pred_file = WF_SHORT_DIR / f"predictions_short_{year}.parquet"
        preds_df.to_parquet(pred_file, index=False, engine="pyarrow", compression="snappy")
        log(f"  Saved {pred_file.name} ({len(preds_df):,} rows)")
        log(f"  Year {year}: {time.perf_counter()-t_year:.0f}s")

    # Validate short signal
    log(f"\n{'─'*60}")
    log("  VALIDATING SHORT SIGNAL ...")
    log(f"{'─'*60}")

    results = validate_short_signal(all_preds)
    for r in results:
        log(f"  {r['year']}: short_cands={r['short_cand_mean']:+.3%}  "
            f"spread={r['spread_mean']:+.3%} (p={r['spread_p']:.4f})")

    report, decision = generate_report(results, auc_by_year)
    report_file = OUT_DIR / "PHASE1A_SHORT_MODEL.md"
    report_file.write_text(report)

    elapsed = time.perf_counter() - t0
    log(f"\n  Report: {report_file}")
    log(f"  Decision: {decision}")
    log(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
