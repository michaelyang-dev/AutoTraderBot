#!/usr/bin/env python3
"""
Walk-Forward Test: Enriched Features vs Baseline
=================================================
Runs walk-forward validation with both baseline (features.parquet) and
enriched (features_enriched.parquet) feature sets, comparing signal quality.

Uses the same methodology as walk_forward_validation.py but focuses on
signal quality metrics (top-5 alpha, AUC) rather than full backtests.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = DATA_DIR / "walkforward_enriched"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    eval_metric="auc", use_label_encoder=False,
)

EARLY_STOP = 100

RANK_FEATURES = [
    ("vol_20d", "vol_rank_20d"),
    ("ret_60d", "momentum_rank_60d"),
    ("rsi_14", "rsi_rank"),
    ("dist_sma50", "dist_sma50_rank"),
]

NEW_FEATURE_COLS = [
    "earnings_beat_streak",
    "revenue_surprise_last",
    "earnings_surprise_3q_avg",
    "revenue_surprise_3q_avg",
    "insider_buy_cluster",
    "insider_net_shares_180d",
]


def log(msg):
    print(msg, flush=True)


def get_feature_cols(df, exclude_new=False):
    exclude = {"date", "symbol", "target", "target_v5", "in_sp500", "pct_rank",
               "fwd_10d_ret", "fwd_ret"}
    forward_kw = {"fwd", "forward", "future"}
    cols = [c for c in df.columns
            if c not in exclude and not any(kw in c.lower() for kw in forward_kw)]
    if exclude_new:
        cols = [c for c in cols if c not in NEW_FEATURE_COLS]
    return cols


def prepare(df):
    for base_col, rank_col in RANK_FEATURES:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)
    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500 = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_v5"] = np.nan
    valid = df[sp500 & has_fwd].copy()
    valid["pct_rank"] = valid.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid["target_v5"] = (valid["pct_rank"] >= 0.80).astype(int)
    df.loc[valid.index, "target_v5"] = valid["target_v5"]
    df["fwd_ret"] = df["fwd_10d_ret"]
    return df


def train_and_predict(df, feature_cols, year):
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    calib_start = pd.Timestamp(f"{year - 2}-01-01")
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    train_all = df[df["date"] <= train_end]
    sp = train_all[train_all["in_sp500"] == True]

    calib_mask = sp["date"] >= calib_start
    pure_mask = sp["date"] < calib_start

    year_data = df[(df["date"] >= year_start) & (df["date"] <= year_end)].copy()

    if pure_mask.sum() < 1000 or calib_mask.sum() < 100 or len(year_data) < 50:
        return None

    X_tr = sp.loc[pure_mask, feature_cols].values
    y_tr = sp.loc[pure_mask, "target_v5"].values
    X_cal = sp.loc[calib_mask, feature_cols].values
    y_cal = sp.loc[calib_mask, "target_v5"].values

    scale = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)

    # LGBM
    m_lgb = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)],
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                         lgb.log_evaluation(period=-1)])
    c_lgb = CalibratedClassifierCV(m_lgb, method="isotonic", cv="prefit")
    c_lgb.fit(X_cal, y_cal)

    # XGB
    try:
        m_xgb = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale)
        m_xgb.fit(X_tr, y_tr, verbose=False)
        c_xgb = CalibratedClassifierCV(m_xgb, method="isotonic", cv="prefit")
        c_xgb.fit(X_cal, y_cal)
        xgb_ok = True
    except Exception:
        xgb_ok = False

    # Predict year
    X_y = year_data[feature_cols].values
    lgb_p = c_lgb.predict_proba(X_y)[:, 1]
    if xgb_ok:
        xgb_p = c_xgb.predict_proba(X_y)[:, 1]
        prob = 0.5 * lgb_p + 0.5 * xgb_p
    else:
        prob = lgb_p

    year_data["prob"] = prob

    # AUC on calibration set
    cal_p = c_lgb.predict_proba(X_cal)[:, 1]
    try:
        auc = roc_auc_score(y_cal, cal_p)
    except:
        auc = 0.5

    # Feature importance (LGBM)
    imp = dict(zip(feature_cols, m_lgb.feature_importances_))

    return year_data, auc, imp


def evaluate_signal(preds_df, year):
    """Compute top-5 forward returns and spread for one year."""
    sp500 = preds_df[preds_df["in_sp500"] == True].copy()
    dates = sorted(sp500["date"].unique())

    top5_rets = []
    bot5_rets = []
    rand_rets = []

    np.random.seed(42)
    for dt in dates:
        day = sp500[sp500["date"] == dt]
        if len(day) < 10 or day["fwd_ret"].isna().all():
            continue
        day = day.dropna(subset=["fwd_ret"]).sort_values("prob", ascending=False)
        if len(day) < 10:
            continue

        top5 = day.head(5)
        bot5 = day.tail(5)
        rand5 = day.sample(min(5, len(day)))

        top5_rets.extend(top5["fwd_ret"].values)
        bot5_rets.extend(bot5["fwd_ret"].values)
        rand_rets.extend(rand5["fwd_ret"].values)

    if not top5_rets:
        return None

    top5_mean = np.mean(top5_rets) * 100
    bot5_mean = np.mean(bot5_rets) * 100
    rand_mean = np.mean(rand_rets) * 100
    spread = top5_mean - bot5_mean
    t, p = stats.ttest_ind(top5_rets, bot5_rets)

    return {
        "year": year,
        "top5_fwd": top5_mean,
        "bot5_fwd": bot5_mean,
        "rand_fwd": rand_mean,
        "spread": spread,
        "spread_p": p,
        "n_days": len(dates),
    }


def main():
    log("=" * 70)
    log("Walk-Forward Test: Enriched Features vs Baseline")
    log("=" * 70)

    # Load enriched features
    enriched_file = DATA_DIR / "features_enriched.parquet"
    if not enriched_file.exists():
        log(f"ERROR: {enriched_file} not found. Run compute_enriched_features.py first.")
        sys.exit(1)

    log("Loading enriched features ...")
    df = pd.read_parquet(enriched_file)
    df["date"] = pd.to_datetime(df["date"])
    log(f"  Shape: {df.shape}")

    df = prepare(df)

    # Define feature sets
    all_cols = get_feature_cols(df, exclude_new=False)
    baseline_cols = get_feature_cols(df, exclude_new=True)

    log(f"\nBaseline features: {len(baseline_cols)}")
    log(f"Enriched features: {len(all_cols)} (+{len(all_cols) - len(baseline_cols)} new)")
    log(f"New features: {[c for c in all_cols if c not in baseline_cols]}")

    # Run both configurations
    configs = {
        "baseline": baseline_cols,
        "enriched": all_cols,
    }

    all_results = {}
    all_importances = {}

    for config_name, feat_cols in configs.items():
        log(f"\n{'='*60}")
        log(f"Config: {config_name} ({len(feat_cols)} features)")
        log(f"{'='*60}")

        results = []
        importances = {}

        for year in YEARS:
            log(f"\n  Year {year} ...")
            t0 = time.time()

            out = train_and_predict(df, feat_cols, year)
            if out is None:
                log(f"  SKIP {year}")
                continue

            preds_df, auc, imp = out
            signal = evaluate_signal(preds_df, year)

            if signal is None:
                log(f"  No signal data for {year}")
                continue

            signal["auc"] = auc
            results.append(signal)

            # Accumulate importance
            for k, v in imp.items():
                importances[k] = importances.get(k, 0) + v

            sig = "***" if signal["spread_p"] < 0.001 else "**" if signal["spread_p"] < 0.01 else "*" if signal["spread_p"] < 0.05 else ""
            log(f"  {year}: AUC={auc:.4f} | top5={signal['top5_fwd']:+.3f}% | spread={signal['spread']:+.3f}%{sig} | {time.time()-t0:.0f}s")

            # Save predictions
            preds_df.to_parquet(OUT_DIR / f"predictions_{config_name}_{year}.parquet", index=False)

        all_results[config_name] = results
        all_importances[config_name] = importances

    # ── Report ──
    write_report(all_results, all_importances, all_cols, baseline_cols)


def write_report(all_results, all_importances, enriched_cols, baseline_cols):
    """Generate ENRICHMENT_RESULTS.md."""
    log("\n" + "=" * 70)
    log("Writing Report")
    log("=" * 70)

    new_cols = [c for c in enriched_cols if c not in baseline_cols]

    lines = [
        "# Feature Enrichment Results",
        f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## New Features Added",
        "",
        "| Feature | Type | PIT Method | Intuition |",
        "|---------|------|-----------|-----------|",
        "| earnings_beat_streak | Earnings | count consecutive quarters where eps_actual > eps_estimated, using reportedDate | Companies on a streak tend to continue outperforming |",
        "| revenue_surprise_last | Earnings | (revenue_actual - revenue_estimated) / |estimate|, available from reportedDate | Revenue beats are strong buy signals, underused vs EPS |",
        "| earnings_surprise_3q_avg | Earnings | avg EPS surprise over last 3 reported quarters | More stable than single-quarter surprise |",
        "| revenue_surprise_3q_avg | Earnings | avg revenue surprise over last 3 reported quarters | More stable than single-quarter |",
        "| insider_buy_cluster | Insiders | binary: 3+ insider buys in trailing 90 calendar days | Clustered buying = high conviction signal |",
        "| insider_net_shares_180d | Insiders | net shares bought minus sold, 180-day window | Longer window captures slower insider trends |",
        "",
    ]

    # Feature importance
    lines.append("## Feature Importance (Top 15)")
    lines.append("")
    for config_name in ["baseline", "enriched"]:
        imp = all_importances.get(config_name, {})
        if not imp:
            continue
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
        total = sum(v for _, v in sorted_imp)
        lines.append(f"### {config_name.capitalize()} Model")
        lines.append("")
        lines.append("| Rank | Feature | Importance | % |")
        lines.append("|------|---------|-----------|---|")
        for i, (feat, val) in enumerate(sorted_imp, 1):
            pct = val / total * 100 if total > 0 else 0
            marker = " **NEW**" if feat in new_cols else ""
            lines.append(f"| {i} | {feat}{marker} | {val:.0f} | {pct:.1f}% |")
        lines.append("")

    # Walk-forward comparison
    lines.append("## Walk-Forward Comparison")
    lines.append("")
    lines.append("| Year | Baseline Top-5 | Enriched Top-5 | Baseline Spread | Enriched Spread | Baseline AUC | Enriched AUC |")
    lines.append("|------|---------------|---------------|----------------|----------------|-------------|-------------|")

    baseline_results = {r["year"]: r for r in all_results.get("baseline", [])}
    enriched_results = {r["year"]: r for r in all_results.get("enriched", [])}

    for year in YEARS:
        b = baseline_results.get(year, {})
        e = enriched_results.get(year, {})
        bt5 = f"{b.get('top5_fwd', 0):+.3f}%" if b else "—"
        et5 = f"{e.get('top5_fwd', 0):+.3f}%" if e else "—"
        bsp = f"{b.get('spread', 0):+.3f}%" if b else "—"
        esp = f"{e.get('spread', 0):+.3f}%" if e else "—"
        bauc = f"{b.get('auc', 0):.4f}" if b else "—"
        eauc = f"{e.get('auc', 0):.4f}" if e else "—"
        lines.append(f"| {year} | {bt5} | {et5} | {bsp} | {esp} | {bauc} | {eauc} |")

    lines.append("")

    # Summary stats
    for config_name in ["baseline", "enriched"]:
        results = all_results.get(config_name, [])
        if not results:
            continue
        top5s = [r["top5_fwd"] for r in results]
        spreads = [r["spread"] for r in results]
        aucs = [r["auc"] for r in results]
        lines.append(f"### {config_name.capitalize()} Summary")
        lines.append(f"- Median top-5 forward return: {np.median(top5s):+.3f}%")
        lines.append(f"- Mean top-5 forward return: {np.mean(top5s):+.3f}%")
        lines.append(f"- Median spread (top5 - bot5): {np.median(spreads):+.3f}%")
        lines.append(f"- Mean AUC: {np.mean(aucs):.4f}")
        lines.append(f"- Positive top-5 years: {sum(1 for t in top5s if t > 0)}/{len(top5s)}")
        lines.append(f"- Positive spread years: {sum(1 for s in spreads if s > 0)}/{len(spreads)}")
        lines.append("")

    # Verdict
    b_top5 = [r["top5_fwd"] for r in all_results.get("baseline", [])]
    e_top5 = [r["top5_fwd"] for r in all_results.get("enriched", [])]
    b_aucs = [r["auc"] for r in all_results.get("baseline", [])]
    e_aucs = [r["auc"] for r in all_results.get("enriched", [])]

    enriched_better_years = sum(1 for b, e in zip(b_top5, e_top5) if e > b)
    auc_improved = sum(1 for b, e in zip(b_aucs, e_aucs) if e > b)

    lines.append("## Recommendation")
    lines.append("")

    if enriched_better_years >= 7 and np.median(e_top5) > np.median(b_top5):
        verdict = "DEPLOY"
        lines.append(f"**{verdict}**: Enriched features improve top-5 returns in {enriched_better_years}/{len(b_top5)} years.")
    elif enriched_better_years >= 5:
        verdict = "MARGINAL"
        lines.append(f"**{verdict}**: Enriched features improve top-5 in {enriched_better_years}/{len(b_top5)} years — marginal improvement.")
    else:
        verdict = "KEEP BASELINE"
        lines.append(f"**{verdict}**: Enriched features only improve {enriched_better_years}/{len(b_top5)} years — not worth production complexity.")

    lines.append(f"- AUC improved in {auc_improved}/{len(b_aucs)} years")
    lines.append(f"- Median top-5 change: {np.median(e_top5) - np.median(b_top5):+.3f}pp")
    lines.append(f"- Mean AUC change: {np.mean(e_aucs) - np.mean(b_aucs):+.4f}")

    report = "\n".join(lines)
    report_file = OUT_DIR / "ENRICHMENT_RESULTS.md"
    with open(report_file, "w") as f:
        f.write(report)
    log(f"\nReport saved to {report_file}")
    log(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
