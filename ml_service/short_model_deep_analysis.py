#!/usr/bin/env python3
"""
Short Model v2 — Deep Analysis & Improvement Research
======================================================
Tests multiple approaches to improve short signal WITHOUT retraining:
  1. Top-N concentration (top-3, top-2, top-1 vs top-5)
  2. Regime filtering (only short when market is weak)
  3. Confidence thresholds (only short when prob_short > X)
  4. Sector analysis (which sectors produce real short signal?)
  5. Year-level failure forensics (why does it fail in 2018/2020/2025?)
  6. Multi-signal confirmation (require multiple deterioration flags)

Output: data/longshort/SHORT_DEEP_ANALYSIS.md
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_SHORT_DIR = DATA_DIR / "walkforward_short"
FEAT_FILE = DATA_DIR / "features_short_enriched.parquet"
OUT_DIR = DATA_DIR / "longshort"
YEARS = list(range(2015, 2026))


def log(msg: str):
    print(msg, flush=True)


def load_predictions():
    """Load all walk-forward short model predictions."""
    all_preds = {}
    for year in YEARS:
        path = WF_SHORT_DIR / f"predictions_short_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["fwd_ret", "prob_short"])
            all_preds[year] = df
    return all_preds


def load_features():
    """Load enriched features for deeper analysis."""
    df = pd.read_parquet(FEAT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ═══════════════════════════════════════════════════════════════════
# 1. TOP-N CONCENTRATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_top_n(all_preds):
    """Test top-1, top-2, top-3, top-5, top-10 short candidates."""
    log("\n  ─── TOP-N CONCENTRATION ───")
    results = {}

    for n in [1, 2, 3, 5, 10]:
        yearly_rets = []
        all_rets = []

        for year in YEARS:
            df = all_preds.get(year)
            if df is None:
                continue

            np.random.seed(42 + year)
            day_rets = []

            for date in sorted(df["date"].unique()):
                day = df[df["date"] == date].sort_values("prob_short", ascending=False)
                if len(day) < 20:
                    continue

                top_n = day.head(n)
                ret = top_n["fwd_ret"].mean()
                day_rets.append(ret)

            if day_rets:
                yearly_rets.append({
                    "year": year,
                    "mean": np.mean(day_rets),
                    "median": np.median(day_rets),
                    "pct_negative": np.mean([r < 0 for r in day_rets]),
                    "n_days": len(day_rets),
                })
                all_rets.extend(day_rets)

        t_stat, p_val = stats.ttest_1samp(all_rets, 0) if all_rets else (0, 1)
        neg_years = sum(1 for r in yearly_rets if r["mean"] < 0)
        avg_ret = np.mean([r["mean"] for r in yearly_rets])
        median_ret = np.median([r["mean"] for r in yearly_rets])

        results[n] = {
            "avg_ret": avg_ret,
            "median_ret": median_ret,
            "t_stat": t_stat,
            "p_val": p_val,
            "neg_years": neg_years,
            "total_years": len(yearly_rets),
            "yearly": yearly_rets,
            "pct_neg_days": np.mean([r < 0 for r in all_rets]),
        }
        log(f"    Top-{n:2d}: avg={avg_ret:+.3%}, median={median_ret:+.3%}, "
            f"neg_years={neg_years}/{len(yearly_rets)}, "
            f"neg_days={results[n]['pct_neg_days']:.1%}, "
            f"t={t_stat:.2f}, p={p_val:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 2. REGIME FILTERING
# ═══════════════════════════════════════════════════════════════════

def analyze_regime_filter(all_preds, features):
    """Only short when market conditions favor shorts."""
    log("\n  ─── REGIME FILTERING ───")

    # Get market features (SPY returns, VIX proxy, etc.)
    spy_data = features[features["symbol"] == "SPY"][["date", "ret_20d", "dist_sma200",
                                                       "ret_60d"]].copy()
    spy_data = spy_data.rename(columns={
        "ret_20d": "spy_ret_20d_raw",
        "dist_sma200": "spy_dist_sma200",
        "ret_60d": "spy_ret_60d_raw",
    })

    # Also get VIX proxy
    vixy_data = features[features["symbol"] == "VIXY"][["date", "ret_5d"]].copy()
    vixy_data = vixy_data.rename(columns={"ret_5d": "vixy_ret_5d_raw"})

    results = {}
    regimes = {
        "all": lambda day_feats: True,
        "spy_below_sma200": lambda day_feats: day_feats.get("spy_dist_sma200", 0) < 0,
        "spy_ret_20d_neg": lambda day_feats: day_feats.get("spy_ret_20d_raw", 0) < 0,
        "spy_ret_60d_neg": lambda day_feats: day_feats.get("spy_ret_60d_raw", 0) < -0.03,
        "spy_weak_or_flat": lambda day_feats: day_feats.get("spy_ret_20d_raw", 0) < 0.01,
        "not_strong_bull": lambda day_feats: day_feats.get("spy_ret_60d_raw", 0) < 0.05,
    }

    for regime_name, regime_fn in regimes.items():
        yearly_rets = []
        all_rets = []
        total_days = 0
        filtered_days = 0

        for year in YEARS:
            df = all_preds.get(year)
            if df is None:
                continue

            day_rets = []
            for date in sorted(df["date"].unique()):
                total_days += 1

                # Get market state for this date
                spy_row = spy_data[spy_data["date"] == date]
                day_feats = {}
                if len(spy_row) > 0:
                    for col in ["spy_ret_20d_raw", "spy_dist_sma200", "spy_ret_60d_raw"]:
                        val = spy_row[col].iloc[0]
                        day_feats[col] = val if not pd.isna(val) else 0

                if not regime_fn(day_feats):
                    filtered_days += 1
                    continue

                day = df[df["date"] == date].sort_values("prob_short", ascending=False)
                if len(day) < 20:
                    continue

                top5 = day.head(5)
                ret = top5["fwd_ret"].mean()
                day_rets.append(ret)

            if day_rets:
                yearly_rets.append({
                    "year": year,
                    "mean": np.mean(day_rets),
                    "n_days": len(day_rets),
                })
                all_rets.extend(day_rets)

        if not all_rets:
            continue

        t_stat, p_val = stats.ttest_1samp(all_rets, 0)
        neg_years = sum(1 for r in yearly_rets if r["mean"] < 0)
        avg_ret = np.mean([r["mean"] for r in yearly_rets])
        active_pct = len(all_rets) / max(total_days, 1)

        results[regime_name] = {
            "avg_ret": avg_ret,
            "t_stat": t_stat,
            "p_val": p_val,
            "neg_years": neg_years,
            "total_years": len(yearly_rets),
            "active_pct": active_pct,
            "n_days": len(all_rets),
            "yearly": yearly_rets,
        }
        log(f"    {regime_name:25s}: avg={avg_ret:+.3%}, "
            f"neg_years={neg_years}/{len(yearly_rets)}, "
            f"active={active_pct:.0%} ({len(all_rets)} days), "
            f"t={t_stat:.2f}, p={p_val:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 3. CONFIDENCE THRESHOLD
# ═══════════════════════════════════════════════════════════════════

def analyze_confidence_threshold(all_preds):
    """Only short when model is very confident (high prob_short)."""
    log("\n  ─── CONFIDENCE THRESHOLDS ───")

    # First, understand prob_short distribution
    all_probs = []
    for year in YEARS:
        df = all_preds.get(year)
        if df is not None:
            all_probs.extend(df["prob_short"].values)

    all_probs = np.array(all_probs)
    log(f"    prob_short distribution: min={all_probs.min():.3f}, "
        f"median={np.median(all_probs):.3f}, "
        f"p75={np.percentile(all_probs, 75):.3f}, "
        f"p90={np.percentile(all_probs, 90):.3f}, "
        f"p95={np.percentile(all_probs, 95):.3f}, "
        f"max={all_probs.max():.3f}")

    results = {}
    thresholds = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for thresh in thresholds:
        yearly_rets = []
        all_rets = []
        all_counts = []

        for year in YEARS:
            df = all_preds.get(year)
            if df is None:
                continue

            day_rets = []
            for date in sorted(df["date"].unique()):
                day = df[df["date"] == date]
                if len(day) < 20:
                    continue

                # Filter by confidence threshold
                confident = day[day["prob_short"] > thresh].sort_values("prob_short", ascending=False)
                if len(confident) == 0:
                    continue

                top5 = confident.head(5)
                ret = top5["fwd_ret"].mean()
                day_rets.append(ret)
                all_counts.append(len(confident))

            if day_rets:
                yearly_rets.append({
                    "year": year,
                    "mean": np.mean(day_rets),
                })
                all_rets.extend(day_rets)

        if not all_rets:
            continue

        t_stat, p_val = stats.ttest_1samp(all_rets, 0)
        neg_years = sum(1 for r in yearly_rets if r["mean"] < 0)
        avg_ret = np.mean([r["mean"] for r in yearly_rets])
        avg_pool = np.mean(all_counts)

        results[thresh] = {
            "avg_ret": avg_ret,
            "t_stat": t_stat,
            "p_val": p_val,
            "neg_years": neg_years,
            "total_years": len(yearly_rets),
            "n_days": len(all_rets),
            "avg_pool_size": avg_pool,
            "yearly": yearly_rets,
        }
        log(f"    thresh>{thresh:.2f}: avg={avg_ret:+.3%}, "
            f"neg_years={neg_years}/{len(yearly_rets)}, "
            f"days={len(all_rets)}, "
            f"avg_pool={avg_pool:.0f}, "
            f"t={t_stat:.2f}, p={p_val:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 4. COMBINED: TOP-N + REGIME + CONFIDENCE
# ═══════════════════════════════════════════════════════════════════

def analyze_combined_filters(all_preds, features):
    """Test combinations of top-N, regime, and confidence filters."""
    log("\n  ─── COMBINED FILTERS ───")

    spy_data = features[features["symbol"] == "SPY"][["date", "ret_20d", "dist_sma200", "ret_60d"]].copy()
    spy_data = spy_data.rename(columns={
        "ret_20d": "spy_ret_20d_raw",
        "dist_sma200": "spy_dist_sma200",
        "ret_60d": "spy_ret_60d_raw",
    })

    combos = [
        {"name": "top3_nofilter", "top_n": 3, "regime": None, "thresh": 0.0},
        {"name": "top3_not_strong_bull", "top_n": 3, "regime": "not_strong_bull", "thresh": 0.0},
        {"name": "top3_spy_weak", "top_n": 3, "regime": "spy_weak", "thresh": 0.0},
        {"name": "top3_high_conf", "top_n": 3, "regime": None, "thresh": 0.30},
        {"name": "top3_spy_weak_high_conf", "top_n": 3, "regime": "spy_weak", "thresh": 0.30},
        {"name": "top5_not_strong_bull", "top_n": 5, "regime": "not_strong_bull", "thresh": 0.0},
        {"name": "top5_spy_weak_high_conf", "top_n": 5, "regime": "spy_weak", "thresh": 0.30},
        {"name": "top2_not_strong_bull", "top_n": 2, "regime": "not_strong_bull", "thresh": 0.0},
        {"name": "top1_not_strong_bull", "top_n": 1, "regime": "not_strong_bull", "thresh": 0.0},
        {"name": "top3_not_strong_bull_conf25", "top_n": 3, "regime": "not_strong_bull", "thresh": 0.25},
    ]

    regime_fns = {
        "not_strong_bull": lambda f: f.get("spy_ret_60d_raw", 0) < 0.05,
        "spy_weak": lambda f: f.get("spy_ret_20d_raw", 0) < 0,
    }

    results = {}
    for combo in combos:
        yearly_rets = []
        all_rets = []
        total_days = 0
        active_days = 0

        for year in YEARS:
            df = all_preds.get(year)
            if df is None:
                continue

            day_rets = []
            for date in sorted(df["date"].unique()):
                total_days += 1

                # Regime check
                if combo["regime"]:
                    spy_row = spy_data[spy_data["date"] == date]
                    day_feats = {}
                    if len(spy_row) > 0:
                        for col in ["spy_ret_20d_raw", "spy_dist_sma200", "spy_ret_60d_raw"]:
                            val = spy_row[col].iloc[0]
                            day_feats[col] = val if not pd.isna(val) else 0
                    if not regime_fns[combo["regime"]](day_feats):
                        continue

                day = df[df["date"] == date]
                if len(day) < 20:
                    continue

                # Confidence filter
                if combo["thresh"] > 0:
                    day = day[day["prob_short"] > combo["thresh"]]
                    if len(day) == 0:
                        continue

                day = day.sort_values("prob_short", ascending=False)
                top_n = day.head(combo["top_n"])
                ret = top_n["fwd_ret"].mean()
                day_rets.append(ret)
                active_days += 1

            if day_rets:
                yearly_rets.append({
                    "year": year,
                    "mean": np.mean(day_rets),
                    "n_days": len(day_rets),
                })
                all_rets.extend(day_rets)

        if not all_rets:
            results[combo["name"]] = {"avg_ret": np.nan, "skip": True}
            continue

        t_stat, p_val = stats.ttest_1samp(all_rets, 0)
        neg_years = sum(1 for r in yearly_rets if r["mean"] < 0)
        avg_ret = np.mean([r["mean"] for r in yearly_rets])
        active_pct = active_days / max(total_days, 1)

        results[combo["name"]] = {
            "avg_ret": avg_ret,
            "t_stat": t_stat,
            "p_val": p_val,
            "neg_years": neg_years,
            "total_years": len(yearly_rets),
            "active_pct": active_pct,
            "n_days": len(all_rets),
            "yearly": yearly_rets,
        }
        log(f"    {combo['name']:40s}: avg={avg_ret:+.3%}, "
            f"neg_years={neg_years}/{len(yearly_rets)}, "
            f"active={active_pct:.0%}, "
            f"t={t_stat:.2f}, p={p_val:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 5. FAILURE FORENSICS — WHY DOES IT FAIL IN SPECIFIC YEARS?
# ═══════════════════════════════════════════════════════════════════

def analyze_failure_years(all_preds, features):
    """Deep dive into years where short candidates go UP."""
    log("\n  ─── FAILURE FORENSICS ───")

    spy_data = features[features["symbol"] == "SPY"][["date", "ret_20d", "ret_60d",
                                                       "dist_sma200"]].copy()

    for year in YEARS:
        df = all_preds.get(year)
        if df is None:
            continue

        # Get short candidates' characteristics
        all_short_cands = []
        for date in sorted(df["date"].unique()):
            day = df[df["date"] == date].sort_values("prob_short", ascending=False)
            if len(day) < 20:
                continue
            top5 = day.head(5)
            all_short_cands.append(top5)

        if not all_short_cands:
            continue

        sc_df = pd.concat(all_short_cands)
        avg_ret = sc_df["fwd_ret"].mean()

        # Get SPY stats for the year
        year_spy = spy_data[(spy_data["date"].dt.year == year)]
        spy_avg_20d = year_spy["ret_20d"].mean() if len(year_spy) > 0 else np.nan
        spy_avg_60d = year_spy["ret_60d"].mean() if len(year_spy) > 0 else np.nan

        # Most shorted symbols
        sym_counts = sc_df["symbol"].value_counts().head(10)
        top_syms = ", ".join([f"{s}({c})" for s, c in sym_counts.items()])

        status = "FAIL" if avg_ret > 0 else "OK"
        log(f"    {year} [{status}]: sc_ret={avg_ret:+.3%}, "
            f"spy_20d_avg={spy_avg_20d:+.3%}, spy_60d_avg={spy_avg_60d:+.3%}")
        log(f"      Top shorted: {top_syms}")


# ═══════════════════════════════════════════════════════════════════
# 6. MULTI-SIGNAL CONFIRMATION
# ═══════════════════════════════════════════════════════════════════

def analyze_multi_signal(all_preds, features):
    """Only short stocks where multiple deterioration signals fire."""
    log("\n  ─── MULTI-SIGNAL CONFIRMATION ───")

    # Load enriched features and merge with predictions
    feat_cols = [
        "date", "symbol",
        "accruals_ratio", "leverage_change_qoq", "gross_margin_qoq_chg",
        "operating_margin_qoq_chg", "eps_miss_streak", "debt_growth_qoq",
        "insider_sell_ratio_90d", "below_sma200", "momentum_collapse",
        "death_cross", "near_52w_low", "interest_coverage_change",
    ]

    avail_cols = [c for c in feat_cols if c in features.columns]
    feat_subset = features[avail_cols].copy()

    # Define deterioration signals
    signal_defs = {
        "high_accruals": ("accruals_ratio", lambda x: x > 0.05),
        "leverage_up": ("leverage_change_qoq", lambda x: x > 0.02),
        "margin_falling": ("gross_margin_qoq_chg", lambda x: x < -0.02),
        "op_margin_falling": ("operating_margin_qoq_chg", lambda x: x < -0.02),
        "eps_miss": ("eps_miss_streak", lambda x: x >= 1),
        "debt_growing": ("debt_growth_qoq", lambda x: x > 0.05),
        "below_sma200": ("below_sma200", lambda x: x > 0.5),
        "momentum_collapse": ("momentum_collapse", lambda x: x > 0.5),
        "death_cross": ("death_cross", lambda x: x > 0.5),
        "near_52w_low": ("near_52w_low", lambda x: x > 0.5),
    }

    results = {}
    for min_signals in [0, 2, 3, 4, 5]:
        yearly_rets = []
        all_rets = []
        all_pool_sizes = []

        for year in YEARS:
            df = all_preds.get(year)
            if df is None:
                continue

            # Merge features
            merged = df.merge(feat_subset, on=["date", "symbol"], how="left")

            day_rets = []
            for date in sorted(merged["date"].unique()):
                day = merged[merged["date"] == date].sort_values("prob_short", ascending=False)
                if len(day) < 20:
                    continue

                # Count deterioration signals per stock
                if min_signals > 0:
                    signal_count = pd.Series(0, index=day.index)
                    for sig_name, (col, fn) in signal_defs.items():
                        if col in day.columns:
                            signal_count += day[col].apply(
                                lambda x: 1 if not pd.isna(x) and fn(x) else 0)

                    # Filter to stocks with enough signals
                    confirmed = day[signal_count >= min_signals]
                    if len(confirmed) < 1:
                        continue
                    all_pool_sizes.append(len(confirmed))
                    top5 = confirmed.sort_values("prob_short", ascending=False).head(5)
                else:
                    top5 = day.head(5)
                    all_pool_sizes.append(len(day))

                ret = top5["fwd_ret"].mean()
                day_rets.append(ret)

            if day_rets:
                yearly_rets.append({
                    "year": year,
                    "mean": np.mean(day_rets),
                    "n_days": len(day_rets),
                })
                all_rets.extend(day_rets)

        if not all_rets:
            log(f"    min_signals={min_signals}: no data")
            continue

        t_stat, p_val = stats.ttest_1samp(all_rets, 0)
        neg_years = sum(1 for r in yearly_rets if r["mean"] < 0)
        avg_ret = np.mean([r["mean"] for r in yearly_rets])
        avg_pool = np.mean(all_pool_sizes) if all_pool_sizes else 0

        results[min_signals] = {
            "avg_ret": avg_ret,
            "t_stat": t_stat,
            "p_val": p_val,
            "neg_years": neg_years,
            "total_years": len(yearly_rets),
            "n_days": len(all_rets),
            "avg_pool": avg_pool,
            "yearly": yearly_rets,
        }
        log(f"    min_signals>={min_signals}: avg={avg_ret:+.3%}, "
            f"neg_years={neg_years}/{len(yearly_rets)}, "
            f"days={len(all_rets)}, pool={avg_pool:.0f}, "
            f"t={t_stat:.2f}, p={p_val:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 7. SECTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_sectors(all_preds, features):
    """Which sectors produce actual short signal?"""
    log("\n  ─── SECTOR ANALYSIS ───")

    # We don't have sector directly in predictions, but can analyze
    # by looking at which symbols consistently appear as short candidates
    # and their forward returns

    # Collect all short candidate returns by symbol
    sym_rets = {}
    for year in YEARS:
        df = all_preds.get(year)
        if df is None:
            continue

        for date in sorted(df["date"].unique()):
            day = df[df["date"] == date].sort_values("prob_short", ascending=False)
            if len(day) < 20:
                continue
            top5 = day.head(5)
            for _, row in top5.iterrows():
                sym = row["symbol"]
                if sym not in sym_rets:
                    sym_rets[sym] = []
                sym_rets[sym].append(row["fwd_ret"])

    # Rank symbols by average short candidate return
    sym_stats = []
    for sym, rets in sym_rets.items():
        if len(rets) >= 20:  # at least 20 appearances
            sym_stats.append({
                "symbol": sym,
                "avg_ret": np.mean(rets),
                "median_ret": np.median(rets),
                "count": len(rets),
                "pct_neg": np.mean([r < 0 for r in rets]),
            })

    sym_stats.sort(key=lambda x: x["avg_ret"])

    log(f"    Symbols appearing as short candidates >= 20 times: {len(sym_stats)}")
    log(f"\n    BEST SHORTS (actually go down when shorted):")
    for s in sym_stats[:15]:
        log(f"      {s['symbol']:6s}: avg={s['avg_ret']:+.3%}, "
            f"median={s['median_ret']:+.3%}, "
            f"count={s['count']:4d}, neg%={s['pct_neg']:.0%}")

    log(f"\n    WORST SHORTS (go UP when shorted):")
    for s in sym_stats[-10:]:
        log(f"      {s['symbol']:6s}: avg={s['avg_ret']:+.3%}, "
            f"median={s['median_ret']:+.3%}, "
            f"count={s['count']:4d}, neg%={s['pct_neg']:.0%}")

    return sym_stats


# ═══════════════════════════════════════════════════════════════════
# 8. ALTERNATIVE TARGETS — What if we predict bottom 10% or absolute loss?
# ═══════════════════════════════════════════════════════════════════

def analyze_quintile_buckets(all_preds):
    """What if we only short stocks in the top-1% of prob_short (most extreme)?"""
    log("\n  ─── EXTREME CONCENTRATION (percentile-based) ───")

    for pct in [1, 2, 5, 10, 20]:
        yearly_rets = []
        all_rets = []

        for year in YEARS:
            df = all_preds.get(year)
            if df is None:
                continue

            day_rets = []
            for date in sorted(df["date"].unique()):
                day = df[df["date"] == date]
                if len(day) < 20:
                    continue

                # Take top X% by prob_short
                n = max(1, int(len(day) * pct / 100))
                top_pct = day.nlargest(n, "prob_short")
                ret = top_pct["fwd_ret"].mean()
                day_rets.append(ret)

            if day_rets:
                yearly_rets.append({
                    "year": year,
                    "mean": np.mean(day_rets),
                })
                all_rets.extend(day_rets)

        if not all_rets:
            continue

        t_stat, p_val = stats.ttest_1samp(all_rets, 0)
        neg_years = sum(1 for r in yearly_rets if r["mean"] < 0)
        avg_ret = np.mean([r["mean"] for r in yearly_rets])

        log(f"    top-{pct:2d}% (~{int(500*pct/100)} stocks): avg={avg_ret:+.3%}, "
            f"neg_years={neg_years}/{len(yearly_rets)}, "
            f"t={t_stat:.2f}, p={p_val:.4f}")


# ═══════════════════════════════════════════════════════════════════
# GENERATE REPORT
# ═══════════════════════════════════════════════════════════════════

def generate_report(top_n_results, regime_results, confidence_results,
                    combined_results, multi_signal_results, sym_stats):
    lines = []
    lines.append("# Short Model v2 — Deep Analysis & Improvement Research")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # Top-N
    lines.append("\n## 1. Top-N Concentration\n")
    lines.append("Does concentrating on fewer, highest-conviction short candidates help?\n")
    lines.append("| Top-N | Avg Fwd Ret | Median Fwd Ret | Neg Years | % Neg Days | t-stat | p-value |")
    lines.append("|-------|-------------|----------------|-----------|------------|--------|---------|")
    for n, r in sorted(top_n_results.items()):
        lines.append(f"| {n} | {r['avg_ret']:+.3%} | {r['median_ret']:+.3%} | "
                     f"{r['neg_years']}/{r['total_years']} | {r['pct_neg_days']:.1%} | "
                     f"{r['t_stat']:.2f} | {r['p_val']:.4f} |")

    # Find best top-N
    best_n = min(top_n_results.items(), key=lambda x: x[1]["avg_ret"])
    lines.append(f"\n**Best**: Top-{best_n[0]} with avg fwd return = {best_n[1]['avg_ret']:+.3%}")

    # Regime
    lines.append("\n## 2. Regime Filtering\n")
    lines.append("Only short when market conditions are favorable for shorts.\n")
    lines.append("| Regime | Avg Fwd Ret | Neg Years | Active % | Days | t-stat | p-value |")
    lines.append("|--------|-------------|-----------|----------|------|--------|---------|")
    for name, r in regime_results.items():
        lines.append(f"| {name} | {r['avg_ret']:+.3%} | "
                     f"{r['neg_years']}/{r['total_years']} | {r['active_pct']:.0%} | "
                     f"{r['n_days']} | {r['t_stat']:.2f} | {r['p_val']:.4f} |")

    # Confidence
    lines.append("\n## 3. Confidence Thresholds\n")
    lines.append("Only short when model confidence exceeds a threshold.\n")
    lines.append("| Threshold | Avg Fwd Ret | Neg Years | Days | Avg Pool | t-stat | p-value |")
    lines.append("|-----------|-------------|-----------|------|----------|--------|---------|")
    for thresh, r in sorted(confidence_results.items()):
        lines.append(f"| >{thresh:.2f} | {r['avg_ret']:+.3%} | "
                     f"{r['neg_years']}/{r['total_years']} | {r['n_days']} | "
                     f"{r['avg_pool_size']:.0f} | {r['t_stat']:.2f} | {r['p_val']:.4f} |")

    # Combined
    lines.append("\n## 4. Combined Filters (Best Combinations)\n")
    lines.append("| Combination | Avg Fwd Ret | Neg Years | Active % | t-stat | p-value |")
    lines.append("|-------------|-------------|-----------|----------|--------|---------|")
    for name, r in sorted(combined_results.items(), key=lambda x: x[1].get("avg_ret", 999)):
        if r.get("skip"):
            continue
        lines.append(f"| {name} | {r['avg_ret']:+.3%} | "
                     f"{r['neg_years']}/{r['total_years']} | {r['active_pct']:.0%} | "
                     f"{r['t_stat']:.2f} | {r['p_val']:.4f} |")

    # Multi-signal
    lines.append("\n## 5. Multi-Signal Confirmation\n")
    lines.append("Require N deterioration signals to fire before shorting.\n")
    lines.append("| Min Signals | Avg Fwd Ret | Neg Years | Days | Avg Pool | t-stat | p-value |")
    lines.append("|-------------|-------------|-----------|------|----------|--------|---------|")
    for n_sig, r in sorted(multi_signal_results.items()):
        lines.append(f"| >={n_sig} | {r['avg_ret']:+.3%} | "
                     f"{r['neg_years']}/{r['total_years']} | {r['n_days']} | "
                     f"{r['avg_pool']:.0f} | {r['t_stat']:.2f} | {r['p_val']:.4f} |")

    # Best shorts
    lines.append("\n## 6. Best & Worst Short Candidates (by symbol)\n")
    lines.append("Symbols that appear ≥20 times as short candidates:\n")
    lines.append("### Best Shorts (actually decline)")
    lines.append("| Symbol | Avg Fwd Ret | Count | % Negative |")
    lines.append("|--------|-------------|-------|------------|")
    for s in sym_stats[:15]:
        lines.append(f"| {s['symbol']} | {s['avg_ret']:+.3%} | {s['count']} | {s['pct_neg']:.0%} |")

    lines.append("\n### Worst Shorts (go UP when shorted)")
    lines.append("| Symbol | Avg Fwd Ret | Count | % Negative |")
    lines.append("|--------|-------------|-------|------------|")
    for s in sym_stats[-10:]:
        lines.append(f"| {s['symbol']} | {s['avg_ret']:+.3%} | {s['count']} | {s['pct_neg']:.0%} |")

    # Overall recommendation
    lines.append("\n## 7. Recommendation\n")

    # Find best overall approach
    best_combined = min(
        [(k, v) for k, v in combined_results.items() if not v.get("skip")],
        key=lambda x: x[1]["avg_ret"]
    )
    best_regime = min(regime_results.items(), key=lambda x: x[1]["avg_ret"])

    lines.append(f"### Best single filter: `{best_regime[0]}`")
    lines.append(f"- Avg fwd return: {best_regime[1]['avg_ret']:+.3%}")
    lines.append(f"- Negative years: {best_regime[1]['neg_years']}/{best_regime[1]['total_years']}")
    lines.append(f"- Active: {best_regime[1]['active_pct']:.0%} of trading days")

    lines.append(f"\n### Best combined filter: `{best_combined[0]}`")
    lines.append(f"- Avg fwd return: {best_combined[1]['avg_ret']:+.3%}")
    lines.append(f"- Negative years: {best_combined[1]['neg_years']}/{best_combined[1]['total_years']}")
    lines.append(f"- Active: {best_combined[1]['active_pct']:.0%} of trading days")

    if best_combined[1]["avg_ret"] < 0:
        lines.append(f"\n**SHORT SIGNAL FOUND** with combined filter `{best_combined[0]}`!")
        lines.append(f"Short candidates avg = {best_combined[1]['avg_ret']:+.3%} (NEGATIVE)")
        lines.append(f"→ Proceed to build B4 long/short with this filter configuration.")
    elif best_regime[1]["avg_ret"] < 0:
        lines.append(f"\n**SHORT SIGNAL FOUND** with regime filter `{best_regime[0]}`!")
        lines.append(f"Short candidates avg = {best_regime[1]['avg_ret']:+.3%} (NEGATIVE)")
        lines.append(f"→ Proceed to build B4 long/short with regime gating.")
    else:
        lines.append(f"\n**NO NEGATIVE SHORT SIGNAL** even with filters.")
        lines.append(f"Best we can achieve: {min(best_combined[1]['avg_ret'], best_regime[1]['avg_ret']):+.3%}")
        lines.append(f"Consider: long/short with relative underperformance (spread-based) instead of absolute short.")

    return "\n".join(lines)


def main():
    log("=" * 70)
    log("  SHORT MODEL v2 — DEEP ANALYSIS & IMPROVEMENT RESEARCH")
    log("=" * 70)

    log("\nLoading predictions ...")
    all_preds = load_predictions()
    log(f"  Loaded {len(all_preds)} years")

    log("\nLoading features ...")
    features = load_features()
    log(f"  Shape: {features.shape}")

    # Run all analyses
    top_n_results = analyze_top_n(all_preds)
    regime_results = analyze_regime_filter(all_preds, features)
    confidence_results = analyze_confidence_threshold(all_preds)
    combined_results = analyze_combined_filters(all_preds, features)
    analyze_failure_years(all_preds, features)
    multi_signal_results = analyze_multi_signal(all_preds, features)
    sym_stats = analyze_sectors(all_preds, features)
    analyze_quintile_buckets(all_preds)

    # Generate report
    report = generate_report(top_n_results, regime_results, confidence_results,
                            combined_results, multi_signal_results, sym_stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUT_DIR / "SHORT_DEEP_ANALYSIS.md"
    report_file.write_text(report)
    log(f"\n  Report: {report_file}")
    log("  Done!")


if __name__ == "__main__":
    main()
