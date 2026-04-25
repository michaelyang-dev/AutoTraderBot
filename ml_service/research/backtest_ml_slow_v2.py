"""
ML Slow v2 Backtest
===================
Tests the expanded-feature ML Slow v2 model in 3 configurations:

  A) ML Slow v2 alone (max 5 positions)
  B) Path B baseline (ML Medium + Momentum + Mean Reversion)
  C) Path B + ML Slow v2 combined (2+4+2+1+2 flex = 11 max)

5 deployment gates:
  1. ML Slow v2 OOS AUC > 0.58
  2. ML Slow v2 standalone alpha > 0
  3. Path B + ML Slow v2 Sharpe >= 1.655
  4. Path B + ML Slow v2 CAGR >= 24.08%
  5. At least 3 picks per year in every year

Run with:
    python3 backtest_ml_slow_v2.py
"""

import warnings
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from unified_backtester import (
    INITIAL_CASH, HOLD_DAYS,
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy, MLSlowStrategy,
    SlotConfig,
    SLOT_ML_MOM_MR,
)
from backtest_utils import load_predictions, calc_metrics, calc_alpha_beta
from diagnose_combined import instrumented_run

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"

# Slot configs
SLOT_SLOW_V2_ONLY = SlotConfig(
    strategy_slots={"ml_slow": 5}, flex_slots=0, max_positions=5,
)
SLOT_PATHB_SLOW_V2 = SlotConfig(
    strategy_slots={"ml_medium": 2, "momentum": 4, "mean_reversion": 2, "ml_slow": 1},
    flex_slots=2, max_positions=11,
)

# Baseline targets
BASELINE_CAGR   = 0.2408
BASELINE_SHARPE = 1.655
AUC_TARGET      = 0.58


def fetch_ohlcv(symbols, start, end):
    all_syms = list(set(["SPY"] + symbols))
    raw = yf.download(all_syms, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume = None
    if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
        volume = raw["Volume"]
        volume.index = pd.to_datetime(volume.index).tz_localize(None)
    return close, volume


def compute_correlations(vals_dict, label_pairs):
    """Compute daily return correlations between strategy pairs."""
    results = {}
    for a, b in label_pairs:
        if a in vals_dict and b in vals_dict:
            ra = vals_dict[a].pct_change().dropna()
            rb = vals_dict[b].pct_change().dropna()
            common = pd.concat([ra, rb], axis=1, join="inner").dropna()
            if len(common) > 10:
                results[f"{a} vs {b}"] = common.iloc[:, 0].corr(common.iloc[:, 1])
    return results


def main():
    t0 = time.perf_counter()
    print("=" * 90)
    print("  ML SLOW v2 BACKTEST — Expanded Features")
    print("=" * 90)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading ML Medium predictions ...")
    df_med = load_predictions()

    pred_file = DATA_DIR / "predictions_slow_v2.parquet"
    if not pred_file.exists():
        print(f"  ERROR: {pred_file} not found — run train_model_slow_v2.py first.")
        return
    df_slow = pd.read_parquet(pred_file)
    df_slow["date"] = pd.to_datetime(df_slow["date"])
    df_slow = df_slow.sort_values(["date", "symbol"])
    print(f"  ML Slow v2 predictions: {len(df_slow):,} rows")

    # Get OOS AUC from predictions
    from sklearn.metrics import roc_auc_score
    oos_auc = roc_auc_score(
        df_slow["target_slow"].values.astype(int),
        df_slow["prob"].values,
    )
    print(f"  ML Slow v2 OOS AUC: {oos_auc:.4f}")

    common_dates = sorted(set(df_med["date"].unique()) & set(df_slow["date"].unique()))
    all_dates = common_dates
    universe_syms = sorted(set(df_med["symbol"].unique()) | set(df_slow["symbol"].unique()))
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    print(f"  {len(all_dates)} trading days | {years:.1f} years")

    # ── Fetch prices ─────────────────────────────────────────────────────
    print("\n  Fetching price & volume data ...")
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    close, volume = fetch_ohlcv(universe_syms, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    volume_aligned = volume.reindex(sim_index, method="ffill") if volume is not None else None

    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close_aligned["SPY"].to_dict()
    m_spy = calc_metrics(spy_bh, [], years, "SPY B&H")

    # ── Build strategies ─────────────────────────────────────────────────
    ml_med_strat = MLMediumStrategy(df_med, threshold=0.55)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)
    ml_slow_strat = MLSlowStrategy(df_slow, price_data=close)

    # ── A) ML Slow v2 alone ──────────────────────────────────────────────
    print("\n  [A] Running ML Slow v2 alone ...")
    slow_vals, slow_trades, d_slow = instrumented_run(
        [ml_slow_strat], SLOT_SLOW_V2_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML Slow v2")
    m_slow = calc_metrics(slow_vals, slow_trades, years, "ML Slow v2")
    a_slow, _ = calc_alpha_beta(slow_vals, spy_bh.reindex(slow_vals.index, method="ffill"))
    m_slow["alpha"] = a_slow
    m_slow["avg_pos"] = np.mean(d_slow["daily_pos_count"])
    print(f"      CAGR {m_slow['cagr']:+.2%} | Sharpe {m_slow['sharpe']:.3f} | "
          f"Alpha {m_slow['alpha']:+.2%} | Trades {m_slow['n_trades']}")

    # ── B) Path B baseline ───────────────────────────────────────────────
    print("\n  [B] Running Path B baseline ...")
    pathb_vals, pathb_trades, d_pathb = instrumented_run(
        [ml_med_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B")
    m_pathb = calc_metrics(pathb_vals, pathb_trades, years, "Path B")
    a_pathb, _ = calc_alpha_beta(pathb_vals, spy_bh.reindex(pathb_vals.index, method="ffill"))
    m_pathb["alpha"] = a_pathb
    m_pathb["avg_pos"] = np.mean(d_pathb["daily_pos_count"])
    print(f"      CAGR {m_pathb['cagr']:+.2%} | Sharpe {m_pathb['sharpe']:.3f} | "
          f"Alpha {m_pathb['alpha']:+.2%} | Trades {m_pathb['n_trades']}")

    # ── C) Path B + ML Slow v2 ───────────────────────────────────────────
    print("\n  [C] Running Path B + ML Slow v2 ...")
    combined_vals, combined_trades, d_combined = instrumented_run(
        [ml_med_strat, mom_strat, mr_strat, ml_slow_strat],
        SLOT_PATHB_SLOW_V2, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="PathB+SlowV2")
    m_combined = calc_metrics(combined_vals, combined_trades, years, "PathB+SlowV2")
    a_combined, _ = calc_alpha_beta(combined_vals, spy_bh.reindex(combined_vals.index, method="ffill"))
    m_combined["alpha"] = a_combined
    m_combined["avg_pos"] = np.mean(d_combined["daily_pos_count"])
    print(f"      CAGR {m_combined['cagr']:+.2%} | Sharpe {m_combined['sharpe']:.3f} | "
          f"Alpha {m_combined['alpha']:+.2%} | Trades {m_combined['n_trades']}")

    # ── Comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  PERFORMANCE COMPARISON")
    print(f"{'='*90}")

    configs = [
        ("SPY B&H",        m_spy),
        ("ML Slow v2",     m_slow),
        ("Path B",         m_pathb),
        ("PathB+SlowV2",   m_combined),
    ]

    header = f"  {'Config':<18} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8} {'Alpha':>8}"
    print(header)
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for label, m in configs:
        alpha_str = f"{m.get('alpha', 0):+.2%}" if 'alpha' in m else "  N/A"
        print(f"  {label:<18} {m['cagr']:>+7.2%} {m['sharpe']:>8.3f} {m['max_dd']:>7.1%} "
              f"{m['n_trades']:>8} {m['win_rate']:>7.1%} {alpha_str:>8}")

    # Delta from Path B
    print(f"\n  Delta (PathB+SlowV2 vs Path B):")
    dcagr = m_combined["cagr"] - m_pathb["cagr"]
    dsharpe = m_combined["sharpe"] - m_pathb["sharpe"]
    ddd = m_combined["max_dd"] - m_pathb["max_dd"]
    print(f"    ΔCAGR:   {dcagr:+.2%}")
    print(f"    ΔSharpe: {dsharpe:+.3f}")
    print(f"    ΔMaxDD:  {ddd:+.1%}")

    # ── Correlations ─────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  STRATEGY RETURN CORRELATIONS")
    print(f"{'='*90}")

    # Run individual strategies for correlation
    # ML Medium alone
    from unified_backtester import SLOT_ML_ONLY, SLOT_MOM_ONLY, SLOT_MR_ONLY
    ml_vals, ml_trades, _ = instrumented_run(
        [ml_med_strat], SLOT_ML_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML Medium")
    mom_vals, mom_trades, _ = instrumented_run(
        [mom_strat], SLOT_MOM_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Momentum")

    from unified_backtester import SLOT_MR_ONLY
    mr_vals, mr_trades, _ = instrumented_run(
        [mr_strat], SLOT_MR_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="MeanRev")

    vals_dict = {
        "ML Medium": ml_vals,
        "Momentum": mom_vals,
        "MeanRev": mr_vals,
        "ML Slow v2": slow_vals,
    }

    pairs = [
        ("ML Slow v2", "ML Medium"),
        ("ML Slow v2", "Momentum"),
        ("ML Slow v2", "MeanRev"),
        ("ML Medium", "Momentum"),
        ("ML Medium", "MeanRev"),
    ]
    corrs = compute_correlations(vals_dict, pairs)
    for pair, corr in corrs.items():
        print(f"  {pair:<30} {corr:+.3f}")

    # ── Pick rate by year ────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  PICK RATE BY YEAR (ML Slow v2, prob >= 0.55)")
    print(f"{'='*90}")

    df_slow["year"] = df_slow["date"].dt.year
    picks = df_slow[df_slow["prob"] >= 0.55]
    all_years = sorted(df_slow["year"].unique())
    sparse_years = []

    print(f"\n  {'Year':>6} {'Picks':>8} {'Total':>8} {'Rate':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
    for yr in all_years:
        yr_total = len(df_slow[df_slow["year"] == yr])
        yr_picks = len(picks[picks["year"] == yr])
        rate = yr_picks / yr_total * 100 if yr_total > 0 else 0
        flag = " ⚠" if yr_picks < 3 else ""
        print(f"  {yr:>6} {yr_picks:>8} {yr_total:>8} {rate:>7.1f}%{flag}")
        if yr_picks < 3:
            sparse_years.append(yr)

    if sparse_years:
        print(f"\n  WARNING: Years with < 3 picks: {', '.join(str(y) for y in sparse_years)}")

    # ── Feature importance ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  FEATURE IMPORTANCE (top 15)")
    print(f"{'='*90}")

    import joblib
    model = joblib.load(DATA_DIR / "model_slow_v2.lgb")
    try:
        cc = model.calibrated_classifiers_[0]
        lgb_model = cc.estimator.estimator  # FrozenEstimator -> LGBMClassifier
        if hasattr(lgb_model, 'booster_'):
            df_feat = pd.read_parquet(DATA_DIR / "features.parquet")
            exclude = {"date", "symbol", "target", "target_slow", "fwd_ret_30d", "fwd_ret"}
            fwd_kw = {"fwd", "forward", "future"}
            feature_cols = [c for c in df_feat.columns if c not in exclude
                           and not any(k in c.lower() for k in fwd_kw)]

            new_feats = [
                "ret_126d","ret_252d","dist_52w_high","dist_52w_low","sma200_slope",
                "max_dd_6m","consec_up_months","consec_down_months",
                "yield_curve_10y2y","yield_curve_30d_change","hy_spread","hy_spread_30d_change",
                "dxy_level","dxy_30d_change","hyg_lqd_ratio","hyg_lqd_30d_change",
                "copper_gold_ratio","copper_gold_30d_change",
                "return_rank_3m","return_rank_6m","return_rank_12m",
                "vol_rank_3m","vol_126d","vol_rank_6m",
            ]

            importance = pd.Series(
                lgb_model.booster_.feature_importance(importance_type="gain"),
                index=feature_cols,
            ).sort_values(ascending=False)
            total = importance.sum()

            for rank, (feat, gain) in enumerate(importance.head(15).items(), 1):
                is_new = "NEW" if feat in new_feats else "   "
                pct = gain / total * 100
                bar = "█" * int(pct * 2)
                print(f"  {rank:2d}. [{is_new}] {feat:<24} {pct:5.1f}%  {bar}")

            n_new_top15 = sum(1 for f in importance.head(15).index if f in new_feats)
            print(f"\n  New features in top 15: {n_new_top15}/15")
    except Exception as e:
        print(f"  Could not extract feature importance: {e}")

    # ── 5 GATE CHECKS ────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  DEPLOYMENT GATE CHECKS")
    print(f"{'='*90}")

    g1 = oos_auc > AUC_TARGET
    g2 = m_slow["alpha"] > 0
    g3 = m_combined["sharpe"] >= BASELINE_SHARPE
    g4 = m_combined["cagr"] >= BASELINE_CAGR
    g5 = len(sparse_years) == 0

    gates = [
        ("Gate 1: OOS AUC > 0.58",           g1, f"AUC = {oos_auc:.4f}"),
        ("Gate 2: Standalone alpha > 0",      g2, f"Alpha = {m_slow['alpha']:+.2%}"),
        ("Gate 3: Combined Sharpe >= 1.655",  g3, f"Sharpe = {m_combined['sharpe']:.3f}"),
        ("Gate 4: Combined CAGR >= 24.08%",   g4, f"CAGR = {m_combined['cagr']:+.2%}"),
        ("Gate 5: >= 3 picks/yr every year",  g5, f"Sparse years: {len(sparse_years)}"),
    ]

    all_pass = True
    for name, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  {status:>6}  {name:<40}  ({detail})")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    if all_pass:
        print("  VERDICT: DEPLOY ML Slow v2")
        print("  All 5 gates passed. ML Slow v2 improves the system.")
    else:
        n_pass = sum(1 for _, p, _ in gates if p)
        print(f"  VERDICT: DO NOT DEPLOY ML Slow v2 ({n_pass}/5 gates passed)")
        print()
        if not g1:
            print(f"  - OOS AUC ({oos_auc:.4f}) below 0.58 threshold")
        if not g2:
            print(f"  - Standalone alpha ({m_slow['alpha']:+.2%}) is not positive")
        if not g3:
            print(f"  - Combined Sharpe ({m_combined['sharpe']:.3f}) below baseline ({BASELINE_SHARPE})")
        if not g4:
            print(f"  - Combined CAGR ({m_combined['cagr']:+.2%}) below baseline ({BASELINE_CAGR:+.2%})")
        if not g5:
            print(f"  - Sparse pick years: {', '.join(str(y) for y in sparse_years)}")
    print(f"{'='*90}")

    print(f"\n  Runtime: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
