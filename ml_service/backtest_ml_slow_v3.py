"""
ML Slow v3 Backtest — Relative Outperformance
==============================================
Tests ML Slow v3 (target: stock 30d ret - SPY 30d ret > 3%) in 4 configs:

  A) ML Slow v3 alone (max 5 positions)
  B) Path B baseline (ML Medium + Momentum + MR)
  C) Path B + ML Slow v3 at threshold 0.55 (max 11)
  D) Path B + ML Slow v3 at threshold 0.50 (max 11, if 0.55 has silent years)

5 deployment gates (must pass ALL):
  1. OOS AUC > 0.58
  2. Standalone alpha > 0
  3. Combined Sharpe >= 1.655
  4. Combined CAGR >= 24.08%
  5. >= 3 picks/yr in EVERY year

Run with:
    python3 backtest_ml_slow_v3.py
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from sklearn.metrics import roc_auc_score

from unified_backtester import (
    INITIAL_CASH, HOLD_DAYS,
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy, MLSlowStrategy,
    SlotConfig,
    SLOT_ML_MOM_MR, SLOT_ML_ONLY, SLOT_MOM_ONLY, SLOT_MR_ONLY,
)
from backtest_ml import load_predictions, calc_metrics, calc_alpha_beta
from diagnose_combined import instrumented_run

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"

# Slot configs
SLOT_SLOW_ONLY = SlotConfig(
    strategy_slots={"ml_slow": 5}, flex_slots=0, max_positions=5,
)
SLOT_PATHB_SLOW = SlotConfig(
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


def main():
    t0 = time.perf_counter()
    print("=" * 90)
    print("  ML SLOW v3 BACKTEST — Relative Outperformance")
    print("  Target: (stock_30d_ret - SPY_30d_ret) > 3%")
    print("=" * 90)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading ML Medium predictions ...")
    df_med = load_predictions()

    pred_file = DATA_DIR / "predictions_slow_v3.parquet"
    if not pred_file.exists():
        print(f"  ERROR: {pred_file} not found")
        return
    df_slow = pd.read_parquet(pred_file)
    df_slow["date"] = pd.to_datetime(df_slow["date"])
    df_slow = df_slow.sort_values(["date", "symbol"])
    print(f"  ML Slow v3 predictions: {len(df_slow):,} rows")

    # OOS AUC
    oos_auc = roc_auc_score(
        df_slow["target_v3"].values.astype(int),
        df_slow["prob"].values,
    )
    print(f"  OOS AUC: {oos_auc:.4f}")

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

    # ── Build strategies ─────────────────────────────────────────────────
    ml_med_strat = MLMediumStrategy(df_med, threshold=0.55)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ── A) ML Slow v3 alone (threshold 0.55) ─────────────────────────────
    print("\n  [A] ML Slow v3 alone (threshold 0.55) ...")
    ml_slow_55 = MLSlowStrategy(df_slow, price_data=close)
    ml_slow_55.PROB_THRESHOLD = 0.55
    ml_slow_55._signals_by_date = {}
    ml_slow_55._build_lookup(df_slow)

    slow_vals_55, slow_trades_55, d_slow_55 = instrumented_run(
        [ml_slow_55], SLOT_SLOW_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="SlowV3@0.55")
    m_slow_55 = calc_metrics(slow_vals_55, slow_trades_55, years, "SlowV3@0.55")
    a_slow_55, _ = calc_alpha_beta(slow_vals_55, spy_bh.reindex(slow_vals_55.index, method="ffill"))
    m_slow_55["alpha"] = a_slow_55
    m_slow_55["avg_pos"] = np.mean(d_slow_55["daily_pos_count"])
    print(f"      CAGR {m_slow_55['cagr']:+.2%} | Sharpe {m_slow_55['sharpe']:.3f} | "
          f"Alpha {m_slow_55['alpha']:+.2%} | Trades {m_slow_55['n_trades']}")

    # ── A2) ML Slow v3 alone (threshold 0.50) ────────────────────────────
    print("\n  [A2] ML Slow v3 alone (threshold 0.50) ...")
    ml_slow_50 = MLSlowStrategy(df_slow, price_data=close)
    ml_slow_50.PROB_THRESHOLD = 0.50
    ml_slow_50._signals_by_date = {}
    ml_slow_50._build_lookup(df_slow)

    slow_vals_50, slow_trades_50, d_slow_50 = instrumented_run(
        [ml_slow_50], SLOT_SLOW_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="SlowV3@0.50")
    m_slow_50 = calc_metrics(slow_vals_50, slow_trades_50, years, "SlowV3@0.50")
    a_slow_50, _ = calc_alpha_beta(slow_vals_50, spy_bh.reindex(slow_vals_50.index, method="ffill"))
    m_slow_50["alpha"] = a_slow_50
    m_slow_50["avg_pos"] = np.mean(d_slow_50["daily_pos_count"])
    print(f"      CAGR {m_slow_50['cagr']:+.2%} | Sharpe {m_slow_50['sharpe']:.3f} | "
          f"Alpha {m_slow_50['alpha']:+.2%} | Trades {m_slow_50['n_trades']}")

    # ── B) Path B baseline ───────────────────────────────────────────────
    print("\n  [B] Path B baseline ...")
    pathb_vals, pathb_trades, d_pathb = instrumented_run(
        [ml_med_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B")
    m_pathb = calc_metrics(pathb_vals, pathb_trades, years, "Path B")
    a_pathb, _ = calc_alpha_beta(pathb_vals, spy_bh.reindex(pathb_vals.index, method="ffill"))
    m_pathb["alpha"] = a_pathb
    m_pathb["avg_pos"] = np.mean(d_pathb["daily_pos_count"])
    print(f"      CAGR {m_pathb['cagr']:+.2%} | Sharpe {m_pathb['sharpe']:.3f} | "
          f"Alpha {m_pathb['alpha']:+.2%} | Trades {m_pathb['n_trades']}")

    # ── C) Path B + ML Slow v3 @ 0.55 ────────────────────────────────────
    print("\n  [C] Path B + ML Slow v3 @ 0.55 ...")
    ml_slow_c55 = MLSlowStrategy(df_slow, price_data=close)
    ml_slow_c55.PROB_THRESHOLD = 0.55
    ml_slow_c55._signals_by_date = {}
    ml_slow_c55._build_lookup(df_slow)

    comb55_vals, comb55_trades, d_comb55 = instrumented_run(
        [ml_med_strat, mom_strat, mr_strat, ml_slow_c55],
        SLOT_PATHB_SLOW, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="PathB+V3@0.55")
    m_comb55 = calc_metrics(comb55_vals, comb55_trades, years, "PathB+V3@0.55")
    a_comb55, _ = calc_alpha_beta(comb55_vals, spy_bh.reindex(comb55_vals.index, method="ffill"))
    m_comb55["alpha"] = a_comb55
    m_comb55["avg_pos"] = np.mean(d_comb55["daily_pos_count"])
    print(f"      CAGR {m_comb55['cagr']:+.2%} | Sharpe {m_comb55['sharpe']:.3f} | "
          f"Alpha {m_comb55['alpha']:+.2%} | Trades {m_comb55['n_trades']}")

    # ── D) Path B + ML Slow v3 @ 0.50 ────────────────────────────────────
    print("\n  [D] Path B + ML Slow v3 @ 0.50 ...")
    ml_slow_c50 = MLSlowStrategy(df_slow, price_data=close)
    ml_slow_c50.PROB_THRESHOLD = 0.50
    ml_slow_c50._signals_by_date = {}
    ml_slow_c50._build_lookup(df_slow)

    comb50_vals, comb50_trades, d_comb50 = instrumented_run(
        [ml_med_strat, mom_strat, mr_strat, ml_slow_c50],
        SLOT_PATHB_SLOW, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="PathB+V3@0.50")
    m_comb50 = calc_metrics(comb50_vals, comb50_trades, years, "PathB+V3@0.50")
    a_comb50, _ = calc_alpha_beta(comb50_vals, spy_bh.reindex(comb50_vals.index, method="ffill"))
    m_comb50["alpha"] = a_comb50
    m_comb50["avg_pos"] = np.mean(d_comb50["daily_pos_count"])
    print(f"      CAGR {m_comb50['cagr']:+.2%} | Sharpe {m_comb50['sharpe']:.3f} | "
          f"Alpha {m_comb50['alpha']:+.2%} | Trades {m_comb50['n_trades']}")

    # ── Comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  PERFORMANCE COMPARISON")
    print(f"{'='*90}")

    m_spy = calc_metrics(spy_bh, [], years, "SPY B&H")

    configs = [
        ("SPY B&H",          m_spy,     None),
        ("SlowV3@0.55",      m_slow_55, d_slow_55),
        ("SlowV3@0.50",      m_slow_50, d_slow_50),
        ("Path B",           m_pathb,   d_pathb),
        ("PathB+V3@0.55",    m_comb55,  d_comb55),
        ("PathB+V3@0.50",    m_comb50,  d_comb50),
    ]

    header = f"  {'Config':<18} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8} {'Alpha':>8} {'AvgPos':>8}"
    print(header)
    print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for label, m, diag in configs:
        alpha_str = f"{m.get('alpha', 0):+.2%}" if 'alpha' in m else "   N/A"
        avg_pos = f"{np.mean(diag['daily_pos_count']):.1f}" if diag else "   N/A"
        print(f"  {label:<18} {m['cagr']:>+7.2%} {m['sharpe']:>8.3f} {m['max_dd']:>7.1%} "
              f"{m['n_trades']:>8} {m['win_rate']:>7.1%} {alpha_str:>8} {avg_pos:>8}")

    # Deltas
    for label, m in [("PathB+V3@0.55", m_comb55), ("PathB+V3@0.50", m_comb50)]:
        dcagr = m["cagr"] - m_pathb["cagr"]
        dsharpe = m["sharpe"] - m_pathb["sharpe"]
        ddd = m["max_dd"] - m_pathb["max_dd"]
        print(f"\n  Delta ({label} vs Path B): ΔCAGR {dcagr:+.2%} | ΔSharpe {dsharpe:+.3f} | ΔMaxDD {ddd:+.1%}")

    # ── Correlations ─────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  STRATEGY RETURN CORRELATIONS (using v3 @ 0.50)")
    print(f"{'='*90}")

    # Individual strategy runs for correlation
    ml_vals, _, _ = instrumented_run(
        [ml_med_strat], SLOT_ML_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML Med")
    mom_vals, _, _ = instrumented_run(
        [mom_strat], SLOT_MOM_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Mom")
    mr_vals, _, _ = instrumented_run(
        [mr_strat], SLOT_MR_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="MR")

    vals = {
        "ML Medium": ml_vals, "Momentum": mom_vals,
        "MeanRev": mr_vals, "ML Slow v3": slow_vals_50,
    }
    pairs = [
        ("ML Slow v3", "ML Medium"),
        ("ML Slow v3", "Momentum"),
        ("ML Slow v3", "MeanRev"),
        ("ML Slow v3", "SPY"),
    ]
    # Add SPY
    vals["SPY"] = spy_bh

    for a, b in pairs:
        if a in vals and b in vals:
            ra = vals[a].pct_change().dropna()
            rb = vals[b].pct_change().dropna()
            common = pd.concat([ra, rb], axis=1, join="inner").dropna()
            if len(common) > 10:
                corr = common.iloc[:, 0].corr(common.iloc[:, 1])
                print(f"  {a:<16} vs {b:<16}  {corr:+.3f}")

    # ── Pick distribution by year ────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  PICK DISTRIBUTION BY YEAR")
    print(f"{'='*90}")

    df_slow["year"] = df_slow["date"].dt.year
    all_years = sorted(df_slow["year"].unique())

    print(f"\n  {'Year':>6}  {'@0.55':>8}  {'@0.50':>8}  {'Total':>8}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")

    sparse_55 = []
    sparse_50 = []
    for yr in all_years:
        yr_data = df_slow[df_slow["year"] == yr]
        n = len(yr_data)
        n_55 = len(yr_data[yr_data["prob"] >= 0.55])
        n_50 = len(yr_data[yr_data["prob"] >= 0.50])
        f55 = " !" if n_55 < 3 else ""
        f50 = " !" if n_50 < 3 else ""
        print(f"  {yr:>6}  {n_55:>8}{f55}  {n_50:>8}{f50}  {n:>8}")
        if n_55 < 3:
            sparse_55.append(yr)
        if n_50 < 3:
            sparse_50.append(yr)

    print(f"\n  Sparse @0.55: {', '.join(str(y) for y in sparse_55) if sparse_55 else 'none'}")
    print(f"  Sparse @0.50: {', '.join(str(y) for y in sparse_50) if sparse_50 else 'none'}")

    # ── Feature importance ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  FEATURE IMPORTANCE (top 15)")
    print(f"{'='*90}")

    import joblib
    model = joblib.load(DATA_DIR / "model_slow_v3.lgb")
    new_feats = [
        "ret_126d","ret_252d","dist_52w_high","dist_52w_low","sma200_slope",
        "max_dd_6m","consec_up_months","consec_down_months",
        "yield_curve_10y2y","yield_curve_30d_change","hy_spread","hy_spread_30d_change",
        "dxy_level","dxy_30d_change","hyg_lqd_ratio","hyg_lqd_30d_change",
        "copper_gold_ratio","copper_gold_30d_change",
        "return_rank_3m","return_rank_6m","return_rank_12m",
        "vol_rank_3m","vol_126d","vol_rank_6m",
    ]
    try:
        cc = model.calibrated_classifiers_[0]
        lgb_model = cc.estimator.estimator
        if hasattr(lgb_model, 'booster_'):
            df_feat = pd.read_parquet(DATA_DIR / "features.parquet")
            exclude = {"date","symbol","target","target_slow","target_v3",
                       "fwd_ret_30d","fwd_ret","fwd_excess_ret_30d"}
            fwd_kw = {"fwd","forward","future"}
            feature_cols = [c for c in df_feat.columns
                           if c not in exclude and not any(k in c.lower() for k in fwd_kw)]
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
            n_new = sum(1 for f in importance.head(15).index if f in new_feats)
            print(f"\n  New features in top 15: {n_new}/15")
    except Exception as e:
        print(f"  Could not extract: {e}")

    # ── GATE CHECKS ──────────────────────────────────────────────────────
    # Determine best threshold: use 0.50 if 0.55 has sparse years
    if sparse_55 and not sparse_50:
        best_thresh = 0.50
        m_best_slow = m_slow_50
        m_best_combined = m_comb50
        best_sparse = sparse_50
    elif not sparse_55:
        best_thresh = 0.55
        m_best_slow = m_slow_55
        m_best_combined = m_comb55
        best_sparse = sparse_55
    else:
        # Both have sparse years — report both
        best_thresh = 0.50
        m_best_slow = m_slow_50
        m_best_combined = m_comb50
        best_sparse = sparse_50

    print(f"\n{'='*90}")
    print(f"  DEPLOYMENT GATE CHECKS (threshold = {best_thresh:.2f})")
    print(f"{'='*90}")

    g1 = oos_auc > AUC_TARGET
    g2 = m_best_slow["alpha"] > 0
    g3 = m_best_combined["sharpe"] >= BASELINE_SHARPE
    g4 = m_best_combined["cagr"] >= BASELINE_CAGR
    g5 = len(best_sparse) == 0

    gates = [
        ("Gate 1: OOS AUC > 0.58",           g1, f"AUC = {oos_auc:.4f}"),
        ("Gate 2: Standalone alpha > 0",      g2, f"Alpha = {m_best_slow['alpha']:+.2%}"),
        ("Gate 3: Combined Sharpe >= 1.655",  g3, f"Sharpe = {m_best_combined['sharpe']:.3f}"),
        ("Gate 4: Combined CAGR >= 24.08%",   g4, f"CAGR = {m_best_combined['cagr']:+.2%}"),
        ("Gate 5: >= 3 picks/yr every year",  g5, f"Sparse years: {len(best_sparse)}"),
    ]

    all_pass = True
    for name, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  {status:>6}  {name:<40}  ({detail})")

    # Also show 0.55 gates if different
    if best_thresh == 0.50 and sparse_55:
        print(f"\n  (Gates for threshold 0.55 — for reference):")
        g5_55 = len(sparse_55) == 0
        gates_55 = [
            ("Gate 2: Standalone alpha > 0",      m_slow_55["alpha"] > 0, f"Alpha = {m_slow_55['alpha']:+.2%}"),
            ("Gate 3: Combined Sharpe >= 1.655",  m_comb55["sharpe"] >= BASELINE_SHARPE, f"Sharpe = {m_comb55['sharpe']:.3f}"),
            ("Gate 4: Combined CAGR >= 24.08%",   m_comb55["cagr"] >= BASELINE_CAGR, f"CAGR = {m_comb55['cagr']:+.2%}"),
            ("Gate 5: >= 3 picks/yr every year",  g5_55, f"Sparse: {', '.join(str(y) for y in sparse_55)}"),
        ]
        for name, passed, detail in gates_55:
            status = "PASS" if passed else "FAIL"
            print(f"  {status:>6}  {name:<40}  ({detail})")

    # ── VERDICT ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    if all_pass:
        print(f"  VERDICT: DEPLOY ML Slow v3 at threshold {best_thresh:.2f}")
        print(f"  All 5 gates passed.")
    else:
        n_pass = sum(1 for _, p, _ in gates if p)
        print(f"  VERDICT: DO NOT DEPLOY ML Slow v3 ({n_pass}/5 gates passed)")
        print()
        if not g1:
            print(f"  - OOS AUC ({oos_auc:.4f}) below 0.58")
        if not g2:
            print(f"  - Standalone alpha ({m_best_slow['alpha']:+.2%}) not positive")
        if not g3:
            print(f"  - Combined Sharpe ({m_best_combined['sharpe']:.3f}) below {BASELINE_SHARPE}")
        if not g4:
            print(f"  - Combined CAGR ({m_best_combined['cagr']:+.2%}) below {BASELINE_CAGR:+.2%}")
        if not g5:
            print(f"  - Sparse pick years: {', '.join(str(y) for y in best_sparse)}")
    print(f"{'='*90}")

    print(f"\n  Runtime: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
