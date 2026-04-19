"""
ML Slow Threshold Sweep
========================
Tests ML Slow at 6 different confidence thresholds to determine if
the 0.55 threshold is the problem or if the model lacks signal.

For each threshold: ML Slow alone + Path B + ML Slow combined.

Run with:
    python3 backtest_ml_slow_thresholds.py
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
from backtest_ml import load_predictions, calc_metrics, calc_alpha_beta
from diagnose_combined import instrumented_run

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"

THRESHOLDS = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55]

SLOT_SLOW_ONLY = SlotConfig(
    strategy_slots={"ml_slow": 5}, flex_slots=0, max_positions=5,
)
SLOT_PATHB_SLOW = SlotConfig(
    strategy_slots={"ml_medium": 2, "momentum": 4, "mean_reversion": 2, "ml_slow": 1},
    flex_slots=2, max_positions=11,
)


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
    print("  ML SLOW THRESHOLD SWEEP")
    print(f"  Thresholds: {', '.join(str(t) for t in THRESHOLDS)}")
    print("=" * 90)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading ML Medium predictions ...")
    df_med = load_predictions()

    pred_file = DATA_DIR / "predictions_slow.parquet"
    df_slow = pd.read_parquet(pred_file)
    df_slow["date"] = pd.to_datetime(df_slow["date"])
    df_slow = df_slow.sort_values(["date", "symbol"])
    print(f"  ML Slow predictions: {len(df_slow):,} rows")

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

    # ── Build non-ML-Slow strategies once ────────────────────────────────
    ml_med_strat = MLMediumStrategy(df_med, threshold=0.55)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ── Run Path B baseline once ─────────────────────────────────────────
    print("\n  Running Path B baseline ...")
    pathb_vals, pathb_trades, _ = instrumented_run(
        [ml_med_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B")
    m_pathb = calc_metrics(pathb_vals, pathb_trades, years, "Path B")
    a_pathb, _ = calc_alpha_beta(pathb_vals, spy_bh.reindex(pathb_vals.index, method="ffill"))
    m_pathb["alpha"] = a_pathb
    print(f"  Path B: CAGR {m_pathb['cagr']:+.2%} | Sharpe {m_pathb['sharpe']:.3f}")

    # ── 1. Pick rate analysis per threshold ──────────────────────────────
    print(f"\n{'='*90}")
    print("  PICK RATE ANALYSIS")
    print(f"{'='*90}")

    df_slow_ts = df_slow.copy()
    df_slow_ts["year"] = df_slow_ts["date"].dt.year
    all_years = sorted(df_slow_ts["year"].unique())

    # Header
    yr_header = "".join(f"{y:>7}" for y in all_years)
    print(f"\n  {'Thresh':>8}  {'Total':>7}  {'Rate':>6}  {yr_header}  {'Zero yrs':>10}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*6}  " + "─" * (7 * len(all_years)) + f"  {'─'*10}")

    pick_stats = {}
    for thresh in THRESHOLDS:
        picks_mask = df_slow_ts["prob"] >= thresh
        total_picks = picks_mask.sum()
        rate = total_picks / len(df_slow_ts) * 100

        yearly_picks = df_slow_ts[picks_mask].groupby("year").size()
        zero_years = []
        yr_vals = []
        for y in all_years:
            cnt = yearly_picks.get(y, 0)
            yr_vals.append(f"{cnt:>7}")
            if cnt < 3:
                zero_years.append(y)

        pick_stats[thresh] = {
            "total": total_picks, "rate": rate,
            "zero_years": zero_years,
            "yearly": {y: yearly_picks.get(y, 0) for y in all_years},
        }

        yr_str = "".join(yr_vals)
        print(f"  {thresh:>8.2f}  {total_picks:>7,}  {rate:>5.1f}%  {yr_str}  {len(zero_years):>10}")

    # Show which years have zero/sparse picks
    print(f"\n  Years with < 3 picks per threshold:")
    for thresh in THRESHOLDS:
        zs = pick_stats[thresh]["zero_years"]
        if zs:
            print(f"    {thresh:.2f}: {', '.join(str(y) for y in zs)} ({len(zs)} years)")
        else:
            print(f"    {thresh:.2f}: none — all years have >= 3 picks")

    # ── 2. Backtest each threshold ───────────────────────────────────────
    print(f"\n{'='*90}")
    print("  BACKTEST RESULTS BY THRESHOLD")
    print(f"{'='*90}")

    results = {}
    for i, thresh in enumerate(THRESHOLDS):
        print(f"\n  [{i+1}/{len(THRESHOLDS)}] Threshold = {thresh:.2f} ...")

        # Build ML Slow strategy at this threshold
        ml_slow_strat = MLSlowStrategy(df_slow, price_data=close)
        ml_slow_strat.PROB_THRESHOLD = thresh
        # Rebuild the signal lookup with new threshold
        ml_slow_strat._signals_by_date = {}
        ml_slow_strat._build_lookup(df_slow)

        # A) ML Slow alone
        slow_vals, slow_trades, d_slow = instrumented_run(
            [ml_slow_strat], SLOT_SLOW_ONLY, all_dates, spy_dict,
            close_aligned, hold_days=HOLD_DAYS, label=f"ML Slow @{thresh}")
        m_slow = calc_metrics(slow_vals, slow_trades, years, f"ML Slow @{thresh}")
        a_slow, _ = calc_alpha_beta(slow_vals, spy_bh.reindex(slow_vals.index, method="ffill"))
        m_slow["alpha"] = a_slow
        m_slow["avg_pos"] = np.mean(d_slow["daily_pos_count"])

        # B) Path B + ML Slow
        four_vals, four_trades, d_four = instrumented_run(
            [ml_med_strat, mom_strat, mr_strat, ml_slow_strat],
            SLOT_PATHB_SLOW, all_dates, spy_dict,
            close_aligned, hold_days=HOLD_DAYS, label=f"PathB+Slow @{thresh}")
        m_four = calc_metrics(four_vals, four_trades, years, f"PathB+Slow @{thresh}")
        a_four, _ = calc_alpha_beta(four_vals, spy_bh.reindex(four_vals.index, method="ffill"))
        m_four["alpha"] = a_four
        m_four["avg_pos"] = np.mean(d_four["daily_pos_count"])

        results[thresh] = {"slow": m_slow, "combined": m_four}
        print(f"    Alone:    CAGR {m_slow['cagr']:+.2%} | Sharpe {m_slow['sharpe']:.3f} | "
              f"Trades {m_slow['n_trades']:>4} | WR {m_slow['win_rate']:.1%} | Alpha {m_slow['alpha']:+.2%}")
        print(f"    Combined: CAGR {m_four['cagr']:+.2%} | Sharpe {m_four['sharpe']:.3f} | "
              f"Trades {m_four['n_trades']:>4} | DD {m_four['max_dd']:.1%}")

    # ── 3. Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  THRESHOLD COMPARISON TABLE")
    print(f"{'='*90}")

    # ML Slow alone table
    print(f"\n  ML Slow Alone:")
    print(f"  {'Thresh':>8} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinRate':>8} {'Alpha':>8} {'AvgPos':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for thresh in THRESHOLDS:
        m = results[thresh]["slow"]
        print(f"  {thresh:>8.2f} {m['cagr']:>+7.2%} {m['sharpe']:>8.3f} {m['max_dd']:>7.1%} "
              f"{m['n_trades']:>8} {m['win_rate']:>7.1%} {m['alpha']:>+7.2%} {m['avg_pos']:>8.2f}")

    # Path B + ML Slow table with deltas
    print(f"\n  Path B + ML Slow (baseline: CAGR {m_pathb['cagr']:+.2%}, Sharpe {m_pathb['sharpe']:.3f}):")
    print(f"  {'Thresh':>8} {'CAGR':>8} {'ΔCAGR':>8} {'Sharpe':>8} {'ΔSharpe':>8} {'MaxDD':>8} {'Trades':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for thresh in THRESHOLDS:
        m = results[thresh]["combined"]
        dcagr = m["cagr"] - m_pathb["cagr"]
        dsharpe = m["sharpe"] - m_pathb["sharpe"]
        print(f"  {thresh:>8.2f} {m['cagr']:>+7.2%} {dcagr:>+7.2%} {m['sharpe']:>8.3f} "
              f"{dsharpe:>+8.3f} {m['max_dd']:>7.1%} {m['n_trades']:>8}")

    # ── 4. Gate checks per threshold ─────────────────────────────────────
    print(f"\n{'='*90}")
    print("  GATE CHECK MATRIX")
    print(f"{'='*90}")

    print(f"\n  {'Thresh':>8} {'Alpha>0':>10} {'Sharpe≥1.655':>14} {'CAGR≥24.08%':>14} {'≥3picks/yr':>12} {'VERDICT':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*14} {'─'*14} {'─'*12} {'─'*10}")

    any_pass = False
    best_thresh = None
    best_sharpe = -999

    for thresh in THRESHOLDS:
        m_s = results[thresh]["slow"]
        m_c = results[thresh]["combined"]
        ps = pick_stats[thresh]

        g1 = m_s["alpha"] > 0
        g2 = m_c["sharpe"] >= m_pathb["sharpe"]
        g3 = m_c["cagr"] >= m_pathb["cagr"]
        g4 = len(ps["zero_years"]) == 0

        g1s = "PASS" if g1 else "FAIL"
        g2s = "PASS" if g2 else "FAIL"
        g3s = "PASS" if g3 else "FAIL"
        g4s = "PASS" if g4 else "FAIL"

        all_pass = g1 and g2 and g3 and g4
        verdict = "PASS" if all_pass else f"{sum([g1,g2,g3,g4])}/4"

        if all_pass:
            any_pass = True
        if m_c["sharpe"] > best_sharpe and g1:
            best_sharpe = m_c["sharpe"]
            best_thresh = thresh

        print(f"  {thresh:>8.2f} {g1s:>10} {g2s:>14} {g3s:>14} {g4s:>12} {verdict:>10}")

    # ── 5. Recommendation ────────────────────────────────────────────────
    print(f"\n{'='*90}")
    if any_pass:
        print(f"  RECOMMENDATION: Deploy ML Slow at threshold {best_thresh:.2f}")
    else:
        print("  RECOMMENDATION: DROP ML Slow — no threshold passes all 4 criteria")
        print()
        # Explain why
        best_alone = max(THRESHOLDS, key=lambda t: results[t]["slow"]["alpha"])
        best_comb = max(THRESHOLDS, key=lambda t: results[t]["combined"]["sharpe"])
        print(f"  Best standalone alpha:   {best_alone:.2f} → alpha {results[best_alone]['slow']['alpha']:+.2%}")
        print(f"  Best combined Sharpe:    {best_comb:.2f} → Sharpe {results[best_comb]['combined']['sharpe']:.3f} (need ≥{m_pathb['sharpe']:.3f})")
        print(f"  Path B baseline:         CAGR {m_pathb['cagr']:+.2%} | Sharpe {m_pathb['sharpe']:.3f}")
        print()

        # Diagnose: is it the model or the threshold?
        all_negative_alpha = all(results[t]["slow"]["alpha"] <= 0 for t in THRESHOLDS)
        all_lower_sharpe = all(results[t]["combined"]["sharpe"] < m_pathb["sharpe"] for t in THRESHOLDS)
        all_sparse = all(len(pick_stats[t]["zero_years"]) > 0 for t in THRESHOLDS)

        if all_negative_alpha:
            print("  DIAGNOSIS: The 30-day prediction model itself lacks alpha.")
            print("  Lowering the threshold does not fix a fundamentally weak model.")
            print("  The model's OOS AUC (0.576) is barely above random (0.50).")
        if all_lower_sharpe:
            print("  DIAGNOSIS: ML Slow dilutes Path B at EVERY threshold tested.")
            print("  Adding it never improves risk-adjusted returns.")
        if all_sparse:
            print("  DIAGNOSIS: Pick distribution is sparse at all thresholds.")
            print("  The calibrated probabilities cluster in narrow, unpredictable bands.")
    print(f"{'='*90}")

    print(f"\n  Runtime: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
