"""
ML Slow Strategy Backtest & Comparison
=======================================
Tests ML Slow (30-day prediction model) in isolation and combined with
the existing 3-strategy system (Path B).

Configurations:
  A. ML Slow alone (5 max positions)
  B. Current Path B baseline (ML Medium + Momentum + MR, 10 max)
  C. Path B + ML Slow (ML Med 2 + Mom 4 + MR 2 + ML Slow 1 + flex 2 = 11 max)

Checks 5 gate conditions before recommending deployment.

Run with:
    python3 backtest_ml_slow.py
"""

import warnings
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from unified_backtester import (
    INITIAL_CASH, HOLD_DAYS,
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy, MLSlowStrategy,
    PortfolioManager, SlotConfig,
    SLOT_ML_ONLY, SLOT_MOM_ONLY, SLOT_ML_MOM_MR,
    SLOT_ML_SLOW_ONLY, SLOT_ML_MOM_MR_SLOW,
)
from backtest_ml import (
    load_predictions, calc_metrics, calc_alpha_beta,
)
from diagnose_combined import instrumented_run

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"

ML_THRESHOLD = 0.55


def load_slow_predictions():
    """Load ML Slow predictions from predictions_slow.parquet."""
    pred_file = DATA_DIR / "predictions_slow.parquet"
    if not pred_file.exists():
        import sys
        sys.exit(f"ERROR: {pred_file} not found — run train_model_slow.py first.")
    df = pd.read_parquet(pred_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"])
    print(f"  ML Slow predictions: {len(df):,} rows  |  {df['symbol'].nunique()} symbols  |  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    n_picks = (df["prob"] >= ML_THRESHOLD).sum()
    print(f"  Picks (prob > {ML_THRESHOLD}): {n_picks:,} ({n_picks/len(df)*100:.1f}%)")
    return df


def fetch_ohlcv(symbols, start, end):
    """Download close prices and volume for all symbols."""
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
    print("=" * 80)
    print("  ML SLOW STRATEGY BACKTEST & COMPARISON")
    print("  Configs: ML Slow alone, Path B baseline, Path B + ML Slow")
    print("=" * 80)

    # ── 1. Load predictions ──────────────────────────────────────────────
    print("\n  Loading ML Medium predictions ...")
    df_med = load_predictions()
    print("\n  Loading ML Slow predictions ...")
    df_slow = load_slow_predictions()

    # Use the intersection of date ranges
    med_dates = set(df_med["date"].unique())
    slow_dates = set(df_slow["date"].unique())
    common_dates = sorted(med_dates & slow_dates)
    print(f"\n  Common date range: {len(common_dates)} trading days")
    print(f"  {common_dates[0].date()} → {common_dates[-1].date()}")

    all_dates = common_dates
    universe_syms = sorted(set(df_med["symbol"].unique()) | set(df_slow["symbol"].unique()))
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    # ── 2. Fetch OHLCV data ──────────────────────────────────────────────
    print("\n  Fetching price & volume data from Yahoo Finance ...")
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    close, volume = fetch_ohlcv(universe_syms,
                                 start.strftime("%Y-%m-%d"),
                                 end.strftime("%Y-%m-%d"))
    print(f"  Price data: {len(close)} days, {len(close.columns)} symbols")

    # Align to simulation dates
    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    volume_aligned = volume.reindex(sim_index, method="ffill") if volume is not None else None

    # SPY benchmark
    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close_aligned["SPY"].to_dict()
    m_spy = calc_metrics(spy_bh, [], years, "SPY B&H")

    # ── 3. Build strategies ──────────────────────────────────────────────
    print("\n  Building strategies ...")
    ml_med_strat = MLMediumStrategy(df_med, threshold=ML_THRESHOLD)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)
    ml_slow_strat = MLSlowStrategy(df_slow, price_data=close)

    # ── 4. Run backtests ─────────────────────────────────────────────────
    configs = {}
    diags = {}

    # 4a. ML Slow alone (5 max positions)
    print("\n  [1/3] Running ML SLOW ALONE (5 max positions) ...")
    slow_vals, slow_trades, d_slow = instrumented_run(
        [ml_slow_strat], SLOT_ML_SLOW_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML Slow")
    m_slow = calc_metrics(slow_vals, slow_trades, years, "ML Slow")
    a_slow, _ = calc_alpha_beta(slow_vals, spy_bh.reindex(slow_vals.index, method="ffill"))
    m_slow["alpha"] = a_slow
    m_slow["avg_pos"] = np.mean(d_slow["daily_pos_count"])
    configs["ML Slow"] = (slow_vals, slow_trades, m_slow)
    diags["ML Slow"] = d_slow
    print(f"    CAGR {m_slow['cagr']:+.2%} | Sharpe {m_slow['sharpe']:.3f} | "
          f"Trades {m_slow['n_trades']:,} | WR {m_slow['win_rate']:.1%}")

    # 4b. Path B baseline (ML Medium + Momentum + MR)
    print("\n  [2/3] Running PATH B BASELINE (ML Med + Mom + MR, 10 max) ...")
    pathb_vals, pathb_trades, d_pathb = instrumented_run(
        [ml_med_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B")
    m_pathb = calc_metrics(pathb_vals, pathb_trades, years, "Path B")
    a_pathb, _ = calc_alpha_beta(pathb_vals, spy_bh.reindex(pathb_vals.index, method="ffill"))
    m_pathb["alpha"] = a_pathb
    m_pathb["avg_pos"] = np.mean(d_pathb["daily_pos_count"])
    configs["Path B"] = (pathb_vals, pathb_trades, m_pathb)
    diags["Path B"] = d_pathb
    print(f"    CAGR {m_pathb['cagr']:+.2%} | Sharpe {m_pathb['sharpe']:.3f} | "
          f"Trades {m_pathb['n_trades']:,} | WR {m_pathb['win_rate']:.1%}")

    # 4c. Path B + ML Slow (4-strategy system)
    print("\n  [3/3] Running PATH B + ML SLOW (4-strategy, 11 max) ...")
    four_vals, four_trades, d_four = instrumented_run(
        [ml_med_strat, mom_strat, mr_strat, ml_slow_strat],
        SLOT_ML_MOM_MR_SLOW, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B + ML Slow")
    m_four = calc_metrics(four_vals, four_trades, years, "Path B + ML Slow")
    a_four, _ = calc_alpha_beta(four_vals, spy_bh.reindex(four_vals.index, method="ffill"))
    m_four["alpha"] = a_four
    m_four["avg_pos"] = np.mean(d_four["daily_pos_count"])
    configs["Path B + ML Slow"] = (four_vals, four_trades, m_four)
    diags["Path B + ML Slow"] = d_four
    print(f"    CAGR {m_four['cagr']:+.2%} | Sharpe {m_four['sharpe']:.3f} | "
          f"Trades {m_four['n_trades']:,} | WR {m_four['win_rate']:.1%}")

    # ── 5. Comparison table ──────────────────────────────────────────────
    labels = ["ML Slow", "Path B", "Path B + ML Slow", "SPY B&H"]
    all_metrics = {
        "ML Slow": m_slow,
        "Path B": m_pathb,
        "Path B + ML Slow": m_four,
        "SPY B&H": m_spy,
    }
    m_spy["alpha"] = 0.0
    m_spy["avg_pos"] = 0.0

    metric_rows = [
        ("CAGR",          "cagr",          lambda v: f"{v:+.2%}"),
        ("Sharpe",        "sharpe",        lambda v: f"{v:.3f}"),
        ("Sortino",       "sortino",       lambda v: f"{v:.3f}"),
        ("Max Drawdown",  "max_dd",        lambda v: f"{v:.1%}"),
        ("Final Value",   "final_value",   lambda v: f"${v:,.0f}"),
        ("Alpha vs SPY",  "alpha",         lambda v: f"{v:+.2%}"),
        ("Total Trades",  "n_trades",      lambda v: f"{v:,}"),
        ("Win Rate",      "win_rate",      lambda v: f"{v:.1%}" if not np.isnan(v) else "—"),
        ("Profit Factor", "profit_factor",
         lambda v: f"{v:.2f}" if np.isfinite(v) else "inf" if not np.isnan(v) else "—"),
        ("Avg Trade Ret", "avg_trade_ret", lambda v: f"{v:+.2%}" if not np.isnan(v) else "—"),
        ("Avg Pos/Day",   "avg_pos",       lambda v: f"{v:.2f}" if v > 0 else "—"),
    ]

    col0 = 18
    col = 20
    print(f"\n{'='*80}")
    print("  FULL COMPARISON TABLE")
    print(f"{'='*80}")
    header = f"  {'Metric':<{col0}}" + "".join(f"{l:<{col}}" for l in labels)
    print(header)
    print(f"  {'─'*(col0-1)}" + "".join(f" {'─'*(col-2)} " for _ in labels))

    for name, key, fmt in metric_rows:
        row = f"  {name:<{col0}}"
        for label in labels:
            m = all_metrics[label]
            val = m.get(key, np.nan)
            row += f"{fmt(val):<{col}}"
        print(row)

    # ── 6. Standalone AUC report ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  ML SLOW MODEL QUALITY")
    print(f"{'='*80}")

    from sklearn.metrics import roc_auc_score
    slow_valid = df_slow.dropna(subset=["target_slow", "prob"])
    if len(slow_valid) > 0:
        auc = roc_auc_score(slow_valid["target_slow"].astype(int), slow_valid["prob"])
        print(f"  OOS AUC-ROC: {auc:.4f}")
        n_picks = (slow_valid["prob"] >= ML_THRESHOLD).sum()
        n_total = len(slow_valid)
        print(f"  Total predictions: {n_total:,}")
        print(f"  Picks (prob > {ML_THRESHOLD}): {n_picks:,} ({n_picks/n_total*100:.1f}%)")

        # Per-year pick distribution
        slow_valid_ts = slow_valid.copy()
        slow_valid_ts["year"] = slow_valid_ts["date"].dt.year
        yearly = slow_valid_ts.groupby("year").apply(
            lambda g: pd.Series({
                "total": len(g),
                "picks": (g["prob"] >= ML_THRESHOLD).sum(),
                "pick_pct": (g["prob"] >= ML_THRESHOLD).mean() * 100,
            })
        )
        print(f"\n  Year-by-year pick rate:")
        for year, row in yearly.iterrows():
            print(f"    {int(year)}: {int(row['picks']):>5} picks / {int(row['total']):>6} total ({row['pick_pct']:.1f}%)")
    else:
        auc = np.nan
        print("  No valid predictions with target_slow — cannot compute AUC")

    # ── 7. Correlation analysis ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  CORRELATION ANALYSIS")
    print(f"{'='*80}")

    slow_daily = slow_vals.pct_change().dropna()
    pathb_daily = pathb_vals.pct_change().dropna()
    four_daily = four_vals.pct_change().dropna()
    spy_daily = spy_bh.pct_change().dropna()

    # Run individual strategies for pairwise correlations
    print("\n  Running ML-only, Mom-only, MR-only for correlation analysis ...")
    ml_only_vals, _, _ = instrumented_run(
        [ml_med_strat], SLOT_ML_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML Only")

    from unified_backtester import SLOT_MR_ONLY
    mr_only_vals, _, _ = instrumented_run(
        [mr_strat], SLOT_MR_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="MR Only")

    mom_only_vals, _, _ = instrumented_run(
        [mom_strat], SLOT_MOM_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=60, label="Mom Only")

    ml_daily = ml_only_vals.pct_change().dropna()
    mom_daily = mom_only_vals.pct_change().dropna()
    mr_daily = mr_only_vals.pct_change().dropna()

    combined = pd.concat([
        slow_daily, ml_daily, mom_daily, mr_daily, spy_daily
    ], axis=1, join="inner")
    combined.columns = ["ML Slow", "ML Medium", "Momentum", "Mean Rev", "SPY"]

    corr = combined.corr()
    col_names = list(combined.columns)
    print(f"\n  Daily Return Correlations:")
    print(f"  {'':18}" + "".join(f"{n:>14}" for n in col_names))
    for name in col_names:
        row = f"  {name:<18}"
        for name2 in col_names:
            row += f"{corr.loc[name, name2]:>14.3f}"
        print(row)

    slow_ml_corr = corr.loc["ML Slow", "ML Medium"]
    slow_mom_corr = corr.loc["ML Slow", "Momentum"]
    slow_mr_corr = corr.loc["ML Slow", "Mean Rev"]
    print(f"\n  ML Slow ↔ ML Medium correlation:  {slow_ml_corr:.3f}")
    print(f"  ML Slow ↔ Momentum correlation:   {slow_mom_corr:.3f}")
    print(f"  ML Slow ↔ Mean Rev correlation:    {slow_mr_corr:.3f}")

    # ── 8. Position & activity analysis ──────────────────────────────────
    print(f"\n{'='*80}")
    print("  POSITION & ACTIVITY ANALYSIS")
    print(f"{'='*80}")

    # 4-strategy system breakdown
    strat_pos_sums = defaultdict(float)
    for day_dict in d_four["daily_pos_by_strat"]:
        for strat, cnt in day_dict.items():
            strat_pos_sums[strat] += cnt
    n_dates_count = len(all_dates)
    print(f"\n  4-Strategy system — avg positions per day by strategy:")
    for strat in sorted(strat_pos_sums.keys()):
        avg = strat_pos_sums[strat] / n_dates_count
        print(f"    {strat:<18} {avg:.2f}")
    print(f"    {'TOTAL':<18} {m_four['avg_pos']:.2f}")

    # Signal acceptance rates
    print(f"\n  4-Strategy system — signal acceptance:")
    gen = d_four["signals_generated"]
    acc = d_four["signals_accepted"]
    for strat in sorted(gen.keys()):
        g = gen[strat]
        a = acc.get(strat, 0)
        print(f"    {strat:<18} {a:>5} / {g:>6} accepted ({a/g:.1%})" if g > 0
              else f"    {strat:<18}     0 /      0 accepted")

    # ML Slow specific analysis
    print(f"\n  ML Slow exit reason breakdown:")
    slow_recs = d_slow.get("trade_records", [])
    if slow_recs:
        exit_counts = Counter()
        exit_pnl = defaultdict(float)
        exit_rets = defaultdict(list)
        for r in slow_recs:
            reason = r.get("exit_reason", "unknown")
            exit_counts[reason] += 1
            exit_pnl[reason] += r["pnl"]
            exit_rets[reason].append(r["return"])

        print(f"  {'Reason':<22} {'Count':>7} {'Win%':>7} {'Avg Ret':>9} {'Total P&L':>12}")
        print(f"  {'─'*21} {'─'*6} {'─'*6} {'─'*8} {'─'*11}")
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            avg_r = np.mean(exit_rets[reason])
            wins = sum(1 for r in exit_rets[reason] if r > 0)
            wr = wins / count if count > 0 else 0
            print(f"  {reason:<22} {count:>7} {wr:>6.1%} {avg_r:>+8.2%} "
                  f"${exit_pnl[reason]:>11,.0f}")

        if slow_recs:
            avg_hold = np.mean([r["days_held"] for r in slow_recs])
            print(f"\n  Avg holding period: {avg_hold:.1f} days")

    # ── 9. Gate condition checks ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  GATE CONDITION CHECKS (5 gates)")
    print(f"{'='*80}")

    gates = []

    # Gate 1: ML Slow standalone AUC > 0.55
    g1 = auc > 0.55
    gates.append(("ML Slow AUC > 0.55", g1,
                  f"AUC = {auc:.4f}"))

    # Gate 2: ML Slow alone has positive alpha vs SPY
    g2 = m_slow["alpha"] > 0
    gates.append(("ML Slow alpha > 0", g2,
                  f"alpha = {m_slow['alpha']:+.2%}"))

    # Gate 3: 4-strategy Sharpe > Path B Sharpe (1.655)
    pathb_sharpe = m_pathb["sharpe"]
    g3 = m_four["sharpe"] > pathb_sharpe
    gates.append(("4-strat Sharpe > Path B Sharpe", g3,
                  f"{m_four['sharpe']:.3f} vs {pathb_sharpe:.3f} (Path B)"))

    # Gate 4: 4-strategy CAGR >= Path B CAGR (24.08%)
    pathb_cagr = m_pathb["cagr"]
    g4 = m_four["cagr"] >= pathb_cagr
    gates.append(("4-strat CAGR >= Path B CAGR", g4,
                  f"{m_four['cagr']:+.2%} vs {pathb_cagr:+.2%} (Path B)"))

    # Gate 5: 4-strategy max DD within 3pp of Path B max DD
    pathb_dd = m_pathb["max_dd"]
    dd_limit = pathb_dd - 0.03
    g5 = m_four["max_dd"] >= dd_limit
    gates.append(("4-strat max DD within 3pp of Path B", g5,
                  f"max DD = {m_four['max_dd']:.1%} vs limit {dd_limit:.1%} (Path B DD = {pathb_dd:.1%})"))

    all_pass = True
    for desc, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")
        print(f"         {detail}")

    # ── 10. Recommendation ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    if all_pass:
        print("  VERDICT: ALL 5 GATES PASS — ML Slow approved for 4-strategy system")
        print("  RECOMMENDATION: Deploy Path B + ML Slow")
    else:
        failed = [desc for desc, passed, _ in gates if not passed]
        print(f"  VERDICT: {len(failed)} gate(s) FAILED — ML Slow NOT recommended for deployment")
        for f in failed:
            print(f"    - {f}")

        # Provide nuanced recommendation
        if m_slow["alpha"] > 0 and auc > 0.55:
            print("\n  NOTE: ML Slow shows signal quality (AUC > 0.55, positive alpha)")
            print("  but may not improve the combined system enough to pass all gates.")
            print("  Consider: adjusting slot allocation, tuning confidence threshold,")
            print("  or retraining with different target definition.")
        elif auc <= 0.55:
            print("\n  NOTE: ML Slow AUC <= 0.55 — the 30-day prediction model lacks")
            print("  sufficient signal. Consider retraining with different features")
            print("  or target definition before retesting.")
    print(f"{'='*80}")

    print(f"\n  Runtime: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
