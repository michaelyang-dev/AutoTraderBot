"""
Mean Reversion Strategy Backtest & Comparison
==============================================
Runs four configurations and compares against SPY B&H:

  1. Mean Reversion alone   (5 max positions)
  2. ML + Momentum baseline (8 max positions)
  3. ML + Mean Reversion    (2+2+2 flex = 6 max)
  4. ML + Mom + MR (3-strat) (2+4+2+2 flex = 10 max)

Prints full metrics table, correlation analysis, overlap frequency,
avg position count per day, and 5 gate condition checks.

Run with:
    python3 backtest_mean_reversion.py
"""

import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from unified_backtester import (
    INITIAL_CASH, HOLD_DAYS,
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    PortfolioManager,
    SLOT_ML_ONLY, SLOT_MOM_ONLY, SLOT_ML_MOM,
    SLOT_MR_ONLY, SLOT_ML_MR, SLOT_ML_MOM_MR,
)
from backtest_utils import load_predictions, calc_metrics, calc_alpha_beta
from diagnose_combined import instrumented_run

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"

ML_THRESHOLD = 0.55


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
    print("  MEAN REVERSION STRATEGY BACKTEST & COMPARISON")
    print("  Configs: MR alone, ML+Mom baseline, ML+MR, ML+Mom+MR (3-strat)")
    print("=" * 80)

    # ── 1. Load ML predictions ───────────────────────────────────────────
    print("\n  Loading ML predictions ...")
    df = load_predictions()
    all_dates = sorted(df["date"].unique().tolist())
    universe_syms = sorted(df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    n_dates = len(all_dates)

    print(f"  {n_dates} trading days | {years:.1f} years | {len(universe_syms)} symbols")

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
    ml_strat = MLMediumStrategy(df, threshold=ML_THRESHOLD)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ── 4. Run backtests ─────────────────────────────────────────────────
    configs = {}
    diags = {}

    # 4a. Mean Reversion alone
    print("\n  [1/4] Running MEAN REVERSION ALONE (5 max positions) ...")
    mr_vals, mr_trades, d_mr = instrumented_run(
        [mr_strat], SLOT_MR_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Mean Reversion")
    m_mr = calc_metrics(mr_vals, mr_trades, years, "Mean Reversion")
    a_mr, _ = calc_alpha_beta(mr_vals, spy_bh.reindex(mr_vals.index, method="ffill"))
    m_mr["alpha"] = a_mr
    m_mr["avg_pos"] = np.mean(d_mr["daily_pos_count"])
    configs["Mean Reversion"] = (mr_vals, mr_trades, m_mr)
    diags["Mean Reversion"] = d_mr
    print(f"    CAGR {m_mr['cagr']:+.2%} | Sharpe {m_mr['sharpe']:.3f} | "
          f"Trades {m_mr['n_trades']:,} | WR {m_mr['win_rate']:.1%}")

    # 4b. ML + Momentum baseline
    print("\n  [2/4] Running ML + MOMENTUM BASELINE (8 max positions) ...")
    mlmom_vals, mlmom_trades, d_mlmom = instrumented_run(
        [ml_strat, mom_strat], SLOT_ML_MOM, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML + Momentum")
    m_mlmom = calc_metrics(mlmom_vals, mlmom_trades, years, "ML + Momentum")
    a_mlmom, _ = calc_alpha_beta(mlmom_vals, spy_bh.reindex(mlmom_vals.index, method="ffill"))
    m_mlmom["alpha"] = a_mlmom
    m_mlmom["avg_pos"] = np.mean(d_mlmom["daily_pos_count"])
    configs["ML + Momentum"] = (mlmom_vals, mlmom_trades, m_mlmom)
    diags["ML + Momentum"] = d_mlmom
    print(f"    CAGR {m_mlmom['cagr']:+.2%} | Sharpe {m_mlmom['sharpe']:.3f} | "
          f"Trades {m_mlmom['n_trades']:,} | WR {m_mlmom['win_rate']:.1%}")

    # 4c. ML + Mean Reversion
    print("\n  [3/4] Running ML + MEAN REVERSION (6 max positions) ...")
    mlmr_vals, mlmr_trades, d_mlmr = instrumented_run(
        [ml_strat, mr_strat], SLOT_ML_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML + MR")
    m_mlmr = calc_metrics(mlmr_vals, mlmr_trades, years, "ML + MR")
    a_mlmr, _ = calc_alpha_beta(mlmr_vals, spy_bh.reindex(mlmr_vals.index, method="ffill"))
    m_mlmr["alpha"] = a_mlmr
    m_mlmr["avg_pos"] = np.mean(d_mlmr["daily_pos_count"])
    configs["ML + MR"] = (mlmr_vals, mlmr_trades, m_mlmr)
    diags["ML + MR"] = d_mlmr
    print(f"    CAGR {m_mlmr['cagr']:+.2%} | Sharpe {m_mlmr['sharpe']:.3f} | "
          f"Trades {m_mlmr['n_trades']:,} | WR {m_mlmr['win_rate']:.1%}")

    # 4d. ML + Momentum + Mean Reversion (3-strategy)
    print("\n  [4/4] Running ML + MOMENTUM + MEAN REVERSION (10 max positions) ...")
    three_vals, three_trades, d_three = instrumented_run(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML + Mom + MR")
    m_three = calc_metrics(three_vals, three_trades, years, "ML + Mom + MR")
    a_three, _ = calc_alpha_beta(three_vals, spy_bh.reindex(three_vals.index, method="ffill"))
    m_three["alpha"] = a_three
    m_three["avg_pos"] = np.mean(d_three["daily_pos_count"])
    configs["ML + Mom + MR"] = (three_vals, three_trades, m_three)
    diags["ML + Mom + MR"] = d_three
    print(f"    CAGR {m_three['cagr']:+.2%} | Sharpe {m_three['sharpe']:.3f} | "
          f"Trades {m_three['n_trades']:,} | WR {m_three['win_rate']:.1%}")

    # ── 5. Comparison table ──────────────────────────────────────────────
    labels = ["Mean Reversion", "ML + Momentum", "ML + MR", "ML + Mom + MR", "SPY B&H"]
    all_metrics = {
        "Mean Reversion": m_mr,
        "ML + Momentum": m_mlmom,
        "ML + MR": m_mlmr,
        "ML + Mom + MR": m_three,
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
    col = 18
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

    # ── 6. Correlation analysis ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  CORRELATION ANALYSIS")
    print(f"{'='*80}")

    mr_daily = mr_vals.pct_change().dropna()
    mlmom_daily = mlmom_vals.pct_change().dropna()
    three_daily = three_vals.pct_change().dropna()
    spy_daily = spy_bh.pct_change().dropna()

    # Also get ML-only and Mom-only for pairwise correlations
    # Run quick ML-only and Mom-only for correlation
    print("\n  Running ML-only and Mom-only for correlation analysis ...")
    ml_only_vals, _, _ = instrumented_run(
        [ml_strat], SLOT_ML_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML Only")
    mom_only_vals, _, _ = instrumented_run(
        [mom_strat], SLOT_MOM_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=60, label="Mom Only")

    ml_daily = ml_only_vals.pct_change().dropna()
    mom_daily = mom_only_vals.pct_change().dropna()

    combined = pd.concat([
        mr_daily, ml_daily, mom_daily, three_daily, spy_daily
    ], axis=1, join="inner")
    combined.columns = ["Mean Rev", "ML Medium", "Momentum", "3-Strat", "SPY"]

    corr = combined.corr()
    col_names = list(combined.columns)
    print(f"\n  Daily Return Correlations:")
    print(f"  {'':18}" + "".join(f"{n:>14}" for n in col_names))
    for name in col_names:
        row = f"  {name:<18}"
        for name2 in col_names:
            row += f"{corr.loc[name, name2]:>14.3f}"
        print(row)

    mr_ml_corr = corr.loc["Mean Rev", "ML Medium"]
    mr_mom_corr = corr.loc["Mean Rev", "Momentum"]
    print(f"\n  MR ↔ ML correlation:       {mr_ml_corr:.3f}")
    print(f"  MR ↔ Momentum correlation: {mr_mom_corr:.3f}")

    # ── 7. Position & activity analysis ──────────────────────────────────
    print(f"\n{'='*80}")
    print("  POSITION & ACTIVITY ANALYSIS")
    print(f"{'='*80}")

    # Avg position count by strategy in 3-strat system
    from collections import defaultdict
    strat_pos_sums = defaultdict(float)
    for day_dict in d_three["daily_pos_by_strat"]:
        for strat, cnt in day_dict.items():
            strat_pos_sums[strat] += cnt
    print(f"\n  3-Strategy system — avg positions per day by strategy:")
    for strat in sorted(strat_pos_sums.keys()):
        avg = strat_pos_sums[strat] / n_dates
        print(f"    {strat:<18} {avg:.2f}")
    print(f"    {'TOTAL':<18} {m_three['avg_pos']:.2f}")

    # Signal acceptance rates
    print(f"\n  3-Strategy system — signal acceptance:")
    gen = d_three["signals_generated"]
    acc = d_three["signals_accepted"]
    for strat in sorted(gen.keys()):
        g = gen[strat]
        a = acc.get(strat, 0)
        print(f"    {strat:<18} {a:>5} / {g:>6} accepted ({a/g:.1%})" if g > 0
              else f"    {strat:<18}     0 /      0 accepted")

    # Block reasons
    br = d_three["block_reasons"]
    if br:
        print(f"\n  3-Strategy system — block reasons:")
        for reason, count in sorted(br.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25} {count:>6}")

    # Monthly activity
    print(f"\n  Monthly Trade Counts:")
    for label_name, (vals, trades_list, m) in configs.items():
        print(f"    {label_name}: {m['n_trades']:,} total "
              f"({m['n_trades']/years/12:.1f} trades/month)")

    # ── 8. Mean Reversion specific analysis ──────────────────────────────
    print(f"\n{'='*80}")
    print("  MEAN REVERSION STRATEGY ANALYSIS")
    print(f"{'='*80}")

    # Exit reason breakdown from MR-only run
    mr_recs = d_mr.get("trade_records", [])
    if mr_recs:
        from collections import Counter
        exit_counts = Counter()
        exit_pnl = defaultdict(float)
        exit_rets = defaultdict(list)
        for r in mr_recs:
            reason = r.get("exit_reason", "unknown")
            exit_counts[reason] += 1
            exit_pnl[reason] += r["pnl"]
            exit_rets[reason].append(r["return"])

        print(f"\n  Exit reason breakdown (MR alone, {len(mr_recs)} trades):")
        print(f"  {'Reason':<22} {'Count':>7} {'Win%':>7} {'Avg Ret':>9} {'Total P&L':>12}")
        print(f"  {'─'*21} {'─'*6} {'─'*6} {'─'*8} {'─'*11}")
        for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
            avg_r = np.mean(exit_rets[reason])
            wins = sum(1 for r in exit_rets[reason] if r > 0)
            wr = wins / count if count > 0 else 0
            print(f"  {reason:<22} {count:>7} {wr:>6.1%} {avg_r:>+8.2%} "
                  f"${exit_pnl[reason]:>11,.0f}")

    # Average holding period
    if mr_recs:
        avg_hold = np.mean([r["days_held"] for r in mr_recs])
        print(f"\n  Avg holding period: {avg_hold:.1f} days")

    # ── 9. Gate condition checks ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  GATE CONDITION CHECKS (5 gates)")
    print(f"{'='*80}")

    gates = []

    # Gate 1: MR positive alpha (standalone)
    g1 = m_mr["alpha"] > 0
    gates.append(("MR alpha > 0", g1,
                  f"alpha = {m_mr['alpha']:+.2%}"))

    # Gate 2: 3-strat Sharpe > ML+Mom Sharpe
    g2 = m_three["sharpe"] > m_mlmom["sharpe"]
    gates.append(("3-strat Sharpe > ML+Mom Sharpe", g2,
                  f"{m_three['sharpe']:.3f} vs {m_mlmom['sharpe']:.3f}"))

    # Gate 3: MR correlation < 0.5 with both ML and Momentum
    g3 = mr_ml_corr < 0.5 and mr_mom_corr < 0.5
    gates.append(("MR corr < 0.5 with ML & Mom", g3,
                  f"ML corr = {mr_ml_corr:.3f}, Mom corr = {mr_mom_corr:.3f}"))

    # Gate 4: 3-strat CAGR >= 22.50%
    # NOTE: This is aspirational — adjust based on actual ML+Mom baseline
    g4 = m_three["cagr"] >= 0.2250
    gates.append(("3-strat CAGR >= 22.50%", g4,
                  f"CAGR = {m_three['cagr']:+.2%}"))

    # Gate 5: 3-strat max DD within 3pp of -16.4%
    target_dd = -0.164
    dd_limit = target_dd - 0.03  # -19.4%
    g5 = m_three["max_dd"] >= dd_limit
    gates.append(("3-strat max DD within 3pp of -16.4%", g5,
                  f"max DD = {m_three['max_dd']:.1%} (limit {dd_limit:.1%})"))

    all_pass = True
    for desc, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {desc}")
        print(f"         {detail}")

    print(f"\n{'='*80}")
    if all_pass:
        print("  VERDICT: ALL 5 GATES PASS — Mean Reversion approved for 3-strategy system")
    else:
        failed = [desc for desc, passed, _ in gates if not passed]
        print(f"  VERDICT: {len(failed)} gate(s) FAILED — review before deploying")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*80}")

    print(f"\n  Runtime: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
