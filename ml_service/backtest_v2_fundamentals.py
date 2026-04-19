"""
Backtest: ML Medium v2 (with Fundamentals) vs Baseline
=======================================================
Compares Path B with the CURRENT model vs Path B with the NEW
fundamentals-enhanced model.

  A) Path B with current model (baseline)
  B) Path B with v2 fundamentals model

Recommendation criteria:
  Deploy if: Sharpe +0.05, CAGR doesn't drop >1%, DD doesn't worsen >2pp
  Consider: improvement in 2 of 3 (CAGR, Sharpe, DD)
  Reject: multiple metrics worsen significantly

Run with:
    python3 backtest_v2_fundamentals.py
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
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    SlotConfig,
    SLOT_ML_MOM_MR,
)
from backtest_ml import load_predictions, calc_metrics, calc_alpha_beta
from diagnose_combined import instrumented_run

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"

# Baseline targets from Path B
BASELINE_CAGR   = 0.2408
BASELINE_SHARPE = 1.655
BASELINE_DD     = -0.263


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
    print("  BACKTEST: ML Medium v2 (Fundamentals) vs Baseline")
    print("=" * 90)

    # ── Load predictions ─────────────────────────────────────────────────
    print("\n  Loading CURRENT model predictions ...")
    df_current = load_predictions()  # uses predictions.parquet (current calibrated)

    # Load v2 fundamentals predictions
    v2_pred_file = DATA_DIR / "predictions_v2_fundamentals.parquet"
    if not v2_pred_file.exists():
        print(f"  ERROR: {v2_pred_file} not found — run train_model_v2_fundamentals.py first")
        return
    df_v2 = pd.read_parquet(v2_pred_file)
    df_v2["date"] = pd.to_datetime(df_v2["date"])
    df_v2 = df_v2.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])
    print(f"  V2 fundamentals predictions: {len(df_v2):,} rows  |  "
          f"{df_v2['symbol'].nunique()} symbols  |  "
          f"{df_v2['date'].min().date()} → {df_v2['date'].max().date()}")

    # OOS AUC comparison
    auc_current = roc_auc_score(df_current["target"].values, df_current["prob"].values)
    auc_v2 = roc_auc_score(df_v2["target"].values, df_v2["prob"].values)
    print(f"\n  OOS AUC — Current: {auc_current:.4f}  |  V2 Fundamentals: {auc_v2:.4f}  "
          f"|  Δ {auc_v2 - auc_current:+.4f}")

    # Use common dates for fair comparison
    common_dates = sorted(set(df_current["date"].unique()) & set(df_v2["date"].unique()))
    all_dates = common_dates
    universe_syms = sorted(
        set(df_current["symbol"].unique()) | set(df_v2["symbol"].unique())
    )
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    print(f"  Common period: {len(all_dates)} trading days | {years:.1f} years")

    # ── Fetch prices ─────────────────────────────────────────────────────
    print("\n  Fetching price & volume data ...")
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    close, volume = fetch_ohlcv(
        universe_syms, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    volume_aligned = volume.reindex(sim_index, method="ffill") if volume is not None else None

    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close_aligned["SPY"].to_dict()

    # ── Build strategies ─────────────────────────────────────────────────
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ── A) Path B with CURRENT model ─────────────────────────────────────
    print("\n  [A] Path B with CURRENT model ...")
    ml_current = MLMediumStrategy(df_current, threshold=0.55)
    vals_a, trades_a, d_a = instrumented_run(
        [ml_current, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="PathB Current")
    m_a = calc_metrics(vals_a, trades_a, years, "PathB Current")
    alpha_a, beta_a = calc_alpha_beta(vals_a, spy_bh.reindex(vals_a.index, method="ffill"))
    m_a["alpha"] = alpha_a
    m_a["beta"] = beta_a
    print(f"    CAGR {m_a['cagr']:+.2%} | Sharpe {m_a['sharpe']:.3f} | "
          f"MaxDD {m_a['max_dd']:.1%} | Alpha {m_a['alpha']:+.2%} | Trades {m_a['n_trades']}")

    # ── B) Path B with V2 FUNDAMENTALS model ─────────────────────────────
    print("\n  [B] Path B with V2 FUNDAMENTALS model ...")
    ml_v2 = MLMediumStrategy(df_v2, threshold=0.55)
    vals_b, trades_b, d_b = instrumented_run(
        [ml_v2, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="PathB V2 Fund")
    m_b = calc_metrics(vals_b, trades_b, years, "PathB V2 Fund")
    alpha_b, beta_b = calc_alpha_beta(vals_b, spy_bh.reindex(vals_b.index, method="ffill"))
    m_b["alpha"] = alpha_b
    m_b["beta"] = beta_b
    print(f"    CAGR {m_b['cagr']:+.2%} | Sharpe {m_b['sharpe']:.3f} | "
          f"MaxDD {m_b['max_dd']:.1%} | Alpha {m_b['alpha']:+.2%} | Trades {m_b['n_trades']}")

    # ── SPY benchmark ────────────────────────────────────────────────────
    m_spy = calc_metrics(spy_bh, [], years, "SPY B&H")

    # ── Comparison Table ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  PERFORMANCE COMPARISON")
    print(f"{'='*90}")

    header = (f"  {'Config':<22} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} "
              f"{'MaxDD':>8} {'WinRate':>8} {'PrftF':>8} {'Trades':>8} {'Alpha':>8}")
    print(header)
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for label, m in [("SPY B&H", m_spy), ("PathB Current", m_a), ("PathB V2 Fund", m_b)]:
        alpha_str = f"{m.get('alpha', 0):+.2%}" if 'alpha' in m else "N/A"
        pf = m['profit_factor']
        pf_str = f"{pf:.2f}" if np.isfinite(pf) else "inf"
        print(f"  {label:<22} {m['cagr']:>+7.2%} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>7.1%} {m['win_rate']:>7.1%} {pf_str:>8} "
              f"{m['n_trades']:>8} {alpha_str:>8}")

    # ── Delta table ──────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  DELTA: V2 Fundamentals vs Current Baseline")
    print(f"{'─'*90}")

    d_cagr = m_b["cagr"] - m_a["cagr"]
    d_sharpe = m_b["sharpe"] - m_a["sharpe"]
    d_sortino = m_b["sortino"] - m_a["sortino"]
    d_dd = m_b["max_dd"] - m_a["max_dd"]
    d_wr = m_b["win_rate"] - m_a["win_rate"]
    d_alpha = m_b["alpha"] - m_a["alpha"]
    d_trades = m_b["n_trades"] - m_a["n_trades"]

    deltas = [
        ("CAGR",         f"{m_a['cagr']:+.2%}",    f"{m_b['cagr']:+.2%}",    f"{d_cagr:+.2%}"),
        ("Sharpe",       f"{m_a['sharpe']:.3f}",    f"{m_b['sharpe']:.3f}",    f"{d_sharpe:+.3f}"),
        ("Sortino",      f"{m_a['sortino']:.3f}",   f"{m_b['sortino']:.3f}",   f"{d_sortino:+.3f}"),
        ("Max Drawdown", f"{m_a['max_dd']:.1%}",    f"{m_b['max_dd']:.1%}",    f"{d_dd:+.1%}"),
        ("Win Rate",     f"{m_a['win_rate']:.1%}",  f"{m_b['win_rate']:.1%}",  f"{d_wr:+.1%}"),
        ("Alpha vs SPY", f"{m_a['alpha']:+.2%}",    f"{m_b['alpha']:+.2%}",    f"{d_alpha:+.2%}"),
        ("Trades",       f"{m_a['n_trades']}",      f"{m_b['n_trades']}",      f"{d_trades:+d}"),
        ("OOS AUC",      f"{auc_current:.4f}",      f"{auc_v2:.4f}",          f"{auc_v2-auc_current:+.4f}"),
    ]

    print(f"  {'Metric':<16} {'Current':>12} {'V2 Fund':>12} {'Delta':>12}")
    print(f"  {'─'*16} {'─'*12} {'─'*12} {'─'*12}")
    for name, cur, v2, delta in deltas:
        print(f"  {name:<16} {cur:>12} {v2:>12} {delta:>12}")

    # ── Annual breakdown ─────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  ANNUAL RETURNS")
    print(f"{'─'*90}")

    annual_a = vals_a.resample("YE").last().pct_change().dropna()
    annual_a.index = annual_a.index.year
    annual_b = vals_b.resample("YE").last().pct_change().dropna()
    annual_b.index = annual_b.index.year
    annual_spy = spy_bh.resample("YE").last().pct_change().dropna()
    annual_spy.index = annual_spy.index.year

    all_years = sorted(set(annual_a.index) | set(annual_b.index))
    print(f"  {'Year':>6}  {'Current':>10}  {'V2 Fund':>10}  {'SPY':>10}  {'Δ CAGR':>10}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for yr in all_years:
        ra = annual_a.get(yr, np.nan)
        rb = annual_b.get(yr, np.nan)
        rs = annual_spy.get(yr, np.nan)
        delta_yr = rb - ra if not (np.isnan(ra) or np.isnan(rb)) else np.nan
        ra_s = f"{ra:+.1%}" if not np.isnan(ra) else "N/A"
        rb_s = f"{rb:+.1%}" if not np.isnan(rb) else "N/A"
        rs_s = f"{rs:+.1%}" if not np.isnan(rs) else "N/A"
        d_s = f"{delta_yr:+.1%}" if not np.isnan(delta_yr) else "N/A"
        print(f"  {yr:>6}  {ra_s:>10}  {rb_s:>10}  {rs_s:>10}  {d_s:>10}")

    # ── Pick distribution comparison ─────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  PICK DISTRIBUTION BY YEAR (ML signals at threshold 0.55)")
    print(f"{'─'*90}")

    df_current["year"] = df_current["date"].dt.year
    df_v2["year"] = df_v2["date"].dt.year

    all_pick_years = sorted(set(df_current["year"].unique()) | set(df_v2["year"].unique()))
    print(f"  {'Year':>6}  {'Current >0.55':>14}  {'V2 Fund >0.55':>14}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*14}")
    for yr in all_pick_years:
        c_picks = len(df_current[(df_current["year"]==yr) & (df_current["prob"]>=0.55)])
        v_picks = len(df_v2[(df_v2["year"]==yr) & (df_v2["prob"]>=0.55)])
        print(f"  {yr:>6}  {c_picks:>14}  {v_picks:>14}")

    # ── Recommendation ───────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  RECOMMENDATION")
    print(f"{'='*90}")

    sharpe_improved = d_sharpe >= 0.05
    cagr_ok = d_cagr >= -0.01
    dd_ok = d_dd >= -0.02  # max_dd is negative, so worsening means more negative

    n_improve = sum([
        d_cagr > 0,
        d_sharpe > 0,
        d_dd > 0,  # less negative = better
    ])

    # Check for significant worsening
    cagr_worse = d_cagr < -0.02
    sharpe_worse = d_sharpe < -0.05
    dd_worse = d_dd < -0.03

    if sharpe_improved and cagr_ok and dd_ok:
        verdict = "DEPLOY"
        reason = (f"Sharpe improved by {d_sharpe:+.3f} (>= +0.05), "
                  f"CAGR delta {d_cagr:+.2%} (within -1%), "
                  f"DD delta {d_dd:+.1%} (within -2pp)")
    elif n_improve >= 2:
        verdict = "CONSIDER DEPLOYING"
        reason = f"{n_improve}/3 main metrics improved"
    elif cagr_worse or sharpe_worse or dd_worse:
        verdict = "REJECT"
        reasons = []
        if cagr_worse:
            reasons.append(f"CAGR dropped {d_cagr:+.2%}")
        if sharpe_worse:
            reasons.append(f"Sharpe dropped {d_sharpe:+.3f}")
        if dd_worse:
            reasons.append(f"DD worsened {d_dd:+.1%}")
        reason = "Multiple metrics worsened: " + ", ".join(reasons)
    else:
        verdict = "NEUTRAL"
        reason = "No clear improvement or worsening"

    print(f"\n  Criteria check:")
    print(f"    Sharpe +0.05?     {'YES' if sharpe_improved else 'NO':>5}  (Δ = {d_sharpe:+.3f})")
    print(f"    CAGR within -1%?  {'YES' if cagr_ok else 'NO':>5}  (Δ = {d_cagr:+.2%})")
    print(f"    DD within -2pp?   {'YES' if dd_ok else 'NO':>5}  (Δ = {d_dd:+.1%})")
    print(f"    Improved 2/3?     {'YES' if n_improve >= 2 else 'NO':>5}  ({n_improve}/3)")
    print(f"\n  VERDICT: {verdict}")
    print(f"  Reason: {reason}")

    print(f"\n{'='*90}")
    print(f"  Runtime: {time.perf_counter() - t0:.0f}s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
