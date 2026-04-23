#!/usr/bin/env python3
"""
Walk-Forward Failure Analysis
==============================
Deep-dive into WHY the walk-forward validation underperforms.

Runs backtests with detail_log=True for all 11 years, then analyzes:
  1. Per-strategy P&L attribution
  2. Signal quality vs execution quality
  3. Exit analysis (MFE/MAE, exit reasons)
  4. Regime sensitivity (bull vs bear)
  5. Concentration / correlation

Produces: ml_service/data/walkforward/FAILURE_ANALYSIS.md
"""

import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    MegaCapStrategy, PortfolioManager, SlotConfig,
    SLOT_LIVE, load_bars_cached, load_predictions_cached,
    INITIAL_CASH, DATA_DIR,
)
from backtest_utils import calc_metrics, calc_alpha_beta

WF_DIR = DATA_DIR / "walkforward"
YEARS = list(range(2015, 2026))


def log(msg: str):
    print(msg, flush=True)


def run_backtest_with_details(year):
    """Run combined_live backtest for year with detail_log=True."""
    pred_file = WF_DIR / f"predictions_{year}.parquet"
    if not pred_file.exists():
        return None

    preds = load_predictions_cached(pred_file=pred_file, no_cache=True)
    all_dates = sorted(preds["date"].unique().tolist())
    universe_syms = sorted(preds["symbol"].unique().tolist())

    if len(all_dates) < 10:
        return None

    years_span = max((all_dates[-1] - all_dates[0]).days / 365.25, len(all_dates) / 252.0)

    start_str = (pd.Timestamp(all_dates[0]) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close = load_bars_cached(universe_syms, start_str, end_str)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close = close.reindex(sim_index, method="ffill")

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        close = load_bars_cached(universe_syms, start_str, end_str, no_cache=True)
        close = close.reindex(sim_index, method="ffill")
        spy_px = close["SPY"].dropna()
        if spy_px.empty:
            return None

    spy_dict = close["SPY"].to_dict()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    if "prob_ensemble" in preds.columns and "prob" not in preds.columns:
        preds = preds.rename(columns={"prob_ensemble": "prob"})

    strategies = [
        MLMediumStrategy(preds, threshold=0.55, top_n=5, selection_mode="top_n"),
        MomentumStrategy(close, volume_data=None, regime_filter=False),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]

    pm = PortfolioManager(strategies=strategies, slot_config=SLOT_LIVE)
    vals, trades, equity_df, trade_log_df = pm.run(
        all_dates, spy_prices=spy_dict, price_data=close, detail_log=True)

    metrics = calc_metrics(vals, trades, years_span, f"combined_live_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None

    # Compute SPY return for this year
    spy_ret = (spy_px.iloc[-1] / spy_px.iloc[0]) - 1.0

    return {
        "year": year,
        "metrics": metrics,
        "trade_log": trade_log_df,
        "equity_df": equity_df,
        "vals": vals,
        "spy_bh": spy_bh,
        "spy_ret": spy_ret,
        "close": close,
        "preds": preds,
        "all_dates": all_dates,
    }


# ── Task 1: Per-strategy attribution ─────────────────────────────────────────

def analyze_strategy_attribution(results):
    """Break down P&L by strategy for each year."""
    rows = []
    for r in results:
        year = r["year"]
        tl = r["trade_log"]
        sells = tl[tl["action"] == "sell"]

        for strat in ["ml_medium", "momentum", "mean_reversion", "mega_cap"]:
            strat_trades = sells[sells["strategy"] == strat]
            if len(strat_trades) == 0:
                rows.append({"year": year, "strategy": strat,
                             "pnl": 0, "trades": 0, "win_rate": 0, "avg_ret": 0})
                continue
            pnl = strat_trades["pnl"].sum()
            n = len(strat_trades)
            wins = (strat_trades["pnl"] > 0).sum()
            avg_ret = strat_trades["ret"].mean()
            rows.append({
                "year": year, "strategy": strat,
                "pnl": pnl, "trades": n,
                "win_rate": wins / n if n > 0 else 0,
                "avg_ret": avg_ret,
            })
    return pd.DataFrame(rows)


# ── Task 2: Signal quality vs execution ───────────────────────────────────────

def analyze_signal_quality(results):
    """Compare ML signal forward returns vs executed trade returns."""
    signal_stats = []

    for r in results:
        year = r["year"]
        preds = r["preds"]
        tl = r["trade_log"]
        sells = tl[(tl["action"] == "sell") & (tl["strategy"] == "ml_medium")]

        # Signal quality: what's the avg forward return of top-5 picks per day?
        if "prob" in preds.columns:
            prob_col = "prob"
        elif "prob_ensemble" in preds.columns:
            prob_col = "prob_ensemble"
        else:
            continue

        # Top-5 picks per day
        top5 = preds.groupby("date").apply(
            lambda g: g.nlargest(5, prob_col), include_groups=False
        ).reset_index(drop=True)

        # Random picks: sample 5 per day
        np.random.seed(42)
        random5 = preds.groupby("date").apply(
            lambda g: g.sample(min(5, len(g)), random_state=42), include_groups=False
        ).reset_index(drop=True)

        top5_fwd = top5["fwd_ret"].dropna()
        random5_fwd = random5["fwd_ret"].dropna()

        # Execution quality: what did executed ML trades actually return?
        executed_ret = sells["ret"].values if len(sells) > 0 else np.array([])

        signal_stats.append({
            "year": year,
            "top5_mean_fwd_ret": top5_fwd.mean() if len(top5_fwd) > 0 else 0,
            "top5_median_fwd_ret": top5_fwd.median() if len(top5_fwd) > 0 else 0,
            "random5_mean_fwd_ret": random5_fwd.mean() if len(random5_fwd) > 0 else 0,
            "signal_alpha": (top5_fwd.mean() - random5_fwd.mean()) if len(top5_fwd) > 0 else 0,
            "executed_mean_ret": executed_ret.mean() if len(executed_ret) > 0 else 0,
            "executed_n": len(executed_ret),
            "execution_gap": (top5_fwd.mean() - executed_ret.mean())
                if len(top5_fwd) > 0 and len(executed_ret) > 0 else 0,
        })

    return pd.DataFrame(signal_stats)


# ── Task 3: Exit analysis ────────────────────────────────────────────────────

def analyze_exits(results):
    """Analyze exit reasons and MFE/MAE for price-based trades."""
    all_exits = []
    mfe_mae_data = []

    for r in results:
        year = r["year"]
        tl = r["trade_log"]
        sells = tl[tl["action"] == "sell"].copy()
        close = r["close"]
        all_dates = r["all_dates"]

        for _, trade in sells.iterrows():
            exit_reason = trade.get("exit_reason", "unknown")
            all_exits.append({
                "year": year,
                "strategy": trade["strategy"],
                "exit_reason": exit_reason,
                "ret": trade["ret"],
                "pnl": trade["pnl"],
                "hold_days": trade["hold_days"],
            })

            # MFE/MAE for price-based trades
            if trade["price_based"] and trade["entry_price"] > 0:
                sym = trade["symbol"]
                entry_date = pd.Timestamp(trade["entry_date"])
                exit_date = pd.Timestamp(trade["exit_date"])

                if sym in close.columns:
                    mask = (close.index >= entry_date) & (close.index <= exit_date)
                    px_during = close.loc[mask, sym].dropna()
                    if len(px_during) > 0:
                        entry_px = trade["entry_price"]
                        rets_during = (px_during / entry_px) - 1.0
                        mfe = rets_during.max()
                        mae = rets_during.min()
                        final_ret = trade["ret"]
                        peak_price = trade.get("peak_price", px_during.max())

                        # How much profit did we leave on the table?
                        left_on_table = mfe - final_ret if final_ret > 0 else 0
                        # How much worse than MAE did we do?
                        exit_vs_worst = final_ret - mae

                        mfe_mae_data.append({
                            "year": year,
                            "strategy": trade["strategy"],
                            "symbol": sym,
                            "entry_date": entry_date,
                            "exit_reason": exit_reason,
                            "mfe": mfe,
                            "mae": mae,
                            "final_ret": final_ret,
                            "left_on_table": left_on_table,
                            "hold_days": trade["hold_days"],
                            "is_winner": final_ret > 0,
                        })

    return pd.DataFrame(all_exits), pd.DataFrame(mfe_mae_data)


# ── Task 4: Regime sensitivity ───────────────────────────────────────────────

def analyze_regime(results):
    """Compare bot performance in bull vs bear regime days."""
    regime_rows = []

    for r in results:
        year = r["year"]
        close = r["close"]
        vals = r["vals"]

        if "SPY" not in close.columns:
            continue

        spy = close["SPY"].dropna()
        spy_sma50 = spy.rolling(50, min_periods=50).mean()

        # Classify each day
        bull_days = set()
        bear_days = set()
        for date in vals.index:
            spy_val = spy.get(date)
            sma_val = spy_sma50.get(date)
            if spy_val is not None and sma_val is not None and not np.isnan(spy_val) and not np.isnan(sma_val):
                if spy_val >= sma_val:
                    bull_days.add(date)
                else:
                    bear_days.add(date)

        # Bot daily returns
        daily_rets = vals.pct_change().dropna()
        spy_daily = spy.reindex(vals.index).pct_change().dropna()

        bull_rets = daily_rets[daily_rets.index.isin(bull_days)]
        bear_rets = daily_rets[daily_rets.index.isin(bear_days)]
        spy_bull = spy_daily[spy_daily.index.isin(bull_days)]
        spy_bear = spy_daily[spy_daily.index.isin(bear_days)]

        # Annualized returns
        bull_ann = bull_rets.mean() * 252 if len(bull_rets) > 0 else 0
        bear_ann = bear_rets.mean() * 252 if len(bear_rets) > 0 else 0
        spy_bull_ann = spy_bull.mean() * 252 if len(spy_bull) > 0 else 0
        spy_bear_ann = spy_bear.mean() * 252 if len(spy_bear) > 0 else 0

        regime_rows.append({
            "year": year,
            "bull_days": len(bull_rets),
            "bear_days": len(bear_rets),
            "bot_bull_ann": bull_ann,
            "bot_bear_ann": bear_ann,
            "spy_bull_ann": spy_bull_ann,
            "spy_bear_ann": spy_bear_ann,
            "bot_vs_spy_bull": bull_ann - spy_bull_ann,
            "bot_vs_spy_bear": bear_ann - spy_bear_ann,
        })

    return pd.DataFrame(regime_rows)


# ── Task 5: Concentration / beta analysis ─────────────────────────────────────

def analyze_concentration(results):
    """How does portfolio drop relative to market?"""
    beta_rows = []

    for r in results:
        year = r["year"]
        vals = r["vals"]
        spy_bh = r["spy_bh"]

        # Beta via daily returns
        bot_rets = vals.pct_change().dropna()
        spy_rets = spy_bh.reindex(vals.index, method="ffill").pct_change().dropna()
        combined = pd.concat([bot_rets, spy_rets], axis=1, join="inner")
        combined.columns = ["bot", "spy"]
        combined = combined.dropna()

        if len(combined) < 20:
            continue

        from scipy import stats
        slope, intercept, r_val, p_val, _ = stats.linregress(combined["spy"], combined["bot"])

        # Downside beta: only on negative spy days
        down = combined[combined["spy"] < 0]
        if len(down) >= 10:
            down_slope, _, _, _, _ = stats.linregress(down["spy"], down["bot"])
        else:
            down_slope = np.nan

        # Worst SPY days: bot performance on SPY's worst 5 days
        worst_spy_days = combined.nsmallest(5, "spy")
        avg_bot_on_worst = worst_spy_days["bot"].mean()
        avg_spy_on_worst = worst_spy_days["spy"].mean()

        beta_rows.append({
            "year": year,
            "beta": slope,
            "downside_beta": down_slope,
            "alpha_daily": intercept,
            "r_squared": r_val ** 2,
            "bot_on_worst5_spy_days": avg_bot_on_worst,
            "spy_on_worst5_days": avg_spy_on_worst,
            "amplification": avg_bot_on_worst / avg_spy_on_worst if avg_spy_on_worst != 0 else np.nan,
        })

    return pd.DataFrame(beta_rows)


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(results, attrib_df, signal_df, exits_df, mfe_df, regime_df, conc_df):
    lines = []
    lines.append("# Walk-Forward Failure Mode Analysis")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    # ══════════════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY — Top 3 failure modes
    # ══════════════════════════════════════════════════════════════════════

    # Compute total P&L by strategy across all years
    strat_totals = attrib_df.groupby("strategy")["pnl"].sum().sort_values()

    # Compute signal quality summary
    avg_signal_alpha = signal_df["signal_alpha"].mean() if len(signal_df) > 0 else 0
    avg_exec_gap = signal_df["execution_gap"].mean() if len(signal_df) > 0 else 0

    # Regime summary
    avg_bear_underperf = regime_df["bot_vs_spy_bear"].mean() if len(regime_df) > 0 else 0
    avg_bull_underperf = regime_df["bot_vs_spy_bull"].mean() if len(regime_df) > 0 else 0

    # Downside beta
    avg_down_beta = conc_df["downside_beta"].mean() if len(conc_df) > 0 else 1.0

    lines.append("\n---")
    lines.append("\n## Executive Summary: Top 3 Causes of Underperformance\n")

    # Rank failure modes by dollar impact
    failure_modes = []

    # 1. Strategy-level losses
    for strat in strat_totals.index:
        total_pnl = strat_totals[strat]
        if total_pnl < 0:
            strat_data = attrib_df[attrib_df["strategy"] == strat]
            losing_years = strat_data[strat_data["pnl"] < 0]["year"].tolist()
            failure_modes.append({
                "name": f"{strat} strategy net losses",
                "dollar_impact": total_pnl,
                "detail": f"Lost ${abs(total_pnl):,.0f} across {len(losing_years)} years: {losing_years}",
            })

    # 2. Bear market amplification
    bear_loss_years = regime_df[regime_df["bot_vs_spy_bear"] < -0.10]
    if len(bear_loss_years) > 0:
        # Estimate dollar impact of bear underperformance
        bear_dollar = 0
        for _, row in bear_loss_years.iterrows():
            bear_dollar += row["bot_vs_spy_bear"] * INITIAL_CASH * row["bear_days"] / 252
        failure_modes.append({
            "name": "Bear market amplification (high downside beta)",
            "dollar_impact": bear_dollar,
            "detail": f"Avg downside beta: {avg_down_beta:.2f}. "
                      f"Bot underperforms SPY by {avg_bear_underperf:+.1%} annualized in bear days",
        })

    # 3. Execution gap
    if avg_exec_gap > 0.001:
        exec_dollar = avg_exec_gap * signal_df["executed_n"].sum() * INITIAL_CASH * 0.12
        failure_modes.append({
            "name": "Execution gap (signal alpha lost in trading)",
            "dollar_impact": -exec_dollar,
            "detail": f"Signal alpha: {avg_signal_alpha:+.3%}/trade, "
                      f"execution gap: {avg_exec_gap:+.3%}/trade",
        })

    failure_modes.sort(key=lambda x: x["dollar_impact"])

    for rank, fm in enumerate(failure_modes[:3], 1):
        lines.append(f"### {rank}. {fm['name']}")
        lines.append(f"- **Dollar impact**: ${fm['dollar_impact']:,.0f}")
        lines.append(f"- **Evidence**: {fm['detail']}")
        lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 1: Per-strategy attribution
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n---")
    lines.append("\n## 1. Per-Strategy P&L Attribution\n")

    # Pivot table
    lines.append("| Year | ML P&L | ML Win% | ML Trades | MOM P&L | MOM Win% | MOM Trades | MR P&L | MR Win% | MCAP P&L | MCAP Win% |")
    lines.append("|------|--------|---------|-----------|---------|----------|------------|--------|---------|----------|-----------|")

    for year in YEARS:
        yr_data = attrib_df[attrib_df["year"] == year]
        if len(yr_data) == 0:
            continue
        ml = yr_data[yr_data["strategy"] == "ml_medium"].iloc[0] if len(yr_data[yr_data["strategy"] == "ml_medium"]) > 0 else None
        mom = yr_data[yr_data["strategy"] == "momentum"].iloc[0] if len(yr_data[yr_data["strategy"] == "momentum"]) > 0 else None
        mr = yr_data[yr_data["strategy"] == "mean_reversion"].iloc[0] if len(yr_data[yr_data["strategy"] == "mean_reversion"]) > 0 else None
        mcap = yr_data[yr_data["strategy"] == "mega_cap"].iloc[0] if len(yr_data[yr_data["strategy"] == "mega_cap"]) > 0 else None

        def fmt(s):
            if s is None:
                return "—", "—", "—"
            return f"${s['pnl']:+,.0f}", f"{s['win_rate']:.0%}", f"{int(s['trades'])}"

        ml_p, ml_w, ml_t = fmt(ml)
        mom_p, mom_w, mom_t = fmt(mom)
        mr_p, mr_w, mr_t = fmt(mr)
        mcap_p, mcap_w, mcap_t = fmt(mcap)

        lines.append(f"| {year} | {ml_p} | {ml_w} | {ml_t} | {mom_p} | {mom_w} | {mom_t} | {mr_p} | {mr_w} | {mcap_p} | {mcap_w} |")

    # Totals
    lines.append("")
    lines.append("**Strategy totals (all years combined):**\n")
    for strat in ["ml_medium", "momentum", "mean_reversion", "mega_cap"]:
        sd = attrib_df[attrib_df["strategy"] == strat]
        total_pnl = sd["pnl"].sum()
        total_trades = sd["trades"].sum()
        avg_wr = sd["win_rate"].mean()
        avg_ret = sd["avg_ret"].mean()
        lines.append(f"- **{strat}**: ${total_pnl:+,.0f} total P&L | "
                     f"{int(total_trades)} trades | {avg_wr:.1%} avg win rate | "
                     f"{avg_ret:+.2%} avg trade return")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 2: Signal quality
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n---")
    lines.append("\n## 2. Signal Quality vs Execution Quality\n")
    lines.append("**Is the ML model predicting?** Comparing top-5 picks forward return vs random.\n")
    lines.append("| Year | Top-5 Fwd Ret | Random Fwd Ret | Signal Alpha | Executed Ret | Exec Gap |")
    lines.append("|------|---------------|----------------|--------------|--------------|----------|")

    for _, row in signal_df.iterrows():
        lines.append(
            f"| {int(row['year'])} "
            f"| {row['top5_mean_fwd_ret']:+.3%} "
            f"| {row['random5_mean_fwd_ret']:+.3%} "
            f"| {row['signal_alpha']:+.3%} "
            f"| {row['executed_mean_ret']:+.3%} "
            f"| {row['execution_gap']:+.3%} |"
        )

    avg_top5 = signal_df["top5_mean_fwd_ret"].mean()
    avg_rand = signal_df["random5_mean_fwd_ret"].mean()
    avg_exec = signal_df["executed_mean_ret"].mean()

    lines.append("")
    lines.append(f"**Averages across all years:**")
    lines.append(f"- Top-5 picks forward return: {avg_top5:+.3%}")
    lines.append(f"- Random picks forward return: {avg_rand:+.3%}")
    lines.append(f"- **Raw signal alpha: {avg_top5 - avg_rand:+.3%}**")
    lines.append(f"- Executed trade return: {avg_exec:+.3%}")
    lines.append(f"- **Execution gap: {avg_top5 - avg_exec:+.3%}**")

    if avg_top5 - avg_rand < 0.001:
        lines.append(f"\n> **Verdict: ML model has negligible predictive power.** "
                     f"Top-5 picks barely outperform random selection.")
    elif avg_top5 - avg_rand < 0.003:
        lines.append(f"\n> **Verdict: ML model has weak but non-zero signal.** "
                     f"Top-5 picks slightly outperform random.")
    else:
        lines.append(f"\n> **Verdict: ML model has meaningful signal.** "
                     f"Top-5 picks outperform random by {(avg_top5 - avg_rand)*100:.1f} bps/trade.")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 3: Exit analysis
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n---")
    lines.append("\n## 3. Exit Analysis\n")

    if len(exits_df) > 0:
        lines.append("### Exit reason distribution\n")
        for strat in ["ml_medium", "momentum", "mean_reversion", "mega_cap"]:
            strat_exits = exits_df[exits_df["strategy"] == strat]
            if len(strat_exits) == 0:
                continue
            lines.append(f"**{strat}:**")
            reason_counts = strat_exits["exit_reason"].value_counts()
            reason_pnl = strat_exits.groupby("exit_reason")["pnl"].sum()
            reason_ret = strat_exits.groupby("exit_reason")["ret"].mean()
            for reason in reason_counts.index:
                n = reason_counts[reason]
                pct = n / len(strat_exits)
                total_pnl = reason_pnl.get(reason, 0)
                avg_r = reason_ret.get(reason, 0)
                lines.append(f"- `{reason}`: {n} trades ({pct:.0%}) | "
                             f"P&L ${total_pnl:+,.0f} | avg ret {avg_r:+.2%}")
            lines.append("")

    if len(mfe_df) > 0:
        lines.append("### MFE / MAE Analysis (price-based trades only)\n")

        winners = mfe_df[mfe_df["is_winner"]]
        losers = mfe_df[~mfe_df["is_winner"]]

        if len(winners) > 0:
            avg_mfe_w = winners["mfe"].mean()
            avg_final_w = winners["final_ret"].mean()
            avg_left = winners["left_on_table"].mean()
            lines.append(f"**Winners ({len(winners)} trades):**")
            lines.append(f"- Avg MFE (max favorable excursion): {avg_mfe_w:+.2%}")
            lines.append(f"- Avg exit return: {avg_final_w:+.2%}")
            lines.append(f"- Avg profit left on table: {avg_left:+.2%} "
                         f"({avg_left/avg_mfe_w*100:.0f}% of peak unrealized)")
            lines.append("")

        if len(losers) > 0:
            avg_mae_l = losers["mae"].mean()
            avg_final_l = losers["final_ret"].mean()
            lines.append(f"**Losers ({len(losers)} trades):**")
            lines.append(f"- Avg MAE (max adverse excursion): {avg_mae_l:+.2%}")
            lines.append(f"- Avg exit return: {avg_final_l:+.2%}")
            lines.append(f"- Losers exiting at MAE (stop loss worked): "
                         f"{(losers['final_ret'] <= losers['mae'] * 1.1).sum()}/{len(losers)}")
            lines.append("")

        # Stop loss analysis
        lines.append("### Stop loss tightness analysis\n")
        for strat in ["momentum", "mega_cap"]:
            strat_mfe = mfe_df[mfe_df["strategy"] == strat]
            if len(strat_mfe) < 10:
                continue
            stopped = strat_mfe[strat_mfe["final_ret"] < -0.05]
            would_recover = 0
            for _, t in stopped.iterrows():
                # Trade was stopped out with >5% loss, but MFE shows it had potential
                if t["mfe"] > 0.02:  # Would have been profitable if held
                    would_recover += 1
            lines.append(f"**{strat}**: {len(stopped)} trades stopped at >-5% loss, "
                         f"{would_recover} ({would_recover/max(len(stopped),1):.0%}) "
                         f"had MFE > +2% (stop was premature)")

        lines.append("")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 4: Regime sensitivity
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n---")
    lines.append("\n## 4. Regime Sensitivity (Bull vs Bear)\n")
    lines.append("Bull = SPY above 50-SMA, Bear = SPY below 50-SMA\n")
    lines.append("| Year | Bull Days | Bear Days | Bot Bull (ann) | SPY Bull (ann) | Bot Bear (ann) | SPY Bear (ann) | Bot vs SPY (Bull) | Bot vs SPY (Bear) |")
    lines.append("|------|-----------|-----------|----------------|----------------|----------------|----------------|-------------------|-------------------|")

    for _, row in regime_df.iterrows():
        lines.append(
            f"| {int(row['year'])} "
            f"| {int(row['bull_days'])} "
            f"| {int(row['bear_days'])} "
            f"| {row['bot_bull_ann']:+.1%} "
            f"| {row['spy_bull_ann']:+.1%} "
            f"| {row['bot_bear_ann']:+.1%} "
            f"| {row['spy_bear_ann']:+.1%} "
            f"| {row['bot_vs_spy_bull']:+.1%} "
            f"| {row['bot_vs_spy_bear']:+.1%} |"
        )

    bull_alpha = regime_df["bot_vs_spy_bull"].mean()
    bear_alpha = regime_df["bot_vs_spy_bear"].mean()
    lines.append("")
    lines.append(f"**Average alpha vs SPY:**")
    lines.append(f"- In bull regime: {bull_alpha:+.1%} annualized")
    lines.append(f"- In bear regime: {bear_alpha:+.1%} annualized")

    if bear_alpha < -0.05:
        lines.append(f"\n> **Verdict: Bot is a leveraged bull.** "
                     f"Underperforms SPY badly in bear markets ({bear_alpha:+.1%}/yr), "
                     f"modestly in bull markets ({bull_alpha:+.1%}/yr).")

    # ══════════════════════════════════════════════════════════════════════
    # TASK 5: Concentration / beta
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n---")
    lines.append("\n## 5. Concentration & Downside Beta\n")
    lines.append("| Year | Beta | Downside Beta | R² | Bot on SPY Worst 5 Days | SPY Worst 5 | Amplification |")
    lines.append("|------|------|---------------|----|-----------------------|-------------|---------------|")

    for _, row in conc_df.iterrows():
        amp = f"{row['amplification']:.2f}x" if not np.isnan(row['amplification']) else "N/A"
        lines.append(
            f"| {int(row['year'])} "
            f"| {row['beta']:.2f} "
            f"| {row['downside_beta']:.2f} "
            f"| {row['r_squared']:.2f} "
            f"| {row['bot_on_worst5_spy_days']:+.2%} "
            f"| {row['spy_on_worst5_days']:+.2%} "
            f"| {amp} |"
        )

    avg_beta = conc_df["beta"].mean()
    avg_dbeta = conc_df["downside_beta"].mean()
    avg_amp = conc_df["amplification"].mean()
    lines.append("")
    lines.append(f"**Averages:**")
    lines.append(f"- Beta: {avg_beta:.2f}")
    lines.append(f"- Downside beta: {avg_dbeta:.2f}")
    lines.append(f"- Amplification on worst SPY days: {avg_amp:.2f}x")

    if avg_dbeta > 1.2:
        lines.append(f"\n> **Verdict: Portfolio amplifies downside.** "
                     f"Downside beta {avg_dbeta:.2f} means the bot drops "
                     f"{avg_dbeta:.0%} for every 1% SPY drops on bad days.")

    # ══════════════════════════════════════════════════════════════════════
    # RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n---")
    lines.append("\n## Recommendations\n")

    # Build recommendation list from evidence
    recs = []

    # Check ML signal quality
    if avg_top5 - avg_rand < 0.001:
        recs.append({
            "priority": 1,
            "fix": "Rebuild ML model — current model has no predictive edge",
            "evidence": f"Top-5 picks: {avg_top5:+.3%}/trade vs random: {avg_rand:+.3%}/trade "
                        f"(delta: {avg_top5 - avg_rand:+.3%})",
            "impact": "High — ML strategy is the core alpha source, currently adds no value",
        })
    elif avg_top5 - avg_rand < 0.003:
        recs.append({
            "priority": 2,
            "fix": "Improve ML model — weak but non-zero signal needs strengthening",
            "evidence": f"Signal alpha only {(avg_top5 - avg_rand)*10000:.0f} bps/trade",
            "impact": "Medium",
        })

    # Check bear market losses
    if bear_alpha < -0.10:
        recs.append({
            "priority": 1,
            "fix": "Add regime-aware position sizing or cash-raise in bear markets",
            "evidence": f"Bot underperforms SPY by {bear_alpha:+.1%}/yr in bear regime, "
                        f"downside beta = {avg_dbeta:.2f}",
            "impact": "High — bear market losses are the single largest dollar leak",
        })

    # Check specific strategy losses
    worst_strat = strat_totals.index[0] if len(strat_totals) > 0 else None
    if worst_strat and strat_totals.iloc[0] < -5000:
        recs.append({
            "priority": 2,
            "fix": f"Fix or remove {worst_strat} strategy",
            "evidence": f"Total P&L: ${strat_totals.iloc[0]:+,.0f} across all years",
            "impact": f"Removing it would recover ${abs(strat_totals.iloc[0]):,.0f}",
        })

    # Check execution gap
    if avg_exec_gap > 0.005:
        recs.append({
            "priority": 3,
            "fix": "Reduce execution gap — signals are better than realized trades",
            "evidence": f"Avg execution gap: {avg_exec_gap:+.3%}/trade",
            "impact": "Medium — improving trade execution converts existing signal into profit",
        })

    # Check downside beta
    if avg_dbeta > 1.3:
        recs.append({
            "priority": 2,
            "fix": "Reduce portfolio concentration — positions are too correlated",
            "evidence": f"Downside beta {avg_dbeta:.2f}, amplification {avg_amp:.2f}x on worst days",
            "impact": "High — concentrated bets amplify market drops",
        })

    recs.sort(key=lambda x: x["priority"])
    for i, rec in enumerate(recs, 1):
        lines.append(f"### Priority {rec['priority']}: {rec['fix']}")
        lines.append(f"- **Evidence**: {rec['evidence']}")
        lines.append(f"- **Expected impact**: {rec['impact']}")
        lines.append("")

    return "\n".join(lines)


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  WALK-FORWARD FAILURE ANALYSIS")
    log("=" * 70)

    # Run backtests with detail logging
    results = []
    for year in YEARS:
        log(f"\n  Year {year}: running backtest with detail_log ...")
        t1 = time.perf_counter()
        r = run_backtest_with_details(year)
        if r:
            results.append(r)
            m = r["metrics"]
            n_sells = len(r["trade_log"][r["trade_log"]["action"] == "sell"])
            log(f"    CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:.2f}  "
                f"SPY={r['spy_ret']:+.1%}  Trades={n_sells}  "
                f"({time.perf_counter()-t1:.0f}s)")

    if not results:
        sys.exit("No results — check walk-forward predictions exist.")

    log(f"\n{'─'*70}")
    log("  ANALYZING ...")
    log(f"{'─'*70}")

    # Task 1
    log("  Task 1: Strategy attribution ...")
    attrib_df = analyze_strategy_attribution(results)

    # Task 2
    log("  Task 2: Signal quality ...")
    signal_df = analyze_signal_quality(results)

    # Task 3
    log("  Task 3: Exit analysis ...")
    exits_df, mfe_df = analyze_exits(results)

    # Task 4
    log("  Task 4: Regime sensitivity ...")
    regime_df = analyze_regime(results)

    # Task 5
    log("  Task 5: Concentration / beta ...")
    conc_df = analyze_concentration(results)

    # Generate report
    log("  Generating report ...")
    report = generate_report(results, attrib_df, signal_df, exits_df, mfe_df, regime_df, conc_df)

    report_file = WF_DIR / "FAILURE_ANALYSIS.md"
    report_file.write_text(report)
    log(f"\n  Report: {report_file}")

    elapsed = time.perf_counter() - t0
    log(f"  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
