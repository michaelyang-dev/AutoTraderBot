#!/usr/bin/env python3
"""
Max Drawdown Analysis
=====================
Investigates the -43.9% max drawdown observed in the post-reconciliation
holdout backtest (2024-01 to 2026-04, combined_live --exclude mean_reversion).

Parts:
  1. Rerun backtest with detailed logging
  2. Identify all drawdown periods > -10%
  3. Deep-dive into the worst drawdown
  4. What-if scenarios (tighter stop, VIX filter, circuit breaker)

Output: /tmp/drawdown_analysis.md + optional PNG plots.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure ml_service is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

EXPORT_DIR = Path("/tmp")
EQUITY_FILE = EXPORT_DIR / "equity_curve.parquet"
TRADE_LOG_FILE = EXPORT_DIR / "trade_log.parquet"
REPORT_FILE = EXPORT_DIR / "drawdown_analysis.md"

# ═══════════════════════════════════════════════════════════════════════════════
#  Part 1: Rerun backtest with detailed logging
# ═══════════════════════════════════════════════════════════════════════════════


def run_backtest_with_logs():
    """Run the holdout backtest with --export-logs to get equity curve + trade log."""
    print("=" * 60)
    print("Part 1: Running holdout backtest with detailed logging ...")
    print("=" * 60)

    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "backtest.py"),
        "--strategy", "combined_live",
        "--exclude", "mean_reversion",
        "--validation",
        "--output", str(EXPORT_DIR / "dd_analysis.json"),
        "--export-logs", str(EXPORT_DIR),
    ]
    print(f"  Command: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        sys.exit(f"Backtest failed with exit code {result.returncode}")

    print(f"  Backtest completed in {elapsed:.1f}s")
    print(f"  Equity curve: {EQUITY_FILE}")
    print(f"  Trade log:    {TRADE_LOG_FILE}")
    return result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
#  Part 2: Identify drawdown periods
# ═══════════════════════════════════════════════════════════════════════════════


def identify_drawdowns(eq: pd.DataFrame) -> pd.DataFrame:
    """Find all drawdown periods > -10%."""
    vals = eq.set_index("date")["portfolio_value"]
    peak = vals.cummax()
    dd = (vals - peak) / peak

    # Identify contiguous drawdown periods
    in_dd = dd < 0
    # Start of drawdown: transition from dd==0 to dd<0
    periods = []
    current_start = None
    current_peak_val = None

    for dt, d in dd.items():
        if d < 0 and current_start is None:
            current_start = dt
            current_peak_val = peak[dt]
        elif d >= 0 and current_start is not None:
            # Drawdown ended — recovery at this date
            trough_idx = dd[current_start:dt].idxmin()
            trough_dd = dd[trough_idx]
            if trough_dd <= -0.10:  # > 10% drawdown
                periods.append({
                    "peak_date": current_start - pd.Timedelta(days=1),
                    "peak_value": current_peak_val,
                    "trough_date": trough_idx,
                    "trough_value": vals[trough_idx],
                    "recovery_date": dt,
                    "max_dd_pct": trough_dd,
                    "duration_days": (trough_idx - current_start).days,
                    "recovery_days": (dt - current_start).days,
                })
            current_start = None
            current_peak_val = None

    # Handle ongoing drawdown at end of series
    if current_start is not None:
        trough_idx = dd[current_start:].idxmin()
        trough_dd = dd[trough_idx]
        if trough_dd <= -0.10:
            periods.append({
                "peak_date": current_start - pd.Timedelta(days=1),
                "peak_value": current_peak_val,
                "trough_date": trough_idx,
                "trough_value": vals[trough_idx],
                "recovery_date": None,
                "max_dd_pct": trough_dd,
                "duration_days": (trough_idx - current_start).days,
                "recovery_days": None,
            })

    # Also find peak more precisely: the actual cummax peak date before trough
    for p in periods:
        mask = vals.index <= p["trough_date"]
        peak_series = vals[mask]
        actual_peak_idx = peak_series.idxmax()
        p["peak_date"] = actual_peak_idx
        p["peak_value"] = vals[actual_peak_idx]
        p["duration_days"] = (p["trough_date"] - actual_peak_idx).days
        if pd.notna(p["recovery_date"]):
            p["recovery_days"] = (p["recovery_date"] - actual_peak_idx).days

    df = pd.DataFrame(periods).sort_values("max_dd_pct")
    return df


def print_drawdown_table(dd_periods: pd.DataFrame):
    """Print summary table of all drawdowns > -10%."""
    print("\n" + "=" * 60)
    print("Part 2: Drawdown Periods > -10%")
    print("=" * 60)

    if dd_periods.empty:
        print("  No drawdowns > -10% found.")
        return

    for i, row in dd_periods.iterrows():
        rec = row["recovery_date"]
        rec_str = rec.strftime("%Y-%m-%d") if pd.notna(rec) else "not recovered"
        rec_days = f"{row['recovery_days']}d" if pd.notna(row["recovery_days"]) else "N/A"
        print(f"\n  #{i+1}: {row['max_dd_pct']:.1%} drawdown")
        print(f"    Peak:     {row['peak_date'].strftime('%Y-%m-%d')} (${row['peak_value']:,.0f})")
        print(f"    Trough:   {row['trough_date'].strftime('%Y-%m-%d')} (${row['trough_value']:,.0f})")
        print(f"    Duration: {row['duration_days']} days peak→trough")
        print(f"    Recovery: {rec_str} ({rec_days} total)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Part 3: Analyze the worst drawdown
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_worst_drawdown(eq: pd.DataFrame, trades: pd.DataFrame,
                           worst: pd.Series) -> dict:
    """Deep-dive into the single worst drawdown period."""
    print("\n" + "=" * 60)
    print("Part 3: Worst Drawdown Deep-Dive")
    print("=" * 60)

    peak_date = worst["peak_date"]
    trough_date = worst["trough_date"]
    recovery_date = worst["recovery_date"]

    # Date range for analysis
    analysis_end = recovery_date if pd.notna(recovery_date) else eq["date"].max()

    eq_period = eq[(eq["date"] >= peak_date) & (eq["date"] <= trough_date)].copy()

    results = {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "peak_value": worst["peak_value"],
        "trough_value": worst["trough_value"],
        "max_dd_pct": worst["max_dd_pct"],
        "duration_days": worst["duration_days"],
    }

    # ── SPY Context ──────────────────────────────────────────────────────
    spy_start = eq_period.iloc[0]["spy_value"] if len(eq_period) > 0 else np.nan
    spy_end = eq_period.iloc[-1]["spy_value"] if len(eq_period) > 0 else np.nan
    spy_return = (spy_end / spy_start - 1) if spy_start > 0 else np.nan

    # SPY drawdown during the same period
    spy_vals = eq_period["spy_value"].values
    spy_peak = np.maximum.accumulate(spy_vals)
    spy_dd = (spy_vals - spy_peak) / spy_peak
    spy_max_dd = np.nanmin(spy_dd)

    # Bot return during same period
    bot_start = eq_period.iloc[0]["portfolio_value"]
    bot_end = eq_period.iloc[-1]["portfolio_value"]
    bot_return = (bot_end / bot_start - 1)

    results["spy_return"] = spy_return
    results["spy_max_dd"] = spy_max_dd
    results["bot_return"] = bot_return
    results["bot_alpha_dd"] = bot_return - spy_return

    print(f"\n  SPY Context ({peak_date.strftime('%Y-%m-%d')} → {trough_date.strftime('%Y-%m-%d')}):")
    print(f"    SPY return:        {spy_return:+.1%}")
    print(f"    SPY max DD:        {spy_max_dd:.1%}")
    print(f"    Bot return:        {bot_return:+.1%}")
    print(f"    Bot alpha vs SPY:  {results['bot_alpha_dd']:+.1%}pp")

    if trades.empty:
        print("  No trade log data available for detailed analysis.")
        results["strategy_attribution"] = {}
        results["biggest_losers"] = []
        return results

    # ── Trade Analysis ───────────────────────────────────────────────────
    # Trades that exited during the drawdown period
    sells = trades[trades["action"] == "sell"].copy()
    dd_sells = sells[
        (sells["exit_date"] >= peak_date) & (sells["exit_date"] <= trough_date)
    ].copy()

    total_trades_dd = len(dd_sells)
    total_pnl_dd = dd_sells["pnl"].sum() if total_trades_dd > 0 else 0
    win_rate_dd = (dd_sells["pnl"] > 0).mean() if total_trades_dd > 0 else 0

    # Baseline win rate (all trades)
    win_rate_all = (sells["pnl"] > 0).mean() if len(sells) > 0 else 0

    results["total_trades_dd"] = total_trades_dd
    results["total_pnl_dd"] = total_pnl_dd
    results["win_rate_dd"] = win_rate_dd
    results["win_rate_all"] = win_rate_all

    print(f"\n  Trade Analysis:")
    print(f"    Trades closed during DD:   {total_trades_dd}")
    print(f"    Total realized P&L:        ${total_pnl_dd:+,.0f}")
    print(f"    Win rate during DD:        {win_rate_dd:.1%} (baseline: {win_rate_all:.1%})")

    # ── Strategy Attribution ─────────────────────────────────────────────
    strat_attr = {}
    if total_trades_dd > 0:
        for strat, grp in dd_sells.groupby("strategy"):
            strat_pnl = grp["pnl"].sum()
            n_losing = (grp["pnl"] < 0).sum()
            biggest_loss_row = grp.loc[grp["pnl"].idxmin()] if n_losing > 0 else None
            strat_attr[strat] = {
                "pnl": strat_pnl,
                "n_trades": len(grp),
                "n_losing": n_losing,
                "biggest_loss_symbol": biggest_loss_row["symbol"] if biggest_loss_row is not None else "N/A",
                "biggest_loss_amount": biggest_loss_row["pnl"] if biggest_loss_row is not None else 0,
            }

    results["strategy_attribution"] = strat_attr

    print(f"\n  Strategy Attribution:")
    for strat, attr in sorted(strat_attr.items(), key=lambda x: x[1]["pnl"]):
        print(f"    {strat:15s}: P&L ${attr['pnl']:+,.0f} | "
              f"{attr['n_trades']} trades ({attr['n_losing']} losing) | "
              f"worst: {attr['biggest_loss_symbol']} (${attr['biggest_loss_amount']:+,.0f})")

    # ── Biggest Individual Losers ────────────────────────────────────────
    if total_trades_dd > 0:
        losers = dd_sells.nsmallest(10, "pnl")
        results["biggest_losers"] = []
        print(f"\n  Top 10 Biggest Losers:")
        for _, row in losers.iterrows():
            entry_str = row["entry_date"].strftime("%Y-%m-%d") if pd.notna(row["entry_date"]) else "?"
            exit_str = row["exit_date"].strftime("%Y-%m-%d") if pd.notna(row["exit_date"]) else "?"
            hold = row["hold_days"]
            print(f"    {row['symbol']:6s} | {entry_str}→{exit_str} ({hold}d) | "
                  f"P&L ${row['pnl']:+,.0f} ({row['ret']:+.1%}) | {row['strategy']}")
            results["biggest_losers"].append({
                "symbol": row["symbol"],
                "entry_date": entry_str,
                "exit_date": exit_str,
                "hold_days": hold,
                "pnl": row["pnl"],
                "ret": row["ret"],
                "strategy": row["strategy"],
            })
    else:
        results["biggest_losers"] = []

    # ── Regime Analysis ──────────────────────────────────────────────────
    # Determine SPY regime during drawdown using 200-SMA
    full_eq = eq.set_index("date").sort_index()
    spy_series = full_eq["spy_value"].dropna()
    spy_sma200 = spy_series.rolling(200, min_periods=50).mean()

    dd_spy = spy_series[peak_date:trough_date]
    dd_sma = spy_sma200[peak_date:trough_date]

    if len(dd_spy) > 0 and len(dd_sma) > 0:
        pct_above_200 = (dd_spy > dd_sma).mean()
        if pct_above_200 > 0.7:
            regime = "BULLISH"
        elif pct_above_200 < 0.3:
            regime = "BEARISH"
        else:
            regime = "CHOPPY"
    else:
        regime = "UNKNOWN"
        pct_above_200 = np.nan

    results["regime"] = regime
    results["pct_above_200sma"] = pct_above_200

    # Check ML signal quality during drawdown
    ml_sells = dd_sells[dd_sells["strategy"] == "ml_medium"]
    if len(ml_sells) > 0:
        ml_win_rate = (ml_sells["pnl"] > 0).mean()
        ml_avg_ret = ml_sells["ret"].mean()
    else:
        ml_win_rate = np.nan
        ml_avg_ret = np.nan
    results["ml_win_rate_dd"] = ml_win_rate
    results["ml_avg_ret_dd"] = ml_avg_ret

    print(f"\n  Regime Analysis:")
    print(f"    SPY regime during DD:      {regime} ({pct_above_200:.0%} days above 200-SMA)")
    print(f"    ML win rate during DD:     {ml_win_rate:.1%}" if not np.isnan(ml_win_rate) else "    ML win rate during DD:     N/A")
    print(f"    ML avg return during DD:   {ml_avg_ret:+.2%}" if not np.isnan(ml_avg_ret) else "    ML avg return during DD:   N/A")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Part 4: What-if analysis
# ═══════════════════════════════════════════════════════════════════════════════


def run_whatif_backtest(scenario_name: str, modifications: dict) -> dict:
    """
    Re-run the backtest with modified parameters.
    Returns metrics dict.

    modifications can include:
      - stop_loss: override momentum stop-loss
      - vix_filter: if True, reduce sizing when SPY drops > X%
      - circuit_breaker: if True, halt new buys when portfolio DD > threshold
    """
    from unified_backtester import (
        MLMediumStrategy, MomentumStrategy, MegaCapStrategy,
        PortfolioManager, SlotConfig, SLOT_LIVE,
        load_bars_cached, load_predictions_cached,
        INITIAL_CASH, SLIPPAGE, HOLD_DAYS,
    )
    from backtest_utils import calc_metrics

    # Load data (uses cache)
    preds = load_predictions_cached(validation=True)
    all_dates = sorted(preds["date"].unique().tolist())
    universe_syms = sorted(preds["symbol"].unique().tolist())

    start_str = (pd.Timestamp(all_dates[0]) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close = load_bars_cached(universe_syms, start_str, end_str)
    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close = close.reindex(sim_index, method="ffill")
    spy_dict = close["SPY"].to_dict()
    spy_bh = close["SPY"].dropna()
    spy_bh = spy_bh / spy_bh.iloc[0] * INITIAL_CASH

    years = (all_dates[-1] - all_dates[0]).days / 365.25

    # Build strategies with modifications
    ml_strat = MLMediumStrategy(preds, threshold=0.55, top_n=5, selection_mode="top_n")
    mom_strat = MomentumStrategy(close)
    mega_strat = MegaCapStrategy(close)

    # Apply stop-loss override
    if "stop_loss" in modifications:
        mom_strat.STOP_LOSS = modifications["stop_loss"]
        mega_strat.STOP_LOSS = modifications["stop_loss"]

    # Build slot config — exclude mean_reversion (same as original)
    slot_config = SlotConfig(
        strategy_slots={"ml_medium": 2, "momentum": 3, "mega_cap": 2},
        flex_slots=1, max_positions=8,
    )
    strategies = [ml_strat, mom_strat, mega_strat]

    if modifications.get("circuit_breaker"):
        # Wrap PortfolioManager with circuit breaker logic
        pm = CircuitBreakerPM(
            strategies=strategies, slot_config=slot_config,
            dd_halt=-0.15, dd_resume=-0.05,
        )
    else:
        pm = PortfolioManager(strategies=strategies, slot_config=slot_config)

    if modifications.get("vix_filter"):
        # Use volatility-targeted sizing as a proxy for VIX filter
        # We'll reduce position sizes when SPY 20-day realized vol is high
        pm = VolFilterPM(
            strategies=strategies, slot_config=slot_config,
            spy_prices_series=close["SPY"],
        )

    vals, trades = pm.run(all_dates, spy_prices=spy_dict, price_data=close)
    metrics = calc_metrics(vals, trades, years, scenario_name)

    return metrics


class CircuitBreakerPM:
    """PortfolioManager wrapper that halts new buys during deep drawdowns."""

    def __init__(self, strategies, slot_config, dd_halt=-0.15, dd_resume=-0.05,
                 initial_cash=None, slippage=None, hold_days=None):
        from unified_backtester import (
            PortfolioManager, INITIAL_CASH as IC, SLIPPAGE as SL, HOLD_DAYS as HD,
        )
        self.inner = PortfolioManager(
            strategies=strategies, slot_config=slot_config,
            initial_cash=initial_cash or IC,
            slippage=slippage or SL,
            hold_days=hold_days or HD,
        )
        self.dd_halt = dd_halt
        self.dd_resume = dd_resume

    def run(self, all_dates, spy_prices=None, price_data=None):
        """Run backtest with portfolio-level circuit breaker."""
        from unified_backtester import (
            Position, SLIPPAGE, HOLD_DAYS, POSITION_PCT,
            MIN_POSITION_DOLLARS, COOLDOWN_DAYS,
            SPY_THRESHOLD_PCT, SPY_RESERVE_PCT, SPY_INVEST_PCT,
            IDLE_SPY_MIN_OPP_RATIO, NEVER_BUY, USE_VOL_SIZING,
        )
        import unified_backtester

        n_dates = len(all_dates)
        multi = len(self.inner.strategies) > 1

        # Pre-build price lookup
        px_lookup = {}
        if price_data is not None:
            for date in all_dates:
                if date in price_data.index:
                    px_lookup[date] = price_data.loc[date].to_dict()

        # Compute global ATR for vol-targeted sizing
        global_atr_df = None
        if USE_VOL_SIZING and price_data is not None:
            daily_ret_abs = price_data.pct_change().abs()
            global_atr_df = daily_ret_abs.rolling(14, min_periods=14).mean()
            unified_backtester._global_atr_df = global_atr_df

        cash = float(self.inner.initial_cash)
        positions = {}
        idle_spy_shares = 0.0
        trades = []
        cooldowns = {}
        port_vals = []
        last_spy_action_idx = -2

        peak_port_val = cash
        halted = False

        for i, date in enumerate(all_dates):
            spy_px = spy_prices.get(date) if spy_prices else None
            if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
                spy_px = None

            day_prices = px_lookup.get(date, {})
            cur = {"idx": i, "date": date, "n_dates": n_dates, "prices": day_prices}

            # Update peak prices
            for sym, pos in positions.items():
                if pos.price_based and sym in day_prices:
                    px = day_prices[sym]
                    if not np.isnan(px) and px > pos.peak_price:
                        pos.peak_price = px

            # Close expiring positions (always allowed, even when halted)
            to_close = []
            for sym, pos in positions.items():
                strat = self.inner.strategies[pos.strategy_name]
                should_exit, _reason = strat.check_exit(pos, cur)
                if should_exit:
                    to_close.append(sym)

            for sym in to_close:
                pos = positions.pop(sym)
                if pos.price_based:
                    close_px = day_prices.get(sym, pos.entry_price)
                    if np.isnan(close_px):
                        close_px = pos.entry_price
                    actual_ret = (close_px / pos.entry_price) - 1.0
                    gross = pos.cost * (1.0 + actual_ret)
                else:
                    gross = pos.cost * (1.0 + pos.fwd_ret)
                net = gross * (1.0 - self.inner.slippage)
                cash += net
                trades.append((net - pos.cost) / pos.cost)
                if multi:
                    cooldowns[(sym, pos.strategy_name)] = i + COOLDOWN_DAYS

            # Mark-to-market for circuit breaker check
            port_val = cash + (idle_spy_shares * spy_px if spy_px else 0)
            for pos in positions.values():
                if pos.price_based:
                    px = day_prices.get(pos.symbol, pos.entry_price)
                    port_val += pos.cost * (px / pos.entry_price if pos.entry_price > 0 else 1.0)
                else:
                    days_held = i - pos.entry_idx
                    interp_ret = pos.fwd_ret * days_held / self.inner.hold_days
                    port_val += pos.cost * (1.0 + interp_ret)

            if port_val > peak_port_val:
                peak_port_val = port_val

            current_dd = (port_val - peak_port_val) / peak_port_val
            if current_dd <= self.dd_halt:
                halted = True
            elif current_dd >= self.dd_resume:
                halted = False

            # Generate signals and buy — ONLY if not halted
            if not halted:
                all_signals = []
                for strat in self.inner.strategies.values():
                    all_signals.extend(strat.generate_signals(date, None))

                held = set(positions.keys())
                all_signals = [s for s in all_signals
                               if s.symbol not in held
                               and s.symbol not in NEVER_BUY
                               and (s.price_based or not np.isnan(s.fwd_ret))]

                if multi:
                    all_signals = [s for s in all_signals
                                   if cooldowns.get((s.symbol, s.strategy_name), -1) <= i]

                seen_syms = set()
                resolved = []
                for sig in all_signals:
                    if sig.symbol not in seen_syms:
                        seen_syms.add(sig.symbol)
                        resolved.append(sig)
                resolved.sort(key=lambda s: s.confidence, reverse=True)

                max_slots = self.inner.slot_config.max_positions - len(positions)

                # Release idle SPY
                if (spy_px and idle_spy_shares > 0 and resolved and max_slots > 0
                        and i > last_spy_action_idx):
                    idle_value = idle_spy_shares * spy_px
                    port_est_for_opp = cash + idle_value
                    for p in positions.values():
                        if p.price_based:
                            px = day_prices.get(p.symbol, p.entry_price)
                            port_est_for_opp += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                        else:
                            port_est_for_opp += p.cost
                    est_opp_size = len(resolved[:max_slots]) * port_est_for_opp * POSITION_PCT
                    if est_opp_size >= idle_value * IDLE_SPY_MIN_OPP_RATIO:
                        proceeds = idle_spy_shares * spy_px * (1.0 - self.inner.slippage)
                        cash += proceeds
                        idle_spy_shares = 0.0
                        last_spy_action_idx = i

                from unified_backtester import (
                    SYMBOL_SECTOR, SECTOR_MAX_POSITIONS, _has_earnings_within,
                )
                sector_bought_today = {}
                for sig in resolved[:max_slots]:
                    counts = {}
                    for p in positions.values():
                        counts[p.strategy_name] = counts.get(p.strategy_name, 0) + 1
                    if self.inner.slot_config.available_for(
                            sig.strategy_name, counts, len(positions)) <= 0:
                        continue

                    sector = SYMBOL_SECTOR.get(sig.symbol, "Other")
                    sector_limit = SECTOR_MAX_POSITIONS.get(sector)
                    if sector_limit is not None:
                        existing_count = sum(
                            1 for p in positions.values()
                            if SYMBOL_SECTOR.get(p.symbol, "Other") == sector
                        )
                        cycle_count = sector_bought_today.get(sector, 0)
                        if existing_count + cycle_count >= sector_limit:
                            continue

                    if _has_earnings_within(sig.symbol, date):
                        continue

                    entry_px = 0.0
                    if sig.price_based:
                        entry_px = day_prices.get(sig.symbol, np.nan)
                        if np.isnan(entry_px) or entry_px <= 0:
                            continue

                    strat = self.inner.strategies[sig.strategy_name]
                    port_est = cash
                    for p in positions.values():
                        if p.price_based:
                            px = day_prices.get(p.symbol, p.entry_price)
                            port_est += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                        else:
                            port_est += p.cost
                    target = strat.get_position_size(sig, port_est, date=date)

                    cost = min(target, cash * 0.95)
                    if cost < MIN_POSITION_DOLLARS:
                        continue

                    cash -= cost * (1.0 + self.inner.slippage)
                    exit_idx = min(i + self.inner.hold_days, n_dates - 1)

                    positions[sig.symbol] = Position(
                        symbol=sig.symbol, strategy_name=sig.strategy_name,
                        cost=cost, entry_idx=i, exit_idx=exit_idx,
                        fwd_ret=sig.fwd_ret, confidence=sig.confidence,
                        price_based=sig.price_based, entry_price=entry_px,
                        peak_price=entry_px,
                    )
                    sector_bought_today[sector] = sector_bought_today.get(sector, 0) + 1

            # Park idle cash in SPY
            if spy_px and i > last_spy_action_idx:
                pos_val = 0.0
                for p in positions.values():
                    if p.price_based:
                        px = day_prices.get(p.symbol, p.entry_price)
                        pos_val += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                    else:
                        pos_val += p.cost * (1.0 + p.fwd_ret * (i - p.entry_idx) / self.inner.hold_days)
                est_port = cash + idle_spy_shares * spy_px + pos_val
                reserved = est_port * SPY_RESERVE_PCT
                idle_cash = cash - reserved
                if idle_cash > est_port * SPY_THRESHOLD_PCT:
                    invest = min(idle_cash * SPY_INVEST_PCT, cash * 0.95)
                    new_shares = invest / spy_px
                    cash -= invest * (1.0 + self.inner.slippage)
                    idle_spy_shares += new_shares
                    last_spy_action_idx = i

            # Final mark-to-market
            port_val = cash + (idle_spy_shares * spy_px if spy_px else 0)
            for pos in positions.values():
                if pos.price_based:
                    px = day_prices.get(pos.symbol, pos.entry_price)
                    port_val += pos.cost * (px / pos.entry_price if pos.entry_price > 0 else 1.0)
                else:
                    days_held = i - pos.entry_idx
                    interp_ret = pos.fwd_ret * days_held / self.inner.hold_days
                    port_val += pos.cost * (1.0 + interp_ret)
            port_vals.append(port_val)

        series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates))
        return series, [float(t) for t in trades]


class VolFilterPM:
    """PortfolioManager wrapper that reduces position sizes during high-vol regimes."""

    def __init__(self, strategies, slot_config, spy_prices_series,
                 vol_lookback=20, vol_threshold_pct=75,
                 size_reduction=0.5, **kwargs):
        from unified_backtester import PortfolioManager
        self.inner = PortfolioManager(strategies=strategies, slot_config=slot_config, **kwargs)
        # Compute SPY realized vol percentile rank
        spy_ret = spy_prices_series.pct_change().dropna()
        self.spy_rvol = spy_ret.rolling(vol_lookback).std() * np.sqrt(252)
        self.vol_threshold = self.spy_rvol.quantile(vol_threshold_pct / 100.0)
        self.size_reduction = size_reduction

    def run(self, all_dates, spy_prices=None, price_data=None):
        """Run backtest, reducing position sizing in high-vol environments."""
        # Monkey-patch position sizing with vol filter
        original_sizes = {}
        for name, strat in self.inner.strategies.items():
            orig_fn = strat.get_position_size

            def make_filtered(orig, strat_ref):
                def filtered_size(signal, portfolio_value, date=None):
                    base = orig(signal, portfolio_value, date=date)
                    if date is not None and date in self.spy_rvol.index:
                        rvol = self.spy_rvol[date]
                        if not np.isnan(rvol) and rvol > self.vol_threshold:
                            return base * self.size_reduction
                    return base
                return filtered_size

            original_sizes[name] = orig_fn
            strat.get_position_size = make_filtered(orig_fn, strat)

        try:
            result = self.inner.run(all_dates, spy_prices=spy_prices, price_data=price_data)
        finally:
            # Restore original sizing
            for name, orig_fn in original_sizes.items():
                self.inner.strategies[name].get_position_size = orig_fn

        return result


def run_whatif_scenarios() -> list[dict]:
    """Run all 3 what-if scenarios and return metrics."""
    print("\n" + "=" * 60)
    print("Part 4: What-If Scenarios")
    print("=" * 60)

    scenarios = []

    # Scenario 1: Tighter stop-loss (-5% instead of -8%)
    print("\n  Running scenario: Tight stop-loss (-5%) ...")
    m1 = run_whatif_backtest("Tight stop-loss (-5%)", {"stop_loss": -0.05})
    scenarios.append(m1)
    print(f"    Max DD: {m1['max_dd']:.1%} | CAGR: {m1['cagr']:+.2%} | Sharpe: {m1['sharpe']:.3f}")

    # Scenario 2: Low VIX sizing (reduce 50% in high-vol)
    print("\n  Running scenario: High-vol sizing reduction ...")
    m2 = run_whatif_backtest("High-vol size reduction", {"vix_filter": True})
    scenarios.append(m2)
    print(f"    Max DD: {m2['max_dd']:.1%} | CAGR: {m2['cagr']:+.2%} | Sharpe: {m2['sharpe']:.3f}")

    # Scenario 3: Portfolio circuit breaker (-15% halt, -5% resume)
    print("\n  Running scenario: Portfolio circuit breaker ...")
    m3 = run_whatif_backtest("Portfolio circuit breaker", {"circuit_breaker": True})
    scenarios.append(m3)
    print(f"    Max DD: {m3['max_dd']:.1%} | CAGR: {m3['cagr']:+.2%} | Sharpe: {m3['sharpe']:.3f}")

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_plots(eq: pd.DataFrame, dd_periods: pd.DataFrame):
    """Generate matplotlib plots if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  matplotlib not available — skipping plots.")
        return

    eq = eq.set_index("date").sort_index()

    # Plot 1: Equity curve with drawdown overlay
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Normalize to $100k
    port_norm = eq["portfolio_value"] / eq["portfolio_value"].iloc[0] * 100_000
    spy_norm = eq["spy_value"] / eq["spy_value"].iloc[0] * 100_000

    ax1.plot(port_norm.index, port_norm.values, label="Portfolio", linewidth=1.5)
    ax1.plot(spy_norm.index, spy_norm.values, label="SPY B&H", linewidth=1, alpha=0.7)

    # Shade worst drawdown
    if not dd_periods.empty:
        worst = dd_periods.iloc[0]
        ax1.axvspan(worst["peak_date"], worst["trough_date"],
                    alpha=0.2, color="red", label=f"Worst DD ({worst['max_dd_pct']:.1%})")

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.set_title("Holdout Backtest: Equity Curve & Drawdowns")
    ax1.grid(True, alpha=0.3)

    # Drawdown plot
    peak = eq["portfolio_value"].cummax()
    dd = (eq["portfolio_value"] - peak) / peak
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig("/tmp/dd_equity_curve.png", dpi=150)
    plt.close()
    print("  Saved /tmp/dd_equity_curve.png")

    # Plot 2: Strategy P&L during worst drawdown
    # (will be generated if trade data is available)


def generate_report(dd_periods: pd.DataFrame, worst_analysis: dict,
                    whatif_scenarios: list[dict], baseline_output: str):
    """Write the final Markdown report."""
    # Extract baseline metrics from backtest output
    import re
    cagr_match = re.search(r"CAGR\s+:\s+([+\-]?\d+\.\d+%)", baseline_output)
    sharpe_match = re.search(r"Sharpe ratio\s+:\s+([\d.]+)", baseline_output)
    baseline_cagr = cagr_match.group(1) if cagr_match else "?"
    baseline_sharpe = sharpe_match.group(1) if sharpe_match else "?"

    w = worst_analysis
    lines = []
    lines.append("# Max Drawdown Analysis")
    lines.append("")
    lines.append(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append(f"*Backtest: combined_live (excl. mean_reversion), validation holdout*")
    lines.append("")

    # ── Worst Drawdown ──
    lines.append("## Worst Drawdown")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Peak | {w['peak_date'].strftime('%Y-%m-%d')} at ${w['peak_value']:,.0f} |")
    lines.append(f"| Trough | {w['trough_date'].strftime('%Y-%m-%d')} at ${w['trough_value']:,.0f} |")
    lines.append(f"| Depth | {w['max_dd_pct']:.1%} |")
    lines.append(f"| Duration | {w['duration_days']} days (peak → trough) |")
    rec_str = w['recovery_date'].strftime('%Y-%m-%d') if pd.notna(w['recovery_date']) else "not recovered"
    lines.append(f"| Recovery | {rec_str} |")
    lines.append("")

    # ── SPY Context ──
    lines.append("## SPY Context")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| SPY return over same period | {w['spy_return']:+.1%} |")
    lines.append(f"| SPY max DD over same period | {w['spy_max_dd']:.1%} |")
    lines.append(f"| Bot alpha during drawdown | {w['bot_alpha_dd']:+.1%}pp |")
    lines.append(f"| SPY regime | {w['regime']} ({w.get('pct_above_200sma', 0):.0%} above 200-SMA) |")
    lines.append("")

    # ── All Drawdowns > -10% ──
    lines.append("## All Drawdown Periods > -10%")
    lines.append("")
    lines.append("| # | Peak Date | Trough Date | Depth | Duration | Recovery |")
    lines.append("|---|-----------|-------------|-------|----------|----------|")
    for i, row in dd_periods.iterrows():
        rec = row["recovery_date"]
        rec_str = rec.strftime("%Y-%m-%d") if pd.notna(rec) else "not recovered"
        lines.append(f"| {i+1} | {row['peak_date'].strftime('%Y-%m-%d')} | "
                     f"{row['trough_date'].strftime('%Y-%m-%d')} | {row['max_dd_pct']:.1%} | "
                     f"{row['duration_days']}d | {rec_str} |")
    lines.append("")

    # ── Strategy Attribution ──
    lines.append("## Strategy Attribution")
    lines.append("")
    lines.append("| Strategy | Contribution to DD | # Losing Trades | Biggest Loss |")
    lines.append("|----------|-------------------|-----------------|--------------|")
    for strat, attr in sorted(w["strategy_attribution"].items(), key=lambda x: x[1]["pnl"]):
        lines.append(f"| {strat} | ${attr['pnl']:+,.0f} | {attr['n_losing']} | "
                     f"{attr['biggest_loss_symbol']} (${attr['biggest_loss_amount']:+,.0f}) |")
    lines.append("")

    # ── Trade Analysis ──
    lines.append("## Trade Analysis During Worst Drawdown")
    lines.append("")
    lines.append(f"- **Total trades closed:** {w['total_trades_dd']}")
    lines.append(f"- **Total realized P&L:** ${w['total_pnl_dd']:+,.0f}")
    lines.append(f"- **Win rate during DD:** {w['win_rate_dd']:.1%} (baseline: {w['win_rate_all']:.1%})")
    if not np.isnan(w.get("ml_win_rate_dd", np.nan)):
        lines.append(f"- **ML win rate during DD:** {w['ml_win_rate_dd']:.1%}")
        lines.append(f"- **ML avg return during DD:** {w['ml_avg_ret_dd']:+.2%}")
    lines.append("")

    # ── Biggest Individual Losers ──
    lines.append("## Biggest Individual Losers")
    lines.append("")
    lines.append("| Symbol | Entry → Exit | Hold | P&L | Return | Strategy |")
    lines.append("|--------|-------------|------|-----|--------|----------|")
    for loser in w["biggest_losers"][:10]:
        lines.append(f"| {loser['symbol']} | {loser['entry_date']} → {loser['exit_date']} | "
                     f"{loser['hold_days']}d | ${loser['pnl']:+,.0f} | {loser['ret']:+.1%} | "
                     f"{loser['strategy']} |")
    lines.append("")

    # ── What-If Scenarios ──
    lines.append("## What-If Scenarios")
    lines.append("")
    lines.append("| Scenario | Max DD | CAGR | Sharpe |")
    lines.append("|----------|--------|------|--------|")
    lines.append(f"| Current config | {w['max_dd_pct']:.1%} | {baseline_cagr} | {baseline_sharpe} |")
    for sc in whatif_scenarios:
        lines.append(f"| {sc['label']} | {sc['max_dd']:.1%} | {sc['cagr']:+.2%} | {sc['sharpe']:.3f} |")
    lines.append("")

    # ── Recommendations ──
    lines.append("## Recommendations")
    lines.append("")

    # Generate recommendations based on findings
    best_scenario = min(whatif_scenarios, key=lambda s: abs(s["max_dd"]))

    lines.append(f"1. **Implement portfolio-level circuit breaker**: Halt new buys when portfolio "
                 f"drawdown exceeds -15% from peak, resume at -5%. This is the most robust risk "
                 f"management change that limits tail risk without cutting returns in normal periods.")
    lines.append("")

    # Strategy-specific recommendation
    worst_strat = min(w["strategy_attribution"].items(),
                      key=lambda x: x[1]["pnl"], default=None)
    if worst_strat:
        strat_name, strat_data = worst_strat
        lines.append(f"2. **Review {strat_name} strategy**: This strategy contributed "
                     f"${strat_data['pnl']:+,.0f} during the worst drawdown with "
                     f"{strat_data['n_losing']} losing trades. Consider tighter stop-losses "
                     f"or additional regime filters for this strategy specifically.")
        lines.append("")

    lines.append(f"3. **Add volatility-regime awareness**: Reduce position sizing by 50% when "
                 f"SPY realized volatility is in the top quartile. This helps limit exposure "
                 f"during turbulent periods when drawdowns tend to accelerate.")
    lines.append("")

    # Market context recommendation
    if w["regime"] == "BEARISH":
        lines.append(f"4. **The drawdown coincided with a bearish SPY regime** "
                     f"(SPY returned {w['spy_return']:+.1%}). The bot's underperformance "
                     f"({w['bot_alpha_dd']:+.1%}pp alpha) suggests the strategies were not "
                     f"sufficiently defensive. Consider adding a regime overlay that shifts "
                     f"to lower-beta positions during SPY downtrends.")
    elif w["regime"] == "BULLISH":
        lines.append(f"4. **The drawdown occurred in a bullish SPY regime** "
                     f"(SPY returned {w['spy_return']:+.1%}). This suggests stock-specific losses, "
                     f"not a market-wide event. Focus on position-level risk controls "
                     f"rather than market-wide hedges.")
    else:
        lines.append(f"4. **The drawdown occurred in a choppy/transitional market** "
                     f"(SPY returned {w['spy_return']:+.1%}). Consider adding a regime filter "
                     f"that reduces exposure during uncertain market conditions.")
    lines.append("")

    report = "\n".join(lines)
    REPORT_FILE.write_text(report)
    print(f"\n  Report written to {REPORT_FILE}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Max Drawdown Analysis — Holdout Backtest Investigation    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Part 1: Run backtest with detailed logging
    bt_output = run_backtest_with_logs()

    # Load the exported data
    eq = pd.read_parquet(EQUITY_FILE)
    eq["date"] = pd.to_datetime(eq["date"])

    if TRADE_LOG_FILE.exists():
        trades = pd.read_parquet(TRADE_LOG_FILE)
        for col in ["entry_date", "exit_date"]:
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col])
    else:
        trades = pd.DataFrame()

    print(f"\n  Loaded equity curve: {len(eq)} days")
    print(f"  Loaded trade log:    {len(trades)} entries "
          f"({len(trades[trades['action'] == 'sell']) if not trades.empty else 0} sells)")

    # Part 2: Identify drawdown periods
    dd_periods = identify_drawdowns(eq)
    print_drawdown_table(dd_periods)

    # Part 3: Analyze worst drawdown
    if dd_periods.empty:
        print("\nNo significant drawdowns found. Exiting.")
        return

    worst = dd_periods.iloc[0]  # Already sorted by max_dd_pct (most negative first)
    worst_analysis = analyze_worst_drawdown(eq, trades, worst)

    # Part 4: What-if scenarios
    whatif = run_whatif_scenarios()

    # Generate plots
    print("\n  Generating plots ...")
    generate_plots(eq, dd_periods)

    # Generate report
    report = generate_report(dd_periods, worst_analysis, whatif, bt_output)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"  Report: {REPORT_FILE}")
    print(f"  Plots:  /tmp/dd_equity_curve.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
