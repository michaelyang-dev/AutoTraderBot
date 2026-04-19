#!/usr/bin/env python3
"""
V4 Backtest with Volatility Targeting
======================================
Applies vol targeting to V4 cross-sectional ranking model.

V4 raw: 37.96% CAGR, 1.77 Sharpe, -39% DD
Hypothesis: V4 always trades (top-N), so it's more equity-like → vol targeting
should work the traditional way (inverse vol scaling).

Uses predictions_v4.parquet — no retraining.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR   = Path(__file__).resolve().parent / "data"
PRED_FILE  = DATA_DIR / "predictions_v4.parquet"

# ── Vol targeting parameters ─────────────────────────────────────────────────
TARGET_VOL    = 0.15    # 15% annualized target
MAX_LEVERAGE  = 1.5
MIN_LEVERAGE  = 0.3
VOL_LOOKBACK  = 20      # trading days

# Top-N selection (no threshold)
TOP_N_PICKS = 5

# Baselines for comparison
V2_CAGR, V2_SHARPE, V2_DD   = 0.2559, 1.65, -0.21
V4_CAGR, V4_SHARPE, V4_DD   = 0.3796, 1.77, -0.39
V3_CAGR, V3_SHARPE, V3_DD   = 0.3280, 1.52, -0.34
V3VT_CAGR, V3VT_SHARPE, V3VT_DD = 0.2865, 1.47, -0.26

# Pass criteria
PASS_CAGR   = 0.32
PASS_SHARPE = 1.70
PASS_DD     = -0.28


def log(msg: str):
    print(msg, flush=True)


def instrumented_run_voltarget(strategies, slot_config, all_dates, spy_prices,
                                price_data, hold_days, label):
    """
    Backtest loop with volatility targeting.
    Scales all new position sizes by TARGET_VOL / realized_portfolio_vol.
    """
    from unified_backtester import (
        INITIAL_CASH, SLIPPAGE,
        COOLDOWN_DAYS, SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
        Position,
    )

    strats = {s.name: s for s in strategies}
    n_dates = len(all_dates)
    multi = len(strats) > 1

    px_lookup = {}
    if price_data is not None:
        for date in all_dates:
            if date in price_data.index:
                px_lookup[date] = price_data.loc[date].to_dict()

    cash = float(INITIAL_CASH)
    positions = {}
    idle_spy_shares = 0.0
    trades = []
    cooldowns = {}
    port_vals = []

    daily_returns = []
    vol_scales = []
    daily_pos_count = []

    for i, date in enumerate(all_dates):
        spy_px = spy_prices.get(date) if spy_prices else None
        if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
            spy_px = None

        day_prices = px_lookup.get(date, {})
        cur = {"idx": i, "date": date, "n_dates": n_dates, "prices": day_prices}

        # ── Compute vol scale factor ────────────────────────────────────
        if len(daily_returns) >= VOL_LOOKBACK:
            recent_rets = np.array(daily_returns[-VOL_LOOKBACK:])
            realized_vol = np.std(recent_rets) * np.sqrt(252)
            if realized_vol > 0:
                scale = TARGET_VOL / realized_vol
                scale = max(MIN_LEVERAGE, min(scale, MAX_LEVERAGE))
            else:
                scale = 1.0
        else:
            scale = 1.0

        vol_scales.append(scale)

        # ── 1. Update peak prices ───────────────────────────────────────
        for sym, pos in positions.items():
            if pos.price_based and sym in day_prices:
                px = day_prices[sym]
                if not np.isnan(px) and px > pos.peak_price:
                    pos.peak_price = px

        # ── 2. Close expiring positions ─────────────────────────────────
        to_close = []
        for sym, pos in positions.items():
            strat = strats[pos.strategy_name]
            should_exit, reason = strat.check_exit(pos, cur)
            if should_exit:
                to_close.append((sym, reason))

        for sym, exit_reason in to_close:
            pos = positions.pop(sym)
            if pos.price_based:
                close_px = day_prices.get(sym, pos.entry_price)
                if np.isnan(close_px):
                    close_px = pos.entry_price
                actual_ret = (close_px / pos.entry_price) - 1.0
                gross = pos.cost * (1.0 + actual_ret)
            else:
                gross = pos.cost * (1.0 + pos.fwd_ret)
            net = gross * (1.0 - SLIPPAGE)
            cash += net
            trades.append((net - pos.cost) / pos.cost)

            if multi:
                cooldowns[(sym, pos.strategy_name)] = i + COOLDOWN_DAYS

        # ── 3. Gather signals ───────────────────────────────────────────
        all_signals = []
        for strat in strats.values():
            all_signals.extend(strat.generate_signals(date, None))

        held = set(positions.keys())
        all_signals = [s for s in all_signals
                       if s.symbol not in held
                       and (s.price_based or not np.isnan(s.fwd_ret))]

        if multi:
            all_signals = [s for s in all_signals
                           if cooldowns.get((s.symbol, s.strategy_name), -1) <= i]

        # ── 4. Resolve overlaps ─────────────────────────────────────────
        seen_syms = set()
        resolved = []
        for sig in all_signals:
            if sig.symbol not in seen_syms:
                seen_syms.add(sig.symbol)
                resolved.append(sig)
        resolved.sort(key=lambda s: s.confidence, reverse=True)

        # ── 5. Execute buys (vol-scaled) ────────────────────────────────
        max_slots = slot_config.max_positions - len(positions)

        if spy_px and idle_spy_shares > 0 and resolved and max_slots > 0:
            proceeds = idle_spy_shares * spy_px * (1.0 - SLIPPAGE)
            cash += proceeds
            idle_spy_shares = 0.0

        for sig_idx, sig in enumerate(resolved):
            if sig_idx >= max_slots:
                break

            counts = {}
            for p in positions.values():
                counts[p.strategy_name] = counts.get(p.strategy_name, 0) + 1
            if slot_config.available_for(sig.strategy_name, counts, len(positions)) <= 0:
                continue

            entry_px = 0.0
            if sig.price_based:
                entry_px = day_prices.get(sig.symbol, np.nan)
                if np.isnan(entry_px) or entry_px <= 0:
                    continue

            strat = strats[sig.strategy_name]
            port_est = cash
            for p in positions.values():
                if p.price_based:
                    px = day_prices.get(p.symbol, p.entry_price)
                    port_est += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                else:
                    port_est += p.cost
            target = strat.get_position_size(sig, port_est)

            # VOL TARGETING: scale position size
            target *= scale

            cost = min(target, cash * 0.95)
            if cost < 50.0:
                continue

            cash -= cost * (1.0 + SLIPPAGE)
            exit_idx = min(i + hold_days, n_dates - 1)

            positions[sig.symbol] = Position(
                symbol=sig.symbol,
                strategy_name=sig.strategy_name,
                cost=cost,
                entry_idx=i,
                exit_idx=exit_idx,
                fwd_ret=sig.fwd_ret,
                confidence=sig.confidence,
                price_based=sig.price_based,
                entry_price=entry_px,
                peak_price=entry_px,
            )

        # ── 6. Park idle cash in SPY ────────────────────────────────────
        if spy_px:
            pos_val = 0.0
            for p in positions.values():
                if p.price_based:
                    px = day_prices.get(p.symbol, p.entry_price)
                    pos_val += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                else:
                    pos_val += p.cost * (1.0 + p.fwd_ret * (i - p.entry_idx) / hold_days)
            est_port = cash + idle_spy_shares * spy_px + pos_val
            reserved = est_port * SPY_RESERVE_PCT
            idle_cash = cash - reserved
            if idle_cash > est_port * SPY_THRESHOLD_PCT:
                invest = min(idle_cash * SPY_INVEST_PCT, cash * 0.95)
                new_shares = invest / spy_px
                cash -= invest * (1.0 + SLIPPAGE)
                idle_spy_shares += new_shares

        # ── 7. Mark-to-market ───────────────────────────────────────────
        port_val = cash + (idle_spy_shares * spy_px if spy_px else 0)
        for pos in positions.values():
            if pos.price_based:
                px = day_prices.get(pos.symbol, pos.entry_price)
                port_val += pos.cost * (px / pos.entry_price if pos.entry_price > 0 else 1.0)
            else:
                days_held = i - pos.entry_idx
                interp_ret = pos.fwd_ret * days_held / hold_days
                port_val += pos.cost * (1.0 + interp_ret)
        port_vals.append(port_val)

        # ── Track daily return ──────────────────────────────────────────
        if len(port_vals) >= 2:
            daily_ret = (port_vals[-1] / port_vals[-2]) - 1.0
        else:
            daily_ret = 0.0
        daily_returns.append(daily_ret)
        daily_pos_count.append(len(positions))

    series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates))

    diag = {
        "daily_pos_count": daily_pos_count,
        "vol_scales": vol_scales,
        "daily_returns": daily_returns,
    }
    return series, [float(t) for t in trades], diag


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log(f"  V4 BACKTEST — Volatility Targeting")
    log(f"  Target Vol: {TARGET_VOL*100:.0f}%  |  Leverage: [{MIN_LEVERAGE:.1f}, {MAX_LEVERAGE:.1f}]")
    log(f"  Lookback: {VOL_LOOKBACK} days  |  Top-N: {TOP_N_PICKS}")
    log("=" * 70)

    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS, POSITION_PCT,
        MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from strategy_base import Strategy, Signal, Position
    from backtest_ml import calc_metrics, calc_alpha_beta

    # ── Load V4 predictions ─────────────────────────────────────────────
    preds_df = pd.read_parquet(PRED_FILE)
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df = preds_df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    log(f"\n  Predictions: {len(preds_df):,} rows  |  {len(universe_syms)} symbols  |  {years:.1f} years")
    log(f"  Date range: {pd.Timestamp(all_dates[0]).date()} → {pd.Timestamp(all_dates[-1]).date()}")

    # ── V4 ML Strategy: top-N selection ─────────────────────────────────

    class MLV4Strategy(Strategy):
        """Top-N cross-sectional ranking strategy."""
        def __init__(self, predictions_df, top_n=TOP_N_PICKS, position_pct=POSITION_PCT):
            self._top_n = top_n
            self._position_pct = position_pct
            self._signals_by_date = {}
            self._build_lookup(predictions_df)

        @property
        def name(self):
            return "ml_medium"

        def _build_lookup(self, df):
            for date, grp in df.groupby("date"):
                sp500_grp = grp[grp["in_sp500"] == True]
                if sp500_grp.empty:
                    continue
                top = sp500_grp.nlargest(self._top_n, "prob_v4")
                self._signals_by_date[date] = [
                    (row.symbol, row.prob_v4, row.fwd_ret)
                    for row in top.itertuples(index=False)
                ]

        def generate_signals(self, date, universe_data):
            raw = self._signals_by_date.get(date, [])
            return [
                Signal(symbol=sym, confidence=prob,
                       strategy_name=self.name, fwd_ret=fwd_ret)
                for sym, prob, fwd_ret in raw
            ]

        def check_exit(self, position, current_data):
            if current_data["idx"] >= position.exit_idx:
                return True, "hold_complete"
            return False, ""

        def get_position_size(self, signal, portfolio_value):
            ml_mult = min(1.0, max(0.60, signal.confidence * 1.6 - 0.28))
            return portfolio_value * self._position_pct * ml_mult

    # ── Fetch OHLCV ─────────────────────────────────────────────────────
    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    all_syms = list(set(["SPY"] + universe_syms))

    log(f"  Fetching OHLCV for {len(all_syms)} symbols ...")
    raw = yf.download(all_syms, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume = None
    if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
        volume = raw["Volume"]
        volume.index = pd.to_datetime(volume.index).tz_localize(None)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")

    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH
    spy_dict = close_aligned["SPY"].to_dict()

    # ── Build strategies ────────────────────────────────────────────────
    ml_v4 = MLV4Strategy(preds_df, top_n=TOP_N_PICKS)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ══════════════════════════════════════════════════════════════════════
    #  Run V4 + VolTarget
    # ══════════════════════════════════════════════════════════════════════
    log(f"\n  Running V4 Path B WITH vol targeting ...")
    vals_vt, trades_vt, diag_vt = instrumented_run_voltarget(
        [ml_v4, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="V4 + VolTarget")

    m_vt = calc_metrics(vals_vt, trades_vt, years, "V4 + VolTarget")
    alpha_vt, beta_vt = calc_alpha_beta(vals_vt, spy_bh.reindex(vals_vt.index, method="ffill"))
    m_vt["alpha"] = alpha_vt
    m_vt["beta"] = beta_vt

    # ══════════════════════════════════════════════════════════════════════
    #  Run V4 raw (no vol targeting, for direct comparison)
    # ══════════════════════════════════════════════════════════════════════
    log(f"  Running V4 Path B WITHOUT vol targeting (baseline) ...")
    from diagnose_combined import instrumented_run
    ml_v4_raw = MLV4Strategy(preds_df, top_n=TOP_N_PICKS)
    mom_strat_raw = MomentumStrategy(close, volume_data=volume)
    mr_strat_raw = MeanReversionStrategy(close, volume_data=volume)

    vals_raw, trades_raw, diag_raw = instrumented_run(
        [ml_v4_raw, mom_strat_raw, mr_strat_raw], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="V4 raw")

    m_raw = calc_metrics(vals_raw, trades_raw, years, "V4 raw")
    alpha_raw, beta_raw = calc_alpha_beta(vals_raw, spy_bh.reindex(vals_raw.index, method="ffill"))
    m_raw["alpha"] = alpha_raw
    m_raw["beta"] = beta_raw

    # ══════════════════════════════════════════════════════════════════════
    #  Vol Scale Analysis
    # ══════════════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("VOLATILITY SCALE ANALYSIS")
    log(f"{'='*70}")

    scales = np.array(diag_vt["vol_scales"])
    log(f"\n  Scale factor statistics:")
    log(f"    Mean:   {np.mean(scales):.3f}")
    log(f"    Median: {np.median(scales):.3f}")
    log(f"    Min:    {np.min(scales):.3f}")
    log(f"    Max:    {np.max(scales):.3f}")
    log(f"    Std:    {np.std(scales):.3f}")

    # Scale distribution
    n_below_1 = np.sum(scales < 1.0)
    n_above_1 = np.sum(scales >= 1.0)
    n_at_min = np.sum(scales <= MIN_LEVERAGE + 0.01)
    n_at_max = np.sum(scales >= MAX_LEVERAGE - 0.01)
    log(f"\n  Scale distribution:")
    log(f"    Days scale < 1.0 (reducing):    {n_below_1:>5} ({n_below_1/len(scales)*100:.1f}%)")
    log(f"    Days scale >= 1.0 (increasing):  {n_above_1:>5} ({n_above_1/len(scales)*100:.1f}%)")
    log(f"    Days at MIN ({MIN_LEVERAGE}):          {n_at_min:>5} ({n_at_min/len(scales)*100:.1f}%)")
    log(f"    Days at MAX ({MAX_LEVERAGE}):          {n_at_max:>5} ({n_at_max/len(scales)*100:.1f}%)")

    # Leverage level buckets
    buckets = [(0.3, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25, 1.5)]
    log(f"\n  Leverage level distribution:")
    for lo, hi in buckets:
        n = np.sum((scales >= lo) & (scales < hi))
        log(f"    [{lo:.2f}, {hi:.2f}): {n:>5} days ({n/len(scales)*100:.1f}%)")
    n_max = np.sum(scales >= 1.5)
    log(f"    [1.50, 1.50]: {n_max:>5} days ({n_max/len(scales)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    #  Year-by-Year Returns
    # ══════════════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("YEAR-BY-YEAR RETURNS")
    log(f"{'='*70}")

    vals_vt.index = pd.to_datetime(vals_vt.index)
    vals_raw.index = pd.to_datetime(vals_raw.index)
    spy_aligned = spy_bh.reindex(vals_vt.index, method="ffill")

    log(f"\n  {'Year':<6} {'V4+VolTgt':>10} {'V4 raw':>10} {'SPY':>10} {'VT Alpha':>10}")
    log(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    unique_years = sorted(set(vals_vt.index.year))
    for yr in unique_years:
        yr_vt = vals_vt[vals_vt.index.year == yr]
        yr_raw = vals_raw[vals_raw.index.year == yr]
        yr_spy = spy_aligned[spy_aligned.index.year == yr]

        if len(yr_vt) < 2:
            continue

        vt_ret = yr_vt.iloc[-1] / yr_vt.iloc[0] - 1
        raw_ret = yr_raw.iloc[-1] / yr_raw.iloc[0] - 1 if len(yr_raw) >= 2 else np.nan
        spy_ret = yr_spy.iloc[-1] / yr_spy.iloc[0] - 1 if len(yr_spy) >= 2 and yr_spy.iloc[0] > 0 else np.nan
        vt_alpha = vt_ret - spy_ret if not np.isnan(spy_ret) else np.nan

        log(f"  {yr:<6} {vt_ret*100:>9.2f}% {raw_ret*100:>9.2f}% {spy_ret*100:>9.2f}% {vt_alpha*100:>+9.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    #  Drawdown Analysis — Key Periods
    # ══════════════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("DRAWDOWN ANALYSIS — KEY PERIODS")
    log(f"{'='*70}")

    def max_dd_period(vals, start_date, end_date):
        mask = (vals.index >= start_date) & (vals.index <= end_date)
        segment = vals[mask]
        if len(segment) < 2:
            return np.nan
        peak = segment.cummax()
        dd = (segment - peak) / peak
        return dd.min()

    periods = [
        ("2020-02-19", "2020-03-23", "COVID crash (Feb-Mar 2020)"),
        ("2022-01-03", "2022-10-12", "2022 bear market"),
        ("2018-09-20", "2018-12-24", "Q4 2018 selloff"),
        ("2015-07-20", "2016-02-11", "2015-16 correction"),
        ("2023-07-31", "2023-10-27", "Q3 2023 rate scare"),
    ]

    log(f"\n  {'Period':<30} {'V4+VolTgt':>12} {'V4 raw':>12} {'Improvement':>12}")
    log(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    for start_d, end_d, label in periods:
        dd_vt = max_dd_period(vals_vt, start_d, end_d)
        dd_raw = max_dd_period(vals_raw, start_d, end_d)
        if not np.isnan(dd_vt) and not np.isnan(dd_raw):
            improvement = dd_vt - dd_raw  # less negative = better
            log(f"  {label:<30} {dd_vt*100:>11.2f}% {dd_raw*100:>11.2f}% {improvement*100:>+11.2f}%")
        else:
            log(f"  {label:<30} {'N/A':>12} {'N/A':>12}")

    # Overall max DD timing
    peak_vt = vals_vt.cummax()
    dd_vt_series = (vals_vt - peak_vt) / peak_vt
    worst_dd_date = dd_vt_series.idxmin()
    # Find when the peak before the worst DD occurred
    peak_before_dd = vals_vt[:worst_dd_date].idxmax()
    log(f"\n  V4+VolTgt worst drawdown:")
    log(f"    Peak date:     {peak_before_dd.date()}")
    log(f"    Trough date:   {worst_dd_date.date()}")
    log(f"    Max DD:        {dd_vt_series.min()*100:.2f}%")

    peak_raw = vals_raw.cummax()
    dd_raw_series = (vals_raw - peak_raw) / peak_raw
    worst_raw_date = dd_raw_series.idxmin()
    peak_before_raw = vals_raw[:worst_raw_date].idxmax()
    log(f"\n  V4 raw worst drawdown:")
    log(f"    Peak date:     {peak_before_raw.date()}")
    log(f"    Trough date:   {worst_raw_date.date()}")
    log(f"    Max DD:        {dd_raw_series.min()*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    #  Comparison Table
    # ══════════════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("FULL COMPARISON (4 VARIANTS)")
    log(f"{'='*70}")

    cagr = m_vt["cagr"]
    sharpe = m_vt["sharpe"]
    sortino = m_vt["sortino"]
    max_dd = m_vt["max_dd"]

    log(f"\n  {'Metric':<20} {'V2 prod':>12} {'V4 raw':>12} {'V4+VolTgt':>12} {'V4VT vs V2':>12}")
    log(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    log(f"  {'CAGR':<20} {V2_CAGR*100:>11.2f}% {m_raw['cagr']*100:>11.2f}% {cagr*100:>11.2f}% {(cagr-V2_CAGR)*100:>+11.2f}%")
    log(f"  {'Sharpe':<20} {V2_SHARPE:>12.3f} {m_raw['sharpe']:>12.3f} {sharpe:>12.3f} {sharpe-V2_SHARPE:>+12.3f}")
    log(f"  {'Sortino':<20} {'—':>12} {m_raw['sortino']:>12.3f} {sortino:>12.3f}")
    log(f"  {'Max Drawdown':<20} {V2_DD*100:>11.1f}% {m_raw['max_dd']*100:>11.1f}% {max_dd*100:>11.1f}% {(max_dd-V2_DD)*100:>+11.1f}%")
    log(f"  {'Win Rate':<20} {'—':>12} {m_raw['win_rate']*100:>11.1f}% {m_vt['win_rate']*100:>11.1f}%")
    log(f"  {'Trades':<20} {'—':>12} {m_raw['n_trades']:>12,} {m_vt['n_trades']:>12,}")
    log(f"  {'Alpha vs SPY':<20} {'—':>12} {alpha_raw*100:>11.2f}% {alpha_vt*100:>11.2f}%")
    log(f"  {'Beta':<20} {'—':>12} {beta_raw:>12.3f} {beta_vt:>12.3f}")
    log(f"  {'Final Value':<20} {'—':>12} ${m_raw['final_value']:>11,.0f} ${m_vt['final_value']:>11,.0f}")
    log(f"  {'Profit Factor':<20} {'—':>12} {m_raw['profit_factor']:>12.3f} {m_vt['profit_factor']:>12.3f}")

    # ══════════════════════════════════════════════════════════════════════
    #  Deployment Decision
    # ══════════════════════════════════════════════════════════════════════
    log(f"\n{'='*70}")
    log("DEPLOYMENT DECISION")
    log(f"{'='*70}")

    pass_cagr = cagr >= PASS_CAGR
    pass_sharpe = sharpe >= PASS_SHARPE
    pass_dd = max_dd >= PASS_DD

    log(f"\n  Pass criteria:")
    log(f"    [{'PASS' if pass_cagr else 'FAIL'}] CAGR >= {PASS_CAGR*100:.0f}%:     {cagr*100:.2f}%")
    log(f"    [{'PASS' if pass_sharpe else 'FAIL'}] Sharpe >= {PASS_SHARPE:.2f}:  {sharpe:.3f}")
    log(f"    [{'PASS' if pass_dd else 'FAIL'}] Max DD >= {PASS_DD*100:.0f}%:   {max_dd*100:.1f}%")

    n_pass = sum([pass_cagr, pass_sharpe, pass_dd])

    log(f"\n  {'='*50}")
    if n_pass == 3:
        log(f"  VERDICT: STRONG DEPLOY CANDIDATE ({n_pass}/3)")
    elif n_pass >= 2:
        log(f"  VERDICT: DEPLOY WITH ACCEPTANCE OF TRADEOFF ({n_pass}/3)")
    else:
        log(f"  VERDICT: ITERATE PARAMETERS ({n_pass}/3)")
    log(f"  {'='*50}")
    log(f"\n  *** Do NOT deploy — report for human review ***")
    log(f"  *** Current model.lgb (V2) remains in production ***")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
