#!/usr/bin/env python3
"""
V3 Backtest with Volatility Targeting
=======================================
Scales ALL new position sizes inverse to realized portfolio volatility.
Uses existing predictions.parquet — no retraining.

Based on Moreira & Muir (2017): vol-managed portfolios improve Sharpe
by reducing exposure during high-vol regimes and increasing during low-vol.
"""

import sys
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR   = Path(__file__).resolve().parent / "data"
PRED_FILE  = DATA_DIR / "predictions.parquet"

# ── Vol targeting parameters ─────────────────────────────────────────────────
TARGET_VOL    = 0.15    # 15% annualized target
MAX_LEVERAGE  = 1.5     # cap upside scaling
MIN_LEVERAGE  = 0.3     # cap downside scaling
VOL_LOOKBACK  = 20      # trading days for realized vol

# ML threshold
THRESHOLD = 0.55

# Baselines
V3_CAGR, V3_SHARPE, V3_DD = 0.3280, 1.520, -0.3405
V2_CAGR, V2_SHARPE, V2_DD = 0.2559, 1.646, -0.2095


def log(msg: str):
    print(msg, flush=True)


def instrumented_run_voltarget(strategies, slot_config, all_dates, spy_prices,
                                price_data, hold_days, label):
    """
    Replica of diagnose_combined.instrumented_run() with volatility targeting.
    Scales all new position sizes by TARGET_VOL / realized_portfolio_vol.
    """
    from unified_backtester import (
        INITIAL_CASH, SLIPPAGE, POSITION_PCT,
        COOLDOWN_DAYS, SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
        Position,
    )

    strats = {s.name: s for s in strategies}
    n_dates = len(all_dates)
    multi = len(strats) > 1

    # Price lookup
    px_lookup = {}
    if price_data is not None:
        for date in all_dates:
            if date in price_data.index:
                px_lookup[date] = price_data.loc[date].to_dict()

    # ── Run state ────────────────────────────────────────────────────────
    cash = float(INITIAL_CASH)
    positions = {}
    idle_spy_shares = 0.0
    trades = []
    cooldowns = {}
    port_vals = []

    # ── Vol targeting state ──────────────────────────────────────────────
    daily_returns = []       # portfolio daily returns for vol calculation
    vol_scales = []          # daily scale factor history
    daily_pos_count = []

    for i, date in enumerate(all_dates):
        spy_px = spy_prices.get(date) if spy_prices else None
        if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
            spy_px = None

        day_prices = px_lookup.get(date, {})
        cur = {"idx": i, "date": date, "n_dates": n_dates, "prices": day_prices}

        # ── Compute vol scale factor (BEFORE any trading) ────────────────
        if len(daily_returns) >= VOL_LOOKBACK:
            recent_rets = np.array(daily_returns[-VOL_LOOKBACK:])
            realized_vol = np.std(recent_rets) * np.sqrt(252)
            if realized_vol > 0:
                scale = TARGET_VOL / realized_vol
                scale = max(MIN_LEVERAGE, min(scale, MAX_LEVERAGE))
            else:
                scale = 1.0
        else:
            scale = 1.0  # not enough history yet

        vol_scales.append(scale)

        # ── 1. Update peak prices ────────────────────────────────────────
        for sym, pos in positions.items():
            if pos.price_based and sym in day_prices:
                px = day_prices[sym]
                if not np.isnan(px) and px > pos.peak_price:
                    pos.peak_price = px

        # ── 2. Close expiring positions ──────────────────────────────────
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

        # ── 3. Gather signals ────────────────────────────────────────────
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

        # ── 4. Resolve overlaps ──────────────────────────────────────────
        seen_syms = set()
        resolved = []
        for sig in all_signals:
            if sig.symbol not in seen_syms:
                seen_syms.add(sig.symbol)
                resolved.append(sig)
        resolved.sort(key=lambda s: s.confidence, reverse=True)

        # ── 5. Execute buys (with vol-scaled position sizes) ─────────────
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

            # Position sizing — strategy provides base target
            strat = strats[sig.strategy_name]
            port_est = cash
            for p in positions.values():
                if p.price_based:
                    px = day_prices.get(p.symbol, p.entry_price)
                    port_est += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                else:
                    port_est += p.cost
            target = strat.get_position_size(sig, port_est)

            # ── VOL TARGETING: scale position size ───────────────────────
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

        # ── 6. Park idle cash in SPY ─────────────────────────────────────
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

        # ── 7. Mark-to-market ────────────────────────────────────────────
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

        # ── Track daily return for vol calculation ───────────────────────
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
    log(f"  V3 BACKTEST — Volatility Targeting")
    log(f"  Target Vol: {TARGET_VOL*100:.0f}%  |  Leverage: [{MIN_LEVERAGE:.1f}, {MAX_LEVERAGE:.1f}]")
    log(f"  Lookback: {VOL_LOOKBACK} days  |  Threshold: {THRESHOLD}")
    log("=" * 70)

    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS,
        MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from backtest_utils import calc_metrics, calc_alpha_beta

    # Load predictions
    preds_df = pd.read_parquet(PRED_FILE)
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    preds_df = preds_df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])

    all_dates = sorted(preds_df["date"].unique().tolist())
    universe_syms = sorted(preds_df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    log(f"\n  Predictions: {len(preds_df):,} rows  |  {len(universe_syms)} symbols  |  {years:.1f} years")
    log(f"  Date range: {pd.Timestamp(all_dates[0]).date()} → {pd.Timestamp(all_dates[-1]).date()}")

    # Fetch OHLCV
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

    # Build strategies
    ml_strat = MLMediumStrategy(preds_df, threshold=THRESHOLD)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ── Run WITH vol targeting ───────────────────────────────────────────
    log(f"\n  Running Path B WITH vol targeting ...")
    vals_vt, trades_vt, diag_vt = instrumented_run_voltarget(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B + VolTarget")

    m_vt = calc_metrics(vals_vt, trades_vt, years, "V3 + Vol Target")
    alpha_vt, beta_vt = calc_alpha_beta(vals_vt, spy_bh.reindex(vals_vt.index, method="ffill"))
    m_vt["alpha"] = alpha_vt
    m_vt["beta"] = beta_vt

    # ── Run WITHOUT vol targeting (baseline) ─────────────────────────────
    log(f"  Running Path B WITHOUT vol targeting (baseline) ...")
    from diagnose_combined import instrumented_run
    vals_base, trades_base, diag_base = instrumented_run(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="Path B baseline")

    m_base = calc_metrics(vals_base, trades_base, years, "V3 baseline")
    alpha_base, beta_base = calc_alpha_beta(vals_base, spy_bh.reindex(vals_base.index, method="ffill"))
    m_base["alpha"] = alpha_base

    # ── Vol scale analysis ───────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("VOLATILITY SCALE ANALYSIS")
    log(f"{'='*70}")

    scales = np.array(diag_vt["vol_scales"])
    log(f"\n  Scale factor statistics:")
    log(f"    Mean:   {np.mean(scales):.3f}")
    log(f"    Median: {np.median(scales):.3f}")
    log(f"    Min:    {np.min(scales):.3f} (max caution — high vol)")
    log(f"    Max:    {np.max(scales):.3f} (max leverage — low vol)")
    log(f"    Std:    {np.std(scales):.3f}")

    # Scale distribution
    n_below_1 = np.sum(scales < 1.0)
    n_above_1 = np.sum(scales >= 1.0)
    n_at_min = np.sum(scales <= MIN_LEVERAGE + 0.01)
    n_at_max = np.sum(scales >= MAX_LEVERAGE - 0.01)
    log(f"\n  Scale distribution:")
    log(f"    Days scale < 1.0 (reducing):  {n_below_1:>5} ({n_below_1/len(scales)*100:.1f}%)")
    log(f"    Days scale >= 1.0 (increasing): {n_above_1:>5} ({n_above_1/len(scales)*100:.1f}%)")
    log(f"    Days at MIN ({MIN_LEVERAGE}):        {n_at_min:>5} ({n_at_min/len(scales)*100:.1f}%)")
    log(f"    Days at MAX ({MAX_LEVERAGE}):        {n_at_max:>5} ({n_at_max/len(scales)*100:.1f}%)")

    # ── Year-by-year returns ─────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("YEAR-BY-YEAR RETURNS")
    log(f"{'='*70}")

    vals_vt.index = pd.to_datetime(vals_vt.index)
    vals_base.index = pd.to_datetime(vals_base.index)
    spy_aligned = spy_bh.reindex(vals_vt.index, method="ffill")

    log(f"\n  {'Year':<6} {'VolTarget':>10} {'No VT':>10} {'SPY':>10} {'VT Alpha':>10}")
    log(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    unique_years = sorted(set(vals_vt.index.year))
    for yr in unique_years:
        yr_mask = vals_vt.index.year == yr
        yr_vt = vals_vt[yr_mask]
        yr_base = vals_base[vals_base.index.year == yr]
        yr_spy = spy_aligned[yr_mask]

        if len(yr_vt) < 2:
            continue

        vt_ret = yr_vt.iloc[-1] / yr_vt.iloc[0] - 1
        base_ret = yr_base.iloc[-1] / yr_base.iloc[0] - 1 if len(yr_base) >= 2 else np.nan
        spy_ret = yr_spy.iloc[-1] / yr_spy.iloc[0] - 1 if len(yr_spy) >= 2 and yr_spy.iloc[0] > 0 else np.nan
        vt_alpha = vt_ret - spy_ret if not np.isnan(spy_ret) else np.nan

        log(f"  {yr:<6} {vt_ret*100:>9.2f}% {base_ret*100:>9.2f}% {spy_ret*100:>9.2f}% {vt_alpha*100:>+9.2f}%")

    # ── Drawdown analysis for key periods ────────────────────────────────
    log(f"\n{'='*70}")
    log("DRAWDOWN ANALYSIS — KEY PERIODS")
    log(f"{'='*70}")

    def max_dd_period(vals, start_date, end_date, label):
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
    ]

    log(f"\n  {'Period':<30} {'VolTarget':>12} {'No VT':>12} {'Improvement':>12}")
    log(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    for start_d, end_d, label in periods:
        dd_vt = max_dd_period(vals_vt, start_d, end_d, label)
        dd_base = max_dd_period(vals_base, start_d, end_d, label)
        if not np.isnan(dd_vt) and not np.isnan(dd_base):
            improvement = dd_vt - dd_base  # less negative = better
            log(f"  {label:<30} {dd_vt*100:>11.2f}% {dd_base*100:>11.2f}% {improvement*100:>+11.2f}%")
        else:
            log(f"  {label:<30} {'N/A':>12} {'N/A':>12}")

    # ── Comparison table ─────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("COMPARISON")
    log(f"{'='*70}")

    cagr = m_vt["cagr"]
    sharpe = m_vt["sharpe"]
    sortino = m_vt["sortino"]
    max_dd = m_vt["max_dd"]
    win_rate = m_vt["win_rate"]
    n_trades = m_vt["n_trades"]

    log(f"\n  {'Metric':<20} {'V3+VolTgt':>12} {'V3 (no VT)':>12} {'V2 prod':>12}")
    log(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")
    log(f"  {'CAGR':<20} {cagr*100:>11.2f}% {V3_CAGR*100:>11.2f}% {V2_CAGR*100:>11.2f}%")
    log(f"  {'Sharpe':<20} {sharpe:>12.3f} {V3_SHARPE:>12.3f} {V2_SHARPE:>12.3f}")
    log(f"  {'Sortino':<20} {sortino:>12.3f}")
    log(f"  {'Max Drawdown':<20} {max_dd*100:>11.2f}% {V3_DD*100:>11.2f}% {V2_DD*100:>11.2f}%")
    log(f"  {'Win Rate':<20} {win_rate*100:>11.2f}%")
    log(f"  {'Trades':<20} {n_trades:>12,}")
    log(f"  {'Alpha':<20} {alpha_vt*100:>11.2f}%")
    log(f"  {'Beta':<20} {beta_vt:>12.3f}")
    log(f"  {'Avg Vol Scale':<20} {np.mean(scales):>12.3f}")

    # ── Sharpe improvement breakdown ─────────────────────────────────────
    log(f"\n  Sharpe improvement breakdown:")
    log(f"    V3 no VT:     {m_base['sharpe']:.3f}")
    log(f"    V3 + VolTgt:  {sharpe:.3f}")
    log(f"    Delta:        {sharpe - m_base['sharpe']:+.3f}")

    # ── Decision ─────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("DEPLOYMENT DECISION")
    log(f"{'='*70}")

    pass_cagr = cagr >= 0.27
    pass_sharpe = sharpe >= 1.65
    pass_dd = max_dd >= -0.27

    log(f"\n  Criteria check:")
    log(f"    CAGR >= 27%:     {cagr*100:.2f}% {'PASS' if pass_cagr else 'FAIL'}")
    log(f"    Sharpe >= 1.65:  {sharpe:.3f}  {'PASS' if pass_sharpe else 'FAIL'}")
    log(f"    Max DD >= -27%:  {max_dd*100:.2f}% {'PASS' if pass_dd else 'FAIL'}")

    n_pass = sum([pass_cagr, pass_sharpe, pass_dd])

    if n_pass == 3:
        verdict = "READY TO DEPLOY V3 + VOL TARGETING"
    elif n_pass >= 1:
        verdict = "PROMISING, ITERATE PARAMETERS"
    else:
        verdict = "VOL TARGETING ALONE INSUFFICIENT, TRY ENSEMBLE NEXT"

    log(f"\n  {'='*50}")
    log(f"  VERDICT: {verdict}")
    log(f"  Criteria passed: {n_pass}/3")
    log(f"  {'='*50}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
