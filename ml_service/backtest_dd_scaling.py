#!/usr/bin/env python3
"""
V3 Backtest with Drawdown-Based Scaling
=========================================
Scales new position sizes based on portfolio drawdown depth.
Preserves alpha during high-vol periods (where this strategy profits)
while cutting exposure only when the portfolio itself is underwater.

Uses existing predictions.parquet — no retraining.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR  = Path(__file__).resolve().parent / "data"
PRED_FILE = DATA_DIR / "predictions.parquet"

# ── Drawdown scaling thresholds ──────────────────────────────────────────────
DD_SCALES = [
    (-0.10, 1.00),   # DD better than -10%: full sizing
    (-0.15, 0.85),   # DD -10% to -15%: trim 15%
    (-0.20, 0.70),   # DD -15% to -20%: trim 30%
    (-0.25, 0.50),   # DD -20% to -25%: half exposure
    (None,  0.35),   # DD worse than -25%: defensive
]

THRESHOLD = 0.55

# Baselines
V3_CAGR, V3_SHARPE, V3_DD = 0.3280, 1.520, -0.3405
VT_CAGR, VT_SHARPE, VT_DD = 0.2865, 1.469, -0.2580


def log(msg: str):
    print(msg, flush=True)


def dd_scale(current_dd: float) -> float:
    """Return position scale factor based on current portfolio drawdown."""
    for threshold, scale in DD_SCALES:
        if threshold is None:
            return scale
        if current_dd > threshold:
            return scale
    return DD_SCALES[-1][1]


def instrumented_run_dd(strategies, slot_config, all_dates, spy_prices,
                        price_data, hold_days):
    """
    Backtest loop with drawdown-based position scaling.
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

    # Drawdown tracking
    peak_value = INITIAL_CASH
    scales_used = []
    daily_pos_count = []
    daily_dd = []

    for i, date in enumerate(all_dates):
        spy_px = spy_prices.get(date) if spy_prices else None
        if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
            spy_px = None

        day_prices = px_lookup.get(date, {})
        cur = {"idx": i, "date": date, "n_dates": n_dates, "prices": day_prices}

        # ── Compute drawdown-based scale ─────────────────────────────────
        if port_vals:
            current_val = port_vals[-1]
            peak_value = max(peak_value, current_val)
            current_dd = (current_val / peak_value) - 1.0
        else:
            current_dd = 0.0

        scale = dd_scale(current_dd)
        scales_used.append(scale)
        daily_dd.append(current_dd)

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

        # ── 5. Execute buys (with DD-scaled position sizes) ──────────────
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

            # ── DRAWDOWN SCALING ─────────────────────────────────────────
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
        daily_pos_count.append(len(positions))

    series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates))
    diag = {
        "daily_pos_count": daily_pos_count,
        "scales_used": scales_used,
        "daily_dd": daily_dd,
    }
    return series, [float(t) for t in trades], diag


def main():
    t0 = time.perf_counter()
    log("=" * 70)
    log("  V3 BACKTEST — Drawdown-Based Scaling")
    log("  Scale: full @ <10% DD, trim @ 10-25%, defensive @ >25%")
    log(f"  Threshold: {THRESHOLD}  |  No upside leverage")
    log("=" * 70)

    from unified_backtester import (
        INITIAL_CASH, HOLD_DAYS,
        MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
        SLOT_ML_MOM_MR,
    )
    from backtest_utils import calc_metrics, calc_alpha_beta

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

    ml_strat = MLMediumStrategy(preds_df, threshold=THRESHOLD)
    mom_strat = MomentumStrategy(close, volume_data=volume)
    mr_strat = MeanReversionStrategy(close, volume_data=volume)

    # ── Run WITH DD scaling ──────────────────────────────────────────────
    log(f"\n  Running Path B WITH drawdown scaling ...")
    vals_dd, trades_dd, diag_dd = instrumented_run_dd(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS)

    m_dd = calc_metrics(vals_dd, trades_dd, years, "V3 + DD Scale")
    alpha_dd, beta_dd = calc_alpha_beta(vals_dd, spy_bh.reindex(vals_dd.index, method="ffill"))
    m_dd["alpha"] = alpha_dd
    m_dd["beta"] = beta_dd

    # ── Run WITHOUT scaling (baseline) ───────────────────────────────────
    log(f"  Running Path B WITHOUT scaling (baseline) ...")
    from diagnose_combined import instrumented_run
    vals_base, trades_base, diag_base = instrumented_run(
        [ml_strat, mom_strat, mr_strat], SLOT_ML_MOM_MR, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="V3 baseline")

    m_base = calc_metrics(vals_base, trades_base, years, "V3 baseline")
    alpha_base, _ = calc_alpha_beta(vals_base, spy_bh.reindex(vals_base.index, method="ffill"))
    m_base["alpha"] = alpha_base

    # ── Scale factor analysis ────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("DRAWDOWN SCALE ANALYSIS")
    log(f"{'='*70}")

    scales = np.array(diag_dd["scales_used"])
    dds = np.array(diag_dd["daily_dd"])

    log(f"\n  Scale factor statistics:")
    log(f"    Mean:   {np.mean(scales):.3f}")
    log(f"    Median: {np.median(scales):.3f}")

    # Time at each scale level
    log(f"\n  Time spent at each scale level:")
    for threshold, scale_val in DD_SCALES:
        n_at = np.sum(np.isclose(scales, scale_val))
        label = f"DD > {threshold*100:.0f}%" if threshold is not None else f"DD <= -25%"
        log(f"    Scale {scale_val:.2f} ({label:<16}): {n_at:>5} days ({n_at/len(scales)*100:.1f}%)")

    log(f"\n  Drawdown statistics:")
    log(f"    Mean DD:    {np.mean(dds)*100:.2f}%")
    log(f"    Worst DD:   {np.min(dds)*100:.2f}%")
    log(f"    Days in DD: {np.sum(dds < -0.01):>5} ({np.sum(dds < -0.01)/len(dds)*100:.1f}%)")
    log(f"    Days > -5%: {np.sum(dds > -0.05):>5} ({np.sum(dds > -0.05)/len(dds)*100:.1f}%)")

    # ── Year-by-year returns ─────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("YEAR-BY-YEAR RETURNS")
    log(f"{'='*70}")

    vals_dd.index = pd.to_datetime(vals_dd.index)
    vals_base.index = pd.to_datetime(vals_base.index)
    spy_aligned = spy_bh.reindex(vals_dd.index, method="ffill")

    log(f"\n  {'Year':<6} {'DD Scale':>10} {'No Scale':>10} {'SPY':>10} {'DD Alpha':>10}")
    log(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    unique_years = sorted(set(vals_dd.index.year))
    for yr in unique_years:
        yr_dd = vals_dd[vals_dd.index.year == yr]
        yr_base = vals_base[vals_base.index.year == yr]
        yr_spy = spy_aligned[vals_dd.index.year == yr]

        if len(yr_dd) < 2:
            continue

        dd_ret = yr_dd.iloc[-1] / yr_dd.iloc[0] - 1
        base_ret = yr_base.iloc[-1] / yr_base.iloc[0] - 1 if len(yr_base) >= 2 else np.nan
        spy_ret = yr_spy.iloc[-1] / yr_spy.iloc[0] - 1 if len(yr_spy) >= 2 and yr_spy.iloc[0] > 0 else np.nan
        dd_alpha = dd_ret - spy_ret if not np.isnan(spy_ret) else np.nan

        log(f"  {yr:<6} {dd_ret*100:>9.2f}% {base_ret*100:>9.2f}% {spy_ret*100:>9.2f}% {dd_alpha*100:>+9.2f}%")

    # ── Drawdown comparison for key periods ──────────────────────────────
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
    ]

    log(f"\n  {'Period':<30} {'DD Scale':>12} {'No Scale':>12} {'Improvement':>12}")
    log(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    for start_d, end_d, label in periods:
        dd_s = max_dd_period(vals_dd, start_d, end_d)
        dd_b = max_dd_period(vals_base, start_d, end_d)
        if not np.isnan(dd_s) and not np.isnan(dd_b):
            improvement = dd_s - dd_b
            log(f"  {label:<30} {dd_s*100:>11.2f}% {dd_b*100:>11.2f}% {improvement*100:>+11.2f}%")
        else:
            log(f"  {label:<30} {'N/A':>12} {'N/A':>12}")

    # ── Comparison table ─────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("COMPARISON")
    log(f"{'='*70}")

    cagr = m_dd["cagr"]
    sharpe = m_dd["sharpe"]
    sortino = m_dd["sortino"]
    max_dd = m_dd["max_dd"]
    win_rate = m_dd["win_rate"]
    n_trades = m_dd["n_trades"]

    log(f"\n  {'Metric':<20} {'V3+DDscale':>12} {'V3 raw':>12} {'V3+VolTgt':>12}")
    log(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")
    log(f"  {'CAGR':<20} {cagr*100:>11.2f}% {V3_CAGR*100:>11.2f}% {VT_CAGR*100:>11.2f}%")
    log(f"  {'Sharpe':<20} {sharpe:>12.3f} {V3_SHARPE:>12.3f} {VT_SHARPE:>12.3f}")
    log(f"  {'Sortino':<20} {sortino:>12.3f}")
    log(f"  {'Max Drawdown':<20} {max_dd*100:>11.2f}% {V3_DD*100:>11.2f}% {VT_DD*100:>11.2f}%")
    log(f"  {'Win Rate':<20} {win_rate*100:>11.2f}%")
    log(f"  {'Trades':<20} {n_trades:>12,}")
    log(f"  {'Alpha':<20} {alpha_dd*100:>11.2f}%")
    log(f"  {'Beta':<20} {beta_dd:>12.3f}")
    log(f"  {'Avg Scale':<20} {np.mean(scales):>12.3f}")

    # ── Decision ─────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    log("DEPLOYMENT DECISION")
    log(f"{'='*70}")

    pass_cagr = cagr >= 0.28
    pass_sharpe = sharpe >= 1.55
    pass_dd = max_dd >= -0.25

    log(f"\n  Criteria check:")
    log(f"    CAGR >= 28%:     {cagr*100:.2f}%  {'PASS' if pass_cagr else 'FAIL'}")
    log(f"    Sharpe >= 1.55:  {sharpe:.3f}   {'PASS' if pass_sharpe else 'FAIL'}")
    log(f"    Max DD >= -25%:  {max_dd*100:.2f}%  {'PASS' if pass_dd else 'FAIL'}")

    n_pass = sum([pass_cagr, pass_sharpe, pass_dd])

    if n_pass == 3:
        verdict = "DEPLOY CANDIDATE"
    elif n_pass == 2:
        verdict = "STRONG IMPROVEMENT, CONSIDER DEPLOY"
    else:
        verdict = "ITERATE PARAMETERS"

    log(f"\n  {'='*50}")
    log(f"  VERDICT: {verdict}")
    log(f"  Criteria passed: {n_pass}/3")
    log(f"  {'='*50}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
