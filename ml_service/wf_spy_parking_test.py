#!/usr/bin/env python3
"""
Walk-Forward Test: SPY Parking for Idle Cash
=============================================
Tests whether parking idle cash in SPY during different regime conditions
improves portfolio returns across 11 annual walk-forward periods (2015-2025).

Configs tested:
  no_parking    — idle cash earns 0% (spy_prices=None)
  always_park   — park idle cash in SPY always (current behavior)
  bull_only     — park idle cash ONLY when SPY is in BULLISH regime

Uses existing walk-forward predictions from ml_service/data/walkforward/.
Uses SLOT_LIVE_V2 (ml:5, momentum:3, max:8) for all tests.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    MegaCapStrategy, PortfolioManager, SlotConfig,
    SLOT_LIVE_V2, load_bars_cached,
    INITIAL_CASH, SLIPPAGE, HOLD_DAYS,
    Position, NEVER_BUY, COOLDOWN_DAYS, MIN_POSITION_DOLLARS,
    POSITION_PCT, SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
    IDLE_SPY_MIN_OPP_RATIO, SYMBOL_SECTOR, SECTOR_MAX_POSITIONS,
    USE_VOL_SIZING, _has_earnings_within,
)
from backtest_utils import calc_metrics, calc_alpha_beta
from wf_regime_test import compute_spy_regime

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"
OUT_DIR = DATA_DIR / "walkforward_spy_parking"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Regime-Conditional SPY Parking Portfolio Manager ─────────────────────────

class SPYParkingPortfolioManager(PortfolioManager):
    """
    Subclass that conditionally parks idle cash based on SPY regime.

    Modes:
      "none"      — never park (ignore spy_prices)
      "always"    — always park (baseline behavior)
      "bull_only" — park only when regime is BULLISH
    """

    def __init__(self, strategies, slot_config, parking_mode="always",
                 regime_series=None,
                 initial_cash=INITIAL_CASH, slippage=SLIPPAGE,
                 hold_days=HOLD_DAYS):
        super().__init__(strategies, slot_config, initial_cash, slippage, hold_days)
        self.parking_mode = parking_mode
        self.regime_series = regime_series or {}

    def run(self, all_dates, spy_prices=None, price_data=None, detail_log=False):
        """Override run to conditionally park idle cash based on regime."""
        n_dates = len(all_dates)
        multi = len(self.strategies) > 1

        import unified_backtester as _ub
        if USE_VOL_SIZING and price_data is not None:
            daily_ret_abs = price_data.pct_change().abs()
            _ub._global_atr_df = daily_ret_abs.rolling(14, min_periods=14).mean()
        else:
            _ub._global_atr_df = None

        px_lookup = {}
        if price_data is not None:
            for date in all_dates:
                if date in price_data.index:
                    px_lookup[date] = price_data.loc[date].to_dict()

        cash = float(self.initial_cash)
        positions = {}
        idle_spy_shares = 0.0
        trades = []
        cooldowns = {}
        port_vals = []
        last_spy_action_idx = -2
        parking_stats = {"parked_days": 0, "total_days": 0}

        for i, date in enumerate(all_dates):
            spy_px = spy_prices.get(date) if spy_prices else None
            if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
                spy_px = None

            day_prices = px_lookup.get(date, {})
            cur = {"idx": i, "date": date, "n_dates": n_dates,
                   "prices": day_prices}

            # Determine if parking allowed today
            regime = self.regime_series.get(date, "CAUTIOUS")
            if self.parking_mode == "none":
                allow_parking = False
            elif self.parking_mode == "always":
                allow_parking = True
            elif self.parking_mode == "bull_only":
                allow_parking = regime == "BULLISH"
            else:
                allow_parking = True

            parking_stats["total_days"] += 1
            if allow_parking and idle_spy_shares > 0:
                parking_stats["parked_days"] += 1

            # ── 1. Update peak prices ──
            for sym, pos in positions.items():
                if pos.price_based and sym in day_prices:
                    px = day_prices[sym]
                    if not np.isnan(px) and px > pos.peak_price:
                        pos.peak_price = px

            # ── 2. Close expiring positions ──
            to_close = []
            for sym, pos in positions.items():
                strat = self.strategies[pos.strategy_name]
                should_exit, _reason = strat.check_exit(pos, cur)
                if should_exit:
                    to_close.append((sym, _reason))

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
                net = gross * (1.0 - self.slippage)
                cash += net
                trades.append((net - pos.cost) / pos.cost)
                if multi:
                    cooldowns[(sym, pos.strategy_name)] = i + COOLDOWN_DAYS

            # ── 3. Gather signals ──
            all_signals = []
            for strat in self.strategies.values():
                all_signals.extend(strat.generate_signals(date, None))

            held = set(positions.keys())
            all_signals = [s for s in all_signals
                           if s.symbol not in held
                           and s.symbol not in NEVER_BUY
                           and (s.price_based or not np.isnan(s.fwd_ret))]

            if multi:
                all_signals = [s for s in all_signals
                               if cooldowns.get((s.symbol, s.strategy_name), -1) <= i]

            # ── 4. Resolve overlaps ──
            seen_syms = set()
            resolved = []
            for sig in all_signals:
                if sig.symbol not in seen_syms:
                    seen_syms.add(sig.symbol)
                    resolved.append(sig)
            resolved.sort(key=lambda s: s.confidence, reverse=True)

            # ── 5. Execute buys ──
            max_slots = self.slot_config.max_positions - len(positions)

            # Release idle SPY before buying
            if (spy_px and idle_spy_shares > 0
                    and resolved and max_slots > 0
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
                    proceeds = idle_spy_shares * spy_px * (1.0 - self.slippage)
                    cash += proceeds
                    idle_spy_shares = 0.0
                    last_spy_action_idx = i

            sector_bought_today = {}
            for sig in resolved[:max_slots]:
                counts = {}
                for p in positions.values():
                    counts[p.strategy_name] = counts.get(p.strategy_name, 0) + 1
                if self.slot_config.available_for(
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

                strat = self.strategies[sig.strategy_name]
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

                cash -= cost * (1.0 + self.slippage)
                exit_idx = min(i + self.hold_days, n_dates - 1)

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
                sector_bought_today[sector] = sector_bought_today.get(sector, 0) + 1

            # ── 6. Park idle cash in SPY (CONDITIONAL) ──
            if allow_parking and spy_px and i > last_spy_action_idx:
                pos_val = 0.0
                for p in positions.values():
                    if p.price_based:
                        px = day_prices.get(p.symbol, p.entry_price)
                        pos_val += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                    else:
                        pos_val += p.cost * (1.0 + p.fwd_ret
                                             * (i - p.entry_idx) / self.hold_days)
                est_port = cash + idle_spy_shares * spy_px + pos_val
                reserved = est_port * SPY_RESERVE_PCT
                idle_cash = cash - reserved
                if idle_cash > est_port * SPY_THRESHOLD_PCT:
                    invest = min(idle_cash * SPY_INVEST_PCT, cash * 0.95)
                    new_shares = invest / spy_px
                    cash -= invest * (1.0 + self.slippage)
                    idle_spy_shares += new_shares
                    last_spy_action_idx = i
            elif not allow_parking and idle_spy_shares > 0 and spy_px:
                # Sell parked SPY when regime changes to non-parking
                proceeds = idle_spy_shares * spy_px * (1.0 - self.slippage)
                cash += proceeds
                idle_spy_shares = 0.0
                last_spy_action_idx = i

            # ── 7. Mark-to-market ──
            port_val = cash + (idle_spy_shares * spy_px if spy_px else 0)
            for pos in positions.values():
                if pos.price_based:
                    px = day_prices.get(pos.symbol, pos.entry_price)
                    port_val += pos.cost * (px / pos.entry_price if pos.entry_price > 0 else 1.0)
                else:
                    days_held = i - pos.entry_idx
                    interp_ret = pos.fwd_ret * days_held / self.hold_days
                    port_val += pos.cost * (1.0 + interp_ret)
            port_vals.append(port_val)

        series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates))
        return series, [float(t) for t in trades], parking_stats


# ── Test Configurations ──────────────────────────────────────────────────────

PARKING_CONFIGS = {
    "no_parking":  "none",
    "always_park": "always",
    "bull_only":   "bull_only",
}


def log(msg: str):
    print(msg, flush=True)


def run_parking_backtest(preds_df, year, close, spy_full, parking_mode,
                         slot_config=None):
    """Run backtest for one year with specified SPY parking mode."""
    all_dates = sorted(preds_df["date"].unique().tolist())
    if len(all_dates) < 10:
        return None

    years_span = (all_dates[-1] - all_dates[0]).days / 365.25
    if years_span <= 0:
        years_span = len(all_dates) / 252.0

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        return None

    spy_dict = close["SPY"].to_dict()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    regime_series = compute_spy_regime(spy_full)
    regime_dict = regime_series.to_dict()

    if "prob_ensemble" in preds_df.columns and "prob" not in preds_df.columns:
        preds_df = preds_df.rename(columns={"prob_ensemble": "prob"})

    sc = slot_config or SLOT_LIVE_V2

    strategies = [
        MLMediumStrategy(preds_df, threshold=0.55, top_n=5, selection_mode="top_n"),
        MomentumStrategy(close, volume_data=None, regime_filter=False),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]

    # For no_parking, pass spy_prices=None to completely disable
    use_spy = spy_dict if parking_mode != "none" else None

    pm = SPYParkingPortfolioManager(
        strategies=strategies,
        slot_config=sc,
        parking_mode=parking_mode,
        regime_series=regime_dict,
    )
    vals, trades, parking_stats = pm.run(
        all_dates, spy_prices=use_spy, price_data=close
    )

    metrics = calc_metrics(vals, trades, years_span, f"parking_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    metrics["parking_stats"] = parking_stats

    return metrics


def generate_report(all_results):
    """Generate markdown report comparing SPY parking configs."""
    lines = []
    lines.append("# SPY Parking for Idle Cash — Walk-Forward Results")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Configurations Tested")
    lines.append("| Config | Description |")
    lines.append("|--------|------------|")
    lines.append("| no_parking | Idle cash earns 0% |")
    lines.append("| always_park | Park idle cash in SPY regardless of regime |")
    lines.append("| bull_only | Park idle cash in SPY only during BULLISH regime |")

    lines.append("\n## Current SPY Parking Parameters")
    lines.append(f"- Reserve: {SPY_RESERVE_PCT:.0%} of portfolio always kept as cash")
    lines.append(f"- Threshold: park when idle cash > {SPY_THRESHOLD_PCT:.0%} of portfolio")
    lines.append(f"- Invest: {SPY_INVEST_PCT:.0%} of idle cash into SPY")

    # Per-config results
    for config_name in PARKING_CONFIGS:
        results = all_results[config_name]
        lines.append(f"\n## Results: {config_name}")
        lines.append("")
        lines.append("| Year | CAGR | Sharpe | Max DD | Alpha | Beta | Trades | Win% |")
        lines.append("|------|------|--------|--------|-------|------|--------|------|")

        for m in results:
            lines.append(
                f"| {m['year']} "
                f"| {m['cagr']:+.1%} "
                f"| {m['sharpe']:.2f} "
                f"| {m['max_dd']:.1%} "
                f"| {'+' if m.get('alpha') and m['alpha'] > 0 else ''}{m.get('alpha', 0) or 0:.1%} "
                f"| {m.get('beta', 0) or 0:.2f} "
                f"| {m['n_trades']} "
                f"| {m['win_rate']:.1%} |"
            )

    # Comparison table
    lines.append("\n## Summary Comparison")
    lines.append("")
    lines.append("| Metric | no_parking | always_park | bull_only |")
    lines.append("|--------|-----------|------------|----------|")

    for metric_name, metric_key, fmt in [
        ("Median CAGR", "cagr", "{:+.1%}"),
        ("Mean CAGR", "cagr", "{:+.1%}"),
        ("Median Sharpe", "sharpe", "{:.2f}"),
        ("Mean Sharpe", "sharpe", "{:.2f}"),
        ("Worst Max DD", "max_dd", None),
        ("Mean Max DD", "max_dd", None),
        ("Median Alpha", "alpha", "{:+.1%}"),
        ("Positive CAGR years", "cagr", None),
        ("Positive Alpha years", "alpha", None),
        ("Mean Win Rate", "win_rate", "{:.1%}"),
    ]:
        row = f"| {metric_name} "
        for config_name in ["no_parking", "always_park", "bull_only"]:
            results = all_results[config_name]
            vals_list = [m[metric_key] for m in results if m.get(metric_key) is not None]
            if not vals_list:
                row += "| N/A "
                continue

            if metric_name == "Positive CAGR years":
                v = f"{sum(1 for v in vals_list if v > 0)}/{len(vals_list)}"
            elif metric_name == "Positive Alpha years":
                v = f"{sum(1 for v in vals_list if v > 0)}/{len(vals_list)}"
            elif "Median" in metric_name:
                v = fmt.format(np.median(vals_list))
            elif "Mean" in metric_name and fmt:
                v = fmt.format(np.mean(vals_list))
            elif "Worst" in metric_name:
                v = f"{min(vals_list):.1%}"
            else:
                v = "?"
            row += f"| {v} "
        row += "|"
        lines.append(row)

    # Recommendation
    lines.append("\n## Recommendation")

    configs_sharpes = {}
    for config_name in PARKING_CONFIGS:
        results = all_results[config_name]
        sharpes = [m["sharpe"] for m in results]
        configs_sharpes[config_name] = np.median(sharpes)

    best = max(configs_sharpes, key=configs_sharpes.get)
    baseline_sharpe = configs_sharpes["always_park"]
    best_sharpe = configs_sharpes[best]

    if best == "always_park":
        lines.append("\n**KEEP ALWAYS_PARK**: Current SPY parking behavior is already optimal.")
    else:
        improvement = best_sharpe - baseline_sharpe
        lines.append(f"\n**BEST CONFIG: {best}**")
        lines.append(f"- Median Sharpe: {best_sharpe:.2f} vs always_park {baseline_sharpe:.2f} ({improvement:+.2f})")

        # Check consistency
        always_results = all_results["always_park"]
        best_results = all_results[best]
        wins = sum(1 for s1, s2 in zip(
            [m["sharpe"] for m in always_results],
            [m["sharpe"] for m in best_results]
        ) if s2 > s1)
        lines.append(f"- Better Sharpe in {wins}/{len(always_results)} years")

    return "\n".join(lines)


def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  SPY PARKING — WALK-FORWARD TEST")
    log(f"  {len(PARKING_CONFIGS)} configs × 11 years = {len(PARKING_CONFIGS) * 11} backtests")
    log("=" * 70)

    years = list(range(2015, 2026))

    # Check predictions exist
    missing = [y for y in years if not (WF_DIR / f"predictions_{y}.parquet").exists()]
    if missing:
        log(f"\nERROR: Missing predictions for years: {missing}")
        sys.exit(1)

    # Load predictions
    all_preds = {}
    for year in years:
        df = pd.read_parquet(WF_DIR / f"predictions_{year}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = df
        log(f"  Loaded predictions_{year}.parquet ({len(df):,} rows)")

    # Get all symbols
    all_syms = set()
    for df in all_preds.values():
        all_syms.update(df["symbol"].unique())
    all_syms = sorted(all_syms)

    # Load price bars
    log(f"\nLoading price bars for {len(all_syms)} symbols ...")
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    log(f"  Price data: {close.shape}")

    spy_full = close["SPY"].dropna()

    # Run backtests
    all_results = {name: [] for name in PARKING_CONFIGS}

    for year in years:
        log(f"\n{'─'*70}")
        log(f"  YEAR {year}")
        log(f"{'─'*70}")

        preds_df = all_preds[year]
        all_dates = sorted(preds_df["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        for config_name, parking_mode in PARKING_CONFIGS.items():
            t_cfg = time.perf_counter()

            metrics = run_parking_backtest(
                preds_df.copy(), year, close_year, spy_full,
                parking_mode=parking_mode,
                slot_config=SLOT_LIVE_V2,
            )

            if metrics:
                all_results[config_name].append(metrics)
                elapsed = time.perf_counter() - t_cfg
                log(f"  {config_name:14s}: CAGR={metrics['cagr']:+.1%}  "
                    f"Sharpe={metrics['sharpe']:.2f}  DD={metrics['max_dd']:.1%}  "
                    f"Trades={metrics['n_trades']}  ({elapsed:.1f}s)")

    # Generate report
    log(f"\n{'='*70}")
    log("  GENERATING REPORT")
    log(f"{'='*70}")

    report = generate_report(all_results)
    report_file = OUT_DIR / "SPY_PARKING_RESULTS.md"
    report_file.write_text(report)
    log(f"\nReport saved to {report_file}")

    # Save raw results
    json_results = {}
    for config_name, results in all_results.items():
        json_results[config_name] = []
        for m in results:
            m_clean = {k: v for k, v in m.items() if k != "parking_stats"}
            json_results[config_name].append(m_clean)
    json_file = OUT_DIR / "spy_parking_results.json"
    json_file.write_text(json.dumps(json_results, indent=2, default=str))

    total = time.perf_counter() - t0
    log(f"\nTotal time: {total:.0f}s ({total/60:.1f}min)")
    log("Done.")


if __name__ == "__main__":
    main()
