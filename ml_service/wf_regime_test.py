#!/usr/bin/env python3
"""
Walk-Forward Test: Regime-Aware Position Sizing
================================================
Tests whether limiting max positions based on SPY regime improves
risk-adjusted returns across 11 annual walk-forward periods (2015-2025).

Regime detection:
  BULLISH  — SPY > SMA50 AND SPY > SMA200
  CAUTIOUS — SPY above one MA but below the other
  BEARISH  — SPY < SMA50 AND SPY < SMA200

Configs tested (max positions per regime: BULL/CAUTIOUS/BEAR):
  baseline    — 8/8/8 (no regime awareness)
  conservative — 8/5/2
  moderate    — 8/6/3
  aggressive  — 8/4/0 (no new buys in bear markets)

Uses existing walk-forward predictions from ml_service/data/walkforward/.
"""

import os
import sys
import json
import time
import copy
import warnings
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    MLMediumStrategy, MomentumStrategy, MeanReversionStrategy,
    MegaCapStrategy, PortfolioManager, SlotConfig,
    SLOT_LIVE, SLOT_LIVE_V2, load_bars_cached,
    INITIAL_CASH, SLIPPAGE, HOLD_DAYS,
    Position, NEVER_BUY, COOLDOWN_DAYS, MIN_POSITION_DOLLARS,
    POSITION_PCT, SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
    IDLE_SPY_MIN_OPP_RATIO, SYMBOL_SECTOR, SECTOR_MAX_POSITIONS,
    USE_VOL_SIZING, _has_earnings_within,
)
from backtest_utils import calc_metrics, calc_alpha_beta

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"
OUT_DIR = DATA_DIR / "walkforward_regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Regime Detection ─────────────────────────────────────────────────────────

def compute_spy_regime(spy_series: pd.Series) -> pd.Series:
    """
    Classify each date as BULLISH, CAUTIOUS, or BEARISH.

    Args:
        spy_series: SPY close prices indexed by date (needs 200+ days of history)

    Returns:
        Series of regime labels indexed by date.
    """
    sma50 = spy_series.rolling(50, min_periods=50).mean()
    sma200 = spy_series.rolling(200, min_periods=200).mean()

    regime = pd.Series("CAUTIOUS", index=spy_series.index)

    bullish = (spy_series > sma50) & (spy_series > sma200)
    bearish = (spy_series < sma50) & (spy_series < sma200)

    regime[bullish] = "BULLISH"
    regime[bearish] = "BEARISH"

    return regime


# ── Regime-Aware Portfolio Manager ───────────────────────────────────────────

class RegimeAwarePortfolioManager(PortfolioManager):
    """
    Subclass that dynamically adjusts max_positions based on SPY regime.

    During the daily loop, the effective max_positions is looked up from
    regime_limits based on that day's regime classification.
    """

    def __init__(self, strategies, slot_config, regime_series, regime_limits,
                 initial_cash=INITIAL_CASH, slippage=SLIPPAGE,
                 hold_days=HOLD_DAYS):
        super().__init__(strategies, slot_config, initial_cash, slippage, hold_days)
        self.regime_series = regime_series  # date -> regime label
        self.regime_limits = regime_limits  # {"BULLISH": 8, "CAUTIOUS": 5, "BEARISH": 2}
        self._base_max = slot_config.max_positions

    def run(self, all_dates, spy_prices=None, price_data=None, detail_log=False):
        """Override run to dynamically adjust max_positions per day."""
        regime_counts = {"BULLISH": 0, "CAUTIOUS": 0, "BEARISH": 0}

        # We need to intercept the daily loop.
        # The simplest approach: temporarily modify slot_config.max_positions
        # before each day's processing.
        # Since the parent's run() doesn't expose per-day hooks, we replicate
        # the critical loop here with regime awareness.

        n_dates = len(all_dates)
        multi = len(self.strategies) > 1

        # Compute global ATR for vol-targeted sizing (if enabled)
        import unified_backtester as _ub
        if USE_VOL_SIZING and price_data is not None:
            daily_ret_abs = price_data.pct_change().abs()
            _ub._global_atr_df = daily_ret_abs.rolling(14, min_periods=14).mean()
        else:
            _ub._global_atr_df = None

        # Pre-build price lookup
        px_lookup = {}
        if price_data is not None:
            for date in all_dates:
                if date in price_data.index:
                    px_lookup[date] = price_data.loc[date].to_dict()

        # State
        cash = float(self.initial_cash)
        positions = {}
        idle_spy_shares = 0.0
        trades = []
        cooldowns = {}
        port_vals = []
        last_spy_action_idx = -2

        for i, date in enumerate(all_dates):
            spy_px = spy_prices.get(date) if spy_prices else None
            if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
                spy_px = None

            day_prices = px_lookup.get(date, {})
            cur = {"idx": i, "date": date, "n_dates": n_dates,
                   "prices": day_prices}

            # ── Determine regime for today ──
            regime = self.regime_series.get(date, "CAUTIOUS")
            effective_max = self.regime_limits.get(regime, self._base_max)
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

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

            # ── 5. Execute buys (REGIME-AWARE max_slots) ──
            max_slots = max(0, effective_max - len(positions))

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

                # Additional regime check: don't exceed effective_max
                if len(positions) >= effective_max:
                    break

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

            # ── 6. Park idle cash in SPY ──
            if spy_px and i > last_spy_action_idx:
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
        return series, [float(t) for t in trades], regime_counts


# ── Test Configurations ──────────────────────────────────────────────────────

CONFIGS = {
    "baseline":     {"BULLISH": 8, "CAUTIOUS": 8, "BEARISH": 8},   # no change
    "conservative": {"BULLISH": 8, "CAUTIOUS": 5, "BEARISH": 2},
    "moderate":     {"BULLISH": 8, "CAUTIOUS": 6, "BEARISH": 3},
    "aggressive":   {"BULLISH": 8, "CAUTIOUS": 4, "BEARISH": 0},
}


def log(msg: str):
    print(msg, flush=True)


def run_regime_backtest(preds_df, year, close, spy_series, regime_limits,
                        slot_config=None):
    """Run backtest for one year with regime-aware position sizing."""
    all_dates = sorted(preds_df["date"].unique().tolist())
    if len(all_dates) < 10:
        return None

    years_span = (all_dates[-1] - all_dates[0]).days / 365.25
    if years_span <= 0:
        years_span = len(all_dates) / 252.0

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        log(f"  [WARN] No SPY data for {year}")
        return None

    spy_dict = close["SPY"].to_dict()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    # Compute regime from full SPY history (need 200d lookback)
    regime_series = compute_spy_regime(spy_series)
    regime_dict = regime_series.to_dict()

    # Normalize prob column
    if "prob_ensemble" in preds_df.columns and "prob" not in preds_df.columns:
        preds_df = preds_df.rename(columns={"prob_ensemble": "prob"})

    sc = slot_config or SLOT_LIVE_V2

    # Build strategies
    strategies = [
        MLMediumStrategy(preds_df, threshold=0.55, top_n=5, selection_mode="top_n"),
        MomentumStrategy(close, volume_data=None, regime_filter=False),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]

    pm = RegimeAwarePortfolioManager(
        strategies=strategies,
        slot_config=sc,
        regime_series=regime_dict,
        regime_limits=regime_limits,
    )
    vals, trades, regime_counts = pm.run(all_dates, spy_prices=spy_dict, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"regime_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    metrics["regime_counts"] = regime_counts

    return metrics


def generate_report(all_results, regime_stats):
    """Generate markdown report comparing regime-aware configs."""
    lines = []
    lines.append("# Regime-Aware Position Sizing — Walk-Forward Results")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Regime Detection")
    lines.append("- **BULLISH**: SPY > SMA50 AND SPY > SMA200")
    lines.append("- **CAUTIOUS**: SPY above one MA but below the other")
    lines.append("- **BEARISH**: SPY < SMA50 AND SPY < SMA200")

    lines.append("\n## Configurations Tested")
    lines.append("| Config | BULL max | CAUTIOUS max | BEAR max |")
    lines.append("|--------|---------|-------------|---------|")
    for name, limits in CONFIGS.items():
        lines.append(f"| {name} | {limits['BULLISH']} | {limits['CAUTIOUS']} | {limits['BEARISH']} |")

    # Regime distribution per year
    lines.append("\n## Regime Distribution by Year")
    lines.append("| Year | BULLISH days | CAUTIOUS days | BEARISH days | % BULLISH |")
    lines.append("|------|-------------|--------------|-------------|-----------|")
    for year, stats in sorted(regime_stats.items()):
        total = stats["BULLISH"] + stats["CAUTIOUS"] + stats["BEARISH"]
        pct_bull = stats["BULLISH"] / total * 100 if total > 0 else 0
        lines.append(f"| {year} | {stats['BULLISH']} | {stats['CAUTIOUS']} | {stats['BEARISH']} | {pct_bull:.0f}% |")

    # Per-config results
    for config_name in CONFIGS:
        results = all_results[config_name]
        lines.append(f"\n## Results: {config_name} ({CONFIGS[config_name]['BULLISH']}/{CONFIGS[config_name]['CAUTIOUS']}/{CONFIGS[config_name]['BEARISH']})")
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
    lines.append("| Metric | baseline | conservative | moderate | aggressive |")
    lines.append("|--------|----------|-------------|----------|-----------|")

    for metric_name, metric_key, fmt, higher_better in [
        ("Median CAGR", "cagr", "{:+.1%}", True),
        ("Mean CAGR", "cagr", "{:+.1%}", True),
        ("Median Sharpe", "sharpe", "{:.2f}", True),
        ("Mean Sharpe", "sharpe", "{:.2f}", True),
        ("Worst Max DD", "max_dd", "{:.1%}", False),
        ("Mean Max DD", "max_dd", "{:.1%}", False),
        ("Median Alpha", "alpha", "{:+.1%}", True),
        ("Positive CAGR years", "cagr", None, True),
        ("Positive Alpha years", "alpha", None, True),
        ("Mean Trades", "n_trades", "{:.0f}", None),
        ("Mean Win Rate", "win_rate", "{:.1%}", True),
    ]:
        row = f"| {metric_name} "
        for config_name in ["baseline", "conservative", "moderate", "aggressive"]:
            results = all_results[config_name]
            vals = [m[metric_key] for m in results if m.get(metric_key) is not None]
            if not vals:
                row += "| N/A "
                continue

            if metric_name == "Positive CAGR years":
                v = f"{sum(1 for v in vals if v > 0)}/{len(vals)}"
            elif metric_name == "Positive Alpha years":
                v = f"{sum(1 for v in vals if v > 0)}/{len(vals)}"
            elif "Median" in metric_name:
                v = fmt.format(np.median(vals))
            elif "Mean" in metric_name:
                v = fmt.format(np.mean(vals))
            elif "Worst" in metric_name:
                v = fmt.format(min(vals))
            else:
                v = "?"
            row += f"| {v} "
        row += "|"
        lines.append(row)

    # Recommendation
    lines.append("\n## Recommendation")

    # Compare median Sharpe ratios to find best config
    best_config = None
    best_sharpe = -999
    for config_name in CONFIGS:
        results = all_results[config_name]
        sharpes = [m["sharpe"] for m in results]
        median_sharpe = np.median(sharpes)
        if median_sharpe > best_sharpe:
            best_sharpe = median_sharpe
            best_config = config_name

    baseline_results = all_results["baseline"]
    baseline_sharpes = [m["sharpe"] for m in baseline_results]
    baseline_median = np.median(baseline_sharpes)

    if best_config == "baseline":
        lines.append("\n**KEEP BASELINE**: Regime-aware sizing does not improve risk-adjusted returns.")
    else:
        best_results = all_results[best_config]
        best_sharpes = [m["sharpe"] for m in best_results]
        improvement = np.median(best_sharpes) - baseline_median

        # Count years where best config beats baseline
        wins = sum(1 for s1, s2 in zip(baseline_sharpes, best_sharpes) if s2 > s1)
        lines.append(f"\n**BEST CONFIG: {best_config}** ({CONFIGS[best_config]['BULLISH']}/{CONFIGS[best_config]['CAUTIOUS']}/{CONFIGS[best_config]['BEARISH']})")
        lines.append(f"- Median Sharpe improvement: {improvement:+.2f}")
        lines.append(f"- Sharpe improvement in {wins}/{len(baseline_sharpes)} years")

        baseline_dds = [m["max_dd"] for m in baseline_results]
        best_dds = [m["max_dd"] for m in best_results]
        dd_improvement = np.mean(best_dds) - np.mean(baseline_dds)
        lines.append(f"- Mean max DD change: {dd_improvement:+.1%}")

        if wins >= 7:
            lines.append(f"\n**DEPLOY**: Consistent improvement across majority of years.")
        elif wins >= 5:
            lines.append(f"\nMarginal improvement. Consider deploying with monitoring.")
        else:
            lines.append(f"\n**KEEP BASELINE**: Improvement not consistent enough ({wins}/{len(baseline_sharpes)} years).")

    return "\n".join(lines)


def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  REGIME-AWARE POSITION SIZING — WALK-FORWARD TEST")
    log(f"  {len(CONFIGS)} configs × 11 years = {len(CONFIGS) * 11} backtests")
    log("=" * 70)

    years = list(range(2015, 2026))

    # Check that walk-forward predictions exist
    missing = []
    for year in years:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            missing.append(year)
    if missing:
        log(f"\nERROR: Missing walk-forward predictions for years: {missing}")
        log("Run walk_forward_validation.py first.")
        sys.exit(1)

    # Load all predictions
    all_preds = {}
    for year in years:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        df = pd.read_parquet(pred_file)
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = df
        log(f"  Loaded predictions_{year}.parquet ({len(df):,} rows)")

    # Get full universe of symbols
    all_syms = set()
    for df in all_preds.values():
        all_syms.update(df["symbol"].unique())
    all_syms = sorted(all_syms)

    # Load price bars for full period (need 200d before 2015 for SMA200)
    log(f"\nLoading price bars for {len(all_syms)} symbols ...")
    start_str = "2013-06-01"  # 200 trading days before 2015
    end_str = "2025-12-31"
    close = load_bars_cached(all_syms, start_str, end_str)
    log(f"  Price data: {close.shape}")

    # Get full SPY series for regime computation
    spy_full = close["SPY"].dropna()
    if len(spy_full) < 200:
        log("ERROR: Insufficient SPY data for SMA200")
        sys.exit(1)

    # Compute regime for full period
    regime_series = compute_spy_regime(spy_full)
    log(f"\nSPY Regime Distribution (full period):")
    regime_vc = regime_series.value_counts()
    for r in ["BULLISH", "CAUTIOUS", "BEARISH"]:
        count = regime_vc.get(r, 0)
        pct = count / len(regime_series) * 100
        log(f"  {r}: {count} days ({pct:.1f}%)")

    # Run backtests for each config × year
    all_results = {name: [] for name in CONFIGS}
    regime_stats = {}

    for year in years:
        log(f"\n{'─'*70}")
        log(f"  YEAR {year}")
        log(f"{'─'*70}")

        preds_df = all_preds[year]
        all_dates = sorted(preds_df["date"].unique().tolist())

        # Get regime stats for this year
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        year_regime = regime_series[(regime_series.index >= year_start) & (regime_series.index <= year_end)]
        year_vc = year_regime.value_counts()
        regime_stats[year] = {
            "BULLISH": int(year_vc.get("BULLISH", 0)),
            "CAUTIOUS": int(year_vc.get("CAUTIOUS", 0)),
            "BEARISH": int(year_vc.get("BEARISH", 0)),
        }
        log(f"  Regime: BULL={regime_stats[year]['BULLISH']}d  "
            f"CAUTIOUS={regime_stats[year]['CAUTIOUS']}d  "
            f"BEAR={regime_stats[year]['BEARISH']}d")

        # Reindex close data for this year's simulation dates
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(
            close.index.union(sim_index), method="ffill"
        )

        for config_name, regime_limits in CONFIGS.items():
            t_cfg = time.perf_counter()

            metrics = run_regime_backtest(
                preds_df.copy(), year, close_year, spy_full,
                regime_limits=regime_limits,
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

    report = generate_report(all_results, regime_stats)
    report_file = OUT_DIR / "REGIME_RESULTS.md"
    report_file.write_text(report)
    log(f"\nReport saved to {report_file}")

    # Save raw results as JSON
    json_results = {}
    for config_name, results in all_results.items():
        json_results[config_name] = []
        for m in results:
            m_clean = {k: v for k, v in m.items() if k != "regime_counts"}
            json_results[config_name].append(m_clean)
    json_file = OUT_DIR / "regime_results.json"
    json_file.write_text(json.dumps(json_results, indent=2, default=str))

    total = time.perf_counter() - t0
    log(f"\nTotal time: {total:.0f}s ({total/60:.1f}min)")
    log("Done.")


if __name__ == "__main__":
    main()
