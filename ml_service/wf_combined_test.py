#!/usr/bin/env python3
"""
Combined Walk-Forward Test
==========================
Tests all combinations of regime-aware sizing × SPY parking
across 11 annual walk-forward periods (2015-2025).

8 configurations = 4 regime × 2 parking × 11 years = 88 backtests.

Uses existing walk-forward predictions from ml_service/data/walkforward/.
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
from wf_regime_test import compute_spy_regime, RegimeAwarePortfolioManager

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"
OUT_DIR = DATA_DIR / "walkforward_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Combined Portfolio Manager ───────────────────────────────────────────────

class CombinedPortfolioManager(PortfolioManager):
    """
    Combines regime-aware position sizing + conditional SPY parking.
    """

    def __init__(self, strategies, slot_config, regime_series, regime_limits,
                 parking_mode="always",
                 initial_cash=INITIAL_CASH, slippage=SLIPPAGE,
                 hold_days=HOLD_DAYS):
        super().__init__(strategies, slot_config, initial_cash, slippage, hold_days)
        self.regime_series = regime_series
        self.regime_limits = regime_limits
        self._base_max = slot_config.max_positions
        self.parking_mode = parking_mode

    def run(self, all_dates, spy_prices=None, price_data=None, detail_log=False):
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

        for i, date in enumerate(all_dates):
            spy_px = spy_prices.get(date) if spy_prices else None
            if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
                spy_px = None

            day_prices = px_lookup.get(date, {})
            cur = {"idx": i, "date": date, "n_dates": n_dates, "prices": day_prices}

            # Regime for today
            regime = self.regime_series.get(date, "CAUTIOUS")
            effective_max = self.regime_limits.get(regime, self._base_max)

            # SPY parking allowed?
            if self.parking_mode == "none":
                allow_parking = False
            elif self.parking_mode == "always":
                allow_parking = True
            elif self.parking_mode == "bull_only":
                allow_parking = regime == "BULLISH"
            else:
                allow_parking = True

            # 1. Update peak prices
            for sym, pos in positions.items():
                if pos.price_based and sym in day_prices:
                    px = day_prices[sym]
                    if not np.isnan(px) and px > pos.peak_price:
                        pos.peak_price = px

            # 2. Close positions
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

            # 3. Gather signals
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

            # 4. Resolve overlaps
            seen_syms = set()
            resolved = []
            for sig in all_signals:
                if sig.symbol not in seen_syms:
                    seen_syms.add(sig.symbol)
                    resolved.append(sig)
            resolved.sort(key=lambda s: s.confidence, reverse=True)

            # 5. Execute buys (regime-aware max)
            max_slots = max(0, effective_max - len(positions))

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

            # 6. Conditional SPY parking
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
                proceeds = idle_spy_shares * spy_px * (1.0 - self.slippage)
                cash += proceeds
                idle_spy_shares = 0.0
                last_spy_action_idx = i

            # 7. Mark-to-market
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
        return series, [float(t) for t in trades]


# ── Configurations ───────────────────────────────────────────────────────────

REGIME_CONFIGS = {
    "off":           {"BULLISH": 8, "CAUTIOUS": 8, "BEARISH": 8},
    "conservative":  {"BULLISH": 8, "CAUTIOUS": 5, "BEARISH": 2},
    "moderate":      {"BULLISH": 8, "CAUTIOUS": 6, "BEARISH": 3},
    "aggressive":    {"BULLISH": 8, "CAUTIOUS": 4, "BEARISH": 0},
}

PARKING_MODES = ["always_park", "no_parking"]

# All 8 combined configs
COMBINED_CONFIGS = []
for regime_name, regime_limits in REGIME_CONFIGS.items():
    for parking in PARKING_MODES:
        config_label = f"{regime_name}+{parking}"
        COMBINED_CONFIGS.append({
            "label": config_label,
            "regime_name": regime_name,
            "regime_limits": regime_limits,
            "parking_mode": "always" if parking == "always_park" else "none",
        })


def log(msg: str):
    print(msg, flush=True)


def run_combined_backtest(preds_df, year, close, spy_full, regime_limits,
                          parking_mode, slot_config=None):
    """Run backtest with combined regime + parking config."""
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

    use_spy = spy_dict if parking_mode != "none" else None

    pm = CombinedPortfolioManager(
        strategies=strategies,
        slot_config=sc,
        regime_series=regime_dict,
        regime_limits=regime_limits,
        parking_mode=parking_mode,
    )
    vals, trades = pm.run(all_dates, spy_prices=use_spy, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"combined_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year

    return metrics


def generate_report(all_results, regime_stats):
    """Generate comprehensive comparison report."""
    lines = []
    lines.append("# Combined Walk-Forward Test — All Enhancement Combinations")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Test Matrix")
    lines.append("8 configs = 4 regime settings × 2 parking modes × 11 years = 88 backtests")
    lines.append("")
    lines.append("| # | Config | Regime (BULL/CAUT/BEAR) | SPY Parking |")
    lines.append("|---|--------|------------------------|-------------|")
    for i, cfg in enumerate(COMBINED_CONFIGS, 1):
        rl = cfg["regime_limits"]
        lines.append(f"| {i} | {cfg['label']} | {rl['BULLISH']}/{rl['CAUTIOUS']}/{rl['BEARISH']} | {cfg['parking_mode']} |")

    # Master comparison table
    lines.append("\n## Summary Comparison (sorted by median Sharpe)")
    lines.append("")

    # Compute summary stats for each config
    summary = []
    for cfg in COMBINED_CONFIGS:
        label = cfg["label"]
        results = all_results[label]
        if not results:
            continue
        cagrs = [m["cagr"] for m in results]
        sharpes = [m["sharpe"] for m in results]
        dds = [m["max_dd"] for m in results]
        alphas = [m["alpha"] for m in results if m.get("alpha") is not None]
        trades_list = [m["n_trades"] for m in results]

        summary.append({
            "label": label,
            "median_cagr": np.median(cagrs),
            "mean_cagr": np.mean(cagrs),
            "median_sharpe": np.median(sharpes),
            "mean_sharpe": np.mean(sharpes),
            "worst_dd": min(dds),
            "mean_dd": np.mean(dds),
            "median_alpha": np.median(alphas) if alphas else np.nan,
            "pos_cagr": sum(1 for c in cagrs if c > 0),
            "pos_alpha": sum(1 for a in alphas if a > 0) if alphas else 0,
            "n_years": len(results),
            "mean_trades": np.mean(trades_list),
        })

    summary.sort(key=lambda x: x["median_sharpe"], reverse=True)

    lines.append("| Rank | Config | Med CAGR | Mean CAGR | Med Sharpe | Mean Sharpe | Worst DD | Med Alpha | +CAGR | +Alpha |")
    lines.append("|------|--------|----------|-----------|------------|-------------|----------|-----------|-------|--------|")

    for rank, s in enumerate(summary, 1):
        lines.append(
            f"| {rank} "
            f"| {s['label']} "
            f"| {s['median_cagr']:+.1%} "
            f"| {s['mean_cagr']:+.1%} "
            f"| {s['median_sharpe']:.2f} "
            f"| {s['mean_sharpe']:.2f} "
            f"| {s['worst_dd']:.1%} "
            f"| {s['median_alpha']:+.1%} "
            f"| {s['pos_cagr']}/{s['n_years']} "
            f"| {s['pos_alpha']}/{s['n_years']} |"
        )

    # Per-year detail for top 3 configs
    lines.append("\n## Per-Year Detail (Top 3 Configs)")
    top3 = [s["label"] for s in summary[:3]]
    current = "off+always_park"

    # Include current baseline if not in top 3
    if current not in top3:
        top3_plus = top3 + [current]
    else:
        top3_plus = top3

    for label in top3_plus:
        results = all_results[label]
        tag = " (CURRENT)" if label == current else ""
        lines.append(f"\n### {label}{tag}")
        lines.append("| Year | CAGR | Sharpe | Max DD | Alpha | Beta | Trades |")
        lines.append("|------|------|--------|--------|-------|------|--------|")

        for m in results:
            lines.append(
                f"| {m['year']} "
                f"| {m['cagr']:+.1%} "
                f"| {m['sharpe']:.2f} "
                f"| {m['max_dd']:.1%} "
                f"| {'+' if m.get('alpha') and m['alpha'] > 0 else ''}{m.get('alpha', 0) or 0:.1%} "
                f"| {m.get('beta', 0) or 0:.2f} "
                f"| {m['n_trades']} |"
            )

    # Head-to-head: best vs current
    lines.append("\n## Head-to-Head: Best Config vs Current Baseline")
    best_label = summary[0]["label"]
    best_results = all_results[best_label]
    curr_results = all_results[current]

    lines.append(f"\n**Best: {best_label}** vs **Current: {current}**")
    lines.append("")
    lines.append("| Year | Current CAGR | Best CAGR | Current Sharpe | Best Sharpe | Current DD | Best DD |")
    lines.append("|------|-------------|-----------|---------------|------------|-----------|--------|")

    sharpe_wins = 0
    cagr_wins = 0
    for m_curr, m_best in zip(curr_results, best_results):
        lines.append(
            f"| {m_curr['year']} "
            f"| {m_curr['cagr']:+.1%} "
            f"| {m_best['cagr']:+.1%} "
            f"| {m_curr['sharpe']:.2f} "
            f"| {m_best['sharpe']:.2f} "
            f"| {m_curr['max_dd']:.1%} "
            f"| {m_best['max_dd']:.1%} |"
        )
        if m_best["sharpe"] > m_curr["sharpe"]:
            sharpe_wins += 1
        if m_best["cagr"] > m_curr["cagr"]:
            cagr_wins += 1

    n = len(curr_results)
    lines.append(f"\n- Best config wins {sharpe_wins}/{n} years by Sharpe")
    lines.append(f"- Best config wins {cagr_wins}/{n} years by CAGR")

    # Deployment recommendation
    lines.append("\n## Deployment Recommendation")

    best = summary[0]
    curr_summary = next(s for s in summary if s["label"] == current)

    sharpe_improve = best["median_sharpe"] - curr_summary["median_sharpe"]
    cagr_improve = best["median_cagr"] - curr_summary["median_cagr"]

    lines.append(f"\n### Recommended Config: **{best['label']}**")
    lines.append(f"- Median Sharpe: {best['median_sharpe']:.2f} vs current {curr_summary['median_sharpe']:.2f} ({sharpe_improve:+.2f})")
    lines.append(f"- Median CAGR: {best['median_cagr']:+.1%} vs current {curr_summary['median_cagr']:+.1%} ({cagr_improve:+.1%}pp)")
    lines.append(f"- Worst DD: {best['worst_dd']:.1%} vs current {curr_summary['worst_dd']:.1%}")
    lines.append(f"- Wins {sharpe_wins}/{n} years by Sharpe")

    if sharpe_wins >= 8:
        lines.append(f"\n**STRONG DEPLOY**: Consistent improvement across {sharpe_wins}/{n} years.")
    elif sharpe_wins >= 6:
        lines.append(f"\n**DEPLOY WITH MONITORING**: Improvement in majority of years ({sharpe_wins}/{n}).")
    elif sharpe_wins >= 4:
        lines.append(f"\nMARGINAL: Only {sharpe_wins}/{n} years improved. Consider partial deployment.")
    else:
        lines.append(f"\n**DO NOT DEPLOY**: Insufficient evidence of improvement ({sharpe_wins}/{n} years).")

    lines.append("\n### Production Changes Required")

    # Parse best config
    regime_part, parking_part = best["label"].split("+")
    if regime_part != "off":
        rl = REGIME_CONFIGS[regime_part]
        lines.append(f"1. **Regime-aware sizing**: Add SPY SMA50/SMA200 regime detection to tradingEngine.js")
        lines.append(f"   - BULLISH: max {rl['BULLISH']} positions")
        lines.append(f"   - CAUTIOUS: max {rl['CAUTIOUS']} positions")
        lines.append(f"   - BEARISH: max {rl['BEARISH']} positions")
    else:
        lines.append("1. **Regime sizing**: No change (keep fixed 8 max)")

    if parking_part == "no_parking":
        lines.append("2. **SPY parking**: DISABLE idle cash SPY parking")
    else:
        lines.append("2. **SPY parking**: Keep current behavior")

    return "\n".join(lines)


def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  COMBINED WALK-FORWARD TEST")
    log(f"  {len(COMBINED_CONFIGS)} configs × 11 years = {len(COMBINED_CONFIGS) * 11} backtests")
    log("=" * 70)

    years = list(range(2015, 2026))

    missing = [y for y in years if not (WF_DIR / f"predictions_{y}.parquet").exists()]
    if missing:
        log(f"\nERROR: Missing predictions for years: {missing}")
        sys.exit(1)

    all_preds = {}
    for year in years:
        df = pd.read_parquet(WF_DIR / f"predictions_{year}.parquet")
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = df
        log(f"  Loaded predictions_{year}.parquet ({len(df):,} rows)")

    all_syms = set()
    for df in all_preds.values():
        all_syms.update(df["symbol"].unique())
    all_syms = sorted(all_syms)

    log(f"\nLoading price bars for {len(all_syms)} symbols ...")
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    log(f"  Price data: {close.shape}")

    spy_full = close["SPY"].dropna()

    # Regime stats per year
    regime_series = compute_spy_regime(spy_full)
    regime_stats = {}
    for year in years:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        yr = regime_series[(regime_series.index >= year_start) & (regime_series.index <= year_end)]
        vc = yr.value_counts()
        regime_stats[year] = {
            "BULLISH": int(vc.get("BULLISH", 0)),
            "CAUTIOUS": int(vc.get("CAUTIOUS", 0)),
            "BEARISH": int(vc.get("BEARISH", 0)),
        }

    # Run all backtests
    all_results = {cfg["label"]: [] for cfg in COMBINED_CONFIGS}

    for year in years:
        log(f"\n{'─'*70}")
        log(f"  YEAR {year}  (BULL={regime_stats[year]['BULLISH']}d  "
            f"CAUT={regime_stats[year]['CAUTIOUS']}d  "
            f"BEAR={regime_stats[year]['BEARISH']}d)")
        log(f"{'─'*70}")

        preds_df = all_preds[year]
        all_dates = sorted(preds_df["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        for cfg in COMBINED_CONFIGS:
            t_cfg = time.perf_counter()

            metrics = run_combined_backtest(
                preds_df.copy(), year, close_year, spy_full,
                regime_limits=cfg["regime_limits"],
                parking_mode=cfg["parking_mode"],
                slot_config=SLOT_LIVE_V2,
            )

            if metrics:
                all_results[cfg["label"]].append(metrics)
                elapsed = time.perf_counter() - t_cfg
                log(f"  {cfg['label']:30s}: CAGR={metrics['cagr']:+.1%}  "
                    f"Sharpe={metrics['sharpe']:.2f}  DD={metrics['max_dd']:.1%}  ({elapsed:.1f}s)")

    # Generate report
    log(f"\n{'='*70}")
    log("  GENERATING REPORT")
    log(f"{'='*70}")

    report = generate_report(all_results, regime_stats)
    report_file = OUT_DIR / "COMBINED_RESULTS.md"
    report_file.write_text(report)
    log(f"\nReport saved to {report_file}")

    # Save raw JSON
    json_file = OUT_DIR / "combined_results.json"
    json_file.write_text(json.dumps(all_results, indent=2, default=str))

    total = time.perf_counter() - t0
    log(f"\nTotal time: {total:.0f}s ({total/60:.1f}min)")
    log("Done.")


if __name__ == "__main__":
    main()
