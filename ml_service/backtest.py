#!/usr/bin/env python3
"""
Backtest CLI
============
Thin wrapper around unified_backtester.py for fast iteration.

Usage examples:
    python3 backtest.py                          # ML-only, default params
    python3 backtest.py --strategy momentum      # momentum only
    python3 backtest.py --strategy combined       # all strategies
    python3 backtest.py --ml-threshold 0.60       # stricter ML threshold
    python3 backtest.py --start 2022-01-01 --end 2024-01-01
    python3 backtest.py --symbols AAPL,MSFT,GOOGL --no-cache
    python3 backtest.py --output results.json --quiet
    python3 backtest.py --live                    # match live production exactly
    python3 backtest.py --no-spy-parking          # disable SPY parking only
    python3 backtest.py --cautious-filter          # enable CAUTIOUS ML filter only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure ml_service is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import (
    MLMediumStrategy, CautiousMLStrategy, compute_regime_live,
    MomentumStrategy, MeanReversionStrategy, MLSlowStrategy,
    MegaCapStrategy, TSMOMStrategy, TrendStrategy, PortfolioManager, SlotConfig,
    SLOT_ML_ONLY, SLOT_MOM_ONLY, SLOT_MR_ONLY, SLOT_ML_SLOW_ONLY,
    SLOT_ML_MOM_MR, SLOT_ML_MOM_MR_SLOW, SLOT_MCAP_ONLY, SLOT_ML_MOM_MR_MCAP,
    SLOT_TSMOM_ONLY, SLOT_ML_MOM_MCAP_TSMOM, SLOT_LIVE,
    SLOT_TREND_ONLY, SLOT_LIVE_TREND,
    load_bars_cached, load_predictions_cached,
    INITIAL_CASH, DATA_DIR,
)
from backtest_utils import calc_metrics, calc_alpha_beta


# ── Strategy / slot config mapping ──────────────────────────────────────────

# Per-strategy primary slot counts (used to build dynamic SlotConfig)
_STRATEGY_SLOTS = {
    "ml":             ("ml_medium", 2),
    "momentum":       ("momentum", 4),
    "mean_reversion": ("mean_reversion", 2),
    "ml_slow":        ("ml_slow", 1),
    "mega_cap":       ("mega_cap", 2),
    "tsmom":          ("tsmom", 2),
    "trend":          ("trend", 3),
}

STRATEGY_MAP = {
    "ml":             (["ml"],                                          SLOT_ML_ONLY),
    "momentum":       (["momentum"],                                    SLOT_MOM_ONLY),
    "mean_reversion": (["mean_reversion"],                              SLOT_MR_ONLY),
    "ml_slow":        (["ml_slow"],                                     SLOT_ML_SLOW_ONLY),
    "mega_cap":       (["mega_cap"],                                    SLOT_MCAP_ONLY),
    "tsmom":          (["tsmom"],                                       SLOT_TSMOM_ONLY),
    "trend":          (["trend"],                                       SLOT_TREND_ONLY),
    "combined":       (["ml", "momentum", "mean_reversion", "mega_cap"], SLOT_ML_MOM_MR_MCAP),
    "combined_v2":    (["ml", "momentum", "mega_cap", "tsmom"],         SLOT_ML_MOM_MCAP_TSMOM),
    "ml_mom":         (["ml", "momentum"],                              SLOT_ML_MOM_MR),
    "combined_live":  (["ml", "momentum", "mean_reversion", "mega_cap"], SLOT_LIVE),
}


def _build_slot_config(strat_names):
    """Build a SlotConfig dynamically from a list of active strategy names."""
    slots = {}
    for name in strat_names:
        if name in _STRATEGY_SLOTS:
            engine_name, n = _STRATEGY_SLOTS[name]
            slots[engine_name] = n
    total = sum(slots.values()) + 2  # +2 flex
    return SlotConfig(strategy_slots=slots, flex_slots=2, max_positions=total)


def build_strategies(names, predictions_df, price_data, volume_data,
                     threshold, selection_mode, top_n,
                     momentum_regime_filter=False,
                     cautious_filter=False, regime_dict=None):
    """Instantiate strategy objects from name list."""
    strats = []
    for name in names:
        if name == "ml":
            if cautious_filter and regime_dict is not None:
                strats.append(CautiousMLStrategy(
                    predictions_df, regime_dict=regime_dict,
                    threshold=threshold,
                    top_n=top_n, selection_mode=selection_mode))
            else:
                strats.append(MLMediumStrategy(
                    predictions_df, threshold=threshold,
                    top_n=top_n, selection_mode=selection_mode))
        elif name == "momentum":
            strats.append(MomentumStrategy(price_data, volume_data=volume_data,
                                           regime_filter=momentum_regime_filter))
        elif name == "mean_reversion":
            strats.append(MeanReversionStrategy(price_data, volume_data=volume_data))
        elif name == "ml_slow":
            strats.append(MLSlowStrategy(predictions_df, price_data))
        elif name == "mega_cap":
            strats.append(MegaCapStrategy(price_data))
        elif name == "tsmom":
            strats.append(TSMOMStrategy(price_data))
        elif name == "trend":
            strats.append(TrendStrategy(price_data, volume_data=volume_data))
        else:
            sys.exit(f"Unknown strategy: {name}")
    return strats


def main():
    parser = argparse.ArgumentParser(
        description="Run backtests using unified_backtester engine")
    parser.add_argument("--strategy", default="ml",
                        choices=list(STRATEGY_MAP.keys()),
                        help="Strategy preset (default: ml)")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbol list (default: full universe)")
    parser.add_argument("--selection-mode", default="top_n",
                        choices=["top_n", "threshold"],
                        help="ML signal selection: top_n (live) or threshold (legacy)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Top-N picks per day in top_n mode (default: 5)")
    parser.add_argument("--ml-threshold", type=float, default=0.55,
                        help="ML probability threshold in threshold mode (default: 0.55)")
    parser.add_argument("--exclude", nargs="+", default=[],
                        choices=["ml", "momentum", "mean_reversion", "mega_cap", "ml_slow", "tsmom", "trend"],
                        help="Strategies to exclude from combined (e.g. --exclude mean_reversion)")
    parser.add_argument("--include-trend", action="store_true",
                        help="Include TrendStrategy in combined runs (matches live trend engine)")
    parser.add_argument("--predictions-path", default=None, metavar="FILE",
                        help="Path to a custom predictions parquet file (overrides --validation)")
    parser.add_argument("--validation", action="store_true",
                        help="Use validation predictions (holdout 2024+, NOT for live)")
    parser.add_argument("--vol-sizing", action="store_true",
                        help="Use volatility-targeted position sizing (equal risk per position)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recompute, bypass disk cache")
    parser.add_argument("--output", default=None,
                        help="Write metrics JSON to this path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    parser.add_argument("--momentum-regime-filter", action="store_true",
                        help="Skip momentum buys when SPY < 50-day SMA (regime filter)")
    parser.add_argument("--no-spy-parking", action="store_true",
                        help="Disable SPY idle-cash parking (matches live production)")
    parser.add_argument("--cautious-filter", action="store_true",
                        help="Enable CAUTIOUS regime ML filter: only top 2 picks during CAUTIOUS (matches live)")
    parser.add_argument("--live", action="store_true",
                        help="Match live production config: --no-spy-parking + --cautious-filter")
    parser.add_argument("--export-logs", default=None, metavar="DIR",
                        help="Export equity_curve.parquet and trade_log.parquet to DIR")
    args = parser.parse_args()

    # --live is a convenience shortcut
    if args.live:
        args.no_spy_parking = True
        args.cautious_filter = True

    # Set vol-sizing global flag before any strategy instantiation
    import unified_backtester
    unified_backtester.USE_VOL_SIZING = args.vol_sizing

    t0 = time.perf_counter()
    no_cache = args.no_cache

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    # ── 1. Load predictions ─────────────────────────────────────────────
    if args.predictions_path:
        pred_path = Path(args.predictions_path)
        log(f"Loading custom predictions from {pred_path} ...")
        preds = load_predictions_cached(pred_file=pred_path, no_cache=True)
    elif args.validation:
        log("Loading VALIDATION predictions (holdout 2024+) ...")
        preds = load_predictions_cached(no_cache=no_cache, validation=args.validation)
    else:
        log("Loading predictions ...")
        preds = load_predictions_cached(no_cache=no_cache, validation=args.validation)
    all_dates = sorted(preds["date"].unique().tolist())
    universe_syms = sorted(preds["symbol"].unique().tolist())

    # Filter by --symbols
    if args.symbols:
        sym_filter = set(s.strip().upper() for s in args.symbols.split(","))
        preds = preds[preds["symbol"].isin(sym_filter)]
        universe_syms = sorted(preds["symbol"].unique().tolist())
        all_dates = sorted(preds["date"].unique().tolist())

    # Filter by --start / --end
    if args.start:
        start_ts = pd.Timestamp(args.start)
        preds = preds[preds["date"] >= start_ts]
        all_dates = [d for d in all_dates if d >= start_ts]
    if args.end:
        end_ts = pd.Timestamp(args.end)
        preds = preds[preds["date"] <= end_ts]
        all_dates = [d for d in all_dates if d <= end_ts]

    if len(all_dates) < 2:
        sys.exit("Not enough trading days in the selected range.")

    years = (all_dates[-1] - all_dates[0]).days / 365.25
    log(f"  {len(all_dates)} days | {len(universe_syms)} symbols | {years:.1f} years")

    # ── 2. Fetch price bars ─────────────────────────────────────────────
    start_str = (pd.Timestamp(all_dates[0]) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    log("Fetching price bars ...")
    close = load_bars_cached(universe_syms, start_str, end_str, no_cache=no_cache)

    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close = close.reindex(sim_index, method="ffill")

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        # Cache may be stale or corrupt — retry without cache
        log("SPY data missing after reindex — retrying without cache ...")
        close = load_bars_cached(universe_syms, start_str, end_str, no_cache=True)
        close = close.reindex(sim_index, method="ffill")
        spy_px = close["SPY"].dropna()
        if spy_px.empty:
            sys.exit("ERROR: No SPY price data available for the requested date range.")

    spy_dict = close["SPY"].to_dict()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    # Compute regime for CAUTIOUS filter (uses full SPY history for SMA accuracy)
    regime_dict = None
    if args.cautious_filter:
        regime_series = compute_regime_live(spy_px)
        regime_dict = regime_series.to_dict()
        regime_counts = regime_series.value_counts()
        log(f"Regime distribution: {dict(regime_counts)}")

    # ── 3. Build strategies ─────────────────────────────────────────────
    strat_names, slot_config = STRATEGY_MAP[args.strategy]

    # Apply --exclude to filter out strategies from combined presets
    if args.exclude:
        strat_names = [s for s in strat_names if s not in args.exclude]
        if not strat_names:
            sys.exit("All strategies excluded — nothing to run.")
        slot_config = _build_slot_config(strat_names)

    # Add trend strategy if requested
    if args.include_trend and "trend" not in strat_names:
        strat_names = list(strat_names) + ["trend"]
        # Use SLOT_LIVE_TREND if base is combined_live, otherwise rebuild
        if args.strategy == "combined_live":
            slot_config = SLOT_LIVE_TREND
        else:
            slot_config = _build_slot_config(strat_names)

    needs_prices = any(s in strat_names for s in ["momentum", "mean_reversion", "ml_slow", "mega_cap", "tsmom", "trend"])

    # Vol-sizing needs price data for ATR computation even in ML-only mode
    price_data = close if (needs_prices or args.vol_sizing) else None
    volume_data = None  # volume not cached yet; strategies handle missing volume

    flags = []
    if args.no_spy_parking:
        flags.append("no-parking")
    if args.cautious_filter:
        flags.append("cautious-filter")
    flag_str = f", flags=[{','.join(flags)}]" if flags else ""
    log(f"Strategy: {args.strategy} ({', '.join(strat_names)}) "
        f"[{args.selection_mode}, top_n={args.top_n}, thresh={args.ml_threshold}, vol_sizing={args.vol_sizing}{flag_str}]")
    strategies = build_strategies(
        strat_names, preds, price_data, volume_data,
        args.ml_threshold, args.selection_mode, args.top_n,
        momentum_regime_filter=args.momentum_regime_filter,
        cautious_filter=args.cautious_filter, regime_dict=regime_dict)

    # ── 4. Run backtest ─────────────────────────────────────────────────
    spy_prices_arg = None if args.no_spy_parking else spy_dict
    log("Running backtest ...")
    pm = PortfolioManager(strategies=strategies, slot_config=slot_config)
    do_detail = args.export_logs is not None
    result = pm.run(all_dates, spy_prices=spy_prices_arg, price_data=price_data,
                    detail_log=do_detail)
    if do_detail:
        vals, trades, equity_df, trade_log_df = result
    else:
        vals, trades = result

    # ── 5. Compute metrics ──────────────────────────────────────────────
    if args.selection_mode == "top_n":
        label = f"{args.strategy} (top-{args.top_n})"
    else:
        label = f"{args.strategy} (>{args.ml_threshold:.2f})"
    metrics = calc_metrics(vals, trades, years, label)
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None

    # ── 6. Print results ────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Final value    : ${metrics['final_value']:>12,.0f}")
    print(f"  CAGR           : {metrics['cagr']:>+8.2%}")
    print(f"  Sharpe ratio   : {metrics['sharpe']:>8.3f}")
    print(f"  Sortino ratio  : {metrics['sortino']:>8.3f}")
    print(f"  Max drawdown   : {metrics['max_dd']:>8.1%}")
    print(f"  Total trades   : {metrics['n_trades']:>8,}")
    print(f"  Win rate       : {metrics['win_rate']:>8.1%}")
    if metrics["alpha"] is not None:
        print(f"  Alpha vs SPY   : {metrics['alpha']:>+8.2%}")
    print(f"  Runtime        : {elapsed:>8.1f}s")
    print(f"{'='*55}")

    # ── 7. Optionally write JSON ────────────────────────────────────────
    if args.output:
        metrics["runtime_s"] = round(elapsed, 1)
        # Convert numpy types for JSON
        out = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in metrics.items()}
        Path(args.output).write_text(json.dumps(out, indent=2))
        log(f"Metrics written to {args.output}")

    # ── 8. Optionally export detailed logs ─────────────────────────────
    if args.export_logs:
        export_dir = Path(args.export_logs)
        export_dir.mkdir(parents=True, exist_ok=True)
        equity_df.to_parquet(export_dir / "equity_curve.parquet", index=False)
        if not trade_log_df.empty:
            trade_log_df.to_parquet(export_dir / "trade_log.parquet", index=False)
        log(f"Logs exported to {export_dir}")


if __name__ == "__main__":
    main()
