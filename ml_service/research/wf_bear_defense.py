#!/usr/bin/env python3
"""
Walk-Forward Test: Bear Market Defense Variants
=================================================
Tests multiple bear-defense strategies against the current live baseline.

Variants:
  1. Baseline        — current live production (no changes)
  2. Regime buffer   — block ML buys if BEARISH in past 5 days
  3. VIX scaling     — half-size when VIX>25, quarter-size when VIX>35
  4. Drawdown brake  — stop buying if portfolio drops >8% from peak, resume after 5% recovery
  5. Adaptive hold   — 5-day hold in CAUTIOUS/BEARISH, 10-day in BULLISH
  6. Combined        — regime buffer + VIX scaling + drawdown brake
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
    MLMediumStrategy, CautiousMLStrategy, compute_regime_live,
    MomentumStrategy,
    PortfolioManager, SlotConfig, Strategy, Signal, Position,
    load_bars_cached,
    INITIAL_CASH, SLIPPAGE, HOLD_DAYS, POSITION_PCT,
    COOLDOWN_DAYS, MIN_POSITION_DOLLARS, NEVER_BUY,
    SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
)
from backtest_utils import calc_metrics, calc_alpha_beta

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"
YEARS = list(range(2015, 2026))

SLOT_LIVE = SlotConfig(
    strategy_slots={"ml_medium": 5, "momentum": 3, "mean_reversion": 0, "mega_cap": 0},
    flex_slots=0,
    max_positions=8,
)


def log(msg: str):
    print(msg, flush=True)


def _fix_prob_col(df):
    if "prob_ensemble" in df.columns and "prob" not in df.columns:
        df = df.rename(columns={"prob_ensemble": "prob"})
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Bear Defense: Regime Buffer Strategy
# ══════════════════════════════════════════════════════════════════════════════

class RegimeBufferMLStrategy(CautiousMLStrategy):
    """Block ML buys if regime was BEARISH any time in the past N days."""

    def __init__(self, predictions_df, regime_dict, buffer_days=5, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._buffer_days = buffer_days
        self._regime_series = pd.Series(regime_dict).sort_index()

    def generate_signals(self, date, universe_data):
        # Check if BEARISH in past buffer_days
        ts = pd.Timestamp(date)
        lookback = self._regime_series[
            (self._regime_series.index <= ts) &
            (self._regime_series.index >= ts - pd.Timedelta(days=self._buffer_days * 2))
        ].tail(self._buffer_days)

        if (lookback == "BEARISH").any():
            return []  # block all ML buys

        return super().generate_signals(date, universe_data)


# ══════════════════════════════════════════════════════════════════════════════
#  Bear Defense: VIX-Scaled ML Strategy
# ══════════════════════════════════════════════════════════════════════════════

class VIXScaledMLStrategy(CautiousMLStrategy):
    """Scale position sizes down when VIX is elevated."""

    def __init__(self, predictions_df, regime_dict, vix_data, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._vix = vix_data  # {date → VIX close}

    def get_position_size(self, signal, portfolio_value, date=None):
        base_size = super().get_position_size(signal, portfolio_value, date=date)
        if date is None:
            return base_size

        vix = self._vix.get(pd.Timestamp(date), None)
        if vix is None:
            # Try nearby dates
            ts = pd.Timestamp(date)
            for offset in range(1, 4):
                for d in [ts - pd.Timedelta(days=offset), ts + pd.Timedelta(days=offset)]:
                    vix = self._vix.get(d, None)
                    if vix is not None:
                        break
                if vix is not None:
                    break

        if vix is not None:
            if vix > 35:
                return base_size * 0.25
            elif vix > 25:
                return base_size * 0.50

        return base_size


# ══════════════════════════════════════════════════════════════════════════════
#  Bear Defense: Adaptive Hold ML Strategy
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveHoldMLStrategy(CautiousMLStrategy):
    """Use shorter hold period during CAUTIOUS/BEARISH regimes."""

    def __init__(self, predictions_df, regime_dict, **kwargs):
        super().__init__(predictions_df, regime_dict=regime_dict, **kwargs)
        self._regime_dict = regime_dict

    def check_exit(self, position, current_data):
        date = current_data["date"]
        regime = self._regime_dict.get(pd.Timestamp(date), "BULLISH")

        # Shorter hold in non-bullish regimes
        if regime in ("CAUTIOUS", "BEARISH"):
            hold_days = 5
        else:
            hold_days = 10

        days_held = current_data["idx"] - position.entry_idx
        if days_held >= hold_days:
            return True, "hold_complete"
        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
#  Bear Defense: Drawdown Brake Portfolio Manager
# ══════════════════════════════════════════════════════════════════════════════

class DrawdownBrakePM(PortfolioManager):
    """
    Wraps PortfolioManager with a portfolio-level drawdown brake:
    if portfolio drops >dd_threshold from peak, block new buys until
    portfolio recovers to within recovery_threshold of peak.
    """

    def __init__(self, strategies, slot_config, dd_threshold=-0.08,
                 recovery_threshold=-0.03, **kwargs):
        super().__init__(strategies, slot_config, **kwargs)
        self.dd_threshold = dd_threshold
        self.recovery_threshold = recovery_threshold

    def run(self, all_dates, spy_prices=None, price_data=None, detail_log=False):
        """Override run to inject drawdown brake logic."""
        # We need to intercept the signal filtering step.
        # The cleanest way is to track peak/drawdown and filter signals.

        # Save original generate_signals methods
        orig_generators = {}
        for name, strat in self.strategies.items():
            orig_generators[name] = strat.generate_signals

        peak_value = self.initial_cash
        braking = False
        pm_self = self

        class SignalBlocker:
            """Wraps a strategy to block signals during drawdown brake."""
            def __init__(self, strat, strat_name):
                self.strat = strat
                self.strat_name = strat_name
                self.orig_fn = strat.generate_signals
                self.blocked = False

            def generate_signals_wrapper(self, date, universe_data):
                if self.blocked:
                    return []
                return self.orig_fn(date, universe_data)

        blockers = {}
        for name, strat in self.strategies.items():
            b = SignalBlocker(strat, name)
            blockers[name] = b
            strat.generate_signals = b.generate_signals_wrapper

        # Run the base PM
        result = super().run(all_dates, spy_prices=spy_prices,
                             price_data=price_data, detail_log=detail_log)

        # Restore original methods
        for name, strat in self.strategies.items():
            strat.generate_signals = orig_generators[name]

        return result

    # Actually, the above approach doesn't work because we can't update
    # the blockers mid-run from outside the loop. Let me take a different approach:
    # override the entire run method with drawdown tracking injected.

    def run(self, all_dates, spy_prices=None, price_data=None, detail_log=False):
        """Full run override with drawdown brake."""
        from unified_backtester import USE_VOL_SIZING

        n_dates = len(all_dates)
        multi = len(self.strategies) > 1

        # Compute global ATR if needed
        import unified_backtester
        if USE_VOL_SIZING and price_data is not None:
            daily_ret_abs = price_data.pct_change().abs()
            unified_backtester._global_atr_df = daily_ret_abs.rolling(14, min_periods=14).mean()

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

        trade_details = [] if detail_log else None
        equity_records = [] if detail_log else None

        # Drawdown brake state
        peak_value = self.initial_cash
        braking = False

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

            # Close expiring positions
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
                if trade_details is not None:
                    trade_details.append({
                        "entry_date": all_dates[pos.entry_idx], "exit_date": date,
                        "symbol": sym, "strategy": pos.strategy_name, "action": "sell",
                        "cost": pos.cost, "entry_price": pos.entry_price,
                        "exit_price": day_prices.get(sym, pos.entry_price) if pos.price_based else 0.0,
                        "gross": gross, "net": net, "pnl": net - pos.cost,
                        "ret": (net - pos.cost) / pos.cost,
                        "hold_days": i - pos.entry_idx, "price_based": pos.price_based,
                        "exit_reason": exit_reason, "peak_price": pos.peak_price,
                    })
                if multi:
                    cooldowns[(sym, pos.strategy_name)] = i + COOLDOWN_DAYS

            # ── Drawdown brake check ──
            # Estimate current portfolio value for brake decision
            est_port = cash + (idle_spy_shares * spy_px if spy_px else 0)
            for pos in positions.values():
                if pos.price_based:
                    px = day_prices.get(pos.symbol, pos.entry_price)
                    est_port += pos.cost * (px / pos.entry_price if pos.entry_price > 0 else 1.0)
                else:
                    days_held = i - pos.entry_idx
                    interp_ret = pos.fwd_ret * days_held / self.hold_days
                    est_port += pos.cost * (1.0 + interp_ret)

            if est_port > peak_value:
                peak_value = est_port
            dd = (est_port - peak_value) / peak_value if peak_value > 0 else 0

            if not braking and dd <= self.dd_threshold:
                braking = True
            elif braking and dd >= self.recovery_threshold:
                braking = False

            # Gather signals (skip if braking)
            if braking:
                all_signals = []
            else:
                all_signals = []
                for strat in self.strategies.values():
                    all_signals.extend(strat.generate_signals(date, None))

            # Filter
            held = set(positions.keys())
            all_signals = [s for s in all_signals
                           if s.symbol not in held
                           and s.symbol not in NEVER_BUY
                           and (s.price_based or not np.isnan(s.fwd_ret))]
            if multi:
                all_signals = [s for s in all_signals
                               if cooldowns.get((s.symbol, s.strategy_name), -1) <= i]

            # Resolve overlaps
            seen_syms = set()
            resolved = []
            for sig in all_signals:
                if sig.symbol not in seen_syms:
                    seen_syms.add(sig.symbol)
                    resolved.append(sig)
            resolved.sort(key=lambda s: s.confidence, reverse=True)

            # Execute buys
            max_slots = self.slot_config.max_positions - len(positions)

            # Sector diversification (from base PM)
            sector_bought_today = {}
            SECTOR_MAP = {}  # simplified — no sector limit for this test

            for sig in resolved[:max_slots]:
                if sig.price_based:
                    entry_px = day_prices.get(sig.symbol, np.nan)
                    if np.isnan(entry_px) or entry_px <= 0:
                        continue
                else:
                    entry_px = 0.0
                    if np.isnan(sig.fwd_ret):
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
                    symbol=sig.symbol, strategy_name=sig.strategy_name,
                    cost=cost, entry_idx=i, exit_idx=exit_idx,
                    fwd_ret=sig.fwd_ret, confidence=sig.confidence,
                    price_based=sig.price_based, entry_price=entry_px,
                    peak_price=entry_px,
                )
                if trade_details is not None:
                    trade_details.append({
                        "entry_date": date, "exit_date": None,
                        "symbol": sig.symbol, "strategy": sig.strategy_name,
                        "action": "buy", "cost": cost, "entry_price": entry_px,
                        "exit_price": 0.0, "gross": 0.0, "net": 0.0, "pnl": 0.0,
                        "ret": 0.0, "hold_days": 0, "price_based": sig.price_based,
                        "exit_reason": "", "peak_price": entry_px,
                    })

            # Mark-to-market
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
            if equity_records is not None:
                equity_records.append({
                    "date": date, "portfolio_value": port_val,
                    "spy_value": spy_px if spy_px else np.nan,
                    "cash": cash, "n_positions": len(positions),
                })

        series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates))
        if detail_log:
            eq_df = pd.DataFrame(equity_records)
            td_df = pd.DataFrame(trade_details) if trade_details else pd.DataFrame()
            return series, trades, eq_df, td_df
        return series, trades


# ══════════════════════════════════════════════════════════════════════════════
#  Config runner
# ══════════════════════════════════════════════════════════════════════════════

def run_config(preds_df, year, close, regime_dict, vix_data, config):
    """Run one year with a specific bear defense config."""
    preds_df = _fix_prob_col(preds_df)
    all_dates = sorted(preds_df["date"].unique().tolist())
    if len(all_dates) < 10:
        return None

    years_span = (all_dates[-1] - all_dates[0]).days / 365.25
    if years_span <= 0:
        years_span = len(all_dates) / 252.0

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        return None
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    name = config["name"]

    # Build ML strategy variant
    if name == "baseline":
        ml = CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                                threshold=0.55, top_n=5, selection_mode="top_n")
    elif name == "regime_buffer":
        ml = RegimeBufferMLStrategy(preds_df, regime_dict=regime_dict,
                                    buffer_days=5,
                                    threshold=0.55, top_n=5, selection_mode="top_n")
    elif name == "vix_scaling":
        ml = VIXScaledMLStrategy(preds_df, regime_dict=regime_dict,
                                 vix_data=vix_data,
                                 threshold=0.55, top_n=5, selection_mode="top_n")
    elif name == "adaptive_hold":
        ml = AdaptiveHoldMLStrategy(preds_df, regime_dict=regime_dict,
                                    threshold=0.55, top_n=5, selection_mode="top_n")
    elif name in ("dd_brake", "combined"):
        ml = CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                                threshold=0.55, top_n=5, selection_mode="top_n")
        if name == "combined":
            ml = RegimeBufferMLStrategy(preds_df, regime_dict=regime_dict,
                                        buffer_days=5,
                                        threshold=0.55, top_n=5, selection_mode="top_n")
            # VIX scaling on top
            ml_base = ml
            class CombinedML(RegimeBufferMLStrategy):
                def __init__(self, base, vix):
                    # Copy all state from base
                    self.__dict__.update(base.__dict__)
                    self._vix = vix
                def get_position_size(self, signal, portfolio_value, date=None):
                    base_size = super().get_position_size(signal, portfolio_value, date=date)
                    if date is None:
                        return base_size
                    vix = self._vix.get(pd.Timestamp(date), None)
                    if vix is None:
                        ts = pd.Timestamp(date)
                        for offset in range(1, 4):
                            for d in [ts - pd.Timedelta(days=offset), ts + pd.Timedelta(days=offset)]:
                                vix = self._vix.get(d, None)
                                if vix is not None:
                                    break
                            if vix is not None:
                                break
                    if vix is not None:
                        if vix > 35:
                            return base_size * 0.25
                        elif vix > 25:
                            return base_size * 0.50
                    return base_size
            ml = CombinedML(ml_base, vix_data)

    mom = MomentumStrategy(close, volume_data=None, regime_filter=True)
    strategies = [ml, mom]

    # Use drawdown brake PM for dd_brake and combined
    if name in ("dd_brake", "combined"):
        pm = DrawdownBrakePM(strategies=strategies, slot_config=SLOT_LIVE,
                             dd_threshold=-0.08, recovery_threshold=-0.03)
    else:
        pm = PortfolioManager(strategies=strategies, slot_config=SLOT_LIVE)

    vals, trades = pm.run(all_dates, spy_prices=None, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"{name}_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Configs
# ══════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    {"name": "baseline",      "label": "Baseline (current live)", "short": "Base"},
    {"name": "regime_buffer",  "label": "Regime buffer (5d)",     "short": "RegBuf"},
    {"name": "vix_scaling",    "label": "VIX scaling",            "short": "VIX"},
    {"name": "dd_brake",       "label": "Drawdown brake (8%)",    "short": "DDBrk"},
    {"name": "adaptive_hold",  "label": "Adaptive hold (5/10d)",  "short": "AdpHld"},
    {"name": "combined",       "label": "RegBuf+VIX+DDBrake",    "short": "Combi"},
]


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  BEAR MARKET DEFENSE — WALK-FORWARD COMPARISON")
    log("=" * 70)

    # Load walk-forward predictions
    all_preds = {}
    for year in YEARS:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            log(f"  SKIP {year}: no predictions")
            continue
        df = pd.read_parquet(pred_file)
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = _fix_prob_col(df)
        log(f"  Loaded predictions_{year}.parquet ({len(df):,} rows)")

    all_syms = sorted(set().union(*(df["symbol"].unique() for df in all_preds.values())))

    # Load price data
    log(f"\nLoading price bars for {len(all_syms)} symbols ...")
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    log(f"  Price data: {close.shape}")

    # Load VIX
    log("Loading VIX data ...")
    import yfinance as yf
    vix_raw = yf.download("^VIX", start="2013-01-01", end="2025-12-31", progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_close = vix_raw["Close"].squeeze()
    else:
        vix_close = vix_raw["Close"]
    vix_data = vix_close.to_dict()
    log(f"  VIX data: {len(vix_data)} days")

    # Regime
    spy_full = close["SPY"].dropna()
    regime_series = compute_regime_live(spy_full)
    regime_dict = regime_series.to_dict()

    # Run all configs × all years
    all_results = {cfg["name"]: [] for cfg in CONFIGS}

    for year in YEARS:
        if year not in all_preds:
            continue
        preds_df = all_preds[year]
        preds_year = preds_df[preds_df["date"].dt.year == year]
        if preds_year.empty:
            continue

        all_dates = sorted(preds_year["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        # Regime stats
        yr = regime_series[(regime_series.index >= pd.Timestamp(f"{year}-01-01"))
                           & (regime_series.index <= pd.Timestamp(f"{year}-12-31"))]
        vc = yr.value_counts()
        log(f"\n  {year}  BULL={vc.get('BULLISH', 0)}  CAUT={vc.get('CAUTIOUS', 0)}  BEAR={vc.get('BEARISH', 0)}")

        for cfg in CONFIGS:
            metrics = run_config(preds_year, year, close_year, regime_dict,
                                 vix_data, cfg)
            if metrics:
                all_results[cfg["name"]].append(metrics)
                log(f"    {cfg['short']:8s}  CAGR={metrics['cagr']:+.1%}  "
                    f"Sharpe={metrics['sharpe']:.2f}  MaxDD={metrics['max_dd']:.1%}  "
                    f"Alpha={metrics.get('alpha', 0):+.1%}")

    # ══════════════════════════════════════════════════════════════════════
    #  Report
    # ══════════════════════════════════════════════════════════════════════
    n = len(all_results["baseline"])
    if n == 0:
        log("No results.")
        return

    print("\n" + "=" * 90)
    print("BEAR DEFENSE — WALK-FORWARD RESULTS")
    print("=" * 90)

    # Summary table
    print(f"\n{'Metric':<22s}", end="")
    for cfg in CONFIGS:
        print(f"  {cfg['short']:>8s}", end="")
    print()
    print("─" * (22 + 10 * len(CONFIGS)))

    def row(label, values, fmt):
        print(f"  {label:<20s}", end="")
        for v in values:
            print(f"  {fmt.format(v):>8s}", end="")
        print()

    def get_vals(field):
        return [
            [m[field] for m in all_results[cfg["name"]]]
            for cfg in CONFIGS
        ]

    cagrs = get_vals("cagr")
    sharpes = get_vals("sharpe")
    dds = get_vals("max_dd")
    alphas_raw = get_vals("alpha")
    alphas = [[a for a in al if a is not None] for al in alphas_raw]

    row("Median CAGR", [np.median(c) for c in cagrs], "{:+.1%}")
    row("Mean CAGR", [np.mean(c) for c in cagrs], "{:+.1%}")
    row("Median Sharpe", [np.median(c) for c in sharpes], "{:.2f}")
    row("Mean Sharpe", [np.mean(c) for c in sharpes], "{:.2f}")
    row("Worst Max DD", [min(c) for c in dds], "{:.1%}")
    row("Mean Max DD", [np.mean(c) for c in dds], "{:.1%}")
    if all(len(a) > 0 for a in alphas):
        row("Median Alpha", [np.median(a) for a in alphas], "{:+.1%}")
    pos_cagr = [sum(1 for c in cs if c > 0) for cs in cagrs]
    print(f"  {'Positive CAGR yrs':<20s}", end="")
    for p in pos_cagr:
        print(f"  {p:>5d}/{n:<2d}", end="")
    print()

    # Per-year head-to-head
    print(f"\n{'Year':<6s}", end="")
    for cfg in CONFIGS:
        print(f"  {cfg['short']:>8s}", end="")
    print(f"  {'Best':>8s}")
    print("─" * (6 + 10 * len(CONFIGS) + 10))

    for yi in range(n):
        yr = all_results["baseline"][yi]["year"]
        year_cagrs = [all_results[cfg["name"]][yi]["cagr"] for cfg in CONFIGS]
        best_idx = int(np.argmax(year_cagrs))
        print(f"{yr:<6d}", end="")
        for c in year_cagrs:
            print(f"  {c:>+8.1%}", end="")
        print(f"  {CONFIGS[best_idx]['short']:>8s}")

    # Delta vs baseline
    print(f"\n{'─' * 70}")
    print("DELTA vs BASELINE")
    print(f"{'Variant':<22s} {'ΔCAGR':>8s} {'ΔSharpe':>8s} {'ΔWorstDD':>9s} {'Sharpe↑':>8s} {'Sharpe↓':>8s}")
    print("─" * 70)

    base_cagrs = cagrs[0]
    base_sharpes = sharpes[0]
    for ci in range(1, len(CONFIGS)):
        cfg = CONFIGS[ci]
        d_cagr = np.median(cagrs[ci]) - np.median(base_cagrs)
        d_sharpe = np.median(sharpes[ci]) - np.median(base_sharpes)
        d_dd = min(dds[ci]) - min(dds[0])
        helped = sum(1 for a, b in zip(base_sharpes, sharpes[ci]) if b > a)
        hurt = sum(1 for a, b in zip(base_sharpes, sharpes[ci]) if b < a)
        print(f"  {cfg['label']:<20s} {d_cagr:>+8.1%} {d_sharpe:>+8.2f} {d_dd:>+9.1%} {helped:>5d}/{n:<2d} {hurt:>5d}/{n:<2d}")

    # 2022 focus
    print(f"\n{'─' * 70}")
    print("2022 BEAR MARKET FOCUS")
    print(f"{'Variant':<22s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Alpha':>8s}")
    print("─" * 70)
    for cfg in CONFIGS:
        m = next((m for m in all_results[cfg["name"]] if m["year"] == 2022), None)
        if m:
            a = f"{m['alpha']:+.1%}" if m.get("alpha") is not None else "N/A"
            print(f"  {cfg['label']:<20s} {m['cagr']:>+8.1%} {m['sharpe']:>8.2f} {m['max_dd']:>8.1%} {a:>8s}")

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save
    out = {cfg["name"]: all_results[cfg["name"]] for cfg in CONFIGS}
    out_path = WF_DIR / "bear_defense_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
