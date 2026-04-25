#!/usr/bin/env python3
"""
Clean Walk-Forward A/B Comparison
=================================
Config A (current LIVE): SLOT_LIVE_V2, momentum regime filter ON, SPY parking ON,
                         ML top_n=5, CAUTIOUS top-2 ML filter
Config B (proposed NEW):  same as A + regime-aware sizing (conservative 8/5/2)

Uses existing walk-forward predictions (trained OOS per year).
Only difference: portfolio construction logic.
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
    MegaCapStrategy, PortfolioManager, SlotConfig, Signal,
    SLOT_LIVE_V2, load_bars_cached,
    INITIAL_CASH, SLIPPAGE, HOLD_DAYS,
    Position, NEVER_BUY, COOLDOWN_DAYS, MIN_POSITION_DOLLARS,
    POSITION_PCT, SYMBOL_SECTOR, SECTOR_MAX_POSITIONS,
    USE_VOL_SIZING, _has_earnings_within,
    SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
    IDLE_SPY_MIN_OPP_RATIO,
)
from backtest_utils import calc_metrics, calc_alpha_beta

DATA_DIR = Path(__file__).resolve().parent / "data"
WF_DIR = DATA_DIR / "walkforward"

YEARS = list(range(2015, 2026))


def log(msg: str):
    print(msg, flush=True)


# ── Regime Detection ────────────────────────────────────────────────────────

def compute_regime_live(spy_series: pd.Series) -> pd.Series:
    """
    Live production regime (tradingEngine.js lines 688-709):
      BULLISH:  SPY > SMA50 AND SPY > SMA200
      CAUTIOUS: SPY > SMA200 (but not above SMA50)
      BEARISH:  SPY <= SMA200
    """
    sma50 = spy_series.rolling(50, min_periods=50).mean()
    sma200 = spy_series.rolling(200, min_periods=200).mean()

    regime = pd.Series("BEARISH", index=spy_series.index)

    cautious = spy_series > sma200
    bullish = (spy_series > sma50) & (spy_series > sma200)

    regime[cautious] = "CAUTIOUS"
    regime[bullish] = "BULLISH"

    return regime


# ── CAUTIOUS ML Filter (matches live tradingEngine.js line 2394) ────────────

class CautiousMLStrategy(MLMediumStrategy):
    """
    Wraps MLMediumStrategy with the live CAUTIOUS regime filter:
    during CAUTIOUS regime, only ML picks with rank <= 2 are allowed.
    """

    def __init__(self, predictions_df, regime_dict, threshold=0.55, top_n=5,
                 selection_mode="top_n", position_pct=POSITION_PCT):
        super().__init__(predictions_df, threshold, top_n, selection_mode, position_pct)
        self._regime_dict = regime_dict

    def generate_signals(self, date, universe_data):
        regime = self._regime_dict.get(date, "CAUTIOUS")
        if regime == "CAUTIOUS":
            # Live behavior: only top 2 ML picks allowed in CAUTIOUS
            raw = self._signals_by_date.get(date, [])
            selected = raw[:2]  # rank 1-2 only
            return [
                Signal(symbol=sym, confidence=prob,
                       strategy_name=self.name, fwd_ret=fwd_ret)
                for sym, prob, fwd_ret in selected
            ]
        return super().generate_signals(date, universe_data)


# ── Config A: Current Live ──────────────────────────────────────────────────

def run_config_a(preds_df, year, close, spy_full):
    """
    Config A — actual current live production:
      SLOT_LIVE_V2 (ml:5, mom:3, max:8)
      Momentum regime filter: ON
      ML top_n: 5 (matches live MOM.TOP_N)
      SPY parking: ON (live has idle-SPY parking enabled)
      CAUTIOUS ML filter: top 2 only (live line 2394)
      Regime-aware sizing: OFF (no dynamic max_positions)
    """
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

    if "prob_ensemble" in preds_df.columns and "prob" not in preds_df.columns:
        preds_df = preds_df.rename(columns={"prob_ensemble": "prob"})

    # Compute live regime for CAUTIOUS ML filter
    regime_live = compute_regime_live(spy_full)
    regime_dict = regime_live.to_dict()

    strategies = [
        CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                           threshold=0.55, top_n=5, selection_mode="top_n"),
        MomentumStrategy(close, volume_data=None, regime_filter=True),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]

    pm = PortfolioManager(strategies=strategies, slot_config=SLOT_LIVE_V2)
    # SPY parking ON (matches live production)
    vals, trades = pm.run(all_dates, spy_prices=spy_dict, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"config_a_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    return metrics


# ── Config B: Proposed New (regime sizing 8/5/2) ───────────────────────────

REGIME_LIMITS = {"BULLISH": 8, "CAUTIOUS": 5, "BEARISH": 2}


class RegimeSizingPM(PortfolioManager):
    """Config B: same as A but with regime-aware max_positions."""

    def __init__(self, strategies, slot_config, regime_dict,
                 initial_cash=INITIAL_CASH, slippage=SLIPPAGE,
                 hold_days=HOLD_DAYS):
        super().__init__(strategies, slot_config, initial_cash, slippage, hold_days)
        self.regime_dict = regime_dict
        self._base_max = slot_config.max_positions

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

            # Regime-aware max positions
            regime = self.regime_dict.get(date, "CAUTIOUS")
            effective_max = REGIME_LIMITS.get(regime, self._base_max)

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

            # 5. Execute buys — regime-aware max_slots
            max_slots = max(0, effective_max - len(positions))

            # Release idle SPY before buying (same as standard PM)
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

            # 6. Park idle cash in SPY (same as standard PM)
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


def run_config_b(preds_df, year, close, spy_full):
    """
    Config B — proposed new:
      Same as Config A + regime-aware sizing (conservative 8/5/2)
      Uses live regime definition for CAUTIOUS ML filter.
      Uses live regime definition for position sizing too (same engine).
      BULLISH: max 8, CAUTIOUS: max 5, BEARISH: max 2
    """
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

    # Compute live regime (same engine as production)
    regime_live = compute_regime_live(spy_full)
    regime_dict = regime_live.to_dict()

    if "prob_ensemble" in preds_df.columns and "prob" not in preds_df.columns:
        preds_df = preds_df.rename(columns={"prob_ensemble": "prob"})

    strategies = [
        CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                           threshold=0.55, top_n=5, selection_mode="top_n"),
        MomentumStrategy(close, volume_data=None, regime_filter=True),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]

    pm = RegimeSizingPM(
        strategies=strategies,
        slot_config=SLOT_LIVE_V2,
        regime_dict=regime_dict,
    )
    # SPY parking ON (same as Config A)
    vals, trades = pm.run(all_dates, spy_prices=spy_dict, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"config_b_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    return metrics


def generate_report(results_a, results_b, regime_stats):
    """Generate AB_COMPARISON.md."""
    lines = []
    lines.append("# Config A (Current Live) vs Config B (Proposed New) — Walk-Forward")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Methodology")
    lines.append("Proper walk-forward: year Y uses model trained only on data <= Y-1.")
    lines.append("Same universe, features, model. Only difference is strategy/portfolio construction.")

    lines.append("\n## Configuration Details")
    lines.append("")
    lines.append("| Setting | Config A (Current Live) | Config B (Proposed New) |")
    lines.append("|---------|----------------------|----------------------|")
    lines.append("| Slot config | ml:5, mom:3, max:8 | ml:5, mom:3, max:8 |")
    lines.append("| Momentum regime filter | ON (SPY < 50-SMA blocks mom buys) | ON |")
    lines.append("| CAUTIOUS ML filter | ON (top 2 only) | ON (top 2 only) |")
    lines.append("| ML top_n | 5 | 5 |")
    lines.append("| Regime-aware sizing | OFF | ON (8/5/2) |")
    lines.append("| SPY parking | ON (30% reserve, 20% threshold) | ON |")

    lines.append("\n## Regime Definition (Live Production)")
    lines.append("- **BULLISH**: SPY > SMA50 AND SPY > SMA200")
    lines.append("- **CAUTIOUS**: SPY > SMA200 (but below SMA50)")
    lines.append("- **BEARISH**: SPY <= SMA200")

    # Regime distribution
    lines.append("\n## Regime Distribution by Year")
    lines.append("")
    lines.append("| Year | BULLISH | CAUTIOUS | BEARISH | % BULLISH |")
    lines.append("|------|---------|----------|---------|-----------|")
    total_bull = total_caut = total_bear = 0
    for year in sorted(regime_stats.keys()):
        s = regime_stats[year]
        total = s["BULLISH"] + s["CAUTIOUS"] + s["BEARISH"]
        pct = s["BULLISH"] / total * 100 if total > 0 else 0
        total_bull += s["BULLISH"]
        total_caut += s["CAUTIOUS"]
        total_bear += s["BEARISH"]
        lines.append(f"| {year} | {s['BULLISH']}d | {s['CAUTIOUS']}d | {s['BEARISH']}d | {pct:.0f}% |")
    total_all = total_bull + total_caut + total_bear
    lines.append(f"| **Total** | **{total_bull}d** | **{total_caut}d** | **{total_bear}d** | **{total_bull/total_all*100:.0f}%** |")

    # Per-year comparison
    lines.append("\n## Per-Year Comparison")
    lines.append("")
    lines.append("| Year | Regime | A CAGR | B CAGR | Delta | A Sharpe | B Sharpe | A MaxDD | B MaxDD |")
    lines.append("|------|--------|--------|--------|-------|----------|----------|---------|---------|")

    for ma, mb in zip(results_a, results_b):
        year = ma["year"]
        s = regime_stats[year]
        total = s["BULLISH"] + s["CAUTIOUS"] + s["BEARISH"]
        regime_str = f"B{s['BULLISH']/total*100:.0f}/C{s['CAUTIOUS']/total*100:.0f}/R{s['BEARISH']/total*100:.0f}"
        delta_cagr = mb["cagr"] - ma["cagr"]
        lines.append(
            f"| {year} "
            f"| {regime_str} "
            f"| {ma['cagr']:+.1%} "
            f"| {mb['cagr']:+.1%} "
            f"| {delta_cagr:+.1%} "
            f"| {ma['sharpe']:.2f} "
            f"| {mb['sharpe']:.2f} "
            f"| {ma['max_dd']:.1%} "
            f"| {mb['max_dd']:.1%} |"
        )

    # Summary metrics
    lines.append("\n## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Config A | Config B | Delta |")
    lines.append("|--------|----------|----------|-------|")

    cagrs_a = [m["cagr"] for m in results_a]
    cagrs_b = [m["cagr"] for m in results_b]
    sharpes_a = [m["sharpe"] for m in results_a]
    sharpes_b = [m["sharpe"] for m in results_b]
    dds_a = [m["max_dd"] for m in results_a]
    dds_b = [m["max_dd"] for m in results_b]
    alphas_a = [m["alpha"] for m in results_a if m.get("alpha") is not None]
    alphas_b = [m["alpha"] for m in results_b if m.get("alpha") is not None]
    betas_a = [m["beta"] for m in results_a if m.get("beta") is not None]
    betas_b = [m["beta"] for m in results_b if m.get("beta") is not None]

    def fmt_delta(a, b, fmt_str):
        return f"{fmt_str.format(b - a)}"

    lines.append(f"| Median CAGR | {np.median(cagrs_a):+.1%} | {np.median(cagrs_b):+.1%} | {np.median(cagrs_b)-np.median(cagrs_a):+.1%} |")
    lines.append(f"| Mean CAGR | {np.mean(cagrs_a):+.1%} | {np.mean(cagrs_b):+.1%} | {np.mean(cagrs_b)-np.mean(cagrs_a):+.1%} |")
    lines.append(f"| Median Sharpe | {np.median(sharpes_a):.2f} | {np.median(sharpes_b):.2f} | {np.median(sharpes_b)-np.median(sharpes_a):+.2f} |")
    lines.append(f"| Mean Sharpe | {np.mean(sharpes_a):.2f} | {np.mean(sharpes_b):.2f} | {np.mean(sharpes_b)-np.mean(sharpes_a):+.02f} |")
    lines.append(f"| Worst Max DD | {min(dds_a):.1%} | {min(dds_b):.1%} | {min(dds_b)-min(dds_a):+.1%} |")
    lines.append(f"| Mean Max DD | {np.mean(dds_a):.1%} | {np.mean(dds_b):.1%} | {np.mean(dds_b)-np.mean(dds_a):+.1%} |")
    if alphas_a and alphas_b:
        lines.append(f"| Median Alpha | {np.median(alphas_a):+.1%} | {np.median(alphas_b):+.1%} | {np.median(alphas_b)-np.median(alphas_a):+.1%} |")
    if betas_a and betas_b:
        lines.append(f"| Mean Beta | {np.mean(betas_a):.2f} | {np.mean(betas_b):.2f} | {np.mean(betas_b)-np.mean(betas_a):+.02f} |")

    pos_cagr_a = sum(1 for c in cagrs_a if c > 0)
    pos_cagr_b = sum(1 for c in cagrs_b if c > 0)
    pos_alpha_a = sum(1 for a in alphas_a if a > 0) if alphas_a else 0
    pos_alpha_b = sum(1 for a in alphas_b if a > 0) if alphas_b else 0
    n = len(results_a)
    lines.append(f"| Positive CAGR years | {pos_cagr_a}/{n} | {pos_cagr_b}/{n} | {pos_cagr_b-pos_cagr_a:+d} |")
    lines.append(f"| Alpha-positive years | {pos_alpha_a}/{n} | {pos_alpha_b}/{n} | {pos_alpha_b-pos_alpha_a:+d} |")

    mean_trades_a = np.mean([m["n_trades"] for m in results_a])
    mean_trades_b = np.mean([m["n_trades"] for m in results_b])
    lines.append(f"| Mean trades/year | {mean_trades_a:.0f} | {mean_trades_b:.0f} | {mean_trades_b-mean_trades_a:+.0f} |")

    mean_wr_a = np.mean([m["win_rate"] for m in results_a])
    mean_wr_b = np.mean([m["win_rate"] for m in results_b])
    lines.append(f"| Mean win rate | {mean_wr_a:.1%} | {mean_wr_b:.1%} | {mean_wr_b-mean_wr_a:+.1%} |")

    # Bear market focus
    lines.append("\n## Bear Market Focus (2018, 2022)")
    for target_year in [2018, 2022]:
        ma = next((m for m in results_a if m["year"] == target_year), None)
        mb = next((m for m in results_b if m["year"] == target_year), None)
        if ma and mb:
            s = regime_stats[target_year]
            lines.append(f"\n### {target_year} (BEAR={s['BEARISH']}d, CAUT={s['CAUTIOUS']}d)")
            lines.append(f"- Config A: CAGR={ma['cagr']:+.1%}, Sharpe={ma['sharpe']:.2f}, DD={ma['max_dd']:.1%}")
            lines.append(f"- Config B: CAGR={mb['cagr']:+.1%}, Sharpe={mb['sharpe']:.2f}, DD={mb['max_dd']:.1%}")
            lines.append(f"- Delta CAGR: {mb['cagr']-ma['cagr']:+.1%}, Delta Sharpe: {mb['sharpe']-ma['sharpe']:+.02f}")

    # Bull market regression check
    lines.append("\n## Bull Year Regression Check")
    lines.append("Did regime sizing hurt any strong bull years?")
    lines.append("")
    bull_years = [2016, 2017, 2019, 2020, 2021, 2023, 2024, 2025]
    regressions = 0
    for y in bull_years:
        ma = next((m for m in results_a if m["year"] == y), None)
        mb = next((m for m in results_b if m["year"] == y), None)
        if ma and mb:
            delta = mb["cagr"] - ma["cagr"]
            flag = " **REGRESSION**" if delta < -0.02 else ""
            if delta < -0.02:
                regressions += 1
            lines.append(f"- {y}: A={ma['cagr']:+.1%} -> B={mb['cagr']:+.1%} (delta {delta:+.1%}){flag}")

    if regressions == 0:
        lines.append(f"\nNo meaningful regressions (>2pp) in bull years.")
    else:
        lines.append(f"\n{regressions} regressions (>2pp) in bull years.")

    # Regime accuracy check
    lines.append("\n## Regime Accuracy Check")
    lines.append(f"- BULLISH: {total_bull}d ({total_bull/total_all*100:.0f}%)")
    lines.append(f"- CAUTIOUS: {total_caut}d ({total_caut/total_all*100:.0f}%)")
    lines.append(f"- BEARISH: {total_bear}d ({total_bear/total_all*100:.0f}%)")
    if total_bear < total_all * 0.05:
        lines.append(f"\nBEARISH triggered only {total_bear/total_all*100:.1f}% of the time — regime sizing has limited impact.")
    elif total_bear > total_all * 0.15:
        lines.append(f"\nBEARISH triggered {total_bear/total_all*100:.1f}% of the time — significant exposure reduction in downturns.")
    else:
        lines.append(f"\nBEARISH triggered {total_bear/total_all*100:.1f}% — moderate impact.")

    # Sharpe wins
    sharpe_wins = sum(1 for sa, sb in zip(sharpes_a, sharpes_b) if sb > sa)
    cagr_wins = sum(1 for ca, cb in zip(cagrs_a, cagrs_b) if cb > ca)

    # Verdict
    lines.append("\n## Verdict")
    lines.append("")

    med_sharpe_delta = np.median(sharpes_b) - np.median(sharpes_a)
    med_cagr_delta = np.median(cagrs_b) - np.median(cagrs_a)
    worst_dd_improved = min(dds_b) > min(dds_a)

    lines.append(f"- Median Sharpe improvement: {med_sharpe_delta:+.02f} (threshold: >0.30)")
    lines.append(f"- Median CAGR change: {med_cagr_delta:+.1%}")
    lines.append(f"- Worst DD {'improved' if worst_dd_improved else 'worsened'}: {min(dds_a):.1%} -> {min(dds_b):.1%}")
    lines.append(f"- Sharpe improved: {sharpe_wins}/{n} years")
    lines.append(f"- CAGR improved: {cagr_wins}/{n} years")
    lines.append(f"- Regressions in bull years: {regressions}")

    if med_sharpe_delta > 0.30 and sharpe_wins >= 7 and regressions <= 1:
        lines.append(f"\n**RECOMMEND DEPLOY**: Clear, consistent improvement across {sharpe_wins}/{n} years.")
    elif med_sharpe_delta > 0.10 and sharpe_wins >= 6:
        lines.append(f"\n**MARGINAL — USER DECISION**: Moderate improvement ({sharpe_wins}/{n} years better).")
        lines.append(f"Not a slam dunk. User should weigh the {med_sharpe_delta:+.02f} Sharpe gain vs added complexity.")
    elif med_sharpe_delta > 0 and sharpe_wins >= 5:
        lines.append(f"\n**MARGINAL**: Small improvement, not consistent enough to recommend outright.")
    else:
        lines.append(f"\n**DO NOT DEPLOY**: Config B does not clearly improve over Config A.")

    return "\n".join(lines)


def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  CLEAN A/B WALK-FORWARD COMPARISON")
    log("  Config A (current live) vs Config B (regime 8/5/2)")
    log("=" * 70)

    # Load predictions
    all_preds = {}
    for year in YEARS:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            log(f"ERROR: {pred_file} not found. Run walk_forward_validation.py first.")
            sys.exit(1)
        df = pd.read_parquet(pred_file)
        df["date"] = pd.to_datetime(df["date"])
        all_preds[year] = df
        log(f"  Loaded predictions_{year}.parquet ({len(df):,} rows)")

    # Get symbols
    all_syms = set()
    for df in all_preds.values():
        all_syms.update(df["symbol"].unique())
    all_syms = sorted(all_syms)

    # Load price bars
    log(f"\nLoading price bars for {len(all_syms)} symbols ...")
    close = load_bars_cached(all_syms, "2013-06-01", "2025-12-31")
    log(f"  Price data: {close.shape}")

    spy_full = close["SPY"].dropna()

    # Compute regime distribution (live production definition)
    regime_series = compute_regime_live(spy_full)
    regime_stats = {}
    for year in YEARS:
        ys = pd.Timestamp(f"{year}-01-01")
        ye = pd.Timestamp(f"{year}-12-31")
        yr = regime_series[(regime_series.index >= ys) & (regime_series.index <= ye)]
        vc = yr.value_counts()
        regime_stats[year] = {
            "BULLISH": int(vc.get("BULLISH", 0)),
            "CAUTIOUS": int(vc.get("CAUTIOUS", 0)),
            "BEARISH": int(vc.get("BEARISH", 0)),
        }

    log(f"\nRegime distribution (live production definition):")
    for r in ["BULLISH", "CAUTIOUS", "BEARISH"]:
        total = sum(regime_stats[y][r] for y in YEARS)
        log(f"  {r}: {total}d")

    # Run backtests
    results_a = []
    results_b = []

    for year in YEARS:
        log(f"\n{'─'*70}")
        s = regime_stats[year]
        log(f"  YEAR {year}  BULL={s['BULLISH']}d  CAUT={s['CAUTIOUS']}d  BEAR={s['BEARISH']}d")
        log(f"{'─'*70}")

        preds_df = all_preds[year]
        all_dates = sorted(preds_df["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        # Config A
        t1 = time.perf_counter()
        ma = run_config_a(preds_df.copy(), year, close_year, spy_full)
        if ma:
            results_a.append(ma)
            log(f"  A (current):  CAGR={ma['cagr']:+.1%}  Sharpe={ma['sharpe']:.2f}  "
                f"DD={ma['max_dd']:.1%}  Trades={ma['n_trades']}  ({time.perf_counter()-t1:.1f}s)")

        # Config B
        t2 = time.perf_counter()
        mb = run_config_b(preds_df.copy(), year, close_year, spy_full)
        if mb:
            results_b.append(mb)
            delta = mb["cagr"] - ma["cagr"] if ma else 0
            log(f"  B (regime):   CAGR={mb['cagr']:+.1%}  Sharpe={mb['sharpe']:.2f}  "
                f"DD={mb['max_dd']:.1%}  Trades={mb['n_trades']}  "
                f"delta={delta:+.1%}  ({time.perf_counter()-t2:.1f}s)")

    # Generate report
    log(f"\n{'='*70}")
    log("  GENERATING REPORT")
    log(f"{'='*70}")

    report = generate_report(results_a, results_b, regime_stats)
    report_file = WF_DIR / "AB_COMPARISON.md"
    report_file.write_text(report)
    log(f"\nReport: {report_file}")

    # Save JSON
    json_file = WF_DIR / "config_A_results.json"
    json_file.write_text(json.dumps([{k: v for k, v in m.items()} for m in results_a], indent=2, default=str))
    json_file = WF_DIR / "config_B_results.json"
    json_file.write_text(json.dumps([{k: v for k, v in m.items()} for m in results_b], indent=2, default=str))

    total = time.perf_counter() - t0
    log(f"\nTotal: {total:.0f}s ({total/60:.1f}min)")
    log("Done.")


if __name__ == "__main__":
    main()
