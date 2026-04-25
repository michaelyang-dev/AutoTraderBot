#!/usr/bin/env python3
"""
Three-Way Walk-Forward Comparison
==================================
Config A: Current live production (top_n=5, SPY parking ON, CAUTIOUS top-2 filter)
Config B: Backtester default aka "mistaken" (top_n=8, no parking, no CAUTIOUS filter)
Config C: Config A + regime-aware sizing (8/5/2)

Uses existing walk-forward predictions (trained OOS per year).
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
    MomentumStrategy, MeanReversionStrategy,
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


# CautiousMLStrategy and compute_regime_live are imported from unified_backtester

# ── Regime Sizing PM (for Config C) ────────────────────────────────────────

REGIME_LIMITS = {"BULLISH": 8, "CAUTIOUS": 5, "BEARISH": 2}


class RegimeSizingPM(PortfolioManager):
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

            regime = self.regime_dict.get(date, "CAUTIOUS")
            effective_max = REGIME_LIMITS.get(regime, self._base_max)

            for sym, pos in positions.items():
                if pos.price_based and sym in day_prices:
                    px = day_prices[sym]
                    if not np.isnan(px) and px > pos.peak_price:
                        pos.peak_price = px

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

            seen_syms = set()
            resolved = []
            for sig in all_signals:
                if sig.symbol not in seen_syms:
                    seen_syms.add(sig.symbol)
                    resolved.append(sig)
            resolved.sort(key=lambda s: s.confidence, reverse=True)

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

            # Park idle cash in SPY
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


# ── Config runners ──────────────────────────────────────────────────────────

def _run_backtest(preds_df, year, close, spy_full, strategies, pm_class,
                  pm_kwargs, spy_parking, label):
    all_dates = sorted(preds_df["date"].unique().tolist())
    if len(all_dates) < 10:
        return None

    years_span = (all_dates[-1] - all_dates[0]).days / 365.25
    if years_span <= 0:
        years_span = len(all_dates) / 252.0

    spy_px = close["SPY"].dropna()
    if spy_px.empty:
        return None

    spy_dict = close["SPY"].to_dict() if spy_parking else None
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    if "prob_ensemble" in preds_df.columns and "prob" not in preds_df.columns:
        preds_df = preds_df.rename(columns={"prob_ensemble": "prob"})

    pm = pm_class(strategies=strategies, slot_config=SLOT_LIVE_V2, **pm_kwargs)
    vals, trades = pm.run(all_dates, spy_prices=spy_dict, price_data=close)

    metrics = calc_metrics(vals, trades, years_span, f"{label}_{year}")
    alpha, beta = calc_alpha_beta(vals, spy_bh.reindex(vals.index, method="ffill"))
    metrics["alpha"] = float(alpha) if not np.isnan(alpha) else None
    metrics["beta"] = float(beta) if not np.isnan(beta) else None
    metrics["year"] = year
    return metrics


def _fix_prob_col(df):
    if "prob_ensemble" in df.columns and "prob" not in df.columns:
        df = df.rename(columns={"prob_ensemble": "prob"})
    return df


def run_live_production(preds_df, year, close, spy_full, regime_dict):
    """Config A: current live (top_n=5, parking ON, CAUTIOUS top-2 filter)."""
    preds_df = _fix_prob_col(preds_df)
    strategies = [
        CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                           threshold=0.55, top_n=5, selection_mode="top_n"),
        MomentumStrategy(close, volume_data=None, regime_filter=True),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]
    return _run_backtest(preds_df, year, close, spy_full, strategies,
                         PortfolioManager, {}, spy_parking=True, label="live")


def run_backtester_default(preds_df, year, close, spy_full):
    """Config B: backtester default (top_n=8, no parking, no CAUTIOUS filter)."""
    preds_df = _fix_prob_col(preds_df)
    strategies = [
        MLMediumStrategy(preds_df, threshold=0.55, top_n=8, selection_mode="top_n"),
        MomentumStrategy(close, volume_data=None, regime_filter=True),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]
    return _run_backtest(preds_df, year, close, spy_full, strategies,
                         PortfolioManager, {}, spy_parking=False, label="bt_default")


def run_regime_sizing(preds_df, year, close, spy_full, regime_dict):
    """Config C: live production + regime sizing (8/5/2)."""
    preds_df = _fix_prob_col(preds_df)
    strategies = [
        CautiousMLStrategy(preds_df, regime_dict=regime_dict,
                           threshold=0.55, top_n=5, selection_mode="top_n"),
        MomentumStrategy(close, volume_data=None, regime_filter=True),
        MeanReversionStrategy(close, volume_data=None),
        MegaCapStrategy(close),
    ]
    return _run_backtest(preds_df, year, close, spy_full, strategies,
                         RegimeSizingPM, {"regime_dict": regime_dict},
                         spy_parking=True, label="regime")


# ── Report ──────────────────────────────────────────────────────────────────

def generate_report(results, regime_stats):
    labels = ["A: Live Production", "B: Backtester Default", "C: Live + Regime(8/5/2)"]
    short = ["A (Live)", "B (BT Default)", "C (Regime)"]

    lines = []
    lines.append("# Three-Way Walk-Forward Comparison")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

    lines.append("\n## Configuration Details")
    lines.append("")
    lines.append("| Setting | A: Live Production | B: Backtester Default | C: Live + Regime |")
    lines.append("|---------|-------------------|----------------------|-----------------|")
    lines.append("| Slot config | ml:5, mom:3, max:8 | ml:5, mom:3, max:8 | ml:5, mom:3, max:8 |")
    lines.append("| ML top_n | 5 | 8 | 5 |")
    lines.append("| CAUTIOUS ML filter | ON (top 2 only) | OFF | ON (top 2 only) |")
    lines.append("| SPY parking | ON | OFF | ON |")
    lines.append("| Momentum regime filter | ON | ON | ON |")
    lines.append("| Regime-aware sizing | OFF | OFF | ON (8/5/2) |")

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
    for cfg_idx, label in enumerate(short):
        lines.append(f"\n### {labels[cfg_idx]}")
        lines.append("")
        lines.append("| Year | CAGR | Sharpe | Max DD | Alpha | Beta | Trades | Win% |")
        lines.append("|------|------|--------|--------|-------|------|--------|------|")
        for m in results[cfg_idx]:
            alpha_str = f"{m['alpha']:+.1%}" if m.get('alpha') is not None else "N/A"
            beta_str = f"{m['beta']:.2f}" if m.get('beta') is not None else "N/A"
            lines.append(
                f"| {m['year']} | {m['cagr']:+.1%} | {m['sharpe']:.2f} | {m['max_dd']:.1%} "
                f"| {alpha_str} | {beta_str} | {m['n_trades']} | {m['win_rate']:.1%} |"
            )

    # Head-to-head CAGR table
    lines.append("\n## Head-to-Head: CAGR by Year")
    lines.append("")
    lines.append("| Year | A (Live) | B (BT Default) | C (Regime) | Best |")
    lines.append("|------|----------|---------------|------------|------|")
    for i_yr in range(len(results[0])):
        yr = results[0][i_yr]["year"]
        cagrs = [results[c][i_yr]["cagr"] for c in range(3)]
        best_idx = np.argmax(cagrs)
        best_label = ["A", "B", "C"][best_idx]
        lines.append(
            f"| {yr} | {cagrs[0]:+.1%} | {cagrs[1]:+.1%} | {cagrs[2]:+.1%} | **{best_label}** |"
        )

    # Head-to-head Sharpe table
    lines.append("\n## Head-to-Head: Sharpe by Year")
    lines.append("")
    lines.append("| Year | A (Live) | B (BT Default) | C (Regime) | Best |")
    lines.append("|------|----------|---------------|------------|------|")
    for i_yr in range(len(results[0])):
        yr = results[0][i_yr]["year"]
        sharpes = [results[c][i_yr]["sharpe"] for c in range(3)]
        best_idx = np.argmax(sharpes)
        best_label = ["A", "B", "C"][best_idx]
        lines.append(
            f"| {yr} | {sharpes[0]:.2f} | {sharpes[1]:.2f} | {sharpes[2]:.2f} | **{best_label}** |"
        )

    # Summary metrics
    lines.append("\n## Summary Metrics")
    lines.append("")
    lines.append("| Metric | A (Live) | B (BT Default) | C (Regime) |")
    lines.append("|--------|----------|---------------|------------|")

    for cfg_idx in range(3):
        r = results[cfg_idx]
        if cfg_idx == 0:
            cagrs = [m["cagr"] for m in r]
            sharpes = [m["sharpe"] for m in r]
            dds = [m["max_dd"] for m in r]
            alphas = [m["alpha"] for m in r if m.get("alpha") is not None]
            betas = [m["beta"] for m in r if m.get("beta") is not None]

    all_cagrs = [[m["cagr"] for m in results[c]] for c in range(3)]
    all_sharpes = [[m["sharpe"] for m in results[c]] for c in range(3)]
    all_dds = [[m["max_dd"] for m in results[c]] for c in range(3)]
    all_alphas = [[m["alpha"] for m in results[c] if m.get("alpha") is not None] for c in range(3)]
    all_betas = [[m["beta"] for m in results[c] if m.get("beta") is not None] for c in range(3)]
    all_trades = [[m["n_trades"] for m in results[c]] for c in range(3)]
    all_wr = [[m["win_rate"] for m in results[c]] for c in range(3)]

    def row(label, vals, fmt):
        return f"| {label} | {fmt.format(vals[0])} | {fmt.format(vals[1])} | {fmt.format(vals[2])} |"

    lines.append(row("Median CAGR", [np.median(c) for c in all_cagrs], "{:+.1%}"))
    lines.append(row("Mean CAGR", [np.mean(c) for c in all_cagrs], "{:+.1%}"))
    lines.append(row("Median Sharpe", [np.median(c) for c in all_sharpes], "{:.2f}"))
    lines.append(row("Mean Sharpe", [np.mean(c) for c in all_sharpes], "{:.2f}"))
    lines.append(row("Worst Max DD", [min(c) for c in all_dds], "{:.1%}"))
    lines.append(row("Mean Max DD", [np.mean(c) for c in all_dds], "{:.1%}"))
    lines.append(row("Median Alpha", [np.median(c) for c in all_alphas], "{:+.1%}"))
    lines.append(row("Mean Beta", [np.mean(c) for c in all_betas], "{:.2f}"))

    pos_cagr = [sum(1 for c in cs if c > 0) for cs in all_cagrs]
    pos_alpha = [sum(1 for a in al if a > 0) for al in all_alphas]
    n = len(results[0])
    lines.append(f"| Positive CAGR years | {pos_cagr[0]}/{n} | {pos_cagr[1]}/{n} | {pos_cagr[2]}/{n} |")
    lines.append(f"| Alpha-positive years | {pos_alpha[0]}/{n} | {pos_alpha[1]}/{n} | {pos_alpha[2]}/{n} |")
    lines.append(row("Mean trades/year", [np.mean(c) for c in all_trades], "{:.0f}"))
    lines.append(row("Mean win rate", [np.mean(c) for c in all_wr], "{:.1%}"))

    # Win counts
    lines.append("\n## Win Counts (Sharpe)")
    lines.append("")
    wins = [0, 0, 0]
    for i_yr in range(n):
        sharpes = [results[c][i_yr]["sharpe"] for c in range(3)]
        wins[np.argmax(sharpes)] += 1
    lines.append(f"- A (Live Production): **{wins[0]}/{n}** years")
    lines.append(f"- B (Backtester Default): **{wins[1]}/{n}** years")
    lines.append(f"- C (Live + Regime): **{wins[2]}/{n}** years")

    # Bear market focus
    lines.append("\n## Bear Market Focus (2022)")
    for target_year in [2022]:
        ms = [next((m for m in results[c] if m["year"] == target_year), None) for c in range(3)]
        if all(ms):
            s = regime_stats[target_year]
            lines.append(f"\n### {target_year} (BEAR={s['BEARISH']}d, CAUT={s['CAUTIOUS']}d)")
            for i, lbl in enumerate(short):
                lines.append(f"- {lbl}: CAGR={ms[i]['cagr']:+.1%}, Sharpe={ms[i]['sharpe']:.2f}, DD={ms[i]['max_dd']:.1%}")

    # Verdict
    lines.append("\n## Verdict")
    lines.append("")
    med_sharpes = [np.median(s) for s in all_sharpes]
    med_cagrs = [np.median(c) for c in all_cagrs]
    best_sharpe_idx = np.argmax(med_sharpes)
    best_label = labels[best_sharpe_idx]
    lines.append(f"**Best config by median Sharpe: {best_label}**")
    lines.append(f"- Median Sharpe: {med_sharpes[best_sharpe_idx]:.2f}")
    lines.append(f"- Median CAGR: {med_cagrs[best_sharpe_idx]:+.1%}")
    lines.append(f"- Wins {wins[best_sharpe_idx]}/{n} years by Sharpe")

    if best_sharpe_idx == 1:
        lines.append(f"\nThe backtester default (no parking, top_n=8, no CAUTIOUS filter) "
                     f"outperforms live production by {med_sharpes[1]-med_sharpes[0]:+.2f} median Sharpe.")
        lines.append("Consider testing individual changes (parking, top_n, CAUTIOUS filter) to isolate impact.")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    log("=" * 70)
    log("  THREE-WAY WALK-FORWARD COMPARISON")
    log("  A: Live Production | B: Backtester Default | C: Live + Regime")
    log("=" * 70)

    # Load predictions
    all_preds = {}
    for year in YEARS:
        pred_file = WF_DIR / f"predictions_{year}.parquet"
        if not pred_file.exists():
            log(f"ERROR: {pred_file} not found.")
            sys.exit(1)
        df = pd.read_parquet(pred_file)
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

    regime_series = compute_regime_live(spy_full)
    regime_dict = regime_series.to_dict()
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

    results = [[], [], []]  # A, B, C

    for year in YEARS:
        log(f"\n{'─'*70}")
        s = regime_stats[year]
        log(f"  YEAR {year}  BULL={s['BULLISH']}d  CAUT={s['CAUTIOUS']}d  BEAR={s['BEARISH']}d")
        log(f"{'─'*70}")

        preds_df = all_preds[year]
        all_dates = sorted(preds_df["date"].unique().tolist())
        sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
        close_year = close.reindex(close.index.union(sim_index), method="ffill")

        # Config A: Live Production
        t1 = time.perf_counter()
        ma = run_live_production(preds_df.copy(), year, close_year, spy_full, regime_dict)
        if ma:
            results[0].append(ma)
            log(f"  A (live):       CAGR={ma['cagr']:+.1%}  Sharpe={ma['sharpe']:.2f}  "
                f"DD={ma['max_dd']:.1%}  Trades={ma['n_trades']}  ({time.perf_counter()-t1:.1f}s)")

        # Config B: Backtester Default
        t2 = time.perf_counter()
        mb = run_backtester_default(preds_df.copy(), year, close_year, spy_full)
        if mb:
            results[1].append(mb)
            log(f"  B (bt default): CAGR={mb['cagr']:+.1%}  Sharpe={mb['sharpe']:.2f}  "
                f"DD={mb['max_dd']:.1%}  Trades={mb['n_trades']}  ({time.perf_counter()-t2:.1f}s)")

        # Config C: Live + Regime Sizing
        t3 = time.perf_counter()
        mc = run_regime_sizing(preds_df.copy(), year, close_year, spy_full, regime_dict)
        if mc:
            results[2].append(mc)
            log(f"  C (regime):     CAGR={mc['cagr']:+.1%}  Sharpe={mc['sharpe']:.2f}  "
                f"DD={mc['max_dd']:.1%}  Trades={mc['n_trades']}  ({time.perf_counter()-t3:.1f}s)")

    log(f"\n{'='*70}")
    log("  GENERATING REPORT")
    log(f"{'='*70}")

    report = generate_report(results, regime_stats)
    report_file = WF_DIR / "THREE_WAY_COMPARISON.md"
    report_file.write_text(report)
    log(f"\nReport: {report_file}")

    # Save JSON
    for i, label in enumerate(["live_production", "backtester_default", "regime_sizing"]):
        json_file = WF_DIR / f"config_{label}_results.json"
        json_file.write_text(json.dumps(results[i], indent=2, default=str))

    total = time.perf_counter() - t0
    log(f"\nTotal: {total:.0f}s ({total/60:.1f}min)")
    log("Done.")


if __name__ == "__main__":
    main()
