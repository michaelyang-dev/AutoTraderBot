"""
Diagnostic analysis: why ML + Momentum combined underperforms both strategies alone.

Instruments the PortfolioManager simulation loop to collect:
  1. Slot utilization distribution
  2. Signal conflict/blocking reasons
  3. Overlap performance (agreed vs solo trades)
  4. Strategy-level P&L attribution
  5. First-to-fire bias
  6. Position count per day
  7. Capital efficiency (cash % of portfolio)

Run with:
    python3 diagnose_combined.py
"""

import warnings
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

from unified_backtester import (
    INITIAL_CASH, HOLD_DAYS, SLIPPAGE, POSITION_PCT,
    COOLDOWN_DAYS,
    SPY_RESERVE_PCT, SPY_THRESHOLD_PCT, SPY_INVEST_PCT,
    Signal, Position, SlotConfig,
    MLMediumStrategy, MomentumStrategy,
    PortfolioManager,
    SLOT_ML_ONLY, SLOT_MOM_ONLY, SLOT_ML_MOM,
)
from backtest_ml import load_predictions, calc_metrics, calc_alpha_beta

warnings.filterwarnings("ignore")
DATA_DIR = Path(__file__).resolve().parent / "data"
ML_THRESHOLD = 0.55


def fetch_ohlcv(symbols, start, end):
    all_syms = list(set(["SPY"] + symbols))
    raw = yf.download(all_syms, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    volume = None
    if isinstance(raw.columns, pd.MultiIndex) and "Volume" in raw.columns.get_level_values(0):
        volume = raw["Volume"]
        volume.index = pd.to_datetime(volume.index).tz_localize(None)
    return close, volume


def instrumented_run(strategies, slot_config, all_dates, spy_prices,
                     price_data, hold_days, label):
    """
    Replica of PortfolioManager.run() with full instrumentation.
    Returns (portfolio_series, trades_list, diagnostics_dict).
    """
    strats = {s.name: s for s in strategies}
    n_dates = len(all_dates)
    multi = len(strats) > 1

    # Price lookup
    px_lookup = {}
    if price_data is not None:
        for date in all_dates:
            if date in price_data.index:
                px_lookup[date] = price_data.loc[date].to_dict()

    # ── Diagnostic accumulators ──────────────────────────────────────────
    daily_pos_count = []           # positions held at end of each day
    daily_pos_by_strat = []        # {strat → count} per day
    daily_cash_pct = []            # cash / portfolio_value per day

    block_reasons = Counter()      # reason → count
    # Per-strategy blocking detail
    block_by_strat = defaultdict(Counter)  # strat → {reason → count}

    # Track which strategy's signals get accepted
    signals_generated = Counter()  # strat → total signals generated
    signals_accepted = Counter()   # strat → signals that became positions

    # Trade-level tracking
    trade_records = []  # list of dicts with strategy, overlap_count, return, cost, etc.

    # Overlap tracking
    overlap_signal_days = 0

    # First-to-fire: on overlap days, which strategy's signal is ranked first?
    first_fire_counts = Counter()  # strat → count of times it won the top ranking

    # Cooldown tracking: who caused the cooldown that blocked whom?
    cooldown_caused_by = Counter()  # (caused_by_strat, blocked_strat) → count

    # ── Run state ────────────────────────────────────────────────────────
    cash = float(INITIAL_CASH)
    positions = {}
    idle_spy_shares = 0.0
    trades = []
    cooldowns = {}          # sym → expiry_idx
    cooldown_owner = {}     # sym → strategy_name that sold it (caused the cooldown)
    port_vals = []

    for i, date in enumerate(all_dates):
        spy_px = spy_prices.get(date) if spy_prices else None
        if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
            spy_px = None

        day_prices = px_lookup.get(date, {})
        cur = {"idx": i, "date": date, "n_dates": n_dates, "prices": day_prices}

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
            trade_ret = (net - pos.cost) / pos.cost
            trades.append(trade_ret)

            trade_records.append({
                "strategy": pos.strategy_name,
                "symbol": pos.symbol,
                "overlap": 1,
                "return": trade_ret,
                "cost": pos.cost,
                "pnl": net - pos.cost,
                "exit_reason": exit_reason,
                "days_held": i - pos.entry_idx,
            })

            if multi:
                cooldowns[(sym, pos.strategy_name)] = i + COOLDOWN_DAYS
                cooldown_owner[sym] = pos.strategy_name

        # ── 3. Gather signals ────────────────────────────────────────────
        raw_signals_by_strat = {}
        for strat in strats.values():
            sigs = strat.generate_signals(date, None)
            raw_signals_by_strat[strat.name] = sigs
            signals_generated[strat.name] += len(sigs)

        all_signals_raw = []
        for sigs in raw_signals_by_strat.values():
            all_signals_raw.extend(sigs)

        # Filter: held symbols
        held = set(positions.keys())
        pre_filter = all_signals_raw
        all_signals = []
        for s in pre_filter:
            if s.symbol in held:
                block_reasons["already_held"] += 1
                block_by_strat[s.strategy_name]["already_held"] += 1
                continue
            if not s.price_based and np.isnan(s.fwd_ret):
                block_reasons["nan_fwd_ret"] += 1
                block_by_strat[s.strategy_name]["nan_fwd_ret"] += 1
                continue
            all_signals.append(s)

        # Filter: cooldowns (strategy-specific)
        if multi:
            filtered = []
            for s in all_signals:
                if cooldowns.get((s.symbol, s.strategy_name), -1) > i:
                    block_reasons["cooldown"] += 1
                    block_by_strat[s.strategy_name]["cooldown"] += 1
                    caused_by = cooldown_owner.get(s.symbol, "unknown")
                    cooldown_caused_by[(caused_by, s.strategy_name)] += 1
                    continue
                filtered.append(s)
            all_signals = filtered

        # ── 4. Resolve overlaps (first-strategy-wins) ────────────────────
        seen_syms = set()
        resolved = []
        day_has_overlap = False
        for sig in all_signals:
            if sig.symbol not in seen_syms:
                seen_syms.add(sig.symbol)
                resolved.append(sig)
            else:
                # Track the overlap (signal discarded)
                day_has_overlap = True
                first_fire_counts[sig.strategy_name] += 1  # track who lost
        if day_has_overlap:
            overlap_signal_days += 1

        resolved.sort(key=lambda s: s.confidence, reverse=True)

        # ── 5. Execute buys ──────────────────────────────────────────────
        max_slots = slot_config.max_positions - len(positions)

        if spy_px and idle_spy_shares > 0 and resolved and max_slots > 0:
            proceeds = idle_spy_shares * spy_px * (1.0 - SLIPPAGE)
            cash += proceeds
            idle_spy_shares = 0.0

        bought_this_day = 0
        for sig_idx, sig in enumerate(resolved):
            if sig_idx >= max_slots:
                block_reasons["slot_full_max"] += 1
                block_by_strat[sig.strategy_name]["slot_full_max"] += 1
                continue

            # Per-strategy slot check
            counts = {}
            for p in positions.values():
                counts[p.strategy_name] = counts.get(p.strategy_name, 0) + 1
            avail = slot_config.available_for(sig.strategy_name, counts, len(positions))
            if avail <= 0:
                block_reasons["slot_full_strategy"] += 1
                block_by_strat[sig.strategy_name]["slot_full_strategy"] += 1
                continue

            # Price-based entry validation
            entry_px = 0.0
            if sig.price_based:
                entry_px = day_prices.get(sig.symbol, np.nan)
                if np.isnan(entry_px) or entry_px <= 0:
                    block_reasons["no_price"] += 1
                    block_by_strat[sig.strategy_name]["no_price"] += 1
                    continue

            # Position sizing
            strat = strats[sig.strategy_name]
            port_est = cash
            for p in positions.values():
                if p.price_based:
                    px = day_prices.get(p.symbol, p.entry_price)
                    port_est += p.cost * (px / p.entry_price if p.entry_price > 0 else 1.0)
                else:
                    port_est += p.cost
            target = strat.get_position_size(sig, port_est)

            cost = min(target, cash * 0.95)
            if cost < 50.0:
                block_reasons["insufficient_cash"] += 1
                block_by_strat[sig.strategy_name]["insufficient_cash"] += 1
                continue

            cash -= cost * (1.0 + SLIPPAGE)
            exit_idx = min(i + hold_days, n_dates - 1)

            pos = Position(
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
            positions[sig.symbol] = pos
            signals_accepted[sig.strategy_name] += 1
            bought_this_day += 1

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

        # ── Diagnostics: daily snapshots ─────────────────────────────────
        daily_pos_count.append(len(positions))
        strat_counts = Counter()
        for p in positions.values():
            strat_counts[p.strategy_name] += 1
        daily_pos_by_strat.append(dict(strat_counts))
        daily_cash_pct.append(cash / port_val if port_val > 0 else 1.0)

    series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates))

    diag = {
        "daily_pos_count": daily_pos_count,
        "daily_pos_by_strat": daily_pos_by_strat,
        "daily_cash_pct": daily_cash_pct,
        "block_reasons": block_reasons,
        "block_by_strat": dict(block_by_strat),
        "signals_generated": dict(signals_generated),
        "signals_accepted": dict(signals_accepted),
        "trade_records": trade_records,
        "overlap_signal_days": overlap_signal_days,
        "first_fire_counts": dict(first_fire_counts),
        "cooldown_caused_by": dict(cooldown_caused_by),
    }
    return series, [float(t) for t in trades], diag


def print_slot_utilization(diags, configs, max_slots_map):
    """Print slot utilization distribution for each configuration."""
    print(f"\n{'='*80}")
    print("  1. SLOT UTILIZATION")
    print(f"{'='*80}")

    for name in configs:
        d = diags[name]
        counts = d["daily_pos_count"]
        max_s = max_slots_map[name]
        n = len(counts)

        full = sum(1 for c in counts if c >= max_s)
        partial = sum(1 for c in counts if 0 < c < max_s)
        empty = sum(1 for c in counts if c == 0)
        avg = np.mean(counts)

        print(f"\n  [{name}] (max {max_s} slots)")
        print(f"    Full ({max_s}/{max_s}):  {full:>5} days ({full/n:.1%})")
        print(f"    Partial:       {partial:>5} days ({partial/n:.1%})")
        print(f"    Empty (0):     {empty:>5} days ({empty/n:.1%})")
        print(f"    Avg positions: {avg:.2f}")

        # Distribution histogram
        dist = Counter(counts)
        print(f"    Distribution:")
        for k in range(max_s + 1):
            cnt = dist.get(k, 0)
            bar = "█" * int(cnt / n * 60)
            print(f"      {k} pos: {cnt:>5} ({cnt/n:>5.1%}) {bar}")


def print_signal_conflicts(diag, name):
    """Print signal blocking reasons."""
    print(f"\n{'='*80}")
    print("  2. SIGNAL CONFLICTS")
    print(f"{'='*80}")

    br = diag["block_reasons"]
    total_blocked = sum(br.values())
    gen = diag["signals_generated"]
    acc = diag["signals_accepted"]
    total_gen = sum(gen.values())
    total_acc = sum(acc.values())

    print(f"\n  [{name}]")
    print(f"    Total signals generated:  {total_gen:>6}")
    print(f"    Total signals accepted:   {total_acc:>6} ({total_acc/total_gen:.1%})")
    print(f"    Total signals blocked:    {total_blocked:>6} ({total_blocked/total_gen:.1%})")

    print(f"\n    Block reasons (all strategies):")
    for reason, count in sorted(br.items(), key=lambda x: -x[1]):
        print(f"      {reason:<25} {count:>6} ({count/total_gen:.1%} of generated)")

    print(f"\n    Per-strategy breakdown:")
    print(f"    {'Strategy':<18} {'Generated':>10} {'Accepted':>10} {'Accept%':>8}")
    print(f"    {'─'*17} {'─'*9} {'─'*9} {'─'*7}")
    for strat in sorted(gen.keys()):
        g = gen[strat]
        a = acc.get(strat, 0)
        print(f"    {strat:<18} {g:>10} {a:>10} {a/g:>7.1%}")

    print(f"\n    Per-strategy block detail:")
    for strat in sorted(diag["block_by_strat"].keys()):
        reasons = diag["block_by_strat"][strat]
        total = sum(reasons.values())
        print(f"    [{strat}] — {total} blocked:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"      {reason:<25} {count:>6}")


def print_overlap_performance(diag):
    """Overlap vs solo trade performance."""
    print(f"\n{'='*80}")
    print("  3. OVERLAP PERFORMANCE")
    print(f"{'='*80}")

    recs = diag["trade_records"]
    if not recs:
        print("  No trades recorded.")
        return

    overlap_trades = [r for r in recs if r["overlap"] > 1]
    solo_trades = [r for r in recs if r["overlap"] == 1]

    print(f"\n  Overlap signal days: {diag['overlap_signal_days']}")
    print(f"  Overlap trades (1.25x size): {len(overlap_trades)}")
    print(f"  Solo trades:                 {len(solo_trades)}")

    if overlap_trades:
        o_rets = [r["return"] for r in overlap_trades]
        o_wins = sum(1 for r in o_rets if r > 0)
        o_pnl = sum(r["pnl"] for r in overlap_trades)
        print(f"\n  OVERLAP trades:")
        print(f"    Win rate:    {o_wins/len(o_rets):.1%}")
        print(f"    Avg return:  {np.mean(o_rets):+.2%}")
        print(f"    Total P&L:   ${o_pnl:,.0f}")

    if solo_trades:
        s_rets = [r["return"] for r in solo_trades]
        s_wins = sum(1 for r in s_rets if r > 0)
        s_pnl = sum(r["pnl"] for r in solo_trades)
        print(f"\n  SOLO trades:")
        print(f"    Win rate:    {s_wins/len(s_rets):.1%}")
        print(f"    Avg return:  {np.mean(s_rets):+.2%}")
        print(f"    Total P&L:   ${s_pnl:,.0f}")


def print_strategy_pnl(diag, name):
    """Strategy-level P&L attribution."""
    print(f"\n{'='*80}")
    print("  4. STRATEGY-LEVEL P&L ATTRIBUTION")
    print(f"{'='*80}")

    recs = diag["trade_records"]
    if not recs:
        print("  No trades.")
        return

    by_strat = defaultdict(list)
    for r in recs:
        by_strat[r["strategy"]].append(r)

    total_pnl = sum(r["pnl"] for r in recs)
    total_cost = sum(r["cost"] for r in recs)

    print(f"\n  [{name}] Total P&L: ${total_pnl:,.0f}")
    print(f"\n  {'Strategy':<18} {'Trades':>7} {'Win%':>7} {'Avg Ret':>9} "
          f"{'Total P&L':>12} {'% of P&L':>10} {'Avg Days':>9}")
    print(f"  {'─'*17} {'─'*6} {'─'*6} {'─'*8} {'─'*11} {'─'*9} {'─'*8}")

    for strat in sorted(by_strat.keys()):
        trades = by_strat[strat]
        rets = [t["return"] for t in trades]
        pnl = sum(t["pnl"] for t in trades)
        wins = sum(1 for r in rets if r > 0)
        avg_days = np.mean([t["days_held"] for t in trades])
        pnl_pct = pnl / total_pnl * 100 if total_pnl != 0 else 0

        print(f"  {strat:<18} {len(trades):>7} {wins/len(trades):>6.1%} "
              f"{np.mean(rets):>+8.2%} ${pnl:>11,.0f} {pnl_pct:>9.1f}% {avg_days:>8.1f}")

    # Exit reason breakdown
    print(f"\n  Exit reason breakdown:")
    exit_counts = Counter()
    exit_pnl = defaultdict(float)
    for r in recs:
        reason = r.get("exit_reason", "unknown")
        exit_counts[reason] += 1
        exit_pnl[reason] += r["pnl"]

    print(f"  {'Reason':<20} {'Count':>7} {'Avg P&L':>10} {'Total P&L':>12}")
    print(f"  {'─'*19} {'─'*6} {'─'*9} {'─'*11}")
    for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        avg_p = exit_pnl[reason] / count
        print(f"  {reason:<20} {count:>7} ${avg_p:>9,.0f} ${exit_pnl[reason]:>11,.0f}")


def print_first_to_fire(diag):
    """Which strategy wins when both signal the same stock."""
    print(f"\n{'='*80}")
    print("  5. FIRST-TO-FIRE BIAS")
    print(f"{'='*80}")

    ff = diag["first_fire_counts"]
    total = sum(ff.values())

    if total == 0:
        print("  No overlap events (no conflicts to analyze).")
        return

    print(f"\n  When both strategies signal the same stock, the MERGED signal")
    print(f"  takes the strategy name of the highest-confidence signal.")
    print(f"  Total overlap events: {total}")
    for strat, count in sorted(ff.items(), key=lambda x: -x[1]):
        print(f"    {strat:<18} wins {count:>5} ({count/total:.1%})")

    # Cooldown cross-blocking
    cc = diag["cooldown_caused_by"]
    if cc:
        print(f"\n  Cooldown cross-blocking (strategy A's sell blocks strategy B's buy):")
        for (caused_by, blocked), count in sorted(cc.items(), key=lambda x: -x[1]):
            print(f"    {caused_by} sell → blocks {blocked} buy: {count} times")


def print_position_count(diags, configs):
    """Average positions held per day."""
    print(f"\n{'='*80}")
    print("  6. POSITION COUNT COMPARISON")
    print(f"{'='*80}")

    for name in configs:
        d = diags[name]
        counts = d["daily_pos_count"]
        avg = np.mean(counts)
        invested_days = sum(1 for c in counts if c > 0)
        n = len(counts)

        print(f"\n  [{name}]")
        print(f"    Avg positions/day:  {avg:.2f}")
        print(f"    Days invested:      {invested_days:>5} / {n} ({invested_days/n:.1%})")

        # Per-strategy positions
        strat_avgs = defaultdict(list)
        for day_dict in d["daily_pos_by_strat"]:
            seen = set()
            for strat, cnt in day_dict.items():
                strat_avgs[strat].append(cnt)
                seen.add(strat)
            # strategies not holding anything this day
            for strat in diags[name].get("_all_strats", []):
                if strat not in seen:
                    strat_avgs[strat].append(0)

        if len(strat_avgs) > 1:
            print(f"    Per-strategy avg:")
            for strat in sorted(strat_avgs.keys()):
                vals = strat_avgs[strat]
                # Pad to full length
                while len(vals) < n:
                    vals.append(0)
                print(f"      {strat:<18} {np.mean(vals):.2f} positions/day")


def print_capital_efficiency(diags, configs):
    """Average cash balance as % of portfolio."""
    print(f"\n{'='*80}")
    print("  7. CAPITAL EFFICIENCY")
    print(f"{'='*80}")

    for name in configs:
        d = diags[name]
        cash_pcts = d["daily_cash_pct"]
        avg_cash = np.mean(cash_pcts)
        max_cash = np.max(cash_pcts)
        min_cash = np.min(cash_pcts)
        # Days with >50% cash
        high_cash_days = sum(1 for c in cash_pcts if c > 0.50)
        n = len(cash_pcts)

        print(f"\n  [{name}]")
        print(f"    Avg cash %:         {avg_cash:.1%}")
        print(f"    Min cash %:         {min_cash:.1%}")
        print(f"    Max cash %:         {max_cash:.1%}")
        print(f"    Days >50% cash:     {high_cash_days:>5} / {n} ({high_cash_days/n:.1%})")
        print(f"    Avg invested %:     {1-avg_cash:.1%}")


def main():
    t0 = time.perf_counter()
    print("=" * 80)
    print("  COMBINED BACKTEST DIAGNOSTIC ANALYSIS")
    print("  Why does ML + Momentum underperform both strategies alone?")
    print("=" * 80)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n  Loading data ...")
    df = load_predictions()
    all_dates = sorted(df["date"].unique().tolist())
    universe_syms = sorted(df["symbol"].unique().tolist())
    years = (all_dates[-1] - all_dates[0]).days / 365.25

    start = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=400)
    end = pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)
    close, volume = fetch_ohlcv(universe_syms,
                                 start.strftime("%Y-%m-%d"),
                                 end.strftime("%Y-%m-%d"))
    sim_index = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    close_aligned = close.reindex(sim_index, method="ffill")
    volume_aligned = volume.reindex(sim_index, method="ffill") if volume is not None else None
    spy_dict = close_aligned["SPY"].to_dict()

    spy_px = close_aligned["SPY"].dropna()
    spy_bh = spy_px / spy_px.iloc[0] * INITIAL_CASH

    # ── Build strategies ─────────────────────────────────────────────────
    ml_strat = MLMediumStrategy(df, threshold=ML_THRESHOLD)
    mom_strat = MomentumStrategy(close, volume_data=volume)

    # ── Run instrumented backtests ───────────────────────────────────────
    configs = ["Momentum alone", "ML alone", "ML + Momentum"]
    diags = {}
    metrics = {}

    print("\n  [1/3] Momentum alone (7 slots) ...")
    v, t, d = instrumented_run(
        [mom_strat], SLOT_MOM_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=60, label="Momentum alone")
    d["_all_strats"] = ["momentum"]
    diags["Momentum alone"] = d
    m = calc_metrics(v, t, years, "Momentum alone")
    a, _ = calc_alpha_beta(v, spy_bh.reindex(v.index, method="ffill"))
    m["alpha"] = a
    metrics["Momentum alone"] = m
    print(f"    CAGR {m['cagr']:+.2%} | Sharpe {m['sharpe']:.3f} | Trades {m['n_trades']}")

    print("\n  [2/3] ML alone (6 slots) ...")
    v2, t2, d2 = instrumented_run(
        [ml_strat], SLOT_ML_ONLY, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML alone")
    d2["_all_strats"] = ["ml_medium"]
    diags["ML alone"] = d2
    m2 = calc_metrics(v2, t2, years, "ML alone")
    a2, _ = calc_alpha_beta(v2, spy_bh.reindex(v2.index, method="ffill"))
    m2["alpha"] = a2
    metrics["ML alone"] = m2
    print(f"    CAGR {m2['cagr']:+.2%} | Sharpe {m2['sharpe']:.3f} | Trades {m2['n_trades']}")

    print("\n  [3/3] ML + Momentum combined (5 slots) ...")
    v3, t3, d3 = instrumented_run(
        [ml_strat, mom_strat], SLOT_ML_MOM, all_dates, spy_dict,
        close_aligned, hold_days=HOLD_DAYS, label="ML + Momentum")
    d3["_all_strats"] = ["ml_medium", "momentum"]
    diags["ML + Momentum"] = d3
    m3 = calc_metrics(v3, t3, years, "ML + Momentum")
    a3, _ = calc_alpha_beta(v3, spy_bh.reindex(v3.index, method="ffill"))
    m3["alpha"] = a3
    metrics["ML + Momentum"] = m3
    print(f"    CAGR {m3['cagr']:+.2%} | Sharpe {m3['sharpe']:.3f} | Trades {m3['n_trades']}")

    # ── Print all analyses ───────────────────────────────────────────────
    max_slots_map = {
        "Momentum alone": 7,
        "ML alone": 6,
        "ML + Momentum": 5,
    }

    print_slot_utilization(diags, configs, max_slots_map)
    print_signal_conflicts(diags["ML + Momentum"], "ML + Momentum")
    print_overlap_performance(diags["ML + Momentum"])
    print_strategy_pnl(diags["ML + Momentum"], "ML + Momentum")
    print_first_to_fire(diags["ML + Momentum"])
    print_position_count(diags, configs)
    print_capital_efficiency(diags, configs)

    # ── Summary diagnosis ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY DIAGNOSIS")
    print(f"{'='*80}")

    avg_pos = {name: np.mean(diags[name]["daily_pos_count"]) for name in configs}
    avg_cash = {name: np.mean(diags[name]["daily_cash_pct"]) for name in configs}

    print(f"\n  {'Config':<22} {'CAGR':>8} {'Sharpe':>8} {'Avg Pos':>9} {'Avg Cash%':>10} {'Trades':>8}")
    print(f"  {'─'*21} {'─'*7} {'─'*7} {'─'*8} {'─'*9} {'─'*7}")
    for name in configs:
        m = metrics[name]
        print(f"  {name:<22} {m['cagr']:>+7.2%} {m['sharpe']:>7.3f} "
              f"{avg_pos[name]:>8.2f} {avg_cash[name]:>9.1%} {m['n_trades']:>7}")

    # Key insights
    d_comb = diags["ML + Momentum"]
    print(f"\n  KEY FINDINGS:")
    print(f"  • Combined system has {avg_pos['ML + Momentum']:.2f} avg positions vs "
          f"{avg_pos['ML alone']:.2f} (ML) and {avg_pos['Momentum alone']:.2f} (Mom)")
    print(f"  • Combined cash idle: {avg_cash['ML + Momentum']:.1%} vs "
          f"{avg_cash['ML alone']:.1%} (ML) and {avg_cash['Momentum alone']:.1%} (Mom)")

    br = d_comb["block_reasons"]
    total_gen = sum(d_comb["signals_generated"].values())
    total_blocked = sum(br.values())
    print(f"  • {total_blocked:,} of {total_gen:,} signals blocked ({total_blocked/total_gen:.1%})")

    top_block = br.most_common(3)
    for reason, count in top_block:
        print(f"    - {reason}: {count:,}")

    print(f"\n{'='*80}")
    print(f"  Runtime: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
