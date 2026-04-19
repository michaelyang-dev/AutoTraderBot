"""
Diagnostic verification of the 60-sym v2 threshold sweep results.
Checks whether the 68.1% win rate at >0.57 is real or a statistical artifact.
"""
import warnings, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from backtest_ml import (
    INITIAL_CASH, MAX_POSITIONS, POSITION_PCT, SLIPPAGE, HOLD_DAYS,
    fetch_benchmarks, make_ml_signal_fn,
)

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent / "data"
PRED_FILE = DATA / "predictions_60universe_v2.parquet"


# ═══════════════════════════════════════════════════════════════════════════════
#  1. WALK-FORWARD WINDOW TREE COUNT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_walk_forward_windows(df):
    """Parse training logs to check tree counts per window.
    Since we don't have the logs, infer from the predictions:
    count how many picks each window produced at various thresholds."""
    print("=" * 75)
    print("  1. WALK-FORWARD WINDOW ANALYSIS")
    print("=" * 75)

    # The OOS predictions span non-overlapping 6-month test windows.
    # We can identify windows by date gaps > 5 days.
    dates_sorted = sorted(df["date"].unique())
    windows = []
    win_start = dates_sorted[0]
    prev = dates_sorted[0]

    for d in dates_sorted[1:]:
        gap = (d - prev).days
        if gap > 15:  # new window
            windows.append((win_start, prev))
            win_start = d
        prev = d
    windows.append((win_start, prev))

    print(f"\n  Detected {len(windows)} walk-forward windows\n")
    print(f"  {'Window':<6} {'Period':<28} {'Rows':>6} {'p>0.55':>8} {'p>0.57':>8} "
          f"{'p>0.60':>8} {'Max Prob':>9} {'Likely Underfit'}")
    print(f"  {'─'*5} {'─'*27} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*15}")

    underfit_windows = []
    for i, (ws, we) in enumerate(windows):
        mask = (df["date"] >= ws) & (df["date"] <= we)
        wdf = df[mask]
        n = len(wdf)
        p55 = (wdf["prob"] >= 0.55).sum()
        p57 = (wdf["prob"] >= 0.57).sum()
        p60 = (wdf["prob"] >= 0.60).sum()
        mx = wdf["prob"].max()

        # Heuristic: if max prob < 0.55, the window likely had very few trees
        underfit = mx < 0.55
        marker = "  *** YES ***" if underfit else ""
        underfit_windows.append((i + 1, underfit, ws, we, p57))

        print(f"  W{i+1:02d}   {str(ws.date()):>10}→{str(we.date()):>10}  "
              f"{n:>6} {p55:>8} {p57:>8} {p60:>8} {mx:>9.4f} {marker}")

    n_underfit = sum(1 for _, uf, *_ in underfit_windows if uf)
    print(f"\n  Windows with max_prob < 0.55 (likely <10 trees): {n_underfit} / {len(windows)}")

    zero_trade_windows = [(w, ws, we) for w, uf, ws, we, p57 in underfit_windows if uf and p57 == 0]
    print(f"  Of those, windows producing ZERO picks at >0.57: {len(zero_trade_windows)}")
    if zero_trade_windows:
        for w, ws, we in zero_trade_windows:
            print(f"    W{w:02d}: {ws.date()} → {we.date()}")

    return windows


# ═══════════════════════════════════════════════════════════════════════════════
#  2-6. DETAILED TRADE SIMULATION WITH METADATA
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_with_metadata(df, threshold, spy_prices):
    """
    Replays the backtest simulation but records per-trade metadata:
    symbol, entry_date, exit_date, return, prob.
    """
    sigs_by_date = {}
    for date, grp in df[["date", "symbol", "prob", "fwd_ret"]].groupby("date"):
        g = grp.sort_values("prob", ascending=False)
        sigs_by_date[date] = list(g[["symbol", "prob", "fwd_ret"]].itertuples(
            index=False, name=None))

    all_dates = sorted(df["date"].unique())
    n_dates = len(all_dates)
    cash = float(INITIAL_CASH)
    idle_spy_shares = 0.0
    positions = {}
    trade_records = []

    signal_fn = make_ml_signal_fn(threshold)

    for i, date in enumerate(all_dates):
        spy_px = spy_prices.get(date) if spy_prices else None
        if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
            spy_px = None

        # Close positions
        for sym in [s for s, p in positions.items() if p["exit_idx"] == i]:
            pos = positions.pop(sym)
            gross = pos["cost"] * (1.0 + pos["fwd_ret"])
            net = gross * (1.0 - SLIPPAGE)
            cash += net
            ret = (net - pos["cost"]) / pos["cost"]
            trade_records.append({
                "symbol": sym,
                "entry_date": pos["entry_date"],
                "exit_date": date,
                "entry_idx": pos["entry_idx"],
                "exit_idx": i,
                "hold_days": i - pos["entry_idx"],
                "prob": pos["prob"],
                "fwd_ret": pos["fwd_ret"],
                "trade_ret": ret,
                "cost": pos["cost"],
                "win": ret > 0,
            })

        # Buy
        slots = MAX_POSITIONS - len(positions)
        held = set(positions.keys())
        day_sigs = signal_fn(date, held, sigs_by_date.get(date, []))

        if spy_px and idle_spy_shares > 0 and day_sigs and slots > 0:
            proceeds = idle_spy_shares * spy_px * (1.0 - SLIPPAGE)
            cash += proceeds
            idle_spy_shares = 0.0

        for sym, prob, fwd_ret in day_sigs[:slots]:
            if np.isnan(fwd_ret):
                continue
            ml_mult = min(1.0, max(0.60, prob * 1.6 - 0.28))
            port_est = cash + sum(p["cost"] for p in positions.values())
            target_val = port_est * POSITION_PCT * ml_mult
            cost = min(target_val, cash * 0.95)
            if cost < 50.0:
                continue
            cash -= cost * (1.0 + SLIPPAGE)
            exit_idx = min(i + HOLD_DAYS, n_dates - 1)
            positions[sym] = dict(
                cost=cost, fwd_ret=fwd_ret, exit_idx=exit_idx,
                entry_idx=i, prob=prob, entry_date=date,
            )

        # Park idle cash
        if spy_px:
            ml_pos_val = sum(
                p["cost"] * (1.0 + p["fwd_ret"] * (i - p["entry_idx"]) / HOLD_DAYS)
                for p in positions.values()
            )
            est_port = cash + idle_spy_shares * spy_px + ml_pos_val
            reserved = est_port * 0.30
            idle_cash = cash - reserved
            if idle_cash > est_port * 0.20:
                invest = min(idle_cash * 0.85, cash * 0.95)
                new_shares = invest / spy_px
                cash -= invest * (1.0 + SLIPPAGE)
                idle_spy_shares += new_shares

    return pd.DataFrame(trade_records)


def print_trades_by_year(trades_57, trades_55):
    """Checks 2-6 from the user's request."""

    # ── 2. Distribution of trades by year at 0.57 ──
    print("\n" + "=" * 75)
    print("  2. TRADE DISTRIBUTION BY YEAR @ threshold >0.57")
    print("=" * 75)

    trades_57["year"] = trades_57["entry_date"].dt.year
    trades_55["year"] = trades_55["entry_date"].dt.year

    years_57 = trades_57.groupby("year").agg(
        trades=("win", "count"),
        wins=("win", "sum"),
        win_rate=("win", "mean"),
        avg_ret=("trade_ret", "mean"),
        total_ret=("trade_ret", "sum"),
    )
    print(f"\n  {'Year':<6} {'Trades':>7} {'Wins':>6} {'WR':>8} {'Avg Ret':>10} {'Total Ret':>11}")
    print(f"  {'─'*5} {'─'*7} {'─'*6} {'─'*8} {'─'*10} {'─'*11}")
    for yr, row in years_57.iterrows():
        flag = " ***" if row["trades"] < 10 else ""
        print(f"  {yr:<6} {int(row['trades']):>7} {int(row['wins']):>6} "
              f"{row['win_rate']:>7.1%} {row['avg_ret']:>+9.2%} "
              f"{row['total_ret']:>+10.2%}{flag}")
    print(f"\n  *** = fewer than 10 trades (low statistical significance)")

    n_years_with_trades = (years_57["trades"] >= 1).sum()
    n_years_sparse = (years_57["trades"] < 10).sum()
    print(f"\n  Years with any trades: {n_years_with_trades}")
    print(f"  Years with <10 trades: {n_years_sparse}")

    # Concentration check
    top3 = years_57.nlargest(3, "trades")
    top3_pct = top3["trades"].sum() / years_57["trades"].sum()
    print(f"  Top 3 years by trade count hold {top3_pct:.1%} of all trades")

    # ── 3. Confidence interval for 68.1% win rate ──
    print("\n" + "=" * 75)
    print("  3. WIN RATE CONFIDENCE INTERVAL @ threshold >0.57")
    print("=" * 75)

    n = len(trades_57)
    wins = int(trades_57["win"].sum())
    wr = wins / n

    # Wilson score interval
    z = 1.96  # 95% CI
    denom = 1 + z**2 / n
    center = (wr + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((wr * (1 - wr) + z**2 / (4 * n)) / n) / denom
    ci_low = center - spread
    ci_high = center + spread

    # Also bootstrap CI
    rng = np.random.default_rng(42)
    boot_wrs = []
    for _ in range(10000):
        sample = rng.choice(trades_57["win"].values, size=n, replace=True)
        boot_wrs.append(sample.mean())
    boot_low = np.percentile(boot_wrs, 2.5)
    boot_high = np.percentile(boot_wrs, 97.5)

    print(f"\n  Total trades: {n}")
    print(f"  Wins: {wins}  |  Losses: {n - wins}")
    print(f"  Win rate: {wr:.1%}")
    print(f"\n  95% Wilson CI:     [{ci_low:.1%}, {ci_high:.1%}]  (width: {(ci_high-ci_low)*100:.1f}pp)")
    print(f"  95% Bootstrap CI:  [{boot_low:.1%}, {boot_high:.1%}]  (width: {(boot_high-boot_low)*100:.1f}pp)")

    if (ci_high - ci_low) < 0.12:
        print(f"\n  VERDICT: CI width < 12pp — reasonably tight for {n} trades")
    else:
        print(f"\n  VERDICT: CI width >= 12pp — wide interval, sample may be too small")

    # ── 4. Years with fewer than 10 trades ──
    print("\n" + "=" * 75)
    print("  4. YEARS WITH FEWER THAN 10 TRADES @ >0.57")
    print("=" * 75)

    sparse_years = years_57[years_57["trades"] < 10]
    if len(sparse_years) == 0:
        print("\n  None — all years have 10+ trades")
    else:
        print(f"\n  {len(sparse_years)} year(s) with <10 trades:")
        for yr, row in sparse_years.iterrows():
            print(f"    {yr}: {int(row['trades'])} trades, WR {row['win_rate']:.0%}")

        # What if we exclude sparse years?
        dense = trades_57[trades_57["year"].isin(years_57[years_57["trades"] >= 10].index)]
        if len(dense) > 0:
            print(f"\n  Win rate EXCLUDING sparse years: {dense['win'].mean():.1%} "
                  f"({len(dense)} trades)")
        print(f"  Win rate INCLUDING all years:     {wr:.1%} ({n} trades)")

    # ── 5. Symbol concentration ──
    print("\n" + "=" * 75)
    print("  5. SYMBOL CONCENTRATION @ >0.57")
    print("=" * 75)

    sym_stats = trades_57.groupby("symbol").agg(
        trades=("win", "count"),
        wins=("win", "sum"),
        win_rate=("win", "mean"),
        avg_ret=("trade_ret", "mean"),
        avg_prob=("prob", "mean"),
        avg_hold=("hold_days", "mean"),
    ).sort_values("trades", ascending=False)

    print(f"\n  Total unique symbols traded: {len(sym_stats)}")
    print(f"\n  Top 15 symbols by trade count:")
    print(f"  {'Symbol':<8} {'Trades':>7} {'Wins':>6} {'WR':>7} {'Avg Ret':>9} "
          f"{'Avg Prob':>9} {'Avg Hold':>9}")
    print(f"  {'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*9} {'─'*9} {'─'*9}")
    for sym, row in sym_stats.head(15).iterrows():
        print(f"  {sym:<8} {int(row['trades']):>7} {int(row['wins']):>6} "
              f"{row['win_rate']:>6.0%} {row['avg_ret']:>+8.2%} "
              f"{row['avg_prob']:>9.3f} {row['avg_hold']:>8.1f}d")

    # Concentration: top 5 symbols
    top5 = sym_stats.head(5)
    top5_pct = top5["trades"].sum() / len(trades_57)
    top5_wins = top5["wins"].sum()
    top5_wr = top5_wins / top5["trades"].sum() if top5["trades"].sum() > 0 else 0
    rest = trades_57[~trades_57["symbol"].isin(top5.index)]
    rest_wr = rest["win"].mean() if len(rest) > 0 else 0

    print(f"\n  Top 5 symbols: {top5_pct:.1%} of all trades "
          f"({int(top5['trades'].sum())}/{len(trades_57)})")
    print(f"    Top 5 win rate: {top5_wr:.1%}")
    print(f"    Rest win rate:  {rest_wr:.1%}")

    if top5_pct > 0.60:
        print(f"\n  WARNING: >60% of trades concentrated in 5 symbols — "
              f"may reflect stock-specific luck")
    elif top5_pct > 0.40:
        print(f"\n  CAUTION: 40-60% concentration in top 5 — moderate symbol bias")
    else:
        print(f"\n  OK: <40% concentration — trades are well-diversified across symbols")

    # Average hold time and position size
    print(f"\n  Average hold time: {trades_57['hold_days'].mean():.1f} days "
          f"(expected: {HOLD_DAYS})")
    print(f"  Average position size: ${trades_57['cost'].mean():,.0f}")
    print(f"  Median position size:  ${trades_57['cost'].median():,.0f}")

    # ── 6. Win rate by year: 0.57 vs 0.55 ──
    print("\n" + "=" * 75)
    print("  6. WIN RATE BY YEAR: >0.57 vs >0.55")
    print("=" * 75)

    years_55 = trades_55.groupby("year").agg(
        trades=("win", "count"),
        win_rate=("win", "mean"),
    )

    all_years = sorted(set(years_57.index) | set(years_55.index))
    print(f"\n  {'Year':<6} {'Trades@55':>10} {'WR@55':>8} {'Trades@57':>10} "
          f"{'WR@57':>8} {'WR Delta':>9}")
    print(f"  {'─'*5} {'─'*10} {'─'*8} {'─'*10} {'─'*8} {'─'*9}")

    big_swings = []
    for yr in all_years:
        t55 = int(years_55.loc[yr, "trades"]) if yr in years_55.index else 0
        wr55 = years_55.loc[yr, "win_rate"] if yr in years_55.index else float("nan")
        t57 = int(years_57.loc[yr, "trades"]) if yr in years_57.index else 0
        wr57 = years_57.loc[yr, "win_rate"] if yr in years_57.index else float("nan")
        delta = wr57 - wr55 if not (np.isnan(wr55) or np.isnan(wr57)) else float("nan")

        wr55_s = f"{wr55:.0%}" if not np.isnan(wr55) else "—"
        wr57_s = f"{wr57:.0%}" if not np.isnan(wr57) else "—"
        delta_s = f"{delta:+.0%}" if not np.isnan(delta) else "—"
        flag = ""
        if not np.isnan(delta) and abs(delta) > 0.20:
            flag = " ***"
            big_swings.append((yr, delta, t57))

        print(f"  {yr:<6} {t55:>10} {wr55_s:>8} {t57:>10} {wr57_s:>8} {delta_s:>9}{flag}")

    print(f"\n  *** = win rate difference > 20pp between thresholds")
    if big_swings:
        print(f"\n  Years with large WR swings (>20pp):")
        for yr, delta, nt in big_swings:
            if nt < 10:
                print(f"    {yr}: {delta:+.0%} swing, but only {nt} trades at >0.57 "
                      f"— LIKELY NOISE")
            else:
                print(f"    {yr}: {delta:+.0%} swing with {nt} trades — "
                      f"{'concerning' if abs(delta) > 0.30 else 'moderate'}")
    else:
        print(f"\n  No large WR swings — threshold change doesn't cherry-pick winners")


def main():
    print("=" * 75)
    print("  THRESHOLD VERIFICATION — 60-Symbol v2 Model")
    print("=" * 75)

    df = pd.read_parquet(PRED_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])
    print(f"  Loaded {len(df):,} rows | {df['symbol'].nunique()} symbols")

    # 1. Walk-forward window analysis
    windows = analyze_walk_forward_windows(df)

    # Fetch SPY for the simulation
    all_dates = sorted(df["date"].unique())
    start = pd.Timestamp(all_dates[0]).strftime("%Y-%m-%d")
    end = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close = fetch_benchmarks(start, end, sorted(df["symbol"].unique()))
    close = close.reindex(
        pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates]), method="ffill")
    spy_dict = close["SPY"].to_dict()

    # 2-6. Simulate with metadata at both thresholds
    print(f"\n  Simulating trades at >0.57 ...")
    trades_57 = simulate_with_metadata(df, 0.57, spy_dict)
    print(f"  → {len(trades_57)} trades")

    print(f"  Simulating trades at >0.55 ...")
    trades_55 = simulate_with_metadata(df, 0.55, spy_dict)
    print(f"  → {len(trades_55)} trades")

    print_trades_by_year(trades_57, trades_55)

    # ── Final verdict ──
    print("\n" + "=" * 75)
    print("  FINAL VERDICT")
    print("=" * 75)

    n = len(trades_57)
    wr = trades_57["win"].mean()
    sym_counts = trades_57["symbol"].value_counts()
    top5_pct = sym_counts.head(5).sum() / n
    years_57 = trades_57.groupby(trades_57["entry_date"].dt.year)["win"].agg(["count", "mean"])
    sparse_years = (years_57["count"] < 10).sum()
    total_years = len(years_57)

    issues = []
    if n < 200:
        issues.append(f"Small sample: {n} trades (want 200+)")
    if top5_pct > 0.50:
        issues.append(f"Symbol concentration: top 5 = {top5_pct:.0%} of trades")
    if sparse_years > total_years * 0.5:
        issues.append(f"{sparse_years}/{total_years} years have <10 trades")
    # Check if WR is driven by a single year
    best_yr_wr = years_57["mean"].max()
    best_yr_n = years_57.loc[years_57["mean"].idxmax(), "count"]
    if best_yr_n > n * 0.3 and best_yr_wr > wr + 0.15:
        issues.append(f"Best year ({years_57['mean'].idxmax()}) has {best_yr_wr:.0%} WR "
                      f"with {best_yr_n:.0f}/{n} trades — dominates overall")

    if not issues:
        print(f"\n  The {wr:.1%} win rate at >0.57 appears GENUINE.")
        print(f"  - {n} trades is a reasonable sample")
        print(f"  - Trades are diversified across symbols and years")
        print(f"  - No single year or symbol dominates the result")
    else:
        print(f"\n  The {wr:.1%} win rate at >0.57 has POTENTIAL ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
        print(f"\n  Treat the headline {wr:.1%} with caution. The true win rate")
        print(f"  is likely within the confidence interval reported above.")

    print("=" * 75)


if __name__ == "__main__":
    main()
