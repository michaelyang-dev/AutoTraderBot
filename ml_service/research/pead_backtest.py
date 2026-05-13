#!/usr/bin/env python3
"""
PEAD (Post-Earnings Announcement Drift) Backtest
=================================================
Tests the classic PEAD anomaly using WRDS IBES surprise data
and CRSP daily prices. No look-ahead bias.

Strategy:
  - On earnings announcement day, compute SUE score
  - Long top quintile (SUE > threshold), short bottom quintile
  - Hold for HOLD_DAYS trading days
  - Equal-weight within each quintile
  - Rebalance as new earnings come in (rolling entry)

Data:
  - IBES surprise: SUE scores, announcement dates
  - CRSP daily: prices for return computation
  - SP1500 membership: universe filter
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))

# ── Configuration ────────────────────────────────────────────────────────────
START_YEAR    = 2005
END_YEAR      = 2025
HOLD_DAYS     = 60       # trading days to hold after earnings
SUE_LONG_THR  = 2.0      # long if SUE > this
SUE_SHORT_THR = -2.0     # short if SUE < this
MAX_LONG_POS  = 20       # max simultaneous long positions
MAX_SHORT_POS = 10       # max simultaneous short positions
MIN_PRICE     = 5.0      # minimum stock price
INITIAL_CAPITAL = 1_000_000
LONG_ALLOC    = 0.75     # 75% of capital to longs
SHORT_ALLOC   = 0.25     # 25% to shorts
STOP_LOSS     = 0.25     # 25% stop loss
COST_PER_TRADE = 0.001   # 10bp per side (slippage + commission)


def load_data():
    """Load IBES surprise and CRSP price data."""
    print("Loading IBES surprise data...")
    ibes = pd.read_parquet(
        ML_DIR / "data/wrds/ibes_surprise.parquet"
    )
    # Filter: US firms, EPS, quarterly
    ibes = ibes[
        (ibes["MEASURE"] == "EPS") &
        (ibes["FISCALP"] == "QTR") &
        (ibes["USFIRM"] == 1)
    ].copy()

    # Use OFTIC as ticker (matches CRSP Ticker)
    ibes["ticker"] = ibes["OFTIC"].str.strip()
    ibes["ann_date"] = pd.to_datetime(ibes["anndats"], errors="coerce")
    ibes = ibes.dropna(subset=["ann_date", "suescore", "ticker"])

    # Remove extreme SUE outliers (data errors)
    ibes = ibes[ibes["suescore"].abs() < 100]

    print(f"  {len(ibes):,} EPS quarterly surprises, {ibes['ticker'].nunique()} tickers")
    print(f"  Date range: {ibes['ann_date'].min().date()} to {ibes['ann_date'].max().date()}")

    print("Loading CRSP daily prices...")
    prices = pd.read_parquet(
        ML_DIR / "data/wrds/crsp_daily_stock_full.parquet",
        columns=["DlyCalDt", "Ticker", "DlyPrc", "DlyVol", "DlyCap"],
    )
    prices = prices.rename(columns={
        "DlyCalDt": "date", "Ticker": "ticker",
        "DlyPrc": "price", "DlyVol": "volume", "DlyCap": "mktcap",
    })
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].str.strip()
    prices = prices.dropna(subset=["date", "ticker", "price"])
    prices["price"] = prices["price"].abs()  # CRSP uses negative for bid/ask
    prices = prices[prices["price"] > 0]

    # Filter date range
    prices = prices[
        (prices["date"].dt.year >= START_YEAR - 1) &
        (prices["date"].dt.year <= END_YEAR)
    ]
    print(f"  {len(prices):,} price rows, {prices['ticker'].nunique()} tickers")

    # Load SP1500 membership for universe filter
    sp1500 = None
    sp_file = ML_DIR / "data/wrds/sp1500_membership_history.parquet"
    if sp_file.exists():
        sp1500 = pd.read_parquet(sp_file)
        print(f"  SP1500 membership loaded: {len(sp1500):,} rows")

    return ibes, prices, sp1500


def build_price_lookup(prices):
    """Build efficient price lookup: {ticker: DataFrame with date index}."""
    print("Building price lookup...")
    lookup = {}
    for ticker, grp in prices.groupby("ticker"):
        g = grp.set_index("date").sort_index()
        g = g[~g.index.duplicated(keep="first")]
        lookup[ticker] = g
    print(f"  {len(lookup)} tickers indexed")
    return lookup


def get_forward_return(lookup, ticker, entry_date, hold_days):
    """Get forward return over hold_days trading days after entry_date."""
    if ticker not in lookup:
        return None, None, None

    df = lookup[ticker]
    # Find entry: first trading day on or after entry_date
    future = df.loc[entry_date:]
    if len(future) < 2:
        return None, None, None

    entry_price = future.iloc[0]["price"]
    if entry_price <= 0 or np.isnan(entry_price):
        return None, None, None

    # Hold for hold_days trading days
    exit_idx = min(hold_days, len(future) - 1)
    exit_price = future.iloc[exit_idx]["price"]

    # Check for stop loss during hold
    prices_during = future.iloc[:exit_idx + 1]["price"]
    min_price = prices_during.min()
    max_price = prices_during.max()

    ret = (exit_price / entry_price) - 1
    return ret, entry_price, exit_price


def run_quintile_analysis(ibes, lookup):
    """Classic academic PEAD: sort into SUE quintiles, measure drift."""
    print("\n" + "=" * 70)
    print("QUINTILE ANALYSIS (Academic Validation)")
    print("=" * 70)

    results_by_year = {}

    for year in range(START_YEAR, END_YEAR + 1):
        year_events = ibes[ibes["ann_date"].dt.year == year].copy()
        if len(year_events) == 0:
            continue

        # Compute forward returns
        rets = []
        for _, row in year_events.iterrows():
            ticker = row["ticker"]
            ann_date = row["ann_date"]
            sue = row["suescore"]

            ret, _, _ = get_forward_return(lookup, ticker, ann_date, HOLD_DAYS)
            if ret is not None and abs(ret) < 2.0:  # filter extreme moves
                rets.append({"sue": sue, "ret": ret, "ticker": ticker, "date": ann_date})

        if len(rets) < 50:
            continue

        df = pd.DataFrame(rets)
        df["quintile"] = pd.qcut(df["sue"], 5, labels=[1, 2, 3, 4, 5])

        q_rets = df.groupby("quintile")["ret"].mean()
        spread = q_rets[5] - q_rets[1]

        results_by_year[year] = {
            "n_events": len(df),
            "q1_ret": q_rets[1],
            "q5_ret": q_rets[5],
            "spread": spread,
            "q_rets": q_rets,
        }

        print(f"  {year}: n={len(df):4d}  Q1(worst)={q_rets[1]:+.2%}  "
              f"Q5(best)={q_rets[5]:+.2%}  Spread={spread:+.2%}")

    # Summary
    if results_by_year:
        spreads = [v["spread"] for v in results_by_year.values()]
        q1s = [v["q1_ret"] for v in results_by_year.values()]
        q5s = [v["q5_ret"] for v in results_by_year.values()]
        print(f"\n  AVERAGE SPREAD (Q5-Q1): {np.mean(spreads):+.2%}")
        print(f"  Average Q1 (worst surprise): {np.mean(q1s):+.2%}")
        print(f"  Average Q5 (best surprise):  {np.mean(q5s):+.2%}")
        print(f"  Spread positive in {sum(1 for s in spreads if s > 0)}/{len(spreads)} years")
        print(f"  Spread t-stat: {np.mean(spreads) / (np.std(spreads) / np.sqrt(len(spreads))):.2f}")

    return results_by_year


def run_portfolio_backtest(ibes, lookup):
    """Realistic portfolio backtest with position management."""
    print("\n" + "=" * 70)
    print("PORTFOLIO BACKTEST (Realistic Simulation)")
    print(f"  Long: SUE > {SUE_LONG_THR}, max {MAX_LONG_POS} positions")
    print(f"  Short: SUE < {SUE_SHORT_THR}, max {MAX_SHORT_POS} positions")
    print(f"  Hold: {HOLD_DAYS} trading days, Stop: {STOP_LOSS:.0%}")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f} ({LONG_ALLOC:.0%} long / {SHORT_ALLOC:.0%} short)")
    print("=" * 70)

    # Get all trading dates
    all_dates = sorted(set(
        d for df in lookup.values()
        for d in df.index
        if START_YEAR <= d.year <= END_YEAR
    ))
    all_dates = pd.DatetimeIndex(all_dates)

    # Sort events by date
    events = ibes[
        (ibes["ann_date"].dt.year >= START_YEAR) &
        (ibes["ann_date"].dt.year <= END_YEAR)
    ].sort_values("ann_date")

    # Track positions
    long_positions = []   # list of {ticker, entry_date, entry_price, exit_date_idx}
    short_positions = []
    daily_equity = []
    capital = INITIAL_CAPITAL
    total_trades = 0
    total_long_trades = 0
    total_short_trades = 0
    long_wins = 0
    short_wins = 0

    # Pre-group events by date
    events_by_date = {}
    for _, row in events.iterrows():
        d = row["ann_date"]
        # Map to next trading day (entry day)
        idx = all_dates.searchsorted(d)
        if idx < len(all_dates) - 1:
            entry_date = all_dates[idx + 1]  # enter NEXT trading day
        else:
            continue
        if entry_date not in events_by_date:
            events_by_date[entry_date] = []
        events_by_date[entry_date].append(row)

    # Simulate day by day
    prev_equity = INITIAL_CAPITAL
    yearly_start = INITIAL_CAPITAL
    yearly_results = {}
    current_year = START_YEAR

    for i, date in enumerate(all_dates):
        if date.year < START_YEAR:
            continue

        # Year tracking
        if date.year != current_year:
            yr_ret = (prev_equity / yearly_start) - 1
            yearly_results[current_year] = yr_ret
            yearly_start = prev_equity
            current_year = date.year

        # Check for exits (time-based)
        new_longs = []
        for pos in long_positions:
            days_held = np.busday_count(
                pos["entry_date"].date(), date.date()
            )
            if days_held >= HOLD_DAYS:
                # Exit
                if pos["ticker"] in lookup:
                    df = lookup[pos["ticker"]]
                    if date in df.index:
                        exit_price = df.loc[date, "price"]
                        ret = (exit_price / pos["entry_price"]) - 1
                        pnl = pos["size"] * ret
                        capital += pos["size"] + pnl - (pos["size"] * COST_PER_TRADE)
                        total_long_trades += 1
                        if ret > 0:
                            long_wins += 1
                    else:
                        capital += pos["size"]  # can't price, return capital
                else:
                    capital += pos["size"]
            else:
                # Check stop loss
                if pos["ticker"] in lookup:
                    df = lookup[pos["ticker"]]
                    if date in df.index:
                        cur_price = df.loc[date, "price"]
                        ret = (cur_price / pos["entry_price"]) - 1
                        if ret < -STOP_LOSS:
                            pnl = pos["size"] * ret
                            capital += pos["size"] + pnl - (pos["size"] * COST_PER_TRADE)
                            total_long_trades += 1
                            continue  # don't keep position
                new_longs.append(pos)
        long_positions = new_longs

        new_shorts = []
        for pos in short_positions:
            days_held = np.busday_count(
                pos["entry_date"].date(), date.date()
            )
            if days_held >= HOLD_DAYS:
                if pos["ticker"] in lookup:
                    df = lookup[pos["ticker"]]
                    if date in df.index:
                        exit_price = df.loc[date, "price"]
                        ret = (pos["entry_price"] / exit_price) - 1  # short: profit when price drops
                        pnl = pos["size"] * ret
                        capital += pos["size"] + pnl - (pos["size"] * COST_PER_TRADE)
                        total_short_trades += 1
                        if ret > 0:
                            short_wins += 1
                    else:
                        capital += pos["size"]
                else:
                    capital += pos["size"]
            else:
                # Check stop loss (short: stock rises)
                if pos["ticker"] in lookup:
                    df = lookup[pos["ticker"]]
                    if date in df.index:
                        cur_price = df.loc[date, "price"]
                        ret_against = (cur_price / pos["entry_price"]) - 1
                        if ret_against > STOP_LOSS:
                            pnl = pos["size"] * (-ret_against)
                            capital += pos["size"] + pnl - (pos["size"] * COST_PER_TRADE)
                            total_short_trades += 1
                            continue
                new_shorts.append(pos)
        short_positions = new_shorts

        # Process new entries
        if date in events_by_date:
            day_events = events_by_date[date]

            # Sort by SUE magnitude for priority
            long_candidates = sorted(
                [e for e in day_events if e["suescore"] > SUE_LONG_THR],
                key=lambda x: x["suescore"], reverse=True
            )
            short_candidates = sorted(
                [e for e in day_events if e["suescore"] < SUE_SHORT_THR],
                key=lambda x: x["suescore"]
            )

            # Filter: min price, has price data, not already in portfolio
            held_tickers = set(p["ticker"] for p in long_positions + short_positions)

            for cand in long_candidates:
                if len(long_positions) >= MAX_LONG_POS:
                    break
                t = cand["ticker"]
                if t in held_tickers or t not in lookup:
                    continue
                df = lookup[t]
                if date not in df.index:
                    continue
                entry_price = df.loc[date, "price"]
                if entry_price < MIN_PRICE:
                    continue

                # Size: equal weight within long allocation
                size = min(
                    capital * 0.10,  # max 10% per position
                    (INITIAL_CAPITAL * LONG_ALLOC) / MAX_LONG_POS
                )
                if size <= 0 or capital < size:
                    continue

                capital -= size + (size * COST_PER_TRADE)
                long_positions.append({
                    "ticker": t,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "size": size,
                    "sue": cand["suescore"],
                })
                held_tickers.add(t)
                total_trades += 1

            for cand in short_candidates:
                if len(short_positions) >= MAX_SHORT_POS:
                    break
                t = cand["ticker"]
                if t in held_tickers or t not in lookup:
                    continue
                df = lookup[t]
                if date not in df.index:
                    continue
                entry_price = df.loc[date, "price"]
                if entry_price < MIN_PRICE:
                    continue

                size = min(
                    capital * 0.05,  # max 5% per short position
                    (INITIAL_CAPITAL * SHORT_ALLOC) / MAX_SHORT_POS
                )
                if size <= 0 or capital < size:
                    continue

                capital -= size + (size * COST_PER_TRADE)
                short_positions.append({
                    "ticker": t,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "size": size,
                    "sue": cand["suescore"],
                })
                held_tickers.add(t)
                total_trades += 1

        # Mark-to-market
        long_mtm = 0
        for pos in long_positions:
            if pos["ticker"] in lookup:
                df = lookup[pos["ticker"]]
                if date in df.index:
                    cur_price = df.loc[date, "price"]
                    ret = (cur_price / pos["entry_price"]) - 1
                    long_mtm += pos["size"] * (1 + ret)
                else:
                    long_mtm += pos["size"]
            else:
                long_mtm += pos["size"]

        short_mtm = 0
        for pos in short_positions:
            if pos["ticker"] in lookup:
                df = lookup[pos["ticker"]]
                if date in df.index:
                    cur_price = df.loc[date, "price"]
                    ret = (cur_price / pos["entry_price"]) - 1
                    short_mtm += pos["size"] * (1 - ret)  # short gains when price drops
                else:
                    short_mtm += pos["size"]
            else:
                short_mtm += pos["size"]

        equity = capital + long_mtm + short_mtm
        prev_equity = equity

        daily_equity.append({"date": date, "equity": equity,
                            "n_long": len(long_positions),
                            "n_short": len(short_positions),
                            "capital": capital})

    # Final year
    if current_year not in yearly_results and yearly_start > 0:
        yearly_results[current_year] = (prev_equity / yearly_start) - 1

    return daily_equity, yearly_results, total_trades, total_long_trades, total_short_trades, long_wins, short_wins


def compute_metrics(daily_equity):
    """Compute portfolio performance metrics."""
    df = pd.DataFrame(daily_equity).set_index("date")
    df["returns"] = df["equity"].pct_change()

    total_ret = (df["equity"].iloc[-1] / df["equity"].iloc[0]) - 1
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1

    ann_ret = df["returns"].mean() * 252
    ann_vol = df["returns"].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cummax = df["equity"].cummax()
    drawdown = (df["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    # Avg positions
    avg_long = df["n_long"].mean()
    avg_short = df["n_short"].mean()

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_long_pos": avg_long,
        "avg_short_pos": avg_short,
        "years": years,
    }


def main():
    print("=" * 70)
    print("PEAD BACKTEST — Post-Earnings Announcement Drift")
    print(f"Period: {START_YEAR}–{END_YEAR}")
    print("=" * 70)

    ibes, prices, sp1500 = load_data()
    lookup = build_price_lookup(prices)

    # Phase 1: Academic quintile validation
    quintile_results = run_quintile_analysis(ibes, lookup)

    # Phase 2: Realistic portfolio simulation
    daily_equity, yearly_results, total_trades, n_long, n_short, long_wins, short_wins = \
        run_portfolio_backtest(ibes, lookup)

    if not daily_equity:
        print("ERROR: No daily equity data generated")
        return

    metrics = compute_metrics(daily_equity)

    print("\n" + "=" * 70)
    print("PORTFOLIO RESULTS")
    print("=" * 70)
    print(f"  CAGR:           {metrics['cagr']:+.1%}")
    print(f"  Sharpe Ratio:   {metrics['sharpe']:.2f}")
    print(f"  Annual Vol:     {metrics['annual_vol']:.1%}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.1%}")
    print(f"  Total Return:   {metrics['total_return']:+.1%}")
    print(f"  Period:         {metrics['years']:.1f} years")
    print(f"  Avg Long Pos:   {metrics['avg_long_pos']:.1f}")
    print(f"  Avg Short Pos:  {metrics['avg_short_pos']:.1f}")
    print(f"  Total Trades:   {total_trades}")
    print(f"  Long Trades:    {n_long} (win rate: {long_wins/max(n_long,1):.1%})")
    print(f"  Short Trades:   {n_short} (win rate: {short_wins/max(n_short,1):.1%})")

    print("\n  YEARLY RETURNS:")
    for year in sorted(yearly_results):
        ret = yearly_results[year]
        bar = "+" * int(max(0, ret) * 100) + "-" * int(max(0, -ret) * 100)
        print(f"    {year}: {ret:+7.1%}  {bar}")

    pos_years = sum(1 for r in yearly_results.values() if r > 0)
    print(f"\n  Positive years: {pos_years}/{len(yearly_results)}")

    # Phase 3: Long-only variant (simpler)
    print("\n" + "=" * 70)
    print("LONG-ONLY VARIANT (for comparison)")
    print("=" * 70)

    global MAX_SHORT_POS, SHORT_ALLOC, LONG_ALLOC
    MAX_SHORT_POS = 0
    SHORT_ALLOC = 0
    LONG_ALLOC = 1.0

    daily_eq_long, yearly_long, _, nl, _, lw, _ = run_portfolio_backtest(ibes, lookup)
    if daily_eq_long:
        m_long = compute_metrics(daily_eq_long)
        print(f"  CAGR:           {m_long['cagr']:+.1%}")
        print(f"  Sharpe Ratio:   {m_long['sharpe']:.2f}")
        print(f"  Max Drawdown:   {m_long['max_drawdown']:.1%}")
        print(f"  Long Win Rate:  {lw/max(nl,1):.1%}")
        print("\n  YEARLY:")
        for year in sorted(yearly_long):
            print(f"    {year}: {yearly_long[year]:+7.1%}")


if __name__ == "__main__":
    main()
