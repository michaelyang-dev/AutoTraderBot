"""
Uncorrelated Strategy Research
================================
Build and test strategies with low correlation to our momentum engine.
Goal: combine two uncorrelated positive-return streams → higher Sharpe.

Three candidates (all using data we already have):
1. Index trend-following (time-series momentum on SPY/QQQ)
2. Short-horizon mean reversion (5-day oversold bounce in SP1500)
3. Cross-asset trend (GLD/TLT trend-following)

Each tested standalone, then combined with our momentum strategy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from main_production_backtest import FastBacktester
import numpy as np, pandas as pd, time


def strategy_index_trend(prices_df, etf_df, start, end, lookback=200, fast=50):
    """
    Time-series momentum on SPY.
    Long SPY when price > SMA(lookback), cash otherwise.
    Uses fast SMA for additional signal.

    This is structurally different from cross-sectional equity momentum:
    - Our main strategy picks WHICH stocks beat others
    - This strategy decides WHEN to be in the market at all
    """
    dates = sorted(prices_df.index)
    dates = [d for d in dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    spy = prices_df["SPY"].reindex(dates).fillna(method='ffill')
    sma_long = spy.rolling(lookback).mean()
    sma_short = spy.rolling(fast).mean()

    capital = 100000
    position = 0  # shares of SPY
    port_values = []

    for date in dates:
        px = spy.get(date)
        sma_l = sma_long.get(date)
        sma_s = sma_short.get(date)

        if pd.isna(px) or pd.isna(sma_l):
            port_values.append((date, capital + position * (px if pd.notna(px) else 0)))
            continue

        total = capital + position * px

        # Signal: long when price > SMA200 AND SMA50 > SMA200 (golden cross)
        should_be_long = px > sma_l

        if should_be_long and position == 0:
            # Buy
            position = total * 0.99 / px  # 99% invested, 1% cash buffer
            capital = total * 0.01
        elif not should_be_long and position > 0:
            # Sell
            capital = position * px * 0.999  # 0.1% cost
            position = 0

        port_values.append((date, capital + position * px))

    return pd.Series([v[1] for v in port_values],
                      index=pd.DatetimeIndex([v[0] for v in port_values]))


def strategy_mean_reversion(prices_df, features_by_date, sp_mem, start, end,
                             lookback=5, n_picks=10, hold_days=5):
    """
    Short-horizon mean reversion in SP1500.
    Buy stocks that dropped most in last 5 days, hold for 5 days.

    Opposite horizon to our 12-month momentum — should be uncorrelated.
    """
    dates = sorted(prices_df.index)
    dates = [d for d in dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    capital = 100000
    holdings = {}  # {sym: {shares, entry_date, entry_px}}
    port_values = []

    for day_idx, date in enumerate(dates):
        # Get current prices
        today = {}
        if date in prices_df.index:
            row = prices_df.loc[date]
            for sym in row.dropna().index:
                today[sym] = row[sym]

        # Update portfolio value
        total = capital + sum(h["shares"] * today.get(s, h["entry_px"])
                               for s, h in holdings.items())

        # Exit positions held for hold_days
        for sym in list(holdings):
            if day_idx - holdings[sym]["entry_day"] >= hold_days:
                px = today.get(sym, holdings[sym]["entry_px"])
                capital += holdings[sym]["shares"] * px * 0.999  # 0.1% cost
                del holdings[sym]

        # Every hold_days, pick new positions
        if day_idx % hold_days == 0 and len(holdings) == 0:
            # Get universe members
            members = set()
            for mem in [sp_mem]:
                if date in mem:
                    members.update(mem[date])
                else:
                    prior = [d for d in mem.keys() if d <= date]
                    if prior:
                        members.update(mem[max(prior)])

            # Compute 5-day returns
            ret5 = {}
            fdate = features_by_date.get(date, {})
            for sym in members:
                sf = fdate.get(sym, {})
                r5 = sf.get("ret_5d")
                if r5 is not None and not np.isnan(r5) and r5 < -0.05:  # dropped >5%
                    # Filter: must be above SMA200 (not in death spiral)
                    d200 = sf.get("dist_sma200")
                    if d200 is not None and not np.isnan(d200) and d200 > -0.10:
                        ret5[sym] = r5

            if not ret5:
                port_values.append((date, total))
                continue

            # Pick the n most oversold
            sorted_syms = sorted(ret5, key=ret5.get)[:n_picks]
            per_stock = total * 0.95 / len(sorted_syms)  # 95% invested

            for sym in sorted_syms:
                px = today.get(sym)
                if px and px > 0:
                    shares = per_stock / px
                    holdings[sym] = {"shares": shares, "entry_px": px, "entry_day": day_idx}
                    capital -= shares * px * 1.001  # 0.1% cost

        port_values.append((date, total))

    return pd.Series([v[1] for v in port_values],
                      index=pd.DatetimeIndex([v[0] for v in port_values]))


def strategy_cross_asset_trend(prices_df, etf_df, start, end):
    """
    Trend-following on GLD and TLT (or bond proxy).
    Long each when price > SMA200, equal weight.

    Structurally uncorrelated with equity momentum.
    """
    dates = sorted(prices_df.index)
    dates = [d for d in dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    capital = 100000
    gld_pos = 0
    tlt_pos = 0  # We'll use a bond proxy from FRED rates if TLT not available
    port_values = []

    # Check if GLD is in ETF data
    has_gld = "GLD" in etf_df.columns if etf_df is not None else False

    for date in dates:
        gld_px = None
        if has_gld and date in etf_df.index:
            gld_px = etf_df.loc[date].get("GLD")
            if pd.isna(gld_px):
                gld_px = None

        total = capital
        if gld_pos > 0 and gld_px:
            total += gld_pos * gld_px

        # GLD trend signal
        if has_gld and gld_px:
            gld_hist = etf_df["GLD"].loc[:date].dropna()
            if len(gld_hist) >= 200:
                gld_sma200 = gld_hist.tail(200).mean()
                should_hold_gld = gld_px > gld_sma200

                if should_hold_gld and gld_pos == 0:
                    alloc = total * 0.50  # 50% to GLD
                    gld_pos = alloc / gld_px
                    capital -= alloc
                elif not should_hold_gld and gld_pos > 0:
                    capital += gld_pos * gld_px * 0.999
                    gld_pos = 0

        total = capital + (gld_pos * gld_px if gld_pos > 0 and gld_px else 0)
        port_values.append((date, max(total, 1)))

    return pd.Series([v[1] for v in port_values],
                      index=pd.DatetimeIndex([v[0] for v in port_values]))


def compute_metrics(values):
    """Compute CAGR, Sharpe, MaxDD from a value series."""
    if len(values) < 2:
        return {}
    n_years = (values.index[-1] - values.index[0]).days / 365.25
    if n_years <= 0:
        return {}
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / n_years) - 1
    rets = values.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    peak = values.cummax()
    max_dd = ((values - peak) / peak).min()
    monthly = values.resample('ME').last().pct_change().dropna()
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "monthly": monthly}


def main():
    print("="*70)
    print("UNCORRELATED STRATEGY TESTING")
    print("="*70)

    bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

    cfg = {
        'mom_w': 0.50, 'val_w': 0.35, 'lv_w': 0.15, 'sec_w': 0.0,
        'top_n': 5, 'rebal_days': 20, 'trailing_stop': 0.40,
        'cap': 0.15, 'use_rp': False,
        'trend_scale': {'bear': 0.40, 'caution': 0.65},
    }

    # ═══ Get our momentum strategy returns ═══
    print("\n[1] Running momentum strategy (baseline)...")
    r_mom = bt.run("2018-01-01", "2025-12-31", cfg)
    mom_daily = r_mom['daily_values']
    mom_monthly = mom_daily.resample('ME').last().pct_change().dropna()
    mom_metrics = compute_metrics(mom_daily)
    print(f"  Momentum: CAGR={mom_metrics['cagr']*100:.1f}%, Sharpe={mom_metrics['sharpe']:.2f}, MaxDD={mom_metrics['max_dd']*100:.1f}%")

    # ═══ Strategy 1: Index Trend-Following ═══
    print("\n[2] Testing index trend-following (SPY SMA200)...")
    trend_daily = strategy_index_trend(bt.prices, bt.etf_df, "2018-01-01", "2025-12-31")
    trend_metrics = compute_metrics(trend_daily)
    print(f"  Index trend: CAGR={trend_metrics['cagr']*100:.1f}%, Sharpe={trend_metrics['sharpe']:.2f}, MaxDD={trend_metrics['max_dd']*100:.1f}%")

    # ═══ Strategy 2: Mean Reversion ═══
    print("\n[3] Testing short-horizon mean reversion...")
    # Build SP membership dict
    sp_mem = {}
    for d in bt.sp500_mem:
        sp_mem[d] = set()
        for mem in [bt.sp500_mem, bt.sp400_mem, bt.sp600_mem]:
            if d in mem:
                sp_mem[d].update(mem[d])

    mr_daily = strategy_mean_reversion(bt.prices, bt.features_by_date, sp_mem,
                                        "2018-01-01", "2025-12-31")
    mr_metrics = compute_metrics(mr_daily)
    print(f"  Mean reversion: CAGR={mr_metrics['cagr']*100:.1f}%, Sharpe={mr_metrics['sharpe']:.2f}, MaxDD={mr_metrics['max_dd']*100:.1f}%")

    # ═══ Strategy 3: Cross-Asset Trend (GLD) ═══
    print("\n[4] Testing cross-asset trend (GLD)...")
    ca_daily = strategy_cross_asset_trend(bt.prices, bt.etf_df, "2018-01-01", "2025-12-31")
    ca_metrics = compute_metrics(ca_daily)
    print(f"  Cross-asset: CAGR={ca_metrics['cagr']*100:.1f}%, Sharpe={ca_metrics['sharpe']:.2f}, MaxDD={ca_metrics['max_dd']*100:.1f}%")

    # ═══ Correlation Matrix ═══
    print(f"\n[5] CORRELATION MATRIX (monthly returns)")
    print("-"*50)

    all_monthly = pd.DataFrame({
        "Momentum": mom_metrics['monthly'],
        "Index Trend": trend_metrics['monthly'],
        "Mean Rev": mr_metrics['monthly'],
        "Cross-Asset": ca_metrics['monthly'],
    }).dropna()

    corr = all_monthly.corr()
    print(f"\n{corr.round(3).to_string()}")

    print(f"\n  Key correlations with Momentum:")
    for col in ["Index Trend", "Mean Rev", "Cross-Asset"]:
        if col in corr.columns:
            c = corr.loc["Momentum", col]
            quality = "EXCELLENT" if abs(c) < 0.2 else "GOOD" if abs(c) < 0.4 else "HIGH"
            print(f"    vs {col}: {c:+.3f} ({quality} diversification)")

    # ═══ Combined Portfolios ═══
    print(f"\n[6] COMBINED PORTFOLIOS")
    print("-"*50)

    # Normalize all strategies to start at 100k
    strategies = {
        "Momentum": mom_daily,
        "Index Trend": trend_daily,
        "Mean Rev": mr_daily,
        "Cross-Asset": ca_daily,
    }

    # Test different combinations
    combos = [
        ("Momentum only", {"Momentum": 1.0}),
        ("70% Mom + 30% Trend", {"Momentum": 0.70, "Index Trend": 0.30}),
        ("70% Mom + 30% MeanRev", {"Momentum": 0.70, "Mean Rev": 0.30}),
        ("70% Mom + 30% CrossAsset", {"Momentum": 0.70, "Cross-Asset": 0.30}),
        ("60% Mom + 20% Trend + 20% MR", {"Momentum": 0.60, "Index Trend": 0.20, "Mean Rev": 0.20}),
        ("50% Mom + 25% Trend + 25% MR", {"Momentum": 0.50, "Index Trend": 0.25, "Mean Rev": 0.25}),
        ("60% Mom + 20% Trend + 10% MR + 10% CA", {"Momentum": 0.60, "Index Trend": 0.20, "Mean Rev": 0.10, "Cross-Asset": 0.10}),
    ]

    print(f"\n  {'Combo':<40} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"  {'-'*66}")

    for name, weights in combos:
        # Combine daily returns
        combined_monthly = pd.Series(0.0, index=all_monthly.index)
        for strat, w in weights.items():
            if strat in all_monthly.columns:
                combined_monthly += all_monthly[strat] * w

        # Reconstruct value series
        combined_values = (1 + combined_monthly).cumprod() * 100000
        m = compute_metrics(combined_values)
        if m:
            print(f"  {name:<40} {m['cagr']*100:>7.1f}% {m['sharpe']:>8.2f} {m['max_dd']*100:>7.1f}%")

    # ═══ Year-by-year for best combo vs baseline ═══
    print(f"\n[7] YEAR-BY-YEAR: Momentum only vs best combo")
    print(f"  {'Year':<6} {'Mom only':>10} {'Best combo':>10}")
    print(f"  {'-'*30}")

    for year in range(2018, 2026):
        mask = all_monthly.index.year == year
        if mask.sum() == 0:
            continue
        mom_yr = (1 + all_monthly.loc[mask, "Momentum"]).prod() - 1
        # Best combo: 60/20/20
        combo_yr = (1 + (all_monthly.loc[mask, "Momentum"] * 0.60 +
                          all_monthly.loc[mask, "Index Trend"] * 0.20 +
                          all_monthly.loc[mask, "Mean Rev"] * 0.20)).prod() - 1
        print(f"  {year:<6} {mom_yr*100:>+9.1f}% {combo_yr*100:>+9.1f}%")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
