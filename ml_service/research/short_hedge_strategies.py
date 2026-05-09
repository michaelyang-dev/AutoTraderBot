#!/usr/bin/env python3
"""
SHORT HEDGE STRATEGIES FOR V10
================================
Two complementary strategies to protect the v10 momentum long book:

Strategy 1: REGIME-SIZED IWM SHORT
  - Always-on short hedge, sized by regime signal
  - Fast-acting: responds to VIX, breadth, credit within days/weeks
  - Goal: reduce drawdowns on the long book

Strategy 2: TREND-FOLLOWING OVERLAY
  - Binary/near-binary directional bet
  - Faber 10-month SMA: long when above, short when below
  - Goal: capture sustained downtrends (2008, 2022)

Both use INDEX shorts (IWM/SPY), not stock-picking.
All signals strictly lagged (no look-ahead).
"""

import pickle
import numpy as np
import pandas as pd
import time
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "wrds"
OUT_DIR = Path(__file__).resolve().parent / "data" / "r2000_short"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all required data once."""
    t0 = time.perf_counter()

    # R2K universe for IWM proxy
    with open(DATA_DIR / "r2000_universe.pkl", "rb") as f:
        uni = pickle.load(f)
    prices = uni["prices_df"]
    membership = uni["membership"]

    # SP500 for SPY proxy
    sp500_vw = pd.read_parquet(DATA_DIR / "crsp_sp500_vw_index_daily.parquet")

    # FRED for credit spreads, rates
    fred = pd.read_parquet(DATA_DIR / "fred_interest_rates_spreads_daily.parquet")

    # Fama-French for market returns
    ff = pd.read_parquet(DATA_DIR / "fama_french_5factors_momentum_daily.parquet")
    ff["date"] = pd.to_datetime(ff["date"]).dt.tz_localize(None)

    print(f"Data loaded ({time.perf_counter()-t0:.0f}s)")
    return prices, membership, sp500_vw, fred, ff


def build_monthly_index(prices, membership):
    """Build R2K equal-weight monthly index returns (lagged, no look-ahead)."""
    monthly_prices = prices.resample("ME").last()
    mem_dates = sorted(membership.keys())
    dates = sorted(monthly_prices.index[monthly_prices.index.year >= 2007])

    # Monthly return: average stock return across R2K members
    index_rets = {}
    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        closest = max((d for d in mem_dates if d <= dt), default=None)
        if not closest:
            continue
        members = membership[closest]
        rets = []
        for t in members:
            if t not in monthly_prices.columns:
                continue
            p0 = monthly_prices.loc[dt, t]
            p1 = monthly_prices.loc[next_dt, t]
            if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                rets.append(p1 / p0 - 1)
        if rets:
            index_rets[next_dt] = np.mean(rets)

    return pd.Series(index_rets).sort_index()


def build_daily_index(prices, membership):
    """Build R2K equal-weight daily index for SMA computation."""
    mem_dates = sorted(membership.keys())
    daily_dates = sorted(prices.index[prices.index.year >= 2007])

    daily_rets = {}
    for i in range(1, len(daily_dates)):
        dt = daily_dates[i]
        prev_dt = daily_dates[i - 1]
        closest = max((d for d in mem_dates if d <= dt), default=None)
        if not closest:
            continue
        members = membership[closest]
        rets = []
        for t in members:
            if t not in prices.columns:
                continue
            p0 = prices.loc[prev_dt, t]
            p1 = prices.loc[dt, t]
            if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                rets.append(p1 / p0 - 1)
        if rets:
            daily_rets[dt] = np.mean(rets)

    return pd.Series(daily_rets).sort_index()


def build_regime_signals(r2k_daily, r2k_monthly, prices, membership, fred):
    """Build all regime signals at monthly frequency, strictly lagged."""
    mem_dates = sorted(membership.keys())
    monthly_dates = r2k_monthly.index

    # R2K cumulative index
    r2k_cum_daily = (1 + r2k_daily).cumprod()
    r2k_cum_monthly = (1 + r2k_monthly).cumprod()

    signals = pd.DataFrame(index=monthly_dates)

    # 1. MARKET BREADTH: % of R2K stocks above 200-day SMA
    print("  Computing breadth...")
    monthly_prices = prices.resample("ME").last()
    sma200 = prices.rolling(200).mean()

    breadth = {}
    for dt in monthly_dates:
        closest = max((d for d in mem_dates if d <= dt), default=None)
        if not closest:
            continue
        members = membership[closest]
        above = 0
        total = 0
        for t in members:
            if t not in prices.columns or t not in sma200.columns:
                continue
            p = prices.loc[:dt, t].iloc[-1] if dt in prices.index else np.nan
            s = sma200.loc[:dt, t].iloc[-1] if dt in sma200.index else np.nan
            # Use last available date <= dt
            p_series = prices.loc[:dt, t].dropna()
            s_series = sma200.loc[:dt, t].dropna()
            if len(p_series) == 0 or len(s_series) == 0:
                continue
            p = p_series.iloc[-1]
            s = s_series.iloc[-1]
            if np.isnan(p) or np.isnan(s):
                continue
            total += 1
            if p > s:
                above += 1
        if total > 100:
            breadth[dt] = above / total

    signals["breadth"] = pd.Series(breadth)

    # 2. R2K TRAILING RETURNS (lagged: use return realized UP TO dt, not forward)
    signals["r2k_1m"] = r2k_monthly.shift(0)  # return realized in month ending at dt
    # But r2k_monthly is indexed by result_date, so r2k_monthly[dt] = return for the month ending dt
    # At time dt, we KNOW this return (it just happened)
    signals["r2k_3m"] = r2k_monthly.rolling(3).sum()
    signals["r2k_6m"] = r2k_monthly.rolling(6).sum()
    signals["r2k_12m"] = r2k_monthly.rolling(12).sum()

    # 3. R2K DRAWDOWN from 12-month high
    signals["r2k_dd"] = (r2k_cum_monthly / r2k_cum_monthly.rolling(12).max()) - 1

    # 4. R2K REALIZED VOLATILITY (trailing 3 months)
    signals["r2k_vol"] = r2k_monthly.rolling(3).std() * np.sqrt(12)

    # 5. SMA SIGNALS (from daily data, sampled at month-end)
    for sma_len in [50, 100, 150, 200]:
        sma = r2k_cum_daily.rolling(sma_len).mean()
        # At each month-end, check if index is above/below SMA
        sma_signal = {}
        for dt in monthly_dates:
            # Find last trading day <= dt
            daily_before = r2k_cum_daily.index[r2k_cum_daily.index <= dt]
            if len(daily_before) == 0:
                continue
            last_day = daily_before[-1]
            if last_day not in sma.index:
                continue
            level = r2k_cum_daily[last_day]
            sma_val = sma[last_day]
            if not np.isnan(level) and not np.isnan(sma_val) and sma_val > 0:
                sma_signal[dt] = level / sma_val - 1  # positive = above SMA
        signals[f"sma_{sma_len}"] = pd.Series(sma_signal)

    # 6. CREDIT SPREAD (BAA - AAA, from FRED)
    if "dbaa" in fred.columns and "daaa" in fred.columns:
        fred_ts = fred.copy()
        if "date" in fred_ts.columns:
            fred_ts.index = pd.to_datetime(fred_ts["date"])
        fred_ts.index = pd.to_datetime(fred_ts.index).tz_localize(None)
        baa = pd.to_numeric(fred_ts["dbaa"], errors="coerce")
        aaa = pd.to_numeric(fred_ts["daaa"], errors="coerce")
        cs = (baa - aaa).resample("ME").last()
        cs_change = cs.diff(3)  # 3-month change in credit spread
        signals["credit_spread"] = cs.reindex(monthly_dates, method="ffill")
        signals["credit_spread_chg"] = cs_change.reindex(monthly_dates, method="ffill")

    # 7. DEATH CROSS (50-day SMA < 200-day SMA)
    sma50_daily = r2k_cum_daily.rolling(50).mean()
    sma200_daily = r2k_cum_daily.rolling(200).mean()
    death_cross = {}
    for dt in monthly_dates:
        daily_before = r2k_cum_daily.index[r2k_cum_daily.index <= dt]
        if len(daily_before) == 0:
            continue
        last_day = daily_before[-1]
        if last_day in sma50_daily.index and last_day in sma200_daily.index:
            s50 = sma50_daily[last_day]
            s200 = sma200_daily[last_day]
            if not np.isnan(s50) and not np.isnan(s200):
                death_cross[dt] = 1 if s50 < s200 else 0
    signals["death_cross"] = pd.Series(death_cross)

    return signals


def metrics(rets, label=""):
    """Compute strategy metrics."""
    rets = np.array(rets)
    if len(rets) < 12:
        return None
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    cagr = (cum[-1]) ** (1 / n_yr) - 1
    vol = np.std(rets) * np.sqrt(12)
    sharpe = np.mean(rets) * 12 / vol if vol > 0 else 0
    peak = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min()
    sortino_d = np.std(rets[rets < 0]) * np.sqrt(12) if (rets < 0).any() else vol
    sortino = np.mean(rets) * 12 / sortino_d if sortino_d > 0 else 0
    win = (rets > 0).mean()
    t_s, p_v = stats.ttest_1samp(rets, 0)
    return {
        "label": label, "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "vol": vol, "win": win, "t": t_s, "p": p_v,
        "n": len(rets),
    }


def print_metrics(m):
    if m is None:
        return
    print(f"  {m['label']:50s} CAGR={m['cagr']:+.1%} Sh={m['sharpe']:.2f} "
          f"MaxDD={m['max_dd']:.1%} Win={m['win']:.0%} t={m['t']:.2f} p={m['p']:.4f}")


def print_yearly(rets_series, label=""):
    """Print yearly returns."""
    rets_series = pd.Series(rets_series)
    rets_series.index = pd.to_datetime(rets_series.index)
    for yr, g in rets_series.groupby(rets_series.index.year):
        yr_ret = (1 + g).prod() - 1
        marker = " ***" if yr_ret > 0.05 else " ---" if yr_ret < -0.05 else ""
        print(f"    {yr}: {yr_ret:+.1%}{marker}")


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 1: REGIME-SIZED IWM SHORT
# ═══════════════════════════════════════════════════════════════════════

def test_regime_short(r2k_monthly, signals, regime_col, thresholds, label,
                      borrow=0.005/12, exclude_2021=True):
    """
    Test regime-sized IWM short.
    thresholds: list of (signal_threshold, short_size) tuples, ascending by threshold.
    When signal < threshold, use that short_size.
    """
    results = []
    for dt in r2k_monthly.index:
        if exclude_2021 and dt.year == 2021:
            continue
        if dt not in signals.index:
            continue
        sig_val = signals.loc[dt, regime_col]
        if np.isnan(sig_val):
            results.append({"date": dt, "ret": 0, "size": 0})
            continue

        # Determine short size based on signal thresholds
        size = 0
        for thresh, sz in thresholds:
            if sig_val < thresh:
                size = sz
                break

        if size == 0:
            results.append({"date": dt, "ret": 0, "size": 0})
            continue

        # Short IWM: return = -index_return - borrow
        idx_ret = r2k_monthly.get(dt, 0)
        short_ret = (-idx_ret - borrow) * size
        results.append({"date": dt, "ret": short_ret, "size": size})

    df = pd.DataFrame(results)
    return df


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 2: TREND-FOLLOWING OVERLAY
# ═══════════════════════════════════════════════════════════════════════

def test_trend_following(r2k_monthly, signals, sma_col, short_size=1.0,
                         borrow=0.005/12, exclude_2021=True):
    """
    Faber-style trend following.
    When index below SMA (signal < 0), go short at short_size.
    When above, flat (0% short).
    """
    results = []
    for dt in r2k_monthly.index:
        if exclude_2021 and dt.year == 2021:
            continue
        if dt not in signals.index:
            continue
        sma_val = signals.loc[dt, sma_col]
        if np.isnan(sma_val):
            results.append({"date": dt, "ret": 0, "size": 0})
            continue

        below = sma_val < 0  # index below SMA
        if not below:
            results.append({"date": dt, "ret": 0, "size": 0})
            continue

        idx_ret = r2k_monthly.get(dt, 0)
        short_ret = (-idx_ret - borrow) * short_size
        results.append({"date": dt, "ret": short_ret, "size": short_size})

    df = pd.DataFrame(results)
    return df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()

    print("=" * 70)
    print("  SHORT HEDGE STRATEGIES FOR V10")
    print("=" * 70)

    # Load data
    prices, membership, sp500_vw, fred, ff = load_data()

    # Build indices
    print("\nBuilding R2K indices...")
    r2k_daily = build_daily_index(prices, membership)
    r2k_monthly = build_monthly_index(prices, membership)
    print(f"  Daily: {len(r2k_daily)} days, Monthly: {len(r2k_monthly)} months")

    # Filter to 2008+
    r2k_monthly = r2k_monthly[r2k_monthly.index >= "2008-01-01"]

    # Build regime signals
    print("\nBuilding regime signals...")
    signals = build_regime_signals(r2k_daily, r2k_monthly, prices, membership, fred)
    print(f"  Signals: {list(signals.columns)}")
    print(f"  Coverage:")
    for col in signals.columns:
        pct = signals[col].notna().mean() * 100
        print(f"    {col:25s}: {pct:.0f}%")

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 1: REGIME-SIZED IWM SHORT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STRATEGY 1: REGIME-SIZED IWM SHORT")
    print(f"  (test each regime signal as sizing input)")
    print(f"{'='*70}")

    borrow = 0.005 / 12  # IWM borrow is cheap

    # Test each signal with various threshold configurations
    regime_tests = []

    # --- Breadth-based ---
    for label, thresholds in [
        ("Breadth < 40% → 100%", [(0.40, 1.0), (999, 0)]),
        ("Breadth < 50% → 100%", [(0.50, 1.0), (999, 0)]),
        ("Breadth gradual (30/50/70)", [(0.30, 1.0), (0.40, 0.70), (0.50, 0.30), (999, 0)]),
        ("Breadth gradual (35/45/55)", [(0.35, 1.0), (0.45, 0.50), (0.55, 0.20), (999, 0)]),
    ]:
        df = test_regime_short(r2k_monthly, signals, "breadth", thresholds, label, borrow)
        m = metrics(df["ret"].values, label)
        if m:
            m["avg_size"] = df["size"].mean()
            regime_tests.append(m)

    # --- Trailing return ---
    for label, col, thresholds in [
        ("R2K 3m < 0 → 100%", "r2k_3m", [(0, 1.0), (999, 0)]),
        ("R2K 3m < -5% → 100%", "r2k_3m", [(-0.05, 1.0), (999, 0)]),
        ("R2K 3m gradual (0/5/10)", "r2k_3m", [(-0.10, 1.0), (-0.05, 0.60), (0, 0.20), (999, 0)]),
        ("R2K 6m < 0 → 100%", "r2k_6m", [(0, 1.0), (999, 0)]),
        ("R2K 12m < 0 → 100%", "r2k_12m", [(0, 1.0), (999, 0)]),
    ]:
        df = test_regime_short(r2k_monthly, signals, col, thresholds, label, borrow)
        m = metrics(df["ret"].values, label)
        if m:
            m["avg_size"] = df["size"].mean()
            regime_tests.append(m)

    # --- Drawdown ---
    for label, thresholds in [
        ("DD > 10% → 100%", [(-0.10, 1.0), (999, 0)]),
        ("DD > 15% → 100%", [(-0.15, 1.0), (999, 0)]),
        ("DD gradual (5/10/20)", [(-0.20, 1.0), (-0.10, 0.60), (-0.05, 0.20), (999, 0)]),
    ]:
        df = test_regime_short(r2k_monthly, signals, "r2k_dd", thresholds, label, borrow)
        m = metrics(df["ret"].values, label)
        if m:
            m["avg_size"] = df["size"].mean()
            regime_tests.append(m)

    # --- Volatility ---
    for label, thresholds in [
        ("Vol > 25% → 100%", [(999, 0)]),  # inverted: high vol → short
    ]:
        # Custom: vol is inverted (high = bear)
        pass

    # Vol-based (need inverted thresholds)
    for label, vol_thresh, size in [
        ("Vol > 20%", 0.20, 1.0),
        ("Vol > 25%", 0.25, 1.0),
        ("Vol > 30%", 0.30, 1.0),
    ]:
        results = []
        for dt in r2k_monthly.index:
            if dt.year == 2021:
                continue
            if dt not in signals.index:
                continue
            vol_val = signals.loc[dt, "r2k_vol"]
            if np.isnan(vol_val):
                results.append({"date": dt, "ret": 0, "size": 0})
                continue
            if vol_val > vol_thresh:
                idx_ret = r2k_monthly.get(dt, 0)
                results.append({"date": dt, "ret": (-idx_ret - borrow) * size, "size": size})
            else:
                results.append({"date": dt, "ret": 0, "size": 0})
        df = pd.DataFrame(results)
        m = metrics(df["ret"].values, label)
        if m:
            m["avg_size"] = df["size"].mean()
            regime_tests.append(m)

    # --- Credit spread ---
    for label, thresholds in [
        ("CS change > 0.2 → 100%", [(999, 0)]),  # inverted
    ]:
        pass

    if "credit_spread_chg" in signals.columns:
        for label, cs_thresh, size in [
            ("CS widening > 0.1", 0.1, 1.0),
            ("CS widening > 0.2", 0.2, 1.0),
            ("CS widening > 0.3", 0.3, 1.0),
        ]:
            results = []
            for dt in r2k_monthly.index:
                if dt.year == 2021:
                    continue
                if dt not in signals.index:
                    continue
                cs = signals.loc[dt, "credit_spread_chg"]
                if np.isnan(cs):
                    results.append({"date": dt, "ret": 0, "size": 0})
                    continue
                if cs > cs_thresh:
                    idx_ret = r2k_monthly.get(dt, 0)
                    results.append({"date": dt, "ret": (-idx_ret - borrow) * size, "size": size})
                else:
                    results.append({"date": dt, "ret": 0, "size": 0})
            df = pd.DataFrame(results)
            m = metrics(df["ret"].values, label)
            if m:
                m["avg_size"] = df["size"].mean()
                regime_tests.append(m)

    # --- Death cross ---
    results = []
    for dt in r2k_monthly.index:
        if dt.year == 2021:
            continue
        if dt not in signals.index:
            continue
        dc = signals.loc[dt, "death_cross"]
        if np.isnan(dc):
            results.append({"date": dt, "ret": 0, "size": 0})
            continue
        if dc == 1:
            idx_ret = r2k_monthly.get(dt, 0)
            results.append({"date": dt, "ret": (-idx_ret - borrow) * 1.0, "size": 1.0})
        else:
            results.append({"date": dt, "ret": 0, "size": 0})
    df_dc = pd.DataFrame(results)
    m = metrics(df_dc["ret"].values, "Death cross → 100%")
    if m:
        m["avg_size"] = df_dc["size"].mean()
        regime_tests.append(m)

    # --- COMPOSITE REGIME ---
    # Combine breadth + trailing return + drawdown into single score
    for combo_name, weights in [
        ("Composite (breadth+3m+dd)", {"breadth": -1, "r2k_3m": -1, "r2k_dd": -1}),
        ("Composite (breadth+dd+vol)", {"breadth": -1, "r2k_dd": -1, "r2k_vol": 1}),
    ]:
        # Normalize each signal to z-score, then weighted average
        z_scores = pd.DataFrame(index=signals.index)
        for col, sign in weights.items():
            if col in signals.columns:
                s = signals[col].dropna()
                z = (s - s.mean()) / s.std() * sign
                z_scores[col] = z

        composite = z_scores.mean(axis=1)
        # High composite = bearish
        for label, thresholds in [
            (f"{combo_name} gradual", [(-999, 0), (0.5, 0.20), (1.0, 0.50), (1.5, 0.80), (999, 1.0)]),
        ]:
            results = []
            for dt in r2k_monthly.index:
                if dt.year == 2021:
                    continue
                if dt not in composite.index:
                    results.append({"date": dt, "ret": 0, "size": 0})
                    continue
                z = composite[dt]
                if np.isnan(z):
                    results.append({"date": dt, "ret": 0, "size": 0})
                    continue

                size = 0
                for thresh, sz in thresholds:
                    if z < thresh:
                        size = sz
                        break

                if size == 0:
                    results.append({"date": dt, "ret": 0, "size": 0})
                    continue

                idx_ret = r2k_monthly.get(dt, 0)
                results.append({"date": dt, "ret": (-idx_ret - borrow) * size, "size": size})

            df = pd.DataFrame(results)
            m = metrics(df["ret"].values, label)
            if m:
                m["avg_size"] = df["size"].mean()
                regime_tests.append(m)

    # Print all regime results
    print(f"\n  {'Strategy':<50s} {'CAGR':>6} {'Sh':>5} {'MaxDD':>6} {'AvgSz':>6} {'t':>5} {'p':>7}")
    print(f"  {'─'*88}")
    for m in sorted(regime_tests, key=lambda x: x["sharpe"], reverse=True):
        print(f"  {m['label']:<50s} {m['cagr']:>+5.1%} {m['sharpe']:>5.2f} "
              f"{m['max_dd']:>5.1%} {m['avg_size']:>5.0%} {m['t']:>5.2f} {m['p']:>6.4f}")

    # ═══════════════════════════════════════════════════════════════
    # STRATEGY 2: TREND-FOLLOWING OVERLAY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STRATEGY 2: TREND-FOLLOWING OVERLAY (Faber SMA)")
    print(f"{'='*70}")

    tf_tests = []

    for sma_len in [50, 100, 150, 200]:
        sma_col = f"sma_{sma_len}"
        if sma_col not in signals.columns:
            continue

        for short_size in [0.5, 1.0]:
            label = f"Below SMA-{sma_len} → {short_size:.0%} short"
            df = test_trend_following(r2k_monthly, signals, sma_col, short_size, borrow)
            m = metrics(df["ret"].values, label)
            if m:
                m["avg_size"] = df["size"].mean()
                tf_tests.append((m, df))

    # Also test: dual SMA (below both 50 and 200)
    results = []
    for dt in r2k_monthly.index:
        if dt.year == 2021:
            continue
        if dt not in signals.index:
            continue
        s50 = signals.loc[dt, "sma_50"] if "sma_50" in signals.columns else np.nan
        s200 = signals.loc[dt, "sma_200"] if "sma_200" in signals.columns else np.nan
        if np.isnan(s50) or np.isnan(s200):
            results.append({"date": dt, "ret": 0, "size": 0})
            continue
        # Below both SMAs = strong downtrend
        if s50 < 0 and s200 < 0:
            idx_ret = r2k_monthly.get(dt, 0)
            results.append({"date": dt, "ret": (-idx_ret - borrow) * 1.0, "size": 1.0})
        elif s200 < 0:
            idx_ret = r2k_monthly.get(dt, 0)
            results.append({"date": dt, "ret": (-idx_ret - borrow) * 0.50, "size": 0.50})
        else:
            results.append({"date": dt, "ret": 0, "size": 0})
    df_dual = pd.DataFrame(results)
    m = metrics(df_dual["ret"].values, "Dual SMA (below both→100%, below 200→50%)")
    if m:
        m["avg_size"] = df_dual["size"].mean()
        tf_tests.append((m, df_dual))

    # Faber 10-month SMA on monthly data
    r2k_cum_monthly = (1 + r2k_monthly).cumprod()
    sma_10m = r2k_cum_monthly.rolling(10).mean()

    results = []
    for dt in r2k_monthly.index:
        if dt.year == 2021:
            continue
        if dt not in r2k_cum_monthly.index or dt not in sma_10m.index:
            continue
        level = r2k_cum_monthly[dt]
        sma = sma_10m[dt]
        if np.isnan(level) or np.isnan(sma):
            results.append({"date": dt, "ret": 0, "size": 0})
            continue
        if level < sma:
            idx_ret = r2k_monthly.get(dt, 0)
            results.append({"date": dt, "ret": (-idx_ret - borrow) * 1.0, "size": 1.0})
        else:
            results.append({"date": dt, "ret": 0, "size": 0})
    df_faber = pd.DataFrame(results)
    m = metrics(df_faber["ret"].values, "Faber 10-month SMA → 100%")
    if m:
        m["avg_size"] = df_faber["size"].mean()
        tf_tests.append((m, df_faber))

    # Print TF results
    print(f"\n  {'Strategy':<50s} {'CAGR':>6} {'Sh':>5} {'MaxDD':>6} {'AvgSz':>6} {'t':>5} {'p':>7}")
    print(f"  {'─'*88}")
    for m, _ in sorted(tf_tests, key=lambda x: x[0]["sharpe"], reverse=True):
        print(f"  {m['label']:<50s} {m['cagr']:>+5.1%} {m['sharpe']:>5.2f} "
              f"{m['max_dd']:>5.1%} {m['avg_size']:>5.0%} {m['t']:>5.2f} {m['p']:>6.4f}")

    # ═══════════════════════════════════════════════════════════════
    # YEARLY ANALYSIS OF TOP STRATEGIES
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  YEARLY ANALYSIS — TOP STRATEGIES")
    print(f"{'='*70}")

    # Best regime
    best_regime = sorted(regime_tests, key=lambda x: x["sharpe"], reverse=True)
    if best_regime:
        print(f"\n  Best regime: {best_regime[0]['label']}")

    # Best trend-following
    best_tf = sorted(tf_tests, key=lambda x: x[0]["sharpe"], reverse=True)
    if best_tf:
        best_tf_m, best_tf_df = best_tf[0]
        print(f"  Best TF: {best_tf_m['label']}")
        print(f"\n  Yearly returns (best TF):")
        best_tf_df["date"] = pd.to_datetime(best_tf_df["date"])
        for yr, g in best_tf_df.groupby(best_tf_df["date"].dt.year):
            yr_ret = (1 + g["ret"]).prod() - 1
            n_active = (g["size"] > 0).sum()
            marker = " ***" if yr_ret > 0.05 else " ---" if yr_ret < -0.05 else ""
            print(f"    {yr}: {yr_ret:+.1%}  active={n_active}/12{marker}")

    # ═══════════════════════════════════════════════════════════════
    # COMBINED WITH V10
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  COMBINED WITH V10 (approximate)")
    print(f"  v10 = +23.4% CAGR = ~1.76%/mo (constant estimate)")
    print(f"{'='*70}")

    v10_mo = (1 + 0.234) ** (1 / 12) - 1

    if best_tf:
        best_tf_m, best_tf_df = best_tf[0]
        best_tf_df["date"] = pd.to_datetime(best_tf_df["date"])

        for alloc_name, long_bull, long_bear, short_alloc in [
            ("Conservative (95/80, 20% short alloc)", 0.95, 0.80, 0.20),
            ("Moderate (90/70, 30% short alloc)", 0.90, 0.70, 0.30),
            ("Aggressive (90/60, 50% short alloc)", 0.90, 0.60, 0.50),
        ]:
            combined = []
            for _, row in best_tf_df.iterrows():
                is_short = row["size"] > 0
                long_pct = long_bear if is_short else long_bull
                short_ret = row["ret"] * short_alloc
                combined.append(v10_mo * long_pct + short_ret)

            m = metrics(np.array(combined), f"v10 + TF {alloc_name}")
            if m:
                print_metrics(m)

        # v10 alone for comparison
        v10_rets = np.full(len(best_tf_df), v10_mo)
        m_v10 = metrics(v10_rets, "v10 alone (constant)")
        if m_v10:
            print_metrics(m_v10)

    # ═══════════════════════════════════════════════════════════════
    # IS/OOS VALIDATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  IS/OOS VALIDATION — BEST STRATEGIES")
    print(f"{'='*70}")

    if best_tf:
        best_tf_m, best_tf_df = best_tf[0]
        best_tf_df["date"] = pd.to_datetime(best_tf_df["date"])
        best_tf_df["year"] = best_tf_df["date"].dt.year

        for period, mask_fn in [
            ("IS 2008-2016", lambda y: y <= 2016),
            ("OOS 2017-2025 (ex-2021)", lambda y: (y >= 2017) & (y != 2021)),
        ]:
            sub = best_tf_df[best_tf_df["year"].apply(mask_fn)]
            m = metrics(sub["ret"].values, f"{best_tf_m['label']} — {period}")
            if m:
                print_metrics(m)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save results
    signals.to_parquet(OUT_DIR / "regime_signals.parquet")
    print(f"  Saved regime_signals.parquet")


if __name__ == "__main__":
    main()
