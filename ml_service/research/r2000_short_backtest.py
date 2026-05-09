#!/usr/bin/env python3
"""
Russell 2000 Short Strategy — Composite Factor Backtest
========================================================
Uses TRUE R2000 membership (Norgate via r2000_universe.pkl) and 6 WRDS signals:
  1. Short interest % float (Compustat)
  2. Accrual anomaly (Compustat quarterly NI-OCF/Assets)
  3. Negative earnings momentum (I/B/E/S SUE < -2)
  4. Negative price momentum (12-1 month, from CRSP prices)
  5. Audit red flags (restatements + SOX404 weakness)
  6. CEO/CFO departures (Audit Analytics severity scores)

Backtest: Monthly rebalance, short top-N, equal-weight, 2008-2025.
"""

import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "wrds"
OUT_DIR = Path(__file__).resolve().parent / "data" / "r2000_short"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD UNIVERSE
# ═══════════════════════════════════════════════════════════════════════

def load_universe():
    """Load R2000 membership and prices from universe pickle."""
    print("[LOAD] Loading R2000 universe...")
    with open(DATA_DIR / "r2000_universe.pkl", "rb") as f:
        uni = pickle.load(f)

    prices = uni["prices_df"]  # DataFrame: dates × tickers, adj close
    membership = uni["membership"]  # dict: date → set of tickers

    print(f"  Prices: {prices.shape[0]} dates × {prices.shape[1]} tickers")
    print(f"  Membership: {len(membership)} dates, {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices, membership


# ═══════════════════════════════════════════════════════════════════════
# 2. LOAD & PREP SIGNAL DATA
# ═══════════════════════════════════════════════════════════════════════

def clean_compustat_ticker(tic):
    """Strip Compustat suffixes: 'ABSI.1' → 'ABSI', '4303B' stays."""
    if pd.isna(tic):
        return None
    t = str(tic).split(".")[0]
    # Remove trailing digits that are Compustat issue identifiers
    # But keep tickers like '3COM' — only strip if suffix after dot
    return t if t else None


def load_short_interest():
    """Load Compustat short interest. Returns: date, ticker, shortint."""
    print("[SI] Loading short interest...")
    si = pd.read_parquet(DATA_DIR / "compustat_short_interest.parquet")
    si["ticker"] = si["tic"].apply(clean_compustat_ticker)
    si["date"] = pd.to_datetime(si["datadate"])
    si = si.dropna(subset=["ticker", "shortint"])
    si = si[si["shortint"] > 0]
    print(f"  {len(si):,} rows, {si['ticker'].nunique()} tickers")
    return si[["date", "ticker", "shortint", "shortintadj"]].copy()


def load_accruals():
    """Load computed accruals (NI-OCF)/Assets. High = aggressive accounting."""
    print("[ACC] Loading accruals...")
    acc = pd.read_parquet(DATA_DIR / "computed_accruals.parquet")
    acc["ticker"] = acc["tic"].apply(clean_compustat_ticker)
    acc["date"] = pd.to_datetime(acc["datadate"])
    acc = acc.dropna(subset=["ticker", "accruals"])
    print(f"  {len(acc):,} rows, {acc['ticker'].nunique()} tickers")
    return acc[["date", "ticker", "accruals"]].copy()


def load_ibes_surprise():
    """Load I/B/E/S SUE scores. Negative SUE = negative earnings surprise."""
    print("[IBES] Loading earnings surprise (SUE)...")
    ibes = pd.read_parquet(DATA_DIR / "ibes_surprise.parquet")
    # Filter to EPS measure only
    ibes = ibes[ibes["MEASURE"] == "EPS"].copy()
    ibes["ticker"] = ibes["OFTIC"].str.strip()
    ibes["date"] = pd.to_datetime(ibes["anndats"])
    ibes = ibes.dropna(subset=["ticker", "suescore"])
    # Keep the most recent SUE per ticker-date
    ibes = ibes.sort_values("date").drop_duplicates(subset=["ticker", "date"], keep="last")
    print(f"  {len(ibes):,} rows, {ibes['ticker'].nunique()} tickers")
    return ibes[["date", "ticker", "suescore"]].copy()


def load_audit_restatements():
    """Load audit restatements. Presence = red flag."""
    print("[AUD] Loading restatements...")
    rest = pd.read_parquet(
        DATA_DIR / "audit_restatements.parquet",
        columns=["company_fkey", "file_date", "best_edgar_ticker"],
    )
    rest["ticker"] = rest["best_edgar_ticker"].str.strip()
    rest["date"] = pd.to_datetime(rest["file_date"])
    rest = rest.dropna(subset=["ticker", "date"])
    # Deduplicate: one flag per ticker per date
    rest = rest.drop_duplicates(subset=["ticker", "date"])
    rest["restatement_flag"] = 1
    print(f"  {len(rest):,} rows, {rest['ticker'].nunique()} tickers")
    return rest[["date", "ticker", "restatement_flag"]].copy()


def load_sox404():
    """Load SOX404 internal control weaknesses (res_adverse = adverse restatement)."""
    print("[SOX] Loading SOX404 internal controls...")
    sox = pd.read_parquet(
        DATA_DIR / "audit_sox404_internal_controls.parquet",
        columns=["company_fkey", "file_date", "best_edgar_ticker",
                 "res_adverse", "res_fraud"],
    )
    sox["ticker"] = sox["best_edgar_ticker"].str.strip()
    sox["date"] = pd.to_datetime(sox["file_date"])
    sox = sox.dropna(subset=["ticker", "date"])
    # Material weakness = adverse restatement or fraud
    sox["sox_weakness"] = ((sox["res_adverse"].fillna(0) == 1) |
                           (sox["res_fraud"].fillna(0) == 1)).astype(int)
    sox = sox[sox["sox_weakness"] == 1]
    sox = sox.drop_duplicates(subset=["ticker", "date"])
    print(f"  {len(sox):,} rows with weakness, {sox['ticker'].nunique()} tickers")
    return sox[["date", "ticker", "sox_weakness"]].copy()


def load_officer_changes():
    """Load CEO/CFO departures with severity scores."""
    print("[OFF] Loading officer changes...")
    off = pd.read_parquet(
        DATA_DIR / "audit_director_officer_changes.parquet",
        columns=["company_fkey", "opinion_file_date", "best_edgar_ticker",
                 "ceo_change_severity", "cfo_change_severity"],
    )
    off["ticker"] = off["best_edgar_ticker"].str.strip()
    off["date"] = pd.to_datetime(off["opinion_file_date"])
    off = off.dropna(subset=["ticker", "date"])

    # Compute max severity across CEO/CFO
    ceo = off["ceo_change_severity"].fillna(0)
    cfo = off["cfo_change_severity"].fillna(0)
    off["officer_severity"] = np.maximum(ceo, cfo)
    off = off[off["officer_severity"] > 0]
    off = off.drop_duplicates(subset=["ticker", "date"])
    print(f"  {len(off):,} rows, {off['ticker'].nunique()} tickers")
    return off[["date", "ticker", "officer_severity"]].copy()


# ═══════════════════════════════════════════════════════════════════════
# 3. BUILD MONTHLY SIGNAL PANELS
# ═══════════════════════════════════════════════════════════════════════

def compute_momentum_12_1(prices, membership):
    """Compute 12-1 month momentum for R2000 members. Returns monthly panel."""
    print("[MOM] Computing 12-1 month price momentum...")

    # Resample prices to month-end
    monthly = prices.resample("ME").last()

    # 12-month return skipping most recent month (classic momentum)
    ret_12m = monthly.shift(1) / monthly.shift(12) - 1  # skip month t, use t-12 to t-1

    records = []
    for dt in ret_12m.index:
        if dt.year < 2007:  # need 12 months lookback
            continue
        # Get R2000 members on this date (find closest membership date)
        mem_dates = sorted(membership.keys())
        closest = max((d for d in mem_dates if d <= dt), default=None)
        if closest is None:
            continue
        members = membership[closest]

        for ticker in members:
            if ticker in ret_12m.columns:
                val = ret_12m.loc[dt, ticker]
                if not np.isnan(val):
                    records.append({"date": dt, "ticker": ticker, "mom_12_1": val})

    df = pd.DataFrame(records)
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers")
    return df


def as_of_signal(signal_df, date_col, ticker_col, value_col, target_dates,
                 lookback_days=180):
    """Map a point-in-time signal to monthly target dates.

    For each (target_date, ticker), find the most recent signal value
    within lookback_days before target_date.
    """
    signal_df = signal_df.copy()
    signal_df["_date"] = pd.to_datetime(signal_df[date_col])
    signal_df["_ticker"] = signal_df[ticker_col]
    signal_df["_value"] = signal_df[value_col]
    signal_df = signal_df.sort_values("_date")

    records = []
    for tgt_date in target_dates:
        cutoff = tgt_date - pd.Timedelta(days=lookback_days)
        mask = (signal_df["_date"] >= cutoff) & (signal_df["_date"] <= tgt_date)
        window = signal_df[mask]
        if window.empty:
            continue
        # Take the most recent value per ticker
        latest = window.drop_duplicates(subset=["_ticker"], keep="last")
        for _, row in latest.iterrows():
            records.append({
                "date": tgt_date,
                "ticker": row["_ticker"],
                value_col: row["_value"],
            })
    return pd.DataFrame(records)


def as_of_binary_signal(signal_df, date_col, ticker_col, flag_col, target_dates,
                        lookback_days=365):
    """Map binary event signal: 1 if event occurred within lookback, 0 otherwise."""
    signal_df = signal_df.copy()
    signal_df["_date"] = pd.to_datetime(signal_df[date_col])
    signal_df["_ticker"] = signal_df[ticker_col]

    records = []
    for tgt_date in target_dates:
        cutoff = tgt_date - pd.Timedelta(days=lookback_days)
        mask = (signal_df["_date"] >= cutoff) & (signal_df["_date"] <= tgt_date)
        tickers_with_event = signal_df.loc[mask, "_ticker"].unique()
        for ticker in tickers_with_event:
            records.append({
                "date": tgt_date,
                "ticker": ticker,
                flag_col: 1,
            })
    return pd.DataFrame(records)


def build_monthly_panel(prices, membership, si_df, acc_df, ibes_df,
                        rest_df, sox_df, off_df, mom_df):
    """Build a monthly panel with all signals for R2000 members."""
    print("\n[PANEL] Building monthly signal panel...")

    # Monthly rebalance dates (month-end)
    monthly_prices = prices.resample("ME").last()
    target_dates = sorted(monthly_prices.index[monthly_prices.index.year >= 2008])
    print(f"  Target dates: {len(target_dates)} months ({target_dates[0].date()} to {target_dates[-1].date()})")

    # Build universe panel: all R2000 members on each month-end
    mem_dates = sorted(membership.keys())
    universe_records = []
    for dt in target_dates:
        closest = max((d for d in mem_dates if d <= dt), default=None)
        if closest is None:
            continue
        members = membership[closest]
        for ticker in members:
            if ticker in monthly_prices.columns:
                price = monthly_prices.loc[dt, ticker]
                if not np.isnan(price) and price > 1.0:  # filter penny stocks
                    universe_records.append({"date": dt, "ticker": ticker, "price": price})

    panel = pd.DataFrame(universe_records)
    print(f"  Universe panel: {len(panel):,} ticker-months, {panel['ticker'].nunique()} unique tickers")

    # --- Merge signals ---
    # 1. Short interest (as-of, 90-day lookback since reported bi-monthly)
    print("  Merging short interest...")
    si_asof = as_of_signal(si_df, "date", "ticker", "shortintadj", target_dates, lookback_days=90)
    panel = panel.merge(si_asof, on=["date", "ticker"], how="left")

    # 2. Accruals (as-of, 180-day lookback for quarterly data)
    print("  Merging accruals...")
    acc_asof = as_of_signal(acc_df, "date", "ticker", "accruals", target_dates, lookback_days=180)
    panel = panel.merge(acc_asof, on=["date", "ticker"], how="left")

    # 3. SUE score (as-of, 120-day lookback)
    print("  Merging earnings surprise (SUE)...")
    sue_asof = as_of_signal(ibes_df, "date", "ticker", "suescore", target_dates, lookback_days=120)
    panel = panel.merge(sue_asof, on=["date", "ticker"], how="left")

    # 4. Momentum (already monthly)
    print("  Merging momentum...")
    panel = panel.merge(mom_df, on=["date", "ticker"], how="left")

    # 5. Restatements (binary, 1-year lookback)
    print("  Merging restatement flags...")
    rest_asof = as_of_binary_signal(rest_df, "date", "ticker", "restatement_flag", target_dates, lookback_days=365)
    panel = panel.merge(rest_asof, on=["date", "ticker"], how="left")
    panel["restatement_flag"] = panel["restatement_flag"].fillna(0)

    # 6. SOX404 weakness (binary, 1-year lookback)
    print("  Merging SOX404 weakness...")
    sox_asof = as_of_binary_signal(sox_df, "date", "ticker", "sox_weakness", target_dates, lookback_days=365)
    panel = panel.merge(sox_asof, on=["date", "ticker"], how="left")
    panel["sox_weakness"] = panel["sox_weakness"].fillna(0)

    # 7. Officer departures (as-of severity, 1-year lookback)
    print("  Merging officer changes...")
    off_asof = as_of_signal(off_df, "date", "ticker", "officer_severity", target_dates, lookback_days=365)
    panel = panel.merge(off_asof, on=["date", "ticker"], how="left")

    # Combine audit flags into single audit_red_flag score
    panel["audit_score"] = (
        panel["restatement_flag"] * 2  # restatement is worse
        + panel["sox_weakness"] * 2
        + panel["officer_severity"].fillna(0)
    )

    # Coverage report
    total = len(panel)
    print(f"\n  Signal coverage (% non-null):")
    for col in ["shortintadj", "accruals", "suescore", "mom_12_1", "audit_score"]:
        if col in panel.columns:
            pct = panel[col].notna().mean() * 100
            print(f"    {col:20s}: {pct:5.1f}%")

    return panel


# ═══════════════════════════════════════════════════════════════════════
# 4. COMPOSITE SHORT SCORE
# ═══════════════════════════════════════════════════════════════════════

def compute_composite_score(panel):
    """Rank each signal cross-sectionally and combine into composite short score.

    Higher composite = stronger short candidate.
    Signal directions:
      - shortintadj: HIGH = crowded short (contrarian: actually BAD to short)
        → we want HIGH short interest as a confirming signal, not primary
        → use moderate weight
      - accruals: HIGH = aggressive accounting = short signal
      - suescore: LOW (negative) = bad earnings = short signal → rank descending
      - mom_12_1: LOW = negative momentum = short signal → rank descending
      - audit_score: HIGH = more red flags = short signal
    """
    print("\n[SCORE] Computing composite short scores...")

    scored = panel.copy()

    # Cross-sectional percentile ranks per month (0 = best, 1 = worst)
    # For short signals, 1 = strongest short candidate

    def rank_ascending(group, col):
        """Rank: highest value → rank near 1.0 (strong short)."""
        return group[col].rank(pct=True, na_option="keep")

    def rank_descending(group, col):
        """Rank: lowest value → rank near 1.0 (strong short)."""
        return (1 - group[col].rank(pct=True, na_option="keep"))

    grouped = scored.groupby("date")

    # Short interest: HIGH SI → stronger short signal
    scored["si_rank"] = grouped.apply(
        lambda g: rank_ascending(g, "shortintadj"), include_groups=False
    ).droplevel(0)

    # Accruals: HIGH accruals → stronger short signal
    scored["acc_rank"] = grouped.apply(
        lambda g: rank_ascending(g, "accruals"), include_groups=False
    ).droplevel(0)

    # SUE: LOW (negative) SUE → stronger short signal
    scored["sue_rank"] = grouped.apply(
        lambda g: rank_descending(g, "suescore"), include_groups=False
    ).droplevel(0)

    # Momentum: LOW momentum → stronger short signal
    scored["mom_rank"] = grouped.apply(
        lambda g: rank_descending(g, "mom_12_1"), include_groups=False
    ).droplevel(0)

    # Audit score: HIGH → stronger short signal
    scored["aud_rank"] = grouped.apply(
        lambda g: rank_ascending(g, "audit_score"), include_groups=False
    ).droplevel(0)

    # Composite: weighted average of available ranks
    # Weights reflect academic evidence strength
    weights = {
        "mom_rank": 0.30,    # momentum is the strongest factor
        "acc_rank": 0.20,    # accrual anomaly well-documented
        "sue_rank": 0.20,    # PEAD is robust
        "si_rank": 0.15,     # short interest has confirming value
        "aud_rank": 0.15,    # audit flags are event-driven
    }

    rank_cols = list(weights.keys())
    w_array = np.array([weights[c] for c in rank_cols])

    # Compute weighted average, handling NaN by re-weighting
    rank_values = scored[rank_cols].values
    valid_mask = ~np.isnan(rank_values)
    w_matrix = np.where(valid_mask, w_array, 0)
    w_sums = w_matrix.sum(axis=1)

    # Require at least 2 signals
    min_signals = valid_mask.sum(axis=1) >= 2
    composite = np.where(
        min_signals & (w_sums > 0),
        np.nansum(rank_values * w_matrix, axis=1) / w_sums,
        np.nan,
    )
    scored["composite_short"] = composite

    n_scored = scored["composite_short"].notna().sum()
    print(f"  Scored: {n_scored:,} / {len(scored):,} ({n_scored/len(scored)*100:.1f}%)")
    print(f"  Composite range: [{np.nanmin(composite):.3f}, {np.nanmax(composite):.3f}]")

    return scored


# ═══════════════════════════════════════════════════════════════════════
# 5. BACKTEST
# ═══════════════════════════════════════════════════════════════════════

def backtest_short(scored_panel, prices, n_shorts=20, annual_borrow_cost=0.02,
                   slippage_bps=30):
    """Backtest short-only strategy: each month, short top-N by composite score.

    Returns monthly returns of the short portfolio.
    Short return = -1 × stock return (profit when stocks go down).
    Costs: borrowing cost + slippage on entry/exit.
    """
    print(f"\n[BT] Backtesting: top-{n_shorts} shorts, {annual_borrow_cost:.0%} borrow, {slippage_bps}bp slippage...")

    monthly_prices = prices.resample("ME").last()
    dates = sorted(scored_panel["date"].unique())

    monthly_borrow = annual_borrow_cost / 12
    round_trip_slip = slippage_bps / 10000 * 2  # entry + exit

    results = []
    holdings_history = []

    for i, dt in enumerate(dates[:-1]):  # skip last month (no forward return)
        next_dt = dates[i + 1]

        month_data = scored_panel[scored_panel["date"] == dt].dropna(subset=["composite_short"])
        if len(month_data) < n_shorts:
            continue

        # Select top-N shorts
        shorts = month_data.nlargest(n_shorts, "composite_short")
        short_tickers = shorts["ticker"].values

        # Compute forward returns (next month)
        rets = []
        for ticker in short_tickers:
            if ticker in monthly_prices.columns:
                p0 = monthly_prices.loc[dt, ticker]
                p1 = monthly_prices.loc[next_dt, ticker] if next_dt in monthly_prices.index else np.nan
                if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                    stock_ret = p1 / p0 - 1
                    short_ret = -stock_ret - monthly_borrow - round_trip_slip
                    rets.append(short_ret)

        if rets:
            port_ret = np.mean(rets)  # equal-weight
            results.append({
                "date": next_dt,
                "short_ret": port_ret,
                "n_shorts": len(rets),
                "avg_stock_ret": -np.mean([r + monthly_borrow + round_trip_slip for r in rets]),
            })
            holdings_history.append({
                "rebal_date": dt,
                "tickers": list(short_tickers),
                "n_held": len(rets),
            })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("  ERROR: No backtest results!")
        return results_df, []

    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df.sort_values("date")
    results_df["cumulative"] = (1 + results_df["short_ret"]).cumprod()

    print(f"  Months: {len(results_df)}, date range: {results_df['date'].min().date()} to {results_df['date'].max().date()}")
    return results_df, holdings_history


# ═══════════════════════════════════════════════════════════════════════
# 6. ANALYSIS & REPORT
# ═══════════════════════════════════════════════════════════════════════

def analyze_results(results_df, label="Short Strategy"):
    """Compute strategy performance metrics."""
    if results_df.empty:
        return {}

    rets = results_df["short_ret"].values
    n_months = len(rets)
    n_years = n_months / 12

    # Basic stats
    total_ret = results_df["cumulative"].iloc[-1] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = np.std(rets) * np.sqrt(12)
    sharpe = (np.mean(rets) * 12) / ann_vol if ann_vol > 0 else 0
    sortino_denom = np.std(rets[rets < 0]) * np.sqrt(12) if (rets < 0).any() else ann_vol
    sortino = (np.mean(rets) * 12) / sortino_denom if sortino_denom > 0 else 0

    # Drawdown
    cum = results_df["cumulative"].values
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Win rate
    win_rate = (rets > 0).mean()

    # Regime analysis (bear = negative avg stock return)
    results_df = results_df.copy()
    results_df["year"] = results_df["date"].dt.year
    yearly = results_df.groupby("year")["short_ret"].agg(["sum", "mean", "std", "count"])
    yearly.columns = ["total_ret", "mean_ret", "vol", "months"]

    metrics = {
        "label": label,
        "months": n_months,
        "total_return": total_ret,
        "cagr": cagr,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_monthly_ret": np.mean(rets),
        "yearly": yearly,
    }

    print(f"\n{'='*60}")
    print(f"  {label} — Performance Summary")
    print(f"{'='*60}")
    print(f"  Period: {results_df['date'].min().date()} to {results_df['date'].max().date()}")
    print(f"  Months: {n_months}")
    print(f"  Total Return:  {total_ret:+.1%}")
    print(f"  CAGR:          {cagr:+.1%}")
    print(f"  Ann Vol:       {ann_vol:.1%}")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Sortino:       {sortino:.2f}")
    print(f"  Max Drawdown:  {max_dd:.1%}")
    print(f"  Win Rate:      {win_rate:.1%}")
    print(f"\n  Yearly Returns:")
    for yr, row in yearly.iterrows():
        marker = " ***" if row["total_ret"] > 0.10 else ""
        print(f"    {yr}: {row['total_ret']:+.1%} ({row['months']:.0f}m){marker}")

    return metrics


def quintile_analysis(scored_panel, prices):
    """Split universe into quintiles by composite score, compare forward returns."""
    print("\n[QUINT] Quintile analysis of composite short score...")

    monthly_prices = prices.resample("ME").last()
    dates = sorted(scored_panel["date"].unique())

    quintile_rets = {q: [] for q in range(1, 6)}

    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        month_data = scored_panel[scored_panel["date"] == dt].dropna(subset=["composite_short"])
        if len(month_data) < 50:
            continue

        # Assign quintiles (1 = lowest composite, 5 = highest/worst)
        month_data = month_data.copy()
        month_data["quintile"] = pd.qcut(month_data["composite_short"], 5, labels=[1, 2, 3, 4, 5])

        for q in range(1, 6):
            q_data = month_data[month_data["quintile"] == q]
            q_rets = []
            for _, row in q_data.iterrows():
                ticker = row["ticker"]
                if ticker in monthly_prices.columns:
                    p0 = monthly_prices.loc[dt, ticker]
                    p1 = monthly_prices.loc[next_dt, ticker] if next_dt in monthly_prices.index else np.nan
                    if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                        q_rets.append(p1 / p0 - 1)
            if q_rets:
                quintile_rets[q].append(np.mean(q_rets))

    print(f"\n  {'Quintile':<12} {'Avg Monthly Ret':>16} {'Ann Ret':>10} {'N Months':>10}")
    print(f"  {'-'*50}")
    for q in range(1, 6):
        rets = quintile_rets[q]
        avg = np.mean(rets)
        ann = (1 + avg) ** 12 - 1
        print(f"  Q{q} {'(best)' if q==1 else '(worst)' if q==5 else '':8s} {avg:>+15.3%} {ann:>+9.1%} {len(rets):>10}")

    # Long-short spread: Q1 - Q5
    if quintile_rets[1] and quintile_rets[5]:
        min_len = min(len(quintile_rets[1]), len(quintile_rets[5]))
        spread = [quintile_rets[1][i] - quintile_rets[5][i] for i in range(min_len)]
        avg_spread = np.mean(spread)
        from scipy import stats
        t, p = stats.ttest_1samp(spread, 0)
        print(f"\n  Q1-Q5 Spread: {avg_spread:+.3%}/month (t={t:.2f}, p={p:.4f})")

    return quintile_rets


def write_report(metrics, quintile_rets, results_df, scored_panel):
    """Write full backtest report."""
    lines = []
    lines.append("# Russell 2000 Short Strategy — Backtest Report")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("\n## Strategy Design")
    lines.append("\n- **Universe**: True Russell 2000 membership (Norgate, via WRDS)")
    lines.append("- **Rebalance**: Monthly (month-end)")
    lines.append("- **Portfolio**: Short top-20 stocks by composite short score, equal-weight")
    lines.append("- **Costs**: 2% annual borrow + 30bp round-trip slippage")
    lines.append("\n### Composite Short Score (6 Signals)")
    lines.append("\n| Signal | Weight | Direction | Source |")
    lines.append("|--------|--------|-----------|--------|")
    lines.append("| Price Momentum 12-1 | 30% | Low → short | CRSP prices |")
    lines.append("| Accrual Anomaly | 20% | High → short | Compustat |")
    lines.append("| Earnings Surprise (SUE) | 20% | Low → short | I/B/E/S |")
    lines.append("| Short Interest | 15% | High → short | Compustat |")
    lines.append("| Audit Red Flags | 15% | High → short | Audit Analytics |")

    lines.append("\n## Performance Summary")
    lines.append(f"\n| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| CAGR | {metrics['cagr']:+.1%} |")
    lines.append(f"| Sharpe | {metrics['sharpe']:.2f} |")
    lines.append(f"| Sortino | {metrics['sortino']:.2f} |")
    lines.append(f"| Max Drawdown | {metrics['max_drawdown']:.1%} |")
    lines.append(f"| Win Rate | {metrics['win_rate']:.1%} |")
    lines.append(f"| Ann Volatility | {metrics['ann_volatility']:.1%} |")

    # Yearly
    lines.append("\n## Yearly Returns")
    lines.append(f"\n| Year | Return | Months |")
    lines.append(f"|------|--------|--------|")
    yearly = metrics["yearly"]
    for yr, row in yearly.iterrows():
        lines.append(f"| {yr} | {row['total_ret']:+.1%} | {row['months']:.0f} |")

    # Quintile
    lines.append("\n## Quintile Analysis (by Composite Score)")
    lines.append("\nQ1 = lowest composite (best stocks), Q5 = highest (worst/short candidates)")
    lines.append(f"\n| Quintile | Avg Monthly | Annualized |")
    lines.append(f"|----------|-------------|------------|")
    for q in range(1, 6):
        rets = quintile_rets[q]
        avg = np.mean(rets)
        ann = (1 + avg) ** 12 - 1
        lines.append(f"| Q{q} | {avg:+.3%} | {ann:+.1%} |")

    # Regime analysis
    lines.append("\n## Regime Analysis")
    bear_years = [2008, 2011, 2015, 2018, 2020, 2022]
    bull_years = [y for y in yearly.index if y not in bear_years]

    bear_rets = yearly.loc[yearly.index.isin(bear_years), "total_ret"]
    bull_rets = yearly.loc[yearly.index.isin(bull_years), "total_ret"]

    if not bear_rets.empty:
        lines.append(f"\n- **Bear years** ({', '.join(str(y) for y in bear_years if y in yearly.index)}): avg {bear_rets.mean():+.1%}")
    if not bull_rets.empty:
        lines.append(f"- **Bull years**: avg {bull_rets.mean():+.1%}")

    # Verdict
    lines.append("\n## Verdict")
    cagr = metrics["cagr"]
    sharpe = metrics["sharpe"]
    max_dd = metrics["max_drawdown"]

    if cagr > 0.03 and sharpe > 0.3:
        verdict = "PROCEED"
        lines.append(f"\n### **PROCEED** — viable short signal")
        lines.append(f"- CAGR {cagr:+.1%} with Sharpe {sharpe:.2f}")
        lines.append("- Next: combine with v10 long strategy, walk-forward validate")
    elif cagr > 0 and sharpe > 0.1:
        verdict = "MARGINAL"
        lines.append(f"\n### **MARGINAL** — weak signal, needs refinement")
        lines.append(f"- CAGR {cagr:+.1%} with Sharpe {sharpe:.2f}")
        lines.append("- Consider: stronger signal filtering, regime conditioning")
    else:
        verdict = "STOP"
        lines.append(f"\n### **STOP** — no viable short signal")
        lines.append(f"- CAGR {cagr:+.1%}, Sharpe {sharpe:.2f}")

    report = "\n".join(lines)
    report_file = OUT_DIR / "R2000_SHORT_BACKTEST_REPORT.md"
    report_file.write_text(report)
    print(f"\n[REPORT] Written to {report_file}")
    return verdict


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("  RUSSELL 2000 SHORT STRATEGY — COMPOSITE FACTOR BACKTEST")
    print("=" * 70)

    # 1. Load universe
    prices, membership = load_universe()

    # 2. Load all signal data
    si_df = load_short_interest()
    acc_df = load_accruals()
    ibes_df = load_ibes_surprise()
    rest_df = load_audit_restatements()
    sox_df = load_sox404()
    off_df = load_officer_changes()

    # 3. Compute momentum from prices
    mom_df = compute_momentum_12_1(prices, membership)

    # 4. Build monthly panel
    panel = build_monthly_panel(prices, membership, si_df, acc_df, ibes_df,
                                rest_df, sox_df, off_df, mom_df)

    # 5. Score
    scored = compute_composite_score(panel)
    scored.to_parquet(OUT_DIR / "scored_panel.parquet", index=False)

    # 6. Backtest
    results_20, holdings = backtest_short(scored, prices, n_shorts=20)
    results_20.to_parquet(OUT_DIR / "backtest_results_top20.parquet", index=False)

    # Also test top-10 and top-30 for robustness
    results_10, _ = backtest_short(scored, prices, n_shorts=10)
    results_30, _ = backtest_short(scored, prices, n_shorts=30)

    # 7. Analyze
    metrics_20 = analyze_results(results_20, "Top-20 Shorts")
    metrics_10 = analyze_results(results_10, "Top-10 Shorts")
    metrics_30 = analyze_results(results_30, "Top-30 Shorts")

    # 8. Quintile analysis
    q_rets = quintile_analysis(scored, prices)

    # 9. Report
    verdict = write_report(metrics_20, q_rets, results_20, scored)

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
