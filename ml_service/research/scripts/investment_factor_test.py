"""
Investment Factor (Low Asset Growth) — Second Long Sleeve
==========================================================
Academic: Cooper, Gulen, Schill (2008) — firms with low total asset growth
outperform by ~7%/yr. Negatively correlated with momentum.

This strategy:
1. Ranks SP1500 stocks by trailing asset growth (low = good)
2. Filters for above-SMA200 trend
3. Picks top-N lowest asset growth stocks
4. Rebalances quarterly (slower signal)

Tests both standalone and combined with momentum.
"""

import numpy as np
import pandas as pd
import time
import logging
import pickle
from pathlib import Path
from functools import partial

from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, COST_BPS, INITIAL_CASH
)

logging.basicConfig(level=logging.WARNING)


class InvestmentFactorBacktester(FastBacktester):
    """Extended backtester with investment factor (asset growth) data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_asset_growth()

    def _load_asset_growth(self):
        """Load and prepare asset growth data."""
        ca = pd.read_parquet("data/wrds/compustat_annual.parquet",
                             columns=["tic", "datadate", "at"])
        ca["datadate"] = pd.to_datetime(ca["datadate"])
        ca = ca.sort_values(["tic", "datadate"])
        ca["at_growth"] = ca.groupby("tic")["at"].pct_change()
        ca["available_date"] = ca["datadate"] + pd.Timedelta(days=90)
        ca = ca.dropna(subset=["at_growth", "at"])
        ca = ca[ca["at"] > 0]
        ca["at_growth"] = ca["at_growth"].clip(-1, 5)

        # Build quarterly snapshots: {quarter_start: {ticker: at_growth}}
        self._ag_by_quarter = {}
        ca_sorted = ca.sort_values("available_date")
        quarters = pd.date_range("2016-01-01", "2026-07-01", freq="QS")
        for q in quarters:
            available = ca_sorted[ca_sorted["available_date"] <= q]
            if len(available) == 0:
                continue
            latest = available.groupby("tic")["at_growth"].last()
            self._ag_by_quarter[q] = latest.to_dict()

        self._ag_quarters = sorted(self._ag_by_quarter.keys())
        print(f"  Asset growth loaded: {len(self._ag_quarters)} quarters", flush=True)

    def get_asset_growth(self, date):
        """Get {ticker: at_growth} for the latest available data as of date."""
        qidx = np.searchsorted(self._ag_quarters, date, side="right") - 1
        if qidx < 0:
            return {}
        q = self._ag_quarters[qidx]
        return self._ag_by_quarter.get(q, {})


def strategy_investment_factor(date, uni, day_idx, top_n=10, rebal_days=10,
                                bt_ref=None, **kwargs):
    """
    Investment factor: buy stocks with LOW asset growth + above SMA200.
    Conservative firms that don't over-invest tend to outperform.
    """
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)
    if len(members) < 50 or bt_ref is None:
        return {}

    # Get asset growth data
    ag_data = bt_ref.get_asset_growth(date)
    if not ag_data:
        return {}

    # Get trend filter
    dist_sma200 = uni.get_feature_map(date, "dist_sma200", members)
    roe_map = uni.get_feature_map(date, "roe", members)

    scores = {}
    for sym in members:
        ag = ag_data.get(sym)
        if ag is None:
            continue

        # Must be above SMA200
        d200 = dist_sma200.get(sym, 0)
        if d200 is None or d200 <= 0:
            continue

        # Quality filter: positive ROE
        roe = roe_map.get(sym)
        if roe is not None and not np.isnan(roe) and roe < 0.05:
            continue

        # Score = negative asset growth (lower growth = better)
        # Exclude extreme shrinkage (distressed firms)
        if ag < -0.30:
            continue

        scores[sym] = -ag  # negative = we WANT low growth

    if not scores:
        return {}

    # Select top N
    sorted_syms = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return {s: 1.0 / len(sorted_syms) for s in sorted_syms}


def run_combined(bt, start, end, mom_pct, inv_pct, config):
    """
    Run combined momentum + investment factor strategy.
    mom_pct + inv_pct should = 1.0 (or less with cash buffer).
    """
    universe = config.get("universe", "sp1500")
    if universe == "sp1500":
        bt.uni.get_sp500 = bt._get_sp1500

    trading_dates = [d for d in sorted(bt.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

    cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
    top_n = config.get("top_n", 8)
    rebal_days = config.get("rebal_days", 10)
    trailing_stop = config.get("trailing_stop", 0.25)
    cap = config.get("cap", 0.20)

    cash = INITIAL_CASH
    holdings = {}
    port_values = []

    for day_idx, date in enumerate(trading_dates):
        today = {}
        if date in bt.prices.index:
            row = bt.prices.loc[date]
            for sym in list(holdings.keys()):
                v = row.get(sym)
                if v is not None and not np.isnan(v):
                    today[sym] = v
            for sym in row.dropna().index:
                today[sym] = row[sym]

        # Trailing stop
        if trailing_stop:
            for sym in list(holdings):
                px = today.get(sym)
                if px:
                    if "peak_px" not in holdings[sym]:
                        holdings[sym]["peak_px"] = px
                    if px > holdings[sym]["peak_px"]:
                        holdings[sym]["peak_px"] = px
                    dd = (px - holdings[sym]["peak_px"]) / holdings[sym]["peak_px"]
                    if dd < -abs(trailing_stop):
                        cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                        del holdings[sym]

        eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                            for s, h in holdings.items())
        total_val = eq_val

        if day_idx % rebal_days != 0:
            port_values.append((date, total_val))
            continue

        # Update SI ranks (pre-computed, O(1))
        if bt._si_months:
            midx = np.searchsorted(bt._si_months, date, side="right") - 1
            if midx >= 0:
                bt.uni._short_interest_rank = bt._si_ranks_by_month.get(
                    bt._si_months[midx], {})

        # Get momentum targets
        t_mom = strategy1_momentum_reversal(date, bt.uni, day_idx,
                                             top_n=top_n, rebal_days=rebal_days)
        if t_mom is None:
            t_mom = {}

        # Get investment factor targets
        t_inv = strategy_investment_factor(date, bt.uni, day_idx,
                                           top_n=top_n, rebal_days=rebal_days,
                                           bt_ref=bt)
        if t_inv is None:
            t_inv = {}

        # Combine
        combined = {}
        for sym, w in t_mom.items():
            combined[sym] = combined.get(sym, 0) + w * mom_pct
        for sym, w in t_inv.items():
            combined[sym] = combined.get(sym, 0) + w * inv_pct

        # Cap and normalize
        for sym in list(combined):
            if combined[sym] > cap:
                combined[sym] = cap
        gross = sum(combined.values())
        if gross > 1.0:
            for sym in combined:
                combined[sym] /= gross
        combined = {s: w for s, w in combined.items() if w >= 0.005}

        target_d = {s: w * total_val for s, w in combined.items()}

        # Execute trades
        for sym in list(holdings):
            if sym not in target_d:
                px = today.get(sym, holdings[sym]["entry_px"])
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                del holdings[sym]

        for sym, tgt in target_d.items():
            px = today.get(sym)
            if not px or px <= 0:
                continue
            cur = holdings[sym]["shares"] * px if sym in holdings else 0
            delta = tgt - cur
            if abs(delta) < total_val * 0.003:
                continue
            cost = abs(delta) * cost_frac
            if delta > 0 and cash >= delta:
                shares = (delta - cost) / px
                if sym in holdings:
                    holdings[sym]["shares"] += shares
                else:
                    holdings[sym] = {"shares": shares, "entry_px": px, "peak_px": px}
                cash -= delta
            elif delta < 0 and sym in holdings:
                sell = min(abs(delta) / px, holdings[sym]["shares"])
                cash += sell * px - cost
                holdings[sym]["shares"] -= sell
                if holdings[sym]["shares"] < 0.01:
                    del holdings[sym]

        eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                            for s, h in holdings.items())
        total_val = eq_val
        port_values.append((date, max(total_val, 0)))

    # Metrics
    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0:
        years = 1
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    yearly = {}
    for year in range(int(start[:4]), int(end[:4]) + 1):
        mask = (vals.index >= f"{year}-01-01") & (vals.index <= f"{year}-12-31")
        yv = vals[mask]
        if len(yv) > 10:
            yr = (yv.iloc[-1] / yv.iloc[0]) - 1
            yearly[year] = {"cagr": yr}
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "yearly": yearly}


if __name__ == "__main__":
    print("Loading data...", flush=True)
    bt = InvestmentFactorBacktester()
    print("Data loaded.\n", flush=True)

    config = {"universe": "sp1500", "top_n": 8, "rebal_days": 10,
              "trailing_stop": 0.25, "cap": 0.20}

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Investment factor standalone
    # ═══════════════════════════════════════════════════════════════
    print("=" * 100, flush=True)
    print("INVESTMENT FACTOR STANDALONE", flush=True)
    print("=" * 100, flush=True)

    for top_n in [8, 10, 15]:
        cfg = {**config, "top_n": top_n}
        r = run_combined(bt, "2018-01-01", "2025-12-31", mom_pct=0.0, inv_pct=1.0, config=cfg)
        if r:
            yrs = " ".join(f"{y}:{r['yearly'][y]['cagr']:+.0%}" for y in sorted(r['yearly'].keys()))
            print(f"  INV top{top_n}: Full={r['cagr']:+.1%} S={r['sharpe']:.2f} DD={r['max_dd']:.0%}", flush=True)
            print(f"    {yrs}", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Combined momentum + investment factor
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}", flush=True)
    print("COMBINED: MOMENTUM + INVESTMENT FACTOR", flush=True)
    print("=" * 100, flush=True)

    blends = [
        (1.0, 0.0, "100% Mom"),
        (0.85, 0.15, "85% Mom + 15% Inv"),
        (0.75, 0.25, "75% Mom + 25% Inv"),
        (0.65, 0.35, "65% Mom + 35% Inv"),
        (0.50, 0.50, "50% Mom + 50% Inv"),
    ]

    print(f"  {'Blend':<25} {'Full CAGR':>9} {'Sharpe':>7} {'DD':>5}  |  {'OOS CAGR':>9} {'OOS Shrp':>9}", flush=True)
    print(f"  {'-'*80}", flush=True)

    for mom_pct, inv_pct, label in blends:
        t0 = time.time()
        full = run_combined(bt, "2018-01-01", "2025-12-31", mom_pct, inv_pct, config)
        oos = run_combined(bt, "2022-01-01", "2025-12-31", mom_pct, inv_pct, config)
        elapsed = time.time() - t0

        if full and oos:
            print(f"  {label:<25} {full['cagr']:>+8.1%} {full['sharpe']:>7.2f} {full['max_dd']:>4.0%}"
                  f"  |  {oos['cagr']:>+8.1%} {oos['sharpe']:>9.2f}  [{elapsed:.0f}s]", flush=True)
            yrs = " ".join(f"{y}:{full['yearly'][y]['cagr']:+.0%}" for y in sorted(full['yearly'].keys()))
            print(f"    {yrs}", flush=True)

    print("\nDone.", flush=True)
