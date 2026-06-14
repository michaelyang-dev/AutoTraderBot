"""
Alpha Stacking Test: Layer additional alpha sources on top of skip-month momentum.
===================================================================================
Goal: Push OOS CAGR from +20.4% toward +25% without overfitting.

Layers tested:
1. ACCRUALS FILTER — exclude high-accrual stocks (earnings quality screen)
2. SHORT INTEREST — penalize crowded shorts, boost low-SI clean momentum
3. UMD REGIME — reduce exposure when momentum factor is crashing
4. COMBINED — all three together
"""

import numpy as np
import pandas as pd
import time
import logging
import pickle
from pathlib import Path
from functools import partial

from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from scoring_variants import strategy1_skip_month
from strategies.multi_strategy_engine import COST_BPS, INITIAL_CASH

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("alpha_stack")


class AlphaStackBacktester(FastBacktester):
    """Extended backtester with additional alpha layers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_accruals()
        self._load_short_interest()

    def _load_accruals(self):
        """Load accruals data and build point-in-time lookup."""
        log.info("Loading accruals...")
        acc = pd.read_parquet("data/wrds/computed_accruals.parquet")
        acc["datadate"] = pd.to_datetime(acc["datadate"])
        # Point-in-time: use accruals available 60 days after quarter end (filing lag)
        acc["available_date"] = acc["datadate"] + pd.Timedelta(days=60)
        acc = acc.dropna(subset=["accruals"])
        # Clip extreme accruals
        acc["accruals"] = acc["accruals"].clip(-2, 2)

        # Build {date: {ticker: accruals_percentile}}
        # For efficiency, pre-compute quarterly snapshots
        self._accruals_by_quarter = {}
        for qdate, grp in acc.groupby(acc["available_date"].dt.to_period("Q")):
            start = qdate.start_time
            latest = grp.sort_values("datadate").groupby("tic")["accruals"].last()
            # Compute percentile ranks
            ranks = latest.rank(pct=True)
            self._accruals_by_quarter[start] = ranks.to_dict()

        self._accruals_quarters = sorted(self._accruals_by_quarter.keys())
        log.info(f"Accruals loaded: {len(self._accruals_quarters)} quarters")

    def get_accruals_rank(self, date, ticker):
        """Get accruals percentile rank for a ticker at a given date (point-in-time)."""
        # Find most recent quarter available
        qidx = np.searchsorted(self._accruals_quarters, date, side="right") - 1
        if qidx < 0:
            return None
        q = self._accruals_quarters[qidx]
        ranks = self._accruals_by_quarter.get(q, {})
        return ranks.get(ticker)

    def _load_short_interest(self):
        """Load short interest and build point-in-time lookup."""
        log.info("Loading short interest...")
        si = pd.read_parquet("data/wrds/compustat_short_interest.parquet",
                             columns=["tic", "datadate", "shortintadj"])
        si["datadate"] = pd.to_datetime(si["datadate"])
        si = si.dropna(subset=["shortintadj"])
        si = si[si["shortintadj"] > 0]

        # Build monthly snapshots {month_start: {ticker: short_interest}}
        self._si_by_month = {}
        si["month"] = si["datadate"].dt.to_period("M")
        for month, grp in si.groupby("month"):
            start = month.start_time
            latest = grp.sort_values("datadate").groupby("tic")["shortintadj"].last()
            self._si_by_month[start] = latest.to_dict()

        self._si_months = sorted(self._si_by_month.keys())
        log.info(f"Short interest loaded: {len(self._si_months)} months")

    def get_short_interest_rank(self, date, ticker, members):
        """Get short interest percentile rank (within universe) for a ticker."""
        midx = np.searchsorted(self._si_months, date, side="right") - 1
        if midx < 0:
            return None
        m = self._si_months[midx]
        si_data = self._si_by_month.get(m, {})
        if ticker not in si_data:
            return None
        # Compute rank within available universe members
        vals = {s: si_data[s] for s in members if s in si_data}
        if len(vals) < 20:
            return None
        sorted_vals = sorted(vals.values())
        rank = np.searchsorted(sorted_vals, si_data[ticker]) / len(sorted_vals)
        return rank


def strategy1_with_alpha_stack(date, uni, day_idx, top_n=8, rebal_days=10,
                                accruals_filter=True, si_boost=True,
                                umd_regime=True, bt_ref=None, **kwargs):
    """
    Skip-month momentum with additional alpha layers:
    1. Accruals filter: penalize high-accrual stocks
    2. Short interest: boost low-SI, penalize high-SI
    3. UMD regime: reduce positions during momentum crashes
    """
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)
    if len(members) < 50:
        return {}

    regime = uni.get_regime(date)
    dist_sma50_all = uni.get_feature_map(date, "dist_sma50")
    if dist_sma50_all:
        mkt_breadth = sum(1 for v in dist_sma50_all.values() if v > 0) / max(len(dist_sma50_all), 1)
    else:
        mkt_breadth = 0.5
    stress = mkt_breadth < 0.30
    n = max(top_n // 2, 5) if stress else top_n

    # UMD regime: reduce positions during momentum factor crashes
    if umd_regime and bt_ref is not None:
        umd = bt_ref.umd_20d.loc[:date]
        if len(umd) > 0 and pd.notna(umd.iloc[-1]):
            umd_val = umd.iloc[-1]
            if umd_val < -0.10:  # severe momentum crash
                n = max(3, n // 2)  # halve positions
            elif umd_val < -0.05:  # moderate crash
                n = max(5, int(n * 0.75))

    # Core: skip-month momentum + SMA200
    ret_20 = uni.get_feature_map(date, "ret_20d", members)
    ret_252 = uni.get_feature_map(date, "ret_252d", members)
    dist_sma200 = uni.get_feature_map(date, "dist_sma200", members)
    eps_surp = uni.get_feature_map(date, "eps_surprise_last", members)
    roe_map = uni.get_feature_map(date, "roe")

    composite = {}
    for sym in members:
        r252 = ret_252.get(sym)
        r20 = ret_20.get(sym)
        if r252 is None or r20 is None or np.isnan(r252) or np.isnan(r20):
            continue

        score = r252 - r20

        # Quality boosts
        es = eps_surp.get(sym)
        if es is not None and not np.isnan(es) and es > 0:
            score *= 1.15
        roe_val = roe_map.get(sym)
        if roe_val is not None and not np.isnan(roe_val) and roe_val > 0.15:
            score *= 1.05
        fg = uni._fin_growth.get(sym)
        if fg:
            rg = fg.get("rev_growth")
            eg = fg.get("eps_growth")
            if rg is not None and not np.isnan(rg) and rg > 0.08:
                score *= 1.10
            if eg is not None and not np.isnan(eg) and eg > 0.10:
                score *= 1.10

        # === ALPHA LAYER 1: Accruals filter ===
        if accruals_filter and bt_ref is not None:
            acc_rank = bt_ref.get_accruals_rank(date, sym)
            if acc_rank is not None:
                if acc_rank > 0.80:  # High accruals = low quality earnings
                    score *= 0.70  # heavy penalty
                elif acc_rank > 0.60:
                    score *= 0.90  # moderate penalty
                elif acc_rank < 0.20:  # Low accruals = high quality
                    score *= 1.10  # quality bonus

        # === ALPHA LAYER 2: Short interest ===
        if si_boost and bt_ref is not None:
            si_rank = bt_ref.get_short_interest_rank(date, sym, members)
            if si_rank is not None:
                if si_rank > 0.90:  # Very high short interest = crowded
                    score *= 0.80
                elif si_rank < 0.10:  # Very low SI = clean momentum
                    score *= 1.10

        # SMA200 trend filter
        d200 = dist_sma200.get(sym, 0)
        if d200 is not None and d200 > 0:
            composite[sym] = score

    # Bear sector tilt
    spy_bull = regime.get("spy_above_sma200", True)
    if not spy_bull and composite:
        sec_rets = {}
        for etf in ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]:
            close = uni.get_close_series(etf, date, 65)
            if len(close) >= 60:
                sec_rets[etf] = (close.iloc[-1] / close.iloc[-60]) - 1.0
        etf_to_sec = {"XLK": "Technology", "XLF": "Financial Services", "XLE": "Energy",
                      "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Consumer Cyclical",
                      "XLP": "Consumer Defensive", "XLB": "Basic Materials", "XLRE": "Real Estate",
                      "XLU": "Utilities", "XLC": "Communication Services"}
        if sec_rets:
            ranked = sorted(sec_rets, key=sec_rets.get, reverse=True)
            top_secs = {etf_to_sec.get(e, "") for e in ranked[:3]}
            bot_secs = {etf_to_sec.get(e, "") for e in ranked[-3:]}
            boosted = {}
            for sym, sc in composite.items():
                sym_sec = uni.sector_map.get(sym, "")
                if sym_sec in top_secs:
                    boosted[sym] = sc * 2.0
                elif sym_sec in bot_secs:
                    continue
                else:
                    boosted[sym] = sc
            composite = boosted

    if not composite:
        return {}

    # Signal-weighted allocation
    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:n]
    scores = [max(composite[s], 0.001) for s in sorted_syms]
    total = sum(scores)
    if total > 0:
        weights = {s: min(sc / total, 2.0 / len(sorted_syms)) for s, sc in zip(sorted_syms, scores)}
        wt = sum(weights.values())
        if wt > 0:
            weights = {s: w / wt for s, w in weights.items()}
        return weights
    return {}


def monkey_patch_run(bt, mom_func, start, end, config):
    import strategies.multi_strategy_engine as mse
    import main_production_backtest as fb
    orig_mse = mse.strategy1_momentum_reversal
    orig_fb = fb.strategy1_momentum_reversal
    mse.strategy1_momentum_reversal = mom_func
    fb.strategy1_momentum_reversal = mom_func
    try:
        result = bt.run(start, end, config)
    finally:
        mse.strategy1_momentum_reversal = orig_mse
        fb.strategy1_momentum_reversal = orig_fb
    return result


if __name__ == "__main__":
    print("Loading data (with accruals + short interest)...", flush=True)
    bt = AlphaStackBacktester()
    print("Data loaded.\n", flush=True)

    config = {
        "universe": "sp1500", "mom_w": 0.85, "val_w": 0.15,
        "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25, "cap": 0.15,
    }

    # Baseline: current production (skip-month, no alpha layers)
    func_base = partial(strategy1_skip_month, trend_filter="sma200", quality_boosts=True)

    # Alpha layers
    func_accruals = partial(strategy1_with_alpha_stack,
                            accruals_filter=True, si_boost=False, umd_regime=False, bt_ref=bt)
    func_si = partial(strategy1_with_alpha_stack,
                      accruals_filter=False, si_boost=True, umd_regime=False, bt_ref=bt)
    func_umd = partial(strategy1_with_alpha_stack,
                       accruals_filter=False, si_boost=False, umd_regime=True, bt_ref=bt)
    func_all = partial(strategy1_with_alpha_stack,
                       accruals_filter=True, si_boost=True, umd_regime=True, bt_ref=bt)
    func_acc_si = partial(strategy1_with_alpha_stack,
                          accruals_filter=True, si_boost=True, umd_regime=False, bt_ref=bt)

    configs_to_test = [
        ("BASELINE: skip12-1 sma200 85/15 qual", func_base),
        ("+ ACCRUALS filter only", func_accruals),
        ("+ SHORT INTEREST only", func_si),
        ("+ UMD REGIME only", func_umd),
        ("+ ACCRUALS + SI", func_acc_si),
        ("+ ALL THREE (accruals + SI + UMD)", func_all),
    ]

    print("=" * 110, flush=True)
    print("ALPHA STACKING TEST — Full sample + OOS (2022-2025)", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'Config':<50} {'Full CAGR':>9} {'Full Shrp':>9} {'OOS CAGR':>9} {'OOS Shrp':>9} {'OOS DD':>7}", flush=True)
    print(f"  {'-'*95}", flush=True)

    for name, func in configs_to_test:
        t0 = time.time()
        full = monkey_patch_run(bt, func, "2018-01-01", "2025-12-31", config)
        oos = monkey_patch_run(bt, func, "2022-01-01", "2025-12-31", config)
        elapsed = time.time() - t0

        if full and oos:
            print(f"  {name:<50} {full['cagr']:>+8.1%} {full['sharpe']:>9.2f} "
                  f"{oos['cagr']:>+8.1%} {oos['sharpe']:>9.2f} {oos['max_dd']:>6.0%}  [{elapsed:.0f}s]", flush=True)
            yrs = " ".join(f"{y}:{full['yearly'][y]['cagr']:+.0%}" for y in sorted(full['yearly'].keys()))
            print(f"    Yearly: {yrs}", flush=True)

    # Test with different position counts for the stacked version
    print(f"\n{'=' * 110}", flush=True)
    print("POSITION COUNT SENSITIVITY (ALL THREE layers)", flush=True)
    print("=" * 110, flush=True)

    for top_n in [6, 8, 10, 12]:
        cfg = {**config, "top_n": top_n}
        full = monkey_patch_run(bt, func_all, "2018-01-01", "2025-12-31", cfg)
        oos = monkey_patch_run(bt, func_all, "2022-01-01", "2025-12-31", cfg)
        if full and oos:
            print(f"  top_{top_n}: Full={full['cagr']:+.1%} S={full['sharpe']:.2f} | "
                  f"OOS={oos['cagr']:+.1%} S={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}", flush=True)

    # Test without value blend (pure momentum + alpha stack)
    print(f"\n{'=' * 110}", flush=True)
    print("BLEND SENSITIVITY (ALL THREE layers)", flush=True)
    print("=" * 110, flush=True)

    for mom_w, val_w in [(1.0, 0.0), (0.85, 0.15), (0.70, 0.30)]:
        cfg = {**config, "mom_w": mom_w, "val_w": val_w}
        full = monkey_patch_run(bt, func_all, "2018-01-01", "2025-12-31", cfg)
        oos = monkey_patch_run(bt, func_all, "2022-01-01", "2025-12-31", cfg)
        if full and oos:
            label = f"{int(mom_w*100)}/{int(val_w*100)}"
            print(f"  {label}: Full={full['cagr']:+.1%} S={full['sharpe']:.2f} | "
                  f"OOS={oos['cagr']:+.1%} S={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}", flush=True)
            yrs = " ".join(f"{y}:{full['yearly'][y]['cagr']:+.0%}" for y in sorted(full['yearly'].keys()))
            print(f"    {yrs}", flush=True)

    print("\nDone.", flush=True)
