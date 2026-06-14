"""
Factor Timing Strategy Backtest
================================
Uses rolling Fama-French factor returns to dynamically shift allocations
between momentum, value, low-vol, and sector strategies.

Logic:
  - Compute 60-day and 20-day cumulative returns for each FF factor
  - A factor is "strong" if BOTH 60d and 20d returns are positive
  - Allocations shift based on which factors are currently working
  - All signals use ONLY data available before the period (no look-ahead)

Compares dynamic factor-timed allocation vs static v10 (85/15).
"""

import numpy as np
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

from wrds_data_provider import WRDSDataProvider
from main_production_backtest import FastBacktester

# ─── Factor Signal Computation ───────────────────────────────────────────────

def compute_factor_signals(ff: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling factor momentum signals.

    Returns DataFrame with columns like 'umd_60d', 'umd_20d', 'umd_strong', etc.
    """
    factors = ["mktrf", "smb", "hml", "umd"]
    signals = pd.DataFrame(index=ff.index)

    for f in factors:
        # Cumulative return over window (sum of daily log-ish returns)
        signals[f"{f}_60d"] = ff[f].rolling(60).sum()
        signals[f"{f}_20d"] = ff[f].rolling(20).sum()
        # "Strong" = positive on both horizons
        signals[f"{f}_strong"] = (signals[f"{f}_60d"] > 0) & (signals[f"{f}_20d"] > 0)
        # "Weak" = negative on both horizons
        signals[f"{f}_weak"] = (signals[f"{f}_60d"] < 0) & (signals[f"{f}_20d"] < 0)

    return signals


def get_allocation_for_date(signals: pd.DataFrame, as_of_date: pd.Timestamp) -> dict:
    """Determine mom_w / val_w / lv_w / sec_w based on factor signals.

    Uses only data available up to as_of_date (no look-ahead).

    Base allocation: mom_w=0.85, val_w=0.15, lv_w=0.0, sec_w=0.0

    Adjustments:
      - Strong UMD: boost momentum (it's working)
      - Weak UMD: cut momentum, add value + lowvol (defensive)
      - Strong HML: boost value
      - Strong SMB: keep momentum (small-cap momentum benefits)
      - Weak MKT-RF: add lowvol for defense
    """
    sig = signals.loc[:as_of_date]
    if len(sig) < 60:
        # Not enough data, use static
        return {"mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0}

    latest = sig.iloc[-1]

    # Start from base
    mom_w = 0.85
    val_w = 0.15
    lv_w = 0.0
    sec_w = 0.0

    # ── UMD (momentum factor) regime ──
    if latest.get("umd_strong", False):
        # Momentum is working: go aggressive
        mom_w = 0.90
        val_w = 0.10
        lv_w = 0.0
        sec_w = 0.0
    elif latest.get("umd_weak", False):
        # Momentum crashing: defensive pivot
        mom_w = 0.40
        val_w = 0.30
        lv_w = 0.25
        sec_w = 0.05

    # ── HML (value factor) modifier ──
    if latest.get("hml_strong", False):
        # Value is working: tilt toward value
        val_w += 0.10
        mom_w -= 0.10

    # ── MKT-RF (market factor) modifier ──
    if latest.get("mktrf_weak", False):
        # Market struggling: add lowvol defense
        lv_w += 0.10
        mom_w = max(0.20, mom_w - 0.10)

    # ── SMB (small-cap factor) — informational but no weight change ──
    # Strong SMB benefits our SP1500 universe already

    # Normalize to sum to 1.0
    total = mom_w + val_w + lv_w + sec_w
    if total > 0:
        mom_w /= total
        val_w /= total
        lv_w /= total
        sec_w /= total

    return {"mom_w": round(mom_w, 3), "val_w": round(val_w, 3),
            "lv_w": round(lv_w, 3), "sec_w": round(sec_w, 3)}


def get_yearly_allocations(signals: pd.DataFrame, start_year=2018, end_year=2025) -> dict:
    """For each year, determine allocation using data available BEFORE that year starts.

    Also computes mid-year check: if factor regime flips mid-year, we note it.
    """
    allocations = {}

    for year in range(start_year, end_year + 1):
        # Use data through Dec 31 of prior year
        as_of = pd.Timestamp(f"{year-1}-12-31")
        alloc = get_allocation_for_date(signals, as_of)
        allocations[year] = alloc

    return allocations


def get_quarterly_allocations(signals: pd.DataFrame, start_year=2018, end_year=2025) -> list:
    """Quarterly rebalancing: determine allocation at the start of each quarter.

    Returns list of (start_date, end_date, config_dict) tuples.
    """
    periods = []
    for year in range(start_year, end_year + 1):
        quarters = [
            (f"{year}-01-01", f"{year}-03-31"),
            (f"{year}-04-01", f"{year}-06-30"),
            (f"{year}-07-01", f"{year}-09-30"),
            (f"{year}-10-01", f"{year}-12-31"),
        ]
        for q_start, q_end in quarters:
            # Use data through the day before the quarter starts
            as_of = pd.Timestamp(q_start) - pd.Timedelta(days=1)
            alloc = get_allocation_for_date(signals, as_of)
            periods.append((q_start, q_end, alloc))

    return periods


# ─── Main Backtest ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 75)
    print("FACTOR TIMING STRATEGY BACKTEST")
    print("=" * 75)

    # Load FF data and compute signals
    print("\n[1] Loading Fama-French factors...")
    ff = WRDSDataProvider().fama_french
    signals = compute_factor_signals(ff)

    # Show factor signals at year boundaries
    print("\n[2] Factor regime at each year start (using prior-year-end data):")
    print(f"  {'Year':<6} {'UMD':^12} {'HML':^12} {'MKTRF':^12} {'SMB':^12}  → {'Alloc'}")
    print("  " + "-" * 80)

    yearly_allocs = get_yearly_allocations(signals)
    for year, alloc in yearly_allocs.items():
        as_of = pd.Timestamp(f"{year-1}-12-31")
        sig = signals.loc[:as_of]
        if len(sig) < 60:
            continue
        s = sig.iloc[-1]

        def regime(factor):
            if s.get(f"{factor}_strong", False):
                return "STRONG"
            elif s.get(f"{factor}_weak", False):
                return "WEAK"
            else:
                return "neutral"

        alloc_str = f"M{alloc['mom_w']:.0%}/V{alloc['val_w']:.0%}/L{alloc['lv_w']:.0%}"
        print(f"  {year:<6} {regime('umd'):^12} {regime('hml'):^12} {regime('mktrf'):^12} {regime('smb'):^12}  → {alloc_str}")

    # Initialize backtester (loads all data once)
    print("\n[3] Loading backtest engine...")
    bt = FastBacktester()

    # ── Static v10 baseline ──
    print("\n[4] Running static v10.1 baseline (85/15)...")
    static_config = {
        "universe": "sp1500",
        "mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
        "vol_scaling": True, "vol_target": 0.18,
        "gld_pct": 0.02, "vixm_pct": 0.02,
    }
    t0 = time.time()
    static_result = bt.run("2018-01-01", "2025-12-31", static_config)
    static_time = time.time() - t0

    # ── Factor-timed: yearly rebalance ──
    print("\n[5] Running YEARLY factor-timed backtest...")
    # Run year-by-year with different allocations
    # We can't easily stitch year-by-year runs (portfolio state doesn't carry over)
    # Instead, we'll test a few representative allocations and also do
    # a full-period run using the AVERAGE optimal allocation

    yearly_results = {}
    for year in range(2018, 2026):
        alloc = yearly_allocs[year]
        cfg = {
            "universe": "sp1500",
            "mom_w": alloc["mom_w"], "val_w": alloc["val_w"],
            "lv_w": alloc["lv_w"], "sec_w": alloc["sec_w"],
            "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
            "vol_scaling": True, "vol_target": 0.18,
            "gld_pct": 0.02, "vixm_pct": 0.02,
        }
        r = bt.run(f"{year}-01-01", f"{year}-12-31", cfg)
        if r:
            yearly_results[year] = {
                "alloc": alloc,
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "max_dd": r["max_dd"],
            }

    # ── Factor-timed: quarterly rebalance ──
    print("\n[6] Running QUARTERLY factor-timed backtest...")
    quarterly_periods = get_quarterly_allocations(signals)
    quarterly_results = {}
    for q_start, q_end, alloc in quarterly_periods:
        cfg = {
            "universe": "sp1500",
            "mom_w": alloc["mom_w"], "val_w": alloc["val_w"],
            "lv_w": alloc["lv_w"], "sec_w": alloc["sec_w"],
            "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
            "vol_scaling": True, "vol_target": 0.18,
            "gld_pct": 0.02, "vixm_pct": 0.02,
        }
        r = bt.run(q_start, q_end, cfg)
        if r and r["cagr"] is not None:
            quarterly_results[q_start] = {
                "alloc": alloc,
                "return": r["final"] / 100000 - 1,  # single-period return
                "sharpe": r["sharpe"],
                "max_dd": r["max_dd"],
            }

    # Compound quarterly returns into annual and full-period
    def compound_quarterly(qr, year):
        """Compound quarterly returns for a given year."""
        qs = [f"{year}-01-01", f"{year}-04-01", f"{year}-07-01", f"{year}-10-01"]
        ret = 1.0
        for q in qs:
            if q in qr:
                ret *= (1 + qr[q]["return"])
        return ret - 1

    def compound_all_quarterly(qr, start_year, end_year):
        """Compound all quarterly returns."""
        ret = 1.0
        for q_start in sorted(qr.keys()):
            ret *= (1 + qr[q_start]["return"])
        return ret

    # ── Also run a full-period "adaptive" backtest using the most common allocation ──
    # Weight-average the yearly allocations
    avg_alloc = {"mom_w": 0, "val_w": 0, "lv_w": 0, "sec_w": 0}
    for year, alloc in yearly_allocs.items():
        for k in avg_alloc:
            avg_alloc[k] += alloc[k]
    n_years = len(yearly_allocs)
    for k in avg_alloc:
        avg_alloc[k] = round(avg_alloc[k] / n_years, 3)

    print(f"\n  Average factor-timed allocation: M{avg_alloc['mom_w']:.1%}/V{avg_alloc['val_w']:.1%}/L{avg_alloc['lv_w']:.1%}/S{avg_alloc['sec_w']:.1%}")

    avg_config = {
        "universe": "sp1500",
        "mom_w": avg_alloc["mom_w"], "val_w": avg_alloc["val_w"],
        "lv_w": avg_alloc["lv_w"], "sec_w": avg_alloc["sec_w"],
        "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
        "vol_scaling": True, "vol_target": 0.18,
        "gld_pct": 0.02, "vixm_pct": 0.02,
    }
    avg_result = bt.run("2018-01-01", "2025-12-31", avg_config)

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 75)
    print("RESULTS")
    print("=" * 75)

    # Static baseline
    print(f"\n  STATIC v10.1 (85/15 mom/val, full period):")
    print(f"    CAGR:    {static_result['cagr']:>+7.1%}")
    print(f"    Sharpe:  {static_result['sharpe']:>7.2f}")
    print(f"    Sortino: {static_result['sortino']:>7.2f}")
    print(f"    Max DD:  {static_result['max_dd']:>7.1%}")
    print(f"    Vol:     {static_result['vol']:>7.1%}")
    print(f"    Final:   ${static_result['final']:>12,.0f}")

    # Year-by-year comparison
    print(f"\n  YEAR-BY-YEAR COMPARISON:")
    print(f"  {'Year':<6} {'Static':>8} {'FT-Year':>8} {'FT-Qtr':>8}  {'Factor-Timed Alloc':<30} {'Regime'}")
    print("  " + "-" * 90)

    for year in range(2018, 2026):
        static_yr = static_result["yearly"].get(year, {}).get("cagr", 0)
        ft_yr = yearly_results.get(year, {}).get("cagr", 0)
        ft_qtr = compound_quarterly(quarterly_results, year)
        alloc = yearly_allocs.get(year, {})
        alloc_str = f"M{alloc.get('mom_w',0):.0%}/V{alloc.get('val_w',0):.0%}/L{alloc.get('lv_w',0):.0%}/S{alloc.get('sec_w',0):.0%}"

        # Determine regime
        as_of = pd.Timestamp(f"{year-1}-12-31")
        sig = signals.loc[:as_of]
        regime = ""
        if len(sig) >= 60:
            s = sig.iloc[-1]
            parts = []
            if s.get("umd_strong", False): parts.append("UMD+")
            if s.get("umd_weak", False): parts.append("UMD-")
            if s.get("hml_strong", False): parts.append("HML+")
            if s.get("hml_weak", False): parts.append("HML-")
            if s.get("mktrf_weak", False): parts.append("MKT-")
            regime = " ".join(parts) if parts else "neutral"

        better = "  <<" if ft_yr > static_yr else ""
        print(f"  {year:<6} {static_yr:>+7.0%} {ft_yr:>+7.0%} {ft_qtr:>+7.0%}  {alloc_str:<30} {regime}{better}")

    # Compound annual results
    static_compound = 1.0
    ft_year_compound = 1.0
    ft_qtr_compound = 1.0
    for year in range(2018, 2026):
        static_compound *= (1 + static_result["yearly"].get(year, {}).get("cagr", 0))
        ft_year_compound *= (1 + yearly_results.get(year, {}).get("cagr", 0))
        ft_qtr_compound *= (1 + compound_quarterly(quarterly_results, year))

    years = 8.0
    static_cagr_comp = static_compound ** (1/years) - 1
    ft_year_cagr = ft_year_compound ** (1/years) - 1
    ft_qtr_cagr = ft_qtr_compound ** (1/years) - 1

    print(f"\n  COMPOUNDED CAGR (from annual returns):")
    print(f"    Static v10.1:            {static_cagr_comp:>+7.1%}")
    print(f"    Factor-Timed (Yearly):   {ft_year_cagr:>+7.1%}")
    print(f"    Factor-Timed (Quarterly):{ft_qtr_cagr:>+7.1%}")

    # Average allocation full-period run
    print(f"\n  AVERAGE ALLOCATION (full period run):")
    print(f"    Config: M{avg_alloc['mom_w']:.0%}/V{avg_alloc['val_w']:.0%}/L{avg_alloc['lv_w']:.0%}/S{avg_alloc['sec_w']:.0%}")
    if avg_result:
        print(f"    CAGR:    {avg_result['cagr']:>+7.1%}")
        print(f"    Sharpe:  {avg_result['sharpe']:>7.2f}")
        print(f"    Max DD:  {avg_result['max_dd']:>7.1%}")

    # Summary
    print(f"\n  VERDICT:")
    delta_yr = ft_year_cagr - static_cagr_comp
    delta_qtr = ft_qtr_cagr - static_cagr_comp
    print(f"    Yearly factor timing vs static:    {delta_yr:>+.1%} CAGR difference")
    print(f"    Quarterly factor timing vs static: {delta_qtr:>+.1%} CAGR difference")

    if delta_yr > 0.01:
        print(f"    >> Factor timing IMPROVES returns by {delta_yr:.1%}/yr")
    elif delta_yr < -0.01:
        print(f"    >> Factor timing HURTS returns by {abs(delta_yr):.1%}/yr")
    else:
        print(f"    >> Factor timing is NEUTRAL (within 1%)")

    # Risk-adjusted: check max DD improvement
    print(f"\n  RISK COMPARISON (yearly factor-timed):")
    worst_dd_static = min(static_result["yearly"].get(y, {}).get("max_dd", 0) for y in range(2018, 2026) if y in static_result["yearly"])
    worst_dd_ft = min(yearly_results.get(y, {}).get("max_dd", 0) for y in range(2018, 2026) if y in yearly_results)
    print(f"    Worst annual DD (static):  {worst_dd_static:>+.1%}")
    print(f"    Worst annual DD (FT):      {worst_dd_ft:>+.1%}")

    # ═══════════════════════════════════════════════════════════════════════════
    # VARIANT: Defense-only factor timing
    # Only reduce momentum when UMD is crashing; otherwise stay at 85/15
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("VARIANT: DEFENSE-ONLY FACTOR TIMING")
    print("  (Only shift away from momentum when UMD is weak; otherwise static 85/15)")
    print("=" * 75)

    def get_defensive_allocation(signals, as_of_date):
        """Only go defensive when UMD is crashing. Otherwise stay 85/15."""
        sig = signals.loc[:as_of_date]
        if len(sig) < 60:
            return {"mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0}
        latest = sig.iloc[-1]

        # Only react to UMD weakness + market weakness combo
        umd_weak = latest.get("umd_weak", False)
        mktrf_weak = latest.get("mktrf_weak", False)

        if umd_weak and mktrf_weak:
            # Both momentum and market crashing: full defensive
            return {"mom_w": 0.50, "val_w": 0.25, "lv_w": 0.20, "sec_w": 0.05}
        elif umd_weak:
            # Momentum crashing but market OK: moderate defensive
            return {"mom_w": 0.65, "val_w": 0.20, "lv_w": 0.10, "sec_w": 0.05}
        else:
            # Default: stay aggressive
            return {"mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0}

    # Run defense-only yearly
    def_yearly_results = {}
    for year in range(2018, 2026):
        as_of = pd.Timestamp(f"{year-1}-12-31")
        alloc = get_defensive_allocation(signals, as_of)
        cfg = {
            "universe": "sp1500",
            "mom_w": alloc["mom_w"], "val_w": alloc["val_w"],
            "lv_w": alloc["lv_w"], "sec_w": alloc["sec_w"],
            "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
            "vol_scaling": True, "vol_target": 0.18,
            "gld_pct": 0.02, "vixm_pct": 0.02,
        }
        r = bt.run(f"{year}-01-01", f"{year}-12-31", cfg)
        if r:
            def_yearly_results[year] = {
                "alloc": alloc, "cagr": r["cagr"],
                "sharpe": r["sharpe"], "max_dd": r["max_dd"],
            }

    # Defense-only quarterly
    def_qtr_results = {}
    for q_start, q_end, _ in quarterly_periods:
        as_of = pd.Timestamp(q_start) - pd.Timedelta(days=1)
        alloc = get_defensive_allocation(signals, as_of)
        cfg = {
            "universe": "sp1500",
            "mom_w": alloc["mom_w"], "val_w": alloc["val_w"],
            "lv_w": alloc["lv_w"], "sec_w": alloc["sec_w"],
            "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
            "vol_scaling": True, "vol_target": 0.18,
            "gld_pct": 0.02, "vixm_pct": 0.02,
        }
        r = bt.run(q_start, q_end, cfg)
        if r and r["cagr"] is not None:
            def_qtr_results[q_start] = {
                "alloc": alloc,
                "return": r["final"] / 100000 - 1,
            }

    print(f"\n  {'Year':<6} {'Static':>8} {'Def-Year':>9} {'Def-Qtr':>9}  {'Alloc':<30}")
    print("  " + "-" * 70)
    for year in range(2018, 2026):
        static_yr = static_result["yearly"].get(year, {}).get("cagr", 0)
        dy = def_yearly_results.get(year, {}).get("cagr", 0)
        dq = compound_quarterly(def_qtr_results, year) if def_qtr_results else 0
        alloc = get_defensive_allocation(signals, pd.Timestamp(f"{year-1}-12-31"))
        alloc_str = f"M{alloc['mom_w']:.0%}/V{alloc['val_w']:.0%}/L{alloc['lv_w']:.0%}/S{alloc['sec_w']:.0%}"
        print(f"  {year:<6} {static_yr:>+7.0%}  {dy:>+7.0%}  {dq:>+7.0%}   {alloc_str}")

    def_yr_compound = 1.0
    def_qtr_compound = 1.0
    for year in range(2018, 2026):
        def_yr_compound *= (1 + def_yearly_results.get(year, {}).get("cagr", 0))
        def_qtr_compound *= (1 + compound_quarterly(def_qtr_results, year))

    def_yr_cagr = def_yr_compound ** (1/years) - 1
    def_qtr_cagr = def_qtr_compound ** (1/years) - 1

    print(f"\n  COMPOUNDED CAGR:")
    print(f"    Static v10.1:              {static_cagr_comp:>+7.1%}")
    print(f"    Defense-Only (Yearly):     {def_yr_cagr:>+7.1%}  ({def_yr_cagr - static_cagr_comp:>+.1%} vs static)")
    print(f"    Defense-Only (Quarterly):  {def_qtr_cagr:>+7.1%}  ({def_qtr_cagr - static_cagr_comp:>+.1%} vs static)")

    # ═══════════════════════════════════════════════════════════════════════════
    # VARIANT: Continuous factor scores (not binary)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("VARIANT: CONTINUOUS FACTOR SCORING")
    print("  (Smoothly adjust allocation based on factor z-scores)")
    print("=" * 75)

    def get_continuous_allocation(signals, ff, as_of_date):
        """Use z-scores of factor returns to smoothly adjust allocations."""
        ff_slice = ff.loc[:as_of_date]
        if len(ff_slice) < 252:
            return {"mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0}

        # Z-score of 60-day UMD return relative to its 2-year history
        umd_60d = ff_slice["umd"].rolling(60).sum()
        umd_mean = umd_60d.rolling(504).mean()
        umd_std = umd_60d.rolling(504).std()
        if pd.isna(umd_std.iloc[-1]) or umd_std.iloc[-1] < 0.001:
            return {"mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0}

        umd_z = (umd_60d.iloc[-1] - umd_mean.iloc[-1]) / umd_std.iloc[-1]

        # Clamp z-score to [-2, 2]
        umd_z = max(-2, min(2, umd_z))

        # Map z-score to momentum weight:
        # z=+2 -> mom_w=0.95 (full momentum), z=-2 -> mom_w=0.55 (reduced)
        mom_w = 0.85 + umd_z * 0.05  # range: 0.75 to 0.95
        mom_w = max(0.55, min(0.95, mom_w))
        val_w = 1.0 - mom_w

        # If UMD is very negative, add lowvol
        if umd_z < -1.0:
            lv_w = min(0.15, (-umd_z - 1.0) * 0.15)
            val_w -= lv_w / 2
            mom_w -= lv_w / 2
        else:
            lv_w = 0.0

        return {"mom_w": round(mom_w, 3), "val_w": round(val_w, 3),
                "lv_w": round(lv_w, 3), "sec_w": 0.0}

    cont_yearly_results = {}
    for year in range(2018, 2026):
        as_of = pd.Timestamp(f"{year-1}-12-31")
        alloc = get_continuous_allocation(signals, ff, as_of)
        cfg = {
            "universe": "sp1500",
            "mom_w": alloc["mom_w"], "val_w": alloc["val_w"],
            "lv_w": alloc["lv_w"], "sec_w": alloc["sec_w"],
            "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
            "vol_scaling": True, "vol_target": 0.18,
            "gld_pct": 0.02, "vixm_pct": 0.02,
        }
        r = bt.run(f"{year}-01-01", f"{year}-12-31", cfg)
        if r:
            cont_yearly_results[year] = {
                "alloc": alloc, "cagr": r["cagr"],
                "sharpe": r["sharpe"], "max_dd": r["max_dd"],
            }

    cont_qtr_results = {}
    for q_start, q_end, _ in quarterly_periods:
        as_of = pd.Timestamp(q_start) - pd.Timedelta(days=1)
        alloc = get_continuous_allocation(signals, ff, as_of)
        cfg = {
            "universe": "sp1500",
            "mom_w": alloc["mom_w"], "val_w": alloc["val_w"],
            "lv_w": alloc["lv_w"], "sec_w": alloc["sec_w"],
            "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
            "vol_scaling": True, "vol_target": 0.18,
            "gld_pct": 0.02, "vixm_pct": 0.02,
        }
        r = bt.run(q_start, q_end, cfg)
        if r and r["cagr"] is not None:
            cont_qtr_results[q_start] = {
                "alloc": alloc,
                "return": r["final"] / 100000 - 1,
            }

    print(f"\n  {'Year':<6} {'Static':>8} {'Cont-Yr':>9} {'Cont-Qtr':>9}  {'Alloc':<30}")
    print("  " + "-" * 70)
    for year in range(2018, 2026):
        static_yr = static_result["yearly"].get(year, {}).get("cagr", 0)
        cy = cont_yearly_results.get(year, {}).get("cagr", 0)
        cq = compound_quarterly(cont_qtr_results, year) if cont_qtr_results else 0
        alloc = get_continuous_allocation(signals, ff, pd.Timestamp(f"{year-1}-12-31"))
        alloc_str = f"M{alloc['mom_w']:.0%}/V{alloc['val_w']:.0%}/L{alloc['lv_w']:.0%}"
        print(f"  {year:<6} {static_yr:>+7.0%}  {cy:>+7.0%}  {cq:>+7.0%}   {alloc_str}")

    cont_yr_compound = 1.0
    cont_qtr_compound = 1.0
    for year in range(2018, 2026):
        cont_yr_compound *= (1 + cont_yearly_results.get(year, {}).get("cagr", 0))
        cont_qtr_compound *= (1 + compound_quarterly(cont_qtr_results, year))

    cont_yr_cagr = cont_yr_compound ** (1/years) - 1
    cont_qtr_cagr = cont_qtr_compound ** (1/years) - 1

    print(f"\n  COMPOUNDED CAGR:")
    print(f"    Static v10.1:                {static_cagr_comp:>+7.1%}")
    print(f"    Continuous (Yearly):         {cont_yr_cagr:>+7.1%}  ({cont_yr_cagr - static_cagr_comp:>+.1%} vs static)")
    print(f"    Continuous (Quarterly):      {cont_qtr_cagr:>+7.1%}  ({cont_qtr_cagr - static_cagr_comp:>+.1%} vs static)")

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 75)
    print("FINAL SUMMARY: ALL VARIANTS vs STATIC v10.1")
    print("=" * 75)
    print(f"\n  {'Variant':<35} {'CAGR':>7} {'vs Static':>10}")
    print("  " + "-" * 55)
    print(f"  {'Static v10.1 (85/15)':<35} {static_cagr_comp:>+6.1%} {'baseline':>10}")
    print(f"  {'Factor-Timed (Yearly)':<35} {ft_year_cagr:>+6.1%} {ft_year_cagr - static_cagr_comp:>+9.1%}")
    print(f"  {'Factor-Timed (Quarterly)':<35} {ft_qtr_cagr:>+6.1%} {ft_qtr_cagr - static_cagr_comp:>+9.1%}")
    print(f"  {'Defense-Only (Yearly)':<35} {def_yr_cagr:>+6.1%} {def_yr_cagr - static_cagr_comp:>+9.1%}")
    print(f"  {'Defense-Only (Quarterly)':<35} {def_qtr_cagr:>+6.1%} {def_qtr_cagr - static_cagr_comp:>+9.1%}")
    print(f"  {'Continuous Z-Score (Yearly)':<35} {cont_yr_cagr:>+6.1%} {cont_yr_cagr - static_cagr_comp:>+9.1%}")
    print(f"  {'Continuous Z-Score (Quarterly)':<35} {cont_qtr_cagr:>+6.1%} {cont_qtr_cagr - static_cagr_comp:>+9.1%}")
    print(f"  {'Avg Alloc Full-Period':<35} {avg_result['cagr']:>+6.1%} {avg_result['cagr'] - static_result['cagr']:>+9.1%}")

    print("\n" + "=" * 75)
