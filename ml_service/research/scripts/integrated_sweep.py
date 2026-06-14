"""
Integrated Sweep: Test scoring variants WITHIN the full production framework.
=============================================================================
Uses the original FastBacktester.run() but swaps out strategy1_momentum_reversal
with scoring variants. This preserves bear sector tilt, signal-weighted allocation,
regime blending, and all other production features.
"""

import numpy as np
import pandas as pd
import time
import logging
import sys
from pathlib import Path
from functools import partial

from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, INITIAL_CASH, COST_BPS,
)
from scoring_variants import strategy1_skip_month, strategy1_risk_adj

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")


def monkey_patch_run(bt, mom_func, start, end, config):
    """
    Run the ORIGINAL FastBacktester.run() but with a different momentum function.
    We temporarily replace strategy1_momentum_reversal in the module namespace.
    """
    import strategies.multi_strategy_engine as mse
    original = mse.strategy1_momentum_reversal
    mse.strategy1_momentum_reversal = mom_func

    # Also need to update the reference in main_production_backtest module
    import main_production_backtest as fb
    fb_original = fb.strategy1_momentum_reversal
    fb.strategy1_momentum_reversal = mom_func

    try:
        result = bt.run(start, end, config)
    finally:
        mse.strategy1_momentum_reversal = original
        fb.strategy1_momentum_reversal = fb_original

    return result


def fmt_yearly(yearly, oos_start=2022):
    parts = []
    for y in sorted(yearly.keys()):
        c = yearly[y]["cagr"]
        marker = "*" if y >= oos_start else " "
        parts.append(f"{y}:{c:+.0%}{marker}")
    return " ".join(parts)


def run_config(bt, name, mom_func, config, full_start="2018-01-01", full_end="2025-04-30",
               oos_start="2022-01-01"):
    t0 = time.time()
    full = monkey_patch_run(bt, mom_func, full_start, full_end, config)
    oos = monkey_patch_run(bt, mom_func, oos_start, full_end, config)
    elapsed = time.time() - t0

    if not full or not oos:
        print(f"  {name:<60} FAILED", flush=True)
        return None

    print(f"  {name:<60} Full: {full['cagr']:>+5.1%} S={full['sharpe']:.2f} DD={full['max_dd']:.0%}"
          f"  |  OOS: {oos['cagr']:>+5.1%} S={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}"
          f"  [{elapsed:.0f}s]", flush=True)
    print(f"    {fmt_yearly(full['yearly'])}", flush=True)
    return {"full": full, "oos": oos, "name": name}


if __name__ == "__main__":
    print("Loading data...", flush=True)
    bt = FastBacktester()
    print("Data loaded.\n", flush=True)

    results = []

    # ═══════════════════════════════════════════════════════════════
    # BASELINE: Current production configs (using original scoring)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 130, flush=True)
    print("BASELINES (original production scoring)", flush=True)
    print("=" * 130, flush=True)

    prod_func = strategy1_momentum_reversal

    # Current production: 85/15 mom/val, top8, r10, trail25, GLD+VIXM
    r = run_config(bt, "PROD: 85/15 t8 r10 trail25 G+V", prod_func, {
        "universe": "sp1500", "mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25, "cap": 0.15,
        "gld_pct": 0.02, "vixm_pct": 0.02,
    })
    if r: results.append(r)

    # Previous best from earlier sweep: 100/0 top10 r5 trail25 nohedge
    r = run_config(bt, "PREV-BEST: 100/0 t10 r5 trail25 nohedge", prod_func, {
        "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0, "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 10, "rebal_days": 5, "trailing_stop": 0.25, "cap": 0.15,
    })
    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SKIP-MONTH MOMENTUM (12-1) with full production framework
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}", flush=True)
    print("SKIP-MONTH (12-1) MOMENTUM — full production framework", flush=True)
    print("=" * 130, flush=True)

    # Test trend filters with skip-month
    for tf in ["sma50", "sma200", "either"]:
        for qual in [True, False]:
            func = partial(strategy1_skip_month, trend_filter=tf, quality_boosts=qual)
            q_label = "qual" if qual else "noQ"
            for top_n in [8, 10, 12]:
                for rebal in [5, 10]:
                    cfg = {
                        "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0,
                        "lv_w": 0.0, "sec_w": 0.0,
                        "top_n": top_n, "rebal_days": rebal,
                        "trailing_stop": 0.25, "cap": 0.15,
                    }
                    name = f"skip12-1 {tf} {q_label} t{top_n} r{rebal}"
                    r = run_config(bt, name, func, cfg)
                    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SKIP-MONTH with regime blending (add back value sleeve)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}", flush=True)
    print("SKIP-MONTH with REGIME BLENDING (value sleeve)", flush=True)
    print("=" * 130, flush=True)

    for tf in ["sma50", "sma200"]:
        func = partial(strategy1_skip_month, trend_filter=tf, quality_boosts=True)
        for mom_w, val_w in [(1.0, 0.0), (0.85, 0.15), (0.70, 0.30)]:
            for top_n in [8, 10]:
                for rebal in [5, 10]:
                    cfg = {
                        "universe": "sp1500", "mom_w": mom_w, "val_w": val_w,
                        "lv_w": 0.0, "sec_w": 0.0,
                        "top_n": top_n, "rebal_days": rebal,
                        "trailing_stop": 0.25, "cap": 0.15,
                    }
                    blend_label = f"{int(mom_w*100)}/{int(val_w*100)}"
                    name = f"skip12-1 {tf} {blend_label} t{top_n} r{rebal}"
                    r = run_config(bt, name, func, cfg)
                    if r: results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SKIP-MONTH with different trailing stops and hedges
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}", flush=True)
    print("SKIP-MONTH: trailing stops and hedges", flush=True)
    print("=" * 130, flush=True)

    func_200 = partial(strategy1_skip_month, trend_filter="sma200", quality_boosts=True)
    func_50 = partial(strategy1_skip_month, trend_filter="sma50", quality_boosts=True)

    for func, tf_name in [(func_200, "sma200"), (func_50, "sma50")]:
        for trail in [None, 0.20, 0.25, 0.30]:
            trail_label = f"trail{int(trail*100)}" if trail else "notrail"
            for gld, vixm in [(0, 0), (0.02, 0.02), (0.05, 0.03)]:
                hedge_label = f"G{int(gld*100)}V{int(vixm*100)}" if gld > 0 else "nohedge"
                cfg = {
                    "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0,
                    "lv_w": 0.0, "sec_w": 0.0,
                    "top_n": 10, "rebal_days": 5,
                    "trailing_stop": trail, "cap": 0.15,
                    "gld_pct": gld, "vixm_pct": vixm,
                }
                name = f"skip12-1 {tf_name} t10 r5 {trail_label} {hedge_label}"
                r = run_config(bt, name, func, cfg)
                if r: results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # RISK-ADJUSTED MOMENTUM with production framework
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}", flush=True)
    print("RISK-ADJUSTED MOMENTUM", flush=True)
    print("=" * 130, flush=True)

    for tf in ["sma50", "sma200"]:
        func = partial(strategy1_risk_adj, trend_filter=tf, quality_boosts=True)
        for top_n in [8, 10, 12]:
            cfg = {
                "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0,
                "lv_w": 0.0, "sec_w": 0.0,
                "top_n": top_n, "rebal_days": 5,
                "trailing_stop": 0.25, "cap": 0.15,
            }
            name = f"risk_adj {tf} t{top_n} r5"
            r = run_config(bt, name, func, cfg)
            if r: results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # VOL SCALING combinations
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}", flush=True)
    print("VOL SCALING", flush=True)
    print("=" * 130, flush=True)

    for func, scoring_name in [(func_200, "skip_sma200"), (func_50, "skip_sma50"),
                                (prod_func, "production")]:
        for vol_tgt in [0.15, 0.18, 0.22]:
            cfg = {
                "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0,
                "lv_w": 0.0, "sec_w": 0.0,
                "top_n": 10, "rebal_days": 5,
                "trailing_stop": 0.25, "cap": 0.15,
                "vol_scaling": True, "vol_target": vol_tgt,
            }
            name = f"{scoring_name} t10 r5 volscale={vol_tgt:.0%}"
            r = run_config(bt, name, func, cfg)
            if r: results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 130}", flush=True)
    print("TOP 15 BY OOS CAGR", flush=True)
    print("=" * 130, flush=True)

    by_oos = sorted(results, key=lambda x: x["oos"]["cagr"], reverse=True)
    for i, r in enumerate(by_oos[:15]):
        oos = r["oos"]
        full = r["full"]
        print(f"  #{i+1}: {r['name']:<60}", flush=True)
        print(f"       Full: CAGR={full['cagr']:+.1%} Sharpe={full['sharpe']:.2f} DD={full['max_dd']:.0%}", flush=True)
        print(f"       OOS:  CAGR={oos['cagr']:+.1%} Sharpe={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}", flush=True)
        print(f"       {fmt_yearly(full['yearly'])}", flush=True)

    print(f"\n{'=' * 130}", flush=True)
    print("TOP 15 BY OOS SHARPE", flush=True)
    print("=" * 130, flush=True)

    by_sharpe = sorted(results, key=lambda x: x["oos"]["sharpe"], reverse=True)
    for i, r in enumerate(by_sharpe[:15]):
        oos = r["oos"]
        full = r["full"]
        print(f"  #{i+1}: {r['name']:<60}", flush=True)
        print(f"       Full: CAGR={full['cagr']:+.1%} Sharpe={full['sharpe']:.2f} DD={full['max_dd']:.0%}", flush=True)
        print(f"       OOS:  CAGR={oos['cagr']:+.1%} Sharpe={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}", flush=True)

    # Also show best Sharpe-CAGR balanced (Sharpe > 0.8, highest CAGR)
    print(f"\n{'=' * 130}", flush=True)
    print("BEST BALANCED (OOS Sharpe > 0.7, ranked by OOS CAGR)", flush=True)
    print("=" * 130, flush=True)

    balanced = [r for r in results if r["oos"]["sharpe"] > 0.7]
    balanced.sort(key=lambda x: x["oos"]["cagr"], reverse=True)
    for i, r in enumerate(balanced[:10]):
        oos = r["oos"]
        full = r["full"]
        print(f"  #{i+1}: {r['name']:<60}", flush=True)
        print(f"       Full: CAGR={full['cagr']:+.1%} Sharpe={full['sharpe']:.2f} DD={full['max_dd']:.0%}", flush=True)
        print(f"       OOS:  CAGR={oos['cagr']:+.1%} Sharpe={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}", flush=True)
        print(f"       {fmt_yearly(full['yearly'])}", flush=True)

    print(f"\nTotal configs tested: {len(results)}", flush=True)
