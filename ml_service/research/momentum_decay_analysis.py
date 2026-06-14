"""
Momentum Decay Analysis — Go/No-Go Check for ML Exit Timing
==============================================================
Before building any ML model, check whether momentum decay patterns
vary by regime. If decay looks the same everywhere → ML can't help.
If decay differs by regime → ML has something to learn.

This is pure statistics, no ML. Takes ~30 minutes to run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, COST_BPS,
)
import numpy as np, pandas as pd, time


def main():
    print("="*70)
    print("MOMENTUM DECAY ANALYSIS — GO/NO-GO FOR ML EXIT TIMING")
    print("="*70)
    print()
    print("Question: Do momentum positions decay differently by regime?")
    print("If YES → ML can learn when to exit early/late")
    print("If NO → fixed 20-day hold is already optimal, stop here")

    # Load universe
    print("\n[1] Loading data...")
    t0 = time.time()
    bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
    bt.uni.get_sp500 = bt._get_sp1500
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load FF and FRED for regime features
    from wrds_data_provider import WRDSDataProvider
    provider = WRDSDataProvider()
    ff = provider.fama_french
    fred = provider.fred_rates

    trading_dates = sorted(bt.prices.index)
    test_dates = [d for d in trading_dates
                  if pd.Timestamp("2016-07-01") <= d <= pd.Timestamp("2025-12-31")]

    # ═══════════════════════════════════════════════════════════
    # Step 1: Generate position entries from momentum strategy
    # ═══════════════════════════════════════════════════════════
    print("\n[2] Generating momentum position entries...")

    entries = []  # {date, symbol, entry_score, entry_price}

    for day_idx, date in enumerate(test_dates):
        if day_idx % 20 != 0:  # every 20 days (rebalance)
            continue

        t1 = strategy1_momentum_reversal(date, bt.uni, day_idx,
                                          top_n=5, rebal_days=20)
        if not t1:
            continue

        for sym, weight in t1.items():
            px = bt.prices.loc[date].get(sym) if date in bt.prices.index else None
            if px and pd.notna(px) and px > 0:
                # Compute regime features at entry
                fdate = bt.features_by_date.get(date, {})
                above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
                total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
                breadth = above / max(total_f, 1)

                umd_val = np.nan
                if date in ff.index and 'umd' in ff.columns:
                    umd_20 = ff['umd'].loc[:date].tail(20).sum()
                    umd_val = umd_20

                sf = fdate.get(sym, {})
                vol = sf.get('vol_20d', np.nan)

                entries.append({
                    'date': date,
                    'symbol': sym,
                    'weight': weight,
                    'entry_price': px,
                    'breadth': breadth,
                    'umd_20d': umd_val,
                    'vol_20d': vol,
                    'entry_momentum': sf.get('ret_252d', 0) - sf.get('ret_20d', 0),
                })

    print(f"  Generated {len(entries)} position entries")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Compute forward return curves for each entry
    # ═══════════════════════════════════════════════════════════
    print("\n[3] Computing forward return curves...")

    horizons = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40]
    results = []

    for entry in entries:
        date = entry['date']
        sym = entry['symbol']
        entry_px = entry['entry_price']

        future = bt.prices.index[bt.prices.index > date]
        if len(future) < max(horizons) + 5:
            continue

        fwd_rets = {}
        for h in horizons:
            if h < len(future):
                fwd_px = bt.prices.loc[future[h]].get(sym)
                if fwd_px and pd.notna(fwd_px) and fwd_px > 0:
                    fwd_rets[f'fwd_{h}d'] = (fwd_px - entry_px) / entry_px

        if len(fwd_rets) >= 8:
            results.append({**entry, **fwd_rets})

    df = pd.DataFrame(results)
    print(f"  {len(df)} entries with full forward return curves")

    # ═══════════════════════════════════════════════════════════
    # Step 3: Split by regime and compare decay curves
    # ═══════════════════════════════════════════════════════════
    print("\n[4] DECAY CURVES BY REGIME")
    print("="*60)

    # Regime 1: High vs low breadth
    median_breadth = df['breadth'].median()
    high_breadth = df[df['breadth'] >= median_breadth]
    low_breadth = df[df['breadth'] < median_breadth]

    print(f"\n  A. BREADTH REGIME (median={median_breadth:.2f})")
    print(f"  {'Horizon':<10}", end="")
    print(f"{'High Breadth':>15} {'Low Breadth':>15} {'Diff':>12}")
    print(f"  {'-'*55}")

    decay_diffs_breadth = []
    for h in horizons:
        col = f'fwd_{h}d'
        if col in df.columns:
            hi = high_breadth[col].mean() * 100
            lo = low_breadth[col].mean() * 100
            diff = hi - lo
            decay_diffs_breadth.append(abs(diff))
            print(f"  {h:>3}d       {hi:>+13.2f}% {lo:>+13.2f}% {diff:>+10.2f}pp")

    # Regime 2: High vs low volatility
    median_vol = df['vol_20d'].dropna().median()
    high_vol = df[df['vol_20d'] >= median_vol]
    low_vol = df[df['vol_20d'] < median_vol]

    print(f"\n  B. VOLATILITY REGIME (median={median_vol:.3f})")
    print(f"  {'Horizon':<10}", end="")
    print(f"{'High Vol':>15} {'Low Vol':>15} {'Diff':>12}")
    print(f"  {'-'*55}")

    decay_diffs_vol = []
    for h in horizons:
        col = f'fwd_{h}d'
        if col in df.columns:
            hi = high_vol[col].mean() * 100
            lo = low_vol[col].mean() * 100
            diff = hi - lo
            decay_diffs_vol.append(abs(diff))
            print(f"  {h:>3}d       {hi:>+13.2f}% {lo:>+13.2f}% {diff:>+10.2f}pp")

    # Regime 3: Strong vs weak entry momentum
    median_mom = df['entry_momentum'].median()
    strong_mom = df[df['entry_momentum'] >= median_mom]
    weak_mom = df[df['entry_momentum'] < median_mom]

    print(f"\n  C. ENTRY MOMENTUM STRENGTH (median={median_mom:.3f})")
    print(f"  {'Horizon':<10}", end="")
    print(f"{'Strong Mom':>15} {'Weak Mom':>15} {'Diff':>12}")
    print(f"  {'-'*55}")

    decay_diffs_mom = []
    for h in horizons:
        col = f'fwd_{h}d'
        if col in df.columns:
            hi = strong_mom[col].mean() * 100
            lo = weak_mom[col].mean() * 100
            diff = hi - lo
            decay_diffs_mom.append(abs(diff))
            print(f"  {h:>3}d       {hi:>+13.2f}% {lo:>+13.2f}% {diff:>+10.2f}pp")

    # Regime 4: UMD crash vs normal
    umd_thresh = -0.03
    umd_stress = df[df['umd_20d'] < umd_thresh]
    umd_normal = df[df['umd_20d'] >= umd_thresh]

    print(f"\n  D. UMD REGIME (stress threshold={umd_thresh})")
    print(f"  {'Horizon':<10}", end="")
    print(f"{'Normal':>15} {'UMD Stress':>15} {'Diff':>12}")
    print(f"  {'-'*55}")

    decay_diffs_umd = []
    for h in horizons:
        col = f'fwd_{h}d'
        if col in df.columns:
            norm = umd_normal[col].mean() * 100
            stress = umd_stress[col].mean() * 100 if len(umd_stress) > 10 else np.nan
            diff = norm - stress if pd.notna(stress) else np.nan
            decay_diffs_umd.append(abs(diff) if pd.notna(diff) else 0)
            stress_str = f"{stress:>+13.2f}%" if pd.notna(stress) else f"{'N/A':>14}"
            diff_str = f"{diff:>+10.2f}pp" if pd.notna(diff) else f"{'N/A':>11}"
            print(f"  {h:>3}d       {norm:>+13.2f}% {stress_str} {diff_str}")

    # ═══════════════════════════════════════════════════════════
    # Step 4: Optimal holding period by regime
    # ═══════════════════════════════════════════════════════════
    print(f"\n[5] OPTIMAL HOLDING PERIOD BY REGIME")
    print("="*60)

    for label, subset in [("All", df), ("High breadth", high_breadth),
                           ("Low breadth", low_breadth), ("High vol", high_vol),
                           ("Low vol", low_vol), ("Strong mom", strong_mom),
                           ("Weak mom", weak_mom)]:
        best_h = 0
        best_ret = -999
        for h in horizons:
            col = f'fwd_{h}d'
            if col in subset.columns:
                avg = subset[col].mean()
                if avg > best_ret:
                    best_ret = avg
                    best_h = h
        # Also compute annualized return per day (efficiency)
        daily_rets = {}
        for h in horizons:
            col = f'fwd_{h}d'
            if col in subset.columns:
                avg = subset[col].mean()
                daily_rets[h] = avg / h * 252  # annualized daily rate
        best_efficient_h = max(daily_rets, key=daily_rets.get) if daily_rets else 0
        print(f"  {label:<15} Best total: {best_h:>3}d ({best_ret*100:>+.2f}%)  "
              f"Best daily rate: {best_efficient_h:>3}d ({daily_rets.get(best_efficient_h,0)*100:>+.1f}% ann)")

    # ═══════════════════════════════════════════════════════════
    # Step 5: GO / NO-GO DECISION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("GO / NO-GO DECISION")
    print(f"{'='*60}")

    avg_breadth_diff = np.mean(decay_diffs_breadth) if decay_diffs_breadth else 0
    avg_vol_diff = np.mean(decay_diffs_vol) if decay_diffs_vol else 0
    avg_mom_diff = np.mean(decay_diffs_mom) if decay_diffs_mom else 0
    avg_umd_diff = np.mean(decay_diffs_umd) if decay_diffs_umd else 0

    print(f"\n  Average absolute difference in forward returns by regime:")
    print(f"    Breadth regime:   {avg_breadth_diff:.2f}pp")
    print(f"    Volatility regime: {avg_vol_diff:.2f}pp")
    print(f"    Momentum strength: {avg_mom_diff:.2f}pp")
    print(f"    UMD stress:        {avg_umd_diff:.2f}pp")

    max_diff = max(avg_breadth_diff, avg_vol_diff, avg_mom_diff, avg_umd_diff)

    if max_diff > 3.0:
        print(f"\n  ✓ GO — Strong regime-dependent decay (max diff {max_diff:.1f}pp)")
        print(f"    ML has potential to learn different exit timing per regime.")
        print(f"    Proceed to Phase 2 (Application A: Adaptive Exit).")
    elif max_diff > 1.5:
        print(f"\n  ~ MARGINAL — Moderate regime-dependent decay (max diff {max_diff:.1f}pp)")
        print(f"    ML might find something but improvement will be small.")
        print(f"    Proceed cautiously — expect marginal results.")
    else:
        print(f"\n  ✗ NO-GO — Decay curves look similar across regimes (max diff {max_diff:.1f}pp)")
        print(f"    Fixed 20-day hold is already near-optimal.")
        print(f"    Do NOT invest in ML exit timing — it will overfit.")


if __name__ == "__main__":
    main()
