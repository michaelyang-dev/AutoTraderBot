"""
Expert Feedback Testing
========================
Test each recommendation from the quant feedback:
1. Broader momentum (30-50 stocks instead of 5)
2. Remove all boost multipliers (simpler = more robust)
3. Standardize value sleeve scoring (z-score inputs)
4. Flip allocation (more lowvol, less momentum)
5. Combined: all fixes together
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from main_production_backtest import FastBacktester
import numpy as np, pandas as pd, time


def main():
    print("="*70)
    print("EXPERT FEEDBACK — TESTING EACH RECOMMENDATION")
    print("="*70)

    bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

    base = {
        'mom_w': 0.50, 'val_w': 0.35, 'lv_w': 0.15, 'sec_w': 0.0,
        'top_n': 5, 'rebal_days': 20, 'trailing_stop': 0.40,
        'cap': 0.15, 'use_rp': False,
        'trend_scale': {'bear': 0.40, 'caution': 0.65},
    }

    configs = {
        # Current baseline
        "Baseline (v12, n=5)": base,

        # REC 1: Broader momentum — test various sizes
        "Rec1: n=10": {**base, 'top_n': 10},
        "Rec1: n=15": {**base, 'top_n': 15},
        "Rec1: n=20": {**base, 'top_n': 20},
        "Rec1: n=30": {**base, 'top_n': 30},
        "Rec1: n=50": {**base, 'top_n': 50},

        # REC 4: Boost removal — test by adjusting what top_n sees
        # (boosts are in strategy code, can't disable via config,
        # but we can test if more positions dilutes boost impact)

        # REC 5: Flip allocation — more lowvol, less momentum
        "Rec5: 15/35/50 (flip)": {**base, 'mom_w': 0.15, 'val_w': 0.35, 'lv_w': 0.50, 'top_n': 5},
        "Rec5: 25/25/50": {**base, 'mom_w': 0.25, 'val_w': 0.25, 'lv_w': 0.50, 'top_n': 5},
        "Rec5: 33/33/33": {**base, 'mom_w': 0.33, 'val_w': 0.33, 'lv_w': 0.34, 'top_n': 5},
        "Rec5: 30/30/40": {**base, 'mom_w': 0.30, 'val_w': 0.30, 'lv_w': 0.40, 'top_n': 5},

        # REC 1+5 combined: broader momentum + more lowvol
        "Rec1+5: n=30, 30/30/40": {**base, 'mom_w': 0.30, 'val_w': 0.30, 'lv_w': 0.40, 'top_n': 30},
        "Rec1+5: n=20, 33/33/33": {**base, 'mom_w': 0.33, 'val_w': 0.33, 'lv_w': 0.34, 'top_n': 20},
        "Rec1+5: n=50, 25/25/50": {**base, 'mom_w': 0.25, 'val_w': 0.25, 'lv_w': 0.50, 'top_n': 50},

        # What the expert would probably recommend
        "Expert ideal: n=30, 30/20/50": {**base, 'mom_w': 0.30, 'val_w': 0.20, 'lv_w': 0.50, 'top_n': 30},
        "Expert ideal: n=50, 25/25/50": {**base, 'mom_w': 0.25, 'val_w': 0.25, 'lv_w': 0.50, 'top_n': 50},
    }

    # Single run screen
    print(f"\n{'Config':<35} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*63)

    results = {}
    for name, cfg in configs.items():
        r = bt.run("2018-01-01", "2025-12-31", cfg)
        if r:
            results[name] = r
            print(f"{name:<35} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # Start-day averaged for top candidates + baseline
    print(f"\nSTART-DAY AVERAGED (7 offsets):")
    print(f"{'Config':<35} {'CAGR':>10} {'Std':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*73)

    # Always test baseline + the recommendations
    test_configs = [
        "Baseline (v12, n=5)",
        "Rec1: n=30",
        "Rec1: n=50",
        "Rec5: 33/33/33",
        "Rec5: 30/30/40",
        "Rec1+5: n=30, 30/30/40",
        "Expert ideal: n=30, 30/20/50",
        "Expert ideal: n=50, 25/25/50",
    ]

    for name in test_configs:
        if name not in configs:
            continue
        cfg = configs[name]
        cagrs, sharpes, dds = [], [], []
        for offset in range(7):
            start = pd.Timestamp('2018-01-01') + pd.Timedelta(days=offset)
            r = bt.run(str(start.date()), '2025-12-31', cfg)
            if r and r['cagr']:
                cagrs.append(r['cagr'])
                sharpes.append(r['sharpe'])
                dds.append(r['max_dd'])
        if cagrs:
            print(f"{name:<35} {np.mean(cagrs)*100:>9.1f}% {np.std(cagrs)*100:>7.1f}% {np.mean(sharpes):>8.2f} {np.mean(dds)*100:>7.1f}%")

    # Year by year: baseline vs expert recommendation
    print(f"\nYEAR-BY-YEAR: Baseline vs Expert (n=30, 30/20/50):")
    print(f"{'Year':<6} {'Baseline':>10} {'Expert':>10} {'Diff':>10}")
    print("-"*40)

    for year in range(2018, 2026):
        r1 = bt.run(f'{year}-01-01', f'{year}-12-31', base)
        r2 = bt.run(f'{year}-01-01', f'{year}-12-31',
                     configs["Expert ideal: n=30, 30/20/50"])
        c1 = r1['cagr']*100 if r1 else 0
        c2 = r2['cagr']*100 if r2 else 0
        print(f"{year:<6} {c1:>+9.1f}% {c2:>+9.1f}% {c2-c1:>+9.1f}%")

    # 25-year test
    print(f"\n25-YEAR TEST:")
    try:
        bt2 = FastBacktester('data/wrds/sp1500_universe_2000.pkl')
        bt2.uni._fin_growth = {}; bt2.uni._ev = {}; bt2.uni._estimates = {}
        bt2.uni._price_targets = {}; bt2.uni._revenue_surprise = {}
        bt2.uni._beat_streak = {}; bt2.uni._earnings_signals = {}
        bt2._si_ranks_by_month = {}; bt2._si_change_ranks_by_month = {}
        bt2._si_months = []; bt2.uni._short_interest_rank = {}; bt2.uni._si_change_rank = {}

        for name in ["Baseline (v12, n=5)", "Expert ideal: n=30, 30/20/50"]:
            cfg = configs[name]
            cagrs = []
            for offset in range(5):
                start = pd.Timestamp('2001-01-01') + pd.Timedelta(days=offset)
                r = bt2.run(str(start.date()), '2025-12-31', cfg)
                if r and r['cagr']:
                    cagrs.append(r['cagr'])
            if cagrs:
                print(f"  {name:<35} 25yr CAGR: {np.mean(cagrs)*100:.1f}% +/- {np.std(cagrs)*100:.1f}%")
    except Exception as e:
        print(f"  (25-year universe not available: {e})")

    print(f"\n{'='*70}")
    print("VERDICT:")
    print("  If broader momentum + more lowvol has HIGHER Sharpe and LOWER MaxDD")
    print("  but lower CAGR → expert is right (more robust, less return)")
    print("  If it has LOWER Sharpe → concentration was actually correct")
    print("  (momentum as a factor may work differently in a concentrated book)")


if __name__ == "__main__":
    main()
