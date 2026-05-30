import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from fast_backtest import FastBacktester
import numpy as np, pandas as pd

bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')
bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}

results = []
count = 0

for mom_w in [0.35, 0.40, 0.45, 0.50, 0.55]:
    for lv_w in [0.15, 0.20, 0.25, 0.30, 0.35]:
        val_w = round(1.0 - mom_w - lv_w, 2)
        if val_w < 0.10 or val_w > 0.45:
            continue
        for tn in [5, 6, 8]:
            for rd in [15, 20]:
                for stop in [None, 0.35, 0.40]:
                    for cap in [0.15, 0.25]:
                        for bear in [0.30, 0.40, 0.50]:
                            count += 1
                            cfg = {
                                'mom_w': mom_w, 'val_w': val_w, 'lv_w': lv_w, 'sec_w': 0.0,
                                'top_n': tn, 'rebal_days': rd, 'trailing_stop': stop,
                                'cap': cap, 'use_rp': False,
                                'trend_scale': {'bear': bear, 'caution': min(bear + 0.25, 0.90)},
                            }
                            r = bt.run("2018-01-01", "2025-12-31", cfg)
                            if r and r['cagr'] and r['cagr'] > 0.17:
                                results.append({
                                    'mom_w': mom_w, 'val_w': val_w, 'lv_w': lv_w,
                                    'top_n': tn, 'rebal_days': rd,
                                    'stop': stop, 'cap': cap, 'bear': bear,
                                    'cagr': r['cagr'], 'sharpe': r['sharpe'], 'max_dd': r['max_dd'],
                                })
                            if count % 100 == 0:
                                print(f"  ...tested {count} configs, {len(results)} promising", flush=True)

print(f"Tested {count} configs, {len(results)} with CAGR > 17%")

results.sort(key=lambda x: x['cagr'] / abs(x['max_dd']), reverse=True)

print(f"\nTOP 20 BY CAGR/DRAWDOWN RATIO:")
hdr = f"{'#':<3} {'Mom':>4} {'Val':>4} {'LV':>4} {'N':>3} {'RD':>3} {'Stop':>5} {'Cap':>4} {'Bear':>5} {'CAGR':>7} {'Shrp':>5} {'MaxDD':>7} {'C/DD':>5}"
print(hdr)
print("-" * len(hdr))
for i, r in enumerate(results[:20]):
    stop_str = f"{r['stop']:.2f}" if r['stop'] else "None"
    ratio = r['cagr'] / abs(r['max_dd'])
    print(f"{i+1:<3} {r['mom_w']:>4.2f} {r['val_w']:>4.2f} {r['lv_w']:>4.2f} {r['top_n']:>3} {r['rebal_days']:>3} {stop_str:>5} {r['cap']:>4.2f} {r['bear']:>5.2f} {r['cagr']*100:>6.1f}% {r['sharpe']:>5.2f} {r['max_dd']*100:>6.1f}% {ratio:>5.2f}")

# Start-day average top 5
print(f"\n{'='*70}")
print(f"START-DAY AVERAGED (7 offsets) - TOP 5:")
print(f"{'='*70}")

for i, r in enumerate(results[:5]):
    cfg = {
        'mom_w': r['mom_w'], 'val_w': r['val_w'], 'lv_w': r['lv_w'], 'sec_w': 0.0,
        'top_n': r['top_n'], 'rebal_days': r['rebal_days'], 'trailing_stop': r['stop'],
        'cap': r['cap'], 'use_rp': False,
        'trend_scale': {'bear': r['bear'], 'caution': min(r['bear'] + 0.25, 0.90)},
    }
    cagrs, sharpes, dds = [], [], []
    for offset in range(7):
        start = pd.Timestamp('2018-01-01') + pd.Timedelta(days=offset)
        res = bt.run(str(start.date()), '2025-12-31', cfg)
        if res and res['cagr']:
            cagrs.append(res['cagr'])
            sharpes.append(res['sharpe'])
            dds.append(res['max_dd'])

    stop_str = f"{r['stop']:.2f}" if r['stop'] else "None"
    label = f"{int(r['mom_w']*100)}/{int(r['val_w']*100)}/{int(r['lv_w']*100)} n{r['top_n']} rd{r['rebal_days']} st{stop_str} cap{int(r['cap']*100)}% bear{int(r['bear']*100)}%"
    print(f"\n  #{i+1}: {label}")
    print(f"      CAGR: {np.mean(cagrs)*100:.1f}% +/- {np.std(cagrs)*100:.1f}%  Sharpe: {np.mean(sharpes):.2f}  MaxDD: {np.mean(dds)*100:.1f}%")

    years_str = ""
    for year in range(2018, 2026):
        res = bt.run(f'{year}-01-01', f'{year}-12-31', cfg)
        if res:
            years_str += f" {year}:{res['cagr']*100:+.0f}%"
    print(f"      Years:{years_str}")

print("\nDONE")
