"""
Robustness sweep for the market-neutral momentum candidate (config 3: perp-restricted
shorts + funding, no vol-target). The question: is the 47% 2023+ / Sharpe 1.17 edge BROAD
across reasonable parameter choices, or did we curve-fit to mom_30 / weekly / quintile?

Sweeps one axis at a time around the base (lookback 30, rebal 7, top_frac 0.2, top_uni 100)
and reports the OUT-OF-SAMPLE 2023+ CAGR / Sharpe for each. A real edge is a broad plateau;
a curve-fit is a lonely spike. Run:  python crypto/backtest/robustness.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from momentum_strategy import run, stats

BASE = dict(perp_short=True, use_funding=True, vol_target=False,
            lookback=30, rebal=7, top_frac=0.2, top_uni=100)


def line(label, **over):
    kw = dict(BASE); kw.update(over)
    d = run(long_short=True, **kw)
    cg, vol, sh, md = stats(d)
    cg23, _, sh23, md23 = stats(d, "2023-01-01")
    print(f"    {label:<22}{cg23*100:>10.1f}%{sh23:>9.2f}{md23*100:>9.1f}%{cg*100:>11.1f}%{sh:>9.2f}", flush=True)


if __name__ == "__main__":
    print("=" * 86)
    print("ROBUSTNESS SWEEP — market-neutral momentum (base = lookback30/rebal7/quintile/top100)")
    print("=" * 86)
    print(f"    {'variant':<22}{'2023+ CAGR':>10}{'Sh23':>9}{'DD23':>9}{'FULL CAGR':>11}{'Sh':>9}")
    print("  -- momentum lookback (days) --")
    for lb in [15, 20, 30, 45, 60, 90]:
        line(("* " if lb == 30 else "  ") + f"lookback {lb}", lookback=lb)
    print("  -- rebalance frequency (days) --")
    for rb in [3, 5, 7, 14, 21, 30]:
        line(("* " if rb == 7 else "  ") + f"rebal {rb}", rebal=rb)
    print("  -- long/short quantile (top_frac) --")
    for tf in [0.10, 0.15, 0.20, 0.30, 0.40]:
        line(("* " if tf == 0.20 else "  ") + f"top_frac {tf:.2f}", top_frac=tf)
    print("  -- tradable universe size (top-N by mcap) --")
    for tu in [40, 60, 100, 150, 200]:
        line(("* " if tu == 100 else "  ") + f"top_uni {tu}", top_uni=tu)
    print("\n  * = base config.  Edge is real if 2023+ stays positive across the board (broad plateau),")
    print("  not just a spike at the base. Watch Sharpe23 — that's the out-of-sample risk-adjusted read.")
