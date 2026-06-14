"""
Locate the source of the 25.4% v12 claim.
Runs the v12 config under 4 data conditions to see which reproduces 25.4%:
  A. DEPLOYED (enhanced OFF, SI OFF)  <- what live actually uses
  B. enhanced ON, SI OFF
  C. enhanced OFF, SI ON
  D. enhanced ON, SI ON (everything)
Same v12 params, start-day averaged.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from fast_backtest import FastBacktester
import numpy as np

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
STARTS = ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-08"]


def clear_enhanced(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}


def clear_si(bt):
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def avg_run(bt):
    c, s, d = [], [], []
    for st in STARTS:
        m = bt.run(st, "2025-12-31", V12)
        if m:
            c.append(m["cagr"]); s.append(m["sharpe"]); d.append(m["max_dd"])
    return np.mean(c)*100, np.std(c)*100, np.mean(s), np.mean(d)*100


def main():
    print("=" * 68)
    print("LOCATING THE 25.4% v12 CLAIM — v12 params, 4 data conditions")
    print("=" * 68)
    # Reload a fresh backtester each time so cleared data is restored.
    conditions = [
        ("A. DEPLOYED (enhanced OFF, SI OFF)", True, True),
        ("B. enhanced ON, SI OFF",            False, True),
        ("C. enhanced OFF, SI ON",            True, False),
        ("D. EVERYTHING ON (enhanced+SI)",    False, False),
    ]
    for label, clr_enh, clr_si in conditions:
        bt = FastBacktester()
        if clr_enh:
            clear_enhanced(bt)
        if clr_si:
            clear_si(bt)
        cagr, std, shrp, dd = avg_run(bt)
        print(f"\n  {label}")
        print(f"     CAGR {cagr:>5.1f}% ± {std:.1f} | Sharpe {shrp:.2f} | MaxDD {dd:.1f}%")
        del bt
    print("\n" + "=" * 68)
    print("README claims 25.4% / 1.00. Which row matches?")
    print("=" * 68)


if __name__ == "__main__":
    main()
