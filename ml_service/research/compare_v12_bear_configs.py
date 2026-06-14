"""
v12 Verification + Bear-Config Comparison
==========================================
Answers two questions:
  1. Is the deployed v12 config (50/35/15, t5, r20, stop40, cap15) actually
     ~25.4% CAGR, or was that number from a different config?
  2. Which bear-regime weights are better — LIVE (10/30/50/10) or
     BACKTEST (10/20/60/10)?

Runs the v12 config with SI + enhanced data CLEARED (matching production
exactly), start-day averaged, for both bear-weight variants.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from fast_backtest import FastBacktester
import numpy as np


def clear_to_v12(bt):
    """Strip SI + enhanced snapshot data so the backtest matches deployed v12."""
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def main():
    print("=" * 70)
    print("v12 VERIFICATION + BEAR-CONFIG COMPARISON")
    print("=" * 70)
    bt = FastBacktester()
    clear_to_v12(bt)

    base = {
        "universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
        "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
        "cap": 0.15,
    }
    variants = {
        "v12 LIVE bear (10/30/50/10)":     {**base, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}},
        "v12 BACKTEST bear (10/20/60/10)": {**base, "bear_weights": {"mom": 0.10, "val": 0.20, "s5": 0.60, "s3": 0.10}},
    }

    # start-day averaging (5 offsets) to control for rebalance-day luck
    starts = ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05", "2018-01-08"]

    for period_label, end in [("2018-2025", "2025-12-31")]:
        print(f"\n### Period: {period_label} (1x, start-day averaged, SI+enhanced CLEARED) ###\n")
        for label, cfg in variants.items():
            cagrs, sharpes, dds = [], [], []
            for st in starts:
                m = bt.run(st, end, cfg)
                if m:
                    cagrs.append(m["cagr"]); sharpes.append(m["sharpe"]); dds.append(m["max_dd"])
            if cagrs:
                print(f"  {label}")
                print(f"     CAGR   {np.mean(cagrs)*100:>6.1f}% ± {np.std(cagrs)*100:.1f}")
                print(f"     Sharpe {np.mean(sharpes):>6.2f}")
                print(f"     MaxDD  {np.mean(dds)*100:>6.1f}%")
                print()

    print("=" * 70)
    print("Compare CAGR/Sharpe above. README claims v12 = 25.4% CAGR, 1.00 Sharpe.")
    print("=" * 70)


if __name__ == "__main__":
    main()
