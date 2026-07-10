"""
DIVERGENCE IMPACT TEST — quantify the two confirmed live-vs-backtest WEIGHT divergences.

  D1 RISK-PARITY: honest backtest runs use_rp=True (inverse-vol reweight within mom/val
     sleeves); the LIVE signal_builder never applies RP. -> A/B: use_rp True vs False.
  D2 CAP BASE: backtest caps a name at 15% of the invested book; live caps at 15% of NAV
     on a ~1.49x book = ~10% of the book. -> A/B: cap 0.15 vs cap 0.10 (the live-effective
     relative cap at full leverage).
  D1+D2 combined = "what the live weight scheme actually is" vs the validated config.

8yr 2018-2025, 3 starts, deployed condition, 1x, everything else exact v12.
Run: OMP_NUM_THREADS=1 python3 research/divergence_impact_test.py
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
from main_production_backtest import FastBacktester

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
STARTS = ["2018-01-02", "2018-01-17", "2018-02-01"]
END = "2025-12-31"
VARIANTS = [
    ("VALIDATED (rp=T, cap .15)",     {}),
    ("D1: no risk-parity (LIVE)",     {"use_rp": False}),
    ("D2: cap .10 (LIVE-effective)",  {"cap": 0.10}),
    ("D1+D2: LIVE weight scheme",     {"use_rp": False, "cap": 0.10}),
]


def main():
    bt = FastBacktester()
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
    hdr = f"{'variant':<30}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}"
    print(hdr); print("-" * len(hdr), flush=True)
    for label, flags in VARIANTS:
        cs, ss, ds = [], [], []
        for st in STARTS:
            cfg = dict(V12); cfg.update(flags)
            m = bt.run(st, END, cfg)
            cs.append(m["cagr"]); ss.append(m["sharpe"]); ds.append(m["max_dd"])
        n = len(STARTS)
        print(f"{label:<30}{sum(cs)/n:>+8.1%}{sum(ss)/n:>8.2f}{sum(ds)/n:>+8.1%}", flush=True)


if __name__ == "__main__":
    main()
