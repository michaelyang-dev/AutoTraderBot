"""
THREAD T VALIDATION — is the OR(credit, rates-vol) gate actually good, or overlay-inflated?

The overlay (threadT) overstates gate benefit vs the real engine (vol-scaling interacts, as
the credit gate already showed +9.6pp overlay -> +6.9pp full-stack). So re-run the composite
INSIDE the full live-mirror ($50k, integer, financing, vol-scaling). Plus a ROBUSTNESS sweep
(threshold p90/p95/p97) to confirm it's not a knife-edge. Both periods, multi-start.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True, "financing_rate": 0.063}
PERIODS = [
    ("SHORT 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("LONG 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
VARIANTS = [
    ("BASELINE (no gate)", {}),
    ("credit only p95", dict(gate_cols=["hy_oas"], gate_pct=0.95)),
    ("rates-vol only p95", dict(gate_cols=["rates_vol"], gate_pct=0.95)),
    ("OR(credit,rates-vol) p95", dict(gate_cols=["hy_oas", "rates_vol"], gate_pct=0.95)),
    ("OR p90 [robustness]", dict(gate_cols=["hy_oas", "rates_vol"], gate_pct=0.90)),
    ("OR p97 [robustness]", dict(gate_cols=["hy_oas", "rates_vol"], gate_pct=0.97)),
]


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stat(v):
    dr = v.pct_change().dropna(); yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1, dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
            ((v - v.cummax()) / v.cummax()).min())


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 92, flush=True)
        print(f"{pname} | FULL live-mirror 1.49x integer $50k | {len(starts)}-start", flush=True)
        print("=" * 92, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<30}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dMaxDD':>9}"
        print(hdr); print("-" * len(hdr), flush=True)
        base = None
        for name, extra in VARIANTS:
            cs, ss, ds = [], [], []
            for st in starts:
                cfg = dict(BASE); cfg.update(extra)
                m = bt.run(st, end, cfg); c, s, d = stat(m["daily_values"])
                cs.append(c); ss.append(s); ds.append(d)
            c, s, d = np.mean(cs), np.mean(ss), np.mean(ds)
            if base is None:
                base = (c, s, d)
            dc = "" if name.startswith("BASELINE") else f"{(c-base[0])*100:>+7.1f}p"
            dd = "" if name.startswith("BASELINE") else f"{(d-base[2])*100:>+8.1f}p"
            print(f"{name:<30}{c:>+8.1%}{s:>8.2f}{d:>+8.1%}{dc:>8}{dd:>9}", flush=True)
        del bt
    print("\nHONEST full-stack numbers (not overlay). dMaxDD>0 = shallower drawdown = better.", flush=True)


if __name__ == "__main__":
    main()
