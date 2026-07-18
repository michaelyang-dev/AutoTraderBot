"""
FINAL LIVE-MIRROR CHECK — everything enabled, $50k, integer shares, 1x & 1.49x, credit gate
run INSIDE the backtest (not an overlay). Both periods, multi-start.

Config = the live v12: 50/35/15 mom/val/lv, top-5, 20d rebal, 40% stop, cap 0.10, no-RP,
bear weights, vol-scaling (de-risk-only), UMD crash — plus $50k capital, integer floor shares,
6.3% financing on the margin debit, and the p95 HY-OAS credit de-risk gate (0.5x) causal inside.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
       "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40}
PERIODS = [
    ("SHORT 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("LONG 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
CAP0, RATE = 50_000.0, 0.063


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


VARIANTS = [
    ("1.00x integer $50k (unlevered live)", dict(leverage=1.0, integer_shares=True)),
    ("1.00x integer + credit p95 gate", dict(leverage=1.0, integer_shares=True, credit_pct=0.95, credit_derisk=0.5)),
    ("1.49x integer $50k (LIVE, +fin)", dict(leverage=1.49, integer_shares=True, financing_rate=RATE)),
    ("1.49x integer + credit p95 gate (+fin)", dict(leverage=1.49, integer_shares=True, financing_rate=RATE, credit_pct=0.95, credit_derisk=0.5)),
    ("1.49x FRACTIONAL (+fin) [rounding ref]", dict(leverage=1.49, integer_shares=False, financing_rate=RATE)),
]


def stats_of(vals):
    dr = vals.pct_change().dropna()
    yrs = max((vals.index[-1] - vals.index[0]).days / 365.25, 1)
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / yrs) - 1
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd = ((vals - vals.cummax()) / vals.cummax()).min()
    return cagr, dr.std() * np.sqrt(252), sh, dd


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 104, flush=True)
        print(f"{pname} | $50k start | integer floor shares | {len(starts)}-start avg | everything enabled", flush=True)
        print("=" * 104, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<42}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'avgGross':>9}{'endNAV':>10}{'fin/yr':>8}"
        print(hdr); print("-" * len(hdr), flush=True)
        for name, extra in VARIANTS:
            cs, vs, ss, ds, gs, fin, endv = [], [], [], [], [], [], []
            for st in starts:
                cfg = dict(V12); cfg.update(extra); cfg["initial_capital"] = CAP0
                m = bt.run(st, end, cfg)
                c, v, s, d = stats_of(m["daily_values"])
                cs.append(c); vs.append(v); ss.append(s); ds.append(d)
                gs.append(m["avg_gross"]); endv.append(m["final"])
                yrs = max((m["daily_values"].index[-1] - m["daily_values"].index[0]).days / 365.25, 1)
                fin.append(m["fin_paid"] / yrs)
            print(f"{name:<42}{np.mean(cs):>+8.1%}{np.mean(vs):>7.1%}{np.mean(ss):>8.2f}"
                  f"{np.mean(ds):>+8.1%}{np.mean(gs):>9.2f}{np.mean(endv):>10,.0f}{np.mean(fin):>8,.0f}", flush=True)
        del bt
    print("\ninteger vs FRACTIONAL row = the $50k rounding drag. Credit-gate rows = full-stack effect.", flush=True)


if __name__ == "__main__":
    main()
