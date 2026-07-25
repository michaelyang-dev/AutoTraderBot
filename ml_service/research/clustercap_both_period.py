"""
CLUSTER-CAP COMPLETION RUN — the missing both-period test on the CURRENT live config.

Prior result (clustercap_run.py, old config: use_rp on, cap 0.15, vol_target 0.20 cap1.5):
cap0.30 = +1.4pp CAGR / +0.06 Sharpe (2-start) and +0.53pp / +0.02 (5 extra starts), DD flat,
parity-exact fork, binds ~25% of rebalances. Never tested on 26yr; never on the live config.

This run: patched fork (vol_scale_cap supported), CURRENT live config (use_rp=False,
cap 0.10, vol_target 0.15, de-risk-only cap 1.0), both periods, multi-start; 1x stats +
1.49x flat-financed overlay. Honest bar: beat baseline CAGR/Sharpe in BOTH periods, DD not
worse, same params.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from clustercap_fork_backtest import FastBacktester  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
       "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
CAPS = [None, 0.40, 0.30]
BASE_L, RATE = 1.49, 0.063


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(r):
    r = r.dropna(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    c = (1 + r).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min())


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 100, flush=True)
        print(f"{pname} | CURRENT live config (use_rp=F cap0.10 vt0.15 derisk-only) | {len(starts)}-start | 1x + 1.49x fin", flush=True)
        print("=" * 100, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = (f"{'variant':<20}{'1x CAGR':>9}{'1x Shrp':>8}{'1x DD':>8} |"
               f"{'1.49x CAGR':>11}{'Shrp':>7}{'DD':>8}{'bind%':>7}")
        print(hdr); print("-" * len(hdr), flush=True)
        for cc in CAPS:
            cs1, ss1, ds1, csL, ssL, dsL, binds = [], [], [], [], [], [], []
            for st in starts:
                cfg = dict(V12)
                if cc is not None:
                    cfg["cluster_cap"] = cc
                m = bt.run(st, end, cfg)
                r = m["daily_values"].pct_change().dropna()
                c1, v1, s1, d1 = stats(r)
                rl = BASE_L * r - max(0.0, BASE_L - 1.0) * (RATE / 252)
                cL, vL, sL, dL = stats(rl)
                cs1.append(c1); ss1.append(s1); ds1.append(d1)
                csL.append(cL); ssL.append(sL); dsL.append(dL)
                ev = getattr(bt, "_cluster_log", [])
                n_reb = getattr(bt, "_cluster_rebals", 0) or 1
                binds.append(len({e["date"] for e in ev}) / n_reb if cc else 0.0)
            label = "baseline" if cc is None else f"cluster_cap {cc:.2f}"
            print(f"{label:<20}{np.mean(cs1):>+9.1%}{np.mean(ss1):>8.2f}{np.mean(ds1):>+8.1%} |"
                  f"{np.mean(csL):>+11.1%}{np.mean(ssL):>7.2f}{np.mean(dsL):>+8.1%}{np.mean(binds):>7.0%}", flush=True)
        del bt
    print("\nBAR: cap variant beats baseline CAGR+Sharpe BOTH periods, DD not worse, same params.", flush=True)


if __name__ == "__main__":
    main()
