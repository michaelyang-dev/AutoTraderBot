"""thread LEV — what does the strategy actually return at LIVE sizing?

Open item #1 from the 2026-08-10/11 audit. The backtest computes
    target_d = w * (nav * leverage)
with `combined` used AS-IS, so every source of under-allocation is preserved: sleeves
returning few names, the per-position cap binding, integer-rounding drag, cash drag.
Measured mean realized gross at nominal 1.49x is only **1.1164x NAV**.

Live does NOT do that. ibkr_engine._calibrate_quantities renormalises w = prob/total_prob so
the weights sum to exactly 1, then closed-loops a multiplier over 4 passes until ACTUAL gross
(after the 15% cap and whole-share truncation) hits nav * leverage * vol_scale * credit_derisk.
It deploys the full target every rebalance AND recovers the rounding drag.

=> live carries **1.33x the exposure (+33.5%)** of every number we quote. So the canonical
+23.58% / 0.84 / -38.23% describes a book running ~1.12x gross, while the real account runs
~1.49x. The drawdown number in particular is NOT the one live is exposed to.

This measures the live behaviour directly rather than inferring it:
    A  backtest sizing (as-is)      -> what every validated number used
    B  LIVE sizing (live_sizing)    -> what the account actually does

Same signals, same costs, same vol-scaling, same credit gate. ONLY the sizing differs.

Run:  python3 research/threadLEV_live_sizing_ab.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063}

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]

ARMS = [
    ("A backtest sizing (as-is)", {}),
    ("B LIVE sizing (renorm+loop)", {"live_sizing": True}),
]


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}
    bt._si_change_ranks_by_month = {}
    bt._si_months = []
    bt.uni._short_interest_rank = {}
    bt.uni._si_change_rank = {}


def stat(v):
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
            dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
            ((v - v.cummax()) / v.cummax()).min())


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 104, flush=True)
        print(f"{pname} | BACKTEST sizing vs LIVE sizing | $50k, integer shares, 6.3% financing "
              f"| {len(starts)}-start", flush=True)
        print("=" * 104, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        hdr = (f"{'arm':<30}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'avgGross':>10}"
               f"{'vol':>8}{'finalNAV':>12}")
        print(hdr)
        print("-" * len(hdr), flush=True)

        res = {}
        for name, extra in ARMS:
            cs, ss, ds, gs, vs, fs = [], [], [], [], [], []
            for st in starts:
                cfg = dict(BASE)
                cfg.update(extra)
                m = bt.run(st, end, cfg)
                c, s, d = stat(m["daily_values"])
                cs.append(c)
                ss.append(s)
                ds.append(d)
                gs.append(m["avg_gross"])
                vs.append(m["vol"])
                fs.append(m["final"])
            r = (float(np.mean(cs)), float(np.mean(ss)), float(np.mean(ds)),
                 float(np.mean(gs)), float(np.mean(vs)), float(np.mean(fs)))
            res[name] = (r, cs, ds)
            print(f"{name:<30}{r[0]:>+9.2%}{r[1]:>8.2f}{r[2]:>+9.2%}{r[3]:>10.4f}"
                  f"{r[4]:>8.1%}{r[5]:>12,.0f}", flush=True)

        a = res[ARMS[0][0]][0]
        b = res[ARMS[1][0]][0]
        print(f"\n  DELTA (live - backtest): CAGR {(b[0]-a[0])*100:+.2f}pp | "
              f"Sharpe {b[1]-a[1]:+.3f} | MaxDD {(b[2]-a[2])*100:+.2f}pp | "
              f"gross {b[3]/max(a[3],1e-9):.2f}x", flush=True)
        print(f"  per-start CAGR  A: " + "  ".join(f"{x:+.2%}" for x in res[ARMS[0][0]][1]), flush=True)
        print(f"  per-start CAGR  B: " + "  ".join(f"{x:+.2%}" for x in res[ARMS[1][0]][1]), flush=True)
        print(f"  per-start MaxDD A: " + "  ".join(f"{x:+.1%}" for x in res[ARMS[0][0]][2]), flush=True)
        print(f"  per-start MaxDD B: " + "  ".join(f"{x:+.1%}" for x in res[ARMS[1][0]][2]), flush=True)
        del bt

    print("\n" + "=" * 104, flush=True)
    print("ARM B is what the live account actually does. If its MaxDD is materially worse,", flush=True)
    print("the drawdown figure we have been quoting understates the real downside exposure.", flush=True)


if __name__ == "__main__":
    main()
