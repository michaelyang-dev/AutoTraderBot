"""thread DYN3 — the UP-leverage side, plus a robustness check on the one survivor.

WHERE WE ARE. threadDYN/DYN2, all scored against a constant-leverage curve interpolated at
each variant's OWN realised avg_gross (so level effects score zero):

  DEAD, on both periods: eqtrend, ddstate, recovery-ramp, asymmetric vol windows.
  MARGINAL: the deployed inverse-vol overlay itself — +0.018 Sharpe skill on 8yr, +0.001 on
    26yr. Nearly all of its value is a LEVEL effect, not timing.
  SURVIVED: applying the SAME vol_scale more often than every 20 days. Drawdown improves on
    BOTH horizons across all 8 cadence/band combinations tested (+4.75 to +6.59pp), at matched
    exposure, for roughly zero CAGR.

TWO GAPS THIS CLOSES.

1. Everything so far is DE-RISK ONLY: vol_scale_cap = 1.0 means the overlay can cut exposure
   but never raise it above base. The complaint that started this — "we de-levered and now the
   market is running away from us" — is about the UP side, which has literally been clamped
   off. Prior research recorded "lever-up Kelly-flat", but that ran on the poisoned universes
   and at a rebalance-only cadence. Re-test it properly: cap > 1.0, WITH the off-cadence
   application that we now know is the part that works.

2. The survivor was measured on 3 starts (8yr) / 2 starts (26yr). Drawdown is a
   single-path statistic and therefore the least stable thing we could have picked. Re-run
   with MORE starts before trusting ~5pp.

Run:  python3 research/threadDYN3_uplever.py
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
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063, "live_sizing": True}

# MORE starts — drawdown is a single-path statistic and needs the extra paths.
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01", "2018-02-15", "2018-03-01", "2018-03-15"],
     "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17", "2001-02-01", "2001-02-15"], "2025-12-31"),
]

CONST_GRID = [0.60, 0.80, 1.00, 1.20, 1.49]

OFF = {"lev_recheck_every": 5, "lev_band": 0.05}

VARIANTS = [
    ("deployed (cap 1.0, rebal)",   {"vol_scale_cap": 1.0}),
    ("off-cadence, cap 1.0",        {"vol_scale_cap": 1.0, **OFF}),
    ("off-cadence, cap 1.15",       {"vol_scale_cap": 1.15, **OFF}),
    ("off-cadence, cap 1.30",       {"vol_scale_cap": 1.30, **OFF}),
    ("off-cadence, cap 1.50",       {"vol_scale_cap": 1.50, **OFF}),
    ("rebal-only,  cap 1.30",       {"vol_scale_cap": 1.30}),
    ("off-cadence 10d, cap 1.0",    {"vol_scale_cap": 1.0, "lev_recheck_every": 10,
                                     "lev_band": 0.05}),
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


def avg(bt, starts, end, cfg):
    cs, ss, ds, gs, la = [], [], [], [], []
    for st in starts:
        m = bt.run(st, end, dict(cfg))
        c, s, d = stat(m["daily_values"])
        cs.append(c); ss.append(s); ds.append(d)
        gs.append(m["avg_gross"]); la.append(m.get("levadj", 0))
    return (float(np.mean(cs)), float(np.mean(ss)), float(np.mean(ds)),
            float(np.mean(gs)), float(np.mean(la)), ds)


def main():
    res = {}
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 108, flush=True)
        print(f"{pname} | UP-LEVER + robustness | {len(starts)} starts", flush=True)
        print("=" * 108, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        gx, cy, sy, dy = [], [], [], []
        for lev in CONST_GRID:
            cfg = dict(BASE); cfg.update({"leverage": lev, "lev_policy": "constant",
                                          "vol_scale_cap": 1.0})
            c, s, d, g, _, _ = avg(bt, starts, end, cfg)
            gx.append(g); cy.append(c); sy.append(s); dy.append(d)
        o = np.argsort(gx)
        gx = np.array(gx)[o]; cy = np.array(cy)[o]; sy = np.array(sy)[o]; dy = np.array(dy)[o]

        def ref(g):
            return (float(np.interp(g, gx, cy)), float(np.interp(g, gx, sy)),
                    float(np.interp(g, gx, dy)))

        h = (f"  {'variant':<28}{'avgGross':>10}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
             f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'wstDD':>9}{'adj':>6}")
        print(h); print("  " + "-" * (len(h) - 2), flush=True)
        for label, extra in VARIANTS:
            cfg = dict(BASE); cfg.update(extra)
            c, s, d, g, la, dlist = avg(bt, starts, end, cfg)
            rc, rs, rd = ref(g)
            res[(pname, label)] = (c - rc, s - rs, d - rd)
            print(f"  {label:<28}{g:>10.4f}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}"
                  f"{(c-rc)*100:>+8.2f}p{s-rs:>+9.3f}{(d-rd)*100:>+8.2f}p"
                  f"{min(dlist):>+9.1%}{la:>6.0f}", flush=True)
        del bt

    print("\n" + "=" * 108, flush=True)
    print("BOTH-PERIOD (matched exposure):", flush=True)
    print(f"  {'variant':<28}{'8yr dSh':>9}{'26yr dSh':>10}{'8yr dDD':>9}{'26yr dDD':>10}   verdict",
          flush=True)
    for label, _ in VARIANTS:
        a = res.get(("8yr 2018-25", label)); b = res.get(("26yr 2001-25", label))
        if not a or not b:
            continue
        dd = a[2] > 0 and b[2] > 0
        sh = a[1] > 0 and b[1] > 0
        v = "SHARPE+DD" if (sh and dd) else ("DD-only" if dd else ("Sharpe-only" if sh else "dead"))
        print(f"  {label:<28}{a[1]:>+9.3f}{b[1]:>+10.3f}{a[2]*100:>+8.2f}p{b[2]*100:>+9.2f}p   {v}",
              flush=True)
    print("\nwstDD = WORST single-start drawdown, not the average — the average hides the", flush=True)
    print("path that would actually have hurt.", flush=True)


if __name__ == "__main__":
    main()
