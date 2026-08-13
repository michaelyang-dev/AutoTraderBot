"""thread DYN2 — push on the ONE thing that consistently helped, and re-test the arm I broke.

threadDYN verdict (matched-exposure, so level effects are already removed):
    invvol (deployed)     8yr +0.018 | 26yr +0.001  Sharpe skill -> survives, but ~zero on 26yr
    volofvol              8yr +0.018 | 26yr +0.005  survives, tiny
    off-cadence           8yr +0.059 | 26yr -0.002  FAILS the Sharpe bar
    eqtrend / ddstate / recovery                    clearly negative on BOTH
    asym windows                                    INVALID — ran unscaled (bug, now fixed)

Two reasons not to close the book there:

1. OFF-CADENCE improved DRAWDOWN on BOTH periods at matched exposure (+5.10pp 8yr, +5.33pp
   26yr) even though its Sharpe residual was ~0 on the 26yr. Drawdown improving consistently
   on both horizons is a different claim from Sharpe, and it is the claim that matters for
   the "we re-lever too late" complaint. Worth knowing whether that is a real effect or an
   artifact of one particular cadence/band pair.
2. The ASYM arm never actually ran. Short windows are the most direct attack on measurement
   lag, so the hypothesis is untested rather than rejected.

Sweeps:
  - cadence: re-scale every 1 / 3 / 5 / 10 / 20 days   (20 ~= current rebalance-only)
  - band:    0.02 / 0.05 / 0.10 turnover deadband
  - asym:    (up, down) window pairs, now that short windows work

Same matched-exposure control as threadDYN: interpolate the constant-leverage curve at each
variant's OWN realised avg_gross. Anything that merely lowers exposure scores zero.

Run:  python3 research/threadDYN2_offcadence.py
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
        "financing_rate": 0.063, "live_sizing": True}

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]

CONST_GRID = [0.60, 0.80, 1.00, 1.20, 1.49]

VARIANTS = [
    ("baseline (rebal-only)",   {}),
    ("cadence 1d  band .05",    {"lev_recheck_every": 1,  "lev_band": 0.05}),
    ("cadence 3d  band .05",    {"lev_recheck_every": 3,  "lev_band": 0.05}),
    ("cadence 5d  band .05",    {"lev_recheck_every": 5,  "lev_band": 0.05}),
    ("cadence 10d band .05",    {"lev_recheck_every": 10, "lev_band": 0.05}),
    ("cadence 5d  band .02",    {"lev_recheck_every": 5,  "lev_band": 0.02}),
    ("cadence 5d  band .10",    {"lev_recheck_every": 5,  "lev_band": 0.10}),
    ("asym 10/40 (rebal-only)", {"vol_up_lookback": 10, "vol_dn_lookback": 40}),
    ("asym 20/60 (rebal-only)", {"vol_up_lookback": 20, "vol_dn_lookback": 60}),
    ("asym 10/40 + cadence 5d", {"vol_up_lookback": 10, "vol_dn_lookback": 40,
                                 "lev_recheck_every": 5, "lev_band": 0.05}),
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
            float(np.mean(gs)), float(np.mean(la)))


def main():
    res = {}
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 104, flush=True)
        print(f"{pname} | OFF-CADENCE + ASYM sweep, matched exposure | {len(starts)}-start",
              flush=True)
        print("=" * 104, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        gx, cy, sy, dy = [], [], [], []
        for lev in CONST_GRID:
            cfg = dict(BASE); cfg.update({"leverage": lev, "lev_policy": "constant"})
            c, s, d, g, _ = avg(bt, starts, end, cfg)
            gx.append(g); cy.append(c); sy.append(s); dy.append(d)
        o = np.argsort(gx)
        gx = np.array(gx)[o]; cy = np.array(cy)[o]; sy = np.array(sy)[o]; dy = np.array(dy)[o]

        def ref(g):
            return (float(np.interp(g, gx, cy)), float(np.interp(g, gx, sy)),
                    float(np.interp(g, gx, dy)))

        h = (f"  {'variant':<26}{'avgGross':>10}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
             f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'adj':>6}")
        print(h); print("  " + "-" * (len(h) - 2), flush=True)
        for label, extra in VARIANTS:
            cfg = dict(BASE); cfg.update(extra)
            c, s, d, g, la = avg(bt, starts, end, cfg)
            rc, rs, rd = ref(g)
            res[(pname, label)] = (c - rc, s - rs, d - rd, g)
            print(f"  {label:<26}{g:>10.4f}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}"
                  f"{(c-rc)*100:>+8.2f}p{s-rs:>+9.3f}{(d-rd)*100:>+8.2f}p{la:>6.0f}", flush=True)
        del bt

    print("\n" + "=" * 104, flush=True)
    print("BOTH-PERIOD RESIDUALS (vs constant at the SAME realised avg_gross):", flush=True)
    print(f"  {'variant':<26}{'8yr dSh':>9}{'26yr dSh':>10}{'8yr dDD':>9}{'26yr dDD':>10}   verdict",
          flush=True)
    for label, _ in VARIANTS:
        a = res.get(("8yr 2018-25", label)); b = res.get(("26yr 2001-25", label))
        if not a or not b:
            continue
        sh = a[1] > 0 and b[1] > 0
        dd = a[2] > 0 and b[2] > 0
        v = "SHARPE+DD" if (sh and dd) else ("DD-only" if dd else ("Sharpe-only" if sh else "dead"))
        print(f"  {label:<26}{a[1]:>+9.3f}{b[1]:>+10.3f}{a[2]*100:>+8.2f}p{b[2]*100:>+9.2f}p   {v}",
              flush=True)
    print("\nDD-only still matters: a variant that cuts drawdown on BOTH horizons at matched", flush=True)
    print("exposure is buying insurance, even if Sharpe is a wash.", flush=True)


if __name__ == "__main__":
    main()
