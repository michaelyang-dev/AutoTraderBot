"""thread DYN — do ANY dynamic-leverage IDEAS beat constant leverage at matched exposure?

Prior leverage-timing research is documented dead (trend gate, DD breaker, downside vol,
hysteresis, Kelly lever-up). Two things changed that make it legitimate to reopen — not a
hope, a reason:

  1. ALL of it ran on the POISONED universes. The 2026-07-28 rebuild fixed 8 defects; the 26yr
     CAGR alone moved -10.4pp. Conclusions drawn on that data are not evidence about this data.
  2. It optimised timing around a base level of 1.49x that threadLEV2 has now shown is well
     past optimum (Sharpe falls monotonically above ~1.0x on BOTH periods). A timing overlay
     tested on top of a mis-set base is not a clean test of timing.

THE CONTROL THAT MAKES THIS HONEST. Any policy that lowers average exposure will beat 1.49x on
Sharpe and drawdown for reasons that have NOTHING to do with timing. So:
  - run CONSTANT leverage across a grid to trace the level curve (avg_gross -> CAGR/Sharpe/DD)
  - run each policy, record its realised avg_gross
  - INTERPOLATE the constant curve at that same avg_gross = what you'd get with zero timing skill
  - TIMING SKILL = policy minus interpolated constant
A policy only earns its complexity if the residual is positive on BOTH periods.

POLICIES (deliberately different THEORIES, not variants of one):
  invvol     size to constant risk (deployed): mult = vol_target / realized_vol
  invvol+off same, but re-scaled every 5 days instead of only at the 20d rebalance
             (isolates APPLICATION lag: vol can collapse the day after a rebalance)
  invvol_ewma EWMA vol (isolates MEASUREMENT lag: 40d simple stays elevated ~40 sessions)
  invvol_asym short window when vol is FALLING, long when RISING (re-lever promptly)
  eqtrend    the strategy's own equity curve trends — levered while shadow NAV > its own MA
  ddstate    risk of ruin is path-dependent — taper linearly with distance from peak
  volofvol   what hurts is UNSTABLE vol, not high vol — gate on the CoV of vol
  recovery   the cost is being LATE — after a de-lever, ramp back on a SCHEDULE, explicitly
             ignoring whether vol has actually fallen

Run:  python3 research/threadDYN_leverage_policies.py
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

# constant-leverage grid -> the "no timing skill" reference curve
CONST_GRID = [0.60, 0.80, 1.00, 1.20, 1.49]

POLICIES = [
    ("invvol (deployed)",     {}),
    ("invvol + off-cadence",  {"lev_recheck_every": 5, "lev_band": 0.05}),
    ("invvol EWMA",           {"vol_mode": "ewma"}),
    ("invvol asym windows",   {"vol_up_lookback": 10, "vol_dn_lookback": 40}),
    ("eqtrend (own equity)",  {"lev_policy": "eqtrend", "eq_ma": 50, "eq_low": 0.60,
                               "lev_recheck_every": 5, "lev_band": 0.05}),
    ("ddstate (dist to peak)", {"lev_policy": "ddstate", "dd_tol": 0.25,
                                "lev_recheck_every": 5, "lev_band": 0.05}),
    ("volofvol (CoV of vol)", {"lev_policy": "volofvol", "vov_thr": 0.35, "vov_low": 0.60,
                               "lev_recheck_every": 5, "lev_band": 0.05}),
    ("recovery ramp",         {"lev_policy": "recovery", "rec_trigger": -0.10,
                               "rec_low": 0.50, "rec_ramp": 20,
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
    verdict = {}
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 104, flush=True)
        print(f"{pname} | DYNAMIC LEVERAGE — policies vs constant at MATCHED exposure | "
              f"{len(starts)}-start", flush=True)
        print("=" * 104, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        # ---- reference curve: constant leverage, no timing ----
        print("\n  CONSTANT-LEVERAGE REFERENCE (no timing skill by construction)", flush=True)
        h = f"  {'nominal':<10}{'avgGross':>10}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
        print(h); print("  " + "-" * (len(h) - 2), flush=True)
        gx, cy, sy, dy = [], [], [], []
        for lev in CONST_GRID:
            cfg = dict(BASE); cfg.update({"leverage": lev, "lev_policy": "constant"})
            c, s, d, g, _ = avg(bt, starts, end, cfg)
            gx.append(g); cy.append(c); sy.append(s); dy.append(d)
            print(f"  {lev:<10.2f}{g:>10.4f}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}", flush=True)
        order = np.argsort(gx)
        gx = np.array(gx)[order]; cy = np.array(cy)[order]
        sy = np.array(sy)[order]; dy = np.array(dy)[order]

        def ref(g):
            return (float(np.interp(g, gx, cy)), float(np.interp(g, gx, sy)),
                    float(np.interp(g, gx, dy)))

        print("\n  POLICIES  (dC/dS/dD = policy MINUS constant-at-same-avgGross = TIMING SKILL)",
              flush=True)
        h2 = (f"  {'policy':<24}{'avgGross':>10}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
              f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'adj':>6}")
        print(h2); print("  " + "-" * (len(h2) - 2), flush=True)
        for label, extra in POLICIES:
            cfg = dict(BASE); cfg.update(extra)
            c, s, d, g, la = avg(bt, starts, end, cfg)
            rc, rs, rd = ref(g)
            verdict[(pname, label)] = (c - rc, s - rs, d - rd)
            print(f"  {label:<24}{g:>10.4f}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}"
                  f"{(c-rc)*100:>+8.2f}p{s-rs:>+9.3f}{(d-rd)*100:>+8.2f}p{la:>6.0f}", flush=True)
        del bt

    print("\n" + "=" * 104, flush=True)
    print("BOTH-PERIOD TIMING SKILL (Sharpe residual vs matched-exposure constant):", flush=True)
    labels = [l for l, _ in POLICIES]
    for l in labels:
        a = verdict.get(("8yr 2018-25", l))
        b = verdict.get(("26yr 2001-25", l))
        if not a or not b:
            continue
        ok = "SURVIVES" if (a[1] > 0 and b[1] > 0) else "dead"
        print(f"  {l:<24} 8yr {a[1]:+.3f} | 26yr {b[1]:+.3f}   -> {ok}", flush=True)
    print("\nA policy only earns its complexity if the Sharpe residual is positive on BOTH.", flush=True)
    print("Anything else is just a leverage-level change wearing a costume.", flush=True)


if __name__ == "__main__":
    main()
