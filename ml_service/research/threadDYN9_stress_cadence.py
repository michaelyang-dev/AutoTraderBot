"""thread DYN9 — does PROMPTNESS generalise from leverage to the BOOK itself?

The one robust result of this whole investigation: applying a rule 20 sessions late destroys it.
That was about LEVERAGE. But the same lateness applies one level up — during a crisis the
HOLDINGS are up to `rebal_days` stale too. The strategy re-picks names every 20 sessions
regardless of regime, so in March 2020 it was trading a book selected in February.

If the insight is real rather than a quirk of the leverage overlay, shortening the rebalance
cadence WHILE THE GATE IS ON should help for the same reason. If it does not, that is evidence
the leverage result is about exposure specifically, not about staleness in general — which is
worth knowing before over-generalising from it.

Counter-argument to hold in mind: prior research found 20d is a genuine cadence optimum
(+3.4pp over neighbours) and that off-cadence trading costs 4-7pp/yr. So the prior is AGAINST
this working. Stress-conditioning is the one variant that prior work did not test — it keeps 20d
in normal regimes and only accelerates when credit is screaming.

  A  20d always (baseline)
  B  20d always + winning off-cadence LEVERAGE (the DYN8 config)
  C  stress cadence 10d + off-cadence leverage
  D  stress cadence 5d  + off-cadence leverage
  E  stress cadence 5d  WITHOUT off-cadence leverage (isolates cadence from leverage)

Scored against a constant-leverage curve interpolated at each variant's own realised avg_gross.

Run:  python3 research/threadDYN9_stress_cadence.py
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

GATE = {"gate_cols": ["hy_oas", "baa_aaa"], "gate_pct": 0.95, "gate_derisk": 0.30}
OFFLEV = {"lev_recheck_every": 5, "lev_band": 0.10, "gate_offcadence": True}

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01", "2018-02-15"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17", "2001-02-01", "2001-02-15"], "2025-12-31"),
]

CONST_GRID = [0.60, 0.80, 1.00, 1.20, 1.49]

VARIANTS = [
    ("A 20d, no off-lev",        {}),
    ("B 20d + off-lev (DYN8)",   {**GATE, **OFFLEV}),
    ("C stress 10d + off-lev",   {**GATE, **OFFLEV, "rebal_stress": 10}),
    ("D stress 5d + off-lev",    {**GATE, **OFFLEV, "rebal_stress": 5}),
    ("E stress 5d, NO off-lev",  {**GATE, "rebal_stress": 5}),
    ("F stress 10d, NO off-lev", {**GATE, "rebal_stress": 10}),
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
    cs, ss, ds, gs = [], [], [], []
    for st in starts:
        m = bt.run(st, end, dict(cfg))
        c, s, d = stat(m["daily_values"])
        cs.append(c); ss.append(s); ds.append(d); gs.append(m["avg_gross"])
    return float(np.mean(cs)), float(np.mean(ss)), float(np.mean(ds)), float(np.mean(gs))


def main():
    res = {}
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 100, flush=True)
        print(f"{pname} | STRESS-CONDITIONAL REBALANCE CADENCE | {len(starts)} starts", flush=True)
        print("=" * 100, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        gx, cy, sy, dy = [], [], [], []
        for lev in CONST_GRID:
            cfg = dict(BASE); cfg.update({"leverage": lev, "lev_policy": "constant"})
            c, s, d, g = avg(bt, starts, end, cfg)
            gx.append(g); cy.append(c); sy.append(s); dy.append(d)
        o = np.argsort(gx)
        gx = np.array(gx)[o]; cy = np.array(cy)[o]; sy = np.array(sy)[o]; dy = np.array(dy)[o]

        def ref(g):
            return (float(np.interp(g, gx, cy)), float(np.interp(g, gx, sy)),
                    float(np.interp(g, gx, dy)))

        h = (f"  {'variant':<26}{'avgGross':>10}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
             f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}")
        print(h); print("  " + "-" * (len(h) - 2), flush=True)
        for label, extra in VARIANTS:
            cfg = dict(BASE); cfg.update(extra)
            c, s, d, g = avg(bt, starts, end, cfg)
            rc, rs, rd = ref(g)
            res[(pname, label)] = (c - rc, s - rs, d - rd)
            print(f"  {label:<26}{g:>10.4f}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}"
                  f"{(c-rc)*100:>+8.2f}p{s-rs:>+9.3f}{(d-rd)*100:>+8.2f}p", flush=True)
        del bt

    print("\n" + "=" * 100, flush=True)
    print("BOTH-PERIOD:", flush=True)
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
    print("\nC/D vs B isolates the CADENCE effect (same leverage handling).", flush=True)
    print("E/F vs A isolates cadence with NO leverage change at all.", flush=True)


if __name__ == "__main__":
    main()
