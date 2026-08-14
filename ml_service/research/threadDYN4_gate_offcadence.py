"""thread DYN4 — apply the ONE proven winner promptly, and finally run the OR-composite.

Two things this session established:
  * the credit gate is the only leverage overlay with a real, documented edge (+7.7pp MaxDD)
  * what actually helps is fixing WHEN a rule is applied, not what it measures
    (off-cadence survives; EWMA / short windows / equity-trend / DD-state / recovery all die)

Nobody has put those two together. The credit gate is evaluated INSIDE the rebalance block, so
it can fire up to `rebal_days` (20 sessions) late. Credit spreads blow out in DAYS — 2008 and
2020 both went from calm to crisis well inside one rebalance cycle — so the gate is at its most
useless exactly when it is supposed to earn its keep. That is a structural lag, not a parameter.

Also finally runs the OR-composite the notes have carried as a "future option (+2pp DD)" without
ever testing: de-risk if ANY of hy_oas / rates_vol / curve is in its own top tail. One spread can
stay calm while another screams; an OR is strictly more sensitive than any single gate.

  A  no gate, rebal-only                 — what threadDYN measured (gate was OFF throughout)
  B  no gate, off-cadence vol            — the surviving vol result, for reference
  C  credit gate, rebal-only             — the DEPLOYED configuration
  D  credit gate, off-cadence GATE only  — isolates gate lag from vol lag
  E  credit gate, off-cadence vol + gate — both lags closed
  F  OR-composite (hy_oas|rates_vol), rebal-only
  G  OR-composite, off-cadence
  H  OR-composite of all three, off-cadence

Scored against a constant-leverage curve interpolated at each variant's OWN realised avg_gross,
so a gate that simply lowers average exposure scores zero. Gates are causal expanding
percentiles already shifted t+1 — no look-ahead.

Run:  python3 research/threadDYN4_gate_offcadence.py
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
     ["2018-01-02", "2018-01-17", "2018-02-01", "2018-02-15"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17", "2001-02-01", "2001-02-15"], "2025-12-31"),
]

CONST_GRID = [0.60, 0.80, 1.00, 1.20, 1.49]
OFF = {"lev_recheck_every": 5, "lev_band": 0.05}
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}

VARIANTS = [
    ("A no gate, rebal-only",        {}),
    ("B no gate, off-cadence vol",   {**OFF}),
    ("C gate, rebal-only (DEPLOYED)", {**GATE}),
    ("D gate off-cadence (gate only)", {**GATE, "gate_offcadence": True,
                                        "lev_recheck_every": 5, "lev_band": 0.0}),
    ("E gate + vol off-cadence",     {**GATE, **OFF, "gate_offcadence": True}),
    ("F OR(hy,rates) rebal-only",    {"gate_cols": ["hy_oas", "rates_vol"],
                                      "gate_pct": 0.95, "gate_derisk": 0.5}),
    ("G OR(hy,rates) off-cadence",   {"gate_cols": ["hy_oas", "rates_vol"],
                                      "gate_pct": 0.95, "gate_derisk": 0.5,
                                      **OFF, "gate_offcadence": True}),
    ("H OR(all 3) off-cadence",      {"gate_cols": ["hy_oas", "rates_vol", "curve"],
                                      "gate_pct": 0.95, "gate_derisk": 0.5,
                                      **OFF, "gate_offcadence": True}),
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
        print("\n" + "=" * 110, flush=True)
        print(f"{pname} | CREDIT GATE off-cadence + OR-composite | {len(starts)} starts",
              flush=True)
        print("=" * 110, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        gx, cy, sy, dy = [], [], [], []
        for lev in CONST_GRID:
            cfg = dict(BASE); cfg.update({"leverage": lev, "lev_policy": "constant"})
            c, s, d, g, _, _ = avg(bt, starts, end, cfg)
            gx.append(g); cy.append(c); sy.append(s); dy.append(d)
        o = np.argsort(gx)
        gx = np.array(gx)[o]; cy = np.array(cy)[o]; sy = np.array(sy)[o]; dy = np.array(dy)[o]

        def ref(g):
            return (float(np.interp(g, gx, cy)), float(np.interp(g, gx, sy)),
                    float(np.interp(g, gx, dy)))

        h = (f"  {'variant':<30}{'avgGross':>10}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
             f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'wstDD':>9}{'adj':>6}")
        print(h); print("  " + "-" * (len(h) - 2), flush=True)
        for label, extra in VARIANTS:
            cfg = dict(BASE); cfg.update(extra)
            c, s, d, g, la, dl = avg(bt, starts, end, cfg)
            rc, rs, rd = ref(g)
            res[(pname, label)] = (c - rc, s - rs, d - rd, min(dl))
            print(f"  {label:<30}{g:>10.4f}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}"
                  f"{(c-rc)*100:>+8.2f}p{s-rs:>+9.3f}{(d-rd)*100:>+8.2f}p"
                  f"{min(dl):>+9.1%}{la:>6.0f}", flush=True)
        del bt

    print("\n" + "=" * 110, flush=True)
    print("BOTH-PERIOD (matched exposure). DD is the claim; Sharpe residuals here are noise-scale.",
          flush=True)
    print(f"  {'variant':<30}{'8yr dSh':>9}{'26yr dSh':>10}{'8yr dDD':>9}{'26yr dDD':>10}   verdict",
          flush=True)
    for label, _ in VARIANTS:
        a = res.get(("8yr 2018-25", label)); b = res.get(("26yr 2001-25", label))
        if not a or not b:
            continue
        dd = a[2] > 0 and b[2] > 0
        sh = a[1] > 0 and b[1] > 0
        v = "SHARPE+DD" if (sh and dd) else ("DD-only" if dd else ("Sharpe-only" if sh else "dead"))
        print(f"  {label:<30}{a[1]:>+9.3f}{b[1]:>+10.3f}{a[2]*100:>+8.2f}p{b[2]*100:>+9.2f}p   {v}",
              flush=True)
    print("\nThe comparison that matters is D/E vs C: same gate, same threshold, only the LAG", flush=True)
    print("differs. If D/E beat C the deployed gate is simply being applied too late.", flush=True)


if __name__ == "__main__":
    main()
