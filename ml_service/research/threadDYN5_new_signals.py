"""thread DYN5 — new stress signals from data we already own, on top of the winning config.

Established: what works is applying a gate PROMPTLY. Config E (credit gate + vol, re-evaluated
every 5 days) is a Pareto improvement over the deployed rebalance-only gate on BOTH horizons
(8yr +2.32pp CAGR / +3.48pp MaxDD; 26yr +0.86pp / +4.23pp).

The deployed gate watches ONE signal: hy_oas. Sitting unused in the universe pickle are the
Fama-French factors (1963->2026) and 82 FRED series (1954->2025). Four new gates from them:

  umd_crash  -(20d sum of the UMD factor). The most interesting by far. The book is 50%
             momentum, and UMD already drives a WEIGHT switch (mom .50 -> .167) -- but has
             NEVER been allowed to touch LEVERAGE. Today, during a momentum crash, the book
             shifts its mix and stays fully levered. It also needs NO new data feed: live
             already computes it from prices (compute_price_umd_20d).
  baa_aaa    dbaa - daaa. Investment-grade risk aversion, independent of junk spreads.
  term_inv   -(dgs10 - dgs3mo). Curve inversion.
  mkt_vol    20d realized vol of mktrf -- MARKET vol, which moves before the book's own 40d
             shadow vol has registered anything.

Each is tested BOTH standalone and OR'd with hy_oas, always off-cadence, always scored against
a constant-leverage curve interpolated at the variant's own realised avg_gross.

WHY OR AND NOT AND: a gate exists to catch regimes the others miss. AND-ing makes it strictly
less sensitive than its most conservative member, which is the opposite of the point.

Run:  python3 research/threadDYN5_new_signals.py
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
OFF = {"lev_recheck_every": 5, "lev_band": 0.05, "gate_offcadence": True}
G = {"gate_pct": 0.95, "gate_derisk": 0.5}


def gate(cols):
    return {"gate_cols": cols, **G, **OFF}


VARIANTS = [
    ("C  deployed (hy, rebal-only)", {"credit_pct": 0.95, "credit_derisk": 0.5}),
    ("E  hy off-cadence (WINNER)",   gate(["hy_oas"])),
    ("umd_crash alone",              gate(["umd_crash"])),
    ("baa_aaa alone",                gate(["baa_aaa"])),
    ("term_inv alone",               gate(["term_inv"])),
    ("mkt_vol alone",                gate(["mkt_vol"])),
    ("OR hy + umd_crash",            gate(["hy_oas", "umd_crash"])),
    ("OR hy + baa_aaa",              gate(["hy_oas", "baa_aaa"])),
    ("OR hy + mkt_vol",              gate(["hy_oas", "mkt_vol"])),
    ("OR hy + umd + mkt_vol",        gate(["hy_oas", "umd_crash", "mkt_vol"])),
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
        print(f"{pname} | NEW STRESS SIGNALS (all off-cadence) | {len(starts)} starts", flush=True)
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
            try:
                c, s, d, g, la, dl = avg(bt, starts, end, cfg)
            except Exception as e:
                print(f"  {label:<30} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            rc, rs, rd = ref(g)
            res[(pname, label)] = (c - rc, s - rs, d - rd, c, d)
            print(f"  {label:<30}{g:>10.4f}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}"
                  f"{(c-rc)*100:>+8.2f}p{s-rs:>+9.3f}{(d-rd)*100:>+8.2f}p"
                  f"{min(dl):>+9.1%}{la:>6.0f}", flush=True)
        del bt

    print("\n" + "=" * 110, flush=True)
    print("BOTH-PERIOD (matched exposure):", flush=True)
    print(f"  {'variant':<30}{'8yr dSh':>9}{'26yr dSh':>10}{'8yr dDD':>9}{'26yr dDD':>10}   verdict",
          flush=True)
    for label, _ in VARIANTS:
        a = res.get(("8yr 2018-25", label)); b = res.get(("26yr 2001-25", label))
        if not a or not b:
            continue
        sh = a[1] > 0 and b[1] > 0
        dd = a[2] > 0 and b[2] > 0
        v = "SHARPE+DD" if (sh and dd) else ("DD-only" if dd else ("Sharpe-only" if sh else "dead"))
        print(f"  {label:<30}{a[1]:>+9.3f}{b[1]:>+10.3f}{a[2]*100:>+8.2f}p{b[2]*100:>+9.2f}p   {v}",
              flush=True)
    print("\nAbsolute CAGR matters too — a gate that only cuts drawdown by cutting return is", flush=True)
    print("not an improvement. Compare the CAGR column against C directly.", flush=True)


if __name__ == "__main__":
    main()
