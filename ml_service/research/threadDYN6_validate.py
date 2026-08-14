"""thread DYN6 — try to KILL the OR(hy_oas, baa_aaa) off-cadence result.

The finding (DYN5): OR(hy,baa) re-evaluated every 5 days beats the deployed rebalance-only gate
on BOTH horizons, on BOTH return and drawdown, with an identical +0.043 Sharpe residual on each.

That is exactly the shape a good result AND a well-fitted one both have. So this script is
written to break it, not to confirm it. Three independent ways:

1. PARAMETER SENSITIVITY. The result was found at (gate_pct .95, derisk .50, cadence 5d). If it
   only survives there it is a fit. Vary each axis ALONE around the base — a real regime signal
   should degrade smoothly and stay positive over a plateau, not spike at one cell.

2. COST STRESS. Off-cadence adds turnover (~366 adjustments over 26yr). Re-run at 2x and 3x
   transaction cost. The backtest already charges 10 bps/leg against 6.10 measured live, so 2x
   is genuinely punitive rather than realistic — if the edge dies there it was turnover noise.

3. SUB-PERIOD STABILITY. Split the 26yr into three ~8yr blocks. A credit gate SHOULD earn most
   of its keep in 2008; the question is whether it is merely harmless elsewhere, or actively
   costly. One dominant block would mean the "both-period" result is really one event.

Everything is scored against a constant-leverage curve interpolated at the variant's own
realised avg_gross, so nothing scores for merely lowering exposure.

Run:  python3 research/threadDYN6_validate.py
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

WIN = {"gate_cols": ["hy_oas", "baa_aaa"], "gate_pct": 0.95, "gate_derisk": 0.5,
       "lev_recheck_every": 5, "lev_band": 0.05, "gate_offcadence": True}
DEPLOYED = {"credit_pct": 0.95, "credit_derisk": 0.5}

CONST_GRID = [0.60, 0.80, 1.00, 1.20, 1.49]

FULL = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01", "2018-02-15"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17", "2001-02-01", "2001-02-15"], "2025-12-31"),
]

SUBS = [
    ("sub 2001-08 (dotcom+GFC start)", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-02-01"], "2008-12-31"),
    ("sub 2009-16 (recovery)", "data/wrds/sp1500_universe_2000.pkl",
     ["2009-01-02", "2009-02-02"], "2016-12-31"),
    ("sub 2017-25 (recent)", "data/wrds/sp1500_universe_2000.pkl",
     ["2017-01-03", "2017-02-01"], "2025-12-31"),
]


def variants():
    v = [("BASE win (.95/.50/5d)", dict(WIN)), ("DEPLOYED (ref)", dict(DEPLOYED))]
    for p in [0.90, 0.93, 0.97]:
        v.append((f"gate_pct {p}", {**WIN, "gate_pct": p}))
    for d in [0.30, 0.70]:
        v.append((f"derisk {d}", {**WIN, "gate_derisk": d}))
    for c in [3, 10, 20]:
        v.append((f"cadence {c}d", {**WIN, "lev_recheck_every": c}))
    for m in [2.0, 3.0]:
        v.append((f"cost x{m:.0f}", {**WIN, "cost_mult": m}))
    return v


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


def block(title, path, starts, end, vlist):
    print("\n" + "=" * 104, flush=True)
    print(f"{title} | {len(starts)} starts", flush=True)
    print("=" * 104, flush=True)
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

    h = (f"  {'variant':<24}{'avgGross':>10}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
         f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}")
    print(h); print("  " + "-" * (len(h) - 2), flush=True)
    out = {}
    for label, extra in vlist:
        cfg = dict(BASE); cfg.update(extra)
        try:
            c, s, d, g = avg(bt, starts, end, cfg)
        except Exception as e:
            print(f"  {label:<24} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        rc, rs, rd = ref(g)
        out[label] = (c - rc, s - rs, d - rd)
        print(f"  {label:<24}{g:>10.4f}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}"
              f"{(c-rc)*100:>+8.2f}p{s-rs:>+9.3f}{(d-rd)*100:>+8.2f}p", flush=True)
    del bt
    return out


def main():
    res = {}
    for t, p, s_, e in FULL:
        res[t] = block(f"{t} | SENSITIVITY + COST STRESS", p, s_, e, variants())

    sub = {}
    for t, p, s_, e in SUBS:
        sub[t] = block(f"{t} | SUB-PERIOD STABILITY", p, s_, e,
                       [("BASE win (.95/.50/5d)", dict(WIN)), ("DEPLOYED (ref)", dict(DEPLOYED))])

    print("\n" + "=" * 104, flush=True)
    print("SENSITIVITY — Sharpe residual by axis (base = .95/.50/5d):", flush=True)
    labels = [l for l, _ in variants()]
    print(f"  {'variant':<24}{'8yr':>9}{'26yr':>9}   both>0?", flush=True)
    nsurv = 0
    for l in labels:
        a = res["8yr 2018-25"].get(l); b = res["26yr 2001-25"].get(l)
        if not a or not b:
            continue
        ok = a[1] > 0 and b[1] > 0
        nsurv += 1 if (ok and l not in ("DEPLOYED (ref)",)) else 0
        print(f"  {l:<24}{a[1]:>+9.3f}{b[1]:>+9.3f}   {'YES' if ok else 'no'}", flush=True)
    print(f"\n  -> {nsurv} of {len(labels)-1} perturbations keep a positive Sharpe residual on BOTH.",
          flush=True)
    print("     A plateau means signal. A single winning cell means fit.", flush=True)

    print("\nSUB-PERIOD STABILITY (is it all one crisis?):", flush=True)
    print(f"  {'block':<34}{'win dSh':>9}{'dep dSh':>9}{'win dDD':>9}{'dep dDD':>9}", flush=True)
    for t, _, _, _ in SUBS:
        w = sub[t].get("BASE win (.95/.50/5d)"); d = sub[t].get("DEPLOYED (ref)")
        if not w or not d:
            continue
        print(f"  {t:<34}{w[1]:>+9.3f}{d[1]:>+9.3f}{w[2]*100:>+8.2f}p{d[2]*100:>+8.2f}p", flush=True)


if __name__ == "__main__":
    main()
