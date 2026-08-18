"""EXP-049 — the final leverage x credit-gate-depth surface, including SUB-1.0 leverage (I-33).

WHY THIS EXISTS
  EXP-046 established that the strategy package is regime-stable and that LEVERAGE is a pure
  risk-preference choice with a known price. That reduces the remaining decision to a 2-D surface:
  how much leverage, and how hard the credit gate bites. Every previous run sampled that surface
  at a handful of hand-picked points; this maps it.

  VERIFY4 already showed the two dials are COUPLED, not separable -- at 1.25x, gate 0.50 gave NO
  drawdown improvement (-1.12pp on 26yr) while gate 0.00 at identical leverage gave +6.01pp, a
  7.1pp swing from the gate multiplier alone, tracing almost entirely to 2008. A grid is the only
  honest way to read a coupled surface; picking points off it one at a time is how I got the
  decomposition wrong twice already.

I-33: LEVERAGE BELOW 1.00x
  EXP-031's iso-drawdown table had a hole -- at a -45% drawdown target only the baseline was
  on-curve, because the no-overlay arms cannot get below ~0.99x realised gross and so floor at
  about -48%. The deployed overlay CAN scale below 1.0x (clamp floor 0.30), reaching exposure
  territory constant leverage never sees. So every "the overlay is harmful" claim in this LOG is
  true only at matched exposure and understates the overlay at low target risk. 0.85x and 1.00x
  close that hole: if constant 0.85x reaches -45% with more CAGR than the overlay does, the
  overlay has no remaining role at ANY risk level. If it cannot, the overlay should be kept for
  drawdown-first operation and I should say so.

GRID: leverage {0.85, 1.00, 1.10, 1.25, 1.40, 1.49} x gate derisk {0.00, 0.25, 0.50} = 18.
  gate 0.25 is included ONLY as the interpolating point of a dose-response. It is NOT a candidate:
  I would be choosing it by looking at the 2008-vs-2020 tradeoff on the same data, which is the
  overfitting this program exists to avoid. If it wins it is labelled IN-SAMPLE and stays unshipped.

Judged on: Sharpe in EVERY sub-period, MaxDD, and iso-drawdown CAGR (what each config delivers at
a MATCHED drawdown, which is the only leverage-fair comparison).

Run:  python3 research/EXP049_lev_gate_grid.py [8yr|26yr] [run|analyse]
"""
import itertools
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
MODE = sys.argv[2] if len(sys.argv) > 2 else "run"
PATH = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
        else "data/wrds/sp1500_universe_2000.pkl")
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13, 2)]
SUBS = ([("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2025", 2023, 2025)]
        if HZ == "8yr" else
        [("2001-2008", 2001, 2008), ("2009-2016", 2009, 2016),
         ("2017-2022", 2017, 2022), ("2023-2025", 2023, 2025)])
CACHE = f"research/_lg_{HZ}"
LEVS = [0.85, 1.00, 1.10, 1.25, 1.40, 1.49]
GATES = [0.00, 0.25, 0.50]
STRAT = dict(credit_pct=0.95, initial_capital=50_000.0, vol_overlay=False,
             mom_w=.70, val_w=.21, lv_w=.09, tranches=4, tranche_stride=5)
LIVE = dict(credit_pct=0.95, credit_derisk=0.50, initial_capital=50_000.0, vol_overlay=True,
            mom_w=.50, val_w=.35, lv_w=.15, tranches=1, tranche_stride=20, leverage=1.49)


def grid():
    return [(f"L{l:.2f}_g{g:.2f}", dict(STRAT, leverage=l, credit_derisk=g))
            for l, g in itertools.product(LEVS, GATES)] + [("LIVE", LIVE)]


def run():
    import inspect
    import textwrap
    from main_production_backtest import FastBacktester
    import VERIFY2_cleanroom as V
    from VERIFY2_cleanroom import CleanRoom
    src = inspect.getsource(CleanRoom.run)
    old = "dd=float(((v - v.cummax()) / v.cummax()).min()))"
    new = "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"
    assert src.count(old) == 1, "patch anchor missing -- refusing to guess"
    ns = dict(V.__dict__)
    exec(compile(textwrap.dedent(src.replace(old, new)), "<p>", "exec"), ns)
    CleanRoom.run = ns["run"]
    os.makedirs(CACHE, exist_ok=True)
    g = grid()
    todo = [(n, c) for n, c in g if not os.path.exists(f"{CACHE}/{n}.parquet")]
    print(f"  {HZ}: {len(g)} configs, {len(todo)} to run", flush=True)
    if not todo:
        return
    t0 = time.time()
    cr = CleanRoom(FastBacktester(universe_path=PATH))
    for i, (n, c) in enumerate(todo, 1):
        pd.DataFrame({s: cr.run(s, c)["curve"] for s in STARTS}).to_parquet(f"{CACHE}/{n}.parquet")
        el = time.time() - t0
        print(f"  [{i:>2}/{len(todo)}] {n:<14} {el:>6.0f}s eta {el/i*(len(todo)-i):>6.0f}s",
              flush=True)


def _st(v):
    v = v.dropna()
    if len(v) < 60:
        return None
    r = v.pct_change()
    y = max((v.index[-1] - v.index[0]).days / 365.25, 0.25)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1,
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            float(((v - v.cummax()) / v.cummax()).min()))


def analyse():
    g = grid()
    have = [(n, c) for n, c in g if os.path.exists(f"{CACHE}/{n}.parquet")]
    D = {}
    for n, _ in have:
        df = pd.read_parquet(f"{CACHE}/{n}.parquet")
        per = {}
        for lab, y0, y1 in SUBS:
            acc = [s for s in (_st(df[c].dropna()[(df[c].dropna().index >= pd.Timestamp(f"{y0}-01-01")) &
                                                  (df[c].dropna().index <= pd.Timestamp(f"{y1}-12-31"))])
                               for c in df.columns) if s]
            if acc:
                per[lab] = np.array(acc)
        per["FULL"] = np.array([x for x in (_st(df[c]) for c in df.columns) if x])
        D[n] = per
    if "LIVE" not in D:
        print("  LIVE missing", flush=True)
        return
    b = D["LIVE"]
    labs = [l for l, _, _ in SUBS]

    print(f"\n  ===== {HZ} — LEVERAGE x GATE SURFACE ({len(STARTS)} starts) =====", flush=True)
    print(f"  LIVE: CAGR {b['FULL'][:,0].mean():+.2%}  Sharpe {b['FULL'][:,1].mean():.3f}  "
          f"MaxDD {b['FULL'][:,2].mean():.1%}\n", flush=True)
    print(f"  {'config':<14}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}"
          f"{'dMaxDD':>9}{'won':>6}" + "".join(f"{l:>12}" for l in labs), flush=True)
    rows = []
    for n, _ in have:
        if n == "LIVE":
            continue
        p = D[n]
        f = p["FULL"]
        ds = {l: p[l][:, 1].mean() - b[l][:, 1].mean() for l in labs if l in p and l in b}
        won = sum(1 for v in ds.values() if v > 0)
        rows.append((n, f, ds, won))
        print(f"  {n:<14}{f[:,0].mean():>+9.2%}{f[:,1].mean():>9.3f}{f[:,2].mean():>9.1%}"
              f"{(f[:,0].mean()-b['FULL'][:,0].mean())*100:>+8.2f}p"
              f"{f[:,1].mean()-b['FULL'][:,1].mean():>+9.3f}"
              f"{(f[:,2].mean()-b['FULL'][:,2].mean())*100:>+8.2f}p{won:>4}/{len(ds)}"
              + "".join(f"{ds.get(l,float('nan')):>+12.3f}" for l in labs), flush=True)

    clean = [r for r in rows if r[3] == len(labs)]
    print(f"\n  POSITIVE ON SHARPE IN EVERY SUB-PERIOD: {len(clean)}", flush=True)
    for n, f, ds, _ in sorted(clean, key=lambda r: -r[1][:, 1].mean()):
        print(f"    {n:<14} Sharpe {f[:,1].mean():.3f}  CAGR {f[:,0].mean():+.2%}  "
              f"MaxDD {f[:,2].mean():.1%}  worst sub {min(ds.values()):+.3f}", flush=True)
    if not clean:
        print("    NONE.", flush=True)

    # ---- ISO-DRAWDOWN: what CAGR does each gate level deliver at a MATCHED MaxDD? ----
    print(f"\n  ===== ISO-DRAWDOWN (the only leverage-fair comparison) =====", flush=True)
    print(f"  For each gate depth, interpolate CAGR at a common MaxDD target.", flush=True)
    live_dd = b["FULL"][:, 2].mean()
    live_cagr = b["FULL"][:, 0].mean()
    targets = sorted({round(live_dd, 3), -0.35, -0.40, -0.45, -0.50})
    print(f"  {'target MaxDD':<14}" + "".join(f"{'g%.2f'%gg:>12}" for gg in GATES)
          + f"{'LIVE':>12}", flush=True)
    for t in targets:
        line = f"  {t:<14.1%}"
        for gg in GATES:
            pts = sorted([(D[f"L{l:.2f}_g{gg:.2f}"]["FULL"][:, 2].mean(),
                           D[f"L{l:.2f}_g{gg:.2f}"]["FULL"][:, 0].mean())
                          for l in LEVS if f"L{l:.2f}_g{gg:.2f}" in D])
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            line += (f"{np.interp(t, xs, ys):>+12.2%}" if xs and min(xs) <= t <= max(xs)
                     else f"{'off-curve':>12}")
        line += f"{live_cagr:>+12.2%}" if abs(t - live_dd) < 1e-9 else f"{'-':>12}"
        print(line, flush=True)
    print(f"\n  (LIVE sits at MaxDD {live_dd:.1%} with CAGR {live_cagr:+.2%}. A config beats it "
          f"only if it shows MORE CAGR in the row at that same drawdown.)", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run() if MODE == "run" else analyse()
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
