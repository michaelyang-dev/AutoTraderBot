"""EXP-053 — gate depth x leverage ON THE WINNING (overlay-ON) BASE. The last open dimension.

WHY THIS IS NEEDED, and why EXP-049 does not already answer it
  EXP-049 mapped the leverage x gate surface on a base with the vol overlay OFF -- it was written
  BEFORE EXP-047 reversed that decision. Its gate-depth finding (deeper gate wins at iso-drawdown:
  g0.00 > g0.25 > g0.50 at every drawdown target on the 26yr) is therefore measured on a base that
  is no longer the candidate, and it cannot be assumed to transfer. The overlay and the gate are
  BOTH de-risking mechanisms; on an overlay-ON book the gate is partly redundant, so its optimal
  depth can be different. Carrying EXP-049's answer across would repeat exactly the decomposition
  error that produced two retractions already.

THE GENUINE TENSION THIS RESOLVES
  On FULL-SAMPLE 26yr numbers the overlay-OFF book looks better: L1.10_g0.00 gives -45.2% MaxDD,
  +15.89% CAGR, Sharpe 0.668, versus winner A (overlay ON, g0.50) at -49.8%, +13.73%, 0.638.
  But L1.10_g0.00's 2023-2025 dSharpe is -0.105 while A's is -0.018 (26yr) and +0.110/+0.065 on the
  CLEAN 8yr file. The overlay-OFF book wins the average and loses the present.

  Since BUGS D9 makes the 26yr 2023-2025 cell the least trustworthy in the program and the 8yr file
  is clean there, the tiebreak goes to the overlay. But that tiebreak rests on ONE dimension I have
  not varied on the winning base: how hard the gate bites. If a deeper gate on the overlay-ON book
  recovers the full-sample CAGR without giving up the recent regime, that is the real answer.

GRID: leverage {1.00, 1.10, 1.25} x gate derisk {0.00, 0.25, 0.50}, overlay ON, tilt 70/21/9,
4 tranches. 24 starts, BOTH samples (12 used + 12 untouched holdout) -- finalist standard, not
screening standard.

  g0.25 remains NOT a candidate. It is here to make the dose-response readable; it was never
  pre-registered and choosing it would be selecting on the 2008-vs-2020 tradeoff after seeing it.
  If it wins it is labelled IN-SAMPLE and stays unshipped.

Run:  python3 research/EXP053_gate_on_winner.py [8yr|26yr] [run|analyse]
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
USED = [f"{y}-{m:02d}-03" for y in YRS for m in (1, 3, 5, 7, 9, 11)]
HOLD = [f"{y}-{m:02d}-03" for y in YRS for m in (2, 4, 6, 8, 10, 12)]
SUBS = ([("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2025", 2023, 2025)]
        if HZ == "8yr" else
        [("2001-2008", 2001, 2008), ("2009-2016", 2009, 2016),
         ("2017-2022", 2017, 2022), ("2023-2025", 2023, 2025)])
CACHE = f"research/_gw_{HZ}"
WIN = dict(credit_pct=0.95, initial_capital=50_000.0, vol_overlay=True,
           mom_w=.70, val_w=.21, lv_w=.09, tranches=4, tranche_stride=5)
LIVE = dict(credit_pct=0.95, credit_derisk=0.50, initial_capital=50_000.0, vol_overlay=True,
            mom_w=.50, val_w=.35, lv_w=.15, tranches=1, tranche_stride=20, leverage=1.49)
LEVS, GATES = [1.00, 1.10, 1.25], [0.00, 0.25, 0.50]


def arms():
    return [("LIVE", LIVE)] + [(f"L{l:.2f}_g{g:.2f}", dict(WIN, leverage=l, credit_derisk=g))
                               for l, g in itertools.product(LEVS, GATES)]


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
    t0 = time.time()
    cr = CleanRoom(FastBacktester(universe_path=PATH))
    for nm, cfg in arms():
        for tag, sts in (("used", USED), ("hold", HOLD)):
            f = f"{CACHE}/{nm}_{tag}.parquet"
            if os.path.exists(f):
                continue
            pd.DataFrame({s: cr.run(s, cfg)["curve"] for s in sts}).to_parquet(f)
            print(f"  {nm:<14}{tag}  {time.time()-t0:>6.0f}s", flush=True)


def _st(v):
    v = v.dropna()
    if len(v) < 60:
        return None
    r = v.pct_change()
    y = max((v.index[-1] - v.index[0]).days / 365.25, 0.25)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1,
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            float(((v - v.cummax()) / v.cummax()).min()))


def load(nm, tag):
    df = pd.read_parquet(f"{CACHE}/{nm}_{tag}.parquet")
    per = {}
    for lab, y0, y1 in SUBS:
        acc = []
        for c in df.columns:
            v = df[c].dropna()
            seg = v[(v.index >= pd.Timestamp(f"{y0}-01-01")) & (v.index <= pd.Timestamp(f"{y1}-12-31"))]
            x = _st(seg)
            if x:
                acc.append(x)
        if acc:
            per[lab] = np.array(acc)
    per["FULL"] = np.array([x for x in (_st(df[c]) for c in df.columns) if x])
    return per


def analyse():
    labs = [l for l, _, _ in SUBS]
    A = arms()
    for tag, title in (("hold", "UNTOUCHED HOLDOUT"), ("used", "previously-used")):
        D = {nm: load(nm, tag) for nm, _ in A if os.path.exists(f"{CACHE}/{nm}_{tag}.parquet")}
        if "LIVE" not in D:
            continue
        b = D["LIVE"]
        print(f"\n  ===== {HZ} — {title} (overlay ON base) =====", flush=True)
        print(f"  LIVE  CAGR {b['FULL'][:,0].mean():+.2%}  Sharpe {b['FULL'][:,1].mean():.3f}  "
              f"MaxDD {b['FULL'][:,2].mean():.1%}", flush=True)
        print(f"  {'config':<14}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}"
              f"{'dMaxDD':>9}{'won':>6}" + "".join(f"{l:>12}" for l in labs), flush=True)
        for nm, _ in A:
            if nm == "LIVE" or nm not in D:
                continue
            p = D[nm]; f = p["FULL"]
            ds = {l: p[l][:, 1].mean() - b[l][:, 1].mean() for l in labs if l in p and l in b}
            print(f"  {nm:<14}{f[:,0].mean():>+9.2%}{f[:,1].mean():>9.3f}{f[:,2].mean():>9.1%}"
                  f"{(f[:,0].mean()-b['FULL'][:,0].mean())*100:>+8.2f}p"
                  f"{f[:,1].mean()-b['FULL'][:,1].mean():>+9.3f}"
                  f"{(f[:,2].mean()-b['FULL'][:,2].mean())*100:>+8.2f}p"
                  f"{sum(1 for v in ds.values() if v>0):>4}/{len(ds)}"
                  + "".join(f"{ds.get(l,float('nan')):>+12.3f}" for l in labs), flush=True)

    print(f"\n  ===== {HZ} — ISO-DRAWDOWN on the overlay-ON base (holdout) =====", flush=True)
    D = {nm: load(nm, "hold") for nm, _ in A if os.path.exists(f"{CACHE}/{nm}_hold.parquet")}
    if "LIVE" not in D:
        return
    ld = D["LIVE"]["FULL"][:, 2].mean(); lc = D["LIVE"]["FULL"][:, 0].mean()
    print(f"  {'target MaxDD':<14}" + "".join(f"{'g%.2f'%g:>12}" for g in GATES), flush=True)
    for t in sorted({round(ld, 3), -0.30, -0.35, -0.40, -0.45, -0.50}):
        line = f"  {t:<14.1%}"
        for g in GATES:
            pts = sorted([(D[f"L{l:.2f}_g{g:.2f}"]["FULL"][:, 2].mean(),
                           D[f"L{l:.2f}_g{g:.2f}"]["FULL"][:, 0].mean())
                          for l in LEVS if f"L{l:.2f}_g{g:.2f}" in D])
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            line += (f"{np.interp(t, xs, ys):>+12.2%}" if xs and min(xs) <= t <= max(xs)
                     else f"{'off-curve':>12}")
        print(line, flush=True)
    print(f"\n  LIVE sits at MaxDD {ld:.1%} / CAGR {lc:+.2%}.", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run() if MODE == "run" else analyse()
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
