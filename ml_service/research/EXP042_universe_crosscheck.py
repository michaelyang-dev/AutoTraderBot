"""EXP-042 — DATA INTEGRITY: do the two universe files agree?

THE TRIGGER
  VERIFY4 ran the identical config over the identical calendar years on two different universe
  pickles and got opposite signs for 2024: REC-minus-LIVE was +0.18pp on complete_sp1500 (8yr
  file) and -4.85pp on sp1500_universe_2000 (26yr file). 2023 (-0.91 vs -3.12) and 2025 (+0.87
  vs -1.45) disagree too.

  Two candidate causes with very different consequences:
    (a) PATH DEPENDENCE -- benign. Those runs enter in 2018/2019 vs 2001/2002, so by 2024 they
        sit at different NAV levels; with integer shares and a 10% cap, different NAV means
        different rounding and different binding caps.
    (b) UNIVERSE DISAGREEMENT -- a DEFECT. If the pickles carry different membership or prices
        for the same dates, every cross-horizon comparison in this program compares two different
        worlds, and the 8yr-vs-26yr agreement I have leaned on as evidence is partly an illusion.

  Hold the start date FIXED and vary ONLY the universe file: any remaining difference cannot be
  path dependence, so it is the data.

MEMORY (this is why the script is staged)
  ONE 26yr loader peaks at 44 GB resident on a 64 GB machine (measured). Loading both files in
  one process would swap the box to death -- the exact failure that crashed this laptop twice.
  So each file is processed in its OWN process which exits before the next starts, handing off
  compact artefacts through disk. Never merge these phases.

Run:  python3 research/EXP042_universe_crosscheck.py {8yrfile|26yrfile|compare}
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "compare"
FILES = {"8yrfile": "data/wrds/complete_sp1500_universe.pkl",
         "26yrfile": "data/wrds/sp1500_universe_2000.pkl"}
OUT = "research/_exp042"
STARTS = ["2018-01-03", "2018-05-03", "2018-09-03", "2019-01-03", "2019-05-03", "2019-09-03"]

LIVE = dict(leverage=1.49, tranches=1, tranche_stride=20, credit_pct=0.95, credit_derisk=0.50,
            mom_w=.50, val_w=.35, lv_w=.15, initial_capital=50_000.0, vol_overlay=True)
REC = dict(leverage=1.25, tranches=4, tranche_stride=5, credit_pct=0.95, credit_derisk=0.50,
           mom_w=.70, val_w=.21, lv_w=.09, initial_capital=50_000.0, vol_overlay=False)


def harvest(tag):
    """Load ONE file, save (i) its price panel restricted to the shared era and (ii) the NAV
    curves for both arms from fixed starts. Then exit so the 44 GB is released."""
    from main_production_backtest import FastBacktester
    import inspect
    import textwrap
    import VERIFY2_cleanroom as V
    from VERIFY2_cleanroom import CleanRoom

    src = inspect.getsource(CleanRoom.run)
    old = "dd=float(((v - v.cummax()) / v.cummax()).min()))"
    new = "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"
    assert src.count(old) == 1, "patch anchor missing -- refusing to guess"
    ns = dict(V.__dict__)
    exec(compile(textwrap.dedent(src.replace(old, new)), "<p>", "exec"), ns)
    CleanRoom.run = ns["run"]

    os.makedirs(OUT, exist_ok=True)
    bt = FastBacktester(universe_path=FILES[tag])
    px = bt.prices
    px = px.loc[px.index >= pd.Timestamp("2017-01-01")]          # shared era only
    px.astype("float32").to_parquet(f"{OUT}/px_{tag}.parquet")
    print(f"  [{tag}] prices saved {px.shape}", flush=True)

    cr = CleanRoom(bt)
    for nm, cfg in (("LIVE", LIVE), ("REC", REC)):
        cur = {}
        for s in STARTS:
            r = cr.run(s, cfg)
            cur[s] = r["curve"]
            print(f"  [{tag}] {nm} {s}  CAGR {r['cagr']:+.2%}  DD {r['dd']:.1%}", flush=True)
        pd.DataFrame(cur).to_parquet(f"{OUT}/nav_{tag}_{nm}.parquet")
    print(f"  [{tag}] done", flush=True)


def yearly(c):
    c = c.dropna()
    out = {}
    for y, seg in c.groupby(c.index.year):
        if len(seg) < 200:
            continue
        pr = c[c.index < pd.Timestamp(f"{y}-01-01")]
        if pr.empty:
            continue
        out[y] = seg.iloc[-1] / pr.iloc[-1] - 1
    return out


def compare():
    print("\n  ===== A. RAW PANEL COMPARISON (2017+) =====", flush=True)
    a = pd.read_parquet(f"{OUT}/px_8yrfile.parquet")
    b = pd.read_parquet(f"{OUT}/px_26yrfile.parquet")
    print(f"  8yr file : {a.shape[0]} dates x {a.shape[1]} symbols", flush=True)
    print(f"  26yr file: {b.shape[0]} dates x {b.shape[1]} symbols", flush=True)
    d = a.index.intersection(b.index)
    s = a.columns.intersection(b.columns)
    print(f"  shared: {len(d)} dates, {len(s)} symbols "
          f"(8yr-only {len(a.columns.difference(b.columns))}, "
          f"26yr-only {len(b.columns.difference(a.columns))})", flush=True)
    A, B = a.loc[d, s], b.loc[d, s]
    both = A.notna() & B.notna()
    n = int(both.values.sum())
    rel = ((A - B).abs() / B.abs().replace(0, np.nan)).where(both)
    fv = rel.values[np.isfinite(rel.values)]
    print(f"\n  PRICE AGREEMENT on {n:,} shared (date,symbol) cells:", flush=True)
    for tol in (1e-6, 1e-4, 1e-2):
        print(f"    within {tol:>7.0e}: {float((fv <= tol).sum())/len(fv):8.4%}", flush=True)
    print(f"    median {np.median(fv):.3e}   p99 {np.percentile(fv,99):.3e}   "
          f"max {fv.max():.3e}", flush=True)
    print(f"\n  {'year':<7}{'cells':>12}{'>1% diff':>11}{'8yr-only px':>13}{'26yr-only px':>14}"
          f"{'names 8yr':>11}{'names 26yr':>12}", flush=True)
    for y in sorted(set(d.year)):
        m = d.year == y
        AA, BB = A[m], B[m]
        bo = AA.notna() & BB.notna()
        nn = int(bo.values.sum())
        if nn == 0:
            continue
        r = ((AA - BB).abs() / BB.abs().replace(0, np.nan)).where(bo).values
        r = r[np.isfinite(r)]
        print(f"  {y:<7}{nn:>12,}{float((r>1e-2).sum())/max(len(r),1):>11.3%}"
              f"{int((AA.notna()&BB.isna()).values.sum()):>13,}"
              f"{int((AA.isna()&BB.notna()).values.sum()):>14,}"
              f"{a.loc[d[m]].notna().sum(axis=1).mean():>11.0f}"
              f"{b.loc[d[m]].notna().sum(axis=1).mean():>12.0f}", flush=True)

    print("\n\n  ===== B. SAME START, SAME CONFIG, ONLY THE FILE DIFFERS =====", flush=True)
    nav = {(t, nm): pd.read_parquet(f"{OUT}/nav_{t}_{nm}.parquet")
           for t in FILES for nm in ("LIVE", "REC")}
    for nm in ("LIVE", "REC"):
        print(f"\n  --- {nm}: per-year return, 8yr file vs 26yr file (same 6 starts) ---",
              flush=True)
        print(f"  {'year':<7}{'8yr file':>11}{'26yr file':>11}{'diff':>10}", flush=True)
        y8 = [yearly(nav[("8yrfile", nm)][c]) for c in nav[("8yrfile", nm)]]
        y26 = [yearly(nav[("26yrfile", nm)][c]) for c in nav[("26yrfile", nm)]]
        for y in sorted(set().union(*[set(x) for x in y8])):
            pr = [(p[y], q[y]) for p, q in zip(y8, y26) if y in p and y in q]
            if not pr:
                continue
            P = float(np.mean([x[0] for x in pr])); Q = float(np.mean([x[1] for x in pr]))
            print(f"  {y:<7}{P:>+11.2%}{Q:>+11.2%}{(P-Q)*100:>+9.2f}p"
                  f"{'   <-- MATERIAL' if abs(P-Q)>0.02 else ''}", flush=True)

    print("\n  --- the VERIFY4 discrepancy, isolated: (REC - LIVE) under each file ---",
          flush=True)
    print(f"  {'year':<7}{'8yr file':>11}{'26yr file':>11}{'gap':>10}", flush=True)
    yl8 = [yearly(nav[("8yrfile", "LIVE")][c]) for c in nav[("8yrfile", "LIVE")]]
    yr8 = [yearly(nav[("8yrfile", "REC")][c]) for c in nav[("8yrfile", "REC")]]
    yl26 = [yearly(nav[("26yrfile", "LIVE")][c]) for c in nav[("26yrfile", "LIVE")]]
    yr26 = [yearly(nav[("26yrfile", "REC")][c]) for c in nav[("26yrfile", "REC")]]
    for y in sorted(set().union(*[set(x) for x in yl8])):
        p8 = [r[y] - l[y] for l, r in zip(yl8, yr8) if y in l and y in r]
        p26 = [r[y] - l[y] for l, r in zip(yl26, yr26) if y in l and y in r]
        if not p8 or not p26:
            continue
        P, Q = float(np.mean(p8)), float(np.mean(p26))
        print(f"  {y:<7}{P*100:>+10.2f}p{Q*100:>+10.2f}p{(P-Q)*100:>+9.2f}p"
              f"{'   <-- MATERIAL' if abs(P-Q)>0.02 else ''}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    if MODE in FILES:
        harvest(MODE)
    else:
        compare()
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
