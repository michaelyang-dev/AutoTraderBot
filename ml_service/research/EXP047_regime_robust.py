"""EXP-047 — find a config that works in EVERY regime, not one that works on average.

WHY THE BAR CHANGED
  EXP-043 showed the current proposal earns its entire headline in 2018-2022 and is Sharpe-NEGATIVE
  in 2023-2025. A mean over 26 years hid that completely. Optimising the mean again would just
  re-select whatever happened to win the biggest sub-period, which is how I got here.

  So the score is no longer the mean. A config must beat LIVE on Sharpe in EVERY sub-period.
  That is a far harder bar and it is much less fittable: to game it a config would have to be
  luckiest in all four eras at once, not one.

  It also cannot be gamed by the leverage confound (EXP-046). Leverage lifts CAGR in bull regimes
  and hurts in crashes, so a pure leverage bet CANNOT be positive in every era. Only something with
  genuine risk-adjusted edge can.

THE GRID -- a full factorial, deliberately NOT a hand-picked list
  overlay   ON / OFF     -- ON is live. EXP-043 hints the overlay HELPED recently (LIVE beat REC on
                            Sharpe in 2023-25), which contradicts the whole overlay-removal thesis.
                            That contradiction is the most interesting thing on the table.
  sleeves   50/35/15 (live) / 70/21/9 (proposed)
  tranches  1 / 4         -- tranching is variance reduction with no alpha claim; it should be
                            regime-INDEPENDENT. If it is not, the EXP-001 mechanism is wrong.
  leverage  1.10 / 1.25 / 1.49
  = 24 configs. Factorial, so each factor's effect is measured with the others averaged out
  rather than confounded -- which is what a hand-picked list of "promising" configs cannot do.

12 starts for the screen (24 is for finalists only). Per-config caching, so this is resumable and
partial results are analysable at any point.

Run:  python3 research/EXP047_regime_robust.py [8yr|26yr] [run|analyse]
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
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13, 2)]      # 12
CACHE = f"research/_sweep_{HZ}"
SUBS = ([("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2025", 2023, 2025)]
        if HZ == "8yr" else
        [("2001-2008", 2001, 2008), ("2009-2016", 2009, 2016),
         ("2017-2022", 2017, 2022), ("2023-2025", 2023, 2025)])

OVL = [("ovl", True), ("noovl", False)]
SLV = [("s503515", (.50, .35, .15)), ("s702109", (.70, .21, .09))]
TRN = [("t1", 1, 20), ("t4", 4, 5)]
LEV = [1.10, 1.25, 1.49]


def grid():
    out = []
    for (on, ov), (sn, sw), (tn, tk, ts), lv in itertools.product(OVL, SLV, TRN, LEV):
        out.append((f"{on}_{sn}_{tn}_L{lv:.2f}",
                    dict(credit_pct=0.95, credit_derisk=0.50, initial_capital=50_000.0,
                         vol_overlay=ov, mom_w=sw[0], val_w=sw[1], lv_w=sw[2],
                         tranches=tk, tranche_stride=ts, leverage=lv)))
    return out


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
    print(f"  {HZ}: {len(g)} configs, {len(todo)} to run, {len(g)-len(todo)} cached", flush=True)
    if not todo:
        return
    t0 = time.time()
    cr = CleanRoom(FastBacktester(universe_path=PATH))
    for i, (n, c) in enumerate(todo, 1):
        pd.DataFrame({s: cr.run(s, c)["curve"] for s in STARTS}).to_parquet(f"{CACHE}/{n}.parquet")
        el = time.time() - t0
        print(f"  [{i:>2}/{len(todo)}] {n:<26} {el:>6.0f}s  eta {el/i*(len(todo)-i):>6.0f}s",
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


def periods(df):
    out = {}
    for lab, y0, y1 in SUBS:
        acc = []
        for c in df.columns:
            v = df[c].dropna()
            seg = v[(v.index >= pd.Timestamp(f"{y0}-01-01")) & (v.index <= pd.Timestamp(f"{y1}-12-31"))]
            s = _st(seg)
            if s:
                acc.append(s)
        if acc:
            a = np.array(acc)
            out[lab] = dict(cagr=a[:, 0], sharpe=a[:, 1], dd=a[:, 2])
    a = np.array([x for x in (_st(df[c]) for c in df.columns) if x])
    out["FULL"] = dict(cagr=a[:, 0], sharpe=a[:, 1], dd=a[:, 2])
    return out


def analyse():
    g = grid()
    have = [(n, c) for n, c in g if os.path.exists(f"{CACHE}/{n}.parquet")]
    print(f"  {HZ}: analysing {len(have)}/{len(g)} configs\n", flush=True)
    P = {n: periods(pd.read_parquet(f"{CACHE}/{n}.parquet")) for n, _ in have}
    BASE = "ovl_s503515_t1_L1.49"
    if BASE not in P:
        print(f"  baseline {BASE} missing", flush=True)
        return
    b = P[BASE]
    labs = [l for l, _, _ in SUBS]

    rows = []
    for n, _ in have:
        if n == BASE:
            continue
        p = P[n]
        ds = {l: p[l]["sharpe"].mean() - b[l]["sharpe"].mean() for l in labs if l in p and l in b}
        dc = {l: p[l]["cagr"].mean() - b[l]["cagr"].mean() for l in labs if l in p and l in b}
        dd = {l: p[l]["dd"].mean() - b[l]["dd"].mean() for l in labs if l in p and l in b}
        nwin = sum(1 for l in ds if ds[l] > 0)
        rows.append(dict(name=n, nwin=nwin, npd=len(ds),
                         worst=min(ds.values()) if ds else 0,
                         fs=p["FULL"]["sharpe"].mean() - b["FULL"]["sharpe"].mean(),
                         fc=p["FULL"]["cagr"].mean() - b["FULL"]["cagr"].mean(),
                         fd=p["FULL"]["dd"].mean() - b["FULL"]["dd"].mean(),
                         ds=ds, dc=dc, dd=dd))
    rows.sort(key=lambda r: (-r["nwin"], -r["worst"]))

    print(f"  ===== {HZ}: ranked by SUB-PERIODS WON, then by WORST sub-period =====")
    print(f"  baseline = {BASE} (= LIVE)\n")
    hdr = f"  {'config':<26}{'won':>5}{'worst':>9}{'fullDS':>9}{'fullDC':>9}{'fullDD':>9}   "
    hdr += "".join(f"{l:>12}" for l in labs)
    print(hdr, flush=True)
    for r in rows:
        line = (f"  {r['name']:<26}{r['nwin']}/{r['npd']:<3}{r['worst']:>+9.3f}"
                f"{r['fs']:>+9.3f}{r['fc']*100:>+8.2f}p{r['fd']*100:>+8.2f}p   ")
        line += "".join(f"{r['ds'].get(l,float('nan')):>+12.3f}" for l in labs)
        print(line, flush=True)

    clean = [r for r in rows if r["nwin"] == r["npd"]]
    print(f"\n  CONFIGS POSITIVE ON SHARPE IN EVERY SUB-PERIOD: {len(clean)}", flush=True)
    for r in clean:
        print(f"    {r['name']:<26} worst {r['worst']:+.3f}  full dSharpe {r['fs']:+.3f}  "
              f"dCAGR {r['fc']*100:+.2f}pp  dMaxDD {r['fd']*100:+.2f}pp", flush=True)
    if not clean:
        print("    NONE. No config in the grid beats LIVE on Sharpe in every regime.", flush=True)

    # factor effects: each factor averaged over all levels of the others
    print(f"\n  ===== MAIN EFFECTS (each factor, others averaged out) =====", flush=True)
    for fac, opts in (("overlay", [o[0] for o in OVL]), ("sleeves", [s[0] for s in SLV]),
                      ("tranche", [t[0] for t in TRN]),
                      ("leverage", [f"L{l:.2f}" for l in LEV])):
        print(f"  {fac}:", flush=True)
        for o in opts:
            sel = [P[n] for n, _ in have if o in n.split("_")]
            if not sel:
                continue
            fs = np.mean([p["FULL"]["sharpe"].mean() for p in sel])
            rec = np.mean([p[labs[-1]]["sharpe"].mean() for p in sel if labs[-1] in p])
            print(f"    {o:<10} mean FULL Sharpe {fs:.3f}   mean {labs[-1]} Sharpe {rec:.3f}",
                  flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run() if MODE == "run" else analyse()
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
