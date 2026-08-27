"""EXP-054 — sweep the parameters that were never swept, on the CURRENT winning base.

WHY NOW
  EXP-047/051/053 settled the STRUCTURE: overlay ON, 4 tranches, sleeves 70/21/9, leverage 1.10,
  credit gate derisk 0.00. But the two components that turned out to CARRY the result -- the vol
  overlay and the credit gate -- have never had their own parameters tested. vol_target, the
  40-day window, the 0.30 clamp floor and the p95 trigger are all inherited assumptions, and so
  are top_n=5, the 40% stop and the position cap. They are hardcoded in the clean-room engine,
  which is exactly why nothing has ever touched them.

  (Made config-driven by an in-memory source patch. VERIFY2_cleanroom.py stays byte-identical on
  disk -- the audited engine is not edited.)

DESIGN: a STAR, three levels per factor, base in the middle
  One factor moves at a time, two steps either side of the current value. Chosen over a factorial
  because a full grid here is 3^7 = 2187 configs, and because the question is "is this parameter
  sitting at a cliff or on a plateau", which a star answers. Three levels means MONOTONICITY is
  testable per factor, which is the only thing separating a real gradient from a lucky draw.

  1 base + 14 variants = 15 configs.

PRE-REGISTERED PROMOTION RULE -- written before running, and this is the whole point
  This is escalation-ladder rung 1 (parameters), the LOWEST-value rung, entered only because the
  structure above it is settled. Fourteen variants means roughly one will look good at p<0.10 by
  pure chance, and this program has already produced two retractions from period-specific
  artefacts. So a variant is promoted ONLY if ALL of:
     (a) its two off-base levels are COHERENT (better on one side, worse on the other -- a
         gradient -- or symmetric decay; NOT "both sides better", which means the base sits in a
         hole and is usually noise),
     (b) positive Sharpe vs base in EVERY sub-period, on BOTH horizons,
     (c) it survives on the untouched holdout at 24 starts.
  Anything passing (a)+(b) on the 8yr screen alone is PRELIMINARY -- UNAUDITED and gets stage (c).
  I expect most of these to be noise and will report them as such.

Run:  python3 research/EXP054_param_frontier.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
        else "data/wrds/sp1500_universe_2000.pkl")
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13, 2)]     # 12 = screen
SUBS = ([("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2025", 2023, 2025)]
        if HZ == "8yr" else
        [("2001-2008", 2001, 2008), ("2009-2016", 2009, 2016),
         ("2017-2022", 2017, 2022), ("2023-2025", 2023, 2025)])
CACHE = f"research/_pf_{HZ}"

BASE = dict(credit_pct=0.95, credit_derisk=0.00, initial_capital=50_000.0, vol_overlay=True,
            mom_w=.70, val_w=.21, lv_w=.09, tranches=4, tranche_stride=5, leverage=1.10,
            top_n=5, trailing_stop=0.40, cap_pos=0.15,
            vol_target=0.15 * 1.49, vol_lookback=40, vol_floor=0.30)

FACTORS = [
    ("vol_lookback", [20, 60]),
    ("vol_target", [0.15 * 1.49 * 0.85, 0.15 * 1.49 * 1.15]),
    ("vol_floor", [0.20, 0.40]),
    ("credit_pct", [0.90, 0.98]),
    ("top_n", [3, 8]),
    ("trailing_stop", [0.30, 0.50]),
    ("cap_pos", [0.10, 0.20]),
]


def _patched_run():
    """Make the hardcoded knobs config-driven, in memory only."""
    import inspect
    import textwrap
    import VERIFY2_cleanroom as V
    from VERIFY2_cleanroom import CleanRoom
    src = inspect.getsource(CleanRoom.run)
    subs = [
        ("dd=float(((v - v.cummax()) / v.cummax()).min()))",
         "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"),
        ("cap_pos = 0.15", 'cap_pos = float(cfg.get("cap_pos", 0.15))'),
        ("stop = 0.40", 'stop = float(cfg.get("trailing_stop", 0.40))'),
        # line-anchored: the bare string also appears in the explanatory comment above it
        ("\n        VOL_TARGET = 0.15 * 1.49\n",
         '\n        VOL_TARGET = float(cfg.get("vol_target", 0.15 * 1.49))\n'),
        ("if use_ov and len(navhist) >= 41:",
         '_vw = int(cfg.get("vol_lookback", 40))\n                if use_ov and len(navhist) >= _vw + 1:'),
        ("rr = np.diff(np.array(navhist[-41:])) / np.array(navhist[-41:-1])",
         "rr = np.diff(np.array(navhist[-(_vw+1):])) / np.array(navhist[-(_vw+1):-1])"),
        ("vs = min(1.0, max(0.30, VOL_TARGET / rv))",
         'vs = min(1.0, max(float(cfg.get("vol_floor", 0.30)), VOL_TARGET / rv))'),
    ]
    for a, b in subs:
        assert src.count(a) == 1, f"anchor not unique/found: {a!r} ({src.count(a)})"
        src = src.replace(a, b)
    ns = dict(V.__dict__)
    exec(compile(textwrap.dedent(src), "<patched>", "exec"), ns)
    CleanRoom.run = ns["run"]
    return CleanRoom


def grid():
    out = [("BASE", dict(BASE))]
    for f, lv in FACTORS:
        for v in lv:
            tag = f"{f}={v:g}" if isinstance(v, float) else f"{f}={v}"
            out.append((tag, dict(BASE, **{f: v})))
    return out


def _st(v):
    v = v.dropna()
    if len(v) < 60:
        return None
    r = v.pct_change()
    y = max((v.index[-1] - v.index[0]).days / 365.25, 0.25)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1,
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            float(((v - v.cummax()) / v.cummax()).min()))


def main():
    from main_production_backtest import FastBacktester
    CleanRoom = _patched_run()
    os.makedirs(CACHE, exist_ok=True)
    g = grid()
    todo = [(n, c) for n, c in g if not os.path.exists(f"{CACHE}/{n}.parquet")]
    print(f"  {HZ}: {len(g)} configs, {len(todo)} to run", flush=True)
    t0 = time.time()
    if todo:
        cr = CleanRoom(FastBacktester(universe_path=PATH))
        # PARITY GATE: the patched engine must reproduce the unpatched BASE exactly when all
        # knobs sit at their hardcoded defaults. If it does not, the patch changed behaviour and
        # every number below is meaningless.
        chk = cr.run(STARTS[0], dict(BASE, cap_pos=0.15, trailing_stop=0.40,
                                     vol_target=0.15 * 1.49, vol_lookback=40, vol_floor=0.30))
        print(f"  parity probe: CAGR {chk['cagr']:+.4%} Sharpe {chk['sharpe']:.4f} "
              f"DD {chk['dd']:.2%}  (must equal the unpatched base)", flush=True)
        for i, (n, c) in enumerate(todo, 1):
            pd.DataFrame({s: cr.run(s, c)["curve"] for s in STARTS}).to_parquet(f"{CACHE}/{n}.parquet")
            el = time.time() - t0
            print(f"  [{i:>2}/{len(todo)}] {n:<22} {el:>6.0f}s eta {el/i*(len(todo)-i):>6.0f}s",
                  flush=True)

    D = {}
    for n, _ in g:
        f = f"{CACHE}/{n}.parquet"
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        per = {}
        for lab, y0, y1 in SUBS:
            acc = []
            for c in df.columns:
                v = df[c].dropna()
                seg = v[(v.index >= pd.Timestamp(f"{y0}-01-01")) &
                        (v.index <= pd.Timestamp(f"{y1}-12-31"))]
                x = _st(seg)
                if x:
                    acc.append(x)
            if acc:
                per[lab] = np.array(acc)
        per["FULL"] = np.array([x for x in (_st(df[c]) for c in df.columns) if x])
        D[n] = per
    if "BASE" not in D:
        return
    b = D["BASE"]
    labs = [l for l, _, _ in SUBS]
    print(f"\n  ===== {HZ} — PARAMETER STAR ({len(STARTS)} starts) =====", flush=True)
    print(f"  BASE  CAGR {b['FULL'][:,0].mean():+.2%}  Sharpe {b['FULL'][:,1].mean():.3f}  "
          f"MaxDD {b['FULL'][:,2].mean():.1%}\n", flush=True)
    print(f"  {'variant':<22}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}"
          f"{'dMaxDD':>9}{'won':>6}" + "".join(f"{l:>12}" for l in labs), flush=True)
    res = {}
    for n, _ in g:
        if n == "BASE" or n not in D:
            continue
        p = D[n]; f = p["FULL"]
        ds = {l: p[l][:, 1].mean() - b[l][:, 1].mean() for l in labs if l in p and l in b}
        res[n] = (f[:, 1].mean() - b["FULL"][:, 1].mean(), ds)
        print(f"  {n:<22}{f[:,0].mean():>+9.2%}{f[:,1].mean():>9.3f}{f[:,2].mean():>9.1%}"
              f"{(f[:,0].mean()-b['FULL'][:,0].mean())*100:>+8.2f}p"
              f"{f[:,1].mean()-b['FULL'][:,1].mean():>+9.3f}"
              f"{(f[:,2].mean()-b['FULL'][:,2].mean())*100:>+8.2f}p"
              f"{sum(1 for v in ds.values() if v>0):>4}/{len(ds)}"
              + "".join(f"{ds.get(l,float('nan')):>+12.3f}" for l in labs), flush=True)

    print(f"\n  ===== COHERENCE TEST (pre-registered rule (a)) =====", flush=True)
    print("  'both sides better' = the base sits in a hole = almost always noise.\n", flush=True)
    for fac, lv in FACTORS:
        tags = [f"{fac}={v:g}" if isinstance(v, float) else f"{fac}={v}" for v in lv]
        if not all(t in res for t in tags):
            continue
        lo, hi = res[tags[0]][0], res[tags[1]][0]
        if lo > 0 and hi > 0:
            verdict = "BOTH better -> base in a hole -> SUSPECT NOISE"
        elif lo < 0 and hi < 0:
            verdict = "both worse -> base is at a local optimum -> KEEP BASE"
        else:
            allsub = all(v > 0 for v in res[tags[0 if lo > 0 else 1]][1].values())
            verdict = ("GRADIENT toward " + tags[0 if lo > 0 else 1] +
                       (" + every sub-period -> PRELIMINARY, needs holdout"
                        if allsub else " but FAILS a sub-period -> reject"))
        print(f"  {fac:<16} lo {lo:+.3f}  hi {hi:+.3f}   {verdict}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
