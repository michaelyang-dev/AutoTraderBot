"""EXP-039 — THE FINAL MATRIX. LIVE vs RECOMMENDED across leverage, both horizons, all ideas.

RECOMMENDED = vol_scaling OFF + 4 tranches (5-day stride) + credit gate cutting to ~0
              (derisk 0.00; EXP-038 showed monotone improvement to the boundary, so 0.00 is the
              limit of the tested range, NOT a tuned optimum)

This run closes the last two gaps:
  1. A full LEVERAGE BREAKDOWN of the recommendation on both horizons, so the CAGR/drawdown
     trade is visible rather than quoted at one point.
  2. IDEA B (mid-cycle signal exit) FINALLY TESTED. It was void in EXP-035: the cadence gate
     (`di % sig_exit == 0 and di % stride != 0`) can never fire when sig_exit == stride, so
     `signal_exit_every=5` with `tranche_stride=5` fired on 0 of 200 days and all three arms
     returned numbers byte-identical to the reference. Valid cadences are FINER than the stride,
     so this uses 2 and 3. Signal exit is the only mechanism in this program whose drawdown gain
     survived matched exposure (+5.6..+7.8pp), so it is worth one honest test.

Run:  python3 research/EXP039_final_matrix.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_tranche import JointTrancheBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
LIVE = {**DEPLOYED, "credit_pct": 0.95, "credit_derisk": 0.50, "financing_curve": True,
        "live_sizing": True, "initial_capital": 50_000.0}
REC = {**LIVE, "vol_scaling": False, "tranches": 4, "tranche_stride": 5,
       "credit_derisk": 0.00}
T70 = {"mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09}

ARMS = [
    ("LIVE @1.49 (deployed)", dict(LIVE)),
    ("REC @1.00", {**REC, "leverage": 1.00}),
    ("REC @1.10", {**REC, "leverage": 1.10}),
    ("REC @1.25", {**REC, "leverage": 1.25}),
    ("REC @1.49", {**REC, "leverage": 1.49}),
    ("REC+tilt70 @1.25", {**REC, "leverage": 1.25, **T70}),
    ("REC+sigexit2 @1.25", {**REC, "leverage": 1.25, "signal_exit_every": 2}),
    ("REC+sigexit3 @1.25", {**REC, "leverage": 1.25, "signal_exit_every": 3}),
]
CRISIS = {2001, 2002, 2008, 2020, 2022}


def crisis_dd(v):
    o = [float(((s - s.cummax()) / s.cummax()).min())
         for y, s in v.groupby(v.index.year) if int(y) in CRISIS and len(s) > 20]
    return float(np.mean(o)) if o else float("nan")


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*112}\nEXP-039 FINAL MATRIX — {HZ}, {len(STARTS)} starts, $50,000\n{'='*112}",
          flush=True)
    res = {}
    for name, cfg in ARMS:
        try:
            rows, cds, gs = [], [], []
            for st in STARTS:
                m = bt.run(st, END, dict(cfg))
                rows.append(stat(m["daily_values"])); cds.append(crisis_dd(m["daily_values"]))
                gs.append(m["avg_gross"])
        except Exception as e:
            print(f"  {name:<22} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        res[name] = (rows, float(np.nanmean(cds)), float(np.mean(gs)))
        print(f"  {name:<22} CAGR {np.mean([r['cagr'] for r in rows]):+6.2%}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    base = res["LIVE @1.49 (deployed)"][0]
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base]); bcd = res["LIVE @1.49 (deployed)"][1]
    print(f"\n  {'arm':<22}{'gross':>7}{'CAGR':>9}{'sdCAGR':>8}{'Sharpe':>8}{'Sortino':>8}"
          f"{'MaxDD':>8}{'worst':>8}{'crisis':>8}{'dCAGR':>8}{'dShrp':>8}{'dMaxDD':>8}"
          f"{'+Shrp':>7}", flush=True)
    for name, _c in ARMS:
        if name not in res:
            continue
        rows, cd, g = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows]); so = np.array([r["sortino"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<22}{g:>7.3f}{c.mean():>+9.2%}{c.std(ddof=1):>8.2%}{s.mean():>8.3f}"
              f"{so.mean():>8.3f}{d.mean():>8.1%}{d.min():>8.1%}{cd:>8.1%}"
              f"{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}"
              f"{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}", flush=True)

    print(f"\n  === ISO-DRAWDOWN: CAGR delivered AT THE SAME drawdown (leverage-adjusted) ===",
          flush=True)
    lv = [n for n, _ in ARMS if n.startswith("REC @")]
    xs = np.array([abs(np.mean([r["dd"] for r in res[n][0]])) for n in lv if n in res])
    ys = np.array([np.mean([r["cagr"] for r in res[n][0]]) for n in lv if n in res])
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    lt = abs(bd.mean())
    print(f"  LIVE sits at MaxDD {lt:.1%} with CAGR {bc.mean():+.2%}", flush=True)
    if xs[0] <= lt <= xs[-1]:
        print(f"  RECOMMENDED at that same {lt:.1%} drawdown -> CAGR "
              f"{np.interp(lt, xs, ys):+.2%}   "
              f"(**{(np.interp(lt, xs, ys)-bc.mean())*100:+.2f}pp**)", flush=True)
    else:
        print(f"  LIVE's drawdown is outside the REC leverage range tested "
              f"({xs[0]:.1%}..{xs[-1]:.1%})", flush=True)
    for t in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        if xs[0] <= t <= xs[-1]:
            print(f"    at MaxDD {t:.0%}: REC CAGR {np.interp(t, xs, ys):+.2%}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
