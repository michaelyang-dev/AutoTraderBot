"""EXP-040 — do the TILT and SIGNAL-EXIT combine, or do they overlap?

They are mechanically independent -- the tilt changes SLEEVE WEIGHTS (what you select), signal
exit changes WHEN YOU LEAVE (timing) -- so a priori they should stack. But "a priori" has been
wrong repeatedly in this program:
  - tranching x overlay-removal: additive (interaction ~0)          -> assumption held
  - ROE quality x overlay-removal: SUBSTITUTES, not complements     -> assumption failed
  - the momentum tilt at 1.00x: clean aggregate AND clean dose-response, FAILED year-by-year
So this gets measured.

There is also a real reason to expect INTERACTION here. Signal exit self-de-risks (it sells into
cash and waits), taking realised gross from ~1.22 down to ~0.91 at nominal 1.25x. The tilt pushes
capital into the highest-vol sleeve. One lowers exposure, the other raises risk per dollar --
they may cancel, or the exit may create exactly the room the tilt needs. Both are plausible;
neither is known.

Because signal exit drops gross so far, it is also tested at HIGHER nominal leverage: if it
self-de-risks to 0.91 at 1.25x, then 1.49x/1.75x may be where it actually belongs.

Run:  python3 research/EXP040_combined_best.py [8yr|26yr]
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
REC = {**LIVE, "vol_scaling": False, "tranches": 4, "tranche_stride": 5, "credit_derisk": 0.00}
T70 = {"mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09}
SE = {"signal_exit_every": 2}

ARMS = [
    ("LIVE @1.49", dict(LIVE)),
    ("REC+tilt @1.25", {**REC, "leverage": 1.25, **T70}),
    ("REC+sigexit @1.25", {**REC, "leverage": 1.25, **SE}),
    ("REC+BOTH @1.25", {**REC, "leverage": 1.25, **T70, **SE}),
    ("REC+BOTH @1.49", {**REC, "leverage": 1.49, **T70, **SE}),
    ("REC+BOTH @1.75", {**REC, "leverage": 1.75, **T70, **SE}),
    ("REC+sigexit @1.75", {**REC, "leverage": 1.75, **SE}),
]
CRISIS = {2001, 2002, 2008, 2020, 2022}


def cdd(v):
    o = [float(((s - s.cummax()) / s.cummax()).min())
         for y, s in v.groupby(v.index.year) if int(y) in CRISIS and len(s) > 20]
    return float(np.mean(o)) if o else float("nan")


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*108}\nEXP-040 TILT x SIGNAL-EXIT — {HZ}, {len(STARTS)} starts\n{'='*108}",
          flush=True)
    res = {}
    for name, cfg in ARMS:
        rows, cds, gs = [], [], []
        for st in STARTS:
            m = bt.run(st, END, dict(cfg))
            rows.append(stat(m["daily_values"])); cds.append(cdd(m["daily_values"]))
            gs.append(m["avg_gross"])
        res[name] = (rows, float(np.nanmean(cds)), float(np.mean(gs)))
        print(f"  {name:<20} CAGR {np.mean([r['cagr'] for r in rows]):+6.2%}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    b = res["LIVE @1.49"][0]
    bc = np.array([r["cagr"] for r in b]); bs = np.array([r["sharpe"] for r in b])
    bd = np.array([r["dd"] for r in b])
    print(f"\n  {'arm':<20}{'gross':>7}{'CAGR':>9}{'sdCAGR':>8}{'Sharpe':>8}{'Sortino':>8}"
          f"{'MaxDD':>8}{'worst':>8}{'crisis':>8}{'dCAGR':>8}{'dShrp':>8}{'dMaxDD':>8}"
          f"{'+Shrp':>7}", flush=True)
    D = {}
    for name, _c in ARMS:
        rows, cd, g = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows]); so = np.array([r["sortino"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        D[name] = ((c - bc).mean(), (s - bs).mean(), (d - bd).mean())
        print(f"  {name:<20}{g:>7.3f}{c.mean():>+9.2%}{c.std(ddof=1):>8.2%}{s.mean():>8.3f}"
              f"{so.mean():>8.3f}{d.mean():>8.1%}{d.min():>8.1%}{cd:>8.1%}"
              f"{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}"
              f"{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}", flush=True)

    print(f"\n  === INTERACTION at 1.25x: (BOTH) - [(tilt) + (sigexit)] ===", flush=True)
    for j, lbl, sc in ((0, "CAGR", 100), (1, "Sharpe", 1), (2, "MaxDD", 100)):
        add = D["REC+tilt @1.25"][j] + D["REC+sigexit @1.25"][j]
        got = D["REC+BOTH @1.25"][j]
        print(f"  {lbl:<8} additive {add*sc:+7.3f}   actual {got*sc:+7.3f}   "
              f"interaction {(got-add)*sc:+7.3f}", flush=True)
    print(f"\n  ~0 => independent, both belong. Negative => they overlap, pick one.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
