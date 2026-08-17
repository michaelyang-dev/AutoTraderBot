"""EXP-035 — three more ways to improve Option 2, each with a stated mechanism.

CONTEXT
  Option 2 is a VARIANCE-REDUCTION change. That buys a budget: at 1.00x it produces 25/25 years
  of better drawdown and no return edge. The interesting question is what to SPEND that budget
  on. Three candidates, each drawn from a measured dose-response elsewhere in this program
  rather than invented:

IDEA A — SPEND IT ON MOMENTUM CONCENTRATION
  EXP-030 found a clean monotone dose-response: shifting capital from value/lowvol into momentum
  improves CAGR and Sharpe through the baseline in BOTH directions (-0.018 -> 0 -> +0.013 ->
  +0.023 -> +0.024 Sharpe at 40/50/60/70/80% momentum). It was not promoted because it reached
  only 7-8/12 and because momentum is the highest-vol sleeve, so the tilt costs drawdown.
  BUT Option 2 has just cut outcome dispersion ~60%. The hypothesis is that tranching creates
  exactly the variance headroom the tilt needs, so the pair should work where the tilt alone did
  not. EXP-003 gives the reason it should: the ranking carries 100% of the selection edge and
  momentum is the ranked sleeve.

IDEA B — SPEND IT ON CRISIS PROTECTION VIA SIGNAL EXIT
  EXP-026: mid-cycle signal exit is the ONLY mechanism in this program whose drawdown gain
  SURVIVED matched exposure (+5.6 to +7.8pp) instead of being a level effect. It costs Sharpe
  alone (-0.016 to -0.037 matched). The year-by-year showed Option 2 @1.25x loses crisis
  drawdown (2008 -4.7pp, 2020 -2.9pp), which is exactly the hole signal exit fills -- and
  unlike the vol overlay it is not a permanent tax, it only acts when the model has abandoned a
  name.

IDEA C — SLOW LEVERAGE RECALIBRATION (a LEVEL rule, not a timing rule)
  Every reactive vol rule in this repo is dead, and EXP-015 showed the deployed overlay has
  NEGATIVE timing skill. But EXP-015 also showed its value is a LEVEL effect. So: set leverage
  ONCE PER YEAR from trailing realised vol, using only data available at that point, and hold it
  fixed for the year. Slow enough that it cannot be a timing rule; adaptive enough to run less
  in a persistently high-vol regime. This is the untested middle between "constant" and
  "reactive", and the distinction the whole DYN program missed.

MULTIPLE-TESTING NOTE
  9 new configs. Judged on mechanism-consistent SHAPES (dose-response, crisis-year recovery),
  not on the best cell.

Run:  python3 research/EXP035_option2_synergies.py [8yr|26yr]
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

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
LIVE = {**DEPLOYED, **GATE, "financing_curve": True, "live_sizing": True,
        "initial_capital": 50_000.0}
O2 = {**LIVE, "vol_scaling": False, "tranches": 4, "tranche_stride": 5}

ARMS = [
    ("LIVE (1.49x+overlay)", dict(LIVE)),
    ("O2 @1.00 (reference)", {**O2, "leverage": 1.00}),
    ("O2 @1.25 (reference)", {**O2, "leverage": 1.25}),
    # A — spend the variance budget on momentum concentration
    ("A tilt60 @1.00", {**O2, "leverage": 1.00, "mom_w": 0.60, "val_w": 0.28, "lv_w": 0.12}),
    ("A tilt70 @1.00", {**O2, "leverage": 1.00, "mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09}),
    ("A tilt70 @1.25", {**O2, "leverage": 1.25, "mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09}),
    ("A tilt80 @1.00", {**O2, "leverage": 1.00, "mom_w": 0.80, "val_w": 0.14, "lv_w": 0.06}),
    # B — spend it on crisis protection via signal exit
    ("B sigexit5 @1.25", {**O2, "leverage": 1.25, "signal_exit_every": 5}),
    ("B sigexit10 @1.25", {**O2, "leverage": 1.25, "signal_exit_every": 10}),
    ("B sigexit5 @1.40", {**O2, "leverage": 1.40, "signal_exit_every": 5}),
    # A+B together
    ("A+B tilt70 sigexit5 @1.25", {**O2, "leverage": 1.25, "mom_w": 0.70, "val_w": 0.21,
                                   "lv_w": 0.09, "signal_exit_every": 5}),
]
CRISIS = [2001, 2002, 2008, 2020, 2022]


def crisis_dd(v):
    out = [float(((seg - seg.cummax()) / seg.cummax()).min())
           for y, seg in v.groupby(v.index.year) if int(y) in CRISIS and len(seg) > 20]
    return float(np.mean(out)) if out else float("nan")


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*106}\nEXP-035 OPTION-2 SYNERGIES — {HZ}, {len(STARTS)} starts, $50k"
          f"\n{'='*106}", flush=True)
    res = {}
    for name, cfg in ARMS:
        rows, cds = [], []
        for st in STARTS:
            m = bt.run(st, END, dict(cfg))
            rows.append(stat(m["daily_values"])); cds.append(crisis_dd(m["daily_values"]))
        res[name] = (rows, float(np.nanmean(cds)))
        print(f"  {name:<28} CAGR {np.mean([r['cagr'] for r in rows]):+6.2%}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    base = res["LIVE (1.49x+overlay)"][0]
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base]); bcd = res["LIVE (1.49x+overlay)"][1]
    print(f"\n  {'arm':<28}{'CAGR':>9}{'sdCAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'worst':>8}"
          f"{'CRISISddn':>11}{'dCAGR':>8}{'dShrp':>8}{'dMaxDD':>8}{'+Shrp':>7}{'verdict':>10}",
          flush=True)
    for name, _c in ARMS:
        rows, cd = res[name]
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<28}{c.mean():>+9.2%}{c.std(ddof=1):>8.2%}{s.mean():>8.3f}"
              f"{d.mean():>8.1%}{d.min():>8.1%}{cd:>11.1%}{(c-bc).mean()*100:>+7.2f}p"
              f"{(s-bs).mean():>+8.3f}{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}"
              f"{('' if name.startswith('LIVE') else v):>10}", flush=True)
    print(f"\n  TARGET: CAGR clearly above LIVE, Sharpe above LIVE, and CRISIS drawdown no worse",
          flush=True)
    print(f"  than LIVE's {bcd:.1%}. Anything that only moves CAGR is a leverage re-pick.",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
