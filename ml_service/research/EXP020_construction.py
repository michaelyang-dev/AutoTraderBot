"""EXP-020 (IDEAS I-04 + I-13) — PORTFOLIO CONSTRUCTION: sleeve risk parity, cross-sleeve overlap.

Escalation-ladder level 5. Both are construction changes with no new signal.

I-04 SLEEVE-LEVEL RISK PARITY
  Sleeve weights are fixed in CAPITAL (50 mom / 35 val / 15 lowvol). Momentum is structurally
  the highest-vol sleeve, so the book's RISK split is nowhere near 50/35/15 and it DRIFTS with
  regime -- an unintended, uncompensated, time-varying bet nobody chose. Rescale the blended
  capital weights by inverse predicted sleeve vol, renormalised so total gross is unchanged
  (otherwise this is a leverage change wearing a construction costume).
  NOTE: `use_rp` is inverse-vol WITHIN mom/val, a different thing, already tested dead.

I-13 CROSS-SLEEVE OVERLAP
  A name picked by two sleeves gets the SUM of both weights, then is capped. Is that
  conviction or concentration? If the sleeves are different views, agreement is information.
  If they share inputs -- and they do, both momentum and lowvol read `roe`, and value and
  lowvol both read `gross_margin` -- agreement is correlated error, and the current behaviour
  silently doubles down on it. Three arms: as-is / boost / flat (no doubling).
  This is the one construction question whose ANSWER determines whether a risk control is
  needed, so it is worth running even if every arm loses.

PRE-REGISTERED
  1. Gross must be unchanged by construction (renormalised) -- verify avgGross matches base,
     otherwise any result is a leverage effect and must be discarded.
  2. Judge on Sharpe and MaxDD; 12 starts; both horizons.
  3. Sleeve-weight changes have been retracted as noise in this repo before (CORE2/3/4 at
     6/12). Anything under 9/12 here is noise too.

Run:  python3 research/EXP020_construction.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import (DEPLOYED, HORIZONS, END, starts, stat,  # noqa: E402
                     sign_verdict, event_concentration)

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)

ARMS = [
    ("sleeveRP p=1.0",   {"sleeve_rp": True, "sleeve_rp_power": 1.0}),
    ("sleeveRP p=0.5",   {"sleeve_rp": True, "sleeve_rp_power": 0.5}),
    ("sleeveRP p=1.5",   {"sleeve_rp": True, "sleeve_rp_power": 1.5}),
    ("overlap flat",     {"overlap_mode": "flat"}),
    ("overlap boost1.25", {"overlap_mode": "boost", "overlap_boost": 1.25}),
    ("overlap boost1.50", {"overlap_mode": "boost", "overlap_boost": 1.50}),
    ("sleeveRP + flat",  {"sleeve_rp": True, "overlap_mode": "flat"}),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*104}\nEXP-020 CONSTRUCTION — {HZ}, {len(STARTS)} starts, REAL financing"
          f"\n{'='*104}", flush=True)

    a = bt.run(STARTS[0], END, dict(DEPLOYED))
    b = bt.run(STARTS[0], END, {**DEPLOYED, "sleeve_rp": False, "overlap_mode": None})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  PARITY hooks off vs untouched: {'IDENTICAL' if same else 'DIFFERENT'}", flush=True)
    if not same:
        print("  ABORT — hooks changed the default path."); return

    base, bcv, bg = [], [], []
    for st in STARTS:
        m = bt.run(st, END, {**DEPLOYED, "financing_curve": True})
        base.append(stat(m["daily_values"])); bcv.append(m["daily_values"])
        bg.append(m["avg_gross"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  MaxDD {bd.mean():.1%}  "
          f"avgGross {np.mean(bg):.4f}", flush=True)

    print(f"\n  {'arm':<20}{'CAGR':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}{'avgGross':>10}"
          f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+Shrp':>7}{'top5%':>9}{'verdict':>10}",
          flush=True)
    for name, extra in ARMS:
        rows, cvs, gs = [], [], []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, **extra, "financing_curve": True})
            rows.append(stat(m["daily_values"])); cvs.append(m["daily_values"])
            gs.append(m["avg_gross"])
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
        cc = [event_concentration(bcv[i], cvs[i]) for i in range(len(rows))]
        ok = [x for x in cc if x["excess_meaningful"]]
        t5 = (f"{np.mean([x['top5_share'] for x in ok]):.0%}" if len(ok) >= len(cc) // 2
              else f"n/a({len(ok)}/{len(cc)})")
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        g = np.mean(gs)
        flag = "  <-- GROSS MOVED, discard" if abs(g - np.mean(bg)) > 0.02 else ""
        print(f"  {name:<20}{c.mean():>+9.2%}{s.mean():>9.3f}{so.mean():>9.3f}{d.mean():>9.1%}"
              f"{g:>10.4f}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
              f"{(d-bd).mean()*100:>+8.2f}p{ns:>4}/{len(s)}{t5:>9}{v:>10}{flag}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
