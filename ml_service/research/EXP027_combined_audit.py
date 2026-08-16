"""EXP-027 — AUDIT GATE for the combined arm: tranching K=4 + constant leverage, no vol overlay.

THIS RESULT MAY BE INFLATED. Auditing before treating it as a finding.

WHAT CAME BACK (EXP-019, both horizons, 12 starts, gate ON, REAL financing, live_sizing)
  arm E = K=4 phase tranching + vol_scaling OFF + leverage 1.00x, vs the deployed system:

                    dCAGR    dSharpe        dMaxDD        worstDD          sd ratio
      26yr        +0.11pp  +0.066 (10/12)  +5.99pp (12/12)  -71.4% -> -59.1%   0.350
       8yr        +0.31pp  +0.096  (8/12)  +4.76pp  (9/12)  -45.5% -> -37.1%   0.555

  Positive on ALL THREE axes on BOTH horizons. Drawdown clears the >=9/12 sign bar on both.
  Sharpe clears on the 26yr (10/12) and MISSES on the 8yr (8/12) -- stated, not smoothed over.

  Stacking is near-additive for Sharpe on both horizons, and the DRAWDOWN interaction is
  super-additive on both (+3.94pp on 26yr, +2.60pp on 8yr) -- i.e. the combination protects
  the tail by MORE than the sum of its parts, consistently.

THE WAYS THIS COULD STILL BE FICTION, each tested below
  1. LOWER EXPOSURE. Arm E runs constant 1.00x with no overlay; the deployed arm averages
     ~1.19x gross. Drawdown improves when you hold less -- that is arithmetic, not skill. The
     matched-exposure control is the single most important test here. EXP-008 was killed by
     exactly this: raw dMaxDD +12.29pp, matched dSharpe -0.099.
  2. TURNOVER. Four sub-books trade on four schedules. Costs are charged inside, but each cost
     multiplier must be paired against the deployed arm at the SAME multiplier (BUGS A8b).
  3. ACCOUNT SIZE. $33k / 4 tranches = $8,250 each; at 1.00x that is ~$360/position and names
     above ~$350/share become unbuyable. This is the constraint most likely to kill it.
  4. EVENT CONCENTRATION, guarded (BUGS A8a).
  5. It is one arm chosen after looking at several. The sd-ratio and drawdown claims were
     pre-registered; the CAGR sign was not, and is treated as incidental.

Run:  python3 research/EXP027_combined_audit.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import (DEPLOYED, HORIZONS, END, starts, stat, sign_verdict,  # noqa: E402
                     matched_exposure_curve, event_concentration)

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
P4 = [0, 5, 10, 15]
CANDS = [("E both 1.00x", P4, {"leverage": 1.00, "vol_scaling": False}),
         ("D both 1.10x", P4, {"leverage": 1.10, "vol_scaling": False})]


def combo(bt, st, phases, extra, cap, cm):
    ck = cap / len(phases)
    per, gs = [], []
    for p in phases:
        m = bt.run(st, END, {**DEPLOYED, **GATE, **extra, "initial_capital": ck,
                             "rebal_phase": p, "live_sizing": True,
                             "financing_curve": True, "cost_mult": cm})
        per.append(m["daily_values"]); gs.append(m["avg_gross"])
    return sum(x.reindex(per[0].index).ffill() for x in per), float(np.mean(gs))


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*118}\nEXP-027 COMBINED-ARM AUDIT GATE — {HZ}, {len(STARTS)} starts, gate ON"
          f"\n{'='*118}", flush=True)

    for cap in (50_000.0, 33_000.0):
        for cm in (1, 2, 3):
            base, bcv, bg = [], [], []
            for st in STARTS:
                c, g = combo(bt, st, [0], {}, cap, cm)
                base.append(stat(c)); bcv.append(c); bg.append(g)
            bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
            bd = np.array([r["dd"] for r in base])
            print(f"\n  --- ${cap:,.0f}  cost x{cm} ---  base CAGR {bc.mean():+.2%} "
                  f"Sharpe {bs.mean():.3f} MaxDD {bd.mean():.1%} worst {bd.min():.1%} "
                  f"gross {np.mean(bg):.4f}", flush=True)
            print(f"  {'arm':<16}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'worst':>8}{'gross':>8}"
                  f"{'dCAGR':>8}{'dShrp':>8}{'dMaxDD':>8}{'+Shrp':>7}{'+DD':>7}{'sdRatio':>8}"
                  f"{'MATCHED dShrp/dDD':>21}{'top5%':>9}", flush=True)
            for name, phases, extra in CANDS:
                rows, cvs, gs, ms, md = [], [], [], [], []
                for st in STARTS:
                    c, g = combo(bt, st, phases, extra, cap, cm)
                    r = stat(c); rows.append(r); cvs.append(c); gs.append(g)
                    ref = matched_exposure_curve(
                        bt, st, END, g,
                        cfg={**DEPLOYED, **GATE, "initial_capital": cap,
                             "live_sizing": True, "financing_curve": True, "cost_mult": cm})
                    ms.append(r["sharpe"] - ref["sharpe"]); md.append(r["dd"] - ref["dd"])
                cc = np.array([r["cagr"] for r in rows]); ss = np.array([r["sharpe"] for r in rows])
                dd = np.array([r["dd"] for r in rows])
                ec = [event_concentration(bcv[i], cvs[i]) for i in range(len(rows))]
                ok = [x for x in ec if x["excess_meaningful"]]
                t5 = (f"{np.mean([x['top5_share'] for x in ok]):.0%}"
                      if len(ok) >= len(ec) // 2 else f"n/a({len(ok)}/{len(ec)})")
                nc, ns, v = sign_verdict(cc - bc, ss - bs, len(STARTS))
                print(f"  {name:<16}{cc.mean():>+9.2%}{ss.mean():>8.3f}{dd.mean():>8.1%}"
                      f"{dd.min():>8.1%}{np.mean(gs):>8.4f}{(cc-bc).mean()*100:>+7.2f}p"
                      f"{(ss-bs).mean():>+8.3f}{(dd-bd).mean()*100:>+7.2f}p{ns:>4}/{len(ss)}"
                      f"{int((dd>bd).sum()):>4}/{len(dd)}"
                      f"{cc.std(ddof=1)/bc.std(ddof=1):>8.3f}"
                      f"{np.mean(ms):>+12.3f}/{np.mean(md)*100:>+6.2f}p{t5:>9}", flush=True)
            print(f"    ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  DECISIVE COLUMN: 'MATCHED dShrp/dDD'. Arm E runs ~1.00x vs the deployed ~1.19x,",
          flush=True)
    print(f"  so some drawdown gain is pure de-levering. If the matched residual is ~0 or", flush=True)
    print(f"  negative, that is ALL it is, and the result is a level effect -- the same way", flush=True)
    print(f"  EXP-008 died (raw dMaxDD +12.29pp, matched dSharpe -0.099).", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
