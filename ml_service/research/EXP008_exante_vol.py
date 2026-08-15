"""EXP-008 (IDEAS I-03) — EX-ANTE, HOLDINGS-BASED PORTFOLIO-VOL ESTIMATE.

THE DEFECT
  Deployed `vol_scale` = clamp(target / trailing-40d-REALISED-vol-of-the-book). After a
  rebalance rotates into five different names, that estimate still describes the OLD book for
  up to 40 sessions. We size today's risk with last month's portfolio.

WHY THIS IS NOT ANOTHER DEAD TIMING RULE
  Every rule in DYNAMIC_LEVERAGE_FINDINGS -- EWMA vol, asymmetric windows, equity-curve trend,
  distance-from-peak, vol-of-vol, market vol, curve inversion, umd_crash -- tried to PREDICT
  when to be levered, and all are dead. This predicts nothing. It replaces a lagged measurement
  of a stale portfolio with a current measurement of the actual one:

      sigma_p^2 = rho*(sum w_i sigma_i)^2 + (1-rho)*sum (w_i sigma_i)^2

  using per-name sigma from the panel's `vol_60d` (already causal) and a FIXED average pairwise
  correlation. rho is fixed on purpose: correlation is slow-moving, composition is fast, and
  composition is the entire point. Sensitivity to rho is swept below.

MANDATORY CONTROL
  This changes realised gross exposure, so it MUST be scored against a constant-leverage curve
  interpolated at its OWN avg_gross. Without that, any estimator that happens to run less
  exposure looks like genius. evalkit.matched_exposure_curve does this.

PRE-REGISTERED
  1. Sharpe up on BOTH horizons AFTER the matched-exposure control, or it is dead.
  2. The gain should be largest where composition changes most -- i.e. it should NOT be
     concentrated in a handful of crisis days. Event-concentration checked.
  3. Robust across rho in [0.25, 0.45]; if only one rho works, it is fitted.

Run:  python3 research/EXP008_exante_vol.py [8yr|26yr]
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
                     matched_exposure_curve, event_concentration, sign_verdict)

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)

ARMS = [
    ("exante pure rho.35", {"exante_vol": "pure",  "exante_rho": 0.35}),
    ("exante pure rho.25", {"exante_vol": "pure",  "exante_rho": 0.25}),
    ("exante pure rho.45", {"exante_vol": "pure",  "exante_rho": 0.45}),
    ("exante blend rho.35", {"exante_vol": "blend", "exante_rho": 0.35}),
    ("exante pure vol20d", {"exante_vol": "pure", "exante_rho": 0.35,
                            "exante_vol_feat": "vol_20d"}),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*100}\nEXP-008 EX-ANTE VOL ESTIMATE — {HZ}, {len(STARTS)} starts\n{'='*100}",
          flush=True)

    # PARITY: no exante_vol key => untouched path, bit-for-bit
    a = bt.run(STARTS[0], END, dict(DEPLOYED))
    b = bt.run(STARTS[0], END, {**DEPLOYED, "exante_vol": None})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  PARITY exante_vol=None vs untouched: {'IDENTICAL' if same else 'DIFFERENT'}",
          flush=True)
    if not same:
        print("  ABORT — hook changed the default path.")
        return

    base, base_g, base_curves = [], [], []
    for st in STARTS:
        m = bt.run(st, END, dict(DEPLOYED))
        base.append(stat(m["daily_values"])); base_g.append(m["avg_gross"])
        base_curves.append(m["daily_values"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  MaxDD {bd.mean():.1%}  "
          f"avgGross {np.mean(base_g):.4f}", flush=True)

    print(f"\n  {'arm':<22}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'avgGr':>8}"
          f"{'dCAGR':>8}{'dShrp':>8}{'dDD':>8}{'+Shrp':>7}{'MATCHED dShrp':>15}"
          f"{'top5%':>8}", flush=True)
    for name, extra in ARMS:
        rows, grs, curves, mres = [], [], [], []
        for i, st in enumerate(STARTS):
            m = bt.run(st, END, {**DEPLOYED, **extra})
            rows.append(stat(m["daily_values"])); grs.append(m["avg_gross"])
            curves.append(m["daily_values"])
            mres.append(matched_exposure_curve(bt, st, END, m["avg_gross"]))
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        ms = np.array([rows[i]["sharpe"] - mres[i]["sharpe"] for i in range(len(rows))])
        mc = np.array([rows[i]["cagr"] - mres[i]["cagr"] for i in range(len(rows))])
        conc = [event_concentration(base_curves[i], curves[i]) for i in range(len(rows))]
        t5 = np.mean([x["top5_share"] for x in conc])
        nc, ns, verdict = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<22}{c.mean():>+9.2%}{s.mean():>8.3f}{d.mean():>8.1%}"
              f"{np.mean(grs):>8.4f}{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}"
              f"{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}"
              f"{ms.mean():>+11.3f} ({int((ms>0).sum())}/{len(ms)})"
              f"{t5:>8.0%}   {verdict}", flush=True)
        print(f"  {'':<22}matched dCAGR {mc.mean()*100:+.2f}pp", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
