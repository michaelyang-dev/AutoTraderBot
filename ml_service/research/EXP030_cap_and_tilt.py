"""EXP-030 (IDEAS I-02 + I-27) — position-cap mechanics, and the momentum tilt.

I-02 CAP MODE. The deployed code truncates any weight above the 10% cap and the excess goes
  NOWHERE -- realised gross just falls. So on days when the cap binds, the book is
  simultaneously (a) capped and (b) under-invested, and nobody chose (b).
  `cap_mode="waterfill"` redistributes the truncated excess across names still under the cap,
  iterating to convergence, then rescales to the pre-cap total. The limit is respected AND the
  intended exposure is preserved.
  This is ALSO a backtest-vs-live parity question: the live engine's closed-loop
  `_calibrate_quantities` grows its multiplier until realised gross hits target, which
  effectively redistributes. So the deployed BACKTEST may be under-investing relative to the
  deployed ENGINE on exactly the days the cap binds. That is the BUGS B-class pattern.
  If the effect is ~0, the cap rarely binds and the long-standing docstring caveat can be closed.

I-27 MOMENTUM TILT. EXP-020 found a monotone dose-response: tilting toward equal RISK across
  sleeves is worse on the 8yr (-0.015 / -0.032 / -0.053 Sharpe as the tilt strengthens) and
  Sharpe-neutral on the 26yr. The gradient points the OTHER way, and EXP-003 gives the reason --
  the ranking carries 100% of the selection edge and momentum is the ranked sleeve, while value
  and lowvol behave close to random draws from a filtered pool.
  So: push capital TOWARD momentum. 50/35/15 -> 60/28/12 -> 70/21/9 -> 80/14/6.
  DANGER, stated up front: momentum is the highest-vol sleeve, so every tilt raises drawdown --
  the axis that matters. And sleeve-weight changes have been RETRACTED as noise in this repo
  before (CORE2/3/4 at 6/12). Judge on Sharpe AND MaxDD, >=9/12, both horizons.

Run:  python3 research/EXP030_cap_and_tilt.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
BASE = {**DEPLOYED, **GATE, "financing_curve": True}

ARMS = [("cap waterfill", {"cap_mode": "waterfill"}),
        ("cap 0.15 (looser)", {"cap": 0.15}),
        ("cap 0.07 (tighter)", {"cap": 0.07}),
        ("tilt 60/28/12", {"mom_w": 0.60, "val_w": 0.28, "lv_w": 0.12}),
        ("tilt 70/21/9", {"mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09}),
        ("tilt 80/14/6", {"mom_w": 0.80, "val_w": 0.14, "lv_w": 0.06}),
        ("tilt 40/42/18 (inverse)", {"mom_w": 0.40, "val_w": 0.42, "lv_w": 0.18})]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*104}\nEXP-030 CAP MECHANICS + MOMENTUM TILT — {HZ}, {len(STARTS)} starts, "
          f"gate ON\n{'='*104}", flush=True)

    a = bt.run(STARTS[0], END, BASE)
    b = bt.run(STARTS[0], END, {**BASE, "cap_mode": "truncate"})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  PARITY cap_mode unset vs 'truncate': {'IDENTICAL' if same else 'DIFFERENT'}",
          flush=True)

    base, bg = [], []
    for st in STARTS:
        m = bt.run(st, END, BASE)
        base.append(stat(m["daily_values"])); bg.append(m["avg_gross"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  MaxDD {bd.mean():.1%}  "
          f"gross {np.mean(bg):.4f}", flush=True)

    print(f"\n  {'arm':<26}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'gross':>9}{'dCAGR':>9}"
          f"{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}{'+Shrp':>7}{'verdict':>10}", flush=True)
    for name, extra in ARMS:
        rows, gs = [], []
        for st in STARTS:
            m = bt.run(st, END, {**BASE, **extra})
            rows.append(stat(m["daily_values"])); gs.append(m["avg_gross"])
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<26}{c.mean():>+9.2%}{s.mean():>9.3f}{d.mean():>9.1%}"
              f"{np.mean(gs):>9.4f}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
              f"{(d-bd).mean()*100:>+8.2f}p{nc:>4}/{len(c)}{ns:>4}/{len(s)}{v:>10}", flush=True)
    print(f"\n  cap waterfill: if avgGross RISES and metrics barely move, the cap binds but the",
          flush=True)
    print(f"  under-investment it causes is immaterial -> close the docstring caveat.", flush=True)
    print(f"  If gross is UNCHANGED, the cap never binds at all -> caveat is moot.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
