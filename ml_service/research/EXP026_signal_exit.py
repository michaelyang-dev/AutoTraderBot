"""EXP-026 (IDEAS I-29) — MID-CYCLE SIGNAL EXIT as a drawdown instrument.

WHERE THIS CAME FROM
  EXP-021, 26yr, 12 starts: `signal_exit_every=5` gave
      MaxDD -64.1% -> -52.7%  (+11.39pp)   Sharpe +0.020 (8/12)   CAGR -1.14pp
  That is the LARGEST single drawdown improvement any arm has produced in this program, on the
  axis EXP-004/EXP-022 identified as the one that matters.

MECHANISM
  The deployed book holds a name for up to 20 sessions after the signal that bought it has
  gone. In a fast decline that is 20 sessions of holding something no sleeve wants any more.
  Exiting is CHEAP -- it is a sale, not a re-pick.

  This is exactly the distinction EXP-007 established and it is why I expect this to work where
  partial-adjustment failed: **cheap adjustments can be prompt, expensive ones cannot.**
  Re-picking the whole book mid-cycle was catastrophic (-7 to -10pp CAGR). Merely EXITING a
  name the model has abandoned is a different operation with a different cost profile.
  DYNAMIC_LEVERAGE_FINDINGS section 3 says the same thing about leverage vs rebalance cadence.

WHY IT MIGHT BE NOTHING
  1. -1.14pp of CAGR is a real cost and sign consistency was only 8/12.
  2. It creates cash that sits idle until the next rebalance (recycling was tested dead
     previously), so part of the drawdown gain is simply LOWER AVERAGE EXPOSURE -- which the
     matched-exposure control exists to strip out. That is the single most likely way this is
     an illusion, and it is tested here rather than assumed away.
  3. It needs new live code (a daily exit check between rebalances).

SWEEP
  cadence 3 / 5 / 10 sessions, and `signal_exit_grace` (tolerate names still inside the
  momentum top-K even if outside the top-5 picks) at None / 10 / 20. A real effect should not
  need one exact cadence.

Run:  python3 research/EXP026_signal_exit.py [8yr|26yr]
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
BASE = {**DEPLOYED, **GATE, "financing_curve": True}

ARMS = [("exit every 3d", {"signal_exit_every": 3}),
        ("exit every 5d", {"signal_exit_every": 5}),
        ("exit every 10d", {"signal_exit_every": 10}),
        ("exit 5d grace10", {"signal_exit_every": 5, "signal_exit_grace": 10}),
        ("exit 5d grace20", {"signal_exit_every": 5, "signal_exit_grace": 20}),
        ("exit 5d cost x2", {"signal_exit_every": 5, "cost_mult": 2})]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*112}\nEXP-026 MID-CYCLE SIGNAL EXIT — {HZ}, {len(STARTS)} starts, gate ON"
          f"\n{'='*112}", flush=True)

    base, bcv, bg = [], [], []
    for st in STARTS:
        m = bt.run(st, END, BASE)
        base.append(stat(m["daily_values"])); bcv.append(m["daily_values"])
        bg.append(m["avg_gross"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  MaxDD {bd.mean():.1%}  "
          f"worst {bd.min():.1%}  gross {np.mean(bg):.4f}", flush=True)

    print(f"\n  {'arm':<18}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'worst':>8}{'gross':>8}"
          f"{'dCAGR':>8}{'dShrp':>8}{'dMaxDD':>8}{'+CAGR':>7}{'+Shrp':>7}{'+DD':>7}"
          f"{'MATCHED dShrp/dDD':>20}{'verdict':>10}", flush=True)
    for name, extra in ARMS:
        cfg = {**BASE, **extra}
        cm = extra.get("cost_mult", 1)
        # cost arms must be paired against the base at the SAME multiplier (BUGS A8b)
        if cm != 1:
            b2 = [stat(bt.run(st, END, {**BASE, "cost_mult": cm})["daily_values"])
                  for st in STARTS]
            rc = np.array([r["cagr"] for r in b2]); rs = np.array([r["sharpe"] for r in b2])
            rd = np.array([r["dd"] for r in b2])
        else:
            rc, rs, rd = bc, bs, bd
        rows, gs, ms, md = [], [], [], []
        for st in STARTS:
            m = bt.run(st, END, cfg)
            r = stat(m["daily_values"]); rows.append(r); gs.append(m["avg_gross"])
            ref = matched_exposure_curve(bt, st, END, m["avg_gross"], cfg=BASE)
            ms.append(r["sharpe"] - ref["sharpe"]); md.append(r["dd"] - ref["dd"])
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - rc, s - rs, len(STARTS))
        print(f"  {name:<18}{c.mean():>+9.2%}{s.mean():>8.3f}{d.mean():>8.1%}{d.min():>8.1%}"
              f"{np.mean(gs):>8.4f}{(c-rc).mean()*100:>+7.2f}p{(s-rs).mean():>+8.3f}"
              f"{(d-rd).mean()*100:>+7.2f}p{nc:>4}/{len(c)}{ns:>4}/{len(s)}"
              f"{int((d>rd).sum()):>4}/{len(d)}"
              f"{np.mean(ms):>+11.3f}/{np.mean(md)*100:>+6.2f}p{v:>10}", flush=True)

    print(f"\n  THE KEY COLUMN IS 'MATCHED dShrp/dDD'. Signal-exit leaves cash idle until the",
          flush=True)
    print(f"  next rebalance, so it LOWERS average exposure. If the drawdown gain vanishes at",
          flush=True)
    print(f"  matched exposure it is a level effect, not a timing skill -- exactly what killed",
          flush=True)
    print(f"  the ex-ante vol estimator in EXP-008 (raw dMaxDD +12pp, matched dSharpe -0.09).",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
