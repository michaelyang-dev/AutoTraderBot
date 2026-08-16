"""EXP-021 (IDEAS I-19) — RE-TEST THE 'DEAD' BATCH at the 12-start, two-horizon standard.

WHY THIS IS WORTH COMPUTE RATHER THAN A SHRUG
  BUGS F6: most "dead" verdicts in this repo were reached on 3-4 starts AND on the pre-audit
  poisoned universes (the rebuild moved 26yr CAGR by -10.4pp). That combination has already
  produced a WRONG SIGN once (the gross-margin bound, logged as -1.50pp, actually +1.17pp) and
  a RETRACTED win (value weights, 6/12). "Dead" is not a safe status for anything tested that
  way.

  Every idea here already has a config hook, so NO new modelling risk is introduced -- only the
  measurement standard changes. That makes this the cheapest possible conversion of "probably
  dead" into "known dead", and cheap negative knowledge is still knowledge: it stops these
  being re-proposed.

  Prior REVIVE run (threadREVIVE_dead_ideas.py) used a 2-stage design with 12 starts on the
  26yr and found NONE survived. This differs in three ways: honest time-varying financing, the
  guarded concentration metric, and per-arm sign consistency reported rather than a pass/fail.

IDEAS RE-TESTED
  vol-managed momentum (Barroso) at two targets, risk parity within sleeves, sector cap,
  momentum quality filters (gp_assets / roe), mid-cycle signal exit, cash parking in IEF,
  bull-lever.

PRE-REGISTERED
  Expect ~all to stay dead; that is the base rate and the point. Anything reaching >=9/12 on
  Sharpe on BOTH horizons gets promoted to its own audit gate, nothing less.

Run:  python3 research/EXP021_revive_batch.py [8yr|26yr]
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

ARMS = [
    ("vol-managed mom 0.20", {"vol_managed_mom": 0.20, "vmm_cap": 1.0}),
    ("vol-managed mom 0.30", {"vol_managed_mom": 0.30, "vmm_cap": 1.0}),
    ("risk parity in-sleeve", {"use_rp": True, "rp_power": 1.0}),
    ("sector cap 2/sleeve",  {"sector_cap": 2}),
    ("mom quality gp_assets", {"mom_quality_filter": "gp_assets", "mom_quality_pool": 3}),
    ("mom quality roe",      {"mom_quality_filter": "roe", "mom_quality_pool": 3}),
    ("signal-exit every 5d", {"signal_exit_every": 5}),
    ("park idle cash IEF",   {"park_etf": "IEF"}),
    ("bull lever 1.15x",     {"bull_lever": 1.15, "bull_breadth": 0.60}),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*100}\nEXP-021 REVIVE BATCH — {HZ}, {len(STARTS)} starts, REAL financing"
          f"\n{'='*100}", flush=True)

    base, bg = [], []
    for st in STARTS:
        m = bt.run(st, END, {**DEPLOYED, "financing_curve": True})
        base.append(stat(m["daily_values"])); bg.append(m["avg_gross"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  MaxDD {bd.mean():.1%}  "
          f"avgGross {np.mean(bg):.4f}", flush=True)

    print(f"\n  {'arm':<24}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'avgGross':>10}"
          f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}{'+Shrp':>7}{'verdict':>10}",
          flush=True)
    for name, extra in ARMS:
        try:
            rows, gs = [], []
            for st in STARTS:
                m = bt.run(st, END, {**DEPLOYED, **extra, "financing_curve": True})
                rows.append(stat(m["daily_values"])); gs.append(m["avg_gross"])
        except Exception as e:
            print(f"  {name:<24} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<24}{c.mean():>+9.2%}{s.mean():>9.3f}{d.mean():>9.1%}"
              f"{np.mean(gs):>10.4f}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
              f"{(d-bd).mean()*100:>+8.2f}p{nc:>4}/{len(c)}{ns:>4}/{len(s)}{v:>10}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
