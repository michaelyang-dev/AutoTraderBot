"""EXP-023 (IDEAS I-28) — MOMENTUM-SLEEVE UNIVERSE TIER. Escalation-ladder level 7.

MECHANISM
  Momentum's premium is documented to be larger in smaller, less-covered names: information
  diffuses more slowly, analyst coverage is thinner, and limits to arbitrage are tighter. The
  momentum sleeve currently takes its top-5 from ~1,100 eligible SP1500 names, so mega-caps and
  small-caps compete on the same score.

  WHO LOSES: in small caps, the marginal seller into a momentum run is more often a
  liquidity-constrained or attention-constrained holder than an informed one.

WHY THIS IS WORTH RUNNING RATHER THAN ASSUMING
  The repo already has a hard data point on the LARGE-cap end: restricting the momentum and
  lowvol sleeves to SP500-only was the B1 live bug, measured at **-15.7pp CAGR / -0.38 Sharpe**
  on a clean 8yr A/B. So the gradient is already known to run BROADER > narrower at the top.
  Nobody has tested the other end.

  And the capacity objection that normally kills small-cap tilts does not apply here: the live
  account is ~$33k. Position sizes are ~$500-1,500. Liquidity is a non-issue at this size, which
  is a genuine structural advantage this account has over institutional money -- one of the few.

IMPLEMENTATION
  `mom_universe` restricts ONLY the momentum and lowvol sleeves' membership, using the SAME PIT
  membership dicts the backtest already trades on. No new universe file, so no new survivorship
  risk. Value and the breadth calculation stay SP1500.

PRE-REGISTERED
  1. If sp400_600 (mid+small) beats sp1500 on BOTH horizons at >=9/12, that is a real finding.
  2. sp500 must reproduce roughly the known -15.7pp disaster, or my hook is wrong. This is the
     built-in correctness check -- an arm whose answer I already know.
  3. Watch MaxDD: small caps are higher-beta, so expect drawdown to worsen. A CAGR-only win is
     not a win (EXP-004/022: drawdown is the axis that matters).

Run:  python3 research/EXP023_universe_tier.py [8yr|26yr]
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
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}   # honest baseline, BUGS A9

ARMS = [
    ("sp500 (known -15.7pp)", {"mom_universe": "sp500"}),
    ("sp500_400",             {"mom_universe": "sp500_400"}),
    ("sp400 (mid only)",      {"mom_universe": "sp400"}),
    ("sp600 (small only)",    {"mom_universe": "sp600"}),
    ("sp400_600 (mid+small)", {"mom_universe": "sp400_600"}),
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*100}\nEXP-023 MOMENTUM UNIVERSE TIER — {HZ}, {len(STARTS)} starts, "
          f"gate ON, REAL financing\n{'='*100}", flush=True)

    a = bt.run(STARTS[0], END, {**DEPLOYED, **GATE})
    b = bt.run(STARTS[0], END, {**DEPLOYED, **GATE, "mom_universe": "sp1500"})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  PARITY mom_universe='sp1500' vs unset: {'IDENTICAL' if same else 'DIFFERENT'}",
          flush=True)
    if not same:
        print("  ABORT — hook changed the default path."); return

    base = []
    for st in STARTS:
        m = bt.run(st, END, {**DEPLOYED, **GATE, "financing_curve": True})
        base.append(stat(m["daily_values"]))
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE sp1500  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  "
          f"MaxDD {bd.mean():.1%}", flush=True)

    print(f"\n  {'arm':<24}{'CAGR':>9}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}"
          f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}{'+Shrp':>7}{'verdict':>10}",
          flush=True)
    for name, extra in ARMS:
        rows = []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, **GATE, **extra, "financing_curve": True})
            rows.append(stat(m["daily_values"]))
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        print(f"  {name:<24}{c.mean():>+9.2%}{s.mean():>9.3f}{so.mean():>9.3f}{d.mean():>9.1%}"
              f"{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}{(d-bd).mean()*100:>+8.2f}p"
              f"{nc:>4}/{len(c)}{ns:>4}/{len(s)}{v:>10}", flush=True)
    print(f"\n  CHECK: the sp500 arm should reproduce roughly the known -15.7pp CAGR live bug.",
          flush=True)
    print(f"         If it does not, the hook is wrong and nothing else here is valid.",
          flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
