"""EXP-007 (IDEAS I-23) — PARTIAL-ADJUSTMENT REBALANCING: the deployable form of EXP-001.

EXP-001 established that rebalance PHASE is worth 7.92pp of 8yr CAGR dispersion, entirely
uncompensated, and that averaging 4 phases removes about half of it. But 4 phases means netting
4 target books inside one IBKR account -- substantial new live code in an engine that has never
had more than one book.

This tests the cheap approximation: ONE book, rebalance every `rebal_days` sessions, move only
`move_frac` of the way to each new target.

  base            20d cadence, move 100%   (deployed)
  5d x 25%         5d cadence, move 25%
  10d x 50%       10d cadence, move 50%
  5d x 40%         5d cadence, move 40%
  10d x 35%       10d cadence, move 35%

WHY IT MIGHT WORK: same intent as tranching -- no single arbitrary session determines the book,
so the phase lottery is averaged away.

WHY IT MIGHT NOT, and this is the honest risk: it is NOT tranching. Tranching averages K
INDEPENDENT sub-books; partial adjustment smooths toward a MOVING target, which is a low-pass
filter on the signal. That introduces LAG -- and the repo has already established (DYNAMIC_
LEVERAGE_FINDINGS section 3) that being late with the BOOK is expensive even when being late
with leverage is not. It also raises trade frequency, and every trade pays full modelled cost
inside the sim.

PRE-REGISTERED PREDICTIONS
  1. sd of CAGR across starts DOWN (the whole point). If not, void.
  2. mean CAGR: slightly DOWN, from lag and extra trading. A large positive is a bug report.
  3. If sd falls close to what K=4 tranching achieved (ratio ~0.48) at a small CAGR cost,
     this is the thing to deploy rather than tranching.

Run:  python3 research/EXP007_partial_adjustment.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)

ARMS = [
    ("base 20d x100%", {}),
    ("5d x25%",  {"rebal_days": 5,  "move_frac": 0.25}),
    ("5d x40%",  {"rebal_days": 5,  "move_frac": 0.40}),
    ("10d x50%", {"rebal_days": 10, "move_frac": 0.50}),
    ("10d x35%", {"rebal_days": 10, "move_frac": 0.35}),
    ("20d x100% CONTROL(5d x100%)", {"rebal_days": 5}),   # cadence WITHOUT partial move
]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*96}\nEXP-007 PARTIAL-ADJUSTMENT REBALANCING — {HZ}, {len(STARTS)} starts"
          f"\n{'='*96}", flush=True)

    # PARITY GATE: move_frac=1.0 must be the untouched path, bit-for-bit
    a = bt.run(STARTS[0], END, dict(DEPLOYED))
    b = bt.run(STARTS[0], END, {**DEPLOYED, "move_frac": 1.0})
    same = bool((a["daily_values"].values == b["daily_values"].values).all())
    print(f"  PARITY  move_frac=1.0 vs untouched: {'IDENTICAL' if same else 'DIFFERENT'}  "
          f"({a['daily_values'].iloc[-1]:,.2f} vs {b['daily_values'].iloc[-1]:,.2f})", flush=True)
    if not same:
        print("  ABORT — the hook changed the default path.")
        return

    res = {n: [] for n, _ in ARMS}
    for si, st in enumerate(STARTS):
        for name, extra in ARMS:
            m = bt.run(st, END, {**DEPLOYED, **extra})
            res[name].append(stat(m["daily_values"]))
        print(f"  [{si+1:2d}/{len(STARTS)}] {st}  " +
              "  ".join(f"{res[n][-1]['cagr']:+6.2%}" for n, _ in ARMS) +
              f"   ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'arm':<30}{'meanCAGR':>10}{'sdCAGR':>9}{'Sharpe':>9}{'sdShrp':>8}"
          f"{'meanDD':>9}{'worstDD':>9}", flush=True)
    agg = {}
    for name, _ in ARMS:
        c = np.array([r["cagr"] for r in res[name]])
        s = np.array([r["sharpe"] for r in res[name]])
        d = np.array([r["dd"] for r in res[name]])
        agg[name] = (c, s, d)
        print(f"  {name:<30}{c.mean():>+10.2%}{c.std(ddof=1):>9.2%}{s.mean():>9.3f}"
              f"{s.std(ddof=1):>8.3f}{d.mean():>9.1%}{d.min():>9.1%}", flush=True)

    bc, bs, bd = agg["base 20d x100%"]
    print(f"\n  {'delta vs base':<30}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}"
          f"{'+CAGR':>8}{'+Shrp':>8}{'sdCAGR ratio':>14}", flush=True)
    for name, _ in ARMS[1:]:
        c, s, d = agg[name]
        print(f"  {name:<30}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
              f"{(d-bd).mean()*100:>+8.2f}p{int((c>bc).sum()):>5}/{len(c)}"
              f"{int((s>bs).sum()):>5}/{len(s)}{c.std(ddof=1)/bc.std(ddof=1):>13.3f}", flush=True)
    print(f"\n  reference: K=4 phase tranching achieved sdCAGR ratio 0.484 at "
          f"dCAGR +2.04pp / dSharpe +0.107 (8yr)", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
