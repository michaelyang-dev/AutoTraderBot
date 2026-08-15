"""EXP-001b — the CONTROL that EXP-001 was missing.

EXP-001 compared K tranches of $12,500 against ONE book of $50,000 and the tranched arm came
out ~3.5pp HIGHER. My own pre-registered prediction was |delta| < 1pp and that a large positive
delta is a bug report. It is: the comparison confounds TWO changes.

  (1) PHASE      -- 4 rebalance phases instead of 1   <- the thing being tested
  (2) CAPITAL    -- $12,500 per book instead of $50,000

(2) is not a nuisance, it changes the STRATEGY. With integer shares at $12.5k, a large share of
target positions round to zero and are dropped, so the tranche holds fewer, cheaper names. That
is a different portfolio, not the same portfolio on a different day.

This script separates them:

    capital effect = base@12.5k        - base@50k
    PHASE effect   = K4@12.5k          - base@12.5k      <-- the only number that tests I-01
    total          = K4@12.5k          - base@50k        (what EXP-001 reported)

and runs the whole thing twice: once under the harness's default sizing, and once under
`live_sizing=True`, the faithful port of `ibkr_engine._calibrate_quantities`. That matters here
more than anywhere else: live closed-loops a multiplier over 4 passes until ACTUAL gross after
whole-share truncation hits the target, i.e. it RECOVERS most of the rounding drag. The default
harness path does not. So the default path systematically overstates the cost of running smaller
books -- exactly the cost this experiment turns on.

Run:  python3 research/EXP001b_phase_with_capital_control.py [8yr|26yr]
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
TOTAL = 50_000.0
PHASES = [0, 5, 10, 15]
K = len(PHASES)
CAP_K = TOTAL / K


def combine(curves):
    idx = curves[0].index
    return sum(c.reindex(idx).ffill() for c in curves)


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*100}\nEXP-001b PHASE vs CAPITAL — {HZ}, {len(STARTS)} starts, "
          f"total ${TOTAL:,.0f}, K={K} phases {PHASES}\n{'='*100}", flush=True)

    arms = {k: [] for k in ("base50", "base12", "K4",
                            "L_base50", "L_base12", "L_K4")}
    phase_cagr = {p: [] for p in PHASES}

    for si, st in enumerate(STARTS):
        for tag, ls in (("", False), ("L_", True)):
            extra = {"live_sizing": True} if ls else {}
            m = bt.run(st, END, {**DEPLOYED, **extra, "initial_capital": TOTAL})
            arms[tag + "base50"].append(stat(m["daily_values"]))
            m = bt.run(st, END, {**DEPLOYED, **extra, "initial_capital": CAP_K})
            arms[tag + "base12"].append(stat(m["daily_values"]))
            cur = []
            for p in PHASES:
                mm = bt.run(st, END, {**DEPLOYED, **extra,
                                      "initial_capital": CAP_K, "rebal_phase": p})
                cur.append(mm["daily_values"])
                if not ls:
                    phase_cagr[p].append(stat(mm["daily_values"])["cagr"])
            arms[tag + "K4"].append(stat(combine(cur)))
        print(f"  [{si+1:2d}/{len(STARTS)}] {st}  "
              f"b50 {arms['base50'][-1]['cagr']:+7.2%} b12 {arms['base12'][-1]['cagr']:+7.2%} "
              f"K4 {arms['K4'][-1]['cagr']:+7.2%}  |  "
              f"Lb50 {arms['L_base50'][-1]['cagr']:+7.2%} "
              f"Lb12 {arms['L_base12'][-1]['cagr']:+7.2%} "
              f"LK4 {arms['L_K4'][-1]['cagr']:+7.2%}   ({time.time()-t0:.0f}s)", flush=True)

    def col(tag, key):
        return np.array([r[key] for r in arms[tag]])

    print(f"\n  {'arm':<12}{'meanCAGR':>10}{'sdCAGR':>9}{'meanShrp':>10}{'sdShrp':>9}"
          f"{'meanDD':>9}{'worstDD':>9}", flush=True)
    for tag in ("base50", "base12", "K4", "L_base50", "L_base12", "L_K4"):
        c, s, d = col(tag, "cagr"), col(tag, "sharpe"), col(tag, "dd")
        print(f"  {tag:<12}{c.mean():>+10.2%}{c.std(ddof=1):>9.2%}{s.mean():>10.3f}"
              f"{s.std(ddof=1):>9.3f}{d.mean():>9.1%}{d.min():>9.1%}", flush=True)

    print(f"\n  {'DECOMPOSITION':<34}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}"
          f"{'+CAGR':>8}{'+Shrp':>8}{'sdCAGR ratio':>14}", flush=True)
    for name, a, b in (("capital  base12 - base50", "base12", "base50"),
                       ("PHASE    K4     - base12", "K4", "base12"),
                       ("total    K4     - base50", "K4", "base50"),
                       ("[live] capital  L_b12-L_b50", "L_base12", "L_base50"),
                       ("[live] PHASE    L_K4 -L_b12", "L_K4", "L_base12"),
                       ("[live] total    L_K4 -L_b50", "L_K4", "L_base50")):
        ca, cb = col(a, "cagr"), col(b, "cagr")
        sa, sb = col(a, "sharpe"), col(b, "sharpe")
        da, db = col(a, "dd"), col(b, "dd")
        print(f"  {name:<34}{(ca-cb).mean()*100:>+8.2f}p{(sa-sb).mean():>+9.3f}"
              f"{(da-db).mean()*100:>+8.2f}p{int((ca>cb).sum()):>5}/{len(ca)}"
              f"{int((sa>sb).sum()):>5}/{len(sa)}"
              f"{ca.std(ddof=1)/cb.std(ddof=1):>13.3f}", flush=True)
    print(f"\n  variance-reduction target for K={K}: sdCAGR ratio ~ {1/np.sqrt(K):.3f}",
          flush=True)

    pm = np.array([phase_cagr[p] for p in PHASES])
    print(f"\n  --- phase risk, measured with START and CAPITAL both held constant ---",
          flush=True)
    print("  per-phase mean CAGR : " +
          "  ".join(f"p{p}={pm[i].mean():+.2%}" for i, p in enumerate(PHASES)), flush=True)
    print(f"  cross-phase sd at a FIXED start: mean {pm.std(axis=0, ddof=1).mean():.2%}  "
          f"max {pm.std(axis=0, ddof=1).max():.2%}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
