"""EXP-041 — FINAL VALIDATION of the leading configs. Year-by-year + cost sensitivity together.

LEADING CONFIGS after EXP-039 (26yr / 8yr, 12 starts, whole shares, real financing):
  REC = vol_scaling OFF + 4 tranches (5d stride) + credit gate cutting to 0
    @1.00  26yr +0.62pp CAGR / +0.098 Sharpe (11/12) / +15.17pp MaxDD
           8yr  -0.65pp CAGR / +0.079 Sharpe (7/12)  / +5.70pp MaxDD   <- CAGR negative on 8yr
    @1.10  26yr +1.38pp / +0.088 (11/12) / +11.58pp
           8yr  +0.90pp / +0.068 (7/12)  / +2.60pp                     <- ALL THREE positive, BOTH horizons
    @1.25  26yr +2.39pp / +0.075 (10/12) / +6.49pp
           8yr  +3.30pp / +0.059 (7/12)  / -1.70pp                     <- drawdown negative on 8yr

  @1.10 is the only setting positive on all three axes on both horizons, so it is the primary
  candidate. @1.00 and @1.25 are the risk-first and return-first alternatives.

WHY THIS TEST AND NOT ANOTHER
  Two arms in this program passed every aggregate test and then died year-by-year (tilt80 @1.00,
  O2+T70 @1.00 d0.20) -- both with clean monotone dose-responses. Aggregates are not evidence of
  breadth. And cost must be paired at the SAME multiplier (BUGS A8b) or it measures nothing.

  Both are run here, on the same configs, so the winner is decided on breadth and cost-robustness
  rather than on a headline.

PASS BAR (pre-registered)
  return: >60% of years won OR total excess survives dropping the best TWO years
  drawdown: >=50% of years won
  crisis years: not materially negative
  cost: dSharpe must not collapse from x1 to x3

Run:  python3 research/EXP041_final_validation.py [8yr|26yr]
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
LIVE = {**DEPLOYED, "credit_pct": 0.95, "credit_derisk": 0.50, "financing_curve": True,
        "live_sizing": True, "initial_capital": 50_000.0}
REC = {**LIVE, "vol_scaling": False, "tranches": 4, "tranche_stride": 5, "credit_derisk": 0.00}
T70 = {"mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09}
SE = {"signal_exit_every": 2}
# EXP-040 made REC+BOTH @1.75 the best all-round arm: positive on CAGR, Sharpe AND drawdown on
# BOTH horizons, at LOWER gross than LIVE. It has never had the year-by-year test -- and that
# test has killed 2 of 2 similar-looking arms (tilt80 @1.00, O2+T70 @1.00 d0.20), both of which
# had clean aggregates AND clean dose-responses. It leads.
CANDS = [("REC+BOTH @1.75", {**REC, "leverage": 1.75, **T70, **SE}),
         ("REC+sigexit @1.75", {**REC, "leverage": 1.75, **SE}),
         ("REC+tilt @1.25", {**REC, "leverage": 1.25, **T70}),
         ("REC @1.10", {**REC, "leverage": 1.10})]
CRISIS = {2001, 2002, 2008, 2020, 2022}


def yearly(v):
    return {int(y): (s.iloc[-1] / s.iloc[0] - 1,
                     float(((s - s.cummax()) / s.cummax()).min()))
            for y, s in v.groupby(v.index.year) if len(s) >= 20}


def main():
    t0 = time.time()
    bt = JointTrancheBacktester(universe_path=PATH)
    print(f"\n{'='*100}\nEXP-041 FINAL VALIDATION — {HZ}, {len(STARTS)} starts\n{'='*100}",
          flush=True)

    # ---------- PART 1: year-by-year breadth ----------
    accL, acc = {}, {n: {} for n, _ in CANDS}
    for si, st in enumerate(STARTS):
        L = yearly(bt.run(st, END, dict(LIVE))["daily_values"])
        for y, v in L.items():
            accL.setdefault(y, []).append(v)
        for n, cfg in CANDS:
            C = yearly(bt.run(st, END, dict(cfg))["daily_values"])
            for y in set(C) & set(L):
                acc[n].setdefault(y, []).append(C[y])
        print(f"  yby [{si+1:2d}/{len(STARTS)}] {st}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'candidate':<14}{'yrs won':>9}{'ddn won':>9}{'sum':>9}{'drop best':>11}"
          f"{'drop 2':>9}{'crisis':>9}{'VERDICT':>9}", flush=True)
    for n, _ in CANDS:
        d = {y: np.mean([x[0] for x in acc[n][y]]) - np.mean([x[0] for x in accL[y]])
             for y in sorted(acc[n])}
        dw = sum(1 for y in d if np.mean([x[1] for x in acc[n][y]])
                 > np.mean([x[1] for x in accL[y]]))
        ys = sorted(d); tot = sum(d.values()); srt = sorted(d.values())
        won = sum(1 for y in ys if d[y] > 0)
        cav = np.mean([d[y] for y in ys if y in CRISIS]) if any(y in CRISIS for y in ys) else 0.0
        broad = (won / len(ys) > 0.60) or (tot - srt[-1] - srt[-2] > 0)
        ok = broad and cav > -0.02 and dw / len(ys) >= 0.50
        print(f"  {n:<14}{won:>5}/{len(ys)}{dw:>6}/{len(ys)}{tot:>+9.1%}{tot-srt[-1]:>+11.1%}"
              f"{tot-srt[-1]-srt[-2]:>+9.1%}{cav:>+9.1%}{'PASS' if ok else 'FAIL':>9}",
              flush=True)

    # ---------- PART 2: cost sensitivity, PAIRED at each multiplier ----------
    print(f"\n  === COST SENSITIVITY (each arm paired vs LIVE at the SAME multiplier) ===",
          flush=True)
    print(f"  {'cost':<6}{'candidate':<14}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}"
          f"{'dShrp':>8}{'dMaxDD':>8}{'+Shrp':>7}", flush=True)
    for cm in (1, 2, 3):
        b = [stat(bt.run(st, END, {**LIVE, "cost_mult": cm})["daily_values"]) for st in STARTS]
        bc = np.array([r["cagr"] for r in b]); bs = np.array([r["sharpe"] for r in b])
        bd = np.array([r["dd"] for r in b])
        print(f"  {'x%d' % cm:<6}{'LIVE':<14}{bc.mean():>+9.2%}{bs.mean():>8.3f}"
              f"{bd.mean():>8.1%}", flush=True)
        for n, cfg in CANDS:
            r = [stat(bt.run(st, END, {**cfg, "cost_mult": cm})["daily_values"]) for st in STARTS]
            c = np.array([x["cagr"] for x in r]); s = np.array([x["sharpe"] for x in r])
            d = np.array([x["dd"] for x in r])
            _, ns, _ = sign_verdict(c - bc, s - bs, len(STARTS))
            print(f"  {'':<6}{n:<14}{c.mean():>+9.2%}{s.mean():>8.3f}{d.mean():>8.1%}"
                  f"{(c-bc).mean()*100:>+7.2f}p{(s-bs).mean():>+8.3f}"
                  f"{(d-bd).mean()*100:>+7.2f}p{ns:>4}/{len(s)}", flush=True)
        print(f"    ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
