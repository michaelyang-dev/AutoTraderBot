"""VERIFY4 — the user's chosen config: 1.25x leverage, CREDIT GATE RETAINED, year-by-year.

Runs on the CLEAN-ROOM independent simulator (VERIFY2), which is imported and left BYTE-
IDENTICAL on disk -- the one line it needs (return the NAV curve, not just the summary stats)
is patched in memory at import time. The audited engine is not modified.

Three arms:
  LIVE            1.49x + vol overlay, gate derisk 0.50, sleeves 50/35/15, 1 book/20d
  REC gate0.50    1.25x, NO overlay, gate derisk 0.50  <- KEEPS THE LIVE GATE (user's pick)
  REC gate0.00    1.25x, NO overlay, gate derisk 0.00  <- what I had recommended

Year-by-year protocol: a single start date would let the 20-session rebalance PHASE drive the
per-year numbers (measured sigma 7.92pp of 8yr CAGR). So every year is averaged over ALL 24
start dates -- holdout AND used -- counting a year only for curves that cover it end to end.
The per-year win count is PAIRWISE per start, not a comparison of the two averages.

Run:  python3 research/VERIFY4_yearly.py [8yr|26yr]
"""
import inspect
import textwrap
import sys
import time

import numpy as np
import pandas as pd

import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom, HOLDOUT, USED, PATH, HZ
from main_production_backtest import FastBacktester

# --- in-memory patch: expose the NAV curve. VERIFY2_cleanroom.py on disk is untouched. ---
_src = inspect.getsource(CleanRoom.run)
_old = "dd=float(((v - v.cummax()) / v.cummax()).min()))"
_new = "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"
assert _src.count(_old) == 1, "patch anchor missing/ambiguous -- refusing to guess"
_ns = dict(V.__dict__)
exec(compile(textwrap.dedent(_src.replace(_old, _new)), "<patched>", "exec"), _ns)
CleanRoom.run = _ns["run"]

STARTS = USED + HOLDOUT          # all 24


def yearly(curve):
    """Calendar-year total returns, only for years the curve covers end to end."""
    out = {}
    for y, seg in curve.groupby(curve.index.year):
        if len(seg) < 200:                       # partial year at either end -- skip
            continue
        prev = curve[curve.index < pd.Timestamp(f"{y}-01-01")]
        if prev.empty:                           # no true opening mark for this year
            continue
        out[y] = seg.iloc[-1] / prev.iloc[-1] - 1
    return out


def main():
    t0 = time.time()
    bt = FastBacktester(universe_path=PATH)
    cr = CleanRoom(bt)

    LIVE = dict(leverage=1.49, tranches=1, tranche_stride=20, credit_pct=0.95,
                credit_derisk=0.50, mom_w=.50, val_w=.35, lv_w=.15,
                initial_capital=50_000.0, vol_overlay=True)
    REC = dict(leverage=1.25, tranches=4, tranche_stride=5, credit_pct=0.95,
               mom_w=.70, val_w=.21, lv_w=.09, initial_capital=50_000.0,
               vol_overlay=False)
    ARMS = [("LIVE", LIVE),
            ("REC gate0.50", {**REC, "credit_derisk": 0.50}),
            ("REC gate0.00", {**REC, "credit_derisk": 0.00})]

    res = {}
    for nm, cfg in ARMS:
        runs = [cr.run(s, cfg) for s in STARTS]
        res[nm] = runs
        print(f"  ran {nm:<14} {time.time()-t0:.0f}s", flush=True)

    print(f"\n  ===== {HZ} — HEADLINE ({len(STARTS)} starts: 12 used + 12 untouched) =====",
          flush=True)
    print(f"  {'arm':<14}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}"
          f"{'dMaxDD':>9}{'+Shrp':>8}{'+CAGR':>8}", flush=True)
    base = res["LIVE"]
    bc = np.array([x["cagr"] for x in base]); bs = np.array([x["sharpe"] for x in base])
    bd = np.array([x["dd"] for x in base])
    for nm, _ in ARMS:
        r = res[nm]
        c = np.array([x["cagr"] for x in r]); s = np.array([x["sharpe"] for x in r])
        d = np.array([x["dd"] for x in r])
        if nm == "LIVE":
            print(f"  {nm:<14}{c.mean():>+10.2%}{s.mean():>9.3f}{d.mean():>9.1%}", flush=True)
        else:
            print(f"  {nm:<14}{c.mean():>+10.2%}{s.mean():>9.3f}{d.mean():>9.1%}"
                  f"{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
                  f"{(d-bd).mean()*100:>+8.2f}p"
                  f"{int((s>bs).sum()):>5}/{len(r)}{int((c>bc).sum()):>5}/{len(r)}", flush=True)

    # ---------------- year by year ----------------
    for nm in ("REC gate0.50", "REC gate0.00"):
        print(f"\n  ===== {HZ} — YEAR BY YEAR: {nm} vs LIVE =====", flush=True)
        print(f"  {'year':<7}{'LIVE':>10}{'REC':>10}{'diff':>10}{'REC wins':>11}", flush=True)
        yl = [yearly(x["curve"]) for x in res["LIVE"]]
        yv = [yearly(x["curve"]) for x in res[nm]]
        years = sorted(set().union(*[set(d) for d in yl]))
        wins = tot = 0
        rows = []
        for y in years:
            pairs = [(a[y], b[y]) for a, b in zip(yl, yv) if y in a and y in b]
            if not pairs:
                continue
            L = float(np.mean([p[0] for p in pairs])); R = float(np.mean([p[1] for p in pairs]))
            w = sum(1 for a, b in pairs if b > a)
            wins += (R > L); tot += 1
            rows.append((y, L, R, R - L, w, len(pairs)))
            flag = "  <-- REC worse" if R < L else ""
            print(f"  {y:<7}{L:>+10.2%}{R:>+10.2%}{(R-L)*100:>+9.2f}p"
                  f"{w:>7}/{len(pairs)}{flag}", flush=True)
        diffs = np.array([r[3] for r in rows])
        print(f"  {'-'*50}", flush=True)
        print(f"  years REC beats LIVE (mean): {wins}/{tot}   "
              f"median diff {np.median(diffs)*100:+.2f}pp   "
              f"mean diff {diffs.mean()*100:+.2f}pp", flush=True)
        print(f"  worst year for REC vs LIVE: {rows[int(np.argmin(diffs))][0]} "
              f"({diffs.min()*100:+.2f}pp)   best: {rows[int(np.argmax(diffs))][0]} "
              f"({diffs.max()*100:+.2f}pp)", flush=True)
        # is the total gain just one year? drop the single best year and re-check.
        k = int(np.argmax(diffs))
        rest = np.delete(diffs, k)
        print(f"  DROP-BEST-YEAR: without {rows[k][0]}, mean diff {rest.mean()*100:+.2f}pp, "
              f"still positive in {int((rest>0).sum())}/{len(rest)} years", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
