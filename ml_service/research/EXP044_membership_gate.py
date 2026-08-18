"""EXP-044 — do the CORRUPT-MOMENTUM names ever pass the PIT membership gate?

THE TRIGGER (EXP-042)
  ~7.1% (8yr) / 5.6% (26yr) of top-5 momentum slots are filled by names carrying absurd 12-1
  momentum (>2000%). Roughly half look like real squeezes (GME 2021, MARA, HTZ, KOPN, CLSK).
  The other half look like DATA ARTEFACTS -- above all WW/WTW, which shows an identical
  15,925.6% ONE-DAY move in BOTH files. WW (Weight Watchers) and WTW (Willis Towers Watson) are
  two different companies; splicing their price series under one ticker manufactures exactly
  that jump, and manufactures a colossal momentum score with it.

WHY THIS COULD BE THE REAL THING
  Momentum is 50-70% of the book and the sleeve takes the TOP FIVE. A fabricated +15,000% return
  is not a small perturbation of a rank -- it is an automatic #1. If those names are investable,
  the backtest buys a fiction, and some unknown share of every momentum number in this program is
  an artefact of a ticker collision.

WHY IT MIGHT BE NOTHING
  The EXP-042 scan ranked the RAW price panel. The real sleeve first filters to PIT SP1500
  membership (plus dist_sma200>0 and liquidity). Most of these names are micro-caps -- ACY, ERNA,
  SOL, TBHC, SPP -- that were plausibly never index members. If membership excludes them, they
  never enter the pool and the finding collapses to a curiosity.

  That is the whole question, and it is a MEMBERSHIP question, not a price question.

ALSO: THE MEMBERSHIP CACHE
  Membership lives only inside the 44 GB pickle, so every check costs a full load. This dumps a
  compact boolean date x symbol matrix once; all later membership analysis is then free.

Run:  python3 research/EXP044_membership_gate.py [8yr|26yr] [dump|analyse]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
MODE = sys.argv[2] if len(sys.argv) > 2 else "dump"
PATH = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
        else "data/wrds/sp1500_universe_2000.pkl")
MEMF = f"research/_exp042/member_{HZ}.parquet"
PXF = {"8yr": "research/_exp042/px_8yrfile.parquet",
       "26yr": "research/_exp042/px_26yrfile.parquet"}[HZ]

# flagged by EXP-042; the "real" set is documented so the judgement is auditable, not hidden
REAL_MOVES = {"GME", "MARA", "HTZ", "RGC", "KOPN", "BNED", "TUES", "EQ", "CLSK", "PENN"}


def dump():
    from main_production_backtest import FastBacktester
    os.makedirs("research/_exp042", exist_ok=True)
    bt = FastBacktester(universe_path=PATH)
    dates = [d for d in bt.prices.index if d >= pd.Timestamp("2017-01-01")]
    syms = list(bt.prices.columns)
    idx = {s: i for i, s in enumerate(syms)}
    M = np.zeros((len(dates), len(syms)), dtype=bool)
    for i, d in enumerate(dates):
        for s in bt._get_sp1500(d):
            j = idx.get(s)
            if j is not None:
                M[i, j] = True
    pd.DataFrame(M, index=pd.DatetimeIndex(dates), columns=syms).to_parquet(MEMF)
    print(f"  saved membership {M.shape}, mean members/day {M.sum(1).mean():.0f}", flush=True)


def analyse():
    mem = pd.read_parquet(MEMF)
    px = pd.read_parquet(PXF)
    d = mem.index.intersection(px.index)
    s = mem.columns.intersection(px.columns)
    mem, px = mem.loc[d, s], px.loc[d, s]
    m = px.shift(20) / px.shift(252) - 1.0
    absurd = (m > 20.0)
    syms = sorted(set(m.columns[absurd.any()]))
    print(f"\n  ===== {HZ}: {len(syms)} names carry >2000% 12-1 momentum =====", flush=True)
    print(f"  mean SP1500 members/day: {mem.sum(1).mean():.0f}\n", flush=True)
    print(f"  {'sym':<8}{'absurd days':>13}{'of which MEMBER':>18}{'verdict':>34}", flush=True)
    reach = 0
    for x in syms:
        n = int(absurd[x].sum())
        k = int((absurd[x] & mem[x]).sum())
        reach += k
        if k == 0:
            v = "NEVER a member -> harmless"
        elif x in REAL_MOVES:
            v = "member, but a REAL move"
        else:
            v = "*** MEMBER + ARTEFACT: REACHES BOOK ***"
        print(f"  {x:<8}{n:>13}{k:>18}{v:>34}", flush=True)

    # the number that matters: share of investable top-5 momentum slots taken by an artefact
    mm = m.where(px.notna() & m.notna() & mem)          # membership-gated, as the sleeve sees it
    rank = mm.rank(axis=1, ascending=False)
    top5 = rank <= 5
    tot = int(top5.values.sum())
    art = [x for x in syms if x not in REAL_MOVES]
    n_art = int(sum((top5[x] & (mm[x] > 20.0)).sum() for x in art))
    n_real = int(sum((top5[x] & (mm[x] > 20.0)).sum() for x in syms if x in REAL_MOVES))
    print(f"\n  MEMBERSHIP-GATED top-5 momentum slots: {tot:,}", flush=True)
    print(f"    taken by a REAL extreme move   : {n_real:>6} ({n_real/tot:.4%})", flush=True)
    print(f"    taken by a LIKELY ARTEFACT     : {n_art:>6} ({n_art/tot:.4%})  <-- the exposure",
          flush=True)
    print(f"  (EXP-042's ungated figure was 7.07% / 5.60%; membership removes "
          f"{100*(1-(n_art+n_real)/max(tot,1)/0.0707):.0f}% of it if this is far lower)",
          flush=True)


if __name__ == "__main__":
    t0 = time.time()
    dump() if MODE == "dump" else analyse()
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
