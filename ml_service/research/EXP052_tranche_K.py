"""EXP-052 — TRANCHE COUNT SWEEP. The largest robust effect found; find its optimum and its cost.

WHY THIS IS THE HIGHEST-VALUE REMAINING TEST
  EXP-047's main effects put tranching FIRST: +0.065 Sharpe (8yr) and +0.033 (26yr), agreeing on
  both horizons -- larger than the overlay, the sleeve tilt or the leverage step, and the only one
  whose mechanism requires no alpha claim at all. The rebalance date carries ZERO information; it
  is a pure nuisance parameter, and per-start 8yr CAGR sigma of 7.07pp is the uncompensated risk
  being taken for free. Averaging K weakly-correlated phases should shrink that by roughly sqrt(K).

  But K=4 was never chosen -- it was assumed. If the sqrt(K) mechanism is right, K=10 should beat
  K=4. If it does not, the mechanism is wrong and the K=4 result needs re-examining even though it
  replicates.

THE COUNTERVAILING FORCE, which is why this is not just "more is better"
  At $50k with INTEGER SHARES, K tranches x 5 names means position size scales as 1/K:
      K=1   ~20 names  ~$3,400 each
      K=4   ~20 names  ~$2,750 each   (tranches share names)
      K=10  up to 50 names ~$1,100 each
  Integer-share truncation error grows as positions shrink, and the harness models this. So the
  sqrt(K) benefit and the granularity cost run in opposite directions and there must be an interior
  optimum. Finding WHERE it sits is the point -- and K=2 is dramatically easier to run live than
  K=4, so if K=2 captures most of the benefit that is the deployable answer.

PRE-REGISTERED READING
  - Benefit rises then falls, peak at K in 4..5   -> mechanism confirmed, granularity binds, ship the peak.
  - Benefit rises monotonically through K=10      -> granularity does NOT bind at $50k; but be
                                                     suspicious, because that contradicts integer-share
                                                     arithmetic and would suggest the cost model is inert.
  - Benefit flat or noisy across K                -> the K=4 result is NOT the sqrt(K) mechanism and
                                                     the whole tranching finding needs re-derivation.

Tested at leverage 1.10 AND 1.49 so the answer is not entangled with the leverage choice.
Overlay ON (EXP-047 restored it). Sleeves left at LIVE 50/35/15, since EXP-047 found the tilt is
noise on the 26yr (+0.002) and negative recently -- so tranching is measured on its own.

Run:  python3 research/EXP052_tranche_K.py [8yr|26yr]
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
PATH = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
        else "data/wrds/sp1500_universe_2000.pkl")
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13)]      # 24 -- K is the whole point,
CACHE = f"research/_tk_{HZ}"                                        # so measure sigma properly
SUBS = ([("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2025", 2023, 2025)]
        if HZ == "8yr" else
        [("2001-2008", 2001, 2008), ("2009-2016", 2009, 2016),
         ("2017-2022", 2017, 2022), ("2023-2025", 2023, 2025)])
KS = [(1, 20), (2, 10), (4, 5), (5, 4), (10, 2)]
LEVS = [1.10, 1.49]
BASE = dict(credit_pct=0.95, credit_derisk=0.50, initial_capital=50_000.0, vol_overlay=True,
            mom_w=.50, val_w=.35, lv_w=.15)


def _st(v):
    v = v.dropna()
    if len(v) < 60:
        return None
    r = v.pct_change()
    y = max((v.index[-1] - v.index[0]).days / 365.25, 0.25)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1,
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            float(((v - v.cummax()) / v.cummax()).min()))


def main():
    import inspect
    import textwrap
    from main_production_backtest import FastBacktester
    import VERIFY2_cleanroom as V
    from VERIFY2_cleanroom import CleanRoom
    src = inspect.getsource(CleanRoom.run)
    old = "dd=float(((v - v.cummax()) / v.cummax()).min()))"
    new = "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"
    assert src.count(old) == 1, "patch anchor missing -- refusing to guess"
    ns = dict(V.__dict__)
    exec(compile(textwrap.dedent(src.replace(old, new)), "<p>", "exec"), ns)
    CleanRoom.run = ns["run"]
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    cr = CleanRoom(FastBacktester(universe_path=PATH))
    for lv in LEVS:
        for k, stride in KS:
            nm = f"L{lv:.2f}_K{k:02d}"
            f = f"{CACHE}/{nm}.parquet"
            if os.path.exists(f):
                continue
            cfg = dict(BASE, leverage=lv, tranches=k, tranche_stride=stride)
            pd.DataFrame({s: cr.run(s, cfg)["curve"] for s in STARTS}).to_parquet(f)
            print(f"  {nm}  {time.time()-t0:>6.0f}s", flush=True)

    labs = [l for l, _, _ in SUBS]
    print(f"\n  ===== {HZ} — TRANCHE COUNT SWEEP ({len(STARTS)} starts) =====", flush=True)
    for lv in LEVS:
        print(f"\n  --- leverage {lv:.2f} ---", flush=True)
        print(f"  {'K':<5}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'sd(CAGR)':>10}"
              f"{'dShrp':>9}{'dMaxDD':>9}{'sd ratio':>10}   "
              + "".join(f"{l:>12}" for l in labs), flush=True)
        ref = None
        for k, _ in KS:
            f = f"{CACHE}/L{lv:.2f}_K{k:02d}.parquet"
            if not os.path.exists(f):
                continue
            df = pd.read_parquet(f)
            a = np.array([x for x in (_st(df[c]) for c in df.columns) if x])
            per = {}
            for lab, y0, y1 in SUBS:
                acc = []
                for c in df.columns:
                    v = df[c].dropna()
                    seg = v[(v.index >= pd.Timestamp(f"{y0}-01-01")) &
                            (v.index <= pd.Timestamp(f"{y1}-12-31"))]
                    x = _st(seg)
                    if x:
                        acc.append(x)
                if acc:
                    per[lab] = np.array(acc)
            sd = a[:, 0].std()
            if ref is None:
                ref = (a, sd, per)
                print(f"  {k:<5}{a[:,0].mean():>+9.2%}{a[:,1].mean():>9.3f}{a[:,2].mean():>9.1%}"
                      f"{sd*100:>9.2f}p" + " " * 28
                      + "".join(f"{0.0:>+12.3f}" for _ in labs), flush=True)
            else:
                ra, rsd, rper = ref
                print(f"  {k:<5}{a[:,0].mean():>+9.2%}{a[:,1].mean():>9.3f}{a[:,2].mean():>9.1%}"
                      f"{sd*100:>9.2f}p{a[:,1].mean()-ra[:,1].mean():>+9.3f}"
                      f"{(a[:,2].mean()-ra[:,2].mean())*100:>+8.2f}p{sd/rsd:>10.3f}   "
                      + "".join(f"{per[l][:,1].mean()-rper[l][:,1].mean():>+12.3f}"
                                if l in per and l in rper else f"{'-':>12}" for l in labs),
                      flush=True)
        print(f"    sqrt(K) prediction for sd ratio: "
              + "  ".join(f"K={k}:{1/np.sqrt(k):.3f}" for k, _ in KS[1:]), flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
