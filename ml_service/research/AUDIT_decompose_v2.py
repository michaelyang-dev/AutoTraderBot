"""Decompose the deployed package on the v2 universes at the deployed 1.49x: LIVE -> +tranches -> +gate 0.00 -> +sleeves 70/21/9 (=FINAL).
8yr: 8 starts; 26yr: 6 starts. Cached FINAL_1.49 and LIVE reused for the same starts."""
import os, sys, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research")
HZ = sys.argv[1]; sys.argv = ["x", HZ]
import EXP057_final_on_v2 as E
from main_production_backtest import FastBacktester
CR = E._engine(); cr = CR(FastBacktester(universe_path=E.PATH)); starts = E.STARTS[::3 if HZ == "8yr" else 4]
L = pd.read_parquet(f"{E.CACHE}/LIVE.parquet"); F = pd.read_parquet(f"{E.CACHE}/FINAL_1.49.parquet")
def st(v): return E.st(v)
base = np.array([st(L[s]) for s in starts]); fin = np.array([st(F[s]) for s in starts])
arms = [("A: LIVE (1 book, gate .50, 50/35/15)", None, base),
        ("B: +tranches only (K=4/5d, gate .50, 50/35/15)", dict(E.LIVE, tranches=4, tranche_stride=5), None),
        ("C: +tranches +gate 0.00 (50/35/15)", dict(E.LIVE, tranches=4, tranche_stride=5, credit_derisk=0.0), None),
        ("D: FINAL = C + sleeves 70/21/9", None, fin)]
print(f"\n{HZ} ({len(starts)} starts) at 1.49x:\n  {'arm':<48}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'vs A: dSharpe':>14}{'+Shrp':>7}{'dMaxDD':>8}{'+DD':>6}")
for nm, cfg, arr in arms:
    a = arr if arr is not None else np.array([[r["cagr"], r["sharpe"], r["dd"]] for r in (cr.run(s, cfg) for s in starts)])
    d = a - base
    print(f"  {nm:<48}{a[:,0].mean():>+9.2%}{a[:,1].mean():>8.3f}{a[:,2].mean():>8.1%}{d[:,1].mean():>+14.3f}{int((d[:,1]>0).sum()):>4}/{len(starts)}{d[:,2].mean()*100:>+8.1f}p{int((d[:,2]>0).sum()):>3}/{len(starts)}", flush=True)
