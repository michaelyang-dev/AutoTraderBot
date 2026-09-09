"""FINAL_1.49 at 2x trading costs (the realistic cost level for a ~$60k account split into 4 books: IBKR $1 minimums make
commissions ~16bp of notional + ~5bp slippage vs the 10bp modelled). 8yr: 8 starts; 26yr: 4 starts."""
import os, sys, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research")
import importlib
for hz, k in [(sys.argv[1], 3 if sys.argv[1] == "8yr" else 6)]:
    sys.argv = ["x", hz]; import EXP057_final_on_v2 as E; importlib.reload(E)
    from main_production_backtest import FastBacktester
    CR = E._engine(); cr = CR(FastBacktester(universe_path=E.PATH)); starts = E.STARTS[::k]
    base = pd.read_parquet(f"{E.CACHE}/FINAL_1.49.parquet"); b = np.array([E.st(base[s]) for s in starts])
    r = np.array([[x["cagr"], x["sharpe"], x["dd"]] for x in (cr.run(s, dict(E.FIN, leverage=1.49, cost_mult=2.0)) for s in starts)])
    l2 = pd.read_parquet(f"{E.CACHE}/LIVE_cost2.parquet"); l = np.array([E.st(l2[s]) for s in starts])
    print(f"{hz} ({len(starts)} starts): FINAL_1.49 1x {b[:,0].mean():+.2%}/{b[:,1].mean():.3f}/{b[:,2].mean():.1%}  ->  2x costs {r[:,0].mean():+.2%}/{r[:,1].mean():.3f}/{r[:,2].mean():.1%}   | LIVE at 2x costs {l[:,0].mean():+.2%}/{l[:,1].mean():.3f}/{l[:,2].mean():.1%}   delta FINAL_1.49(2x)-LIVE(2x): {(r[:,0]-l[:,0]).mean()*100:+.2f}pp CAGR {(r[:,1]-l[:,1]).mean():+.3f} Sharpe {(r[:,2]-l[:,2]).mean()*100:+.1f}pp MaxDD", flush=True)
    del cr
