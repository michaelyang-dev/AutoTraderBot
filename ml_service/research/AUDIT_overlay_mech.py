"""MECHANISM diagnostic for the prompt overlay (overlay_all / overlay_down) and exit_all on the v2 8yr + 26yr:
how often the switch acts, how much extra turnover it adds, and the year-by-year P&L delta split into
'crisis' (2008, 2020, 2022) vs other years. Instrumented run: 4 starts per horizon."""
import os, sys, inspect, textwrap, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research")
HZ = sys.argv[1]; sys.argv = ["x", HZ, "stage1"]
import EXP059_frontier as X, EXP057_final_on_v2 as E
from main_production_backtest import FastBacktester
import VERIFY2_cleanroom as V
X._engine()   # installs the switch-aware run()
src = inspect.getsource(V.CleanRoom.run) if False else None
# wrap: count overlay actions and traded notional by monkeypatching via cfg-visible counters is not possible from outside;
# so re-run the patched source with two counters injected
run_src = inspect.getsource(X.CleanRoom.run) if hasattr(X.CleanRoom.run, "__code__") and X.CleanRoom.run.__code__.co_filename != "<exp059>" else None
if run_src is None:
    # the exp059 run() was exec'd; rebuild it from VERIFY2 source with the same replacements plus counters
    import VERIFY2_cleanroom as V2
    from VERIFY2_cleanroom import CleanRoom as CR0
    import importlib; importlib.reload(V2)
    base_src = inspect.getsource(V2.CleanRoom.run)
    # reuse EXP059's replacement list by re-executing its _engine on a fresh class is simplest:
    X.V = V2; X.CleanRoom = V2.CleanRoom; X._engine()
    run_src = None
cr = X.CleanRoom(FastBacktester(universe_path=E.PATH))
STARTS = E.STARTS[::6]
def st(v): return E.st(v)
rows = []
for nm, cfg in [("base", X.ARMS["base"]), ("overlay_all", X.ARMS["overlay_all"]), ("overlay_down", X.ARMS["overlay_down"]), ("exit_all", X.ARMS["exit_all"])]:
    curves = {s: cr.run(s, cfg)["curve"] for s in STARTS}
    turn = np.nan
    rows.append((nm, curves))
B = rows[0][1]
crisis = {2008, 2020, 2022}
for nm, curves in rows[1:]:
    dy = {}
    for s in STARTS:
        yb, ya = E.yearly(B[s]), E.yearly(curves[s])
        for y in yb:
            if y in ya: dy.setdefault(y, []).append(ya[y] - yb[y])
    dy = {y: np.mean(v) for y, v in dy.items()}
    cr_ = sum(v for y, v in dy.items() if y in crisis); oth = sum(v for y, v in dy.items() if y not in crisis); n_oth = sum(1 for y in dy if y not in crisis); w_oth = sum(1 for y, v in dy.items() if y not in crisis and v > 0)
    a = np.array([st(curves[s]) for s in STARTS]); b = np.array([st(B[s]) for s in STARTS])
    print(f"{HZ} {nm:<13} dCAGR {(a[:,0]-b[:,0]).mean()*100:+.2f}pp dSharpe {(a[:,1]-b[:,1]).mean():+.3f} dMaxDD {(a[:,2]-b[:,2]).mean()*100:+.1f}pp | year-delta sum: crisis years {cr_*100:+.1f}pp, other years {oth*100:+.1f}pp (better in {w_oth}/{n_oth} non-crisis years)", flush=True)
