"""EXP-056 — extend the FINAL base's leverage ladder to 1.40x / 1.49x for an ISO-DRAWDOWN option.

EXP-053 measured the final base (overlay ON, K=4/5, 70/21/9, gate p95->0.00) at 1.00/1.10/1.25x
only. The free audit shows the package at 1.10x is a pure risk reshaping: CAGR flat, Sharpe up,
MaxDD 8-13pp shallower, but it LOSES most calendar years by return (2/7, 10/24) because it runs
less gross than LIVE's 1.49x. The Sharpe gain can be spent the other way -- run the final base at
LIVE's own leverage and read CAGR at LIVE's own drawdown. That is the only leverage-fair way to
answer "is the new package better in most years". Same cache dir/naming as EXP-053 so all five
levels analyse together. 12 used + 12 untouched starts, both horizons.
Run: python3 research/EXP056_ladder_ext.py [8yr|26yr]
"""
import os, sys, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "research"))
import pandas as pd
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = "data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr" else "data/wrds/sp1500_universe_2000.pkl"
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
USED = [f"{y}-{m:02d}-03" for y in YRS for m in (1,3,5,7,9,11)]; HOLD = [f"{y}-{m:02d}-03" for y in YRS for m in (2,4,6,8,10,12)]
CACHE = f"research/_gw_{HZ}"
WIN = dict(credit_pct=0.95, credit_derisk=0.00, initial_capital=50_000.0, vol_overlay=True, mom_w=.70, val_w=.21, lv_w=.09, tranches=4, tranche_stride=5)
def main():
    import inspect, textwrap
    from main_production_backtest import FastBacktester
    import VERIFY2_cleanroom as V
    from VERIFY2_cleanroom import CleanRoom
    src = inspect.getsource(CleanRoom.run); a = "dd=float(((v - v.cummax()) / v.cummax()).min()))"
    assert src.count(a) == 1; ns = dict(V.__dict__)
    exec(compile(textwrap.dedent(src.replace(a, a[:-1] + ", curve=v)")), "<p>", "exec"), ns); CleanRoom.run = ns["run"]
    t0 = time.time(); cr = None
    for lev in (1.40, 1.49):
        for tag, sts in (("used", USED), ("hold", HOLD)):
            f = f"{CACHE}/L{lev:.2f}_g0.00_{tag}.parquet"
            if os.path.exists(f): continue
            cr = cr or CleanRoom(FastBacktester(universe_path=PATH))
            pd.DataFrame({s: cr.run(s, dict(WIN, leverage=lev))["curve"] for s in sts}).to_parquet(f)
            print(f"  L{lev:.2f} {tag}  {time.time()-t0:6.0f}s", flush=True)
    print(f"  done {time.time()-t0:.0f}s")
if __name__ == "__main__": main()
