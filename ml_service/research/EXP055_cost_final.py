"""EXP-055 — PAIRED COST SENSITIVITY for the FINAL config. Last open item of the audit gate.

The final package (overlay ON, K=4/stride 5, sleeves 70/21/9, leverage 1.10, credit gate
p95 -> derisk 0.00) has cleared: 24 starts, two horizons, untouched holdout, an independent
engine, sub-period consistency. It has NOT had its own cost-sensitivity run -- the earlier 2x/3x/5x
pass was on the overlay-OFF package that EXP-047 retired. Tranching changes trade granularity
(four small books), so this is not a formality.

PAIRED by construction (BUGS A8b): LIVE and FINAL are both run at the same multiplier and only the
DELTA is read. Reading FINAL-at-5x against LIVE-at-1x would be theatre.

Cost is hardcoded in the clean room; patched in memory, file untouched on disk.
Run: python3 research/EXP055_cost_final.py [8yr|26yr]
"""
import os, sys, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "research"))
import numpy as np, pandas as pd
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = "data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr" else "data/wrds/sp1500_universe_2000.pkl"
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13, 2)]
CACHE = f"research/_cost_{HZ}"
_C = dict(credit_pct=0.95, initial_capital=50_000.0)
LIVE  = dict(_C, credit_derisk=0.50, vol_overlay=True, mom_w=.50, val_w=.35, lv_w=.15, tranches=1, tranche_stride=20, leverage=1.49)
FINAL = dict(_C, credit_derisk=0.00, vol_overlay=True, mom_w=.70, val_w=.21, lv_w=.09, tranches=4, tranche_stride=5,  leverage=1.10)
MULTS = [1.0, 2.0, 3.0, 5.0]

def _patched():
    import inspect, textwrap, VERIFY2_cleanroom as V
    from VERIFY2_cleanroom import CleanRoom
    src = inspect.getsource(CleanRoom.run)
    subs = [("dd=float(((v - v.cummax()) / v.cummax()).min()))", "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"),
            ("cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0", 'cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0 * float(cfg.get("cost_mult", 1.0))')]
    for a, b in subs:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    ns = dict(V.__dict__); exec(compile(textwrap.dedent(src), "<p>", "exec"), ns); CleanRoom.run = ns["run"]; return CleanRoom

def _st(v):
    v = v.dropna(); r = v.pct_change(); y = max((v.index[-1]-v.index[0]).days/365.25, 1)
    return ((v.iloc[-1]/v.iloc[0])**(1/y)-1, r.mean()/r.std()*np.sqrt(252) if r.std() > 0 else 0.0, float(((v-v.cummax())/v.cummax()).min()))

def main():
    from main_production_backtest import FastBacktester
    CR = _patched(); os.makedirs(CACHE, exist_ok=True); t0 = time.time()
    jobs = [(f"{a}_x{m:g}", dict(cfg, cost_mult=m)) for a, cfg in (("LIVE", LIVE), ("FINAL", FINAL)) for m in MULTS]
    todo = [(n, c) for n, c in jobs if not os.path.exists(f"{CACHE}/{n}.parquet")]
    if todo:
        cr = CR(FastBacktester(universe_path=PATH))
        for i, (n, c) in enumerate(todo, 1):
            pd.DataFrame({s: cr.run(s, c)["curve"] for s in STARTS}).to_parquet(f"{CACHE}/{n}.parquet")
            print(f"  [{i}/{len(todo)}] {n:<12} {time.time()-t0:6.0f}s", flush=True)
    print(f"\n  ===== {HZ} — PAIRED COST SENSITIVITY, FINAL vs LIVE ({len(STARTS)} starts) =====")
    print(f"  {'cost':<6}{'LIVE Shrp':>10}{'FINAL Shrp':>11}{'dCAGR':>9}{'dShrp':>9}{'dMaxDD':>9}{'+Shrp':>7}{'+CAGR':>7}   FINAL abs: CAGR / MaxDD")
    for m in MULTS:
        L = np.array([_st(pd.read_parquet(f"{CACHE}/LIVE_x{m:g}.parquet")[c]) for c in STARTS])
        F = np.array([_st(pd.read_parquet(f"{CACHE}/FINAL_x{m:g}.parquet")[c]) for c in STARTS])
        print(f"  {m:<6g}{L[:,1].mean():>10.3f}{F[:,1].mean():>11.3f}{(F[:,0].mean()-L[:,0].mean())*100:>+8.2f}p"
              f"{F[:,1].mean()-L[:,1].mean():>+9.3f}{(F[:,2].mean()-L[:,2].mean())*100:>+8.2f}p"
              f"{int((F[:,1]>L[:,1]).sum()):>4}/{len(STARTS)}{int((F[:,0]>L[:,0]).sum()):>4}/{len(STARTS)}   {F[:,0].mean():+.2%} / {F[:,2].mean():.1%}")
    print("  (delta must stay positive on Sharpe at 5x; if the edge is turnover-fragile it dies here)")
    print(f"\n  total {time.time()-t0:.0f}s")
if __name__ == "__main__": main()
