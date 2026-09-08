"""EXP-057 — LIVE vs FINAL tranche structure on the REBUILT (v2) universes. The deliverable.
Clean-room engine, 24 starts (12 used odd months + 12 untouched even months), both horizons,
leverage ladder for FINAL, sub-periods, year-by-year, paired cost 1x/2x. Nothing here is new
methodology — it is the audited protocol re-run on corrected data.
Run: python3 research/EXP057_final_on_v2.py [8yr|26yr] [run|analyse]"""
import os, sys, time, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "research"))
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"; MODE = sys.argv[2] if len(sys.argv) > 2 else "run"
PATH = "data/wrds/complete_sp1500_universe_v2.pkl" if HZ == "8yr" else "data/wrds/sp1500_universe_2000_v2.pkl"
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13)]
END = pd.Timestamp("2026-08-31")
CACHE = f"research/_v2_{HZ}"
SUBS = ([("2018-2020",2018,2020),("2021-2022",2021,2022),("2023-2026",2023,2026)] if HZ=="8yr" else [("2001-2008",2001,2008),("2009-2016",2009,2016),("2017-2022",2017,2022),("2023-2026",2023,2026)])
_C = dict(credit_pct=0.95, initial_capital=50_000.0)
LIVE = dict(_C, credit_derisk=0.50, vol_overlay=True, mom_w=.50, val_w=.35, lv_w=.15, tranches=1, tranche_stride=20, leverage=1.49)
FIN = dict(_C, credit_derisk=0.00, vol_overlay=True, mom_w=.70, val_w=.21, lv_w=.09, tranches=4, tranche_stride=5)
ARMS = [("LIVE", LIVE), ("FINAL_1.10", dict(FIN, leverage=1.10)), ("FINAL_1.25", dict(FIN, leverage=1.25)), ("FINAL_1.49", dict(FIN, leverage=1.49)),
        ("LIVE_cost2", dict(LIVE, cost_mult=2.0)), ("FINAL_1.25_cost2", dict(FIN, leverage=1.25, cost_mult=2.0))]
def _engine():
    import inspect, textwrap, VERIFY2_cleanroom as V
    from VERIFY2_cleanroom import CleanRoom
    src = inspect.getsource(CleanRoom.run)
    for a, b in [("dd=float(((v - v.cummax()) / v.cummax()).min()))", "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"),
                 ("cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0", 'cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0 * float(cfg.get("cost_mult", 1.0))')]:
        assert src.count(a) == 1, a; src = src.replace(a, b)
    V.END = END                        # MUST precede the dict copy: the exec'd run() reads END from ns, a COPY of V.__dict__
    ns = dict(V.__dict__); assert ns["END"] == END; exec(compile(textwrap.dedent(src), "<p>", "exec"), ns); CleanRoom.run = ns["run"]
    # BUG found 2026-09-07: the first version set V.END AFTER copying, so every curve silently ended 2025-12-31 while the
    # header claimed 2026-08-31. Caught by the ledger audit's end-date check. All v2 caches from that version were deleted.
    return CleanRoom
def run():
    from main_production_backtest import FastBacktester
    CR = _engine(); os.makedirs(CACHE, exist_ok=True); t0 = time.time(); cr = None
    for nm, cfg in ARMS:
        f = f"{CACHE}/{nm}.parquet"
        if os.path.exists(f): continue
        cr = cr or CR(FastBacktester(universe_path=PATH))
        pd.DataFrame({s: cr.run(s, cfg)["curve"] for s in STARTS}).to_parquet(f); print(f"  {nm:<18} {time.time()-t0:6.0f}s", flush=True)
def st(v):
    v = v.dropna(); r = v.pct_change(); y = max((v.index[-1]-v.index[0]).days/365.25, .25)
    return ((v.iloc[-1]/v.iloc[0])**(1/y)-1, r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0, float(((v-v.cummax())/v.cummax()).min())) if len(v) > 60 else None
def yearly(c):
    c = c.dropna(); out = {}
    for y, seg in c.groupby(c.index.year):
        if len(seg) < 150: continue
        pr = c[c.index < pd.Timestamp(f"{y}-01-01")]
        if pr.empty: continue
        out[y] = seg.iloc[-1]/pr.iloc[-1]-1
    return out
def analyse():
    D = {nm: pd.read_parquet(f"{CACHE}/{nm}.parquet") for nm, _ in ARMS if os.path.exists(f"{CACHE}/{nm}.parquet")}
    L = D["LIVE"]; cols = list(L.columns); l = np.array([st(L[c]) for c in cols]); n = len(cols)
    print(f"\n{'='*96}\n{HZ} on REBUILT v2 universe — LIVE vs FINAL ({n} starts: 12 used + 12 untouched), through {END.date()}\n{'='*96}")
    print(f"  {'arm':<18}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'worstDD':>9}{'dCAGR':>9}{'dShrp':>8}{'dMaxDD':>9}{'+Shrp':>7}{'+DD':>7}" + "".join(f"{s[0]:>12}" for s in SUBS))
    def per(df):
        out = {}
        for lab, y0, y1 in SUBS:
            acc = [st(df[c].dropna()[(df[c].dropna().index >= f"{y0}-01-01") & (df[c].dropna().index <= f"{y1}-12-31")]) for c in cols]; acc = [x for x in acc if x]
            if acc: out[lab] = np.array(acc)
        return out
    lp = per(L)
    for nm, _ in ARMS:
        if nm not in D or "cost2" in nm: continue
        F = D[nm]; f = np.array([st(F[c]) for c in cols])
        if nm == "LIVE": print(f"  {nm:<18}{f[:,0].mean():>+9.2%}{f[:,1].mean():>8.3f}{f[:,2].mean():>8.1%}{f[:,2].min():>9.1%}"); continue
        d = f - l; fp = per(F)
        print(f"  {nm:<18}{f[:,0].mean():>+9.2%}{f[:,1].mean():>8.3f}{f[:,2].mean():>8.1%}{f[:,2].min():>9.1%}{d[:,0].mean()*100:>+8.2f}p{d[:,1].mean():>+8.3f}{d[:,2].mean()*100:>+8.2f}p{int((d[:,1]>0).sum()):>4}/{n}{int((d[:,2]>0).sum()):>4}/{n}"
              + "".join(f"{(fp[s][:,1].mean()-lp[s][:,1].mean()) if s in fp and s in lp else float('nan'):>+12.3f}" for s, _, _ in SUBS))
    print("  (sub-period columns = dSharpe vs LIVE)")
    if "LIVE_cost2" in D and "FINAL_1.25_cost2" in D:
        l2 = np.array([st(D["LIVE_cost2"][c]) for c in cols]); f2 = np.array([st(D["FINAL_1.25_cost2"][c]) for c in cols])
        print(f"  PAIRED COST 2x: FINAL_1.25 - LIVE  dSharpe {(f2[:,1]-l2[:,1]).mean():+.3f} (1x: {(np.array([st(D['FINAL_1.25'][c]) for c in cols])[:,1]-l[:,1]).mean():+.3f})  dCAGR {(f2[:,0]-l2[:,0]).mean()*100:+.2f}pp")
    for nm in ("FINAL_1.25", "FINAL_1.49"):
        if nm not in D: continue
        yl = [yearly(L[c]) for c in cols]; yf = [yearly(D[nm][c]) for c in cols]; rows = []
        for y in sorted(set().union(*[set(x) for x in yl])):
            pr = [(a[y], b[y]) for a, b in zip(yl, yf) if y in a and y in b]
            if pr: rows.append((y, np.mean([p[1]-p[0] for p in pr]), sum(1 for a, b in pr if b > a), len(pr)))
        dd = np.array([r[1] for r in rows])
        print(f"\n  YEAR-BY-YEAR {nm} vs LIVE: better {int((dd>0).sum())}/{len(dd)}  median {np.median(dd)*100:+.2f}pp  mean {dd.mean()*100:+.2f}pp")
        print("   " + "  ".join(f"{y}:{d*100:+.1f}({w}/{m})" for y, d, w, m in rows))
if __name__ == "__main__":
    t0 = time.time(); run() if MODE == "run" else analyse(); print(f"\n  total {time.time()-t0:.0f}s")
