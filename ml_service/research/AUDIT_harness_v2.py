"""HARNESS AUDIT (user checks 10 and 15): (a) NEXT-CLOSE fills instead of signal-bar-close fills; (b) RANDOM signal through the
identical harness (same universe, leverage, overlay, gate, stops, costs). Run: python3 research/AUDIT_harness_v2.py [8yr|26yr]"""
import os, sys, time, inspect, textwrap, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "research"))
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
import EXP057_final_on_v2 as E
from main_production_backtest import FastBacktester
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
STARTS = ["2018-01-03", "2018-07-03", "2019-02-03", "2019-08-03"] if HZ == "8yr" else ["2001-01-03", "2002-02-03"]
def _engine():
    src = inspect.getsource(CleanRoom.run); reps = [
        ("dd=float(((v - v.cummax()) / v.cummax()).min()))", "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"),
        # next-close fills: pf = tomorrow's closes; used ONLY as the execution price when cfg['fill']=='nextclose'
        ("            prc = {} if row is None else {s: v for s, v in row.dropna().items()}\n",
         "            prc = {} if row is None else {s: v for s, v in row.dropna().items()}\n            pf = prc if (cfg.get('fill') != 'nextclose' or i + 1 >= len(dates)) else {s: v for s, v in px.loc[dates[i + 1]].dropna().items()}\n"),
        ("                        p = prc.get(s, lastpx.get(s, 0.0))\n                        q = books[t].pop(s); peaks[t].pop(s, None)\n                        cash += q * p\n",
         "                        p = pf.get(s, prc.get(s, lastpx.get(s, 0.0)))\n                        q = books[t].pop(s); peaks[t].pop(s, None)\n                        cash += q * p\n"),
        ("                for s, q in tgt.items():\n                    p = prc.get(s)\n                    if not p:\n                        continue\n",
         "                for s, q in tgt.items():\n                    p = pf.get(s, prc.get(s))\n                    if not p:\n                        continue\n"),
        # random signal: replace the sleeves' combined weights with a random equal-weight basket of MEMBERS
        ("                comb = {s: v for s, v in comb.items() if v >= 0.005}\n",
         "                comb = {s: v for s, v in comb.items() if v >= 0.005}\n                if cfg.get('random_signal'):\n                    rs_ = np.random.RandomState((int(pd.Timestamp(d).value // 86400e9) * 7919 + int(cfg['random_signal'])) % (2**32)); pool = sorted(s for s in mem if s in prc)\n                    pick = rs_.choice(pool, size=min(25, len(pool)), replace=False); comb = {s: 1.0 / len(pick) for s in pick}\n"),
    ]
    for a, b in reps: assert src.count(a) == 1, a[:60]; src = src.replace(a, b)
    V.END = E.END; ns = dict(V.__dict__); assert ns["END"] == E.END; exec(compile(textwrap.dedent(src), "<audit>", "exec"), ns); CleanRoom.run = ns["run"]
def st(v): return E.st(v)
def main():
    t0 = time.time(); _engine(); print(f"\n{'#'*100}\nHARNESS AUDIT {HZ}: next-close fills + random signal, starts {STARTS}\n{'#'*100}", flush=True)
    bt = FastBacktester(universe_path=E.PATH); cr = CleanRoom(bt)
    ARMS = [("LIVE_nextclose", dict(E.LIVE, fill="nextclose")), ("FINAL_1.25_nextclose", dict(E.FIN, leverage=1.25, fill="nextclose")),
            ("RANDOM_livestructure_s1", dict(E.LIVE, random_signal=1)), ("RANDOM_livestructure_s2", dict(E.LIVE, random_signal=2)),
            ("RANDOM_tranche1.25_s1", dict(E.FIN, leverage=1.25, random_signal=1)), ("RANDOM_tranche1.25_s2", dict(E.FIN, leverage=1.25, random_signal=2))]
    base = {nm: pd.read_parquet(f"{E.CACHE}/{nm}.parquet") for nm in ("LIVE", "FINAL_1.25")}
    print(f"\n  {'arm':<26}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}   per-start CAGR")
    for nm in base: a = np.array([st(base[nm][s]) for s in STARTS]); print(f"  {nm+' (cached, close fill)':<26}{a[:,0].mean():>+9.2%}{a[:,1].mean():>8.3f}{a[:,2].mean():>8.1%}   " + " ".join(f"{x:+.1%}" for x in a[:,0]), flush=True)
    out = {}
    for nm, cfg in ARMS:
        res = [cr.run(s, cfg) for s in STARTS]; a = np.array([[r["cagr"], r["sharpe"], r["dd"]] for r in res]); out[nm] = a
        print(f"  {nm:<26}{a[:,0].mean():>+9.2%}{a[:,1].mean():>8.3f}{a[:,2].mean():>8.1%}   " + " ".join(f"{x:+.1%}" for x in a[:,0]) + f"   ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame({s: r["curve"] for s, r in zip(STARTS, res)}).to_parquet(f"{E.CACHE}/audit_{nm}.parquet")
    L = np.array([st(base["LIVE"][s]) for s in STARTS]); F = np.array([st(base["FINAL_1.25"][s]) for s in STARTS])
    print(f"\n  10. NEXT-CLOSE FILL vs signal-close fill: LIVE dCAGR {(out['LIVE_nextclose'][:,0]-L[:,0]).mean()*100:+.2f}pp dSharpe {(out['LIVE_nextclose'][:,1]-L[:,1]).mean():+.3f} | FINAL_1.25 dCAGR {(out['FINAL_1.25_nextclose'][:,0]-F[:,0]).mean()*100:+.2f}pp dSharpe {(out['FINAL_1.25_nextclose'][:,1]-F[:,1]).mean():+.3f} | FINAL-LIVE delta under next-close: dSharpe {(out['FINAL_1.25_nextclose'][:,1]-out['LIVE_nextclose'][:,1]).mean():+.3f} dMaxDD {(out['FINAL_1.25_nextclose'][:,2]-out['LIVE_nextclose'][:,2]).mean()*100:+.2f}pp")
    rl = np.vstack([out['RANDOM_livestructure_s1'], out['RANDOM_livestructure_s2']]); rt = np.vstack([out['RANDOM_tranche1.25_s1'], out['RANDOM_tranche1.25_s2']])
    print(f"  15. RANDOM SIGNAL through the same harness: LIVE-structure random CAGR {rl[:,0].mean():+.2%} Sharpe {rl[:,1].mean():.3f} MaxDD {rl[:,2].mean():.1%} vs real LIVE {L[:,0].mean():+.2%}/{L[:,1].mean():.3f} | tranche random {rt[:,0].mean():+.2%}/{rt[:,1].mean():.3f}/{rt[:,2].mean():.1%} vs real FINAL {F[:,0].mean():+.2%}/{F[:,1].mean():.3f}")
    print(f"  total {time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__": main()
