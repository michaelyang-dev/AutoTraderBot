"""ENGINE AGREEMENT: the ORIGINAL live-mirror harness (research/livemirror_backtest.py, written months before the clean room,
shares no sizing/cost/financing code with it) on the SAME v2 universe, same 4 starts, deployed config. Run: python3 research/AUDIT_origengine_v2.py [8yr|26yr]"""
import os, sys, time, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research")
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
from livemirror_backtest import LiveMirrorBacktester
PATH = "data/wrds/complete_sp1500_universe_v2.pkl" if HZ == "8yr" else "data/wrds/sp1500_universe_2000_v2.pkl"
STARTS = ["2018-01-03", "2018-07-03", "2019-02-03", "2019-08-03"] if HZ == "8yr" else ["2001-01-03", "2002-02-03"]
B = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
     "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0, "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
     "financing_rate": 0.063, "credit_pct": 0.95, "credit_derisk": 0.5}
def st(v): v = v.dropna(); r = v.pct_change().dropna(); y = (v.index[-1] - v.index[0]).days / 365.25; return (v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1, r.mean() / r.std() * np.sqrt(252), float(((v - v.cummax()) / v.cummax()).min())
t0 = time.time(); bt = LiveMirrorBacktester(universe_path=PATH); C = pd.read_parquet(f"research/_v2_{HZ}/LIVE.parquet")
print(f"\n{'#'*100}\nORIGINAL ENGINE (LiveMirror) vs CLEAN ROOM on v2 {HZ}, LIVE config, starts {STARTS}\n{'#'*100}")
for end in ("2025-12-31", "2026-08-31"):
    rows = []
    for s in STARTS:
        m = bt.run(s, end, dict(B)); a = st(m["daily_values"]); b = st(C[s].loc[:end]); rows.append((s, a, b, m.get("fin_paid", float("nan")), m.get("avg_gross", float("nan"))))
        print(f"  {s} to {end}: ORIGINAL {a[0]:+.2%} / {a[1]:.3f} / {a[2]:.1%}   CLEAN-ROOM {b[0]:+.2%} / {b[1]:.3f} / {b[2]:.1%}   (orig fin paid {rows[-1][3]:,.0f}, avg gross {rows[-1][4]:.2f})  [{time.time()-t0:.0f}s]", flush=True)
    A = np.array([r[1] for r in rows]); Bc = np.array([r[2] for r in rows])
    print(f"  MEAN to {end}: ORIGINAL {A[:,0].mean():+.2%} / {A[:,1].mean():.3f} / {A[:,2].mean():.1%}  vs CLEAN-ROOM {Bc[:,0].mean():+.2%} / {Bc[:,1].mean():.3f} / {Bc[:,2].mean():.1%}   dCAGR {(Bc[:,0]-A[:,0]).mean()*100:+.2f}pp dSharpe {(Bc[:,1]-A[:,1]).mean():+.3f}", flush=True)
print(f"  total {time.time()-t0:.0f}s")
