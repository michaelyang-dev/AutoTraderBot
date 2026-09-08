"""ENGINE GAP DECOMPOSITION: why does the clean room report ~3pp more CAGR than the original live-mirror on the same v2 data?
Candidates: position cap (clean room 0.15 vs mirror 0.10), financing (curve vs fixed 6.3%). Run: python3 research/AUDIT_enginegap_v2.py"""
import os, sys, time, inspect, textwrap, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research")
import EXP057_final_on_v2 as E
from main_production_backtest import FastBacktester
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
from livemirror_backtest import LiveMirrorBacktester
STARTS = ["2018-01-03", "2018-07-03", "2019-02-03", "2019-08-03"]; END = "2025-12-31"
src = inspect.getsource(CleanRoom.run)
for a, b in [("        cap_pos = 0.15\n", "        cap_pos = float(cfg.get('cap_pos', 0.15))\n"),
             ("        fr = fr.ffill().reindex(pd.DatetimeIndex(dates)).ffill().bfill()\n", "        fr = fr.ffill().reindex(pd.DatetimeIndex(dates)).ffill().bfill()\n        if cfg.get('fin_fixed'): fr = fr * 0 + float(cfg['fin_fixed'])\n"),
             ("dd=float(((v - v.cummax()) / v.cummax()).min()))", "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)")]:
    assert src.count(a) == 1, a; src = src.replace(a, b)
V.END = pd.Timestamp(END); ns = dict(V.__dict__); exec(compile(textwrap.dedent(src), "<gap>", "exec"), ns); CleanRoom.run = ns["run"]
def st(v): v = v.dropna(); r = v.pct_change().dropna(); y = (v.index[-1] - v.index[0]).days / 365.25; return (v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1, r.mean() / r.std() * np.sqrt(252), float(((v - v.cummax()) / v.cummax()).min())
t0 = time.time(); bt = FastBacktester(universe_path=E.PATH); cr = CleanRoom(bt); mir = LiveMirrorBacktester(universe_path=E.PATH)
B = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False, "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0, "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True, "financing_rate": 0.063, "credit_pct": 0.95, "credit_derisk": 0.5}
print(f"\n{'#'*100}\nENGINE GAP DECOMPOSITION, 8yr v2, LIVE config, 4 starts, to {END}\n{'#'*100}\n  {'arm':<44}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}")
def show(nm, res): a = np.array(res); print(f"  {nm:<44}{a[:,0].mean():>+9.2%}{a[:,1].mean():>8.3f}{a[:,2].mean():>8.1%}   [{time.time()-t0:.0f}s]", flush=True); return a
show("CLEAN ROOM as reported (cap 0.15, fin curve)", [st(cr.run(s, E.LIVE)["curve"]) for s in STARTS])
show("CLEAN ROOM cap 0.10", [st(cr.run(s, dict(E.LIVE, cap_pos=0.10))["curve"]) for s in STARTS])
show("CLEAN ROOM fin fixed 6.3%", [st(cr.run(s, dict(E.LIVE, fin_fixed=0.063))["curve"]) for s in STARTS])
show("CLEAN ROOM cap 0.10 + fin 6.3%", [st(cr.run(s, dict(E.LIVE, cap_pos=0.10, fin_fixed=0.063))["curve"]) for s in STARTS])
show("ORIGINAL mirror as reported (cap 0.10, fin 6.3%)", [st(mir.run(s, END, dict(B))["daily_values"]) for s in STARTS])
show("ORIGINAL mirror cap 0.15", [st(mir.run(s, END, dict(B, cap=0.15))["daily_values"]) for s in STARTS])
print(f"  total {time.time()-t0:.0f}s")
