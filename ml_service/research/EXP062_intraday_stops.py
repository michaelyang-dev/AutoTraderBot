"""EXP-062 — live-vs-backtest parity: trailing stops on CLOSES (backtest) vs INTRADAY (live engine).

The validated clean room updates each book's peak with the day's close and stops when the close is <= 60% of it,
exiting at that close. The live engine polls prices all session: its peak rises with intraday highs and it sells at
the market as soon as a price is <= 60% of the peak. Live-like model here, per held name per day, using CRSP
open/high/low scaled onto the universe's total-return close series (factor = adjusted close / raw close):
  threshold = prior peak x 0.60; open <= threshold -> exit at the open (gap through the stop);
  else low <= threshold -> exit at the threshold; else peak = max(prior peak, high, close).
Same engine, same deployed package (cap 0.15, overlay_down, equal-weight momentum, 80/15/5), 2018-2025 (CRSP has
OHLC through 2025), 12 monthly starts. Read-only. Run from ml_service/: python3 research/EXP062_intraday_stops.py"""
import ast
import inspect
import os
import sys
import textwrap
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
sys.argv = ["x", HZ]
import EXP057_final_on_v2 as E  # noqa: E402  (chdirs to ml_service)
import VERIFY2_cleanroom as V  # noqa: E402
from VERIFY2_cleanroom import CleanRoom  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402

END = pd.Timestamp("2025-12-31")
STARTS = E.STARTS[0::2]
BASE = dict(E.FIN, leverage=1.49)
PACKAGE = dict(BASE, cap=0.15, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05)
ARMS = {"close_stops": PACKAGE, "intraday_stops": dict(PACKAGE, stop_mode="intraday"),
        "highpeak_closetrig": dict(PACKAGE, stop_mode="highpeak"),     # peaks from intraday highs, trigger/exit at the close
        "closepeak_lowtrig": dict(PACKAGE, stop_mode="lowtrig")}       # peaks from closes, trigger on the low / exit intraday

# ---- the frontier harness's validated patches, taken verbatim from its source (no drift) ----------------------
tree = ast.parse(open("research/EXP059_frontier.py").read())
eng = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_engine")
reps_node = next(n for n in ast.walk(eng) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "reps")
REPS = ast.literal_eval(reps_node.value)
EXTRA = [
    ("        px = self.px\n", "        px = self.px\n        _hlo = getattr(self, '_hlo', None)\n"),
    ("                        peaks[t][s] = max(peaks[t].get(s, p), p)\n                        if p <= peaks[t][s] * (1.0 - stop_of.get(s, stop)):\n",
     "                        if cfg.get('stop_mode') in ('intraday', 'highpeak', 'lowtrig') and _hlo is not None and s in _hlo[0].columns and d in _hlo[0].index:\n"
     "                            _o, _h, _l = _hlo[0].at[d, s], _hlo[1].at[d, s], _hlo[2].at[d, s]\n"
     "                            if _o == _o and _h == _h and _l == _l:\n"
     "                                _pk = peaks[t].get(s, p)\n"
     "                                _thr = _pk * (1.0 - stop_of.get(s, stop))\n"
     "                                _m = cfg.get('stop_mode')\n"
     "                                if _m == 'highpeak': _xp = p if p <= _thr else None\n"
     "                                else: _xp = _o if _o <= _thr else (_thr if _l <= _thr else None)\n"
     "                                if _xp is None:\n"
     "                                    peaks[t][s] = max(_pk, p) if _m == 'lowtrig' else max(_pk, _h, p)\n"
     "                                    continue\n"
     "                                q = books[t].pop(s); peaks[t].pop(s, None)\n"
     "                                cash += q * _xp; cash -= q * _xp * cost_r\n"
     "                                self._nstop = getattr(self, '_nstop', 0) + 1\n"
     "                                continue\n"
     "                        peaks[t][s] = max(peaks[t].get(s, p), p)\n"
     "                        if p <= peaks[t][s] * (1.0 - stop_of.get(s, stop)):\n"
     "                            self._nstop = getattr(self, '_nstop', 0) + 1\n"),
]
src = inspect.getsource(CleanRoom.run)
for a, b in REPS + EXTRA:
    assert src.count(a) == 1, a[:90]
    src = src.replace(a, b)
V.END = END
ns = dict(V.__dict__)
assert ns["END"] == END
exec(compile(textwrap.dedent(src), "<exp062>", "exec"), ns)
CleanRoom.run = ns["run"]

t0 = time.time()
cr = CleanRoom(FastBacktester(universe_path=E.PATH))
px = cr.px
cols = [c for c in px.columns if str(c).isdigit()]
permnos = [int(c) for c in cols]
dset = ds.dataset("data/wrds/crsp_daily_stock_full.parquet")
flt = (ds.field("PERMNO").isin(permnos)) & (ds.field("DlyCalDt") >= ("2017-06-01" if HZ == "8yr" else "2000-06-01")) & (ds.field("DlyCalDt") <= "2025-12-31")
raw = dset.to_table(columns=["PERMNO", "DlyCalDt", "DlyClose", "DlyPrc", "DlyOpen", "DlyHigh", "DlyLow"], filter=flt).to_pandas()
raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"])
raw["close_raw"] = raw["DlyClose"].where(raw["DlyClose"] > 0, raw["DlyPrc"].abs())
raw = raw.drop_duplicates(["PERMNO", "DlyCalDt"], keep="last")
piv = {k: raw.pivot(index="DlyCalDt", columns="PERMNO", values=v) for k, v in
       (("c", "close_raw"), ("o", "DlyOpen"), ("h", "DlyHigh"), ("l", "DlyLow"))}
for k in piv:
    piv[k].columns = [str(c) for c in piv[k].columns]
    piv[k] = piv[k].reindex(index=px.index, columns=cols)
fac = px[cols] / piv["c"]
OP, HI, LO = piv["o"] * fac, piv["h"] * fac, piv["l"] * fac
ok = OP.notna() & HI.notna() & LO.notna()
bad = (LO > HI * 1.0001) | (OP > HI * 1.02) | (OP < LO * 0.98)
OP, HI, LO = OP.where(ok & ~bad), HI.where(ok & ~bad), LO.where(ok & ~bad)
cov = float(ok.sum().sum() / px[cols].loc[:END].notna().sum().sum())
print(f"OHLC built in {time.time() - t0:.0f}s: coverage {cov:.1%} of close cells, rejected inconsistent cells {int(bad.sum().sum())}", flush=True)
cr._hlo = (OP, HI, LO)

res = {}
for nm, cfg in ARMS.items():
    rows = []
    for s_ in STARTS:
        cr._nstop = 0
        t1 = time.time()
        r = cr.run(s_, cfg)
        cagr, sh, dd = E.st(r["curve"])
        rows.append((s_, cagr, sh, dd, cr._nstop))
        print(f"  {nm:15s} {s_}  CAGR {cagr:+.2%}  Sharpe {sh:.3f}  MaxDD {dd:.1%}  stops {cr._nstop:4d}  ({time.time() - t1:.0f}s)", flush=True)
    res[nm] = pd.DataFrame(rows, columns=["start", "cagr", "sharpe", "maxdd", "stops"]).set_index("start")
for nm in ("highpeak_closetrig", "closepeak_lowtrig"):
    dd_ = res[nm] - res["close_stops"]
    print(f"  {nm:20s} vs close: dCAGR {dd_['cagr'].mean() * 100:+.2f}pp ({int((dd_['cagr'] > 0).sum())}/{len(dd_)})  dSharpe {dd_['sharpe'].mean():+.3f}  "
          f"dMaxDD {dd_['maxdd'].mean() * 100:+.2f}pp  stops/run {res[nm]['stops'].mean():.0f}")
A, B = res["close_stops"], res["intraday_stops"]
d = B - A
print(f"\n{HZ} (to 2025-12-31), {len(STARTS)} starts — deployed package")
print(f"  close stops    : CAGR {A['cagr'].mean():+.2%}  Sharpe {A['sharpe'].mean():.3f}  MaxDD {A['maxdd'].mean():.1%}  stops/run {A['stops'].mean():.0f}")
print(f"  intraday stops : CAGR {B['cagr'].mean():+.2%}  Sharpe {B['sharpe'].mean():.3f}  MaxDD {B['maxdd'].mean():.1%}  stops/run {B['stops'].mean():.0f}")
print(f"  intraday - close: dCAGR {d['cagr'].mean() * 100:+.2f}pp (better on {int((d['cagr'] > 0).sum())}/{len(d)})  "
      f"dSharpe {d['sharpe'].mean():+.3f} ({int((d['sharpe'] > 0).sum())}/{len(d)})  dMaxDD {d['maxdd'].mean() * 100:+.2f}pp ({int((d['maxdd'] > 0).sum())}/{len(d)})")
pd.concat(res, axis=1).to_parquet(f"research/_exp062_intraday_stops_{HZ}.parquet")
print(f"total {time.time() - t0:.0f}s")
