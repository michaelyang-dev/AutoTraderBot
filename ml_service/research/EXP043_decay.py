"""EXP-043 — IS THE EDGE DECAYING? Sub-period and rolling analysis. Plus a CURVE CACHE.

THE TRIGGER
  VERIFY4's 26yr year-by-year shows the proposed config LOSING to live in each of the last three
  calendar years: 2023 -3.12pp, 2024 -4.85pp, 2025 -1.45pp. A 26-year mean is worthless for a
  deployment decision if the edge died in 2022. This is the single most deployment-relevant
  question outstanding, and it cuts against my own recommendation, so it goes first.

WHAT WOULD MAKE THE RECENT LOSSES BENIGN
  Three consecutive negative years is not by itself evidence of decay -- with a median year-diff
  of only +0.71pp and a per-year spread of many points, three negatives in a row happens easily by
  chance. The test has to distinguish:
    (a) NOISE            -- the recent run is inside the historical spread of 3-year windows.
    (b) REGIME           -- the edge pays in V-recoveries (2020-21) and gives some back in
                            grinding bull markets; 2023-25 is the latter, and 2013/2016/2004 were
                            too (they were also negative).
    (c) GENUINE DECAY    -- a monotone downward trend in the rolling delta, ending near zero.
  (b) and (c) look identical over three years, which is exactly why the rolling series and the
  historical distribution of 3-year windows are needed rather than an eyeball.

ALSO: THE CURVE CACHE
  Every analysis so far has paid ~25 minutes of 26yr compute to re-derive NAV curves it then
  reduces to three numbers. This saves every curve to parquet, so sub-period, rolling, event-study
  and drawdown-shape analyses afterwards cost nothing. Do this before, not after, the ideas queue.

Run:  python3 research/EXP043_decay.py [8yr|26yr] [run|analyse]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
MODE = sys.argv[2] if len(sys.argv) > 2 else "run"
PATH = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
        else "data/wrds/sp1500_universe_2000.pkl")
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13)]     # all 24
CACHE = f"research/_curves_{HZ}"

_L = dict(credit_pct=0.95, mom_w=.50, val_w=.35, lv_w=.15, initial_capital=50_000.0)
_R = dict(credit_pct=0.95, mom_w=.70, val_w=.21, lv_w=.09, initial_capital=50_000.0,
          vol_overlay=False, tranches=4, tranche_stride=5)
ARMS = [
    ("LIVE",         dict(_L, leverage=1.49, tranches=1, tranche_stride=20,
                          credit_derisk=0.50, vol_overlay=True)),
    ("REC125_g50",   dict(_R, leverage=1.25, credit_derisk=0.50)),
    ("REC125_g00",   dict(_R, leverage=1.25, credit_derisk=0.00)),
    ("REC110_g50",   dict(_R, leverage=1.10, credit_derisk=0.50)),
]
SUBS = ([("2001-2008", 2001, 2008), ("2009-2016", 2009, 2016), ("2017-2025", 2017, 2025)]
        if HZ == "26yr" else
        [("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2025", 2023, 2025)])


def run():
    import inspect
    import textwrap
    from main_production_backtest import FastBacktester
    import VERIFY2_cleanroom as V
    from VERIFY2_cleanroom import CleanRoom
    src = inspect.getsource(CleanRoom.run)
    old = "dd=float(((v - v.cummax()) / v.cummax()).min()))"
    new = "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"
    assert src.count(old) == 1, "patch anchor missing -- refusing to guess"
    ns = dict(V.__dict__)
    exec(compile(textwrap.dedent(src.replace(old, new)), "<p>", "exec"), ns)
    CleanRoom.run = ns["run"]

    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    cr = CleanRoom(FastBacktester(universe_path=PATH))
    for nm, cfg in ARMS:
        f = f"{CACHE}/{nm}.parquet"
        if os.path.exists(f):
            print(f"  {nm:<12} cached, skip", flush=True)
            continue
        cur = {s: cr.run(s, cfg)["curve"] for s in STARTS}
        pd.DataFrame(cur).to_parquet(f)
        print(f"  {nm:<12} saved {len(STARTS)} curves  {time.time()-t0:.0f}s", flush=True)


def _st(v):
    v = v.dropna()
    r = v.pct_change().dropna()
    if len(r) < 60:
        return None
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 0.25)
    return dict(cagr=(v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
                dd=float(((v - v.cummax()) / v.cummax()).min()))


def analyse():
    nav = {nm: pd.read_parquet(f"{CACHE}/{nm}.parquet") for nm, _ in ARMS}
    base = nav["LIVE"]

    print(f"\n  ===== {HZ} — SUB-PERIOD (each period re-based; {len(STARTS)} starts) =====",
          flush=True)
    for lab, y0, y1 in SUBS:
        print(f"\n  --- {lab} ---", flush=True)
        print(f"  {'arm':<13}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}"
              f"{'dMaxDD':>9}{'+Shrp':>8}", flush=True)
        got = {}
        for nm, _ in ARMS:
            per = []
            for c in nav[nm].columns:
                v = nav[nm][c].dropna()
                seg = v[(v.index >= pd.Timestamp(f"{y0}-01-01")) &
                        (v.index <= pd.Timestamp(f"{y1}-12-31"))]
                s = _st(seg)
                if s:
                    per.append(s)
            if not per:
                continue
            got[nm] = {k: np.array([p[k] for p in per]) for k in ("cagr", "sharpe", "dd")}
        b = got.get("LIVE")
        for nm, _ in ARMS:
            if nm not in got:
                continue
            g = got[nm]
            if nm == "LIVE":
                print(f"  {nm:<13}{g['cagr'].mean():>+9.2%}{g['sharpe'].mean():>9.3f}"
                      f"{g['dd'].mean():>9.1%}", flush=True)
            else:
                n = min(len(g["sharpe"]), len(b["sharpe"]))
                print(f"  {nm:<13}{g['cagr'].mean():>+9.2%}{g['sharpe'].mean():>9.3f}"
                      f"{g['dd'].mean():>9.1%}"
                      f"{(g['cagr'].mean()-b['cagr'].mean())*100:>+8.2f}p"
                      f"{g['sharpe'].mean()-b['sharpe'].mean():>+9.3f}"
                      f"{(g['dd'].mean()-b['dd'].mean())*100:>+8.2f}p"
                      f"{int((g['sharpe'][:n]>b['sharpe'][:n]).sum()):>5}/{n}", flush=True)

    # ---------------- rolling 3-year delta, and where the recent window sits ----------------
    for nm, _ in ARMS:
        if nm == "LIVE":
            continue
        print(f"\n  ===== {HZ} — ROLLING 3-YEAR CAGR DELTA: {nm} vs LIVE =====", flush=True)
        rows = []
        yrs = sorted({y for c in base.columns for y in base[c].dropna().index.year})
        for y in yrs:
            if y + 2 > max(yrs):
                break
            ds = []
            for c in base.columns:
                if c not in nav[nm].columns:
                    continue
                lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+2}-12-31")
                a = base[c].dropna(); v = nav[nm][c].dropna()
                a = a[(a.index >= lo) & (a.index <= hi)]
                v = v[(v.index >= lo) & (v.index <= hi)]
                sa, sv = _st(a), _st(v)
                if sa and sv:
                    ds.append(sv["cagr"] - sa["cagr"])
            if len(ds) >= 6:
                rows.append((y, float(np.mean(ds)), len(ds)))
        for y, d, n in rows:
            bar = "#" * int(min(abs(d) * 200, 40))
            print(f"  {y}-{y+2}  {d*100:>+7.2f}pp  n={n:<3} "
                  f"{'' if d>=0 else '-'}{bar}", flush=True)
        if len(rows) >= 6:
            ds = np.array([r[1] for r in rows])
            ys = np.array([r[0] for r in rows], float)
            slope = np.polyfit(ys, ds, 1)[0]
            last = ds[-1]
            pct = float((ds <= last).mean())
            print(f"  {'-'*58}", flush=True)
            print(f"  windows positive: {int((ds>0).sum())}/{len(ds)}   "
                  f"mean {ds.mean()*100:+.2f}pp   median {np.median(ds)*100:+.2f}pp", flush=True)
            print(f"  TREND slope {slope*100:+.3f}pp per year  "
                  f"({'decaying' if slope<0 else 'improving'})", flush=True)
            print(f"  most recent window {rows[-1][0]}-{rows[-1][0]+2}: {last*100:+.2f}pp "
                  f"= {pct:.0%} percentile of all windows "
                  f"({'INSIDE the historical spread -> consistent with noise/regime' if pct>0.10 else 'WORST DECILE -> decay cannot be ruled out'})",
                  flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run() if MODE == "run" else analyse()
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
