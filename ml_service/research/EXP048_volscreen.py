"""EXP-048 — IDIOSYNCRATIC-VOL SCREEN (IDEAS I-11), implemented as a membership gate.

MECHANISM
  AUDIT01's forward-IC sweep found `vol_60d` the SINGLE STRONGEST feature in the whole panel
  (IC -0.050: low vol -> high forward return) -- stronger than any return feature. The momentum
  sleeve currently harvests this only through a soft x1.15 nudge for vol_20d < 0.25. The low-vol
  anomaly (leverage-constrained investors bid up high-beta names, so they are systematically
  overpriced) is among the most replicated effects in the literature, it is visible in our OWN
  data, and we are barely trading it.

  Momentum and low-vol are also known to combine well: momentum's worst property is its crash
  risk, which is concentrated in exactly the high-beta names a vol screen removes.

IMPLEMENTATION -- as a membership gate, deliberately
  Excluding high-vol names from the investable set BEFORE the sleeves run needs no change to any
  sleeve, so there is no new modelling risk and no chance I quietly alter a sleeve's behaviour
  while "adding a screen". It is the same mechanism EXP-045 uses. Trailing 60-day realised vol,
  ranked cross-sectionally each day -> strictly PIT-honest.

  Note this screens the WHOLE BOOK, not just momentum. That is a stronger intervention than I-11
  proposed and it must be judged as such: the value and lowvol sleeves lose names too.

ARMS
  p67  drop the top vol TERCILE       (I-11 as written)
  p80  drop the top vol QUINTILE      (softer)
  p50  drop the top vol HALF          (aggressive -- included to give a dose-response; a real
                                       effect should be monotone, a fitted one will not be)

  The dose-response is the point. A genuine low-vol premium should strengthen monotonically as the
  screen tightens, then eventually cost too much breadth. A non-monotone result is a fitting
  artefact and I will read it that way.

JUDGED ON: Sharpe in EVERY sub-period (the EXP-047 bar), not the mean. And on MaxDD -- a vol
screen that improves Sharpe by shrinking exposure is not an edge, it is de-grossing in disguise.

Run:  python3 research/EXP048_volscreen.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
        else "data/wrds/sp1500_universe_2000.pkl")
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13, 2)]
SUBS = ([("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2025", 2023, 2025)]
        if HZ == "8yr" else
        [("2001-2008", 2001, 2008), ("2009-2016", 2009, 2016),
         ("2017-2022", 2017, 2022), ("2023-2025", 2023, 2025)])
CACHE = f"research/_vol_{HZ}"

BASE = dict(credit_pct=0.95, credit_derisk=0.50, initial_capital=50_000.0, vol_overlay=False,
            mom_w=.70, val_w=.21, lv_w=.09, tranches=4, tranche_stride=5, leverage=1.25)
CUTS = [("p80", 0.80), ("p67", 0.67), ("p50", 0.50)]


def _st(v):
    v = v.dropna()
    if len(v) < 60:
        return None
    r = v.pct_change()
    y = max((v.index[-1] - v.index[0]).days / 365.25, 0.25)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1,
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            float(((v - v.cummax()) / v.cummax()).min()))


def main():
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
    bt = FastBacktester(universe_path=PATH)
    px = bt.prices
    orig = bt._get_sp1500

    # trailing 60d realised vol, cross-sectional percentile each day -- PIT-honest
    rv = px.pct_change(fill_method=None).rolling(60, min_periods=40).std()
    pct = rv.rank(axis=1, pct=True)
    print(f"  vol panel built {time.time()-t0:.0f}s", flush=True)

    arms = [("base", None)] + [(n, q) for n, q in CUTS]
    for nm, q in arms:
        f = f"{CACHE}/{nm}.parquet"
        if os.path.exists(f):
            print(f"  {nm:<7} cached", flush=True)
            continue
        if q is None:
            bt._get_sp1500 = orig
        else:
            keep = (pct <= q)
            cols = np.array(keep.columns)
            cache = {}

            def gated(date, _k=keep, _c=cols, _e=cache, _o=orig):
                base = _o(date)
                if date not in _e:
                    try:
                        _e[date] = set(_c[_k.loc[date].values])
                    except KeyError:
                        _e[date] = None
                allow = _e[date]
                return base if allow is None else (base & allow)
            bt._get_sp1500 = gated
        bt._sp1500_cache = {}
        cr = CleanRoom(bt)
        pd.DataFrame({s: cr.run(s, BASE)["curve"] for s in STARTS}).to_parquet(f)
        print(f"  {nm:<7} saved  {time.time()-t0:.0f}s", flush=True)
        bt._get_sp1500 = orig
        bt._sp1500_cache = {}

    # ---------------- report ----------------
    print(f"\n  ===== {HZ} — VOL SCREEN, dose-response ({len(STARTS)} starts) =====", flush=True)
    D = {}
    for nm, _ in arms:
        df = pd.read_parquet(f"{CACHE}/{nm}.parquet")
        per = {}
        for lab, y0, y1 in SUBS:
            acc = []
            for c in df.columns:
                v = df[c].dropna()
                seg = v[(v.index >= pd.Timestamp(f"{y0}-01-01")) &
                        (v.index <= pd.Timestamp(f"{y1}-12-31"))]
                s = _st(seg)
                if s:
                    acc.append(s)
            if acc:
                per[lab] = np.array(acc)
        per["FULL"] = np.array([x for x in (_st(df[c]) for c in df.columns) if x])
        D[nm] = per
    b = D["base"]
    labs = [l for l, _, _ in SUBS]
    print(f"  {'arm':<7}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}{'dMaxDD':>9}"
          + "".join(f"{l:>12}" for l in labs), flush=True)
    for nm, _ in arms:
        p = D[nm]
        f = p["FULL"]
        line = f"  {nm:<7}{f[:,0].mean():>+9.2%}{f[:,1].mean():>9.3f}{f[:,2].mean():>9.1%}"
        if nm == "base":
            line += " " * 27
        else:
            line += (f"{(f[:,0].mean()-b['FULL'][:,0].mean())*100:>+8.2f}p"
                     f"{f[:,1].mean()-b['FULL'][:,1].mean():>+9.3f}"
                     f"{(f[:,2].mean()-b['FULL'][:,2].mean())*100:>+8.2f}p")
        for l in labs:
            if l in p and l in b:
                line += f"{p[l][:,1].mean()-b[l][:,1].mean():>+12.3f}"
            else:
                line += f"{'-':>12}"
        print(line, flush=True)
    print(f"\n  (sub-period columns are dSharpe vs base. Monotone across p80->p67->p50 = real "
          f"dose-response; non-monotone = fitting artefact.)", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
