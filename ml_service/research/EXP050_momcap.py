"""EXP-050 — MOMENTUM CAP dose-response. Follow-up on an unplanned EXP-045 signal.

THE LEAD
  EXP-045 ran `momcap` (exclude names whose 12-1 momentum exceeds 500%) purely as a control arm --
  it was there to be the STRATEGY question next to the two bug-fix guards, and I expected it to do
  nothing or to cost money. On the 8yr it did something better than nothing:

    unguarded  REC-LIVE  dCAGR +5.04pp  dSharpe +0.059  dMaxDD -0.73pp  sign 14/24
    momcap     REC-LIVE  dCAGR +4.59pp  dSharpe +0.062  dMaxDD +0.37pp  sign 17/24

  It gave up 0.45pp of CAGR and bought a full point of drawdown plus 3 extra starts of sign
  consistency. On its own that is nowhere near enough to believe -- it is ONE threshold on ONE
  horizon, found while looking at something else, which is the exact profile of a fitting artefact.

MECHANISM (required before spending compute -- an idea with no mechanism gets downranked)
  Momentum's documented failure mode is the momentum CRASH: extreme past winners carry strong
  negative skew and unwind violently, and that unwind is concentrated in the most parabolic names.
  A cap does not try to time anything -- it declines to buy the right tail of a distribution whose
  right tail is where the crashes live. That is the same logic as the 40% trailing stop, applied at
  entry instead of at exit. It should cost a little CAGR and buy drawdown, which is exactly the
  shape observed.

THE TEST: DOSE-RESPONSE at 200% / 300% / 500% / 1000%
  A real effect must be MONOTONE -- tighter cap, more drawdown protection, more CAGR given up --
  and must flatten as the cap rises past where names actually sit. A single threshold that helps
  while its neighbours do nothing or hurt is noise, and I will read a non-monotone result that way.

  Judged on BOTH horizons and on Sharpe in EVERY sub-period (the EXP-047 bar). Cost sensitivity at
  2x/3x/5x is NOT run here because a cap REDUCES turnover -- it can only look better under higher
  costs, so that test cannot falsify it and would be theatre.

Run:  python3 research/EXP050_momcap.py [8yr|26yr]
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
CACHE = f"research/_mc_{HZ}"
REC = dict(credit_pct=0.95, credit_derisk=0.50, initial_capital=50_000.0, vol_overlay=False,
           mom_w=.70, val_w=.21, lv_w=.09, tranches=4, tranche_stride=5, leverage=1.25)
LIVE = dict(credit_pct=0.95, credit_derisk=0.50, initial_capital=50_000.0, vol_overlay=True,
            mom_w=.50, val_w=.35, lv_w=.15, tranches=1, tranche_stride=20, leverage=1.49)
CAPS = [None, 10.0, 5.0, 3.0, 2.0]          # None = uncapped, then 1000/500/300/200%


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
    mom = px.shift(20) / px.shift(252) - 1.0

    jobs = [(f"{a}_cap{('inf' if c is None else int(c*100))}", cfg, c)
            for a, cfg in (("REC", REC), ("LIVE", LIVE)) for c in CAPS]
    for nm, cfg, cap in jobs:
        f = f"{CACHE}/{nm}.parquet"
        if os.path.exists(f):
            print(f"  {nm:<16} cached", flush=True)
            continue
        if cap is None:
            bt._get_sp1500 = orig
            nex = 0.0
        else:
            excl = (mom > cap).fillna(False)
            nex = float(excl.sum(axis=1).mean())
            cols = np.array(excl.columns)
            cache = {}

            def gated(date, _x=excl, _c=cols, _e=cache, _o=orig):
                base = _o(date)
                if date not in _e:
                    try:
                        _e[date] = set(_c[_x.loc[date].values])
                    except KeyError:
                        _e[date] = set()
                return base - _e[date]
            bt._get_sp1500 = gated
        bt._sp1500_cache = {}
        cr = CleanRoom(bt)
        pd.DataFrame({s: cr.run(s, cfg)["curve"] for s in STARTS}).to_parquet(f)
        print(f"  {nm:<16} excl {nex:>5.1f}/day  {time.time()-t0:>6.0f}s", flush=True)
        bt._get_sp1500 = orig
        bt._sp1500_cache = {}

    # ---------------- dose-response report ----------------
    D = {}
    for nm, _, _ in jobs:
        df = pd.read_parquet(f"{CACHE}/{nm}.parquet")
        per = {}
        for lab, y0, y1 in SUBS:
            acc = []
            for c in df.columns:
                v = df[c].dropna()
                seg = v[(v.index >= pd.Timestamp(f"{y0}-01-01")) &
                        (v.index <= pd.Timestamp(f"{y1}-12-31"))]
                x = _st(seg)
                if x:
                    acc.append(x)
            if acc:
                per[lab] = np.array(acc)
        per["FULL"] = np.array([x for x in (_st(df[c]) for c in df.columns) if x])
        D[nm] = per
    labs = [l for l, _, _ in SUBS]

    print(f"\n  ===== {HZ} — MOMENTUM CAP DOSE-RESPONSE ({len(STARTS)} starts) =====", flush=True)
    for arm in ("LIVE", "REC"):
        print(f"\n  --- {arm} ---", flush=True)
        print(f"  {'cap':<10}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}"
              f"{'dMaxDD':>9}", flush=True)
        ref = D[f"{arm}_capinf"]["FULL"]
        for c in CAPS:
            k = f"{arm}_cap{('inf' if c is None else int(c*100))}"
            f = D[k]["FULL"]
            lab = "none" if c is None else f"{int(c*100)}%"
            print(f"  {lab:<10}{f[:,0].mean():>+9.2%}{f[:,1].mean():>9.3f}{f[:,2].mean():>9.1%}"
                  f"{(f[:,0].mean()-ref[:,0].mean())*100:>+8.2f}p"
                  f"{f[:,1].mean()-ref[:,1].mean():>+9.3f}"
                  f"{(f[:,2].mean()-ref[:,2].mean())*100:>+8.2f}p", flush=True)

    print(f"\n  ===== REC - LIVE at each cap (both arms capped identically) =====", flush=True)
    print(f"  {'cap':<10}{'dCAGR':>9}{'dShrp':>9}{'dMaxDD':>9}{'+Shrp':>8}"
          + "".join(f"{l:>12}" for l in labs), flush=True)
    for c in CAPS:
        sfx = 'inf' if c is None else int(c*100)
        R, L = D[f"REC_cap{sfx}"], D[f"LIVE_cap{sfx}"]
        r, l = R["FULL"], L["FULL"]
        n = min(len(r), len(l))
        line = (f"  {('none' if c is None else str(int(c*100))+'%'):<10}"
                f"{(r[:,0].mean()-l[:,0].mean())*100:>+8.2f}p"
                f"{r[:,1].mean()-l[:,1].mean():>+9.3f}"
                f"{(r[:,2].mean()-l[:,2].mean())*100:>+8.2f}p"
                f"{int((r[:n,1]>l[:n,1]).sum()):>5}/{n}")
        for lb in labs:
            line += (f"{R[lb][:,1].mean()-L[lb][:,1].mean():>+12.3f}"
                     if lb in R and lb in L else f"{'-':>12}")
        print(line, flush=True)
    print(f"\n  MONOTONE across 200/300/500/1000 = real dose-response. "
          f"A lone threshold that helps while its neighbours do not = noise.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
