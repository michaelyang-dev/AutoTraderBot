"""EXP-046 — is the 2023-2025 weakness SIGNAL DECAY or just LESS LEVERAGE?

THE TRIGGER (EXP-043)
  The rolling 3-year REC-minus-LIVE CAGR delta is positive in 17/23 windows, but the last two
  (2022-2024 -1.93pp, 2023-2025 -2.43pp) are the 4th percentile of all 23 windows. Taken at face
  value that is decay, and it would sink the recommendation.

THE CONFOUND THAT MAKES THAT READING UNSAFE
  REC and LIVE do not run the same exposure, and the gap is regime-dependent:
    LIVE = 1.49x x vol overlay. In a CALM bull market realised vol is low, the overlay clamps to
           its 1.00 cap, and LIVE runs the full ~1.49x.
    REC  = 1.25x constant, no overlay -> ~1.25x, always.
  So in a quiet strong bull market -- exactly 2023-2025 -- LIVE runs ~19% more gross than REC, and
  loses that advantage only when vol spikes. A pure leverage difference would therefore produce
  precisely the observed pattern: REC wins big in V-recoveries (2019-2022) and loses in calm
  melt-ups (2023-2025), with NO change in signal quality anywhere.

  Decay and this confound predict the SAME rolling series. They are separated only by holding
  leverage fixed.

THE DECOMPOSITION
    REC@1.25 - LIVE@1.25   = the STRATEGY change alone (overlay removal + sleeves + tranching)
    LIVE@1.49 - LIVE@1.25  = the LEVERAGE effect alone
  If (REC@1.25 - LIVE@1.25) is stable across the rolling windows while the leverage term swings,
  there is no decay -- the recommendation is simply less levered and that shows up in bull markets.
  If the strategy term itself trends to zero, the edge really is decaying.

  Also runs REC@1.49 so the comparison exists at both leverage levels.

Run:  python3 research/EXP046_leverage_decomp.py [8yr|26yr] [run|analyse]
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
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13)]
CACHE = f"research/_curves_{HZ}"

_L = dict(credit_pct=0.95, mom_w=.50, val_w=.35, lv_w=.15, initial_capital=50_000.0,
          tranches=1, tranche_stride=20, credit_derisk=0.50, vol_overlay=True)
_R = dict(credit_pct=0.95, mom_w=.70, val_w=.21, lv_w=.09, initial_capital=50_000.0,
          vol_overlay=False, tranches=4, tranche_stride=5, credit_derisk=0.50)
NEW = [("LIVE_125", dict(_L, leverage=1.25)),
       ("REC_149", dict(_R, leverage=1.49))]


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
    for nm, cfg in NEW:
        f = f"{CACHE}/{nm}.parquet"
        if os.path.exists(f):
            print(f"  {nm:<10} cached, skip", flush=True)
            continue
        pd.DataFrame({s: cr.run(s, cfg)["curve"] for s in STARTS}).to_parquet(f)
        print(f"  {nm:<10} saved  {time.time()-t0:.0f}s", flush=True)


def cagr(v):
    v = v.dropna()
    if len(v) < 60:
        return None
    y = max((v.index[-1] - v.index[0]).days / 365.25, 0.25)
    return (v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1


def roll(a, b, lab):
    yrs = sorted({y for c in a.columns for y in a[c].dropna().index.year})
    rows = []
    for y in yrs:
        if y + 2 > max(yrs):
            break
        ds = []
        for c in a.columns:
            if c not in b.columns:
                continue
            lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+2}-12-31")
            x = a[c].dropna(); z = b[c].dropna()
            x = x[(x.index >= lo) & (x.index <= hi)]
            z = z[(z.index >= lo) & (z.index <= hi)]
            ca, cb = cagr(x), cagr(z)
            if ca is not None and cb is not None:
                ds.append(cb - ca)
        if len(ds) >= 6:
            rows.append((y, float(np.mean(ds))))
    return rows


def analyse():
    need = ["LIVE", "LIVE_125", "REC125_g50", "REC_149"]
    nav = {}
    for n in need:
        f = f"{CACHE}/{n}.parquet"
        if not os.path.exists(f):
            print(f"  MISSING {f} -- run EXP043 and EXP046 run first", flush=True)
            return
        nav[n] = pd.read_parquet(f)

    series = {
        "STRATEGY  (REC@1.25 - LIVE@1.25)": roll(nav["LIVE_125"], nav["REC125_g50"], ""),
        "LEVERAGE  (LIVE@1.49 - LIVE@1.25)": roll(nav["LIVE_125"], nav["LIVE"], ""),
        "COMBINED  (REC@1.25 - LIVE@1.49)": roll(nav["LIVE"], nav["REC125_g50"], ""),
        "STRATEGY@1.49 (REC@1.49 - LIVE@1.49)": roll(nav["LIVE"], nav["REC_149"], ""),
    }
    yrs = [y for y, _ in series["COMBINED  (REC@1.25 - LIVE@1.49)"]]
    print(f"\n  ===== {HZ} — ROLLING 3-YEAR CAGR DELTA, DECOMPOSED =====", flush=True)
    hdr = f"  {'window':<12}"
    for k in series:
        hdr += f"{k.split('(')[0].strip():>16}"
    print(hdr, flush=True)
    for i, y in enumerate(yrs):
        line = f"  {y}-{y+2:<7}"
        for k, v in series.items():
            d = dict(v).get(y)
            line += f"{d*100:>+15.2f}p" if d is not None else f"{'-':>16}"
        print(line, flush=True)
    print(f"  {'-'*76}", flush=True)
    for k, v in series.items():
        d = np.array([x for _, x in v])
        last3 = d[-3:]
        print(f"  {k:<38} mean {d.mean()*100:+6.2f}pp  median {np.median(d)*100:+6.2f}pp  "
              f"pos {int((d>0).sum())}/{len(d)}  last3 {last3.mean()*100:+6.2f}pp", flush=True)

    s = np.array([x for _, x in series["STRATEGY  (REC@1.25 - LIVE@1.25)"]])
    c = np.array([x for _, x in series["COMBINED  (REC@1.25 - LIVE@1.49)"]])
    print(f"\n  VERDICT", flush=True)
    print(f"    strategy term, last 3 windows: {s[-3:].mean()*100:+.2f}pp  "
          f"vs its own history {s[:-3].mean()*100:+.2f}pp", flush=True)
    print(f"    combined term, last 3 windows: {c[-3:].mean()*100:+.2f}pp  "
          f"vs its own history {c[:-3].mean()*100:+.2f}pp", flush=True)
    if s[-3:].mean() > 0 and c[-3:].mean() < 0:
        print("    => the recent negative COMBINED delta is the LEVERAGE term, not decay.",
              flush=True)
    elif s[-3:].mean() <= 0:
        print("    => the STRATEGY term itself is negative recently. Decay cannot be excused "
              "by leverage.", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run() if MODE == "run" else analyse()
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
