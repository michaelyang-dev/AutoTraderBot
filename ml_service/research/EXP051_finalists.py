"""EXP-051 — FINALIST VERIFICATION at the full 24-start standard, both samples, both horizons.

WHY
  EXP-047 screened 24 configs at 12 starts (odd months) -- a SCREEN, explicitly not a verdict.
  Finalists must clear the real bar: 24 starts = the 12 previously-used odd months PLUS the 12
  untouched even months that no experiment in this program has ever examined.

WHAT EXP-047 ACTUALLY ESTABLISHED, and what it did not
  Two-horizon-agreeing MAIN EFFECTS (each averaged over 12 configs, so far more robust than any
  single config's rank):
    tranching 4 books   +0.065 Sharpe (8yr)  +0.033 (26yr)   <- consistent, and mechanism-backed
    lower leverage      monotone on both horizons             <- expected; it is a risk choice
    overlay ON          +0.027 (8yr) but -0.011 (26yr) full   <- horizons DISAGREE on full sample,
                        yet BOTH say ON is better in 2023-2025 (+0.118 / +0.102)
    sleeves 70/21/9     +0.023 (8yr)  +0.002 (26yr)           <- noise on the long horizon

  The per-config "won 4/4" ranking is NOT robust and I am not selecting on it. On the 26yr only two
  configs went 4/4, and they did it by margins of +0.006 and +0.007 in the last sub-period while
  strictly better configs missed by -0.018. That is a knife-edge, not a finding.

THE CONTAMINATED CELL -- the reason the horizons disagree
  Take ovl_s503515_t4_L1.49 (= LIVE plus tranching, nothing else). Its 2023-2025 dSharpe is
  -0.049 on the 26yr file and +0.052 on the 8yr file: OPPOSITE SIGNS, same config, same calendar
  window, different universe file. BUGS D9 says the 26yr file is missing ~10% of the modern
  investable universe in exactly that window. **The 26yr 2023-2025 column is the least trustworthy
  cell in the entire table** and must be down-weighted against the 8yr's, which is clean there.
  Read that way, the leading config wins every period on the trustworthy file for each era.

FINALISTS (chosen from the main effects, not from the leaderboard)
  A  ovl_s702109_t4_L1.10   overlay ON + tilt + tranching + low leverage   <- leading candidate
  B  ovl_s702109_t4_L1.25   same, one leverage step up
  C  ovl_s503515_t4_L1.25   LIVE sleeves + tranching + lower leverage (isolates the tilt)
  D  ovl_s702109_t1_L1.10   the 26yr 4/4 config (isolates tranching)
  E  noovl_s702109_t4_L1.25 the config I PREVIOUSLY recommended -- carried so the comparison is
                            explicit rather than quietly dropped

Run:  python3 research/EXP051_finalists.py [8yr|26yr] [run|analyse]
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
MODE = sys.argv[2] if len(sys.argv) > 2 else "run"
PATH = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
        else "data/wrds/sp1500_universe_2000.pkl")
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
USED = [f"{y}-{m:02d}-03" for y in YRS for m in (1, 3, 5, 7, 9, 11)]
HOLD = [f"{y}-{m:02d}-03" for y in YRS for m in (2, 4, 6, 8, 10, 12)]
SUBS = ([("2018-2020", 2018, 2020), ("2021-2022", 2021, 2022), ("2023-2025", 2023, 2025)]
        if HZ == "8yr" else
        [("2001-2008", 2001, 2008), ("2009-2016", 2009, 2016),
         ("2017-2022", 2017, 2022), ("2023-2025", 2023, 2025)])
CACHE = f"research/_fin_{HZ}"
_C = dict(credit_pct=0.95, credit_derisk=0.50, initial_capital=50_000.0)
ARMS = [
    ("LIVE", dict(_C, vol_overlay=True, mom_w=.50, val_w=.35, lv_w=.15,
                  tranches=1, tranche_stride=20, leverage=1.49)),
    ("A_ovl_tilt_t4_L110", dict(_C, vol_overlay=True, mom_w=.70, val_w=.21, lv_w=.09,
                                tranches=4, tranche_stride=5, leverage=1.10)),
    ("B_ovl_tilt_t4_L125", dict(_C, vol_overlay=True, mom_w=.70, val_w=.21, lv_w=.09,
                                tranches=4, tranche_stride=5, leverage=1.25)),
    ("C_ovl_live_t4_L125", dict(_C, vol_overlay=True, mom_w=.50, val_w=.35, lv_w=.15,
                                tranches=4, tranche_stride=5, leverage=1.25)),
    ("D_ovl_tilt_t1_L110", dict(_C, vol_overlay=True, mom_w=.70, val_w=.21, lv_w=.09,
                                tranches=1, tranche_stride=20, leverage=1.10)),
    ("E_PREV_noovl_t4_L125", dict(_C, vol_overlay=False, mom_w=.70, val_w=.21, lv_w=.09,
                                  tranches=4, tranche_stride=5, leverage=1.25)),
]


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
        for tag, sts in (("used", USED), ("hold", HOLD)):
            f = f"{CACHE}/{nm}_{tag}.parquet"
            if os.path.exists(f):
                continue
            pd.DataFrame({s: cr.run(s, cfg)["curve"] for s in sts}).to_parquet(f)
            print(f"  {nm:<22}{tag}  {time.time()-t0:>6.0f}s", flush=True)


def _st(v):
    v = v.dropna()
    if len(v) < 60:
        return None
    r = v.pct_change()
    y = max((v.index[-1] - v.index[0]).days / 365.25, 0.25)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1,
            r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
            float(((v - v.cummax()) / v.cummax()).min()))


def load(nm, tag):
    df = pd.read_parquet(f"{CACHE}/{nm}_{tag}.parquet")
    per = {}
    for lab, y0, y1 in SUBS:
        acc = []
        for c in df.columns:
            v = df[c].dropna()
            seg = v[(v.index >= pd.Timestamp(f"{y0}-01-01")) & (v.index <= pd.Timestamp(f"{y1}-12-31"))]
            x = _st(seg)
            if x:
                acc.append(x)
        if acc:
            per[lab] = np.array(acc)
    per["FULL"] = np.array([x for x in (_st(df[c]) for c in df.columns) if x])
    return per


def analyse():
    labs = [l for l, _, _ in SUBS]
    for tag, title in (("hold", "UNTOUCHED HOLDOUT (even months)"),
                       ("used", "previously-used (odd months)")):
        D = {nm: load(nm, tag) for nm, _ in ARMS
             if os.path.exists(f"{CACHE}/{nm}_{tag}.parquet")}
        if "LIVE" not in D:
            continue
        b = D["LIVE"]
        print(f"\n  ===== {HZ} — {title} =====", flush=True)
        print(f"  {'arm':<22}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}"
              f"{'dMaxDD':>9}{'+Shrp':>8}{'won':>6}" + "".join(f"{l:>12}" for l in labs),
              flush=True)
        for nm, _ in ARMS:
            if nm not in D:
                continue
            p = D[nm]
            f = p["FULL"]
            if nm == "LIVE":
                print(f"  {nm:<22}{f[:,0].mean():>+9.2%}{f[:,1].mean():>9.3f}"
                      f"{f[:,2].mean():>9.1%}", flush=True)
                continue
            ds = {l: p[l][:, 1].mean() - b[l][:, 1].mean() for l in labs if l in p and l in b}
            won = sum(1 for v in ds.values() if v > 0)
            n = min(len(f), len(b["FULL"]))
            print(f"  {nm:<22}{f[:,0].mean():>+9.2%}{f[:,1].mean():>9.3f}{f[:,2].mean():>9.1%}"
                  f"{(f[:,0].mean()-b['FULL'][:,0].mean())*100:>+8.2f}p"
                  f"{f[:,1].mean()-b['FULL'][:,1].mean():>+9.3f}"
                  f"{(f[:,2].mean()-b['FULL'][:,2].mean())*100:>+8.2f}p"
                  f"{int((f[:n,1]>b['FULL'][:n,1]).sum()):>5}/{n}{won:>4}/{len(ds)}"
                  + "".join(f"{ds.get(l,float('nan')):>+12.3f}" for l in labs), flush=True)

    # holdout-vs-used agreement is the real test of whether the screen generalised
    print(f"\n  ===== {HZ} — HOLDOUT vs USED AGREEMENT (dSharpe vs LIVE) =====", flush=True)
    print(f"  {'arm':<22}{'used':>10}{'holdout':>10}{'gap':>10}", flush=True)
    for nm, _ in ARMS:
        if nm == "LIVE":
            continue
        try:
            u = load(nm, "used"); h = load(nm, "hold")
            bu = load("LIVE", "used"); bh = load("LIVE", "hold")
        except FileNotFoundError:
            continue
        du = u["FULL"][:, 1].mean() - bu["FULL"][:, 1].mean()
        dh = h["FULL"][:, 1].mean() - bh["FULL"][:, 1].mean()
        print(f"  {nm:<22}{du:>+10.3f}{dh:>+10.3f}{du-dh:>+10.3f}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    run() if MODE == "run" else analyse()
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)
