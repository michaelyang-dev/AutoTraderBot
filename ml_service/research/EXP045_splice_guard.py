"""EXP-045 — REMEDIATION for BUGS D10: does a PIT-honest splice guard change the answer?

THE PROBLEM (D10)
  Spliced ticker series manufacture impossible returns -- WW/WTW carries a 15,925.6% ONE-DAY move
  in both universe files. A number that large is an automatic #1 on a top-5 momentum rank, so if
  such names are investable the backtest buys a fiction.

WHY NOT A BLACKLIST
  The obvious fix is to list the offending tickers and drop them. I am deliberately NOT doing that:
  I chose those names by looking at the results, so a hand-curated list is fitted to the outcome and
  would not generalise to the splices I have not spotted. A general rule that never sees a ticker
  name is both a real fix and an honest test.

THE GUARD (PIT-honest by construction)
  A name is excluded from the investable set for the 252 sessions FOLLOWING any single-day move
  beyond a threshold. That is exactly the window in which the bad print contaminates 12-1 momentum,
  it uses only trailing data, and it needs no knowledge of which ticker is bad. Excluding on
  full-history knowledge would be look-ahead; this is not.

THREE ARMS, and they answer DIFFERENT questions -- do not conflate them:
  splice300  |ret| > 300% in one day  -> essentially only data artefacts. This is the BUG FIX.
  splice100  |ret| > 100% in one day  -> artefacts + genuine squeezes (GME, MARA, HTZ).
  momcap     12-1 momentum > 500%     -> a STRATEGY question (should we chase extreme momentum?),
                                         NOT a bug fix. Labelled separately for that reason.

INTERPRETATION SET IN ADVANCE, so I cannot move it afterwards:
  - splice300 moves the answer by ~0  => D10 does not reach the book; conclusions stand as written.
  - splice300 moves the answer a lot  => some share of every momentum result in this program is an
                                         artefact, and the whole program needs re-running on a
                                         guarded panel. That is the bad outcome and I will report
                                         it as such.

Run:  python3 research/EXP045_splice_guard.py [8yr|26yr]
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
STARTS = [f"{y}-{m:02d}-03" for y in YRS for m in range(1, 13)]
CACHE = f"research/_curves_{HZ}"

_L = dict(credit_pct=0.95, mom_w=.50, val_w=.35, lv_w=.15, initial_capital=50_000.0)
_R = dict(credit_pct=0.95, mom_w=.70, val_w=.21, lv_w=.09, initial_capital=50_000.0,
          vol_overlay=False, tranches=4, tranche_stride=5)
LIVE = dict(_L, leverage=1.49, tranches=1, tranche_stride=20, credit_derisk=0.50, vol_overlay=True)
REC = dict(_R, leverage=1.25, credit_derisk=0.50)

GUARDS = [("splice300", "ret", 3.00), ("splice100", "ret", 1.00), ("momcap", "mom", 5.00)]


def build_mask(px, kind, thr):
    """True where the name is EXCLUDED. Trailing-window only -- no look-ahead."""
    if kind == "ret":
        r = px.pct_change(fill_method=None)
        bad = (r.abs() > thr)
        # excluded for the 252 sessions AFTER the bad print (the contamination window)
        return bad.rolling(252, min_periods=1).max().fillna(0).astype(bool)
    m = px.shift(20) / px.shift(252) - 1.0
    return (m > thr).fillna(False)


def _st(v):
    v = v.dropna()
    r = v.pct_change()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return dict(cagr=(v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
                dd=float(((v - v.cummax()) / v.cummax()).min()))


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

    t0 = time.time()
    bt = FastBacktester(universe_path=PATH)
    px = bt.prices
    orig_get = bt._get_sp1500

    out = {}
    for gname, kind, thr in GUARDS:
        mask = build_mask(px, kind, thr)
        excl_by_date = {}
        cols = np.array(mask.columns)

        def gated(date, _m=mask, _c=cols, _e=excl_by_date, _o=orig_get):
            base = _o(date)
            if date not in _e:
                try:
                    _e[date] = set(_c[_m.loc[date].values])
                except KeyError:
                    _e[date] = set()
            return base - _e[date]

        bt._get_sp1500 = gated
        bt._sp1500_cache = {}
        cr = CleanRoom(bt)
        nex = float(mask.sum(axis=1).mean())
        print(f"\n  --- guard {gname} ({kind} > {thr:.0%}) : excludes {nex:.1f} names/day ---",
              flush=True)
        for nm, cfg in (("LIVE", LIVE), ("REC", REC)):
            rs = [cr.run(s, cfg) for s in STARTS]
            out[(gname, nm)] = rs
            print(f"    {nm:<5} CAGR {np.mean([x['cagr'] for x in rs]):+.2%}  "
                  f"Sharpe {np.mean([x['sharpe'] for x in rs]):.3f}  "
                  f"MaxDD {np.mean([x['dd'] for x in rs]):.1%}   {time.time()-t0:.0f}s",
                  flush=True)
        bt._get_sp1500 = orig_get
        bt._sp1500_cache = {}

    # -------- compare against the UNGUARDED cache --------
    print(f"\n\n  ===== {HZ} — GUARD vs UNGUARDED ({len(STARTS)} starts) =====", flush=True)
    ref = {}
    for nm, f in (("LIVE", "LIVE"), ("REC", "REC125_g50")):
        p = f"{CACHE}/{f}.parquet"
        if os.path.exists(p):
            d = pd.read_parquet(p)
            ref[nm] = [_st(d[c]) for c in d.columns]
    if not ref:
        print("  (curve cache missing -- run EXP043 first)", flush=True)
        return
    print(f"  {'guard':<11}{'arm':<6}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}"
          f"{'dCAGR':>9}{'dShrp':>9}{'dMaxDD':>9}", flush=True)
    for nm in ("LIVE", "REC"):
        b = {k: np.array([x[k] for x in ref[nm]]) for k in ("cagr", "sharpe", "dd")}
        print(f"  {'UNGUARDED':<11}{nm:<6}{b['cagr'].mean():>+9.2%}{b['sharpe'].mean():>9.3f}"
              f"{b['dd'].mean():>9.1%}", flush=True)
        for gname, _, _ in GUARDS:
            g = {k: np.array([x[k] for x in out[(gname, nm)]]) for k in ("cagr", "sharpe", "dd")}
            print(f"  {gname:<11}{nm:<6}{g['cagr'].mean():>+9.2%}{g['sharpe'].mean():>9.3f}"
                  f"{g['dd'].mean():>9.1%}{(g['cagr'].mean()-b['cagr'].mean())*100:>+8.2f}p"
                  f"{g['sharpe'].mean()-b['sharpe'].mean():>+9.3f}"
                  f"{(g['dd'].mean()-b['dd'].mean())*100:>+8.2f}p", flush=True)

    print(f"\n  ===== THE NUMBER THAT DECIDES IT: (REC - LIVE) under each guard =====", flush=True)
    print(f"  {'guard':<11}{'dCAGR':>9}{'dSharpe':>10}{'dMaxDD':>10}{'+Shrp':>8}", flush=True)
    bl = {k: np.array([x[k] for x in ref["LIVE"]]) for k in ("cagr", "sharpe", "dd")}
    br = {k: np.array([x[k] for x in ref["REC"]]) for k in ("cagr", "sharpe", "dd")}
    n = min(len(bl["sharpe"]), len(br["sharpe"]))
    print(f"  {'UNGUARDED':<11}{(br['cagr'].mean()-bl['cagr'].mean())*100:>+8.2f}p"
          f"{br['sharpe'].mean()-bl['sharpe'].mean():>+10.3f}"
          f"{(br['dd'].mean()-bl['dd'].mean())*100:>+9.2f}p"
          f"{int((br['sharpe'][:n]>bl['sharpe'][:n]).sum()):>5}/{n}", flush=True)
    for gname, _, _ in GUARDS:
        L = {k: np.array([x[k] for x in out[(gname, 'LIVE')]]) for k in ("cagr", "sharpe", "dd")}
        R = {k: np.array([x[k] for x in out[(gname, 'REC')]]) for k in ("cagr", "sharpe", "dd")}
        print(f"  {gname:<11}{(R['cagr'].mean()-L['cagr'].mean())*100:>+8.2f}p"
              f"{R['sharpe'].mean()-L['sharpe'].mean():>+10.3f}"
              f"{(R['dd'].mean()-L['dd'].mean())*100:>+9.2f}p"
              f"{int((R['sharpe']>L['sharpe']).sum()):>5}/{len(R['sharpe'])}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
