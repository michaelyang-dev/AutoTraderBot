"""EXP-024 — AUDIT GATE for the one revived idea: momentum-pool ROE quality filter.

THIS RESULT MAY BE INFLATED. Auditing before treating it as a finding.

WHAT CAME BACK (EXP-021, 26yr, 12 starts, honest financing, audited universe):
    mom_quality_filter="roe", mom_quality_pool=3
      -> take the top 15 by the momentum composite, keep the top 5 by ROE
    dCAGR +0.27pp   dSharpe +0.021   dMaxDD +3.81pp   sign 9/12 CAGR, 10/12 Sharpe  SURVIVES

  This idea was previously logged DEAD. It came back under the many-start standard on the
  post-audit universe -- exactly the BUGS F6 scenario the REVIVE batch exists to catch.

MECHANISM (why it could be real)
  Quality-momentum interaction, Novy-Marx. Among names that have already run, the profitable
  ones are the ones whose run reflects improving fundamentals rather than sentiment or a
  short squeeze; the unprofitable ones are the ones that reverse. It refines the ORDERING,
  which EXP-003 showed is the only part of selection that carries any edge -- so it is
  operating on the right axis, not a new bolt-on.
  WHO LOSES: buyers of unprofitable momentum names near the top of a run.

WHY IT MIGHT STILL BE NOTHING
  1. `roe` is a fundamental. If its timestamping is by PERIOD rather than RELEASE date anywhere
     in the panel, this arm eats a look-ahead the pure-price arms do not. The shift test is the
     check that matters most here and is run first.
  2. avgGross rose 1.1939 -> 1.2115. More exposure normally *worsens* drawdown, yet drawdown
     IMPROVED by 3.81pp -- which argues the effect is real, but the matched-exposure control is
     run anyway rather than assumed.
  3. gp_assets -- the OTHER quality metric, same mechanism, same hook -- was catastrophic
     (-2.67pp, 0/12). If the mechanism were robust, a closely related metric should not be that
     bad. That asymmetry is the strongest argument that `roe` here is a lucky draw, and it is
     the reason the pool-size sweep below matters: a real effect should survive pool changes.
  4. It is one arm out of nine in EXP-021. Bonferroni on 9 tests turns a nominal p~0.03 into
     ~0.24.

TESTS
  A  8yr confirmation (the two-horizon bar)
  B  SHIFT: lag every signal one full bar. Look-ahead dies here; a real edge degrades gently.
  C  pool sweep 2 / 3 / 4 / 6 -- a real effect should not need exactly 15 candidates
  D  cost x1 / x2 / x3, each paired against the base at the SAME cost (BUGS A8b)
  E  event concentration of the excess, guarded (BUGS A8a)
  F  matched-exposure residual

Run:  python3 research/EXP024_roe_quality_audit.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import (DEPLOYED, HORIZONS, END, starts, stat, sign_verdict,  # noqa: E402
                     event_concentration, matched_exposure_curve)

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
BASE = {**DEPLOYED, **GATE, "financing_curve": True}
ROE = {"mom_quality_filter": "roe", "mom_quality_pool": 3}


def run_arms(bt, arms, base_cfg, label):
    base, bcv, bg = [], [], []
    for st in STARTS:
        m = bt.run(st, END, base_cfg)
        base.append(stat(m["daily_values"])); bcv.append(m["daily_values"])
        bg.append(m["avg_gross"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"\n  --- {label} ---   base CAGR {bc.mean():+.2%} Sharpe {bs.mean():.3f} "
          f"MaxDD {bd.mean():.1%} gross {np.mean(bg):.4f}", flush=True)
    print(f"  {'arm':<22}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'gross':>8}{'dCAGR':>9}"
          f"{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}{'+Shrp':>7}{'top5%':>9}{'verdict':>10}",
          flush=True)
    out = {}
    for name, extra in arms:
        rows, cvs, gs = [], [], []
        for st in STARTS:
            m = bt.run(st, END, {**base_cfg, **extra})
            rows.append(stat(m["daily_values"])); cvs.append(m["daily_values"])
            gs.append(m["avg_gross"])
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        cc = [event_concentration(bcv[i], cvs[i]) for i in range(len(rows))]
        ok = [x for x in cc if x["excess_meaningful"]]
        t5 = (f"{np.mean([x['top5_share'] for x in ok]):.0%}" if len(ok) >= len(cc) // 2
              else f"n/a({len(ok)}/{len(cc)})")
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        out[name] = (c - bc, s - bs, d - bd)
        print(f"  {name:<22}{c.mean():>+9.2%}{s.mean():>9.3f}{d.mean():>9.1%}"
              f"{np.mean(gs):>8.4f}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
              f"{(d-bd).mean()*100:>+8.2f}p{nc:>4}/{len(c)}{ns:>4}/{len(s)}{t5:>9}{v:>10}",
              flush=True)
    return out


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*112}\nEXP-024 ROE-QUALITY AUDIT GATE — {HZ}, {len(STARTS)} starts\n{'='*112}",
          flush=True)

    # A + C: headline and pool sweep
    run_arms(bt, [(f"roe pool={p}", {"mom_quality_filter": "roe", "mom_quality_pool": p})
                  for p in (2, 3, 4, 6)]
             + [("gp_assets pool=3", {"mom_quality_filter": "gp_assets",
                                      "mom_quality_pool": 3})],
             BASE, "A/C headline + pool sweep + the gp_assets contrast")

    # D: cost sensitivity, paired at the SAME multiplier
    for cm in (2, 3):
        run_arms(bt, [("roe pool=3", ROE)], {**BASE, "cost_mult": cm}, f"D cost x{cm} (paired)")

    # F: matched exposure
    print(f"\n  --- F matched-exposure residual ---", flush=True)
    ms, mc = [], []
    for st in STARTS:
        m = bt.run(st, END, {**BASE, **ROE})
        r = stat(m["daily_values"])
        ref = matched_exposure_curve(bt, st, END, m["avg_gross"], cfg=BASE)
        ms.append(r["sharpe"] - ref["sharpe"]); mc.append(r["cagr"] - ref["cagr"])
    print(f"  vs constant-leverage curve at its OWN gross: dSharpe {np.mean(ms):+.3f} "
          f"({int((np.array(ms)>0).sum())}/{len(ms)})  dCAGR {np.mean(mc)*100:+.2f}pp",
          flush=True)

    # B: SHIFT test -- the one that matters most, because roe is a FUNDAMENTAL
    print(f"\n  --- B SHIFT test: every signal lagged one full bar ---", flush=True)
    of, ofb, oc, ou = (bt.uni._feat_by_date, bt.features_by_date, bt.uni._close, bt.umd_20d)
    dts = sorted(of.keys())
    bt.uni._feat_by_date = {dts[i]: of[dts[i - 1]] for i in range(1, len(dts))}
    bt.features_by_date = {dts[i]: ofb[dts[i - 1]] for i in range(1, len(dts))
                           if dts[i - 1] in ofb}
    bt.uni._close = {k: v.shift(1) for k, v in oc.items()}
    bt.umd_20d = ou.shift(1)
    try:
        run_arms(bt, [("roe pool=3 SHIFTED", ROE)], BASE, "B shift +1 bar")
    finally:
        bt.uni._feat_by_date, bt.features_by_date = of, ofb
        bt.uni._close, bt.umd_20d = oc, ou
    print(f"\n  READING: if the SHIFTED arm keeps a similar dSharpe, no look-ahead. If it dies "
          f"or flips,\n           `roe` is timestamped by period rather than release date "
          f"somewhere and the result is fiction.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
