"""EXP-025 (IDEAS I-30) — is the ROE result a mechanism, or a lucky draw?

THE PROBLEM IT SOLVES
  EXP-021 revived `mom_quality_filter="roe"` (+0.021 Sharpe, 10/12) but the SAME hook with
  `gp_assets` was catastrophic (-0.072 Sharpe, 0/12). Both are profitability measures. Two
  measures of the same thing should not disagree that violently if the mechanism is real.

  That asymmetry is either
    (a) the tell that `roe` is a lucky draw out of nine EXP-021 arms (Bonferroni: p~0.03 -> ~0.24), or
    (b) informative -- i.e. it matters WHICH profitability signal you use, for a reason.

  This settles it independently of EXP-024's own audit. If two MORE profitability metrics side
  with `roe`, the mechanism is real and gp_assets is the outlier. If they side with gp_assets,
  `roe` is noise and EXP-024 should not be believed however clean its shift test comes back.

  This is the multiple-testing question asked CONSTRUCTIVELY -- adding independent evidence
  rather than just deflating a p-value.

THE FAMILY (all already in the causal feature panel, all "profitability", no new data)
  roe               net income / equity              <- the survivor
  net_margin        net income / sales
  operating_margin  operating income / sales
  gp_assets         gross profit / assets            <- the catastrophe (Novy-Marx's own metric)
  gross_margin      gross profit / sales             <- bounded [-1,1] since commit 74d57cb

WHAT WOULD MAKE ME BELIEVE (a) vs (b)
  (b) mechanism real: >=3 of 5 positive, and the negatives explainable. Note roe/net_margin/
      operating_margin share a NUMERATOR (income after costs) while gp_assets/gross_margin use
      GROSS profit -- so an income-vs-gross-profit split would be a real, interpretable pattern.
  (a) lucky draw: results scattered with no pattern, ~half positive, magnitudes similar to noise.

Run:  python3 research/EXP025_profitability_family.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat, sign_verdict  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
BASE = {**DEPLOYED, **GATE, "financing_curve": True}

FAMILY = [("roe", "income/equity"), ("net_margin", "income/sales"),
          ("operating_margin", "opinc/sales"), ("gp_assets", "grossprofit/assets"),
          ("gross_margin", "grossprofit/sales")]


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*104}\nEXP-025 PROFITABILITY FAMILY — {HZ}, {len(STARTS)} starts, gate ON"
          f"\n{'='*104}", flush=True)

    base, bg = [], []
    for st in STARTS:
        m = bt.run(st, END, BASE)
        base.append(stat(m["daily_values"])); bg.append(m["avg_gross"])
    bc = np.array([r["cagr"] for r in base]); bs = np.array([r["sharpe"] for r in base])
    bd = np.array([r["dd"] for r in base])
    print(f"  BASE  CAGR {bc.mean():+.2%}  Sharpe {bs.mean():.3f}  MaxDD {bd.mean():.1%}  "
          f"gross {np.mean(bg):.4f}", flush=True)

    print(f"\n  {'metric':<20}{'numerator':<20}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}"
          f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}{'+CAGR':>7}{'+Shrp':>7}{'verdict':>10}",
          flush=True)
    res = {}
    for feat, desc in FAMILY:
        rows = []
        for st in STARTS:
            m = bt.run(st, END, {**BASE, "mom_quality_filter": feat, "mom_quality_pool": 3})
            rows.append(stat(m["daily_values"]))
        c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
        d = np.array([r["dd"] for r in rows])
        nc, ns, v = sign_verdict(c - bc, s - bs, len(STARTS))
        res[feat] = (s - bs).mean()
        print(f"  {feat:<20}{desc:<20}{c.mean():>+9.2%}{s.mean():>9.3f}{d.mean():>9.1%}"
              f"{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}{(d-bd).mean()*100:>+8.2f}p"
              f"{nc:>4}/{len(c)}{ns:>4}/{len(s)}{v:>10}", flush=True)

    inc = [res[f] for f in ("roe", "net_margin", "operating_margin")]
    gp = [res[f] for f in ("gp_assets", "gross_margin")]
    npos = sum(1 for v in res.values() if v > 0)
    print(f"\n  income-based  (roe / net_margin / operating_margin): "
          f"mean dSharpe {np.mean(inc):+.3f}   [{', '.join(f'{v:+.3f}' for v in inc)}]",
          flush=True)
    print(f"  grossprofit   (gp_assets / gross_margin):             "
          f"mean dSharpe {np.mean(gp):+.3f}   [{', '.join(f'{v:+.3f}' for v in gp)}]",
          flush=True)
    print(f"\n  {npos}/5 positive on Sharpe.", flush=True)
    print(f"  >=3/5 positive with a clean income-vs-grossprofit split -> MECHANISM, roe is real.",
          flush=True)
    print(f"  scattered, ~half positive, noise-sized -> LUCKY DRAW, do not believe EXP-024.",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
