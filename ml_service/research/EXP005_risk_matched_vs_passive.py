"""EXP-005 — RISK-MATCHED comparison against passive, because EXP-004 was not one.

EXP-004 put the deployed strategy (26yr, 1.49x levered: +11.26% / 0.51 / -64.8%) essentially
LEVEL with an unlevered quarterly equal-weight SP1500 index (+11.17% / 0.59 / -58.5%) -- same
return, worse Sharpe, 6pp worse drawdown. Serious, but NOT risk-matched: one side runs 1.49x
with a 6.3% financing drag and the other runs 1.0x.

Two questions this settles:

  Q1  DE-LEVERED, does the strategy beat passive? Leverage 1.00 / 1.25 / 1.49 over 12 starts.
      If the strategy at 1.0x has a materially better Sharpe than EW SP1500, the security
      selection IS adding something and leverage is what turns it into a worse risk-adjusted
      product. If it does not, the core thesis is in question -- escalation-ladder level 10.

  Q2  What is leverage actually BUYING? Repo canon says 1.49x is "about optimal", but that was
      established on pre-audit universes and few starts (BUGS F6). Re-measured at 12 starts.

STATISTICAL HONESTY
  The EXP-004 benchmark is ONE path from one start date; the strategy figure is a 12-start mean
  with a 95% CI of +/-1.56pp (26yr). Those overlap heavily, so "level with passive" describes a
  difference we cannot resolve, not a measured underperformance. This script therefore reports
  the strategy's full per-start DISTRIBUTION against each single passive path, so the overlap is
  visible instead of buried in a point estimate.

Run:  python3 research/EXP005_risk_matched_vs_passive.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
LEVS = [1.00, 1.25, 1.49]

# from EXP-004 (_exp004b.out): single path, same price matrix, buy&hold between reconstitutions
PASSIVE = {
    "8yr":  [("SPY buy&hold", 0.1399, 0.77, -0.337),
             ("SPY 1.49x financed", 0.1623, 0.67, -0.472),
             ("SP1500 EW qtrly buy&hold", 0.0979, 0.52, -0.415)],
    "26yr": [("SPY buy&hold", 0.0869, 0.53, -0.552),
             ("SPY 1.49x financed", 0.0832, 0.42, -0.732),
             ("SP1500 EW qtrly buy&hold", 0.1117, 0.59, -0.585)],
}


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*92}\nEXP-005 RISK-MATCHED vs PASSIVE — {HZ}, {len(STARTS)} starts\n{'='*92}",
          flush=True)

    res = {L: [] for L in LEVS}
    gross = {L: [] for L in LEVS}
    for si, st in enumerate(STARTS):
        for L in LEVS:
            m = bt.run(st, END, {**DEPLOYED, "leverage": L})
            res[L].append(stat(m["daily_values"]))
            gross[L].append(m["avg_gross"])
        print(f"  [{si+1:2d}/{len(STARTS)}] {st}  " +
              "  ".join(f"{L:.2f}x {res[L][-1]['cagr']:+6.2%}/{res[L][-1]['sharpe']:.2f}"
                        for L in LEVS) + f"   ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  {'arm':<26}{'meanCAGR':>10}{'sdCAGR':>8}{'Sharpe':>9}{'sdShrp':>8}"
          f"{'Sortino':>9}{'meanDD':>9}{'worstDD':>9}{'avgGross':>10}", flush=True)
    for L in LEVS:
        c = np.array([r["cagr"] for r in res[L]]); s = np.array([r["sharpe"] for r in res[L]])
        so = np.array([r["sortino"] for r in res[L]]); d = np.array([r["dd"] for r in res[L]])
        print(f"  {'STRATEGY %.2fx' % L:<26}{c.mean():>+10.2%}{c.std(ddof=1):>8.2%}"
              f"{s.mean():>9.3f}{s.std(ddof=1):>8.3f}{so.mean():>9.3f}"
              f"{d.mean():>9.1%}{d.min():>9.1%}{np.mean(gross[L]):>10.3f}", flush=True)
    print(f"  {'-'*88}", flush=True)
    for name, c, s, d in PASSIVE[HZ]:
        print(f"  {'PASSIVE ' + name:<26}{c:>+10.2%}{'n/a':>8}{s:>9.3f}{'n/a':>8}"
              f"{'n/a':>9}{d:>9.1%}", flush=True)

    print(f"\n  --- how often does the strategy beat each passive path, across "
          f"{len(STARTS)} starts? ---", flush=True)
    for name, pc, ps, pd_ in PASSIVE[HZ]:
        rc = "".join(f"{int((np.array([r['cagr'] for r in res[L]]) > pc).sum()):>4}/{len(STARTS)}"
                     for L in LEVS)
        rs = "".join(f"{int((np.array([r['sharpe'] for r in res[L]]) > ps).sum()):>4}/{len(STARTS)}"
                     for L in LEVS)
        rd = "".join(f"{int((np.array([r['dd'] for r in res[L]]) > pd_).sum()):>4}/{len(STARTS)}"
                     for L in LEVS)
        print(f"  {name:<28} CAGR {rc}    Sharpe {rs}    MaxDD {rd}    "
              f"(cols = {', '.join('%.2fx' % L for L in LEVS)})", flush=True)

    print(f"\n  --- Q2: what does leverage buy? (paired deltas vs 1.00x) ---", flush=True)
    b = res[1.00]
    for L in LEVS[1:]:
        dc = np.array([res[L][i]["cagr"] - b[i]["cagr"] for i in range(len(STARTS))])
        ds = np.array([res[L][i]["sharpe"] - b[i]["sharpe"] for i in range(len(STARTS))])
        dd = np.array([res[L][i]["dd"] - b[i]["dd"] for i in range(len(STARTS))])
        print(f"  {L:.2f}x vs 1.00x: dCAGR {dc.mean()*100:+.2f}pp  dSharpe {ds.mean():+.3f}  "
              f"dMaxDD {dd.mean()*100:+.2f}pp   "
              f"Sharpe better in {int((ds>0).sum())}/{len(STARTS)} starts", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
