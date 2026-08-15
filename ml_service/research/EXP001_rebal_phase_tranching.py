"""EXP-001 (IDEAS I-01) — REBALANCE-PHASE TRANCHING.

HYPOTHESIS
----------
The rebalance date carries zero information. It is a nuisance parameter. Yet the measured
per-start CAGR sigma is 7.07pp on the 8yr (range +15.03%..+34.55% on entry month alone) --
that entire spread is PHASE risk, and the deployed book takes 100% of it uncompensated.

Running K sub-books on different phases of the rebal_days cycle and combining them should
shrink that dispersion by roughly sqrt(K) with NO assumption that any signal works better.
This is diversification applied to the time axis instead of the name axis.

WHY THIS IS NOT A FITTING ARTEFACT
----------------------------------
No parameter is being chosen on performance. The claim is purely that averaging K draws of a
zero-mean nuisance reduces its variance. If mean CAGR moves much at all in EITHER direction,
that is evidence something else is going on and the result should be distrusted.

THE HONEST COST, WHICH IS MODELLED
----------------------------------
Each tranche gets capital/K, so integer-share rounding bites K times harder. At $50k / K=4 a
tranche runs $12.5k across ~23 names ~ $540/position -- whole-share truncation is material.
The harness floors every quantity, so this cost is fully paid, not assumed away.

PREDICTIONS (stated BEFORE running, so they can falsify)
--------------------------------------------------------
  1. mean CAGR:  ~unchanged (|delta| < 1pp).  A large POSITIVE delta is a bug report.
  2. sigma of CAGR across starts: DOWN, by roughly sqrt(K) toward the phase-risk floor.
  3. Sharpe: UP slightly (same return, less path noise).
  4. MaxDD: modestly better.
  5. turnover per dollar: ~unchanged.
If (2) fails, phases are far more correlated than assumed and the idea is void.

Run:  python3 research/EXP001_rebal_phase_tranching.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

B = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
     "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
     "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
     "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
     "financing_rate": 0.063}

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
if HZ == "8yr":
    PATH, YEARS = "data/wrds/complete_sp1500_universe.pkl", [2018, 2019]
else:
    PATH, YEARS = "data/wrds/sp1500_universe_2000.pkl", [2001, 2002]
STARTS = [f"{y}-{m:02d}-03" for y in YEARS for m in range(1, 13, 2)]   # 12 monthly starts
END = "2025-12-31"
CAPITAL = 50_000.0
SCHEMES = [("K=2", [0, 10]), ("K=4", [0, 5, 10, 15])]


def stat(v):
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return dict(cagr=(v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
                sharpe=dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0,
                dd=((v - v.cummax()) / v.cummax()).min())


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*94}\nEXP-001 REBALANCE-PHASE TRANCHING — {HZ}, {len(STARTS)} starts, "
          f"${CAPITAL:,.0f}\n{'='*94}", flush=True)

    # ---- PARITY GATE: rebal_phase=0 must reproduce the untouched code path exactly ----
    m0 = bt.run(STARTS[0], END, dict(B))
    m0p = bt.run(STARTS[0], END, {**B, "rebal_phase": 0})
    same = bool((m0["daily_values"].values == m0p["daily_values"].values).all())
    print(f"  PARITY  rebal_phase=0 vs untouched path: "
          f"{'IDENTICAL' if same else 'DIFFERENT'}  "
          f"({m0['daily_values'].iloc[-1]:,.2f} vs {m0p['daily_values'].iloc[-1]:,.2f})", flush=True)
    if not same:
        print("  ABORT — the phase hook changed the default path. Nothing below is comparable.")
        return

    rows = {"base": [], "K=2": [], "K=4": []}
    per_phase = {p: [] for p in (0, 5, 10, 15)}

    for si, st in enumerate(STARTS):
        m = bt.run(st, END, {**B, "initial_capital": CAPITAL})
        rows["base"].append(stat(m["daily_values"]))

        for tag, phases in SCHEMES:
            cap_k = CAPITAL / len(phases)
            curves = []
            for p in phases:
                mm = bt.run(st, END, {**B, "initial_capital": cap_k, "rebal_phase": p})
                curves.append(mm["daily_values"])
                if tag == "K=4":
                    per_phase[p].append(stat(mm["daily_values"])["cagr"])
            comb = sum(c.reindex(curves[0].index).ffill() for c in curves)
            rows[tag].append(stat(comb))
        print(f"  [{si+1:2d}/{len(STARTS)}] {st}  base {rows['base'][-1]['cagr']:+7.2%}  "
              f"K=2 {rows['K=2'][-1]['cagr']:+7.2%}  K=4 {rows['K=4'][-1]['cagr']:+7.2%}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------- report -------------------------------
    print(f"\n  {'arm':<10}{'meanCAGR':>10}{'sdCAGR':>9}{'meanShrp':>10}{'sdShrp':>8}"
          f"{'meanDD':>9}{'worstDD':>9}", flush=True)
    agg = {}
    for tag in ("base", "K=2", "K=4"):
        c = np.array([r["cagr"] for r in rows[tag]])
        s = np.array([r["sharpe"] for r in rows[tag]])
        d = np.array([r["dd"] for r in rows[tag]])
        agg[tag] = (c, s, d)
        print(f"  {tag:<10}{c.mean():>+10.2%}{c.std(ddof=1):>9.2%}{s.mean():>10.3f}"
              f"{s.std(ddof=1):>8.3f}{d.mean():>9.1%}{d.min():>9.1%}", flush=True)

    bc, bs, bd = agg["base"]
    print(f"\n  {'delta vs base':<16}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}"
          f"{'+CAGR':>8}{'+Shrp':>8}{'sigma ratio':>13}", flush=True)
    for tag in ("K=2", "K=4"):
        c, s, d = agg[tag]
        k = len(dict(SCHEMES)[tag])
        print(f"  {tag:<16}{(c-bc).mean()*100:>+8.2f}p{(s-bs).mean():>+9.3f}"
              f"{(d-bd).mean()*100:>+8.2f}p{int((c>bc).sum()):>5}/{len(c)}"
              f"{int((s>bs).sum()):>5}/{len(s)}"
              f"{c.std(ddof=1)/bc.std(ddof=1):>10.3f} (1/sqrt{k}={1/np.sqrt(k):.3f})", flush=True)

    print("\n  --- direct measurement of PHASE RISK (K=4 tranches, same start, "
          "same capital) ---", flush=True)
    pm = np.array([per_phase[p] for p in (0, 5, 10, 15)])          # 4 x nstarts
    print(f"  per-phase mean CAGR : " +
          "  ".join(f"p{p}={pm[i].mean():+.2%}" for i, p in enumerate((0, 5, 10, 15))), flush=True)
    print(f"  cross-phase sigma at a FIXED start: mean {pm.std(axis=0, ddof=1).mean():.2%}  "
          f"max {pm.std(axis=0, ddof=1).max():.2%}", flush=True)
    print("  -> this is the uncompensated dispersion tranching is meant to remove; it is "
          "measured\n     with start date held constant, so it cannot be calendar luck.",
          flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)



if __name__ == "__main__":
    main()
