"""EXP-031 — THE LEVERAGE FRONTIER for the baseline and both candidate options.

WHY
  Options 1 and 2 were each tested at ONE or TWO leverage settings, chosen because they roughly
  matched the deployed book's realised gross. That answered "is the change good at the exposure
  we already run" but NOT "what is the best exposure to run the change at". Those are different
  questions and the owner needs the second one to choose.

  It also matters because the two changes alter realised gross in opposite ways: removing the
  vol overlay RAISES gross at a given nominal leverage (no more de-risking), while tranching
  lowers it slightly (four smaller books hit the position cap differently). So "1.10x" does not
  mean the same exposure across arms, and the frontier has to be read on realised avg_gross,
  which is reported for every cell.

ARMS
  A  BASELINE   deployed: vol overlay ON
  B  OPTION 1   vol_scaling OFF                       (config change only)
  C  OPTION 2   vol_scaling OFF + tranching K=4        (option 1 PLUS tranching -- a superset)

  each at nominal leverage 1.00 / 1.10 / 1.25 / 1.49 / 1.75 / 2.00

WHAT TO WATCH
  Leverage is a CAGR-for-drawdown trade, already quantified (EXP-010: 1.49x vs 1.00x buys
  +2.60pp CAGR and costs -0.022 Sharpe and -15.87pp MaxDD, 0/12 on Sharpe). Nothing here will
  repeal that. The question is whether options 1/2 move the WHOLE frontier -- i.e. whether at
  any given drawdown they deliver more CAGR than the baseline does at the leverage that
  produces the same drawdown. That is the only honest way to compare arms that run different
  exposure.

  Everything is at the real account size (ACCT_CAP, default $50,000) with the credit gate ON,
  real time-varying financing, live sizing, integer shares and full costs.
  NOTE: EXP-027 measured $33k vs $50k across 12 audit cells and they matched to within
  0.1pp, so account size is not a material driver here -- but it is set correctly anyway.

Run:  python3 research/EXP031_leverage_frontier.py [8yr|26yr]
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

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
GATE = {"credit_pct": 0.95, "credit_derisk": 0.5}
CAP = float(os.environ.get("ACCT_CAP", 50_000.0))   # live account; was 33k, now ~50k
LEVS = [1.00, 1.10, 1.25, 1.49, 1.75, 2.00]
ARMS = [("A BASELINE (overlay ON)", [0], {}),
        ("B OPTION1 (no overlay)", [0], {"vol_scaling": False}),
        ("C OPTION2 (no ov + K=4)", [0, 5, 10, 15], {"vol_scaling": False})]


def run_cell(bt, phases, extra, lev):
    ck = CAP / len(phases)
    rows, gs = [], []
    for st in STARTS:
        per, g = [], []
        for p in phases:
            m = bt.run(st, END, {**DEPLOYED, **GATE, **extra, "leverage": lev,
                                 "initial_capital": ck, "rebal_phase": p,
                                 "live_sizing": True, "financing_curve": True})
            per.append(m["daily_values"]); g.append(m["avg_gross"])
        rows.append(stat(sum(x.reindex(per[0].index).ffill() for x in per)))
        gs.append(float(np.mean(g)))
    return rows, float(np.mean(gs))


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*104}\nEXP-031 LEVERAGE FRONTIER — {HZ}, {len(STARTS)} starts, ${CAP:,.0f}, "
          f"gate ON, real financing\n{'='*104}", flush=True)
    print(f"  {'arm':<26}{'lev':>6}{'avgGross':>10}{'CAGR':>10}{'sdCAGR':>9}{'Sharpe':>9}"
          f"{'Sortino':>9}{'MaxDD':>9}{'worstDD':>9}", flush=True)
    store = {}
    for name, phases, extra in ARMS:
        for lev in LEVS:
            rows, g = run_cell(bt, phases, extra, lev)
            c = np.array([r["cagr"] for r in rows]); s = np.array([r["sharpe"] for r in rows])
            so = np.array([r["sortino"] for r in rows]); d = np.array([r["dd"] for r in rows])
            store[(name, lev)] = (c, s, d, g)
            print(f"  {name:<26}{lev:>6.2f}{g:>10.4f}{c.mean():>+10.2%}{c.std(ddof=1):>9.2%}"
                  f"{s.mean():>9.3f}{so.mean():>9.3f}{d.mean():>9.1%}{d.min():>9.1%}",
                  flush=True)
        print(f"    ({time.time()-t0:.0f}s)", flush=True)

    # --- the only fair comparison: CAGR at MATCHED DRAWDOWN -------------------------------
    print(f"\n  === ISO-DRAWDOWN COMPARISON ===", flush=True)
    print(f"  Arms run different exposure, so comparing them at the same NOMINAL leverage is",
          flush=True)
    print(f"  meaningless. For each target drawdown, interpolate each arm's leverage curve and",
          flush=True)
    print(f"  report the CAGR it delivers AT that drawdown. Higher = strictly better.", flush=True)
    print(f"\n  {'target MaxDD':<14}" + "".join(f"{n.split()[0]+' '+n.split()[1]:>26}"
                                                for n, _, _ in ARMS), flush=True)
    for tgt in (-0.35, -0.40, -0.45, -0.50, -0.55, -0.60):
        row = f"  {tgt:<14.0%}"
        for name, _p, _e in ARMS:
            xs = np.array([abs(store[(name, L)][2].mean()) for L in LEVS])
            ys = np.array([store[(name, L)][0].mean() for L in LEVS])
            o = np.argsort(xs)
            xs, ys = xs[o], ys[o]
            if abs(tgt) < xs[0] or abs(tgt) > xs[-1]:
                row += f"{'  (off curve)':>26}"
            else:
                row += f"{np.interp(abs(tgt), xs, ys):>+26.2%}"
        print(row, flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
