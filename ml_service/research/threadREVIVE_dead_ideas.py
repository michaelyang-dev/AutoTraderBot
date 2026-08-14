"""thread REVIVE — re-test previously-DEAD ideas under the many-start standard, on clean data.

Two things invalidate most of the "dead" verdicts in this repo:
  1. they were reached on the PRE-AUDIT universes (the rebuild moved 26yr CAGR by -10.4pp), and
  2. they were reached on 3-4 starts, which threadCANON has now shown cannot detect a 2pp edge
     (8yr per-start sigma 7.07pp; a delta's sigma ~2.30pp).

That combination already flipped the SIGN of the gross_margin result and killed a value-weight
finding that looked airtight. So "dead" is not a safe status for anything tested that way.

This re-runs the ideas that already have config hooks in livemirror, so no new modelling risk is
introduced — only the measurement standard changes.

TWO-STAGE DESIGN, deliberately. Stage 1 runs 12 starts on the 26yr ONLY: it is ~2.5x tighter
(sigma 2.75pp vs 7.07pp) and therefore the better discriminator, and it costs half the compute.
Only ideas that clear stage 1 earn an 8yr confirmation. Judging on the noisier horizon first
would be exactly backwards.

Judged on SIGN-CONSISTENCY across starts, never the mean. 6/12 is a coin flip.

Run:  python3 research/threadREVIVE_dead_ideas.py
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

B = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
     "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
     "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
     "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
     "financing_rate": 0.063}

# Previously reported dead/marginal. Hooks already exist, so only the standard changes.
IDEAS = [
    ("vol-managed mom 0.20", {"vol_managed_mom": 0.20, "vmm_cap": 1.0}),
    ("vol-managed mom 0.30", {"vol_managed_mom": 0.30, "vmm_cap": 1.0}),
    ("risk parity weights", {"use_rp": True, "rp_power": 1.0}),
    ("sector cap 2/sleeve", {"sector_cap": 2}),
    ("mom quality: gp_assets", {"mom_quality_filter": "gp_assets", "mom_quality_pool": 15}),
    ("mom quality: roe", {"mom_quality_filter": "roe", "mom_quality_pool": 15}),
    ("signal-exit every 5d", {"signal_exit_every": 5}),
    ("park idle cash in IEF", {"park_etf": "IEF"}),
    ("bull lever 1.15x", {"bull_lever": 1.15, "bull_breadth": 0.60}),
]

STAGE1 = ("26yr", "data/wrds/sp1500_universe_2000.pkl", [2001, 2002])
STAGE2 = ("8yr", "data/wrds/complete_sp1500_universe.pkl", [2018, 2019])


def cd(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}
    bt._si_change_ranks_by_month = {}
    bt._si_months = []
    bt.uni._short_interest_rank = {}
    bt.uni._si_change_rank = {}


def stat(v):
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
            dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
            ((v - v.cummax()) / v.cummax()).min())


def run_stage(label, path, yrs_, ideas):
    starts = [f"{y}-{m:02d}-03" for y in yrs_ for m in range(1, 13, 2)]
    bt = LiveMirrorBacktester(universe_path=path)
    cd(bt)
    base = []
    for st in starts:
        m = bt.run(st, "2025-12-31", dict(B))
        base.append(stat(m["daily_values"]))
    print(f"\n=== {label}: {len(starts)} starts | baseline CAGR "
          f"{np.mean([b[0] for b in base]):+.2%} ===", flush=True)
    print(f"  {'idea':<26}{'dCAGR':>9}{'dSharpe':>9}{'+CAGR':>8}{'+Sharpe':>9}{'verdict':>12}",
          flush=True)
    survivors = []
    for name, extra in ideas:
        dc, ds = [], []
        ok = True
        for i, st in enumerate(starts):
            cfg = dict(B); cfg.update(extra)
            try:
                m = bt.run(st, "2025-12-31", cfg)
            except Exception as e:
                print(f"  {name:<26} FAILED {type(e).__name__}: {e}", flush=True)
                ok = False
                break
            a, s, _ = stat(m["daily_values"])
            dc.append(a - base[i][0]); ds.append(s - base[i][1])
        if not ok:
            continue
        a, b = np.array(dc), np.array(ds)
        nc, ns = int((a > 0).sum()), int((b > 0).sum())
        # survive = clearly better than a coin flip on BOTH metrics
        v = "SURVIVES" if (ns >= 9 and nc >= 8) else ("marginal" if ns >= 7 else "dead")
        if v == "SURVIVES":
            survivors.append((name, extra))
        print(f"  {name:<26}{a.mean()*100:>+8.2f}p{b.mean():>+9.3f}"
              f"{nc:>5}/{len(a)}{ns:>6}/{len(b)}{v:>12}", flush=True)
    del bt
    return survivors


def main():
    t0 = time.time()
    lbl, path, yrs_ = STAGE1
    print("=" * 92, flush=True)
    print("STAGE 1 — 26yr only (tighter: sigma 2.75pp vs 8yr's 7.07pp)", flush=True)
    print("=" * 92, flush=True)
    surv = run_stage(lbl, path, yrs_, IDEAS)
    print(f"\n  stage-1 survivors: {[s[0] for s in surv] or 'NONE'}  ({time.time()-t0:.0f}s)",
          flush=True)

    if not surv:
        print("\nNothing cleared the 26yr. The dead verdicts hold under the better standard —",
              flush=True)
        print("which is itself worth knowing: they were not merely artefacts of few starts.",
              flush=True)
        return

    lbl2, path2, yrs2 = STAGE2
    print("\n" + "=" * 92, flush=True)
    print("STAGE 2 — 8yr confirmation of stage-1 survivors", flush=True)
    print("=" * 92, flush=True)
    final = run_stage(lbl2, path2, yrs2, surv)
    print(f"\n  CLEARED BOTH HORIZONS: {[s[0] for s in final] or 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
