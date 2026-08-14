"""thread CORE4 — try to break the regime-value result.

CORE3 found: value is a DEFENSIVE factor. Removing it from the bull leg while RAISING it in the
bear/crash legs improves CAGR, Sharpe and drawdown on both horizons (8yr +2.65pp/+0.044/+0.94pp;
26yr +1.28pp/+0.035/+0.56pp). Removing it everywhere makes things worse, which is what makes the
story mechanistic rather than a fit.

A clean Pareto improvement is exactly what both a real effect and a well-fitted one look like,
and this session has already produced two one-horizon traps (cap 0.15, umd_crash). So:

1. SENSITIVITY. The winner sits at bull val 0.00 / bear val 0.50 / crash val 0.65. Vary each
   axis alone. A real regime effect degrades smoothly over a plateau; a fit spikes at one cell.
   Included deliberately: bull val 0.05/0.10 (is 0 special, or is "low" enough?) and bear/crash
   values BELOW the deployed 0.3333 (does raising them actually matter, or is the whole effect
   just the bull-leg removal?).

2. SUB-PERIOD STABILITY. Three ~8yr blocks. A defensive factor SHOULD earn its keep in 2001-08
   and cost little in 2009-16/2017-25. If instead one block carries everything, "both horizons"
   is really one event seen twice.

Note the standing confound this cannot resolve: the value sleeve's inputs were broken until this
week (unbounded gross_margin, missing roe seqq>0 guard in the overlay, 66% 26yr fundamentals
coverage pre-rebuild). "Value does not work in bull regimes" and "the value sleeve is fed bad
data in bull regimes" remain entangled; this only tests whether the WEIGHTING result is stable.

Run:  python3 research/threadCORE4_validate.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063}


def W(mom, val, s5):
    return {"mom": mom, "val": val, "s5": s5, "s3": 0.0}


def cfg(bull_val, bear_val, crash_val):
    """bull leg gets the residual in momentum; bear/crash keep lowvol as the remainder."""
    return {"mom_w": 0.85 - (bull_val - 0.0), "val_w": bull_val, "lv_w": 0.15,
            "bear_weights": W(0.08, bear_val, 1.0 - 0.08 - bear_val),
            "crash_weights": W(0.12, crash_val, 1.0 - 0.12 - crash_val)}


WIN = cfg(0.00, 0.50, 0.65)

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01", "2018-02-15"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17", "2001-02-01", "2001-02-15"], "2025-12-31"),
]

SUBS = [
    ("sub 2001-08", "data/wrds/sp1500_universe_2000.pkl", ["2001-01-02", "2001-02-01"], "2008-12-31"),
    ("sub 2009-16", "data/wrds/sp1500_universe_2000.pkl", ["2009-01-02", "2009-02-02"], "2016-12-31"),
    ("sub 2017-25", "data/wrds/sp1500_universe_2000.pkl", ["2017-01-03", "2017-02-01"], "2025-12-31"),
]

VARIANTS = [
    ("DEPLOYED", {}),
    ("WIN b0/br.50/cr.65", dict(WIN)),
    # bull-leg value: is 0 special, or is "low" enough?
    ("bull val .05", cfg(0.05, 0.50, 0.65)),
    ("bull val .10", cfg(0.10, 0.50, 0.65)),
    ("bull val .20", cfg(0.20, 0.50, 0.65)),
    # does RAISING bear/crash value matter, or is it all the bull removal?
    ("bear .3333 (deployed lvl)", cfg(0.00, 0.3333, 0.65)),
    ("bear .40", cfg(0.00, 0.40, 0.65)),
    ("bear .60", cfg(0.00, 0.60, 0.65)),
    ("crash .50 (deployed lvl)", cfg(0.00, 0.50, 0.50)),
    ("crash .75", cfg(0.00, 0.50, 0.75)),
    ("bear+crash at deployed lvls", cfg(0.00, 0.3333, 0.50)),
]


def clear_deployed(bt):
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


def block(title, path, starts, end, vlist):
    print("\n" + "=" * 96, flush=True)
    print(f"{title} | {len(starts)} starts", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=path)
    clear_deployed(bt)
    print(f"(loaded {time.time() - t0:.0f}s)", flush=True)
    h = (f"  {'variant':<28}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
         f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}")
    print(h); print("  " + "-" * (len(h) - 2), flush=True)
    base, out = None, {}
    for label, extra in vlist:
        c = dict(BASE); c.update(extra)
        cs, ss, ds = [], [], []
        for st in starts:
            m = bt.run(st, end, dict(c))
            a, b, d = stat(m["daily_values"])
            cs.append(a); ss.append(b); ds.append(d)
        r = (float(np.mean(cs)), float(np.mean(ss)), float(np.mean(ds)))
        if base is None:
            base = r; dc = dsh = dd = ""
        else:
            dc = f"{(r[0]-base[0])*100:>+8.2f}p"
            dsh = f"{r[1]-base[1]:>+9.3f}"
            dd = f"{(r[2]-base[2])*100:>+8.2f}p"
        out[label] = (r[0] - base[0], r[1] - base[1], r[2] - base[2])
        print(f"  {label:<28}{r[0]:>+9.2%}{r[1]:>8.2f}{r[2]:>+9.2%}{dc:>9}{dsh:>9}{dd:>9}",
              flush=True)
    del bt
    return out


def main():
    res = {}
    for t, p, s, e in PERIODS:
        res[t] = block(f"{t} | SENSITIVITY", p, s, e, VARIANTS)

    sub = {}
    for t, p, s, e in SUBS:
        sub[t] = block(f"{t} | SUB-PERIOD", p, s, e,
                       [("DEPLOYED", {}), ("WIN b0/br.50/cr.65", dict(WIN))])

    print("\n" + "=" * 96, flush=True)
    print("SENSITIVITY — beats deployed on BOTH horizons?", flush=True)
    print(f"  {'variant':<28}{'8yr dSh':>9}{'26yr dSh':>10}{'8yr dC':>9}{'26yr dC':>10}  verdict",
          flush=True)
    n = 0
    for label, _ in VARIANTS[1:]:
        a = res["8yr 2018-25"].get(label); b = res["26yr 2001-25"].get(label)
        if not a or not b:
            continue
        ok = a[1] > 0 and b[1] > 0 and a[0] > 0 and b[0] > 0
        n += 1 if ok else 0
        print(f"  {label:<28}{a[1]:>+9.3f}{b[1]:>+10.3f}{a[0]*100:>+8.2f}p{b[0]*100:>+9.2f}p  "
              f"{'YES' if ok else 'no'}", flush=True)
    print(f"\n  {n} of {len(VARIANTS)-1} perturbations beat deployed on both horizons.", flush=True)

    print("\nSUB-PERIOD (is it one event?):", flush=True)
    print(f"  {'block':<16}{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}", flush=True)
    for t, _, _, _ in SUBS:
        w = sub[t].get("WIN b0/br.50/cr.65")
        if w:
            print(f"  {t:<16}{w[0]*100:>+8.2f}p{w[1]:>+9.3f}{w[2]*100:>+8.2f}p", flush=True)


if __name__ == "__main__":
    main()
