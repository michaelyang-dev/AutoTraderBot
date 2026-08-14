"""thread CORE3 — TRUE value-off: remove value from the BEAR and CRASH legs too.

Motivation, and it is not a small one. Every parameter the strategy runs on — sleeve weights
50/35/15, top_n 5, rebal 20d, trailing stop 40%, position cap 15% — was selected on the
PRE-AUDIT universes. The 2026-07-28 rebuild fixed 8 defects and moved the 26yr CAGR by -10.4pp
(look-ahead membership, survivorship via suffixed tickers, ticker-keyed price splicing, deleted
delisting losses, ticker-joined fundamentals, zero prices...). A parameter chosen on data with
survivorship bias and spliced price series is not a parameter chosen on this data.

This is potentially a bigger prize than any overlay: if an optimum moved, it moved in the
strategy itself, not in a bolt-on.

Method: ONE-AT-A-TIME around the deployed configuration, both horizons, multiple starts. No
joint optimisation — that would just manufacture an overfit. The question is narrow and
falsifiable: does the DEPLOYED value still sit at or near the top of each axis?

Read the results with discipline:
  * a deployed value that is clearly beaten on BOTH horizons is a real finding
  * a deployed value beaten on ONE horizon is noise (the session has produced several such
    traps already: umd_crash was +0.026 / -0.020)
  * differences inside ~0.6pp CAGR per start are at the documented noise floor

Run:  python3 research/threadCORE_reoptimise.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

# DEPLOYED configuration — the thing being questioned.
BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063}

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17", "2001-02-01"], "2025-12-31"),
]

# CORE2 varied only mom_w/val_w/lv_w, which feed ONLY the BULL leg. Measured on the 26yr,
# just 50.5% of rebalance dates are pure bull; 18.7% are pure BEAR where val_w is ignored
# outright, and the mean blend is 0.676. So CORE2's "85/00/15 value OFF" actually ran ~10.8%
# effective value (0.00*0.676 + 0.3333*0.324), not 0%. The gradient it found is real but the
# label was wrong.
#
# Here bear_weights and crash_weights are overridden too, so value can genuinely be removed
# everywhere -- and, separately, removed ONLY in bull while kept in bear/crash. If the
# bull-only version wins, value is regime protection rather than a return source, and the
# deployed 35% is simply mis-sized rather than wrong.
_B = lambda mom, val, s5: {"mom": mom, "val": val, "s5": s5, "s3": 0.0}

VARIANTS = [
    ("DEPLOYED 50/35/15", {}),
    # CORE2's winner, for reference (bull-leg only -> ~10.8% effective value)
    ("bull-only val 0 (CORE2)", {"mom_w": 0.85, "val_w": 0.00, "lv_w": 0.15}),
    # TRUE value-off: bull AND bear AND crash
    ("TRUE value OFF everywhere", {"mom_w": 0.85, "val_w": 0.00, "lv_w": 0.15,
                                   "bear_weights": _B(0.20, 0.00, 0.80),
                                   "crash_weights": _B(0.25, 0.00, 0.75)}),
    # value kept ONLY for the defensive legs (bull value = 0, bear/crash unchanged)
    ("value ONLY in bear/crash", {"mom_w": 0.85, "val_w": 0.00, "lv_w": 0.15}),
    # halve value everywhere rather than remove it
    ("value halved everywhere", {"mom_w": 0.68, "val_w": 0.17, "lv_w": 0.15,
                                 "bear_weights": _B(0.14, 0.17, 0.69),
                                 "crash_weights": _B(0.21, 0.25, 0.54)}),
    # the converse: MORE value in the defensive legs, none in bull
    ("bull val 0, bear/crash val UP", {"mom_w": 0.85, "val_w": 0.00, "lv_w": 0.15,
                                       "bear_weights": _B(0.08, 0.50, 0.42),
                                       "crash_weights": _B(0.12, 0.65, 0.23)}),
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


def main():
    res = {}
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 100, flush=True)
        print(f"{pname} | CORE PARAMETER RE-VALIDATION on audited data | {len(starts)} starts",
              flush=True)
        print("=" * 100, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        h = (f"  {'variant':<30}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
             f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}")
        print(h); print("  " + "-" * (len(h) - 2), flush=True)
        base = None
        for label, extra in VARIANTS:
            cfg = dict(BASE); cfg.update(extra)
            cs, ss, ds = [], [], []
            for st in starts:
                m = bt.run(st, end, dict(cfg))
                c, s, d = stat(m["daily_values"])
                cs.append(c); ss.append(s); ds.append(d)
            r = (float(np.mean(cs)), float(np.mean(ss)), float(np.mean(ds)))
            if base is None:
                base = r
                dc = dsh = dd = ""
            else:
                dc = f"{(r[0]-base[0])*100:>+8.2f}p"
                dsh = f"{r[1]-base[1]:>+9.3f}"
                dd = f"{(r[2]-base[2])*100:>+8.2f}p"
            res[(pname, label)] = (r[0] - base[0], r[1] - base[1], r[2] - base[2])
            print(f"  {label:<30}{r[0]:>+9.2%}{r[1]:>8.2f}{r[2]:>+9.2%}{dc:>9}{dsh:>9}{dd:>9}",
                  flush=True)
        del bt

    print("\n" + "=" * 100, flush=True)
    print("BEATS DEPLOYED ON BOTH HORIZONS (Sharpe AND CAGR)? — the only claim worth making",
          flush=True)
    print(f"  {'variant':<30}{'8yr dSh':>9}{'26yr dSh':>10}{'8yr dC':>9}{'26yr dC':>10}   verdict",
          flush=True)
    hits = []
    for label, _ in VARIANTS[1:]:
        a = res.get(("8yr 2018-25", label)); b = res.get(("26yr 2001-25", label))
        if not a or not b:
            continue
        sh = a[1] > 0 and b[1] > 0
        cg = a[0] > 0 and b[0] > 0
        v = "BEATS BOTH" if (sh and cg) else ("Sharpe-only" if sh else ("CAGR-only" if cg else "no"))
        if sh and cg:
            hits.append(label)
        print(f"  {label:<30}{a[1]:>+9.3f}{b[1]:>+10.3f}{a[0]*100:>+8.2f}p{b[0]*100:>+9.2f}p   {v}",
              flush=True)
    print(f"\n  {len(hits)} variant(s) beat the deployed config on both horizons: {hits}", flush=True)
    print("  If that list is empty, the deployed parameters survived the universe rebuild and", flush=True)
    print("  the original tuning was not an artefact of the poisoned data.", flush=True)


if __name__ == "__main__":
    main()
