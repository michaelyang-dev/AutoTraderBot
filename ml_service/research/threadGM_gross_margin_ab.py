"""thread GM — A/B the gross_margin plausibility bound (commit 74d57cb).

WHY: gross_margin = (saleq-cogsq)/saleq, so |gm| > 1 is definitionally impossible.
Compustat rows with saleq ~ 0 produce garbage (VIR live: 4561.72; 8yr universe holds
values down to -13,878 and the 26yr down to -39,217). Two separate failure modes:

  1. strategy_value filtered only on the LOWER side (g < 0.15), so a huge positive gm
     passed and dominated the score (gm carries a 0.25 weight).
  2. strategy5_lowvol_quality feeds gm through zscore(). ONE absurd value drags the mean
     and inflates the std so every legitimate name collapses toward z ~ 0 — measured on
     synthetic data: legit-name z-spread 1.0000 -> 0.0012, an 830x loss of discriminating
     power. The quality factor is switched OFF, not merely tilted.

Contamination is near-universal by DATE, which is why this is worth measuring rather than
assuming it is a rounding error:
    8yr  : 80,398 obs outside [-1,1] over 2,391/2,411 dates (99.2%)
    26yr : 186,842 obs outside [-1,1] over 6,519/6,539 dates (99.7%)

ARM A = pre-fix  (_sane_gross_margin patched to identity -> old behavior)
ARM B = post-fix (current shipped code)

Both arms call clear_deployed() so the deployed-only aux maps (short interest, price
targets, fin growth, ...) are zeroed and the run mirrors what live can actually reproduce.

Run:  python3 research/threadGM_gross_margin_ab.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import strategies.multi_strategy_engine as M  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063}

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]

# the shipped guard, captured before we patch anything
_SHIPPED = M._sane_gross_margin


def _identity(gm_map):
    """Pre-fix behavior: pass gross_margin straight through, absurd values included."""
    return gm_map


ARMS = [
    ("A pre-fix  (no gm bound)", _identity),
    ("B post-fix (gm in [-1,1])", _SHIPPED),
]


def clear_deployed(bt):
    """Zero the deployed-only aux maps so the backtest models what live can reproduce."""
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals"]:
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
    results = {}
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 96, flush=True)
        print(f"{pname} | gross_margin bound A/B | live-mirror 1.49x, integer, $50k | "
              f"{len(starts)}-start", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(universe loaded {time.time() - t0:.0f}s)", flush=True)

        hdr = (f"{'arm':<28}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
               f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}")
        print(hdr)
        print("-" * len(hdr), flush=True)

        base = None
        for name, fn in ARMS:
            M._sane_gross_margin = fn          # swap the guard for this arm
            cs, ss, ds = [], [], []
            for st in starts:
                m = bt.run(st, end, dict(BASE))
                c, s, d = stat(m["daily_values"])
                cs.append(c)
                ss.append(s)
                ds.append(d)
            c, s, d = float(np.mean(cs)), float(np.mean(ss)), float(np.mean(ds))
            if base is None:
                base = (c, s, d)
                dc = dsh = dd = ""
            else:
                dc = f"{(c - base[0]) * 100:>+8.2f}p"
                dsh = f"{(s - base[1]):>+9.3f}"
                dd = f"{(d - base[2]) * 100:>+8.2f}p"
            results[(pname, name)] = (c, s, d, cs, ss, ds)
            print(f"{name:<28}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}{dc:>9}{dsh:>9}{dd:>9}", flush=True)

        # per-start detail so a single lucky start cannot carry the verdict
        print(f"\n  per-start CAGR ({len(starts)} starts):", flush=True)
        for name, _ in ARMS:
            cs = results[(pname, name)][3]
            print(f"    {name:<28} " + "  ".join(f"{x:+.2%}" for x in cs), flush=True)
        del bt

    M._sane_gross_margin = _SHIPPED            # always restore
    print("\n" + "=" * 96, flush=True)
    print("VERDICT: B >= A on BOTH periods -> the bound is free or positive, keep it.", flush=True)
    print("         B <  A on BOTH periods -> the corrupt values were accidentally", flush=True)
    print("         load-bearing; investigate before trusting either number.", flush=True)
    print("NOTE: differences inside ~0.6pp/start are at the documented noise floor.", flush=True)


if __name__ == "__main__":
    main()
