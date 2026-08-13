"""thread CAP2 — does making the 2/N sleeve cap ACTUALLY BIND help?

Every sleeve ends with the same pattern:

    weights = {s: min(v/total, 2.0/N) for ...}   # cap
    weights = {s: w/sum(weights) for ...}        # renormalise  <-- undoes the cap

Capping the dominant name shrinks the denominator by almost exactly what was removed, so a
name holding 98% of raw score returns to ~93% despite a nominal 20% cap. Verified live: VIR
took ~93% of the value sleeve with the cap "applied". The cap is decorative.

That flaw is what turned ONE corrupt gross_margin into a whole-sleeve takeover. The [-1,1] gm
bound (74d57cb) removed that particular trigger, but the amplifier is still armed for any
future bad value — and it also means the documented "capped 2/N" behaviour has never actually
been what the strategy did, in live OR backtest.

This tests the proper fix: iterative WATER-FILLING. Cap the over-weight names AT the cap and
redistribute the remainder proportionally among the rest, repeating until nothing exceeds it.
The cap then genuinely binds and the weights still sum to 1.

  A  current  — cap-then-renormalise (cap does not bind)
  B  waterfill — cap binds

Applied to all three sleeves by wrapping their OUTPUT, so no production file is touched.
Cap per sleeve is 2/N over the names it actually returned, matching the intended rule.

NOTE ON METHOD: the sleeves are imported BY NAME into livemirror/main_production_backtest, so
patching only multi_strategy_engine leaves those references untouched and the probe silently
measures nothing (this bit the SI probe — it reported "0 calls" and looked like a result).
Every namespace holding the name is patched here, and the script HARD-FAILS if it patches none.

Run:  python3 research/threadCAP2_waterfill.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import strategies.multi_strategy_engine as M  # noqa: E402
import main_production_backtest as MPB  # noqa: E402
import livemirror_backtest as LM  # noqa: E402
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

WATERFILL = {"on": False}
STATS = {"calls": 0, "rebalanced": 0, "max_before": 0.0, "max_after": 0.0}


def water_fill(w, cap):
    """Cap that actually binds: pin over-cap names AT cap, redistribute the rest pro-rata."""
    w = {k: max(float(v), 0.0) for k, v in w.items()}
    tot = sum(w.values())
    if tot <= 0 or not w:
        return w
    w = {k: v / tot for k, v in w.items()}
    for _ in range(25):
        over = {k for k, v in w.items() if v > cap + 1e-12}
        if not over:
            break
        free = 1.0 - cap * len(over)
        rest = {k: v for k, v in w.items() if k not in over}
        rtot = sum(rest.values())
        if free <= 0 or rtot <= 0:            # cap too tight to satisfy -> equal weight
            return {k: 1.0 / len(w) for k in w}
        w = {k: (cap if k in over else v / rtot * free) for k, v in w.items()}
    return w


def _wrap(fn):
    def inner(*a, **kw):
        out = fn(*a, **kw)
        if not WATERFILL["on"] or not out:
            return out
        cap = 2.0 / max(len(out), 1)
        before = max(out.values())
        STATS["calls"] += 1
        if before > cap + 1e-12:
            STATS["rebalanced"] += 1
            STATS["max_before"] = max(STATS["max_before"], before)
        new = water_fill(out, cap)
        STATS["max_after"] = max(STATS["max_after"], max(new.values()) if new else 0)
        return new
    return inner


_NAMES = ["strategy_value", "strategy1_momentum_reversal", "strategy5_lowvol_quality"]
_patched = []
for _n in _NAMES:
    _orig = getattr(M, _n, None)
    if _orig is None:
        continue
    _w = _wrap(_orig)
    for _mod in (M, MPB, LM):
        if getattr(_mod, _n, None) is _orig:
            setattr(_mod, _n, _w)
            _patched.append(f"{_mod.__name__}.{_n}")
if not _patched:
    raise SystemExit("FATAL: patched nothing — the probe would silently measure zero effect.")
print("patched:", _patched, flush=True)


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
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 96, flush=True)
        print(f"{pname} | cap-then-renormalise vs WATER-FILL | {len(starts)}-start", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        hdr = f"{'arm':<30}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
        print(hdr); print("-" * len(hdr), flush=True)
        out = {}
        for label, on in [("A current (cap does NOT bind)", False),
                          ("B water-fill (cap binds)", True)]:
            WATERFILL["on"] = on
            for k in STATS:
                STATS[k] = 0 if isinstance(STATS[k], int) else 0.0
            cs, ss, ds = [], [], []
            for st in starts:
                m = bt.run(st, end, dict(BASE))
                c, s, d = stat(m["daily_values"])
                cs.append(c); ss.append(s); ds.append(d)
            r = (float(np.mean(cs)), float(np.mean(ss)), float(np.mean(ds)))
            out[label] = r
            print(f"{label:<30}{r[0]:>+9.2%}{r[1]:>8.2f}{r[2]:>+9.2%}", flush=True)
            if on:
                print(f"    sleeve calls {STATS['calls']}, cap was breached on "
                      f"{STATS['rebalanced']} ({STATS['rebalanced']/max(STATS['calls'],1):.1%}); "
                      f"max weight before {STATS['max_before']:.1%} -> after "
                      f"{STATS['max_after']:.1%}", flush=True)
        a = out["A current (cap does NOT bind)"]
        b = out["B water-fill (cap binds)"]
        print(f"\n  DELTA (waterfill - current): CAGR {(b[0]-a[0])*100:+.2f}pp | "
              f"Sharpe {b[1]-a[1]:+.3f} | MaxDD {(b[2]-a[2])*100:+.2f}pp", flush=True)
        WATERFILL["on"] = False
        del bt

    print("\n" + "=" * 96, flush=True)
    print("If the cap is rarely breached, this is dormant insurance — near-zero cost, and it", flush=True)
    print("disarms the amplifier that turned one bad gross_margin into a sleeve takeover.", flush=True)


if __name__ == "__main__":
    main()
