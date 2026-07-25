"""
FRACTIONAL vs INTEGER shares — how much does the whole-share constraint on the SMALL
$34K IBKR live account cost in CAGR vs fractional sizing (Alpaca-style / the backtest's
continuous weights)?

Forks the production backtester to inject the LIVE integer-share rounding into position
sizing: int() whole shares + a closed-loop multiplier rescale to still hit the target
gross (exactly ibkr_engine._calibrate_quantities), at the live account NAV. Compares
CAGR/Sharpe/MaxDD: fractional vs integer, at $34K (current size — drag is highest on a
small book because a $546 WDC share is a lumpy chunk) and $100K/$500K/$2M (the drag
shrinks as the account grows). Fractional is scale-invariant (sanity check: same CAGR
at every size). 3-start 8yr, live v12 config, no vol-scaling (isolates the share effect).
"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = os.path.join(ML, "main_production_backtest.py")
FORK = os.path.join(ML, "research", "_intshare_fork.py")

A1 = "        cash = INITIAL_CASH"
P1 = "        cash = globals().get('_START_CAP', INITIAL_CASH)"
A2 = "            target_d = {s: w * total_val * eq_pct for s, w in combined.items()}"
P2 = A2 + """
            if globals().get('_INTEGER'):
                # LIVE integer-share sizing: int() whole shares + closed-loop multiplier
                # rescale to hit the target gross (ibkr_engine._calibrate_quantities).
                _tg = sum(target_d.values())
                _pm = {s: today[s] for s in list(target_d) if today.get(s, 0) > 0}
                _sh, _m = {}, 1.0
                for _ in range(4):
                    _sh = {s: int(target_d[s] * _m / _pm[s]) for s in _pm}
                    _ag = sum(_sh[s] * _pm[s] for s in _sh)
                    if _ag <= 0:
                        break
                    _m = min(_m * (_tg / _ag), _m * 1.8 / 1.49)
                target_d = {s: _sh[s] * _pm[s] for s in _sh if _sh[s] > 0}"""

src = open(SRC).read()
assert src.count(A1) == 1 and src.count(A2) == 1, "fork anchors changed — re-check"
open(FORK, "w").write(src.replace(A1, P1).replace(A2, P2))
print("fork written", flush=True)

import _intshare_fork as F  # noqa: E402
from live_config import V12_LIVE_BACKTEST_CONFIG  # noqa: E402

V12 = dict(V12_LIVE_BACKTEST_CONFIG)
for k in ("vol_scaling", "vol_target", "vol_lookback", "vol_scale_cap"):
    V12.pop(k, None)
STARTS = ["2018-01-02", "2018-01-17", "2018-02-01"]
END = "2025-12-31"


def stats(vals):
    r = vals.pct_change().dropna()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    curve = (1 + r).cumprod()
    return (curve.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((curve - curve.cummax()) / curve.cummax()).min())


def main():
    bt = F.FastBacktester()
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

    t0 = time.time()
    frac_cagr = None
    print(f"\n{'mode':<26}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'vs fractional':>15}", flush=True)
    print("-" * 65, flush=True)
    for cap in [34_000, 100_000, 500_000, 2_000_000]:
        for integer in [False, True]:
            F._START_CAP = float(cap)
            F._INTEGER = bool(integer)
            cs, ss, ds = [], [], []
            for st in STARTS:
                m = bt.run(st, END, V12)
                c, v, s, d = stats(m["daily_values"])
                cs.append(c); ss.append(s); ds.append(d)
            cagr = float(np.mean(cs))
            label = f"${cap//1000}K {'INTEGER' if integer else 'fractional'}"
            if not integer and frac_cagr is None:
                frac_cagr = cagr
            drag = "" if not integer else f"{(cagr - frac_cagr) * 100:>+8.2f}pp"
            print(f"{label:<26}{cagr:>+8.1%}{np.mean(ss):>8.2f}{np.mean(ds):>+8.1%}{drag:>15}",
                  flush=True)
    # Isolate the CURRENT-size drag: rolling 1-yr windows each starting fresh at $34K
    # (the 8-yr runs above compound away from $34K, diluting the small-account effect).
    print(f"\n{'rolling 1yr @ $34K':<26}{'frac CAGR':>10}{'int CAGR':>10}{'int - frac':>12}",
          flush=True)
    print("-" * 58, flush=True)
    F._START_CAP = 34_000.0
    diffs = []
    for yr in range(2018, 2025):
        w0, w1 = f"{yr}-01-02", f"{yr}-12-31"
        row = {}
        for integer in (False, True):
            F._INTEGER = integer
            cs = []
            for st_off in (w0, f"{yr}-01-17", f"{yr}-02-01"):
                m = bt.run(st_off, w1, V12)
                cs.append(stats(m["daily_values"])[0])
            row[integer] = float(np.mean(cs))
        d = (row[True] - row[False]) * 100
        diffs.append(d)
        print(f"{yr:<26}{row[False]:>+10.1%}{row[True]:>+10.1%}{d:>+10.2f}pp", flush=True)
    print("-" * 58, flush=True)
    print(f"{'MEAN drag (7 windows)':<26}{'':>10}{'':>10}{np.mean(diffs):>+10.2f}pp", flush=True)
    print(f"{'std across windows':<26}{'':>10}{'':>10}{np.std(diffs):>+10.2f}pp", flush=True)

    print(f"\n(fractional is scale-invariant; INTEGER drag = the whole-share cost at that "
          f"account size. {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
