"""
DEPLOYED LIVE METHOD at $50K — the byte-for-byte production config:
  1.49x leverage + financing + DE-RISK-ONLY vol-scaling (cap 1.0) + INTEGER round-down shares.
Compares to the same with FRACTIONAL shares (the +28.3% headline) to show what the whole-
number round-down does to the number at a $50K account.

One fork, four injections into the backtester:
  1. start capital = $50K (so integer truncation is at the live account scale)
  2. vol_scale cap = config (1.0 = de-risk only, exactly live)
  3. vs_log (records the rebalance-day vol_scale for the financing overlay)
  4. LIVE integer sizing: int() whole shares + closed-loop rescale (ibkr_engine._calibrate_quantities)
Leverage overlay: r_lev = 1.49*r_1x - financing on borrowed(=max(0,1.49*vs-1)), exactly
final_live_config_test. Integer truncation is applied at the UNLEVERED position scale
(~$2-7.5k/pos) — slightly SMALLER than the live's levered positions, so this is CONSERVATIVE
(the real levered $50K account has even less integer lumpiness). Both periods.
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
FORK = os.path.join(ML, "research", "_dep50_fork.py")
src = open(SRC).read()

A1 = "        cash = INITIAL_CASH"
A2 = "        self._rebal_log = []  # research: (date, {sym: target_weight}) per rebalance when record_targets set"
A3 = "                    vol_scale = min(1.5, max(0.3, vol_target / realized_vol))"
A4 = "            target_d = {s: w * total_val * eq_pct for s, w in combined.items()}"
for a in (A1, A2, A3, A4):
    assert src.count(a) == 1, f"anchor changed: {a[:40]}"
INT_INJ = A4 + """
            if globals().get('_INTEGER'):
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
src = src.replace(A1, "        cash = globals().get('_START_CAP', INITIAL_CASH)")
src = src.replace(A2, A2 + "\n        self._vs_log = []")
src = src.replace(A3, "                    vol_scale = min(config.get('vol_scale_cap', 1.5), "
                      "max(0.3, vol_target / realized_vol))\n                    self._vs_log.append((date, vol_scale))")
src = src.replace(A4, INT_INJ)
open(FORK, "w").write(src)
print("fork written", flush=True)

import _dep50_fork as F  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
LIVE_FLAGS = {"vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
BASE_L, RATE = 1.49, 0.063


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def lever(r1x, vs):
    gross = BASE_L * vs
    fin = np.maximum(0.0, gross - 1.0) * (RATE / 252)
    return BASE_L * r1x - fin, float(gross.mean())


def stats(rl):
    yrs = (rl.index[-1] - rl.index[0]).days / 365.25
    c = (1 + rl).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1, rl.std() * np.sqrt(252),
            (rl.mean() / rl.std() * np.sqrt(252)) if rl.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min())


def main():
    t0 = time.time()
    for pname, path, starts, end in PERIODS:
        print("=" * 80, flush=True)
        print(f"PERIOD {pname} | {len(starts)}-start | 1.49x + fin + de-risk vol-scale | $50K", flush=True)
        print("=" * 80, flush=True)
        bt = F.FastBacktester(universe_path=path)
        clear_deployed(bt)
        hdr = f"{'shares':<26}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'avgGross':>9}"
        print(hdr); print("-" * len(hdr), flush=True)
        frac_cagr = None
        for label, integer in [("FRACTIONAL (the +28%)", False),
                               ("INTEGER round-down $50K", True)]:
            F._START_CAP = 50_000.0
            F._INTEGER = bool(integer)
            cs, vs_, ss, ds, gs = [], [], [], [], []
            for st in starts:
                bt._vs_log = []
                m = bt.run(st, end, {**V12, **LIVE_FLAGS})
                r = m["daily_values"].pct_change().dropna()
                vser = pd.Series(1.0, index=r.index)
                if bt._vs_log:
                    sp = pd.Series({d: v for d, v in bt._vs_log})
                    sp.index = pd.to_datetime(sp.index)
                    vser = sp.reindex(r.index.union(sp.index)).ffill().reindex(r.index).fillna(1.0)
                rl, g = lever(r, vser)
                c, v, s, d = stats(rl)
                cs.append(c); vs_.append(v); ss.append(s); ds.append(d); gs.append(g)
            cagr = float(np.mean(cs))
            if frac_cagr is None:
                frac_cagr = cagr
            tag = "" if integer is False else f"  ({(cagr-frac_cagr)*100:+.2f}pp vs fractional)"
            print(f"{label:<26}{cagr:>+8.1%}{np.mean(vs_):>7.1%}{np.mean(ss):>8.2f}"
                  f"{np.mean(ds):>+8.1%}{np.mean(gs):>9.2f}{tag}", flush=True)
        del bt
    print(f"\n(both = full deployed config; integer = live's whole-share round-down. {time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
