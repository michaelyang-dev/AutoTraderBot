"""
LEVERAGE POLICY SEARCH — can we get MORE CAGR and LESS drawdown than the live vol-scaled
1.49x policy (cap 1.0)? The matched-risk test showed vol-scaling creates re-leverable risk
budget; here we try to SPEND it (cap>1.0 in calm) while a tail overlay holds the drawdown.

Tested on BOTH periods (2018-25 and 2001-25 incl. 2008+2020), multi-start:
  - vol-scale CAP sweep 1.0 -> 2.0 (lever up in calm periods)
  - portfolio DRAWDOWN breaker: existing get_drawdown_scale thresholds (NOT tuned here)
  - TREND gate: SPY>200-SMA (standard, NOT tuned) — no levering up into a downtrend
  - combinations

Overlay methodology == final_live_config_test: r_policy = gross_t * r_1x - financing, with
r_1x the UNSCALED (OFF) daily return and gross_t the policy's leverage path (causal — the
drawdown breaker uses YESTERDAY's drawdown, no look-ahead). Financing 6.3% on borrowed
(gross-1); de-levered cash earns 0% (conservative). Validated by reproducing
final_live_config's LIVE@1.49x baseline (+28.3%/-33.9% 8yr).

OVERFIT GUARD: a policy only "wins" if it beats the baseline on BOTH CAGR and MaxDD in
BOTH periods with the SAME params. Everything is reported, wins and losses.
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
FORK = os.path.join(ML, "research", "_levpol_fork.py")
A1 = '        self._rebal_log = []  # research: (date, {sym: target_weight}) per rebalance when record_targets set'
P1 = A1 + '\n        self._vs_log = []'
A2 = '                    vol_scale = min(1.5, max(0.3, vol_target / realized_vol))'
P2 = ('                    vol_scale = min(config.get("vol_scale_cap", 1.5), max(0.3, vol_target / realized_vol))\n'
      '                    self._vs_log.append((date, vol_scale))')
src = open(SRC).read()
assert src.count(A1) == 1 and src.count(A2) == 1, "fork anchors changed — re-check"
open(FORK, "w").write(src.replace(A1, P1).replace(A2, P2))
print("fork written", flush=True)

from _levpol_fork import FastBacktester  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
VS = {"vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 2.0}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
BASE_L, RATE = 1.49, 0.063
# EXISTING get_drawdown_scale thresholds (ibkr_engine.py) — NOT tuned in this search:
DD_THRESH = [(0.10, 1.00), (0.15, 0.85), (0.20, 0.70), (0.25, 0.50)]


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(r):
    r = r.dropna()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    c = (1 + r).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min())


def dd_scale(dd):
    for lvl, sc in DD_THRESH:
        if dd <= lvl:
            return sc
    return 0.35


def dvol_series(r_off, target, lookback=40, rebal=20, cap=1.0):
    """DOWNSIDE-vol targeting: piecewise-constant (every `rebal` days) leverage scale from
    the trailing-`lookback` NEGATIVE-return semideviation. Levers up when DOWNSIDE risk is
    contained even if upside vol is high — directly targets 'more CAGR, less DD'. Causal."""
    r = r_off.values
    scale = np.ones(len(r))
    cur = 1.0
    for t in range(len(r)):
        if t % rebal == 0 and t >= lookback:
            w = r[t - lookback:t]
            neg = w[w < 0]
            dv = (np.sqrt(np.mean(neg ** 2)) * np.sqrt(252)) if len(neg) > 1 else 0.15
            cur = min(cap, max(0.3, target / dv)) if dv > 0.01 else cap
        scale[t] = cur
    return pd.Series(scale, index=r_off.index)


def apply_policy(r_off, vs, trend, cap=1.0, dd_break=False, trend_gate=False,
                 dvol_target=None, rate=RATE):
    """base_gross = BASE_L*min(cap, vs) [or downside-vol scale if dvol_target set]; optional
    trend-gate (no lever>base in downtrend) and dd-break (scale by yesterday's drawdown).
    Sequential + causal. Returns (daily lev return series, avg gross)."""
    if dvol_target is not None:
        base_gross = BASE_L * dvol_series(r_off, dvol_target, cap=cap).values
    else:
        base_gross = BASE_L * np.minimum(cap, vs.values)
    if trend_gate:
        base_gross = np.where(trend.values, base_gross, BASE_L * np.minimum(1.0, vs.values))
    rr = r_off.values
    out = np.empty(len(rr)); gross_realized = np.empty(len(rr))
    eq = 1.0; peak = 1.0; dd = 0.0
    for t in range(len(rr)):
        g = base_gross[t] * (dd_scale(dd) if dd_break else 1.0)
        gross_realized[t] = g
        r = g * rr[t] - max(0.0, g - 1.0) * rate / 252
        out[t] = r
        eq *= (1 + r); peak = max(peak, eq); dd = 1 - eq / peak
    return pd.Series(out, index=r_off.index), float(np.mean(gross_realized))


POLICIES = [
    ("LIVE baseline (cap1.0)", dict(cap=1.0)),
    ("cap1.5", dict(cap=1.5)),
    ("cap2.0", dict(cap=2.0)),
    ("cap1.5 +DDbreak", dict(cap=1.5, dd_break=True)),
    ("cap1.5 +trend", dict(cap=1.5, trend_gate=True)),
    ("cap2.0 +trend +DDbreak", dict(cap=2.0, trend_gate=True, dd_break=True)),
    ("baseline +DDbreak (cap1.0)", dict(cap=1.0, dd_break=True)),
    ("DOWNSIDE-vol t0.09 cap1.0", dict(dvol_target=0.09, cap=1.0)),
    ("DOWNSIDE-vol t0.105 cap1.0", dict(dvol_target=0.105, cap=1.0)),
    ("DOWNSIDE-vol t0.12 cap1.0", dict(dvol_target=0.12, cap=1.0)),
    ("DOWNSIDE-vol t0.105 cap1.5", dict(dvol_target=0.105, cap=1.5)),
]


def main():
    results = {name: {} for name, _ in POLICIES}
    for pname, path, starts, end in PERIODS:
        print("=" * 96, flush=True)
        print(f"PERIOD {pname} | {len(starts)}-start avg | financing {RATE:.1%} on borrowed", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        runs = []
        for st in starts:
            r_off = bt.run(st, end, dict(V12))["daily_values"].pct_change().dropna()
            bt._vs_log = []
            cfg = dict(V12); cfg.update(VS)
            bt.run(st, end, cfg)
            vs = pd.Series({d: v for d, v in bt._vs_log})
            vs.index = pd.to_datetime(vs.index)
            vs = vs.reindex(r_off.index.union(vs.index)).ffill().reindex(r_off.index).fillna(1.0)
            try:
                spy = bt.prices["SPY"]
                sma = spy.rolling(200).mean()
                trend = (spy.reindex(r_off.index).ffill() > sma.reindex(r_off.index).ffill())
            except Exception:
                trend = pd.Series(True, index=r_off.index)
            runs.append((r_off, vs, trend))
        print(f"(loaded+ran {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'policy':<42}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'avgGro':>8}"
        print(hdr); print("-" * len(hdr), flush=True)
        for name, kw in POLICIES:
            cs, vls, ss, ds, gs = [], [], [], [], []
            for r_off, vs, trend in runs:
                ser, g = apply_policy(r_off, vs, trend, **kw)
                c, v, s, d = stats(ser)
                cs.append(c); vls.append(v); ss.append(s); ds.append(d); gs.append(g)
            cagr, vol, sh, dd, gross = (np.mean(cs), np.mean(vls), np.mean(ss),
                                        np.mean(ds), np.mean(gs))
            results[name][pname] = (cagr, dd, sh)
            print(f"{name:<42}{cagr:>+8.1%}{vol:>7.1%}{sh:>8.2f}{dd:>+8.1%}{gross:>8.2f}", flush=True)
        del bt

    print("\n" + "=" * 96, flush=True)
    print("VERDICT — policies beating baseline on CAGR *and* MaxDD in BOTH periods:", flush=True)
    print("=" * 96, flush=True)
    b = {p: results["LIVE baseline (cap1.0)"][p] for p in results["LIVE baseline (cap1.0)"]}
    any_win = False
    for name, per in results.items():
        if name.startswith("LIVE baseline"):
            continue
        wins = all(per[p][0] > b[p][0] + 0.001 and per[p][1] > b[p][1] + 0.001 for p in per)
        if wins:
            any_win = True
            deltas = " | ".join(f"{p}: +{(per[p][0]-b[p][0])*100:.1f}pp CAGR, {(per[p][1]-b[p][1])*100:+.1f}pp DD"
                                for p in per)
            print(f"  WIN  {name}: {deltas}", flush=True)
    if not any_win:
        print("  NONE — no policy Pareto-beats the baseline in both periods.", flush=True)
        print("  (that itself is the finding: the live config is near-optimal on this axis)", flush=True)


if __name__ == "__main__":
    main()
