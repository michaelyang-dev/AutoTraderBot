"""
FINAL LIVE-CONFIG TEST: exact live weight scheme (use_rp=False, cap 0.10-of-book ~ 15%-of-NAV at 1.49x) + vol-scaling + leverage + financing.

Prior A/B was at 1x with zero financing. Live reality: closed-loop 1.49x, margin on the
borrowed ~0.49x at IBKR's small-account rate (~6.3%/yr; 5%/7% sensitivity). Vol-scaling
changes the financing bill too: when scaled to s, gross = 1.49*s and borrowing (and its
cost) shrinks — max(0, 1.49*s - 1). This test measures everything:

  OFF  @1x, @1.49x(+fin)          — the unscaled book, unlevered vs levered
  LIVE @1x, @1.49x(+fin)          — the deployed vol-scaled policy (t0.15, cap 1.0)
  LIVE @matched-risk(+fin)        — LIVE levered until its vol equals OFF@1.49x's vol

Overlay math on the fork-run daily series: r_lev = L*r_1x - max(0, L*s_t - 1)*rate/252,
with s_t the rebalance-day vol_scale (piecewise-constant, exactly how live applies it;
logged by the fork). Excess cash when de-levered is credited NOTHING (conservative).
NOT modeled: margin calls / forced liquidation in the levered drawdowns — read MaxDD
with that in mind. 2 periods x multi-start, deployed data condition.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np
import pandas as pd

SRC = os.path.join(ML, "main_production_backtest.py")
FORK = os.path.join(ML, "research", "_levfin_fork.py")
A1 = '        self._rebal_log = []  # research: (date, {sym: target_weight}) per rebalance when record_targets set'
P1 = A1 + '\n        self._vs_log = []  # research: (date, vol_scale) applied at each rebalance'
A2 = '                    vol_scale = min(1.5, max(0.3, vol_target / realized_vol))'
P2 = ('                    vol_scale = min(config.get("vol_scale_cap", 1.5), max(0.3, vol_target / realized_vol))\n'
      '                    self._vs_log.append((date, vol_scale))')
src = open(SRC).read()
assert src.count(A1) == 1 and src.count(A2) == 1, "fork anchors changed — re-check"
open(FORK, "w").write(src.replace(A1, P1).replace(A2, P2))
print("fork written")

from _levfin_fork import FastBacktester  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
LIVE_FLAGS = {"vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0}
PERIODS = [
    ("8yr 2018-2025", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-2025", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
BASE_L, RATE, RATES = 1.49, 0.063, (0.05, 0.063, 0.07)


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(r):
    r = r.dropna()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    curve = (1 + r).cumprod()
    return (curve.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((curve - curve.cummax()) / curve.cummax()).min())


def lever(r1x, vs, L, rate):
    """r_lev = L*r - financing. gross_t = L*s_t; borrowed = max(0, gross-1)."""
    gross = L * vs
    fin = np.maximum(0.0, gross - 1.0) * (rate / 252)
    return L * r1x - fin, float(gross.mean()), float(fin.sum() / (len(r1x) / 252))


def run_policy(bt, starts, end, flags):
    """avg daily-return series list + vol-scale series list across starts."""
    out = []
    for st in starts:
        cfg = dict(V12); cfg.update(flags)
        m = bt.run(st, end, cfg)
        r = m["daily_values"].pct_change().dropna()
        vs = pd.Series(1.0, index=r.index)
        if getattr(bt, "_vs_log", None):
            s = pd.Series({d: v for d, v in bt._vs_log})
            s.index = pd.to_datetime(s.index)
            vs = s.reindex(r.index.union(s.index)).ffill().reindex(r.index).fillna(1.0)
        out.append((r, vs))
    return out


def main():
    for pname, path, starts, end in PERIODS:
        print("=" * 100, flush=True)
        print(f"PERIOD {pname} | {len(starts)}-start avg | financing on borrowed only | no margin-call modeling", flush=True)
        print("=" * 100, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        off = run_policy(bt, starts, end, {})
        live = run_policy(bt, starts, end, LIVE_FLAGS)

        hdr = f"{'variant':<34}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'avgGross':>9}{'fin/yr':>7}"
        print(hdr); print("-" * len(hdr), flush=True)

        def row(name, runs, L, rate):
            cs, vls, ss, ds, gs, fs = [], [], [], [], [], []
            for r, vs in runs:
                rl, g, f = lever(r, vs, L, rate)
                c, v, s, d = stats(rl)
                cs.append(c); vls.append(v); ss.append(s); ds.append(d); gs.append(g); fs.append(f)
            n = len(runs)
            print(f"{name:<34}{np.mean(cs):>+8.1%}{np.mean(vls):>7.1%}{np.mean(ss):>8.2f}"
                  f"{np.mean(ds):>+8.1%}{np.mean(gs):>9.2f}{np.mean(fs):>7.2%}", flush=True)
            return np.mean(vls)

        row("OFF   @1.00x", off, 1.0, 0.0)
        v_off_lev = row(f"OFF   @{BASE_L}x  fin {RATE:.1%}", off, BASE_L, RATE)
        row("LIVE  @1.00x", live, 1.0, 0.0)
        v_live_lev = row(f"LIVE  @{BASE_L}x  fin {RATE:.1%}", live, BASE_L, RATE)
        # matched-risk: lever LIVE until its levered vol == OFF's levered vol
        lm = BASE_L * (v_off_lev / v_live_lev) if v_live_lev > 0 else BASE_L
        row(f"LIVE  @{lm:.2f}x MATCHED-RISK fin", live, lm, RATE)
        for rr in RATES:
            if rr != RATE:
                row(f"LIVE  @{BASE_L}x  fin {rr:.1%}", live, BASE_L, rr)
        del bt
    print("\nHONEST NOTES: overlay assumes leverage reset each rebalance (matches the live", flush=True)
    print("closed-loop sizing); intraday margin-call risk in the deep levered DDs NOT modeled;", flush=True)
    print("de-levered excess cash credited 0% (conservative).", flush=True)


if __name__ == "__main__":
    main()
