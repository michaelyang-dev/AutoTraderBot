"""
VOL-POLICY AT LEVERAGE — user challenge: "t0.20/cap1.5 sounds better than de-risk-only".

At 1x, B(t.20/cap1.5) had more CAGR, D(live t.15/cap1.0) more Sharpe/less DD. But live runs
1.49x with financing, where the game changes: an up-scale of 1.5 wants 1.49*1.5 = 2.24x gross
(beyond Reg-T 2.0 — clamped here), borrows more (financing bill), and levered compounding
punishes the extra vol. This test settles it on the LIVE weight scheme (use_rp=False, cap .10):

  B  t0.20 cap1.5   |  C  t0.20 cap1.0   |  D  t0.15 cap1.0 (LIVE)   — all @1.49x, fin 6.3%,
  gross clamped to 2.0 (Reg-T), plus each policy's 1x row for reference.

8yr + 26yr, multi-start, deployed condition. Uses the _levfin_fork (vol_scale logged per rebalance).
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np
import pandas as pd
from _levfin_fork import FastBacktester  # created by leverage_financing_test.py runs

V12_LIVE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
            "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
            "cap": 0.10, "use_rp": False,
            "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
POLICIES = [
    ("B t.20 cap1.5", {"vol_scaling": True, "vol_target": 0.20, "vol_lookback": 40, "vol_scale_cap": 1.5}),
    ("C t.20 cap1.0", {"vol_scaling": True, "vol_target": 0.20, "vol_lookback": 40, "vol_scale_cap": 1.0}),
    ("D t.15 cap1.0 LIVE", {"vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0}),
]
PERIODS = [
    ("8yr 2018-2025", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-2025", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
L, RATE, REGT = 1.49, 0.063, 2.0


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


def main():
    for pname, path, starts, end in PERIODS:
        print("=" * 100, flush=True)
        print(f"PERIOD {pname} | live weight scheme | @{L}x fin {RATE:.1%} | gross clamped {REGT}x (Reg-T)", flush=True)
        print("=" * 100, flush=True)
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        hdr = f"{'policy':<22}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'avgGross':>9}{'fin/yr':>7}{'clamp%':>7}"
        print(hdr); print("-" * len(hdr), flush=True)
        for label, flags in POLICIES:
            cs, vls, ss, ds, gs, fs, cl = [], [], [], [], [], [], []
            for st in starts:
                cfg = dict(V12_LIVE); cfg.update(flags)
                m = bt.run(st, end, cfg)
                r = m["daily_values"].pct_change().dropna()
                vs = pd.Series(1.0, index=r.index)
                if getattr(bt, "_vs_log", None):
                    s = pd.Series({d: v for d, v in bt._vs_log}); s.index = pd.to_datetime(s.index)
                    vs = s.reindex(r.index.union(s.index)).ffill().reindex(r.index).fillna(1.0)
                gross_want = L * vs
                gross = np.minimum(gross_want, REGT)          # Reg-T feasibility clamp
                eff_mult = gross / vs                          # actual leverage applied to the 1x series
                fin = np.maximum(0.0, gross - 1.0) * (RATE / 252)
                rl = eff_mult * r - fin
                c, v, sh, d = stats(rl)
                cs.append(c); vls.append(v); ss.append(sh); ds.append(d)
                gs.append(float(gross.mean())); fs.append(float(fin.sum() / (len(r) / 252)))
                cl.append(float((gross_want > REGT).mean()))
            n = len(starts)
            print(f"{label:<22}{np.mean(cs):>+8.1%}{np.mean(vls):>7.1%}{np.mean(ss):>8.2f}"
                  f"{np.mean(ds):>+8.1%}{np.mean(gs):>9.2f}{np.mean(fs):>7.2%}{np.mean(cl):>7.1%}", flush=True)
        del bt
    print("\nclamp% = share of days the policy WANTED >2.0x gross (infeasible; clamped).", flush=True)


if __name__ == "__main__":
    main()
