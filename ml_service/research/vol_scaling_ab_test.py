"""
VOL-SCALING A/B — does it actually help CAGR/DD, and is the LIVE calibration right?

Policies tested (fork parameterizes the up-scale cap as config["vol_scale_cap"]):
  A. OFF                      — no vol scaling (pure v12)
  B. tgt0.20 cap1.5           — the backtest's validated "honest bar" config
  C. tgt0.20 cap1.0           — same target, de-risk-only (no up-lever)
  D. tgt0.15 cap1.0           — THE LIVE ENGINE'S POLICY (VOL_TARGET_1X=0.15, min(1.0,...))
  E. tgt0.15 cap1.5           — live target, backtest cap

Periods: 2018-2025 (3 starts) on complete_sp1500_universe.pkl and 2001-2025 (2 starts)
on sp1500_universe_2000.pkl (the through-cycle test: 2008 -54% DD is what vol-scaling
is supposed to fix). Deployed data condition (enhanced OFF, SI OFF). 1x throughout —
live leverage multiplies everything ~1.49x proportionally.

Run: OMP_NUM_THREADS=1 python3 research/vol_scaling_ab_test.py
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, "research"))

SRC = os.path.join(ML, "main_production_backtest.py")
FORK = os.path.join(ML, "research", "_volscale_fork.py")
ORIG = "                    vol_scale = min(1.5, max(0.3, vol_target / realized_vol))"
PATCH = "                    vol_scale = min(config.get(\"vol_scale_cap\", 1.5), max(0.3, vol_target / realized_vol))"
src = open(SRC).read()
assert src.count(ORIG) == 1, "vol_scale anchor not unique — backtest changed"
open(FORK, "w").write(src.replace(ORIG, PATCH))
print("fork written")

from _volscale_fork import FastBacktester  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}

POLICIES = [
    ("A OFF",            {}),
    ("B t.20 cap1.5",    {"vol_scaling": True, "vol_target": 0.20, "vol_lookback": 40, "vol_scale_cap": 1.5}),
    ("C t.20 cap1.0",    {"vol_scaling": True, "vol_target": 0.20, "vol_lookback": 40, "vol_scale_cap": 1.0}),
    ("D t.15 cap1.0 LIVE", {"vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0}),
    ("E t.15 cap1.5",    {"vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.5}),
]

PERIODS = [
    ("8yr 2018-2025", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-2025", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def main():
    for pname, path, starts, end in PERIODS:
        print("=" * 96, flush=True)
        print(f"PERIOD {pname} | starts {starts} | deployed condition | 1x", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(universe loaded in {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'policy':<20}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}   worst-yr | 2008/2020/2022"
        print(hdr); print("-" * len(hdr), flush=True)
        for label, flags in POLICIES:
            cs, vs, ss, ds, wys, keyyrs = [], [], [], [], [], []
            for st in starts:
                cfg = dict(V12); cfg.update(flags)
                m = bt.run(st, end, cfg)
                cs.append(m["cagr"]); ss.append(m["sharpe"]); ds.append(m["max_dd"])
                dv = m["daily_values"].pct_change().dropna()
                vs.append(dv.std() * (252 ** 0.5))
                yr = m.get("yearly", {})
                if yr:
                    wys.append(min(y["cagr"] for y in yr.values()))
                    keyyrs.append({y: yr[y]["cagr"] for y in yr if y in (2008, 2020, 2022)})
            n = len(starts)
            wy = sum(wys) / n if wys else float("nan")
            key = keyyrs[0] if keyyrs else {}
            keystr = "/".join(f"{key.get(y, float('nan'))*100:+.0f}%" for y in (2008, 2020, 2022) if y in key) or "-"
            print(f"{label:<20}{sum(cs)/n:>+8.1%}{sum(vs)/n:>7.1%}{sum(ss)/n:>8.2f}{sum(ds)/n:>+8.1%}"
                  f"   {wy:>+7.1%} | {keystr}", flush=True)
        del bt

    print("\nHONEST READ GUIDE: policy D is what runs live. If D's Sharpe/DD do not beat A", flush=True)
    print("meaningfully, the live calibration is wrong even if some vol-scaling variant helps.", flush=True)


if __name__ == "__main__":
    main()
