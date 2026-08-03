"""Break-even sweep: extra one-way cost charged ONLY on non-SP500 (mid/small-cap) legs
of the fixed SP1500 book. How big must it be to erase the advantage over the SP500-only book?
Also: matched-size cost comparison (what a $3,000 trade costs in each book's names)."""
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "1"
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from costreal_cache import SP, ensure_cache  # noqa: E402
ensure_cache()
import numpy as np, pandas as pd  # noqa
from costreal_fork import CostedBacktester, clear_deployed, stats_of  # noqa

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
       "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40}
CAP0, RATE = 53_000.0, 0.063
STARTS = ["2018-01-02", "2018-01-17", "2018-02-01"]
END = "2025-12-31"

t0 = time.time()
bt = CostedBacktester(universe_path="data/wrds/complete_sp1500_universe.pkl")
clear_deployed(bt)
bt.load_liquidity()
print(f"loaded {time.time()-t0:.0f}s", flush=True)


def go(pool_kw, **ck):
    cs, ss, ds, fs, trs = [], [], [], [], []
    for st in STARTS:
        cfg = dict(V12); cfg.update(pool_kw)
        cfg.update(dict(leverage=1.49, integer_shares=True, financing_rate=RATE,
                        initial_capital=CAP0))
        m = bt.run_cost(st, END, cfg, **ck)
        c, v, s, d = stats_of(m["daily_values"])
        cs.append(c); ss.append(s); ds.append(d); fs.append(m["final"]); trs.append(m["trades"])
    tr = pd.concat(trs)
    return (np.mean(cs), np.mean(ss), np.mean(ds), np.mean(fs),
            (tr.cf * tr.dollars).sum() / tr.dollars.sum() * 1e4, tr)


print("\nreference: SP500-only (bug) book under the REALISTIC cost model")
ref = go({"mom_pool_sp500": True}, cost_mode="real", spread_mult=1.0)
print(f"  CAGR {ref[0]:+.1%}  Sharpe {ref[1]:.2f}  DD {ref[2]:+.1%}  end ${ref[3]:,.0f}  "
      f"cost {ref[4]:.1f}bps/leg", flush=True)

print("\nSP1500 book with an EXTRA one-way penalty charged only on non-SP500 legs:")
print(f"{'extra bps/leg (mid+small only)':<34}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
      f"{'end $':>12}{'avg cost/leg':>14}{'vs SP500-only':>15}")
out = []
for x in [0, 25, 50, 100, 150, 200, 300, 500]:
    r = go({}, cost_mode="real", spread_mult=1.0, smallcap_extra_bps=float(x))
    out.append((x,) + r[:5])
    print(f"{x:<34}{r[0]:>+9.1%}{r[1]:>8.2f}{r[2]:>+9.1%}{r[3]:>12,.0f}{r[4]:>14.1f}"
          f"{r[0]-ref[0]:>+15.1%}", flush=True)
pickle.dump({"ref": ref[:5], "sweep": out}, open(SP + "sweep.pkl", "wb"))

# ---- matched-size cost: price a $3,000 trade in each book's actual names ----
print("\nMATCHED-SIZE check — cost of a $3,000 trade in the names each book actually holds:")
for lbl, pkw in [("SP1500 (fixed)", {}), ("SP500-only (bug)", {"mom_pool_sp500": True})]:
    _, _, _, _, _, tr = go(pkw, cost_mode="real", spread_mult=1.0)
    bt.cost_mode = "real"; bt.spread_mult = 1.0; bt.extra_bps = 0.0
    bt.smallcap_extra_bps = 0.0
    bt.fallback_spread_bps = 25.0; bt.fallback_adv = 3e6
    bt._trades = []
    u = tr.drop_duplicates(["date", "sym"])
    cfs = [bt._cf(s, d, 3000.0) for s, d in zip(u["sym"], u["date"])]
    cfs = np.array(cfs) * 1e4
    print(f"  {lbl:<20} n={len(cfs):>6}  mean {cfs.mean():>5.2f}bps  median {np.median(cfs):>5.2f}"
          f"  p90 {np.percentile(cfs,90):>5.2f}  p99 {np.percentile(cfs,99):>6.2f}  max {cfs.max():>6.2f}")
