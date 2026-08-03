"""SP1500 vs SP500-pool at $53k under FLAT-10bps vs REALISTIC per-name costs."""
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

POOLS = [("SP1500 (fixed)", {}), ("SP500-only (bug)", {"mom_pool_sp500": True})]
COSTS = [
    ("A flat 10bps/leg (backtest default)", dict(cost_mode="flat")),
    ("B realistic: half-quoted-spread+impact+comm", dict(cost_mode="real", spread_mult=1.0)),
    ("C pessimistic: FULL quoted spread+imp+comm", dict(cost_mode="real", spread_mult=2.0)),
    ("D paranoid: 2x spread + 10bps extra/leg", dict(cost_mode="real", spread_mult=2.0, extra_bps=10.0)),
]

t0 = time.time()
bt = CostedBacktester(universe_path="data/wrds/complete_sp1500_universe.pkl")
clear_deployed(bt)
bt.load_liquidity()
print(f"loaded {time.time()-t0:.0f}s", flush=True)

rows = []
trade_store = {}
for cname, ckw in COSTS:
    for pname, pkw in POOLS:
        cs, ss, ds, vs, fs = [], [], [], [], []
        trs = []
        for st in STARTS:
            cfg = dict(V12); cfg.update(pkw)
            cfg.update(dict(leverage=1.49, integer_shares=True, financing_rate=RATE,
                            initial_capital=CAP0))
            tt = time.time()
            m = bt.run_cost(st, END, cfg, **ckw)
            c, v, s, d = stats_of(m["daily_values"])
            cs.append(c); vs.append(v); ss.append(s); ds.append(d); fs.append(m["final"])
            trs.append(m["trades"])
        tr = pd.concat(trs)
        wavg_bps = (tr.cf * tr.dollars).sum() / tr.dollars.sum() * 1e4
        rows.append(dict(cost=cname, pool=pname, cagr=np.mean(cs), sharpe=np.mean(ss),
                         dd=np.mean(ds), final=np.mean(fs), legs=len(tr) / len(STARTS),
                         traded=tr.dollars.sum() / len(STARTS), wcost_bps=wavg_bps))
        trade_store[(cname, pname)] = tr
        print(f"{cname:<44}{pname:<20}CAGR{np.mean(cs):>+7.1%} Sh{np.mean(ss):>6.2f} "
              f"DD{np.mean(ds):>+7.1%} end${np.mean(fs):>10,.0f} cost/leg{wavg_bps:>6.1f}bps "
              f"({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(rows)
R.to_csv(SP + "ab_results.csv", index=False)
pickle.dump({k: v for k, v in trade_store.items()}, open(SP + "trades.pkl", "wb"))

print("\n" + "=" * 100)
print(f"{'cost model':<44}{'SP1500':>10}{'SP500':>10}{'Δ CAGR':>10}{'Δ Sharpe':>10}")
for cname, _ in COSTS:
    a = R[(R.cost == cname) & (R.pool == "SP1500 (fixed)")].iloc[0]
    b = R[(R.cost == cname) & (R.pool == "SP500-only (bug)")].iloc[0]
    print(f"{cname:<44}{a.cagr:>+10.1%}{b.cagr:>+10.1%}{a.cagr-b.cagr:>+10.1%}"
          f"{a.sharpe-b.sharpe:>+10.2f}")
