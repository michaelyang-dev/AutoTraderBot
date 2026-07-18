"""
THREAD S — two theory-backed, previously-UNTESTED momentum risk controls:
  (1) SECTOR-concentration cap: max K of the top-5 momentum names per 2-digit SIC group
      (momentum crashes are usually concentrated sector unwinds; the book had NO sector control).
  (2) VOL-MANAGED momentum (Barroso/Daniel-Moskowitz): scale the momentum SLEEVE by
      target/own-realized-vol — distinct from portfolio vol-scaling; targets momentum crashes.
Full live-mirror stack (1.49x integer $50k), both periods, multi-start. Honest bar: beat
baseline on Sharpe/DD in BOTH periods (or a clean DD-first trade).
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True, "financing_rate": 0.063}
PERIODS = [
    ("SHORT 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("LONG 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
VARIANTS = [
    ("BASELINE", {}),
    ("sector cap 2/SIC (pool2)", dict(sector_cap=2, mom_pool=2.0)),
    ("sector cap 1/SIC (pool3)", dict(sector_cap=1, mom_pool=3.0)),
    ("vol-managed mom t0.25", dict(vol_managed_mom=0.25)),
    ("vol-managed mom t0.35", dict(vol_managed_mom=0.35)),
    ("vol-managed mom t0.35 +sector cap2", dict(vol_managed_mom=0.35, sector_cap=2, mom_pool=2.0)),
]


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stat(v):
    dr = v.pct_change().dropna(); yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1, dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
            ((v - v.cummax()) / v.cummax()).min())


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 92, flush=True)
        print(f"{pname} | 1.49x integer $50k full stack | {len(starts)}-start", flush=True)
        print("=" * 92, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<38}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dSharpe':>9}"
        print(hdr); print("-" * len(hdr), flush=True)
        base = None
        for name, extra in VARIANTS:
            cs, ss, ds = [], [], []
            for st in starts:
                cfg = dict(BASE); cfg.update(extra)
                m = bt.run(st, end, cfg); c, s, d = stat(m["daily_values"])
                cs.append(c); ss.append(s); ds.append(d)
            c, s, d = np.mean(cs), np.mean(ss), np.mean(ds)
            if base is None:
                base = (c, s, d)
            dc = "" if name == "BASELINE" else f"{(c-base[0])*100:>+7.1f}p"
            dsh = "" if name == "BASELINE" else f"{(s-base[1]):>+8.2f}"
            print(f"{name:<38}{c:>+8.1%}{s:>8.2f}{d:>+8.1%}{dc:>8}{dsh:>9}", flush=True)
        del bt


if __name__ == "__main__":
    main()
