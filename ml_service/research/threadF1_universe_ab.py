"""
THREAD F1 — the audit's headline finding, quantified: live momentum+lowvol sleeves select
from SP500-ONLY membership (audit F1: live get_sp500() serves sp500_constituents.json, no
override in the live path), while every validated number selects from SP1500.

This run measures what that divergence is WORTH, in the full live-mirror at live parity
(vol_scale_cap=1.0 — also fixing audit F5: prior livemirror runs up-scaled to 1.5 in calm):
  1. SP1500 all sleeves, no gate      (validated config, corrected cap)
  2. SP1500 all sleeves, credit gate  (re-verifies the DEPLOYED gate delta at correct cap)
  3. SP500 mom/s5 + SP1500 value, no gate   (the audit's live-replica)
  4. SP500 mom/s5 + SP1500 value, gate      (what live is ACTUALLY running today)
  5. #2 + TLT de-risk parking          (corrected parking read)
Both periods, multi-start. The 1-vs-3 and 2-vs-4 deltas = the cost (or benefit) of F1.
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
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063}
GATE = dict(credit_pct=0.95, credit_derisk=0.5)
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
VARIANTS = [
    ("SP1500 (validated), no gate", {}),
    ("SP1500 + credit gate", dict(GATE)),
    ("SP500-pool replica, no gate", dict(mom_pool_sp500=True)),
    ("SP500-pool replica + gate (LIVE TODAY)", dict(mom_pool_sp500=True, **GATE)),
    ("SP1500 + gate + TLT parking", dict(park_etf="TLT", **GATE)),
]


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stat(v):
    dr = v.pct_change().dropna(); yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
            dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
            ((v - v.cummax()) / v.cummax()).min())


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 100, flush=True)
        print(f"{pname} | live-mirror 1.49x integer $50k, vol_scale_cap=1.0 (LIVE parity) | {len(starts)}-start", flush=True)
        print("=" * 100, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<42}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}"
        print(hdr); print("-" * len(hdr), flush=True)
        for name, extra in VARIANTS:
            cs, ss, ds = [], [], []
            for st in starts:
                cfg = dict(BASE); cfg.update(extra)
                m = bt.run(st, end, cfg)
                c, s, d = stat(m["daily_values"])
                cs.append(c); ss.append(s); ds.append(d)
            print(f"{name:<42}{np.mean(cs):>+8.1%}{np.mean(ss):>8.2f}{np.mean(ds):>+8.1%}", flush=True)
        del bt
    print("\nREAD: (1 vs 3) and (2 vs 4) = the F1 SP500-pool divergence cost. (1 vs 2) = corrected gate delta.", flush=True)


if __name__ == "__main__":
    main()
