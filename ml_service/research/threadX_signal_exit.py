"""
THREAD X — user idea (2026-07-22): sell a holding MID-CYCLE when it drops out of the
current signals, instead of waiting out the fixed 20-day cadence.

Motivation: on 7/22 the day's big losers (NOW/DOCU/BOX) were exactly the names already
rotated out of the live signals but still held by the 20d cadence, while the current picks
were green. Does exiting them early beat holding to the rebalance?

Prior evidence AGAINST (to be honestly re-tested, not assumed): C1 diagnostic — held names
whose momentum rolled over OUTPERFORM forward (hold-the-dip, +2.1/+1.1pp fwd20 both
periods); Different-Jobs exit threads null; 20d uniform cadence beat 10d. But an exit-only-
on-signal-drop rule (asymmetric cadence: exits fast, entries every 20d) was never tested
in THIS exact form. Cash from exits WAITS for the rebalance (recycle tested-dead).

Variants (full live-mirror stack, 1.49x integer $50k + financing + vol-scaling):
  strict/5d  — sell if absent from ALL sleeve targets (mom top-5 ∪ val-10 ∪ s3 ∪ s5), check every 5d
  grace15/5d — tolerate names still in the mom top-15 (user's "out by however many ranks")
  strict/1d  — strict checked daily (8yr only; the most aggressive version)
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
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
VARIANTS = [
    ("BASELINE hold-to-rebalance", {}, False),
    ("signal-exit STRICT /5d", dict(signal_exit_every=5), False),
    ("signal-exit grace mom15 /5d", dict(signal_exit_every=5, signal_exit_grace=15), False),
    ("signal-exit STRICT /daily", dict(signal_exit_every=1), True),   # 8yr only
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
        print("\n" + "=" * 96, flush=True)
        print(f"{pname} | full live-mirror 1.49x integer $50k | {len(starts)}-start", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<32}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dSharpe':>9}{'exits/yr':>9}"
        print(hdr); print("-" * len(hdr), flush=True)
        base = None
        for name, extra, only8 in VARIANTS:
            if only8 and pname.startswith("26yr"):
                continue
            cs, ss, ds, xs = [], [], [], []
            for st in starts:
                cfg = dict(BASE); cfg.update(extra)
                m = bt.run(st, end, cfg)
                c, s, d = stat(m["daily_values"])
                yrs = max((m["daily_values"].index[-1] - m["daily_values"].index[0]).days / 365.25, 1)
                cs.append(c); ss.append(s); ds.append(d); xs.append(bt._sigexit_count / yrs)
            c, s, d, x = np.mean(cs), np.mean(ss), np.mean(ds), np.mean(xs)
            if base is None:
                base = (c, s, d)
            dc = "" if name.startswith("BASELINE") else f"{(c-base[0])*100:>+7.1f}p"
            dsh = "" if name.startswith("BASELINE") else f"{(s-base[1]):>+8.2f}"
            print(f"{name:<32}{c:>+8.1%}{s:>8.2f}{d:>+8.1%}{dc:>8}{dsh:>9}{x:>9.1f}", flush=True)
        del bt
    print("\nHONEST BAR: beats baseline on Sharpe (or clean DD-first) in BOTH periods, same params.", flush=True)


if __name__ == "__main__":
    main()
