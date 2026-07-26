"""
THREAD S3 — sector-sleeve live-parity A/B (found 2026-07-26, ACTIVE divergence).

BUG: in bear/crash regimes the sector sleeve (s3) gets 10% of the book and returns sector
ETFs (XLK/XLE/XLI/XLRE). The BACKTEST buys them (they are in its price matrix and enter
`combined`). LIVE DROPS them: signal_builder emits only SP1500 *members*, and ETFs are not
members -> zero XL* names ever reach the engine. The engine's closed-loop sizing then
redistributes that 10% across the remaining stocks.
=> live runs MORE single-stock concentrated than the validated backtest, in exactly the
defensive regime where the sector diversification was designed to help.
Only bites in bear/crash. The UMD crash detector is FIRING as of 2026-07-26 (20d UMD
-0.079 < -0.05 threshold), so this is live TODAY.

ARMS (identical except the s3 treatment):
  A "backtest (holds sector ETFs)"  bear s3=.10, crash s3=.10   <- what was validated
  B "live replica (ETFs dropped)"   s3=0 in bear AND crash, remaining sleeves renormalized
                                     (bear .10/.30/.50 -> .111/.333/.556;
                                      crash .15/.45/.30 -> .1667/.500/.3333)
Full live-mirror stack (1.49x, integer $50k, financing, vol_scale_cap 1.0, credit gate ON),
both periods, multi-start.

BAR: only change live if a variant wins on BOTH periods. If A > B -> emit ETFs live (restore
parity). If B >= A -> live behavior is fine/better; document and close the divergence.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063, "credit_pct": 0.95, "credit_derisk": 0.5}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
BEAR_A = {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}          # validated backtest
BEAR_B = {"mom": 0.1111, "val": 0.3333, "s5": 0.5556, "s3": 0.0}     # live: s3 dropped, renorm
CRASH_A = {"mom": 0.15, "val": 0.45, "s5": 0.30, "s3": 0.10}
CRASH_B = {"mom": 0.1667, "val": 0.5000, "s5": 0.3333, "s3": 0.0}

VARIANTS = [
    ("A backtest — holds sector ETFs", dict(bear_weights=BEAR_A, crash_weights=CRASH_A)),
    ("B live replica — ETFs dropped", dict(bear_weights=BEAR_B, crash_weights=CRASH_B)),
]


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stat(v):
    dr = v.pct_change().dropna(); yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
            dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
            ((v - v.cummax()) / v.cummax()).min())


def main():
    out = {}
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 92, flush=True)
        print(f"{pname} | s3 sector-ETF parity A/B | live-mirror 1.49x + gate | {len(starts)}-start", flush=True)
        print("=" * 92, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<34}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dSharpe':>9}{'dDD':>8}"
        print(hdr); print("-" * len(hdr), flush=True)
        base = None
        for name, extra in VARIANTS:
            cs, ss, ds = [], [], []
            for st in starts:
                cfg = dict(BASE); cfg.update(extra)
                m = bt.run(st, end, cfg)
                c, s, d = stat(m["daily_values"])
                cs.append(c); ss.append(s); ds.append(d)
            c, s, d = np.mean(cs), np.mean(ss), np.mean(ds)
            if base is None:
                base = (c, s, d); dc = dsh = dd = ""
            else:
                dc = f"{(c-base[0])*100:>+7.1f}p"; dsh = f"{(s-base[1]):>+8.2f}"; dd = f"{(d-base[2])*100:>+7.1f}p"
            out[(pname, name)] = (c, s, d)
            print(f"{name:<34}{c:>+8.1%}{s:>8.2f}{d:>+8.1%}{dc:>8}{dsh:>9}{dd:>8}", flush=True)
        del bt
    print("\nVERDICT GUIDE: B(live) >= A(backtest) on BOTH periods -> live behavior is fine, close as", flush=True)
    print("documented-benign. A > B on BOTH -> emit sector ETFs live to restore parity.", flush=True)


if __name__ == "__main__":
    main()
