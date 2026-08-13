"""thread LEV2 — is live's over-investment harmful because of the MECHANISM or just the LEVEL?

threadLEV showed live sizing (renormalise weights to sum 1, then closed-loop to the full
target) realises ~1.49x gross vs the backtest's ~1.12x, and that the extra exposure bought
+0.01pp CAGR over 26yr while adding 11.25pp of drawdown.

But that comparison CONFLATES two different things:
  (a) the MECHANISM — renormalising away under-allocation, and recovering rounding drag
  (b) the LEVEL     — the realised gross that results (1.49x vs 1.12x)

They imply completely different fixes. If the mechanism is neutral and only the level matters,
the correct change is a ONE-CONSTANT edit (lower EFFECTIVE_LEVERAGE) — trivially safe and
reversible. If the mechanism itself hurts at matched exposure, live's sizing routine has to be
rewritten, which is a far riskier change to order sizing.

Arms (all live-mirror, $50k, integer shares, 6.3% financing, vol-scaling, credit gate off):
  A  backtest sizing @ nominal 1.49  -> ~1.12x realised   (the validated baseline)
  B  live sizing     @ nominal 1.49  -> ~1.49x realised   (what the account actually does)
  C  live sizing     @ nominal 1.12  -> ~1.12x realised   (SAME exposure as A, other mechanism)

A vs C is the decisive comparison: matched realised gross, mechanism the only difference.

Then a LEVERAGE SWEEP under live sizing, to find what level the strategy actually wants —
because if we are going to pick a constant, pick it on evidence rather than inheritance.

Run:  python3 research/threadLEV2_mechanism_vs_level.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "integer_shares": True, "financing_rate": 0.063}

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]

MECHANISM_ARMS = [
    ("A backtest sizing @1.49",   {"leverage": 1.49}),
    ("B live sizing     @1.49",   {"leverage": 1.49, "live_sizing": True}),
    ("C live sizing     @1.12",   {"leverage": 1.12, "live_sizing": True}),
]

SWEEP = [0.90, 1.00, 1.12, 1.25, 1.35, 1.49]


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}
    bt._si_change_ranks_by_month = {}
    bt._si_months = []
    bt.uni._short_interest_rank = {}
    bt.uni._si_change_rank = {}


def stat(v):
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
            dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
            ((v - v.cummax()) / v.cummax()).min())


def avg_over_starts(bt, starts, end, cfg):
    cs, ss, ds, gs = [], [], [], []
    for st in starts:
        m = bt.run(st, end, dict(cfg))
        c, s, d = stat(m["daily_values"])
        cs.append(c); ss.append(s); ds.append(d); gs.append(m["avg_gross"])
    return (float(np.mean(cs)), float(np.mean(ss)),
            float(np.mean(ds)), float(np.mean(gs)))


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 96, flush=True)
        print(f"{pname} | MECHANISM vs LEVEL | {len(starts)}-start", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        hdr = f"{'arm':<28}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'avgGross':>10}"
        print(hdr); print("-" * len(hdr), flush=True)
        res = {}
        for name, extra in MECHANISM_ARMS:
            cfg = dict(BASE); cfg.update(extra)
            r = avg_over_starts(bt, starts, end, cfg)
            res[name] = r
            print(f"{name:<28}{r[0]:>+9.2%}{r[1]:>8.2f}{r[2]:>+9.2%}{r[3]:>10.4f}", flush=True)

        a = res[MECHANISM_ARMS[0][0]]
        c = res[MECHANISM_ARMS[2][0]]
        print(f"\n  >>> DECISIVE (A vs C — matched exposure, mechanism differs):", flush=True)
        print(f"      realised gross  A {a[3]:.4f}  vs  C {c[3]:.4f}", flush=True)
        print(f"      CAGR   {(c[0]-a[0])*100:+.2f}pp | Sharpe {c[1]-a[1]:+.3f} | "
              f"MaxDD {(c[2]-a[2])*100:+.2f}pp", flush=True)
        print(f"      -> if these are within noise, the MECHANISM is neutral and only the", flush=True)
        print(f"         LEVEL matters (fix = lower one constant, not rewrite sizing).", flush=True)

        print(f"\n  LEVERAGE SWEEP under LIVE sizing:", flush=True)
        hdr2 = f"{'nominal':<10}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'avgGross':>10}{'CAGR/DD':>9}"
        print("  " + hdr2); print("  " + "-" * len(hdr2), flush=True)
        for lev in SWEEP:
            cfg = dict(BASE); cfg.update({"leverage": lev, "live_sizing": True})
            r = avg_over_starts(bt, starts, end, cfg)
            ratio = r[0] / abs(r[2]) if r[2] else 0
            print(f"  {lev:<10.2f}{r[0]:>+9.2%}{r[1]:>8.2f}{r[2]:>+9.2%}{r[3]:>10.4f}{ratio:>9.3f}",
                  flush=True)
        del bt

    print("\n" + "=" * 96, flush=True)
    print("Read Sharpe and CAGR/DD, not CAGR: raising leverage raises CAGR almost by", flush=True)
    print("construction, so a bare CAGR comparison always flatters more leverage.", flush=True)


if __name__ == "__main__":
    main()
