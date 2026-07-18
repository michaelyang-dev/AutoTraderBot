"""
THREAD R — richer regimes (not just bull/bear leverage) + a bigger QUALITY sleeve.

Two ideas the user pushed:
  (1) a STRONG-BULL state that tilts HARDER into momentum (mom 0.70) when breadth is high +
      SPY uptrend (+ optional credit-calm), instead of the flat base 0.50 mom. Optionally also
      levers up in that state.
  (2) more QUALITY: raise the lowvol/quality sleeve (s5) weight — does quality improve
      risk-adjusted return / drawdown, or does it just drag momentum (prior: factors HURT mom)?

Full live-mirror stack: $50k, integer shares, 1.49x + fin, vol-scaling + 40% stops + 20d rebal.
Both periods, multi-start. Honest bar: beat baseline on Sharpe/DD in BOTH periods, or a clean
CAGR-first / DD-first trade.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASEW = {"mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0}
BASE = {"universe": "sp1500", "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10,
        "use_rp": False, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True, "financing_rate": 0.063}
PERIODS = [
    ("SHORT 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("LONG 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
BULL_MOM = {"mom": 0.70, "val": 0.20, "s5": 0.10, "s3": 0.0}

VARIANTS = [
    ("BASELINE 50/35/15", dict(BASEW)),
    ("strong-bull mom0.70 (breadth+trend)", dict(BASEW, bull_weights=BULL_MOM, bull_breadth=0.60, bull_need_trend=True)),
    ("strong-bull mom0.70 + credit-calm", dict(BASEW, bull_weights=BULL_MOM, bull_breadth=0.60, bull_need_trend=True, bull_credit_calm=0.30)),
    ("strong-bull mom0.70 + lever1.3", dict(BASEW, bull_weights=BULL_MOM, bull_breadth=0.60, bull_need_trend=True, bull_credit_calm=0.30, bull_lever=1.3)),
    ("more-quality 40/35/25", dict(mom_w=0.40, val_w=0.35, lv_w=0.25, sec_w=0.0)),
    ("more-quality 45/30/25", dict(mom_w=0.45, val_w=0.30, lv_w=0.25, sec_w=0.0)),
    ("more-momentum 60/25/15", dict(mom_w=0.60, val_w=0.25, lv_w=0.15, sec_w=0.0)),
    ("bull-tilt + credit down-gate", dict(BASEW, bull_weights=BULL_MOM, bull_breadth=0.60, bull_need_trend=True, bull_credit_calm=0.30, credit_pct=0.95, credit_derisk=0.5)),
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
        print("\n" + "=" * 96, flush=True)
        print(f"{pname} | 1.49x integer $50k full stack | {len(starts)}-start", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<40}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dSharpe':>9}"
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
            dc = "" if name.startswith("BASELINE") else f"{(c-base[0])*100:>+7.1f}p"
            dsh = "" if name.startswith("BASELINE") else f"{(s-base[1]):>+8.2f}"
            print(f"{name:<40}{c:>+8.1%}{s:>8.2f}{d:>+8.1%}{dc:>8}{dsh:>9}", flush=True)
        del bt
    print("\nStrong-bull = momentum tilt only in high-breadth+uptrend(+calm). Quality = raise s5.", flush=True)


if __name__ == "__main__":
    main()
