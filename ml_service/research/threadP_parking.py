"""
THREAD P — DE-RISK TREASURY PARKING (+ stop-removal cross-check).

Genuinely untested cross-asset idea (verified: no stock/bond allocation test exists in the
repo; phase6 '60/40' was mom/val ratio, phase4b trend sleeve was TSMOM not bonds):
when the leverage target < 1.0x NAV (vol-scaling de-levered and/or credit gate ON), park the
idle fraction in a treasury ETF (total-return series incl. distributions) instead of
0%-yield cash. Treasuries rally in exactly those flight-to-quality windows (TLT 2008-window
+20.6%, COVID +15.9%) — landmine pre-registered: 2022 (TLT −32.3%, IEF −16.1%, SHY −4.5%).
SHY-parking also doubles as the honest model of live reality (IBKR pays ~short-rate on cash).

Bonus arm: trailing_stop=None in the full stack — phase0_stop_attribution said the stop
costs −1.5pp CAGR / +2.3pp DD, and threadC's flat-lever sweep showed NONE beat 40% on 26yr;
never tested with vol-scaling+gate active.

Full live-mirror (1.49x integer $50k + financing + vol-scaling), both periods, multi-start.
Note: bond ETFs exist 2002-07+; before that parking stays cash (honest, disclosed).
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
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063, "credit_pct": 0.95, "credit_derisk": 0.5}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
VARIANTS = [
    ("BASELINE (gate, cash 0%)", {}),
    ("park SHY (cash-yield model)", dict(park_etf="SHY")),
    ("park IEF (7-10y flight-to-quality)", dict(park_etf="IEF")),
    ("park TLT (20y max convexity)", dict(park_etf="TLT")),
    ("no trailing stop (stop=None)", dict(trailing_stop=None)),
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
        print("\n" + "=" * 98, flush=True)
        print(f"{pname} | full live-mirror 1.49x integer $50k + credit gate | {len(starts)}-start", flush=True)
        print("=" * 98, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<38}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dSharpe':>9}{'dDD':>8}"
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
                base = (c, s, d)
            dc = "" if name.startswith("BASELINE") else f"{(c-base[0])*100:>+7.1f}p"
            dsh = "" if name.startswith("BASELINE") else f"{(s-base[1]):>+8.2f}"
            dd = "" if name.startswith("BASELINE") else f"{(d-base[2])*100:>+7.1f}p"
            print(f"{name:<38}{c:>+8.1%}{s:>8.2f}{d:>+8.1%}{dc:>8}{dsh:>9}{dd:>8}", flush=True)
        del bt
    print("\nBAR: both periods, dSharpe>=0 with dCAGR>=0 (or clean DD-first). dDD>0 = shallower.", flush=True)


if __name__ == "__main__":
    main()
