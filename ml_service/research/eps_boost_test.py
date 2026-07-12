"""
EPS-BOOST A/B — divergence #61: live applies a x1.15 momentum boost on positive last
earnings surprise (eps_surprise_last from IBES); the backtest feature store never had
this populated, so NO canonical number includes the boost. This test builds the feature
POINT-IN-TIME from IBES history (surprise known only after its announcement date) and
measures the boost's actual effect. Decides: remove from live (parity) vs canonize.

BOOST OFF = canonical config (empty map, reproduces known numbers)
BOOST ON  = live behavior (PIT surprise served to the momentum sleeve)

8yr 2018-2025 (IBES history starts 2014), 3 starts, deployed condition, 1x.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
import bisect
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester
from live_config import V12_LIVE_BACKTEST_CONFIG

STARTS = ["2018-01-02", "2018-01-17", "2018-02-01"]
END = "2025-12-31"


def build_pit_surprise():
    ib = pd.read_parquet(os.path.join(ML, "data/wrds/ibes_summary_latest.parquet"))
    ib = ib[ib.MEASURE == "EPS"].dropna(subset=["ACTUAL", "MEANEST", "ANNDATS_ACT"])
    ib["ANNDATS_ACT"] = pd.to_datetime(ib["ANNDATS_ACT"])
    ib["surp"] = (ib.ACTUAL - ib.MEANEST) / ib.MEANEST.abs().replace(0, np.nan)
    ib = ib.dropna(subset=["surp"]).sort_values("ANNDATS_ACT")
    out = {}
    for sym, g in ib.groupby("OFTIC"):
        out[sym] = (list(g.ANNDATS_ACT), list(g.surp))
    return out


def main():
    pit = build_pit_surprise()
    print(f"PIT surprise built: {len(pit)} tickers", flush=True)
    bt = FastBacktester()
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
    orig = bt.uni.get_feature_map

    def pit_map(date, feature, members=None):
        if feature != "eps_surprise_last":
            return orig(date, feature, members)
        cutoff = pd.Timestamp(date) - pd.Timedelta(days=1)   # known strictly before signal date
        res = {}
        syms = members if members is not None else pit.keys()
        for s in syms:
            e = pit.get(s)
            if not e:
                continue
            i = bisect.bisect_right(e[0], cutoff) - 1
            if i >= 0:
                res[s] = e[1][i]
        return res

    cfg = dict(V12_LIVE_BACKTEST_CONFIG)
    for k in ("vol_scaling", "vol_target", "vol_lookback"):
        cfg.pop(k, None)

    print(f"{'variant':<22}{'start':<13}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}", flush=True)
    for label, fn in [("BOOST OFF (canonical)", orig), ("BOOST ON (live now)", pit_map)]:
        bt.uni.get_feature_map = fn
        for st in STARTS:
            m = bt.run(st, END, cfg)
            print(f"{label:<22}{st:<13}{m['cagr']:>+8.1%}{m['sharpe']:>8.2f}{m['max_dd']:>+8.1%}", flush=True)
    bt.uni.get_feature_map = orig


if __name__ == "__main__":
    main()
