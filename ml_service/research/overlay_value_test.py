"""
STALE-FUNDAMENTALS TEST — how much does the live system's quarterly-stale WRDS data cost?

Live loads Compustat/IBES from quarterly manual downloads, so fundamentals lag 0-13 weeks
(avg ~6). This test measures the WORST CASE: every fundamental feature (roe, gross_margin,
debt_to_equity, eps_surprise_last) served with a FULL 63-trading-day lag, price features
untouched. Baseline vs lagged, live config, 1x, deployed condition, 2 starts x 8yr.
If the worst case is small, the average real-world staleness cost is smaller still.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
from main_production_backtest import FastBacktester
from live_config import V12_LIVE_BACKTEST_CONFIG

FUND_KEYS = {"roe", "gross_margin", "debt_to_equity", "eps_surprise_last"}
LAG_TDAYS = 63
STARTS = ["2018-01-02", "2018-01-17"]
END = "2025-12-31"


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def main():
    bt = FastBacktester()
    clear_deployed(bt)
    dates = sorted(bt.uni._feat_by_date.keys())
    idx = {d: i for i, d in enumerate(dates)}
    orig = bt.uni.get_feature_map

    def lagged_map(date, feature, members=None):
        if feature in FUND_KEYS and date in idx:
            date = dates[max(0, idx[date] - LAG_TDAYS)]
        return orig(date, feature, members)

    cfg = dict(V12_LIVE_BACKTEST_CONFIG)
    cfg.pop("vol_scaling", None); cfg.pop("vol_target", None); cfg.pop("vol_lookback", None)

    def lagged_subset_map(date, feature, members=None):
        # overlay world: roe + eps stay FRESH (deployable set), gm/d2e stale
        if feature in {"gross_margin", "debt_to_equity"} and date in idx:
            date = dates[max(0, idx[date] - LAG_TDAYS)]
        return orig(date, feature, members)

    print(f"{'variant':<30}{'start':<13}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}", flush=True)
    for label, fn in [("ALL FRESH (ceiling)", orig),
                      (f"ALL STALE {LAG_TDAYS}td (no overlay)", lagged_map),
                      ("OVERLAY WORLD (roe+eps fresh)", lagged_subset_map)]:
        bt.uni.get_feature_map = fn
        for st in STARTS:
            m = bt.run(st, END, cfg)
            print(f"{label:<30}{st:<13}{m['cagr']:>+8.1%}{m['sharpe']:>8.2f}{m['max_dd']:>+8.1%}", flush=True)
    bt.uni.get_feature_map = orig


if __name__ == "__main__":
    main()
