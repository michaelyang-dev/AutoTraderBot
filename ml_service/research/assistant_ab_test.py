"""
THE TRUE ASSISTANT A/B — with vs without, using the assistant's REAL extracted values.

Arms (identical everything except how the roe feature is served):
  CEILING   — backtest's own PIT-fresh roe (ideal, for reference)
  ARM A     — "life without the assistant" = LIVE TODAY: all three Compustat
              fundamentals (roe, gross_margin, debt_to_equity) frozen to quarterly
              manual uploads (grid Feb/May/Aug/Nov 15; values held until next upload)
  ARM B     — "life with the assistant" = LIVE AFTER THE FLIP: identical to ARM A
              (same upload-stale gm/d2e, same base), except roe for GATE-COVERED
              symbols is overridden by the EDGAR-extracted value (real specs, real
              filed dates) available the day after filing. Uncovered symbols keep
              ARM A staleness — mirrors production coverage exactly.

Conservative choice (disclosed): arm B serves the extraction even for quarters the
upload already covers (production would serve exact Compustat there via the vintage
guard) — so arm B carries the FULL 1.1% extraction-error burden. Bias runs AGAINST
the assistant; the real flip can only do better.
8yr, 2 starts, live config, 1x. Run AFTER build_edgar_pit_roe.py + scp of the parquet.
"""
import bisect
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
import pandas as pd
from main_production_backtest import FastBacktester
from live_config import V12_LIVE_BACKTEST_CONFIG

STARTS = ["2018-01-02", "2018-01-17"]
END = "2025-12-31"
UPLOADS = pd.date_range("2015-02-15", "2026-05-15", freq="3MS") + pd.Timedelta(days=14)


def main():
    pit = pd.read_parquet(os.path.join(ML, "data/edgar_pit_roe.parquet"))
    pit = pit.sort_values("filed")
    pit_map = {s: (list(g.filed), list(g.roe)) for s, g in pit.groupby("symbol")}
    print(f"assistant PIT coverage: {len(pit_map)} symbols, {len(pit)} quarter-values")

    bt = FastBacktester()
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
    orig = bt.uni.get_feature_map
    uploads = sorted(UPLOADS)

    def last_upload(date):
        i = bisect.bisect_right(uploads, pd.Timestamp(date)) - 1
        return uploads[max(i, 0)]

    STALE_FEATS = {"roe", "gross_margin", "debt_to_equity"}   # the upload-fed features

    def arm_a(date, feature, members=None):
        if feature not in STALE_FEATS:
            return orig(date, feature, members)
        u = last_upload(date)
        return orig(min(pd.Timestamp(date), u), feature, members) if u <= pd.Timestamp(date) \
            else orig(date, feature, members)

    def arm_b(date, feature, members=None):
        if feature != "roe":
            return arm_a(date, feature, members)   # SAME stale base as arm A (gm/d2e)
        base = arm_a(date, feature, members)          # stale fallback for uncovered
        cutoff = pd.Timestamp(date) - pd.Timedelta(days=1)
        syms = members if members is not None else list(base.keys())
        for s in syms:
            e = pit_map.get(s)
            if not e:
                continue
            i = bisect.bisect_right(e[0], cutoff) - 1
            if i >= 0:
                base[s] = e[1][i]
        return base

    cfg = dict(V12_LIVE_BACKTEST_CONFIG)
    for k in ("vol_scaling", "vol_target", "vol_lookback"):
        cfg.pop(k, None)

    print(f"{'arm':<28}{'start':<13}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}", flush=True)
    for label, fn in [("CEILING (ideal fresh)", orig),
                      ("A: NO assistant (uploads)", arm_a),
                      ("B: WITH assistant", arm_b)]:
        bt.uni.get_feature_map = fn
        for st in STARTS:
            m = bt.run(st, END, cfg)
            print(f"{label:<28}{st:<13}{m['cagr']:>+8.1%}{m['sharpe']:>8.2f}{m['max_dd']:>+8.1%}", flush=True)
    bt.uni.get_feature_map = orig


if __name__ == "__main__":
    main()
