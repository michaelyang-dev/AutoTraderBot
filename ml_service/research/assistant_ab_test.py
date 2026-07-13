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

ARM B implements the PRODUCTION VINTAGE GUARD exactly: the extraction is used ONLY
when its period_end is NEWER than the quarter available in the last upload; otherwise
the upload's exact Compustat value serves (v1 of this test skipped the guard and let
full-time extraction noise swamp the freshness gain — wash result, now corrected).
SUMMER arms model the ACTUAL current situation: uploads stop after May 15 (WRDS closed
until Sept) — the decision-relevant scenario for flipping now.
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
    pit_pe = {s: list(g.period_end) for s, g in pit.groupby("symbol")}
    print(f"assistant PIT coverage: {len(pit_map)} symbols, {len(pit)} quarter-values")
    # full three-feature PIT set (for the B++ "is 92-96%-accurate fresh gm/d2e worth it" arm)
    pitf_path = os.path.join(ML, "data/edgar_pit_features.parquet")
    pitf_map, pitf_pe = {}, {}
    if os.path.exists(pitf_path):
        pf = pd.read_parquet(pitf_path).sort_values("filed")
        for (feat, sym), g in pf.groupby(["feature", "symbol"]):
            pitf_map.setdefault(feat, {})[sym] = (list(g.filed), list(g.value))
            pitf_pe.setdefault(feat, {})[sym] = list(g.period_end)
        print("full-feature PIT coverage:",
              {f: len(v) for f, v in pitf_map.items()})
    # WALK-FORWARD set (per-year re-resolved specs — the drift-fair rebuild)
    pitwf_path = os.path.join(ML, "data/edgar_walkforward_features.parquet")
    pitwf_map, pitwf_pe = {}, {}
    if os.path.exists(pitwf_path):
        pw = pd.read_parquet(pitwf_path).sort_values("filed")
        for (feat, sym), g in pw.groupby(["feature", "symbol"]):
            pitwf_map.setdefault(feat, {})[sym] = (list(g.filed), list(g.value))
            pitwf_pe.setdefault(feat, {})[sym] = list(g.period_end)
        print("walk-forward PIT coverage:",
              {f: len(v) for f, v in pitwf_map.items()})

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

    # per-symbol (rdq -> datadate) for the vintage guard: which quarter an upload holds
    fundq = pd.read_parquet(os.path.join(ML, "data/wrds/compustat_fundamentals_quarterly.parquet"),
                            columns=["tic", "datadate", "rdq"]).dropna()
    fundq["datadate"] = pd.to_datetime(fundq["datadate"]); fundq["rdq"] = pd.to_datetime(fundq["rdq"])
    fundq = fundq.sort_values("rdq")
    upload_q = {t: (list(g.rdq), list(g.datadate)) for t, g in fundq.groupby("tic")}

    def make_arm_b(upload_fn, uploads_list):
        def arm_b(date, feature, members=None):
            if feature != "roe":
                return upload_fn(date, feature, members)   # SAME stale base (gm/d2e)
            base = upload_fn(date, feature, members)
            d = pd.Timestamp(date)
            i_u = bisect.bisect_right(uploads_list, d) - 1
            u = uploads_list[max(i_u, 0)]
            cutoff = d - pd.Timedelta(days=1)
            syms = members if members is not None else list(base.keys())
            for s in syms:
                e = pit_map.get(s)
                if not e:
                    continue
                i = bisect.bisect_right(e[0], cutoff) - 1
                if i < 0:
                    continue
                # VINTAGE GUARD: use extraction only if newer than the upload quarter
                uq = upload_q.get(s)
                q_upload = None
                if uq:
                    j = bisect.bisect_right(uq[0], u) - 1
                    if j >= 0:
                        q_upload = uq[1][j]
                p_end = pit_pe[s][i]
                if q_upload is None or p_end > q_upload:
                    base[s] = e[1][i]
            return base
        return arm_b

    cfg = dict(V12_LIVE_BACKTEST_CONFIG)
    for k in ("vol_scaling", "vol_target", "vol_lookback"):
        cfg.pop(k, None)

    # SUMMER world: uploads STOP after May 15 each "gap year" — model the real 2026
    # situation by dropping uploads between May 15 and Nov 15 every year (worst-case
    # recurring summer gap; matches the current WRDS closure).
    # grid months are 3/6/9/12 (pd "3MS" snaps to month starts) -> drop the SEPTEMBER
    # upload: gap Jun 15 -> Dec 15 each year (v2 dropped month 8 = nothing; arms were
    # silently identical to normal -- caught by identical-to-the-decimal results)
    uploads_summer = [u for u in uploads if u.month not in (9,)]
    def last_upload_summer(date):
        i = bisect.bisect_right(uploads_summer, pd.Timestamp(date)) - 1
        return uploads_summer[max(i, 0)]
    def arm_a_summer(date, feature, members=None):
        if feature not in STALE_FEATS:
            return orig(date, feature, members)
        u = last_upload_summer(date)
        return orig(min(pd.Timestamp(date), u), feature, members) if u <= pd.Timestamp(date) \
            else orig(date, feature, members)

    arm_b = make_arm_b(arm_a, uploads)
    arm_b_summer = make_arm_b(arm_a_summer, uploads_summer)

    # DIAGNOSTIC: assistant roe + FRESH gm/d2e — isolates how much of the remaining
    # ceiling gap is gm/d2e staleness (certifiable someday) vs roe coverage/noise.
    def arm_b_plus(date, feature, members=None):
        if feature == "roe":
            return arm_b(date, feature, members)
        return orig(date, feature, members)

    # B++: ALL THREE features from REAL extractions (with their real 92-96% gm/d2e
    # accuracy), real filed dates, vintage guard — tests whether fresh-but-imperfect
    # beats stale-but-exact under the real upload calendar (the 99%-bar assumption).
    def make_bpp(src_map, src_pe):
        """Overlay ALL three features from a source (pitf=anchor-once, pitwf=walk-
        forward) under the production vintage guard: use extraction only for quarters
        newer than the last upload holds."""
        def arm(date, feature, members=None):
            if feature not in STALE_FEATS or feature not in src_map:
                return arm_a(date, feature, members) if feature in STALE_FEATS \
                    else orig(date, feature, members)
            base = arm_a(date, feature, members)
            fm, fpe = src_map[feature], src_pe[feature]
            d = pd.Timestamp(date)
            i_u = bisect.bisect_right(uploads, d) - 1
            u = uploads[max(i_u, 0)]
            cutoff = d - pd.Timedelta(days=1)
            syms = members if members is not None else list(base.keys())
            for s in syms:
                e = fm.get(s)
                if not e:
                    continue
                i = bisect.bisect_right(e[0], cutoff) - 1
                if i < 0:
                    continue
                uq = upload_q.get(s)
                q_upload = None
                if uq:
                    j = bisect.bisect_right(uq[0], u) - 1
                    if j >= 0:
                        q_upload = uq[1][j]
                if q_upload is None or fpe[s][i] > q_upload:
                    base[s] = e[1][i]
            return base
        return arm

    arm_bpp = make_bpp(pitf_map, pitf_pe)                       # anchor-once (drift-contaminated)
    arm_bwf = make_bpp(pitwf_map, pitwf_pe) if pitwf_map else None   # walk-forward (drift-fair)

    arms = [("CEILING (ideal fresh)", orig),
            ("A: no assistant (uploads)", arm_a),
            ("B: assistant roe only", arm_b),
            ("DIAG: B + fresh gm/d2e", arm_b_plus),
            ("B++: anchor-once gm/d2e", arm_bpp)]
    if arm_bwf is not None:
        arms.append(("B-WF: walkfwd gm/d2e", arm_bwf))
    print(f"{'arm':<30}{'start':<13}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}", flush=True)
    for label, fn in arms:
        bt.uni.get_feature_map = fn
        for st in STARTS:
            m = bt.run(st, END, cfg)
            print(f"{label:<30}{st:<13}{m['cagr']:>+8.1%}{m['sharpe']:>8.2f}{m['max_dd']:>+8.1%}", flush=True)
    bt.uni.get_feature_map = orig


if __name__ == "__main__":
    main()
