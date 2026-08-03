"""
TAIL-RISK AUDIT of the 2026-07-25 universe-parity fix (SP500-only pool -> full SP1500).

Question: the fix adds ~1000 S&P400/600 mid/small caps to the momentum + low-vol pools.
Those names are more volatile and can gap catastrophically. Does the DEPLOYED risk stack
(15%-of-NAV single-name cap, 40% trailing stop, vol-scaling, credit gate) still hold, or
does the fix buy return with a tail-risk bill?

Everything runs in the LIVE-MIRROR engine at live parity ($50k, integer shares, 1.49x gross,
6.3% financing, vol_scale_cap=1.0, credit p95 gate) — same BASE as research/threadF1_universe_ab.py.

Usage:  python3 research/tailrisk_run.py 8yr|26yr <outdir>
"""
import os, sys, time, pickle, bisect
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _tailrisk_fork import TailRiskBacktester  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063}
GATE = dict(credit_pct=0.95, credit_derisk=0.5)

PERIODS = {
    "8yr":  ("data/wrds/complete_sp1500_universe.pkl",
             ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    "26yr": ("data/wrds/sp1500_universe_2000.pkl",
             ["2001-01-02", "2001-01-17"], "2025-12-31"),
}
MEMPKL = os.environ.get("TR_MEMPKL", "/tmp/_tailrisk_membership.pkl")


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def risk_stats(v):
    """Full risk panel from a daily NAV series."""
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1
    vol = dr.std() * np.sqrt(252)
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dsd = dr[dr < 0].std() * np.sqrt(252)
    sortino = (dr.mean() * 252) / dsd if dsd > 0 else 0
    dd = (v - v.cummax()) / v.cummax()
    mdd = dd.min()
    # rolling 20d NAV return (worst calendar month-ish window)
    r20 = (v / v.shift(20) - 1).dropna()
    uw = (dd < -1e-9)
    longest, cur = 0, 0
    for x in uw.values:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    return {
        "CAGR": cagr, "Vol": vol, "Sharpe": sh, "Sortino": sortino, "MaxDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd else np.nan,
        "worst1d": dr.min(), "worst20d": r20.min(),
        "VaR95": dr.quantile(0.05), "VaR99": dr.quantile(0.01),
        "CVaR99": dr[dr <= dr.quantile(0.01)].mean(),
        "ulcer": np.sqrt((dd ** 2).mean()),
        "d_lt_-3%": float((dr < -0.03).mean()), "d_lt_-5%": float((dr < -0.05).mean()),
        "uw_days": longest, "finalNAV": v.iloc[-1],
    }


def main():
    period = sys.argv[1] if len(sys.argv) > 1 else "8yr"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
    path, starts, end = PERIODS[period]
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    bt = TailRiskBacktester(universe_path=path); clear_deployed(bt)
    print(f"[{period}] loaded {time.time()-t0:.0f}s", flush=True)

    # ---- clean PIT membership (the 26yr pickle's own mem dicts are polluted) ----
    snaps = pickle.load(open(MEMPKL, "rb"))
    snap_keys = sorted(snaps.keys())

    def bucket(d, sym):
        i = bisect.bisect_right(snap_keys, d) - 1
        if i < 0:
            return "?"
        s = snaps[snap_keys[i]]
        if sym in s.get("500", ()):
            return "SP500"
        if sym in s.get("400", ()):
            return "SP400"
        if sym in s.get("600", ()):
            return "SP600"
        return "other"

    clean500 = {d: snaps[d]["500"] for d in snap_keys if "500" in snaps[d]}

    # ---- PARITY: fork must be bit-identical to the unmodified live-mirror ----
    ref = LiveMirrorBacktester(universe_path=path); clear_deployed(ref)
    cfgp = dict(BASE); cfgp.update(GATE)
    a = bt.run(starts[0], end, cfgp); b = ref.run(starts[0], end, cfgp)
    same = np.allclose(a["daily_values"].values, b["daily_values"].values)
    print(f"[{period}] PARITY fork==livemirror: {same}  (fork final {a['final']:,.0f} / "
          f"ref final {b['final']:,.0f})", flush=True)
    del ref

    # SP500-pool replica needs a CLEAN membership dict on the 26yr pickle
    sp500_over = clean500 if period == "26yr" else None

    VARIANTS = [
        ("A SP1500  stop40  1.49x  (POST-FIX LIVE)", dict(GATE)),
        ("B SP500   stop40  1.49x  (PRE-FIX LIVE)", dict(mom_pool_sp500=True, sp500_mem_override=sp500_over, **GATE)),
        ("C SP1500  NOSTOP  1.49x", dict(trailing_stop=None, **GATE)),
        ("D SP500   NOSTOP  1.49x", dict(mom_pool_sp500=True, sp500_mem_override=sp500_over, trailing_stop=None, **GATE)),
        ("E SP1500  stop40  1.00x", dict(leverage=1.0, financing_rate=0.0, **GATE)),
        ("F SP500   stop40  1.00x", dict(mom_pool_sp500=True, sp500_mem_override=sp500_over, leverage=1.0, financing_rate=0.0, **GATE)),
        ("G SP1500  stop40  1.49x  NO CREDIT GATE", {}),
        ("H SP500   stop40  1.49x  NO CREDIT GATE", dict(mom_pool_sp500=True, sp500_mem_override=sp500_over)),
    ]

    rows = {}
    logs = {}
    for name, extra in VARIANTS:
        per_start = []
        for i, st in enumerate(starts):
            cfg = dict(BASE); cfg.update(extra)
            m = bt.run(st, end, cfg)
            per_start.append(risk_stats(m["daily_values"]))
            if i == 0:                      # keep the primary-start logs + NAV
                logs[name] = {
                    "stops": pd.DataFrame(bt._stop_log),
                    "pos": pd.DataFrame(bt._pos_daily, columns=["date", "sym", "value", "nav"]),
                    "mom": pd.DataFrame([(d, s) for d, syms in bt._mom_log for s in syms],
                                        columns=["date", "sym"]),
                    "tgt": pd.DataFrame(bt._tgt_log,
                                        columns=["date", "sym", "w_book", "lev_t", "is_mom", "n_capped"]),
                    "nav": m["daily_values"],
                    "avg_gross": m["avg_gross"],
                }
        rows[name] = {k: float(np.mean([p[k] for p in per_start])) for k in per_start[0]}
        r = rows[name]
        print(f"  {name:<44} CAGR {r['CAGR']:>+7.1%}  Vol {r['Vol']:>6.1%}  Sh {r['Sharpe']:>5.2f} "
              f" Sort {r['Sortino']:>5.2f}  MaxDD {r['MaxDD']:>+7.1%}  w1d {r['worst1d']:>+6.1%} "
              f" w20d {r['worst20d']:>+7.1%}  ulcer {r['ulcer']:>5.1%}  gross {logs[name]['avg_gross']:.2f}",
              flush=True)

    # ---------------- persist raw material ----------------
    pd.DataFrame(rows).T.to_parquet(f"{outdir}/tr_{period}_variants.parquet")
    for name, L in logs.items():
        tag = name.split()[0]
        for k in ["stops", "pos", "mom", "tgt"]:
            df = L[k]
            if len(df):
                df = df.copy()
                if "sym" in df and "date" in df:
                    df["bucket"] = [bucket(d, s) for d, s in zip(df["date"], df["sym"])]
                df.to_parquet(f"{outdir}/tr_{period}_{tag}_{k}.parquet")
        L["nav"].to_frame("nav").to_parquet(f"{outdir}/tr_{period}_{tag}_nav.parquet")

    # price matrix (needed for forward-path / gap analysis) — save once, restricted to
    # the tickers we ever touched, to keep the file small
    touched = set()
    for L in logs.values():
        touched |= set(L["mom"]["sym"]) | set(L["pos"]["sym"])
    cols = [c for c in bt.prices.columns if c in touched]
    bt.prices[cols].to_parquet(f"{outdir}/tr_{period}_prices.parquet")
    # per-date vol_20d for the picks
    vrows = []
    for L in logs.values():
        for d, s in zip(L["mom"]["date"], L["mom"]["sym"]):
            fd = bt.features_by_date.get(d, {}).get(s, {})
            vrows.append((d, s, fd.get("vol_20d"), fd.get("vol_60d"),
                          fd.get("ret_252d"), fd.get("ret_20d"), fd.get("dist_sma200")))
    pd.DataFrame(vrows, columns=["date", "sym", "vol_20d", "vol_60d", "ret_252d",
                                 "ret_20d", "dist_sma200"]).drop_duplicates(
        subset=["date", "sym"]).to_parquet(f"{outdir}/tr_{period}_pickfeat.parquet")
    print(f"[{period}] wrote logs to {outdir}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
