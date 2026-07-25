"""
THREAD V — give the "value" sleeve an actual VALUATION signal (WRDS financial ratios, PIT).

Finding: strategy_value scores on -ret_252d*0.30 + gm*0.25 + roe*0.25 — quality + long-term
reversal, ZERO price-vs-fundamentals content. We own WRDS's professional monthly PIT ratio
library (research/_ratio_panel.parquet: bm, evm, pcf, GProf..., public_date-stamped,
1999->2025) — never read by any code. The expert's "use a real valuation metric" test
(tier1_improvements) never persisted a verdict. 35% of the book rides on this sleeve.

Pre-committed variants (filters & weighting mechanics UNCHANGED — only the score changes;
ratio lookup is PIT: latest public_date <= date, max 400d stale):
  V0 baseline    current score
  V1 real-value  score = z(bm) - z(evm)          (cheap on book AND EV-multiple)
  V2 val+prof    score = 0.5*(z(bm)-z(evm)) + 0.5*z(GProf)   (cheap AND profitable)
  V3 additive    current score + 0.25*(z(bm)-z(evm))          (keep reversal, add valuation)
Full live-mirror (1.49x integer $50k fin, vol_scale_cap=1.0, credit gate ON), both periods.
Bar: beat baseline Sharpe at >= -0.5pp CAGR in BOTH periods (or clean Pareto), same params.
"""
import os, sys, time, bisect
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd  # noqa: E402
import livemirror_backtest as lm  # noqa: E402
from strategies.multi_strategy_engine import strategy_value as sv_orig  # noqa: E402

# ---- PIT ratio lookup ----
_panel = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ratio_panel.parquet"))
_panel = _panel.sort_values(["ticker", "public_date"])
_BY = {}
for t, g in _panel.groupby("ticker"):
    _BY[t] = (g["public_date"].values.astype("datetime64[ns]"),
              g[["bm", "evm", "GProf"]].values)


def pit_ratios(sym, date):
    rec = _BY.get(sym)
    if rec is None:
        return None
    dates, vals = rec
    i = np.searchsorted(dates, np.datetime64(date)) - 1
    if i < 0:
        return None
    if (np.datetime64(date) - dates[i]) > np.timedelta64(400, "D"):
        return None
    return vals[i]   # [bm, evm, GProf]


def zmap(d):
    v = np.array([x for x in d.values() if x is not None and not np.isnan(x)])
    if len(v) < 5:
        return {k: 0.0 for k in d}
    mu, sd = v.mean(), v.std() or 1.0
    return {k: ((x - mu) / sd if (x is not None and not np.isnan(x)) else 0.0) for k, x in d.items()}


def make_sv(mode):
    def sv(uni, date, members, top_n=10):
        roe = uni.get_feature_map(date, "roe", members)
        gm = uni.get_feature_map(date, "gross_margin", members)
        r252 = uni.get_feature_map(date, "ret_252d", members)
        d200 = uni.get_feature_map(date, "dist_sma200", members)
        de = uni.get_feature_map(date, "debt_to_equity", members)
        elig, bmv, evv, gpv = [], {}, {}, {}
        for sym in members:
            r = roe.get(sym); g = gm.get(sym); rv = r252.get(sym)
            dv = d200.get(sym); debt = de.get(sym)
            if r is None or g is None or rv is None:
                continue
            if np.isnan(r) or np.isnan(g) or np.isnan(rv):
                continue
            if r < 0.05 or g < 0.15:
                continue
            if dv is None or np.isnan(dv) or dv < -0.15:
                continue
            if debt is not None and not np.isnan(debt) and debt > 3.0:
                continue
            elig.append(sym)
            pr = pit_ratios(sym, date)
            bmv[sym] = pr[0] if pr is not None else np.nan
            evv[sym] = pr[1] if pr is not None else np.nan
            gpv[sym] = pr[2] if pr is not None else np.nan
        if not elig:
            return {}
        zbm, zev, zgp = zmap(bmv), zmap(evv), zmap(gpv)
        scores = {}
        for sym in elig:
            cur = -r252[sym] * 0.30 + gm[sym] * 0.25 + min(roe[sym], 0.5) * 0.25
            valz = zbm[sym] - zev[sym]
            if mode == "V1":
                scores[sym] = valz
            elif mode == "V2":
                scores[sym] = 0.5 * valz + 0.5 * zgp[sym]
            elif mode == "V3":
                scores[sym] = cur + 0.25 * valz
            else:
                scores[sym] = cur
        ss = sorted(scores, key=scores.get, reverse=True)[:top_n]
        sc = [max(scores[s] - min(scores.values()) + 0.001, 0.001) for s in ss]  # shift +ve for weighting
        total = sum(sc)
        w = {s: min(v / total, 2.0 / len(ss)) for s, v in zip(ss, sc)}
        wt = sum(w.values())
        return {s: x / wt for s, x in w.items()} if wt > 0 else {s: 1.0 / len(ss) for s in ss}
    return sv


BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063, "credit_pct": 0.95, "credit_derisk": 0.5}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
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
        print(f"{pname} | value-sleeve score variants | live-mirror cap1.0 + gate | {len(starts)}-start", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = lm.LiveMirrorBacktester(universe_path=path); clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'variant':<40}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}"
        print(hdr); print("-" * len(hdr), flush=True)
        for mode, label in [("V0", "V0 current (reversal+quality)"),
                            ("V1", "V1 real-value z(bm)-z(evm)"),
                            ("V2", "V2 value+profitability"),
                            ("V3", "V3 current + 0.25*valuation")]:
            lm.strategy_value = make_sv(mode)
            cs, ss, ds = [], [], []
            for st in starts:
                m = bt.run(st, end, dict(BASE))
                c, s, d = stat(m["daily_values"])
                cs.append(c); ss.append(s); ds.append(d)
            print(f"{label:<40}{np.mean(cs):>+8.1%}{np.mean(ss):>8.2f}{np.mean(ds):>+8.1%}", flush=True)
        lm.strategy_value = sv_orig
        del bt
    print("\nBAR: beat V0 Sharpe at >=-0.5pp CAGR BOTH periods (or Pareto).", flush=True)


if __name__ == "__main__":
    main()
