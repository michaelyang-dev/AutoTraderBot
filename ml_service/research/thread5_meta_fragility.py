"""
Thread #5 — meta-model book-fragility -> exposure. Predict the STRATEGY's own
forward 20d return from book-state + market features (all PIT), use it to de-risk.
HONEST CAVEAT: one NAV path -> few independent 20d episodes -> high overfit risk
(same small-sample issue that capped regime detection at 58%). So: simple/regularized
model + LINEAR baseline + report OOS predictive corr BEFORE judging the overlay.
Compared at matched exposure vs incumbents (vol-scaling Sharpe ~1.19).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from main_production_backtest import FastBacktester
from research.exposure_lib import (metrics, exposure_from_signal, spy_trend_exposure,
                                   vol_scale_exposure, apply_overlay)

BASE = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
            top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
            bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
            record_targets=True)


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def show(name, m, avg_e):
    print(f"  {name:<32} CAGR {m['cagr']*100:5.1f}%  Sharpe {m['sharpe']:.2f}  "
          f"Sortino {m['sortino']:.2f}  MaxDD {m['mdd']*100:6.1f}%  avgExpo {avg_e*100:3.0f}%")


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester(); clear(bt)
    r = bt.run("2016-01-01", "2025-12-31", BASE)
    nav = r["daily_values"].dropna(); base_r = nav.pct_change().fillna(0); idx = nav.index
    print(f"[loaded+ran {time.time()-t0:.0f}s]")

    # daily held book (ffill between rebalances)
    log = {d: set(w) for d, w in bt._rebal_log}
    held = {}
    cur = set()
    for d in idx:
        if d in log:
            cur = log[d]
        held[d] = cur

    fbd = bt.features_by_date
    rows = []
    for d in idx:
        names = [s for s in held[d] if d in fbd and s in fbd[d]]
        if len(names) < 5:
            rows.append({}); continue
        def avg(f):
            v = [fbd[d][s].get(f, np.nan) for s in names]
            v = [x for x in v if x == x]
            return np.mean(v) if v else np.nan
        r20 = [fbd[d][s].get("ret_20d", np.nan) for s in names]
        r20 = [x for x in r20 if x == x]
        rows.append(dict(b_mom=avg("ret_252d"), b_vol=avg("vol_20d"),
                         b_strch=avg("dist_sma200"), b_disp=np.std(r20) if r20 else np.nan,
                         b_r20=avg("ret_20d")))
    X = pd.DataFrame(rows, index=idx)

    # market features
    vix = pd.read_parquet("data/enhanced_data/vix_cache.parquet"); vix.index = pd.to_datetime(vix.index)
    X["vix"] = vix["^VIX"].reindex(idx).ffill()
    X["vixterm"] = (vix["^VIX"] / vix["^VIX3M"]).reindex(idx).ffill()
    spy = bt.prices["SPY"].reindex(idx).ffill()
    X["spy_dist200"] = (spy - spy.rolling(200).mean()) / spy.rolling(200).mean()
    X["strat_mom20"] = base_r.rolling(20).sum()
    X["strat_vol20"] = base_r.rolling(20).std()

    # label: forward 20d strategy return = sum of base_r over days t+1..t+20
    br = base_r.values
    fwd20 = np.array([br[i + 1:i + 21].sum() if i + 21 <= len(br) else np.nan
                      for i in range(len(br))])
    y = pd.Series(fwd20, index=idx)

    feat = ["b_mom", "b_vol", "b_strch", "b_disp", "b_r20", "vix", "vixterm",
            "spy_dist200", "strat_mom20", "strat_vol20"]
    data = X[feat].copy(); data["y"] = y
    data = data.dropna()

    # walk-forward yearly w/ 20d embargo
    pred_lgb = pd.Series(index=data.index, dtype=float)
    pred_lin = pd.Series(index=data.index, dtype=float)
    for yr in range(2019, 2026):
        tr = data[data.index < f"{yr}-01-01"]
        tr = tr.iloc[:-20] if len(tr) > 20 else tr   # embargo overlap
        te = data[(data.index >= f"{yr}-01-01") & (data.index < f"{yr+1}-01-01")]
        if len(te) == 0 or len(tr) < 200:
            continue
        gb = lgb.LGBMRegressor(n_estimators=120, num_leaves=7, learning_rate=0.03,
                               min_child_samples=50, n_jobs=1, verbose=-1,
                               subsample=0.8, colsample_bytree=0.8)
        gb.fit(tr[feat], tr["y"]); pred_lgb.loc[te.index] = gb.predict(te[feat])
        ln = Ridge(alpha=10.0)
        ln.fit((tr[feat] - tr[feat].mean()) / tr[feat].std(), tr["y"])
        pred_lin.loc[te.index] = ln.predict((te[feat] - tr[feat].mean()) / tr[feat].std())

    act = data["y"]
    oos = pred_lgb.dropna().index
    print(f"\n=== OOS predictive power (forward 20d strat return), 2019-2025 ===")
    print(f"  LGB  corr(pred, actual) = {pred_lgb.loc[oos].corr(act.loc[oos]):+.3f}")
    print(f"  Ridge corr(pred, actual) = {pred_lin.loc[oos].corr(act.loc[oos]):+.3f}")

    # overlay: de-risk when predicted forward return is LOW (high_is_risk on -pred)
    e_spy = spy_trend_exposure(bt.prices, idx); _, target = apply_overlay(base_r.values, idx, e_spy)
    e_vol = vol_scale_exposure(base_r.values, idx); rv, av = apply_overlay(base_r.values, idx, e_vol)

    def m_oos(rr):
        s = pd.Series(rr, index=idx).reindex(oos); return metrics(s.values, oos)
    print(f"\n=== meta-fragility exposure vs incumbents (matched {target*100:.0f}%, OOS) ===")
    show("NONE", m_oos(base_r.values), 1.0)
    rs, as_ = apply_overlay(base_r.values, idx, e_spy); show("SPY<SMA200", m_oos(rs), as_)
    show("vol-scaling", m_oos(rv), av)
    for nm, pr in [("meta LGB", pred_lgb), ("meta Ridge", pred_lin)]:
        e = exposure_from_signal(-pr.reindex(idx), idx, floor=0.40, target_avg=target, high_is_risk=True)
        rr, ae = apply_overlay(base_r.values, idx, e); show(nm, m_oos(rr), ae)
    print(f"\n[total {time.time()-t0:.0f}s]")
