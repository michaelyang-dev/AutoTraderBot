"""
Thread #4 — ML for RISK (not return): a fresh crash CLASSIFIER used as a de-risk
trigger. Build a NEW model (not the IC=0.014 return ranker). Label = 15d forward
return < -10% (a crash). Walk-forward LightGBM, aggregate OOS predicted crash-prob
across the universe per day -> market crash signal -> exposure overlay. Compare at
matched avg exposure vs incumbents (vol-scaling Sharpe ~1.11 is the bar).
Reports OOS AUC (does ML predict crashes at all?) + whether the AGGREGATE times de-risking.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from main_production_backtest import FastBacktester
from research.exposure_lib import (metrics, exposure_from_signal, spy_trend_exposure,
                                   vol_scale_exposure, apply_overlay)

BASE = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
            top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
            bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10})


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def show(name, m, avg_e):
    print(f"  {name:<34} CAGR {m['cagr']*100:5.1f}%  Sharpe {m['sharpe']:.2f}  "
          f"Sortino {m['sortino']:.2f}  MaxDD {m['mdd']*100:6.1f}%  avgExpo {avg_e*100:3.0f}%")


if __name__ == "__main__":
    t0 = time.time()

    # ---- train fresh crash classifier (walk-forward) ----
    df = pd.read_parquet("data/ml_training_enhanced.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret_15d"]).sort_values("date")
    # EXCLUDE leaked feature: target_rank == cross-sectional rank of fwd_ret_15d
    # (corr 0.61 with the label — confirmed leakage, thread4_diag.py)
    LEAK = {"date", "symbol", "fwd_ret_15d", "target_rank"}
    feat_cols = [c for c in df.columns if c not in LEAK and df[c].dtype != "O"]
    print(f"[dropped leaked 'target_rank'; using {len(feat_cols)} clean features]")
    df["crash"] = (df["fwd_ret_15d"] < -0.10).astype(int)
    print(f"[ml data {df.shape}, {len(feat_cols)} feats, crash base rate {df['crash'].mean()*100:.1f}%]")

    preds = []
    for yr in range(2019, 2026):
        tr = df[df["date"] < f"{yr}-01-01"]
        te = df[(df["date"] >= f"{yr}-01-01") & (df["date"] < f"{yr+1}-01-01")]
        if len(te) == 0 or len(tr) < 5000:
            continue
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               n_jobs=1, verbose=-1, subsample=0.8, colsample_bytree=0.8)
        m.fit(tr[feat_cols], tr["crash"])
        p = m.predict_proba(te[feat_cols])[:, 1]
        auc = roc_auc_score(te["crash"], p) if te["crash"].nunique() > 1 else float("nan")
        print(f"  {yr} OOS AUC {auc:.3f}  (n={len(te)}, crash {te['crash'].mean()*100:.0f}%)")
        tmp = te[["date"]].copy(); tmp["p"] = p
        preds.append(tmp)
    allp = pd.concat(preds)
    crash_prob = allp.groupby("date")["p"].mean()   # market-aggregate crash probability
    crash_prob.index = pd.to_datetime(crash_prob.index)

    # ---- base book returns + overlay comparison ----
    bt = FastBacktester(); clear(bt)
    r = bt.run("2016-01-01", "2025-12-31", BASE)
    nav = r["daily_values"].dropna(); base_r = nav.pct_change().fillna(0).values; idx = nav.index

    e_spy = spy_trend_exposure(bt.prices, idx); _, target = apply_overlay(base_r, idx, e_spy)
    e_vol = vol_scale_exposure(base_r, idx); r_vol, a_vol = apply_overlay(base_r, idx, e_vol)

    print(f"\n=== ML crash-prob de-risk vs incumbents (matched avg {target*100:.0f}%, 2019-2025 OOS) ===")
    # restrict to OOS window where we have crash_prob
    oos = idx[idx >= crash_prob.index.min()]
    def m_on(rr):
        s = pd.Series(rr, index=idx).reindex(oos)
        return metrics(s.values, oos)
    show("NONE", m_on(base_r), 1.0)
    rs, as_ = apply_overlay(base_r, idx, e_spy); show("SPY<SMA200", m_on(rs), as_)
    show("vol-scaling", m_on(r_vol), a_vol)
    e_ml = exposure_from_signal(crash_prob, idx, floor=0.40, target_avg=target, high_is_risk=True)
    rml, aml = apply_overlay(base_r, idx, e_ml); show("ML crash-prob (aggregate)", m_on(rml), aml)
    # combo with vol-scaling
    e_cb = pd.concat([e_vol, e_ml], axis=1).min(axis=1)
    rcb, acb = apply_overlay(base_r, idx, e_cb); show("vol-scaling + ML crash", m_on(rcb), acb)
    print(f"\n[total {time.time()-t0:.0f}s]")
