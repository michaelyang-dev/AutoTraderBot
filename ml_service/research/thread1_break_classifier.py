"""
Thread #1 — structural-break vs bounce classifier (the exit layer's core job).
Foundational study: hard idiosyncratic gap-downs BOUNCE on average. So a useful
exit layer must identify the RARE 'breaks' (keep falling) inside the gap-down crowd.
Decisive question: can features DISCRIMINATE break-from-bounce?

Population: ALL SP1500 idiosyncratic gap-downs (1d ret - SPY < -gap), 2016-2025
(big sample). Label: BREAK = forward 40d market-relative return < -10% (didn't
recover). Features (PIT at the gap): gap size, vols, momentum, trend distance, rsi,
quality/valuation, market state. Walk-forward LightGBM -> OOS AUC + precision at a
high-confidence cut (if we exit the top-decile most-likely breaks, how many are
real breaks vs bounces we wrongly dump?).

AUC>~0.60 -> price features discriminate -> exit layer viable (NLP would add).
AUC~0.50 -> need news CONTENT to tell breaks from bounces -> scope GDELT/NLP.
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

GAP = 0.10        # idiosyncratic 1d drop threshold (name - SPY)
BREAK_H = 40      # forward horizon for break label
BREAK_THR = -0.10 # break = fwd40 mkt-rel < -10%
FEATS = ["gap", "vol_20d", "vol_60d", "ret_252d", "ret_20d", "ret_60d",
         "dist_sma200", "dist_sma50", "rsi_14", "gp_assets", "net_margin",
         "debt_to_equity", "asset_growth", "net_issuance", "spy_dist200"]


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    print(f"[loaded {time.time()-t0:.0f}s]")

    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    spy = prices["SPY"]; spy_d200 = (spy - spy.rolling(200).mean()) / spy.rolling(200).mean()
    allidx = list(prices.index); di_map = {d: i for i, d in enumerate(allidx)}
    fbd = bt.features_by_date
    dates = [d for d in allidx if pd.Timestamp("2016-06-01") <= d <= pd.Timestamp("2025-10-01")]

    recs = []
    for d in dates:
        gi = di_map[d]
        if gi + BREAK_H >= len(allidx):
            continue
        members = bt._get_sp1500(d)
        if d not in fbd:
            continue
        row = dr.loc[d]; sr = spy_r.loc[d]
        if pd.isna(sr):
            continue
        # candidate gap-down names this day
        for sym in members:
            g = row.get(sym)
            if g is None or pd.isna(g):
                continue
            idio = g - sr
            if idio >= -GAP:
                continue
            fd = fbd[d].get(sym)
            if not fd:
                continue
            # forward 40d mkt-rel (from day after gap)
            seg = dr[sym].iloc[gi+1:gi+1+BREAK_H]
            if seg.isna().all():
                continue
            fwd = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+1+BREAK_H].sum()
            rec = {f: fd.get(f, np.nan) for f in FEATS if f not in ("gap", "spy_dist200")}
            rec["gap"] = idio
            rec["spy_dist200"] = spy_d200.loc[d]
            rec["date"] = d
            rec["brk"] = int(fwd < BREAK_THR)
            recs.append(rec)

    df = pd.DataFrame(recs).dropna(subset=["gap", "vol_20d", "ret_252d"])
    print(f"[{len(df)} gap-down events (<-{int(GAP*100)}% idio), break rate "
          f"{df['brk'].mean()*100:.0f}% (fwd{BREAK_H}d mkt-rel < {int(BREAK_THR*100)}%)]")

    # walk-forward
    preds = []
    for yr in range(2019, 2026):
        tr = df[df["date"] < f"{yr}-01-01"]; te = df[(df["date"] >= f"{yr}-01-01") & (df["date"] < f"{yr+1}-01-01")]
        if len(te) < 30 or len(tr) < 300 or te["brk"].nunique() < 2:
            continue
        m = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.04,
                               min_child_samples=40, n_jobs=1, verbose=-1,
                               subsample=0.8, colsample_bytree=0.8)
        m.fit(tr[FEATS], tr["brk"])
        p = m.predict_proba(te[FEATS])[:, 1]
        auc = roc_auc_score(te["brk"], p)
        t = te[["brk"]].copy(); t["p"] = p; preds.append(t)
        print(f"  {yr} OOS AUC {auc:.3f}  (n={len(te)}, break {te['brk'].mean()*100:.0f}%)")

    allp = pd.concat(preds)
    print(f"\n  POOLED OOS AUC {roc_auc_score(allp['brk'], allp['p']):.3f}")
    # precision at top-decile most-likely-break (the names we'd exit)
    cut = allp["p"].quantile(0.90)
    flagged = allp[allp["p"] >= cut]
    base = allp["brk"].mean()
    print(f"  if we EXIT the top-10% most-likely breaks: precision {flagged['brk'].mean()*100:.0f}% "
          f"(base rate {base*100:.0f}%) -> {flagged['brk'].mean()/base:.1f}x lift")
    print(f"  meaning {100-flagged['brk'].mean()*100:.0f}% of exits would be BOUNCES wrongly dumped")
    print(f"\n[total {time.time()-t0:.0f}s]")
