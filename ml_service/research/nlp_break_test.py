"""
DEEP NLP test on the structural-break task. Price features gave AUC 0.55 (thread1).
Does NEWS CONTENT around the gap separate breaks from bounces?

News features per (gap ticker, gap date) from FNSPID headlines in [gap-2, gap+3]:
  - n_news: article count (coverage/attention spike)
  - brk_kw: # headlines hitting PERMANENT-IMPAIRMENT keywords (fraud/SEC/probe/
            guidance-cut/going-concern/downgrade/trial-fail/bankruptcy/...)
  - rec_kw: # headlines hitting RECOVERABLE/overreaction keywords (beats/upgrade/
            oversold/raises/...)
  - net_kw: rec_kw - brk_kw ; brk_frac: brk_kw / n_news
Compare break-classifier OOS AUC: price-only vs news-only vs price+news, plus
precision at the top-decile exit. AUC>>0.55 with news -> the exit layer is viable.

Run after nlp_fetch_fnspid.py: cd ml_service && ./venv/bin/python research/nlp_break_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import re, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from main_production_backtest import FastBacktester

GAP, BREAK_H, BREAK_THR = 0.10, 40, -0.10
PRICE_FEATS = ["gap", "vol_20d", "vol_60d", "ret_252d", "ret_20d", "ret_60d",
               "dist_sma200", "dist_sma50", "rsi_14", "gp_assets", "net_margin",
               "debt_to_equity", "asset_growth", "net_issuance", "spy_dist200"]
NEWS_FEATS = ["n_news", "brk_kw", "rec_kw", "net_kw", "brk_frac"]

BRK_KW = re.compile(r"\b(fraud|probe|investigat|sec\b|s\.e\.c|doj|lawsuit|sued|"
    r"subpoena|litigation|misconduct|whistleblow|scandal|restat|accounting|"
    r"bankrupt|chapter 11|default|going concern|insolven|covenant|dilut|"
    r"cut guidance|lowered guidance|lowers guidance|slash|withdrawn|withdraws|"
    r"profit warning|warns|warning|downgrad|plunge|plummet|collapse|layoff|"
    r"restructur|write-?down|impair|recall|halt|delist|suspend|resign|steps down|"
    r"fired|ousted|departure|fails|failed|misses|missed|disappoint|weak|"
    r"rejection|complete response letter|crl|discontinu|guts|tumbl)\b", re.I)
REC_KW = re.compile(r"\b(beats?|tops?|raises?|raised|upgrad|outperform|buy rating|"
    r"overweight|price target rais|oversold|rebound|record|strong|surge|jumps?|"
    r"soar|rally|wins?|approval|approved|launch|partnership|upbeat|momentum)\b", re.I)


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def auc_cv(df, feats, label="brk"):
    preds = []
    for yr in range(2018, 2021):
        tr = df[df["date"] < f"{yr}-01-01"]; te = df[(df["date"] >= f"{yr}-01-01") & (df["date"] < f"{yr+1}-01-01")]
        if len(te) < 25 or len(tr) < 200 or te[label].nunique() < 2:
            continue
        m = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.04,
                               min_child_samples=30, n_jobs=1, verbose=-1,
                               subsample=0.8, colsample_bytree=0.8)
        m.fit(tr[feats], tr[label]); p = m.predict_proba(te[feats])[:, 1]
        t = te[[label]].copy(); t["p"] = p; preds.append(t)
    allp = pd.concat(preds)
    auc = roc_auc_score(allp[label], allp["p"])
    cut = allp["p"].quantile(0.90); fl = allp[allp["p"] >= cut]
    return auc, fl[label].mean(), allp[label].mean(), len(allp)


if __name__ == "__main__":
    t0 = time.time()
    news = pd.read_parquet("research/_fnspid_headlines.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news["brk_hit"] = news["title"].str.contains(BRK_KW).astype(int)
    news["rec_hit"] = news["title"].str.contains(REC_KW).astype(int)
    print(f"[news {len(news)} headlines, {news['symbol'].nunique()} tickers, "
          f"{news['date'].min().date()}..{news['date'].max().date()}]")
    # index news by (symbol)->df sorted by date for window lookups
    news = news.sort_values("date")
    by_sym = {s: g for s, g in news.groupby("symbol")}

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    spy = prices["SPY"]; spy_d200 = (spy - spy.rolling(200).mean()) / spy.rolling(200).mean()
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    fbd = bt.features_by_date
    # align to news coverage (FNSPID slice 2016..2020-06)
    nmax = news["date"].max()
    dates = [d for d in ai if pd.Timestamp("2016-06-01") <= d <= (nmax - pd.Timedelta(days=70))]
    print(f"[loaded {time.time()-t0:.0f}s]")

    recs = []
    for d in dates:
        gi = di[d]
        if gi + BREAK_H >= len(ai):
            continue
        sr = spy_r.loc[d]
        if pd.isna(sr) or d not in fbd:
            continue
        row = dr.loc[d]
        for sym in bt._get_sp1500(d):
            g = row.get(sym)
            if g is None or pd.isna(g) or (g - sr) >= -GAP or sym not in by_sym:
                continue
            fd = fbd[d].get(sym)
            if not fd:
                continue
            seg = dr[sym].iloc[gi+1:gi+1+BREAK_H]
            if seg.isna().all():
                continue
            fwd = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+1+BREAK_H].sum()
            # news window [gap-2, gap+3]
            g_sym = by_sym[sym]
            w = g_sym[(g_sym["date"] >= d - pd.Timedelta(days=2)) & (g_sym["date"] <= d + pd.Timedelta(days=3))]
            rec = {f: fd.get(f, np.nan) for f in PRICE_FEATS if f not in ("gap", "spy_dist200")}
            rec["gap"] = g - sr; rec["spy_dist200"] = spy_d200.loc[d]
            rec["n_news"] = len(w); rec["brk_kw"] = int(w["brk_hit"].sum())
            rec["rec_kw"] = int(w["rec_hit"].sum())
            rec["net_kw"] = rec["rec_kw"] - rec["brk_kw"]
            rec["brk_frac"] = rec["brk_kw"] / max(len(w), 1)
            rec["date"] = d; rec["brk"] = int(fwd < BREAK_THR)
            recs.append(rec)

    df = pd.DataFrame(recs).dropna(subset=["gap", "vol_20d", "ret_252d"])
    covered = (df["n_news"] > 0).mean()
    print(f"\n[{len(df)} gap-down events WITH news coverage; {covered*100:.0f}% have >=1 headline; "
          f"break rate {df['brk'].mean()*100:.0f}%]")
    print(f"  avg headlines/event {df['n_news'].mean():.1f}; break-kw events {(df['brk_kw']>0).mean()*100:.0f}%")

    # raw separation check
    print(f"\n=== raw keyword separation (break vs bounce) ===")
    for b, g in df.groupby("brk"):
        print(f"  {'BREAK ' if b else 'BOUNCE'}: brk_kw {g['brk_kw'].mean():.2f}  rec_kw {g['rec_kw'].mean():.2f}  "
              f"net_kw {g['net_kw'].mean():+.2f}  n_news {g['n_news'].mean():.1f}")

    print(f"\n=== break-classifier OOS AUC (2018-2023) ===")
    for name, feats in [("PRICE only", PRICE_FEATS), ("NEWS only", NEWS_FEATS),
                        ("PRICE + NEWS", PRICE_FEATS + NEWS_FEATS)]:
        auc, prec, base, n = auc_cv(df, feats)
        print(f"  {name:<14} AUC {auc:.3f}  top-decile exit precision {prec*100:.0f}% "
              f"(base {base*100:.0f}%, {prec/base:.1f}x)  n={n}")
    print(f"\n[total {time.time()-t0:.0f}s]")
