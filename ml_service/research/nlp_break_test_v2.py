"""
DEEP NLP break test v2 — dig past the confounds:
  (1) FILTER listicle/noise headlines (biggest movers, premarket, 52-week, etc.)
  (2) EXCLUDE COVID crash (2020-02-15..2020-04-30) = market-wide, not news-driven
  (3) REFINED event-type features (downgrade / legal-SEC / guidance-cut / earnings-miss
      / analyst-action / dilution-offering) as binary flags, not raw counts
  (4) multiple break horizons (20d/40d)
Compare price vs +news AUC on the IDIOSYNCRATIC, de-noised population.
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

GAP = 0.10
PRICE_FEATS = ["gap", "vol_20d", "vol_60d", "ret_252d", "ret_20d", "ret_60d",
               "dist_sma200", "dist_sma50", "rsi_14", "gp_assets", "net_margin",
               "debt_to_equity", "asset_growth", "net_issuance", "spy_dist200"]
NEWS_FEATS = ["n_news", "f_downgrade", "f_legal", "f_guidance", "f_miss",
              "f_analyst_cut", "f_dilution", "f_struct_any", "f_upgrade"]

NOISE = re.compile(r"biggest mover|stocks moving|stocks that hit|52-?week|mid-?day|"
    r"premarket|pre-market|after hours|after-hours|what you need to know|"
    r"\bgainers?\b|\blosers?\b|movers from|moving in|watch list|unusual options|"
    r"options activity|here's what|things to know|market update|stocks to watch", re.I)
F = {
 "f_downgrade": re.compile(r"downgrad|cut to (sell|underweight|hold)|lowered to|"
    r"reduces? (rating|to)|underperform", re.I),
 "f_legal": re.compile(r"\bsec\b|s\.e\.c|investigat|probe|lawsuit|sued|subpoena|"
    r"fraud|misconduct|whistleblow|litigation|class action|doj|settlement|restat", re.I),
 "f_guidance": re.compile(r"cuts? guidance|lowers? guidance|lowered guidance|"
    r"withdraw|slashes? (guidance|outlook|forecast)|profit warning|warns", re.I),
 "f_miss": re.compile(r"miss(es|ed)?\b|falls short|disappoint|weaker|below estimate|"
    r"below expectation", re.I),
 "f_analyst_cut": re.compile(r"price target (cut|lowered|reduce|slash)|lowers? (pt|target)|"
    r"cuts? price target", re.I),
 "f_dilution": re.compile(r"offering|dilut|prices? (\$|[0-9].* (million|billion).* (shares|notes|stock))|"
    r"convertible notes|capital raise|secondary", re.I),
 "f_upgrade": re.compile(r"upgrad|raises? (rating|to|price target|pt)|outperform|"
    r"overweight|initiates? (buy|overweight)|beats?\b|tops? estimate", re.I),
}


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def auc_cv(df, feats, label):
    preds = []
    for yr in range(2018, 2021):
        tr = df[df["date"] < f"{yr}-01-01"]; te = df[(df["date"] >= f"{yr}-01-01") & (df["date"] < f"{yr+1}-01-01")]
        if len(te) < 20 or len(tr) < 150 or te[label].nunique() < 2:
            continue
        m = lgb.LGBMClassifier(n_estimators=120, num_leaves=12, learning_rate=0.04,
                               min_child_samples=25, n_jobs=1, verbose=-1,
                               subsample=0.8, colsample_bytree=0.8)
        m.fit(tr[feats], tr[label]); p = m.predict_proba(te[feats])[:, 1]
        t = te[[label]].copy(); t["p"] = p; preds.append(t)
    if not preds:
        return None, None, None, 0
    allp = pd.concat(preds)
    auc = roc_auc_score(allp[label], allp["p"])
    cut = allp["p"].quantile(0.85); fl = allp[allp["p"] >= cut]
    return auc, fl[label].mean(), allp[label].mean(), len(allp)


if __name__ == "__main__":
    t0 = time.time()
    news = pd.read_parquet("research/_fnspid_headlines.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    n0 = len(news)
    news = news[~news["title"].str.contains(NOISE)]
    print(f"[news {n0}->{len(news)} after noise filter ({(1-len(news)/n0)*100:.0f}% dropped)]")
    for k, rgx in F.items():
        news[k] = news["title"].str.contains(rgx).astype(int)
    news = news.sort_values("date")
    by_sym = {s: g for s, g in news.groupby("symbol")}

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    spy = prices["SPY"]; spy_d200 = (spy - spy.rolling(200).mean()) / spy.rolling(200).mean()
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    fbd = bt.features_by_date
    nmax = news["date"].max()
    dates = [d for d in ai if pd.Timestamp("2016-06-01") <= d <= (nmax - pd.Timedelta(days=70))]
    COVID = (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30"))
    print(f"[loaded {time.time()-t0:.0f}s]")

    recs = []
    for d in dates:
        if COVID[0] <= d <= COVID[1]:
            continue
        gi = di[d]
        if gi + 40 >= len(ai):
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
            seg = dr[sym].iloc[gi+1:gi+41]; seg20 = dr[sym].iloc[gi+1:gi+21]
            if seg.isna().all():
                continue
            fwd40 = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+41].sum()
            fwd20 = seg20.fillna(0).sum() - spy_r.iloc[gi+1:gi+21].sum()
            gw = by_sym[sym]
            w = gw[(gw["date"] >= d - pd.Timedelta(days=2)) & (gw["date"] <= d + pd.Timedelta(days=3))]
            rec = {f: fd.get(f, np.nan) for f in PRICE_FEATS if f not in ("gap", "spy_dist200")}
            rec["gap"] = g - sr; rec["spy_dist200"] = spy_d200.loc[d]; rec["n_news"] = len(w)
            for k in F:
                rec[k] = int(w[k].sum() > 0)
            rec["f_struct_any"] = int(rec["f_legal"] or rec["f_guidance"] or rec["f_dilution"])
            rec["date"] = d; rec["brk40"] = int(fwd40 < -0.10); rec["brk20"] = int(fwd20 < -0.08)
            recs.append(rec)

    df = pd.DataFrame(recs).dropna(subset=["gap", "vol_20d", "ret_252d"])
    print(f"\n[{len(df)} idiosyncratic gap events (COVID excluded), break40 rate {df['brk40'].mean()*100:.0f}%]")
    print("=== event-type flag rates: break vs bounce (label=brk40) ===")
    for k in ["f_downgrade", "f_legal", "f_guidance", "f_miss", "f_analyst_cut", "f_dilution", "f_struct_any", "f_upgrade"]:
        b1 = df[df.brk40 == 1][k].mean(); b0 = df[df.brk40 == 0][k].mean()
        print(f"  {k:<14} BREAK {b1*100:4.0f}%  BOUNCE {b0*100:4.0f}%  lift {b1/max(b0,1e-9):.2f}x")

    for label in ["brk40", "brk20"]:
        print(f"\n=== OOS AUC (label={label}, idiosyncratic, de-noised) ===")
        for name, feats in [("PRICE", PRICE_FEATS), ("NEWS", NEWS_FEATS), ("PRICE+NEWS", PRICE_FEATS + NEWS_FEATS)]:
            auc, prec, base, n = auc_cv(df, feats, label)
            if auc is None:
                print(f"  {name:<11} insufficient"); continue
            print(f"  {name:<11} AUC {auc:.3f}  top-15% exit precision {prec*100:.0f}% (base {base*100:.0f}%, {prec/base:.1f}x)  n={n}")
    print(f"\n[total {time.time()-t0:.0f}s]")
