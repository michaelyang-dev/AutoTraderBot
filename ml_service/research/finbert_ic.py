"""
FinBERT semantic sentiment as a CROSS-SECTIONAL return signal (2017-2019 clean).
Score unique headlines once (cached), per-stock trailing-30d mean FinBERT tone at
each 20d rebalance, IC vs forward 20d mkt-rel return + quintile spread + momentum
redundancy. Removes the 'keyword-crude / didn't try FinBERT cross-sectionally' gap.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re, time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from main_production_backtest import FastBacktester

NOISE = re.compile(r"biggest mover|stocks moving|stocks that hit|52-?week|mid-?day|"
    r"premarket|pre-market|after hours|after-hours|\bgainers?\b|\blosers?\b|"
    r"movers from|moving in|watch list|unusual options|options activity|"
    r"here's what|things to know|market update|stocks to watch", re.I)


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


@torch.no_grad()
def score_unique(titles, t0):
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert"); model.eval()
    id2 = model.config.id2label
    pi = [k for k, v in id2.items() if v == "positive"][0]
    ni = [k for k, v in id2.items() if v == "negative"][0]
    out = np.empty(len(titles))
    bs = 64
    for i in range(0, len(titles), bs):
        enc = tok(titles[i:i+bs], padding=True, truncation=True, max_length=64, return_tensors="pt")
        p = torch.softmax(model(**enc).logits, dim=1).numpy()
        out[i:i+bs] = p[:, pi] - p[:, ni]
        if i % 12800 == 0:
            print(f"   scored {i}/{len(titles)} ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    news = pd.read_parquet("research/_fnspid_headlines.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news = news[~news["title"].str.contains(NOISE)]
    news = news[(news["date"] >= "2016-12-01") & (news["date"] <= "2019-12-31")].copy()
    cache = "research/_finbert_titlecache.parquet"
    uniq = news["title"].drop_duplicates().tolist()
    print(f"[{len(news)} headlines 2017-2019, {len(uniq)} unique to score]", flush=True)
    if os.path.exists(cache):
        sc = pd.read_parquet(cache); m = dict(zip(sc["title"], sc["tone"]))
        print(f"[loaded cache {len(m)}]")
    else:
        tones = score_unique(uniq, t0)
        pd.DataFrame({"title": uniq, "tone": tones}).to_parquet(cache)
        m = dict(zip(uniq, tones))
    news["tone"] = news["title"].map(m)
    daily = news.groupby(["symbol", "date"]).agg(n=("tone", "size"), tone=("tone", "mean")).reset_index()
    dn = {s: g.set_index("date") for s, g in daily.groupby("symbol")}
    print(f"[scoring done {time.time()-t0:.0f}s; running IC]", flush=True)

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    dates = [d for d in ai if pd.Timestamp("2017-01-01") <= d <= pd.Timestamp("2019-11-15")]
    rebal = list(range(0, len(dates), 20))

    ics, spreads, momcorr = [], [], []
    for k in rebal[:-1]:
        d = dates[k]; gi = di[d]; rows = []
        ret252 = bt.uni.get_feature_map(d, "ret_252d")
        for sym in bt._get_sp1500(d):
            if sym not in dn:
                continue
            g = dn[sym]; w = g[(g.index >= d - pd.Timedelta(days=30)) & (g.index <= d)]
            if len(w) == 0 or sym not in dr.columns:
                continue
            seg = dr[sym].iloc[gi+1:gi+21]
            if seg.isna().all():
                continue
            tone = (w["tone"] * w["n"]).sum() / max(w["n"].sum(), 1)
            fwd = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+21].sum()
            rows.append((tone, ret252.get(sym, np.nan), fwd))
        if len(rows) < 30:
            continue
        a = pd.DataFrame(rows, columns=["tone", "mom", "fwd"]).dropna(subset=["fwd"])
        if a["tone"].nunique() > 3:
            ics.append(spearmanr(a["tone"], a["fwd"]).statistic)
            am = a.dropna(subset=["mom"])
            if len(am) > 20:
                momcorr.append(spearmanr(am["tone"], am["mom"]).statistic)
            try:
                a["q"] = pd.qcut(a["tone"].rank(method="first"), 5, labels=False)
                spreads.append(a[a.q == 4]["fwd"].mean() - a[a.q == 0]["fwd"].mean())
            except Exception:
                pass
    ics = np.array(ics); sp = np.array(spreads)
    print(f"\n=== FinBERT cross-sectional IC (fwd20 mkt-rel, {len(ics)} periods, 2017-2019) ===")
    print(f"  IC mean {ics.mean():+.4f}  t-stat {ics.mean()/(ics.std()/np.sqrt(len(ics))+1e-9):+.2f}")
    print(f"  top-bottom quintile fwd20 spread mean {sp.mean()*100:+.2f}%  t {sp.mean()/(sp.std()/np.sqrt(len(sp))+1e-9):+.2f}")
    print(f"  corr(FinBERT tone, momentum) {np.mean(momcorr):+.3f}")
    print(f"\n[total {time.time()-t0:.0f}s]")
