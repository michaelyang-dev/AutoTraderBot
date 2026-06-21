"""
FinBERT (semantic) on the break task — removes the 'crude keywords' objection.
Score event-window headlines with ProsusAI/finbert (tone = P(pos)-P(neg)), then
test whether FinBERT-negative gap-downs (overall and within legal/guidance events)
break more / have negative forward returns. If even semantic sentiment can't find
an exitable population, the exit thread is dead for real.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from main_production_backtest import FastBacktester

GAP = 0.10
NOISE = re.compile(r"biggest mover|stocks moving|stocks that hit|52-?week|mid-?day|"
    r"premarket|pre-market|after hours|after-hours|\bgainers?\b|\blosers?\b|"
    r"movers from|moving in|watch list|unusual options|options activity|"
    r"here's what|things to know|market update|stocks to watch", re.I)
LEGAL = re.compile(r"\bsec\b|investigat|probe|lawsuit|sued|fraud|misconduct|litigation|doj|restat", re.I)
GUID = re.compile(r"cuts? guidance|lowers? guidance|withdraw|slash|profit warning|warns|going concern", re.I)


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


class FinBERT:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        # label order for ProsusAI/finbert: 0=positive,1=negative,2=neutral
        self.id = self.model.config.id2label

    @torch.no_grad()
    def tone(self, titles, bs=64):
        out = []
        for i in range(0, len(titles), bs):
            b = titles[i:i+bs]
            enc = self.tok(b, padding=True, truncation=True, max_length=64, return_tensors="pt")
            p = torch.softmax(self.model(**enc).logits, dim=1).numpy()
            # map by label name
            pos = np.array([p[j][k] for j in range(len(b)) for k, v in self.id.items() if v == "positive"]).reshape(-1)
            neg = np.array([p[j][k] for j in range(len(b)) for k, v in self.id.items() if v == "negative"]).reshape(-1)
            out.extend((pos - neg).tolist())
        return np.array(out)


if __name__ == "__main__":
    t0 = time.time()
    news = pd.read_parquet("research/_fnspid_headlines.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news = news[~news["title"].str.contains(NOISE)].copy()
    news = news.sort_values("date")
    by_sym = {s: g for s, g in news.groupby("symbol")}

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    nmax = news["date"].max()
    dates = [d for d in ai if pd.Timestamp("2016-06-01") <= d <= (nmax - pd.Timedelta(days=70))]

    # collect events + their window headlines
    events = []
    all_titles = []
    for d in dates:
        gi = di[d]
        if gi + 40 >= len(ai):
            continue
        sr = spy_r.loc[d]
        if pd.isna(sr):
            continue
        row = dr.loc[d]
        for sym in bt._get_sp1500(d):
            g = row.get(sym)
            if g is None or pd.isna(g) or (g - sr) >= -GAP or sym not in by_sym:
                continue
            seg = dr[sym].iloc[gi+1:gi+41]
            if seg.isna().all():
                continue
            fwd = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+41].sum()
            gw = by_sym[sym]
            w = gw[(gw["date"] >= d - pd.Timedelta(days=2)) & (gw["date"] <= d + pd.Timedelta(days=3))]
            titles = w["title"].tolist()
            if not titles:
                continue
            i0 = len(all_titles); all_titles.extend(titles)
            legal = int(w["title"].str.contains(LEGAL).any())
            guid = int(w["title"].str.contains(GUID).any())
            events.append({"sym": sym, "date": d, "fwd": fwd, "brk": int(fwd < -0.10),
                           "i0": i0, "i1": len(all_titles), "legal": legal, "guid": guid})
    print(f"[{len(events)} events, {len(all_titles)} window headlines to score]", flush=True)

    fb = FinBERT()
    print(f"[FinBERT loaded {time.time()-t0:.0f}s; scoring...]", flush=True)
    tones = fb.tone(all_titles)
    print(f"[scored in {time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(events)
    df["fb_mean"] = [tones[e["i0"]:e["i1"]].mean() for e in events]
    df["fb_min"] = [tones[e["i0"]:e["i1"]].min() for e in events]
    df["struct"] = ((df.legal == 1) | (df.guid == 1)).astype(int)

    print(f"\n[base break {df.brk.mean()*100:.0f}%, fwd40 {df.fwd.mean()*100:+.1f}%]")
    print("=== forward return by FinBERT tone quintile (most-neg -> most-pos) ===")
    df["q"] = pd.qcut(df["fb_mean"], 5, labels=["Q1neg","Q2","Q3","Q4","Q5pos"])
    for q, g in df.groupby("q"):
        print(f"  {q:<7} fwd40 mean {g.fwd.mean()*100:+5.1f}%  median {g.fwd.median()*100:+5.1f}%  "
              f"P(break) {g.brk.mean()*100:3.0f}%  hit {(g.fwd>0).mean()*100:3.0f}%  n={len(g)}")
    print(f"\n  IC(FinBERT mean tone, fwd40) = {df['fb_mean'].corr(df['fwd'], method='spearman'):+.3f}")
    print(f"  IC(FinBERT MIN tone, fwd40)  = {df['fb_min'].corr(df['fwd'], method='spearman'):+.3f}")
    # most-negative decile: do they finally break?
    cut = df["fb_min"].quantile(0.10); mneg = df[df.fb_min <= cut]
    print(f"\n  most-negative-headline decile (fb_min): P(break) {mneg.brk.mean()*100:.0f}%  "
          f"fwd40 {mneg.fwd.mean()*100:+.1f}%  n={len(mneg)}")
    # struct + finbert-negative
    sn = df[(df.struct == 1) & (df.fb_mean < df.fb_mean.median())]
    print(f"  legal/guidance AND FinBERT-negative: P(break) {sn.brk.mean()*100:.0f}%  "
          f"fwd40 {sn.fwd.mean()*100:+.1f}%  n={len(sn)}")
    df.to_parquet("research/_finbert_events.parquet")
    print(f"\n[total {time.time()-t0:.0f}s]")
