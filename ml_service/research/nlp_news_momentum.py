"""
Push on the one real signal: news drift is real but sub-cost ON AVERAGE. Is there
a high-magnitude SUBSET that clears costs? Condition forward 1/3/5d market-relative
drift on attention-spike x tone strength. If big-attention strong-negative news ->
continuation of -1%+ over a few days, that's a tradeable bad-news-momentum (avoid/
trim) signal; if it mean-reverts, dead. Also check the positive extreme.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import re, time
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester

NOISE = re.compile(r"biggest mover|stocks moving|stocks that hit|52-?week|mid-?day|"
    r"premarket|pre-market|after hours|after-hours|\bgainers?\b|\blosers?\b|"
    r"movers from|moving in|watch list|unusual options|options activity|"
    r"here's what|things to know|market update|stocks to watch", re.I)
POS = re.compile(r"\b(beats?|tops?|raises?|raised|upgrad|outperform|buy rating|overweight|"
    r"price target rais|record|strong|surge|jumps?|soar|rally|wins?|approval|approved|launch|upbeat|growth|gains?)\b", re.I)
NEG = re.compile(r"\b(miss(es|ed)?|downgrad|cuts?|lowers?|lowered|sec\b|investigat|probe|"
    r"lawsuit|sued|fraud|warns?|warning|plunge|plummet|falls?|drops?|weak|disappoint|slash|halt|recall|resign|layoff|loss|losses|decline)\b", re.I)


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


if __name__ == "__main__":
    t0 = time.time()
    news = pd.read_parquet("research/_fnspid_headlines.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news = news[~news["title"].str.contains(NOISE)].copy()
    news["pos"] = news["title"].str.contains(POS).astype(int)
    news["neg"] = news["title"].str.contains(NEG).astype(int)
    daily = news.groupby(["symbol", "date"]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    daily["tone"] = (daily["pos"] - daily["neg"]) / daily["n"]   # normalized tone
    daily = daily.sort_values(["symbol", "date"])
    daily["att_base"] = daily.groupby("symbol")["n"].transform(lambda s: s.rolling(20, min_periods=3).mean().shift(1))
    daily["att"] = daily["n"] / daily["att_base"].replace(0, np.nan)

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    print(f"[loaded {time.time()-t0:.0f}s]")

    rows = []
    for sym, g in daily.groupby("symbol"):
        if sym not in dr.columns:
            continue
        s = dr[sym]
        for r in g.itertuples():
            d = r.date
            gi = di.get(d)
            if gi is None:
                fut = [x for x in ai if x > d][:1]
                if not fut:
                    continue
                gi = di[fut[0]]
            if gi + 5 >= len(ai):
                continue
            rec = {"tone": r.tone, "att": r.att, "n": r.n}
            for h in (1, 3, 5):
                seg = s.iloc[gi+1:gi+1+h]
                if not seg.isna().all():
                    rec[f"f{h}"] = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+1+h].sum()
            rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["att", "f5"])
    print(f"[{len(df)} obs]\n")

    print("=== forward 5d MARKET-RELATIVE drift by attention x tone ===")
    df["att_b"] = pd.cut(df["att"], [0, 1.5, 3, 6, 1e9], labels=["low", "med", "high(3-6x)", "extreme(>6x)"])
    df["tone_b"] = pd.cut(df["tone"], [-1.01, -0.5, -0.01, 0.01, 0.5, 1.01],
                          labels=["strongNEG", "negish", "neutral", "posish", "strongPOS"])
    for ab in ["high(3-6x)", "extreme(>6x)"]:
        print(f"\n  attention {ab}:")
        sub = df[df["att_b"] == ab]
        for tb, gg in sub.groupby("tone_b", observed=True):
            if len(gg) < 20:
                continue
            print(f"     {str(tb):<10} f1 {gg['f1'].mean()*100:+5.2f}%  f3 {gg['f3'].mean()*100:+5.2f}%  "
                  f"f5 {gg['f5'].mean()*100:+5.2f}%  n={len(gg)}")
    # the tradeable extreme: extreme attention + strong negative
    ext = df[(df["att"] > 6) & (df["tone"] <= -0.5)]
    print(f"\n  EXTREME attention(>6x) + strongNEG: f1 {ext['f1'].mean()*100:+.2f}%  "
          f"f3 {ext['f3'].mean()*100:+.2f}%  f5 {ext['f5'].mean()*100:+.2f}%  hit<0 {(ext['f5']<0).mean()*100:.0f}%  n={len(ext)}")
    print(f"  (vs ~28bps round-trip cost; tradeable only if drift >> cost & consistent)")
    print(f"\n[total {time.time()-t0:.0f}s]")
