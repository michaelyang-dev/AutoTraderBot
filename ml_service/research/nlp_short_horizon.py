"""
Short-horizon news drift — the angle I hadn't tested. News effects are strongest
in the first days; a real 1-5d signal could be washed out by the 20d horizon.
For each stock-day WITH news (2016-2020), net keyword tone + attention (n articles),
measure forward 1/3/5/10d MARKET-RELATIVE return (from t+1, no same-day look-ahead).
Bucket by tone sign and by attention spike. If pos-tone -> short-horizon up-drift
and neg-tone -> down-drift (monotone, sized), there's a tradeable (if high-turnover)
signal momentum at 20d misses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import re, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from main_production_backtest import FastBacktester

NOISE = re.compile(r"biggest mover|stocks moving|stocks that hit|52-?week|mid-?day|"
    r"premarket|pre-market|after hours|after-hours|\bgainers?\b|\blosers?\b|"
    r"movers from|moving in|watch list|unusual options|options activity|"
    r"here's what|things to know|market update|stocks to watch", re.I)
POS = re.compile(r"\b(beats?|tops?|raises?|raised|upgrad|outperform|buy rating|overweight|"
    r"price target rais|record|strong|surge|jumps?|soar|rally|wins?|approval|approved|"
    r"launch|upbeat|growth|gains?)\b", re.I)
NEG = re.compile(r"\b(miss(es|ed)?|downgrad|cuts?|lowers?|lowered|sec\b|investigat|probe|"
    r"lawsuit|sued|fraud|warns?|warning|plunge|plummet|falls?|drops?|weak|disappoint|"
    r"slash|halt|recall|resign|layoff|loss|losses|decline)\b", re.I)
HOR = [1, 3, 5, 10]


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
    daily["tone"] = daily["pos"] - daily["neg"]
    # attention spike: n vs the stock's trailing 60-day mean news count
    daily = daily.sort_values(["symbol", "date"])
    daily["att_base"] = daily.groupby("symbol")["n"].transform(lambda s: s.rolling(20, min_periods=3).mean().shift(1))
    daily["att_spike"] = daily["n"] / daily["att_base"].replace(0, np.nan)
    print(f"[{len(daily)} stock-news-days]")

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    aidate = set(ai)
    # map each news day to the trading-day index (next trading day if not a trading day)
    print(f"[loaded {time.time()-t0:.0f}s]")

    rows = []
    cumret = {}
    for sym, g in daily.groupby("symbol"):
        if sym not in dr.columns:
            continue
        s = dr[sym]
        for r in g.itertuples():
            d = r.date
            # find trading index at or after d
            if d in di:
                gi = di[d]
            else:
                fut = [x for x in ai if x > d][:1]
                if not fut:
                    continue
                gi = di[fut[0]]
            rec = {"tone": r.tone, "att": r.att_spike, "n": r.n}
            ok = False
            for h in HOR:
                if gi + h < len(ai):
                    seg = s.iloc[gi+1:gi+1+h]; mseg = spy_r.iloc[gi+1:gi+1+h]
                    if not seg.isna().all():
                        rec[f"f{h}"] = seg.fillna(0).sum() - mseg.sum(); ok = True
            if ok:
                rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"[{len(df)} news-day observations with forward returns]\n")

    print("=== mean MARKET-RELATIVE forward return by news tone sign ===")
    df["sign"] = np.sign(df["tone"]).map({1: "POS", -1: "NEG", 0: "neutral"})
    for h in HOR:
        col = f"f{h}"
        sub = df.dropna(subset=[col])
        g = sub.groupby("sign")[col].mean()
        spread = (g.get("POS", np.nan) - g.get("NEG", np.nan)) * 100
        ic = spearmanr(sub["tone"], sub[col]).statistic
        print(f"  +{h:2d}d:  POS {g.get('POS',np.nan)*100:+.2f}%  NEG {g.get('NEG',np.nan)*100:+.2f}%  "
              f"neutral {g.get('neutral',np.nan)*100:+.2f}%  | POS-NEG spread {spread:+.2f}%  IC {ic:+.3f}")

    print("\n=== high attention-spike (att>3x baseline) forward return + |move| ===")
    hi = df[df["att"] > 3].dropna(subset=["f5"]); lo = df[(df["att"] <= 1.5)].dropna(subset=["f5"])
    print(f"  att>3x: f5 mean {hi['f5'].mean()*100:+.2f}%  abs {hi['f5'].abs().mean()*100:.2f}%  n={len(hi)}")
    print(f"  att<=1.5x: f5 mean {lo['f5'].mean()*100:+.2f}%  abs {lo['f5'].abs().mean()*100:.2f}%  n={len(lo)}")
    print(f"\n[total {time.time()-t0:.0f}s]")
