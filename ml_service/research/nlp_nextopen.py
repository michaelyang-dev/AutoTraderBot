"""
Realistic-execution test: how much of the news-momentum alpha survives NEXT-DAY-OPEN
entry (you detect overnight news, trade the open) vs SAME-DAY-CLOSE (needs intraday
detection+execution)? Uses FMP open+close (2016-2026, 719 tickers). Per strongPOS
signal, market-relative HOLD-day return for each entry timing, net of cost. If
next-open >> 0 -> deployable with a morning layer; if the alpha is the overnight gap
(close[E]->open[E+1]) only -> HFT-only. Runs on whatever news files are present.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import re
import numpy as np
import pandas as pd

NOISE = re.compile(r"biggest mover|stocks moving|stocks that hit|52-?week|mid-?day|"
    r"premarket|pre-market|after hours|after-hours|\bgainers?\b|\blosers?\b|"
    r"movers from|moving in|watch list|unusual options|options activity|"
    r"here's what|things to know|market update|stocks to watch", re.I)
POS = re.compile(r"\b(beats?|tops?|raises?|raised|upgrad|outperform|buy rating|overweight|"
    r"price target rais|record|strong|surge|jumps?|soar|rally|wins?|approval|approved|launch|upbeat|growth|gains?)\b", re.I)
NEG = re.compile(r"\b(miss(es|ed)?|downgrad|cuts?|lowers?|lowered|sec\b|investigat|probe|"
    r"lawsuit|sued|fraud|warns?|warning|plunge|plummet|falls?|drops?|weak|disappoint|slash|halt|recall|resign|layoff|loss|losses|decline)\b", re.I)
HOLD = 5


def build_sig(path):
    news = pd.read_parquet(path); news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news = news[~news["title"].str.contains(NOISE)].copy()
    news["pos"] = news["title"].str.contains(POS).astype(int)
    news["neg"] = news["title"].str.contains(NEG).astype(int)
    d = news.groupby(["symbol", "date"]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    d["tone"] = (d["pos"] - d["neg"]) / d["n"]
    d = d.sort_values(["symbol", "date"])
    d["ab"] = d.groupby("symbol")["n"].transform(lambda s: s.rolling(20, min_periods=3).mean().shift(1))
    d["att"] = d["n"] / d["ab"].replace(0, np.nan)
    return d[(d["att"].between(3, 6)) & (d["tone"] >= 0.5)]


op = pd.read_parquet("data/cached_open_prices.parquet"); op.index = pd.to_datetime(op.index)
cl = pd.read_parquet("data/cached_close_prices.parquet"); cl.index = pd.to_datetime(cl.index)
idx = list(cl.index); pos = {d: i for i, d in enumerate(idx)}

files = [("IN-SAMPLE 2016-2020", "research/_fnspid_headlines.parquet")]
if os.path.exists("research/_fnspid_oos.parquet"):
    files.append(("OOS 2020-2023", "research/_fnspid_oos.parquet"))


def perf(rets, cps_rt):
    r = np.array(rets) - cps_rt
    return r.mean() * 100, (r.mean() / (r.std() + 1e-9)), len(r)


for label, path in files:
    sig = build_sig(path)
    sdc, nop = [], []   # same-day-close, next-open  (market-relative HOLD-day returns)
    for r in sig.itertuples():
        d = r.date
        i = pos.get(d)
        if i is None:
            fut = [x for x in idx if x >= d]
            if not fut:
                continue
            i = pos[fut[0]]
        if i + HOLD + 1 >= len(idx) or r.symbol not in cl.columns:
            continue
        s = r.symbol
        c0, cH = cl[s].iloc[i], cl[s].iloc[i+HOLD]
        o1 = op[s].iloc[i+1]
        sc0, scH, so1 = cl["SPY"].iloc[i], cl["SPY"].iloc[i+HOLD], op["SPY"].iloc[i+1]
        if any(pd.isna(x) or x <= 0 for x in [c0, cH, o1, sc0, scH, so1]):
            continue
        sdc.append((cH/c0 - 1) - (scH/sc0 - 1))
        nop.append((cH/o1 - 1) - (scH/so1 - 1))
    print(f"\n=== {label}: {len(sdc)} strongPOS signals ===")
    for nm, rr in [("same-day CLOSE entry", sdc), ("next-day OPEN entry", nop)]:
        for cps in (0.0014, 0.0030):  # round-trip 14 / 30 bps
            m, sh, n = perf(rr, cps)
            print(f"  {nm:<22} RT{cps*1e4:2.0f}bps: per-trade {m:+.2f}%  per-trade Sharpe {sh:+.3f}  n={n}")
