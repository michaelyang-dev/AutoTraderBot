"""
Does news sentiment predict returns CROSS-SECTIONALLY (a signal/sleeve, not an exit)?
The core 'sentiment alpha' question, now on real ticker-tagged headlines (FNSPID
2016-2020, de-noised). At each 20d rebalance, per stock: trailing-20d net news tone
(rec - brk keyword hits / article) + article count (attention). Compute IC (Spearman
corr with forward 20d mkt-rel return), top-minus-bottom quintile spread, and corr
with momentum (redundancy check). IC>~0.03 and a positive spread -> worth a sleeve.
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
POS = re.compile(r"\b(beats?|tops?|raises?|raised|upgrad|outperform|buy rating|"
    r"overweight|price target rais|record|strong|surge|jumps?|soar|rally|wins?|"
    r"approval|approved|launch|upbeat|growth|profit|gains?)\b", re.I)
NEG = re.compile(r"\b(miss(es|ed)?|downgrad|cuts?|lowers?|lowered|sec\b|investigat|"
    r"probe|lawsuit|sued|fraud|warns?|warning|plunge|plummet|falls?|drops?|"
    r"weak|disappoint|slash|halt|recall|resign|layoff|loss|losses|decline)\b", re.I)


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
    # daily per-stock aggregates
    daily = news.groupby(["symbol", "date"]).agg(
        n=("title", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    daily["tone"] = daily["pos"] - daily["neg"]
    print(f"[news daily aggregates: {len(daily)} stock-days, {daily['symbol'].nunique()} tickers]")

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    nmax = news["date"].max()
    dates = [d for d in ai if pd.Timestamp("2016-06-01") <= d <= (nmax - pd.Timedelta(days=35))]
    rebal = list(range(0, len(dates), 20))
    # index daily news by symbol for window sums
    dn = {s: g.set_index("date") for s, g in daily.groupby("symbol")}
    print(f"[loaded {time.time()-t0:.0f}s]")

    ics_tone, ics_att, ics_mom, spreads, mom_corr = [], [], [], [], []
    for k in rebal[:-1]:
        d = dates[k]; gi = di[d]
        rows = []
        ret252 = bt.uni.get_feature_map(d, "ret_252d")
        for sym in bt._get_sp1500(d):
            if sym not in dn:
                continue
            g = dn[sym]
            w = g[(g.index >= d - pd.Timedelta(days=30)) & (g.index <= d)]
            if len(w) == 0:
                continue
            tone = w["tone"].sum() / max(w["n"].sum(), 1)
            att = w["n"].sum()
            # forward 20d mkt-rel return
            if sym not in dr.columns:
                continue
            seg = dr[sym].iloc[gi+1:gi+21]
            if seg.isna().all():
                continue
            fwd = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+21].sum()
            rows.append((tone, att, ret252.get(sym, np.nan), fwd))
        if len(rows) < 30:
            continue
        a = pd.DataFrame(rows, columns=["tone", "att", "mom", "fwd"]).dropna(subset=["fwd"])
        if a["tone"].nunique() > 3:
            ics_tone.append(spearmanr(a["tone"], a["fwd"]).statistic)
            ics_att.append(spearmanr(a["att"], a["fwd"]).statistic)
            am = a.dropna(subset=["mom"])
            if len(am) > 20:
                ics_mom.append(spearmanr(am["mom"], am["fwd"]).statistic)
                mom_corr.append(spearmanr(am["tone"], am["mom"]).statistic)
            # quintile spread on tone
            try:
                a["q"] = pd.qcut(a["tone"].rank(method="first"), 5, labels=False)
                spreads.append(a[a.q == 4]["fwd"].mean() - a[a.q == 0]["fwd"].mean())
            except Exception:
                pass

    def stat(x):
        x = np.array(x); return x.mean(), x.mean() / (x.std() / np.sqrt(len(x)) + 1e-9), len(x)
    print(f"\n=== news-sentiment cross-sectional IC (fwd 20d mkt-rel, {len(ics_tone)} periods) ===")
    for nm, ic in [("news TONE", ics_tone), ("news ATTENTION (count)", ics_att), ("momentum (ret252) [ref]", ics_mom)]:
        m, t, nN = stat(ic)
        print(f"  IC {nm:<26} mean {m:+.4f}  t-stat {t:+.2f}  (n={nN})")
    sm, st, sn = stat(spreads)
    print(f"\n  top-bottom tone quintile fwd20 spread: mean {sm*100:+.2f}%  t {st:+.2f}")
    mc = np.mean(mom_corr)
    print(f"  corr(news tone, momentum) = {mc:+.3f}  (redundancy w/ momentum)")
    print(f"\n[total {time.time()-t0:.0f}s]")
