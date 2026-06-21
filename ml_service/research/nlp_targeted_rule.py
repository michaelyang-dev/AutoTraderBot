"""
Targeted high-conviction exit rule: the break signal lives in SPECIFIC event types
(legal/SEC, guidance-cut), not generic sentiment. Quantify directly: among gap-down
events, does the presence of a LEGAL or GUIDANCE headline raise the break rate and
push forward returns negative enough to justify a targeted exit? Report P(break),
forward 40d mkt-rel return, and counts (how rare) — vs the no-structural-news group.
This is the honest test of whether news adds a usable (if narrow) exit rule.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import re, time
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester

GAP = 0.10
NOISE = re.compile(r"biggest mover|stocks moving|stocks that hit|52-?week|mid-?day|"
    r"premarket|pre-market|after hours|after-hours|what you need to know|\bgainers?\b|"
    r"\blosers?\b|movers from|moving in|watch list|unusual options|options activity|"
    r"here's what|things to know|market update|stocks to watch", re.I)
LEGAL = re.compile(r"\bsec\b|s\.e\.c|investigat|probe|lawsuit|sued|subpoena|fraud|"
    r"misconduct|whistleblow|litigation|class action|doj|restat", re.I)
GUID = re.compile(r"cuts? guidance|lowers? guidance|lowered guidance|withdraw|"
    r"slashes? (guidance|outlook|forecast)|profit warning|warns|going concern", re.I)


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


if __name__ == "__main__":
    t0 = time.time()
    news = pd.read_parquet("research/_fnspid_headlines.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news = news[~news["title"].str.contains(NOISE)]
    news["legal"] = news["title"].str.contains(LEGAL).astype(int)
    news["guid"] = news["title"].str.contains(GUID).astype(int)
    news = news.sort_values("date")
    by_sym = {s: g for s, g in news.groupby("symbol")}

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    nmax = news["date"].max()
    dates = [d for d in ai if pd.Timestamp("2016-06-01") <= d <= (nmax - pd.Timedelta(days=70))]
    print(f"[loaded {time.time()-t0:.0f}s]")

    recs = []
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
            recs.append({"date": d, "sym": sym, "legal": int(w["legal"].sum() > 0),
                         "guid": int(w["guid"].sum() > 0), "fwd": fwd,
                         "brk": int(fwd < -0.10)})
    df = pd.DataFrame(recs)
    df["struct"] = ((df.legal == 1) | (df.guid == 1)).astype(int)
    n = len(df)
    print(f"\n[{n} gap-down events w/ news, base break rate {df.brk.mean()*100:.0f}%, "
          f"base fwd40 {df.fwd.mean()*100:+.1f}%]\n")

    def grp(name, mask):
        s = df[mask]
        if len(s) == 0:
            print(f"  {name:<26} n=0"); return
        print(f"  {name:<26} n={len(s):3d} ({len(s)/n*100:2.0f}%)  P(break) {s.brk.mean()*100:3.0f}%  "
              f"fwd40 mean {s.fwd.mean()*100:+5.1f}%  median {s.fwd.median()*100:+5.1f}%  hit {(s.fwd>0).mean()*100:3.0f}%")

    print("=== conditional break rate & forward return by news event type ===")
    grp("LEGAL headline", df.legal == 1)
    grp("GUIDANCE headline", df.guid == 1)
    grp("LEGAL or GUIDANCE", df.struct == 1)
    grp("neither (no struct news)", df.struct == 0)
    print()
    # the actual rule payoff: if we EXIT struct-news gap-downs vs HOLD them
    s = df[df.struct == 1]; h = df[df.struct == 0]
    print(f"  RULE: exit on legal/guidance gap-down avoids fwd40 {s.fwd.mean()*100:+.1f}% "
          f"(vs holding non-struct {h.fwd.mean()*100:+.1f}%)")
    print(f"  events/yr ~ {len(s)/4.0:.0f} across SP1500 universe; in top-5 book far rarer")
    print(f"\n[total {time.time()-t0:.0f}s]")
