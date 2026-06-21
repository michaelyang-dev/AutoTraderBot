"""
TRUE OOS test (2020-2023) of the news-momentum sleeve, identical construction to
the in-sample version. If Sharpe stays >~0.5 at realistic 15bps/side same-day-close
-> robust, worth considering infra. If it collapses -> regime-luck, not worth it.
Also reports the latency check (1-day late) on OOS.
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
HOLD = 5


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def sim(ent, dr, ai, sgi, egi, delay, cps):
    rl, dts = [], []
    for gi in range(sgi, egi):
        rets = []
        for k in range(1 + delay, HOLD + 1 + delay):
            for sym in ent.get(gi - k, []):
                if sym not in dr.columns:
                    continue
                rr = dr[sym].iloc[gi]
                if pd.isna(rr):
                    continue
                kk = k - delay
                c = cps if kk == 1 else (cps if kk == HOLD else 0)
                rets.append(rr - dr["SPY"].iloc[gi] - c)
        rl.append(np.mean(rets) if rets else 0.0); dts.append(ai[gi])
    s = pd.Series(rl, index=pd.DatetimeIndex(dts))
    eq = (1 + s).cumprod(); yrs = (s.index[-1]-s.index[0]).days/365.25
    return eq.iloc[-1]**(1/yrs)-1, (s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0), ((eq-eq.cummax())/eq.cummax()).min()


if __name__ == "__main__":
    t0 = time.time()
    news = pd.read_parquet("research/_fnspid_oos.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news = news[~news["title"].str.contains(NOISE)].copy()
    news["pos"] = news["title"].str.contains(POS).astype(int)
    news["neg"] = news["title"].str.contains(NEG).astype(int)
    daily = news.groupby(["symbol", "date"]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    daily["tone"] = (daily["pos"] - daily["neg"]) / daily["n"]
    daily = daily.sort_values(["symbol", "date"])
    daily["att_base"] = daily.groupby("symbol")["n"].transform(lambda s: s.rolling(20, min_periods=3).mean().shift(1))
    daily["att"] = daily["n"] / daily["att_base"].replace(0, np.nan)
    sig = daily[(daily["att"].between(3, 6)) & (daily["tone"] >= 0.5)].copy()
    print(f"[OOS news {news['date'].min().date()}..{news['date'].max().date()}, {len(sig)} strongPOS signals]")

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    dr = bt.prices.pct_change(); ai = list(bt.prices.index); di = {d: i for i, d in enumerate(ai)}
    ent = {}
    for r in sig.itertuples():
        gi = di.get(r.date)
        if gi is not None and gi + HOLD + 2 < len(ai):
            ent.setdefault(gi, []).append(r.symbol)
    sgi = di[[d for d in ai if d >= max(pd.Timestamp("2020-07-01"), sig["date"].min())][0]]
    egi = di[[d for d in ai if d <= sig["date"].max()][-1]]
    print(f"[loaded {time.time()-t0:.0f}s; OOS test window {ai[sgi].date()}..{ai[egi].date()}]\n")

    print("=== OOS 2020-2023 news-momentum (strongPOS, same-day close) ===")
    for cps in (0.0007, 0.0015, 0.0025):
        c, sh, dd = sim(ent, dr, ai, sgi, egi, 0, cps)
        print(f"  {cps*1e4:2.0f}bps/side: CAGR {c*100:6.1f}%  Sharpe {sh:5.2f}  MaxDD {dd*100:6.1f}%")
    c, sh, dd = sim(ent, dr, ai, sgi, egi, 1, 0.0007)
    print(f"  1-day-late 7bps:  CAGR {c*100:6.1f}%  Sharpe {sh:5.2f}  (latency check)")
    print(f"\n[total {time.time()-t0:.0f}s]")
