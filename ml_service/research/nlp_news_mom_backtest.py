"""
Proper daily-NAV backtest of the news-momentum sleeve + orthogonality to the
existing strategy. Each day: open positions in stocks with high-attention(3-6x)
strong-tone(|t|>=.5) news; hold HOLD days (overlapping cohorts, equal weight across
active positions). Daily portfolio return net of cost. Report CAGR/Sharpe/MaxDD,
correlation to SPY and to the deployed momentum book, for long-only and long-short.
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
CPS = 0.0007  # cost per side


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def metrics(daily, idx):
    s = pd.Series(daily, index=idx)
    eq = (1 + s).cumprod(); yrs = (idx[-1]-idx[0]).days/365.25
    return (eq.iloc[-1]**(1/yrs)-1, s.mean()/s.std()*np.sqrt(252) if s.std()>0 else 0,
            ((eq-eq.cummax())/eq.cummax()).min(), s)


if __name__ == "__main__":
    t0 = time.time()
    news = pd.read_parquet("research/_fnspid_headlines.parquet")
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news = news[~news["title"].str.contains(NOISE)].copy()
    news["pos"] = news["title"].str.contains(POS).astype(int)
    news["neg"] = news["title"].str.contains(NEG).astype(int)
    daily = news.groupby(["symbol", "date"]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    daily["tone"] = (daily["pos"] - daily["neg"]) / daily["n"]
    daily = daily.sort_values(["symbol", "date"])
    daily["att_base"] = daily.groupby("symbol")["n"].transform(lambda s: s.rolling(20, min_periods=3).mean().shift(1))
    daily["att"] = daily["n"] / daily["att_base"].replace(0, np.nan)
    sig = daily[(daily["att"].between(3, 6)) & (daily["tone"].abs() >= 0.5)].copy()
    sig["side"] = np.sign(sig["tone"])

    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    prices = bt.prices; dr = prices.pct_change()
    ai = list(prices.index); di = {d: i for i, d in enumerate(ai)}
    print(f"[loaded {time.time()-t0:.0f}s; {len(sig)} signal-events]")

    # map signals to entry trading-day index
    entries = []  # (entry_gi, sym, side)
    for r in sig.itertuples():
        gi = di.get(r.date)
        if gi is None or gi + HOLD + 1 >= len(ai):
            continue
        entries.append((gi, r.symbol, int(r.side)))
    ent_by_day = {}
    for gi, sym, side in entries:
        ent_by_day.setdefault(gi, []).append((sym, side))

    # simulate: active positions = those opened in last HOLD days; daily ret = mean(side*stock_ret)
    start_gi = di[[d for d in ai if d >= pd.Timestamp("2016-06-01")][0]]
    end_gi = di[[d for d in ai if d <= sig["date"].max()][-1]]
    rl_ls, rl_lo, dates_out = [], [], []
    for gi in range(start_gi, end_gi):
        active = []
        for k in range(1, HOLD + 1):
            for sym, side in ent_by_day.get(gi - k, []):
                active.append((sym, side, k))  # k = days held
        if not active:
            rl_ls.append(0.0); rl_lo.append(0.0); dates_out.append(ai[gi+1]); continue
        rets_ls, rets_lo = [], []
        spy_ret = dr["SPY"].iloc[gi]
        spy_ret = 0.0 if pd.isna(spy_ret) else spy_ret
        for sym, side, k in active:
            if sym not in dr.columns:
                continue
            rr = dr[sym].iloc[gi]  # position entered at gi-k earns day gi return (k-th day held)
            if pd.isna(rr):
                continue
            rel = rr - spy_ret  # MARKET-RELATIVE (apples-to-apples w/ event study)
            c = CPS if k == 1 else (CPS if k == HOLD else 0)  # cost at entry & exit day
            rets_ls.append(side * rel - c)
            if side > 0:
                rets_lo.append(rel - c)
        rl_ls.append(np.mean(rets_ls) if rets_ls else 0.0)
        rl_lo.append(np.mean(rets_lo) if rets_lo else 0.0)
        dates_out.append(ai[gi])
    idx = pd.DatetimeIndex(dates_out)
    spy = dr["SPY"].reindex(idx).fillna(0)

    print(f"\n=== news-momentum sleeve daily-NAV backtest ({idx[0].date()}..{idx[-1].date()}) ===")
    for nm, rl in [("LONG-SHORT", rl_ls), ("LONG-ONLY (strongPOS)", rl_lo)]:
        cagr, sh, mdd, s = metrics(rl, idx)
        corr_spy = pd.Series(rl, index=idx).corr(spy)
        print(f"  {nm:<22} CAGR {cagr*100:6.1f}%  Sharpe {sh:.2f}  MaxDD {mdd*100:6.1f}%  corr(SPY) {corr_spy:+.2f}")

    # orthogonality to deployed momentum book
    BASE = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0, top_n=5,
                cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
                bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10})
    r = bt.run("2016-06-01", str(idx[-1].date()), BASE)
    mom = r["daily_values"].pct_change().reindex(idx).fillna(0)
    print(f"\n  corr(news-mom LONG-SHORT, deployed momentum book) = {pd.Series(rl_ls, index=idx).corr(mom):+.2f}")
    print(f"  corr(news-mom LONG-ONLY,  deployed momentum book) = {pd.Series(rl_lo, index=idx).corr(mom):+.2f}")
    print(f"  -> low corr => genuinely additive/orthogonal sleeve")
    print(f"\n[total {time.time()-t0:.0f}s]")
