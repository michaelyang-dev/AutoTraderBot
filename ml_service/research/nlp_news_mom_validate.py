"""
Validate the news-momentum lead (high-attention x strong-tone -> 5d drift).
Stress tests before believing it:
  (1) sub-period robustness: 2016-2018 vs 2018-2020 (does the drift hold OOS?)
  (2) net-of-cost portfolio: daily long high-att strongPOS / short high-att strongNEG,
      5d hold, ~7bps/side cost -> Sharpe/CAGR + capacity (signals/day)
  (3) is it just beta/momentum? control with market-relative already; report long-only leg too
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
COST = 0.0007  # 7bps/side


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
    daily["tone"] = (daily["pos"] - daily["neg"]) / daily["n"]
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
            gi = di.get(r.date)
            if gi is None or gi + HOLD >= len(ai):
                continue
            if not (3 <= r.att <= 6) or abs(r.tone) < 0.5:
                continue
            seg = s.iloc[gi+1:gi+1+HOLD]
            if seg.isna().all():
                continue
            f5 = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+1+HOLD].sum()
            rows.append({"date": ai[gi], "sym": sym, "side": 1 if r.tone > 0 else -1, "f5": f5})
    ev = pd.DataFrame(rows)
    print(f"[{len(ev)} high-att strong-tone events; {len(ev)/((ev.date.max()-ev.date.min()).days/365.25):.0f}/yr]\n")

    # (1) sub-period robustness
    print("=== sub-period robustness: mean 5d mkt-rel drift (signal = side) ===")
    for lbl, lo, hi in [("2016-2018", "2016-01-01", "2018-06-30"), ("2018-2020", "2018-07-01", "2020-12-31")]:
        s = ev[(ev.date >= lo) & (ev.date <= hi)]
        L = s[s.side == 1]["f5"].mean(); S = s[s.side == -1]["f5"].mean()
        print(f"  {lbl}: strongPOS {L*100:+.2f}% (n={ (s.side==1).sum() })  "
              f"strongNEG {S*100:+.2f}% (n={ (s.side==-1).sum() })  L-S {(L-S)*100:+.2f}%")

    # (2) net-of-cost portfolio: each event = a 5d position; daily-rebalanced equal weight
    # build a daily return series: on each date, avg of (side * f5)/HOLD per active position, minus entry/exit cost
    ev["gross5"] = ev["side"] * ev["f5"]
    print(f"\n=== net-of-cost economics (per 5d position, {COST*1e4:.0f}bps/side) ===")
    g = ev["gross5"].mean(); net = g - 2 * COST
    print(f"  long-short avg per-position 5d: gross {g*100:+.2f}%  net {net*100:+.2f}%")
    Lonly = ev[ev.side == 1]["f5"].mean() - 2 * COST
    print(f"  long-only (strongPOS) avg per-position 5d net: {Lonly*100:+.2f}%")
    # annualized Sharpe of the per-position returns (treat each as ~independent, scale)
    r = ev["gross5"].values - 2 * COST
    sharpe = r.mean() / r.std() * np.sqrt(252 / HOLD * (len(ev) / ((ev.date.max()-ev.date.min()).days/365.25*252/HOLD)))
    # simpler: per-trade Sharpe annualized assuming ~N trades/yr held 5d
    ntrades_yr = len(ev) / ((ev.date.max()-ev.date.min()).days/365.25)
    per_trade_sharpe = r.mean() / (r.std() + 1e-9)
    print(f"  per-trade net mean {r.mean()*100:+.2f}%  std {r.std()*100:.2f}%  per-trade Sharpe {per_trade_sharpe:+.3f}")
    print(f"  capacity: ~{ntrades_yr:.0f} trades/yr, ~{ntrades_yr*HOLD/252:.0f} concurrent positions (LOW capacity)")
    print(f"\n[total {time.time()-t0:.0f}s]")
