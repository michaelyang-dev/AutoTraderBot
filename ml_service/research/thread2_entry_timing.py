"""
Thread #2 — entry-timing around earnings. When the strategy initiates a fresh
momentum pick that has EARNINGS within the next few days, is holding through the
event a net NEGATIVE (binary gap risk -> delay entry) or net POSITIVE (post-
earnings drift continues momentum -> don't)? Event study on top-5 picks, split by
"earnings within next K trading days", market-relative forward return + worst-path
drawdown. If imminent-earnings picks are clearly worse -> a delay rule is worth a
backtest.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester
from strategies.multi_strategy_engine import strategy1_momentum_reversal

K = 7  # "imminent" = earnings within next K trading days


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    print(f"[loaded {time.time()-t0:.0f}s]")

    # earnings dates per ticker (OFTIC) from IBES
    ibes = pd.read_parquet("data/wrds/ibes_surprise.parquet", columns=["OFTIC", "anndats", "suescore"])
    ibes["anndats"] = pd.to_datetime(ibes["anndats"], errors="coerce")
    ibes = ibes.dropna(subset=["OFTIC", "anndats"])
    ibes = ibes[ibes["anndats"] >= "2015-01-01"]
    edates = {t: np.array(sorted(g["anndats"].values)) for t, g in ibes.groupby("OFTIC")}
    sue = {(r.OFTIC, pd.Timestamp(r.anndats).normalize()): r.suescore for r in ibes.itertuples()}
    print(f"[earnings dates for {len(edates)} tickers]")

    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    allidx = list(prices.index); di_map = {d: i for i, d in enumerate(allidx)}
    dates = [d for d in allidx if pd.Timestamp("2016-01-01") <= d <= pd.Timestamp("2025-12-31")]
    rebal = list(range(0, len(dates), 20))

    def fwd(sym, gi, h):
        j = gi + h
        if j >= len(allidx) or sym not in dr.columns:
            return None, None
        seg = dr[sym].iloc[gi+1:j+1]; m = spy_r.iloc[gi+1:j+1]
        if seg.isna().all():
            return None, None
        rel = seg.fillna(0).sum() - m.sum()
        cum = (1 + seg.fillna(0)).cumprod()
        ddv = (cum / cum.cummax() - 1).min()
        return rel, ddv

    imm = {"ret": [], "dd": []}
    noi = {"ret": [], "dd": []}
    n_imm = 0
    for k in rebal[:-1]:
        d = dates[k]; gi = di_map[d]
        w = strategy1_momentum_reversal(d, bt.uni, k, top_n=5, rebal_days=20)
        if not w:
            continue
        horizon_end = allidx[min(gi + K, len(allidx)-1)]
        for sym in w:
            ed = edates.get(sym)
            imminent = False
            if ed is not None:
                nxt = ed[(ed > np.datetime64(d)) & (ed <= np.datetime64(horizon_end))]
                imminent = len(nxt) > 0
            rel, ddv = fwd(sym, gi, 20)
            if rel is None:
                continue
            (imm if imminent else noi)["ret"].append(rel)
            (imm if imminent else noi)["dd"].append(ddv)
            if imminent:
                n_imm += 1

    def rep(name, d):
        a = np.array(d["ret"]); dd = np.array(d["dd"])
        print(f"  {name:<28} fwd20 mkt-rel mean {a.mean()*100:+6.2f}%  median {np.median(a)*100:+6.2f}%  "
              f"hit {(a>0).mean()*100:4.0f}%  worstpathDD {dd.mean()*100:6.2f}%  n={len(a)}")

    print(f"\n=== fresh top-5 picks split by earnings within next {K} trading days (2016-2025) ===")
    rep(f"IMMINENT earnings", imm)
    rep(f"no imminent earnings", noi)
    print(f"\n  {n_imm}/{n_imm+len(noi['ret'])} picks had imminent earnings "
          f"({n_imm/max(n_imm+len(noi['ret']),1)*100:.0f}%)")
    print(f"  => if IMMINENT clearly worse (ret/DD) -> delay-entry rule worth testing")
    print(f"\n[total {time.time()-t0:.0f}s]")
