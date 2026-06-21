"""
Opportunity-set test — lean cross-sectional momentum, run IDENTICALLY on any
(prices_df, members_by_date). The ONLY thing that differs between a small-cap and
a large-cap run is the universe, so any return-alpha gap is the opportunity set,
not strategy tuning.

Signal: classic 12-1 skip-month momentum (P[t-21]/P[t-252]-1), trend filter
(above 200d SMA), min-price filter, top-N equal-weight, rebalance every rebal_days,
round-trip cost in bps. Reports CAGR / Sharpe / MaxDD / yearly.

Usage:
    from alpha_lean_momentum import lean_momentum, load_members
    r = lean_momentum(prices_df, members_by_date, "2006-01-01","2025-12-31",
                      top_n=20, rebal_days=20, cost_bps=10, min_price=5)
"""
import numpy as np
import pandas as pd


def lean_momentum(prices, members_by_date, start, end, top_n=20, rebal_days=20,
                  cost_bps=10, min_price=5.0, trend_filter=True, signal_prop=False,
                  skip=21, lookback=252):
    prices = prices.sort_index()
    dates = [d for d in prices.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(dates) < lookback + 5:
        return None
    # fast membership lookup: sorted dates -> as-of
    mem_dates = sorted(members_by_date.keys()) if members_by_date else []

    def members_asof(d):
        if not mem_dates:
            return None
        i = np.searchsorted(mem_dates, d, side="right") - 1
        return set(members_by_date[mem_dates[i]]) if i >= 0 else set()

    pos = {start_i: None for start_i in []}
    nav = [1.0]
    nav_dates = [dates[0]]
    holdings = {}   # sym -> weight (target book between rebalances)
    px_at_entry = {}

    full_idx = prices.index
    di_map = {d: i for i, d in enumerate(full_idx)}

    rebal_set = set(range(0, len(dates), rebal_days))
    prev_d = dates[0]
    for k, d in enumerate(dates):
        gi = di_map[d]
        # mark-to-market daily return of current book
        if holdings:
            r = 0.0
            tot_w = sum(holdings.values())
            for s, w in holdings.items():
                p_now = prices[s].iloc[gi] if s in prices.columns else np.nan
                p_prev = prices[s].iloc[di_map[prev_d]] if s in prices.columns else np.nan
                if pd.notna(p_now) and pd.notna(p_prev) and p_prev > 0:
                    r += (w / tot_w) * (p_now / p_prev - 1.0)
            nav.append(nav[-1] * (1 + r))
        else:
            nav.append(nav[-1])
        nav_dates.append(d)
        prev_d = d

        if k not in rebal_set:
            continue
        # ---- rebalance: compute 12-1 momentum on members ----
        if gi < lookback:
            continue
        members = members_asof(d)
        p_now = prices.iloc[gi]
        p_skip = prices.iloc[gi - skip]
        p_look = prices.iloc[gi - lookback]
        sma200 = prices.iloc[gi - 199:gi + 1].mean()

        scores = {}
        cand = members if members else prices.columns
        for s in cand:
            if s not in prices.columns:
                continue
            pn, ps, pl, sm = p_now.get(s), p_skip.get(s), p_look.get(s), sma200.get(s)
            if not (pd.notna(pn) and pd.notna(ps) and pd.notna(pl)) or pl <= 0 or pn < min_price:
                continue
            if trend_filter and (pd.isna(sm) or pn <= sm):
                continue
            scores[s] = ps / pl - 1.0   # 12-1 momentum (price at skip vs lookback)
        if len(scores) < top_n:
            new_book = {}
        else:
            top = sorted(scores, key=scores.get, reverse=True)[:top_n]
            if signal_prop:
                vals = np.array([max(scores[s], 1e-4) for s in top])
                w = vals / vals.sum()
                new_book = dict(zip(top, w))
            else:
                new_book = {s: 1.0 / top_n for s in top}

        # turnover cost
        old = set(holdings); new = set(new_book)
        turnover = len(old.symmetric_difference(new)) / max(2 * top_n, 1)
        nav[-1] *= (1 - turnover * cost_bps / 10000)
        holdings = new_book

    s = pd.Series(nav, index=pd.DatetimeIndex(nav_dates))
    s = s[~s.index.duplicated(keep="last")]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    dr = s.pct_change().dropna()
    cagr = s.iloc[-1] ** (1 / yrs) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    mdd = ((s - s.cummax()) / s.cummax()).min()
    yearly = {}
    for y in range(s.index[0].year, s.index[-1].year + 1):
        ys = s[(s.index >= f"{y}-01-01") & (s.index <= f"{y}-12-31")]
        if len(ys) > 20:
            yearly[y] = ys.iloc[-1] / ys.iloc[0] - 1
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": mdd,
            "vol": dr.std() * np.sqrt(252), "yearly": yearly, "n": len(scores)}


def show(name, r):
    if r is None:
        print(f"{name:<42} None (insufficient data)"); return
    yr = " ".join(f"{k}:{v*100:+.0f}" for k, v in sorted(r["yearly"].items()))
    print(f"{name:<42} CAGR {r['cagr']*100:6.1f}%  Sharpe {r['sharpe']:.2f}  "
          f"MaxDD {r['max_dd']*100:6.1f}%  Vol {r['vol']*100:4.1f}%")
    print(f"{'':42} {yr}")
