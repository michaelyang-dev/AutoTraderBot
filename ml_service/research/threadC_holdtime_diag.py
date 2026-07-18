"""
THREAD C — C1 MANDATORY DIAGNOSTIC GATE (per-stock / conditional hold times).

Event study: bucket the held top-5 momentum book's forward 20d/60d return by candidate
state features, over BOTH periods (8yr 2018-25 AND 26yr 2001-25). A feature GRADUATES to a
trading rule (C2/C3/C6) ONLY if it separates forward returns MONOTONICALLY in the SAME
direction in BOTH periods. Flat / sign-flipping across periods -> dead, do not build (honest).

Prior work (memory): aging flat/inverted, only rollover + dist-52w-high showed promise.
This re-tests that and adds the C3-relevant features: vol_20d (for a vol-scaled stop),
dist_sma50/200, rsi_14, max_dd_6m, sma200_slope, dist_52w_high (price-derived where the
feature map is empty on this universe).

Extends research/alpha_aging_diag.py.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402
from strategies.multi_strategy_engine import strategy1_momentum_reversal  # noqa: E402

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl", "2018-01-01", "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl", "2001-01-01", "2025-12-31"),
]


def _fwd(prices, sym, d0, d1):
    s = prices.get(sym)
    if s is None:
        return None
    p0 = s.loc[:d0].dropna(); p1 = s.loc[:d1].dropna()
    if len(p0) == 0 or len(p1) == 0:
        return None
    a, b = p0.iloc[-1], p1.iloc[-1]
    if a <= 0 or np.isnan(a) or np.isnan(b):
        return None
    return b / a - 1.0


def _price_feats(s, d0):
    """price-derived state features at d0 (dist_52w_high, max_dd_6m, sma200_slope)."""
    p = s.loc[:d0].dropna()
    if len(p) < 60:
        return {}
    px = p.iloc[-1]
    hi252 = p.iloc[-252:].max() if len(p) >= 60 else p.max()
    dd52 = px / hi252 - 1.0 if hi252 > 0 else np.nan
    w126 = p.iloc[-126:]
    mdd6 = ((w126 - w126.cummax()) / w126.cummax()).min() if len(w126) > 10 else np.nan
    sma200 = p.iloc[-200:].mean() if len(p) >= 200 else np.nan
    sma200_prev = p.iloc[-220:-20].mean() if len(p) >= 220 else np.nan
    slope = (sma200 / sma200_prev - 1.0) if (sma200_prev and sma200_prev > 0) else np.nan
    return {"dist_52w_high": dd52, "max_dd_6m": mdd6, "sma200_slope": slope}


def collect(bt, start, end, top_n=5, rebal_days=20):
    prices = {c: bt.prices[c].dropna() for c in bt.prices.columns}
    bt.uni.get_sp500 = bt._get_sp1500
    dates = [d for d in sorted(bt.prices.index) if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    ridx = list(range(0, len(dates), rebal_days))
    age = {}; rows = []
    for k, di in enumerate(ridx):
        date = dates[di]
        w = strategy1_momentum_reversal(date, bt.uni, di, top_n=top_n, rebal_days=rebal_days)
        if not w:
            age = {}; continue
        held = list(w.keys())
        d_next = dates[ridx[k + 1]] if k + 1 < len(ridx) else None
        d_60 = dates[min(di + 60, len(dates) - 1)]
        fmaps = {key: bt.uni.get_feature_map(date, key) for key in
                 ("ret_20d", "ret_60d", "dist_sma50", "dist_sma200", "rsi_14", "vol_20d")}
        na = {}
        for sym in held:
            a = age.get(sym, 0) + 1; na[sym] = a
            r20 = fmaps["ret_20d"].get(sym)
            rec = {"date": date, "sym": sym, "age": a,
                   "rollover": (r20 is not None and r20 < 0),
                   "ret_60d": fmaps["ret_60d"].get(sym),
                   "dist_sma50": fmaps["dist_sma50"].get(sym),
                   "dist_sma200": fmaps["dist_sma200"].get(sym),
                   "rsi_14": fmaps["rsi_14"].get(sym),
                   "vol_20d": fmaps["vol_20d"].get(sym),
                   "fwd20": _fwd(prices, sym, date, d_next) if d_next else None,
                   "fwd60": _fwd(prices, sym, date, d_60)}
            if sym in prices:
                rec.update(_price_feats(prices[sym], date))
            rows.append(rec)
        age = na
    return pd.DataFrame(rows)


TERCILE_FEATS = ["ret_60d", "dist_sma50", "dist_sma200", "rsi_14", "vol_20d",
                 "dist_52w_high", "max_dd_6m", "sma200_slope"]


def tercile_means(df, col, fwd):
    sub = df.dropna(subset=[col, fwd])
    if len(sub) < 60:
        return None
    try:
        q = pd.qcut(sub[col], 3, labels=["lo", "mid", "hi"], duplicates="drop")
    except Exception:
        return None
    g = sub.groupby(q, observed=True)[fwd].mean()
    if len(g) < 3:
        return None
    return g["lo"], g["mid"], g["hi"], (sub.groupby(q, observed=True)[fwd].count()).sum()


def rollover_means(df, fwd):
    sub = df.dropna(subset=[fwd])
    g = sub.groupby("rollover")[fwd].mean()
    return g.get(False), g.get(True)


def main():
    perdata = {}
    for pname, path, start, end in PERIODS:
        print("\n" + "=" * 96, flush=True)
        print(f"PERIOD {pname}", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path)
        df = collect(bt, start, end)
        print(f"[{len(df)} obs, {df['date'].nunique()} rebalances, {time.time()-t0:.0f}s]", flush=True)
        perdata[pname] = df
        # rollover
        for fwd in ("fwd20", "fwd60"):
            a, b = rollover_means(df, fwd)
            if a is not None and b is not None:
                print(f"  rollover(1m<0) {fwd}: not-rolled {a*100:+.2f}%  rolled {b*100:+.2f}%  "
                      f"spread {(b-a)*100:+.2f}pp", flush=True)
        # tercile features
        for col in TERCILE_FEATS:
            r = tercile_means(df, col, "fwd20")
            if r:
                lo, mid, hi, n = r
                mono = "MONO+" if lo < mid < hi else ("MONO-" if lo > mid > hi else "flat")
                print(f"  {col:16s} fwd20 lo {lo*100:+.2f}%  mid {mid*100:+.2f}%  hi {hi*100:+.2f}%  "
                      f"[{mono}] n={int(n)}", flush=True)
        del bt

    # -------- graduation gate: monotone SAME direction in BOTH periods --------
    print("\n" + "=" * 96, flush=True)
    print("C1 GRADUATION GATE — feature must separate fwd20 MONOTONICALLY, SAME sign, BOTH periods", flush=True)
    print("=" * 96, flush=True)
    names = list(perdata)
    # rollover
    spreads = []
    for pn in names:
        a, b = rollover_means(perdata[pn], "fwd20")
        spreads.append((b - a) if (a is not None and b is not None) else None)
    if all(s is not None for s in spreads):
        same = all(s < 0 for s in spreads) or all(s > 0 for s in spreads)
        print(f"  rollover: spreads {[f'{s*100:+.2f}pp' for s in spreads]}  "
              f"-> {'GRADUATES' if same else 'DROP (sign flip / flat)'}", flush=True)
    for col in TERCILE_FEATS:
        dirs = []
        for pn in names:
            r = tercile_means(perdata[pn], col, "fwd20")
            if not r:
                dirs.append(None); continue
            lo, mid, hi, _ = r
            dirs.append(1 if lo < mid < hi else (-1 if lo > mid > hi else 0))
        if any(d is None for d in dirs):
            print(f"  {col:16s}: insufficient data -> DROP", flush=True)
            continue
        grad = (all(d == 1 for d in dirs) or all(d == -1 for d in dirs))
        print(f"  {col:16s}: dirs {dirs}  -> {'GRADUATES' if grad else 'DROP (not monotone both periods)'}", flush=True)


if __name__ == "__main__":
    main()
