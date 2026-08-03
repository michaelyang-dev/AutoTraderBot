"""
THREAD V1 — What does the momentum sleeve ACTUALLY pick on the full SP1500, and did it pay?

Motivation: after the SP500-only -> SP1500 parity fix (c8a097a) the #1 served name is VIR
(S&P600 smallcap, broken fundamentals). Question: is smallcap momentum junk, or the engine?

Method: replay strategy1_momentum_reversal (EXACT production function, top_n=5) at every 20th
trading day across both validated periods, with clear_deployed() PIT hygiene (same as the
validated live-mirror). For every pick record:
   - index tier (S&P500 / S&P400 / S&P600) from the PIT membership dicts in the pickle
   - the raw 12-1 momentum score (ret_252d - ret_20d) and the composite score used to rank
   - the sleeve weight the strategy assigned
   - realized forward 20d and 60d total return from bt.prices, and SPY's over the same window
Then compare SP500 vs non-SP500 picks: mean/median fwd ret, hit rate, worst decile, excess vs SPY.

Output: research/_v1_mom_picks.parquet + printed report.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402
from strategies.multi_strategy_engine import strategy1_momentum_reversal  # noqa: E402

STEP = 20  # v12 rebalance cadence
TOP_N = 5


def clear_deployed(bt):
    """Same PIT hygiene as research/livemirror_run.py — the deployed-today enhanced blobs
    are NOT point-in-time, so they are cleared for any historical replay."""
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def mem_at(mem, date):
    if date in mem:
        return set(mem[date])
    keys = mem.setdefault("__keys__", None)
    return None


class MemLookup:
    def __init__(self, mem):
        self.mem = mem
        self.keys = sorted(mem.keys())

    def at(self, date):
        if date in self.mem:
            return set(self.mem[date])
        i = np.searchsorted(self.keys, date, side="right") - 1
        return set(self.mem[self.keys[i]]) if i >= 0 else set()


def collect(period_name, path, start, end):
    t0 = time.time()
    bt = FastBacktester(universe_path=path)
    clear_deployed(bt)
    bt.uni.get_sp500 = bt._get_sp1500
    print(f"[{period_name}] loaded {time.time()-t0:.0f}s", flush=True)

    m500 = MemLookup(bt.sp500_mem)
    m400 = MemLookup(bt.sp400_mem)
    m600 = MemLookup(bt.sp600_mem)

    px = bt.prices
    dates = [d for d in sorted(px.index) if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    all_dates = list(px.index)
    pos = {d: i for i, d in enumerate(all_dates)}

    def fwd(sym, date, h):
        """Forward total return over h trading days. Returns (ret, status).
        status: 'ok' | 'trunc' (price series ends early -> use last available) | 'na'."""
        i = pos[date]
        p0 = px[sym].iloc[i] if sym in px.columns else np.nan
        if not np.isfinite(p0) or p0 <= 0:
            return np.nan, "na"
        j = min(i + h, len(all_dates) - 1)
        seg = px[sym].iloc[i:j + 1].dropna()
        if len(seg) < 2:
            return np.nan, "na"
        p1 = seg.iloc[-1]
        status = "ok" if (j == i + h and np.isfinite(px[sym].iloc[j])) else "trunc"
        return p1 / p0 - 1.0, status

    rows = []
    for k in range(0, len(dates), STEP):
        date = dates[k]
        if pos[date] + 20 >= len(all_dates):
            break
        tgt = strategy1_momentum_reversal(date, bt.uni, 0, top_n=TOP_N, rebal_days=1)
        if not tgt:
            continue
        members = bt.uni.get_sp500(date)
        r252 = bt.uni.get_feature_map(date, "ret_252d", members)
        r20 = bt.uni.get_feature_map(date, "ret_20d", members)
        s500, s400, s600 = m500.at(date), m400.at(date), m600.at(date)
        spy20, _ = fwd("SPY", date, 20)
        spy60, _ = fwd("SPY", date, 60)
        for rank, sym in enumerate(sorted(tgt, key=tgt.get, reverse=True), 1):
            if sym in s500:
                tier = "SP500"
            elif sym in s400:
                tier = "SP400"
            elif sym in s600:
                tier = "SP600"
            else:
                tier = "OTHER"
            f20, st20 = fwd(sym, date, 20)
            f60, st60 = fwd(sym, date, 60)
            rows.append(dict(
                period=period_name, date=date, sym=sym, rank=rank, tier=tier,
                weight=tgt[sym],
                mom_raw=(r252.get(sym, np.nan) - r20.get(sym, np.nan)),
                ret252=r252.get(sym, np.nan), ret20=r20.get(sym, np.nan),
                fwd20=f20, fwd60=f60, st20=st20, st60=st60,
                spy20=spy20, spy60=spy60,
                pool=len(members),
            ))
    del bt
    return pd.DataFrame(rows)


def wavg(x, w):
    m = np.isfinite(x) & np.isfinite(w)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.sum() else np.nan


def group_stats(df, col, spycol):
    x = df[col].values.astype(float)
    w = df["weight"].values.astype(float)
    ex = x - df[spycol].values.astype(float)
    ok = np.isfinite(x)
    x, exf = x[ok], ex[np.isfinite(ex)]
    if len(x) == 0:
        return None
    return dict(
        n=len(x),
        mean=float(np.mean(x)), med=float(np.median(x)),
        wmean=wavg(df[col].values.astype(float), w),
        hit=float(np.mean(x > 0)),
        beat_spy=float(np.mean(exf > 0)) if len(exf) else np.nan,
        exmean=float(np.mean(exf)) if len(exf) else np.nan,
        p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
        worst_dec=float(np.mean(np.sort(x)[:max(1, len(x) // 10)])),
        best_dec=float(np.mean(np.sort(x)[-max(1, len(x) // 10):])),
        std=float(np.std(x)), mn=float(np.min(x)), mx=float(np.max(x)),
    )


def hdr(s):
    print("\n" + "=" * 100); print(s); print("=" * 100, flush=True)


def report(df, label):
    hdr(label)
    n_dates = df["date"].nunique()
    print(f"rebalance dates sampled: {n_dates}   picks: {len(df)}   "
          f"pool size (mean SP1500 members): {df['pool'].mean():.0f}")
    tc = df["tier"].value_counts()
    print("\n(a) TIER MIX of momentum top-5 picks")
    for t in ["SP500", "SP400", "SP600", "OTHER"]:
        c = int(tc.get(t, 0))
        wsh = df.loc[df.tier == t, "weight"].sum() / df["weight"].sum()
        print(f"    {t:<7} {c:>5} picks  {c/len(df):>6.1%} of picks   {wsh:>6.1%} of sleeve weight")
    nonsp = (df.tier != "SP500")
    print(f"    NON-SP500 total: {nonsp.sum()}/{len(df)} = {nonsp.mean():.1%} of picks, "
          f"{df.loc[nonsp,'weight'].sum()/df['weight'].sum():.1%} of sleeve weight")

    print("\n    momentum score (12-1 raw) by tier:  mean / median")
    for t in ["SP500", "SP400", "SP600"]:
        s = df.loc[df.tier == t, "mom_raw"].dropna()
        if len(s):
            print(f"    {t:<7} {s.mean():>+7.1%} / {s.median():>+7.1%}   (n={len(s)})")

    for h, spycol in [("fwd20", "spy20"), ("fwd60", "spy60")]:
        print(f"\n(b/c) REALIZED {h.upper()} — equal-weight across picks")
        cols = ("n", "mean", "med", "wmean", "hit", "beat_spy", "exmean", "worst_dec", "best_dec", "p10", "p90", "mn", "mx")
        print("    " + f"{'group':<12}" + "".join(f"{c:>10}" for c in cols))
        groups = [("SP500", df[df.tier == "SP500"]), ("SP400", df[df.tier == "SP400"]),
                  ("SP600", df[df.tier == "SP600"]),
                  ("NON-SP500", df[df.tier != "SP500"]), ("ALL", df)]
        for gname, g in groups:
            st = group_stats(g, h, spycol)
            if st is None:
                continue
            print("    " + f"{gname:<12}" + f"{st['n']:>10}" +
                  "".join(f"{st[c]:>+10.2%}" if c not in ("n",) else "" for c in cols[1:]))
    return df


def main():
    out = []
    for pname, path, start, end in [
        ("2018-25", "data/wrds/complete_sp1500_universe.pkl", "2018-01-01", "2025-12-31"),
        ("2001-25", "data/wrds/sp1500_universe_2000.pkl", "2001-01-01", "2025-12-31"),
    ]:
        d = collect(pname, path, start, end)
        out.append(d)
        report(d, f"PERIOD {pname}  (momentum sleeve top-{TOP_N}, every {STEP}th trading day, SP1500 pool)")
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v1_mom_picks.parquet"))
    print("\nwrote research/_v1_mom_picks.parquet", flush=True)


if __name__ == "__main__":
    main()
