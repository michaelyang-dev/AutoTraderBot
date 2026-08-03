"""
THREAD V2 — momentum-sleeve pick census, CLEAN tier labels + head-to-head SP1500 vs SP500-only.

Fixes/extends threadV1:
  * V1 labelled tiers from each pickle's own *_mem dicts. VERIFIED BAD for the 26yr pickle:
    sp1500_universe_2000.pkl has sp500_mem sized 656-874, sp400_mem 846-1140, sp600_mem
    1060-1531 (vs the clean pickle's 487-507 / 394-402 / 581-603). Those are not PIT index
    memberships. So here ALL tier labels come from complete_sp1500_universe.pkl's membership
    dicts, which are clean and span 1990-2026 (sp500), 1991- (sp400), 1994- (sp600).
  * Adds the direct counterfactual: at each rebalance date run the EXACT production
    strategy1_momentum_reversal twice -- once on the full pool (SP1500, post-fix) and once
    restricted to clean PIT S&P500 members (the pre-fix live bug) -- and compare realized
    forward sleeve returns. Per-DATE aggregation so a few dates cannot dominate.
  * Adds a "broken fundamentals" (VIR-profile) tag per pick and its realized returns.
"""
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402
from strategies.multi_strategy_engine import strategy1_momentum_reversal  # noqa: E402

STEP, TOP_N = 20, 5
CLEAN = "data/wrds/complete_sp1500_universe.pkl"
RESEARCH = os.path.dirname(os.path.abspath(__file__))


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


class MemLookup:
    """PIT membership with as-of (backward) lookup."""
    def __init__(self, mem):
        self.mem = mem
        self.keys = sorted(mem.keys())
        self.cache = {}

    def at(self, date):
        if date in self.cache:
            return self.cache[date]
        if date in self.mem:
            s = set(self.mem[date])
        else:
            i = np.searchsorted(self.keys, date, side="right") - 1
            s = set(self.mem[self.keys[i]]) if i >= 0 else set()
        self.cache[date] = s
        return s


print("loading CLEAN membership labels from", CLEAN, flush=True)
_c = pickle.load(open(CLEAN, "rb"))
M500, M400, M600 = MemLookup(_c["sp500_mem"]), MemLookup(_c["sp400_mem"]), MemLookup(_c["sp600_mem"])
del _c
print("  ok", flush=True)

BROKEN_FEATS = ("revenue_growth_yoy", "eps_growth_yoy", "net_margin", "roe", "gp_assets")


def collect(period_name, path, start, end):
    t0 = time.time()
    bt = FastBacktester(universe_path=path)
    clear_deployed(bt)
    print(f"[{period_name}] loaded {time.time()-t0:.0f}s", flush=True)

    px = bt.prices
    all_dates = list(px.index)
    pos = {d: i for i, d in enumerate(all_dates)}
    dates = [d for d in sorted(px.index) if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    colset = set(px.columns)

    def fwd(sym, date, h):
        if sym not in colset:
            return np.nan
        i = pos[date]
        s = px[sym]
        p0 = s.iloc[i]
        if not np.isfinite(p0) or p0 <= 0:
            return np.nan
        j = min(i + h, len(all_dates) - 1)
        seg = s.iloc[i:j + 1].dropna()
        if len(seg) < 2:
            return np.nan
        return seg.iloc[-1] / p0 - 1.0

    rows, sleeve = [], []
    for k in range(0, len(dates), STEP):
        date = dates[k]
        if pos[date] + 20 >= len(all_dates):
            break

        # --- POST-FIX: full pool (SP1500 union as the backtest defines it) ---
        bt.uni.get_sp500 = bt._get_sp1500
        tgt_full = strategy1_momentum_reversal(date, bt.uni, 0, top_n=TOP_N, rebal_days=1)
        pool_full = bt.uni.get_sp500(date)

        # --- PRE-FIX replica: clean PIT S&P500 members only ---
        s500 = M500.at(date)
        bt.uni.get_sp500 = lambda d, _s=s500: _s
        tgt_500 = strategy1_momentum_reversal(date, bt.uni, 0, top_n=TOP_N, rebal_days=1)

        bt.uni.get_sp500 = bt._get_sp1500
        if not tgt_full:
            continue

        r252 = bt.uni.get_feature_map(date, "ret_252d")
        r20 = bt.uni.get_feature_map(date, "ret_20d")
        fm = {f: bt.uni.get_feature_map(date, f) for f in BROKEN_FEATS}
        s400, s600 = M400.at(date), M600.at(date)
        spy20, spy60 = fwd("SPY", date, 20), fwd("SPY", date, 60)

        for rank, sym in enumerate(sorted(tgt_full, key=tgt_full.get, reverse=True), 1):
            tier = ("SP500" if sym in s500 else "SP400" if sym in s400
                    else "SP600" if sym in s600 else "OTHER")
            rg, nm, ro = fm["revenue_growth_yoy"].get(sym), fm["net_margin"].get(sym), fm["roe"].get(sym)
            def neg(v):
                return (v is not None) and np.isfinite(v) and v < 0
            rows.append(dict(
                period=period_name, date=date, sym=sym, rank=rank, tier=tier,
                weight=tgt_full[sym],
                mom_raw=(r252.get(sym, np.nan) - r20.get(sym, np.nan)),
                ret252=r252.get(sym, np.nan),
                rev_g=rg if rg is not None else np.nan,
                net_margin=nm if nm is not None else np.nan,
                roe=ro if ro is not None else np.nan,
                broken=int(neg(rg) and neg(nm)),          # VIR profile: shrinking + lossmaking
                unprofitable=int(neg(nm)),
                fwd20=fwd(sym, date, 20), fwd60=fwd(sym, date, 60),
                spy20=spy20, spy60=spy60, pool=len(pool_full),
                in_500_pick=int(sym in (tgt_500 or {})),
            ))

        def sleeve_ret(tgt, h):
            if not tgt:
                return np.nan
            num = den = 0.0
            for s, w in tgt.items():
                r = fwd(s, date, h)
                if np.isfinite(r):
                    num += w * r; den += w
            return num / den if den > 0 else np.nan

        sleeve.append(dict(period=period_name, date=date,
                           full20=sleeve_ret(tgt_full, 20), full60=sleeve_ret(tgt_full, 60),
                           only500_20=sleeve_ret(tgt_500, 20), only500_60=sleeve_ret(tgt_500, 60),
                           spy20=spy20, spy60=spy60,
                           overlap=len(set(tgt_full) & set(tgt_500 or {})),
                           n500=len(tgt_500 or {})))
    del bt
    return pd.DataFrame(rows), pd.DataFrame(sleeve)


def gstats(g, col, spycol):
    x = g[col].values.astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    ex = (g[col] - g[spycol]).values.astype(float); ex = ex[np.isfinite(ex)]
    xs = np.sort(x); d = max(1, len(x) // 10)
    return dict(n=len(x), mean=x.mean(), med=np.median(x),
                trim=x[(x >= np.percentile(x, 5)) & (x <= np.percentile(x, 95))].mean(),
                hit=(x > 0).mean(), beat=(ex > 0).mean() if len(ex) else np.nan,
                exmean=ex.mean() if len(ex) else np.nan,
                worst10=xs[:d].mean(), best10=xs[-d:].mean(),
                p10=np.percentile(x, 10), mn=x.min())


COLS = ["n", "mean", "med", "trim", "hit", "beat", "exmean", "worst10", "best10", "p10", "mn"]


def ptable(groups, col, spycol, title):
    print(f"\n  {title}")
    print("    " + f"{'group':<13}" + "".join(f"{c:>9}" for c in COLS))
    for name, g in groups:
        st = gstats(g, col, spycol)
        if st is None:
            continue
        line = "    " + f"{name:<13}" + f"{st['n']:>9}"
        for c in COLS[1:]:
            line += f"{st[c]:>+9.1%}"
        print(line)


def report(df, sl, label):
    print("\n" + "=" * 112); print(label); print("=" * 112, flush=True)
    print(f"rebalance dates {df['date'].nunique()}  picks {len(df)}  "
          f"mean pool {df['pool'].mean():.0f} names  "
          f"({df['date'].min().date()} .. {df['date'].max().date()})")

    print("\n(a) TIER MIX (clean PIT labels)")
    tot_w = df["weight"].sum()
    for t in ["SP500", "SP400", "SP600", "OTHER"]:
        c = int((df.tier == t).sum())
        print(f"    {t:<7} {c:>5}  {c/len(df):>6.1%} of picks   "
              f"{df.loc[df.tier==t,'weight'].sum()/tot_w:>6.1%} of sleeve weight")
    ns = df.tier != "SP500"
    print(f"    NON-SP500: {int(ns.sum())}/{len(df)} = {ns.mean():.1%} of picks, "
          f"{df.loc[ns,'weight'].sum()/tot_w:.1%} of weight")
    print(f"    12-1 mom score median by tier: " + "  ".join(
        f"{t}={df.loc[df.tier==t,'mom_raw'].median():+.0%}" for t in ["SP500", "SP400", "SP600"]
        if (df.tier == t).any()))

    groups = [("SP500", df[df.tier == "SP500"]), ("SP400", df[df.tier == "SP400"]),
              ("SP600", df[df.tier == "SP600"]), ("OTHER", df[df.tier == "OTHER"]),
              ("NON-SP500", df[df.tier != "SP500"]), ("ALL", df)]
    ptable(groups, "fwd20", "spy20", "(b/c) realized FWD 20d, per pick")
    ptable(groups, "fwd60", "spy60", "(b/c) realized FWD 60d, per pick")

    print("\n(d) 'BROKEN FUNDAMENTALS' (rev growth<0 AND net margin<0 -- the VIR profile)")
    for tag, sub in [("broken", df[df.broken == 1]), ("not-broken", df[df.broken == 0]),
                     ("unprofitable", df[df.unprofitable == 1])]:
        st20, st60 = gstats(sub, "fwd20", "spy20"), gstats(sub, "fwd60", "spy60")
        if st20:
            print(f"    {tag:<13} n={st20['n']:>5} ({len(sub)/len(df):>5.1%} of picks)  "
                  f"fwd20 mean {st20['mean']:>+7.1%} med {st20['med']:>+7.1%} hit {st20['hit']:>5.1%}  |  "
                  f"fwd60 mean {st60['mean']:>+7.1%} med {st60['med']:>+7.1%} hit {st60['hit']:>5.1%}")

    print("\n(e) HEAD-TO-HEAD sleeve: full pool (post-fix) vs clean-PIT-SP500-only (pre-fix bug)")
    s = sl.dropna(subset=["full20", "only500_20"])
    s60 = sl.dropna(subset=["full60", "only500_60"])
    print(f"    n dates {len(s)}   mean overlap of the two top-5 lists: {sl['overlap'].mean():.2f}/5 names")
    for h, a, b, spy in [("20d", "full20", "only500_20", "spy20"), ("60d", "full60", "only500_60", "spy60")]:
        d = sl.dropna(subset=[a, b])
        diff = d[a] - d[b]
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if diff.std(ddof=1) > 0 else np.nan
        # compound the per-period sleeve returns (non-overlapping 20d windows for h=20)
        comp_a = float(np.prod(1 + d[a].values)) if h == "20d" else np.nan
        comp_b = float(np.prod(1 + d[b].values)) if h == "20d" else np.nan
        print(f"    {h}: FULL mean {d[a].mean():>+7.2%} med {d[a].median():>+7.2%} | "
              f"SP500-ONLY mean {d[b].mean():>+7.2%} med {d[b].median():>+7.2%} | "
              f"diff {diff.mean():>+7.2%} (t={t:>5.2f})  SPY {d[spy].mean():>+6.2%}")
        if h == "20d":
            yrs = (d["date"].max() - d["date"].min()).days / 365.25
            print(f"         compounded over {len(d)} non-overlapping 20d windows ({yrs:.1f}y): "
                  f"FULL x{comp_a:.2f} ({comp_a**(1/yrs)-1:+.1%}/yr)  "
                  f"SP500-ONLY x{comp_b:.2f} ({comp_b**(1/yrs)-1:+.1%}/yr)")


def main():
    dfs, sls = [], []
    for pname, path, start, end in [
        ("2018-25", CLEAN, "2018-01-01", "2025-12-31"),
        ("2001-25", "data/wrds/sp1500_universe_2000.pkl", "2001-01-01", "2025-12-31"),
    ]:
        d, s = collect(pname, path, start, end)
        dfs.append(d); sls.append(s)
        report(d, s, f"PERIOD {pname} | momentum sleeve top-{TOP_N} | every {STEP}th trading day")
        if pname == "2001-25":
            pre = d[d.date < "2018-01-01"]
            pres = s[s.date < "2018-01-01"]
            report(pre, pres, "SUB-PERIOD 2001-2017 ONLY (disjoint from the 8yr sample above)")
    pd.concat(dfs, ignore_index=True).to_parquet(os.path.join(RESEARCH, "_v2_mom_picks.parquet"))
    pd.concat(sls, ignore_index=True).to_parquet(os.path.join(RESEARCH, "_v2_mom_sleeve.parquet"))
    print("\nwrote research/_v2_mom_picks.parquet, research/_v2_mom_sleeve.parquet", flush=True)


if __name__ == "__main__":
    main()
