"""Analysis of the tail-risk logs produced by research/tailrisk_run.py.
Usage: python3 research/tailrisk_analyze.py 8yr|26yr <outdir>"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
period = sys.argv[1] if len(sys.argv) > 1 else "8yr"
D = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
P = lambda t, k: pd.read_parquet(f"{D}/tr_{period}_{t}_{k}.parquet")  # noqa: E731

px = pd.read_parquet(f"{D}/tr_{period}_prices.parquet")
rets = px.pct_change()
dates = px.index
dpos = {d: i for i, d in enumerate(dates)}
var = pd.read_parquet(f"{D}/tr_{period}_variants.parquet")
pf = pd.read_parquet(f"{D}/tr_{period}_pickfeat.parquet")

LBL = {"A": "SP1500 (POST-FIX)", "B": "SP500 (PRE-FIX)", "C": "SP1500 no-stop",
       "Dv": "SP500 no-stop", "E": "SP1500 1.00x", "F": "SP500 1.00x",
       "G": "SP1500 no-gate", "H": "SP500 no-gate"}
print("=" * 118)
print(f"TAIL-RISK AUDIT — {period} — live-mirror $50k / integer / 1.49x / 6.3% fin / vol_scale_cap 1.0 / credit p95 gate")
print("=" * 118)

# ───────────────────────────── (d) BOOK-LEVEL RISK A/B ─────────────────────────────
print("\n### (d) BOOK-LEVEL: does SP1500 cost risk?  (multi-start average)")
cols = ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "worst1d", "worst20d",
        "VaR99", "CVaR99", "ulcer", "d_lt_-3%", "uw_days", "finalNAV"]
fmt = var[cols].copy()
for c in ["CAGR", "Vol", "MaxDD", "worst1d", "worst20d", "VaR99", "CVaR99", "ulcer", "d_lt_-3%"]:
    fmt[c] = (fmt[c] * 100).round(2)
fmt["finalNAV"] = fmt["finalNAV"].round(0)
print(fmt.to_string())
A, B = var.loc[var.index[0]], var.loc[var.index[1]]
print(f"\n  A−B (SP1500 minus SP500 pool, 1.49x, stop on):  CAGR {100*(A.CAGR-B.CAGR):+.2f}pp | "
      f"Vol {100*(A.Vol-B.Vol):+.2f}pp | Sharpe {A.Sharpe-B.Sharpe:+.2f} | Sortino {A.Sortino-B.Sortino:+.2f} | "
      f"MaxDD {100*(A.MaxDD-B.MaxDD):+.2f}pp | worst1d {100*(A.worst1d-B.worst1d):+.2f}pp | "
      f"worst20d {100*(A.worst20d-B.worst20d):+.2f}pp | ulcer {100*(A.ulcer-B.ulcer):+.2f}pp")
E, F = var.loc[var.index[4]], var.loc[var.index[5]]
print(f"  E−F (same, UNLEVERED 1.00x):                    CAGR {100*(E.CAGR-F.CAGR):+.2f}pp | "
      f"Vol {100*(E.Vol-F.Vol):+.2f}pp | Sharpe {E.Sharpe-F.Sharpe:+.2f} | MaxDD {100*(E.MaxDD-F.MaxDD):+.2f}pp")
print(f"  return/risk: SP1500 CAGR-per-unit-vol {A.CAGR/A.Vol:.2f} vs SP500 {B.CAGR/B.Vol:.2f}; "
      f"Calmar {A.Calmar:.2f} vs {B.Calmar:.2f}")

# ───────────────────────── (a) VOL OF MOMENTUM PICKS ─────────────────────────
print("\n### (a) REALIZED vol_20d (ANNUALIZED) OF MOMENTUM PICKS AT SELECTION")


def pickstats(tag):
    m = P(tag, "mom").merge(pf, on=["date", "sym"], how="left")
    return m


rowsv = []
for tag, lab in [("A", "SP1500 pool (POST-FIX)"), ("B", "SP500 pool (PRE-FIX)")]:
    m = pickstats(tag)
    v = m["vol_20d"].dropna()
    rowsv.append({"pool": lab, "n_picks": len(m), "n_names": m["sym"].nunique(),
                  "mean": v.mean(), "med": v.median(), "p75": v.quantile(.75),
                  "p90": v.quantile(.90), "p95": v.quantile(.95), "p99": v.quantile(.99),
                  "max": v.max(), ">40%": (v > .40).mean(), ">60%": (v > .60).mean(),
                  ">80%": (v > .80).mean()})
dv = pd.DataFrame(rowsv).set_index("pool")
print((dv.assign(**{c: (dv[c] * 100).round(1) for c in dv.columns if c not in ("n_picks", "n_names")})).to_string())

mA = pickstats("A")
print("\n  SP1500 momentum picks BY INDEX BUCKET (clean PIT membership):")
g = mA.groupby("bucket")["vol_20d"]
bk = pd.DataFrame({"n_picks": mA.groupby("bucket").size(),
                   "share_of_picks": mA.groupby("bucket").size() / len(mA),
                   "mean_vol": g.mean(), "med_vol": g.median(), "p90_vol": g.quantile(.9),
                   "max_vol": g.max(), "pct>40%": g.apply(lambda x: (x > .40).mean()),
                   "pct>60%": g.apply(lambda x: (x > .60).mean())})
print(bk.round(3).to_string())

# ───────────── (b) SINGLE-NAME CATASTROPHE AMONG HELD MOMENTUM NAMES ─────────────
print("\n### (b) FORWARD 20-DAY PATH OF EACH MOMENTUM PICK (the holding cycle)")


def fwd(tag, H=20):
    m = P(tag, "mom")
    out = []
    for d, s, bkt in zip(m["date"], m["sym"], m["bucket"]):
        if s not in px.columns or d not in dpos:
            continue
        i = dpos[d]
        w = px[s].iloc[i:i + H + 1].dropna()
        if len(w) < 5:
            continue
        p0 = w.iloc[0]
        cum = w / p0 - 1.0
        r = w.pct_change().dropna()
        out.append({"date": d, "sym": s, "bucket": bkt, "ret20": cum.iloc[-1],
                    "mincum": cum.min(), "worst1d": r.min()})
    return pd.DataFrame(out)


fw = {}
for tag, lab in [("A", "SP1500 pool"), ("B", "SP500 pool")]:
    f = fwd(tag); fw[tag] = f
    yrs = (f["date"].max() - f["date"].min()).days / 365.25
    print(f"\n  {lab}: n={len(f)} pick-cycles over {yrs:.1f}y  (mean fwd-20d ret {f.ret20.mean():+.2%})")
    print(f"    worst-1d-in-window : mean {f.worst1d.mean():+.2%} | p5 {f.worst1d.quantile(.05):+.2%} | "
          f"p1 {f.worst1d.quantile(.01):+.2%} | MIN {f.worst1d.min():+.2%}")
    for thr in [-.15, -.20, -.30, -.50]:
        n = int((f.worst1d <= thr).sum())
        print(f"    single-day gap <= {thr:+.0%}: {n:>4} events ({n/len(f):.2%} of pick-cycles, {n/yrs:.1f}/yr)")
    print(f"    intra-cycle drawdown from entry: mean {f.mincum.mean():+.2%} | p5 {f.mincum.quantile(.05):+.2%} | "
          f"p1 {f.mincum.quantile(.01):+.2%} | MIN {f.mincum.min():+.2%}")
    for thr in [-.30, -.40, -.50]:
        n = int((f.mincum <= thr).sum())
        print(f"    cycle drawdown <= {thr:+.0%}: {n:>4} ({n/len(f):.2%}, {n/yrs:.1f}/yr)")

fA = fw["A"]
print("\n  SP1500 pick-cycles BY BUCKET:")
gb = fA.groupby("bucket")
print(pd.DataFrame({"n": gb.size(), "mean_ret20": gb.ret20.mean(), "min_worst1d": gb.worst1d.min(),
                    "p1_worst1d": gb.worst1d.quantile(.01), "pct_gap<=-20%": gb.worst1d.apply(lambda x: (x <= -.20).mean()),
                    "pct_dd<=-40%": gb.mincum.apply(lambda x: (x <= -.40).mean()),
                    "min_cycle_dd": gb.mincum.min()}).round(4).to_string())

# ── position-level: worst DAILY return actually experienced on a HELD name, and its NAV bite
print("\n### (b2) ACTUAL HELD POSITIONS — worst single-day move and its NAV impact")
for tag, lab in [("A", "SP1500 (POST-FIX)"), ("B", "SP500 (PRE-FIX)")]:
    pos = P(tag, "pos")
    pos["w"] = pos["value"] / pos["nav"]
    syms = [s for s in pos["sym"].unique() if s in rets.columns]
    rr = rets[syms].stack().rename("ret").reset_index()
    rr.columns = ["date", "sym", "ret"]
    j = pos.merge(rr, on=["date", "sym"], how="left").dropna(subset=["ret"])
    j["nav_bite"] = j["w"] * j["ret"]
    yrs = (pos["date"].max() - pos["date"].min()).days / 365.25
    print(f"\n  {lab}: {len(j):,} position-days, {pos['sym'].nunique()} names, {yrs:.1f}y")
    print(f"    worst single-name 1d return while held: {j.ret.min():+.1%}  "
          f"(p0.1 {j.ret.quantile(.001):+.1%}, p1 {j.ret.quantile(.01):+.1%})")
    for thr in [-.15, -.20, -.30, -.50]:
        n = int((j.ret <= thr).sum())
        print(f"    position-days with 1d move <= {thr:+.0%}: {n:>4} ({n/yrs:.1f}/yr)")
    print(f"    worst NAV bite from ONE name in ONE day: {j.nav_bite.min():+.2%} of NAV "
          f"(p0.1 {j.nav_bite.quantile(.001):+.2%})")
    top = j.nsmallest(8, "nav_bite")[["date", "sym", "bucket", "w", "ret", "nav_bite"]]
    print("    worst 8 single-name/single-day NAV bites:")
    print(top.assign(w=(top.w * 100).round(1), ret=(top.ret * 100).round(1),
                     nav_bite=(top.nav_bite * 100).round(2)).to_string(index=False))

# ───────────────────────────── (c) 40% TRAILING STOP ─────────────────────────────
print("\n### (c) 40% TRAILING STOP: firing rate and value")
for tag, lab in [("A", "SP1500 (POST-FIX)"), ("B", "SP500 (PRE-FIX)")]:
    try:
        st = P(tag, "stops")
    except Exception:
        print(f"  {lab}: no stop events"); continue
    pos = P(tag, "pos")
    yrs = (pos["date"].max() - pos["date"].min()).days / 365.25
    print(f"\n  {lab}: {len(st)} stop fires over {yrs:.1f}y = {len(st)/yrs:.1f}/yr")
    print(f"    by bucket: {st.groupby('bucket').size().to_dict()}")
    print(f"    was a momentum pick at the time: {int(st.was_mom.sum())}/{len(st)}")
    print(f"    loss vs ENTRY at the stop: mean {st.ret_vs_entry.mean():+.1%} | "
          f"median {st.ret_vs_entry.median():+.1%} | worst {st.ret_vs_entry.min():+.1%}")
    print(f"    drawdown from PEAK when it fired: mean {st.dd_from_peak.mean():+.1%} | "
          f"worst {st.dd_from_peak.min():+.1%}  (trigger is −40%; overshoot = gap-through)")
    print(f"    position weight at the stop: mean {st.wgt_nav.mean():.1%} of NAV | max {st.wgt_nav.max():.1%}")
    # post-stop forward return of the stopped name = save (negative) vs regret (positive)
    for H in [20, 60]:
        fr = []
        for d, s in zip(st["date"], st["sym"]):
            if s not in px.columns or d not in dpos:
                continue
            i = dpos[d]
            w = px[s].iloc[i:i + H + 1].dropna()
            if len(w) >= 5:
                fr.append(w.iloc[-1] / w.iloc[0] - 1)
        fr = pd.Series(fr)
        print(f"    stopped name's next {H}d: median {fr.median():+.1%} | mean {fr.mean():+.1%} | "
              f"kept falling {(fr<0).mean():.0%} of the time | worst {fr.min():+.1%} | best {fr.max():+.1%}")
print("\n  BOOK-LEVEL value of the stop (stop ON minus stop OFF):")
for on, off, lab in [(var.index[0], var.index[2], "SP1500"), (var.index[1], var.index[3], "SP500")]:
    o, f = var.loc[on], var.loc[off]
    print(f"    {lab:<7} CAGR {100*(o.CAGR-f.CAGR):+.2f}pp | MaxDD {100*(o.MaxDD-f.MaxDD):+.2f}pp | "
          f"Sharpe {o.Sharpe-f.Sharpe:+.2f} | worst20d {100*(o.worst20d-f.worst20d):+.2f}pp | "
          f"ulcer {100*(o.ulcer-f.ulcer):+.2f}pp")

# ───────────────────── (e) CONCENTRATION / GAP SCENARIO ─────────────────────
print("\n### (e) REALIZED SINGLE-NAME CONCENTRATION and the −50% GAP SCENARIO")
for tag, lab in [("A", "SP1500 (POST-FIX)"), ("B", "SP500 (PRE-FIX)")]:
    pos = P(tag, "pos")
    pos["w"] = pos["value"] / pos["nav"]
    mx = pos.groupby("date")["w"].max()
    gross = pos.groupby("date")["value"].sum() / pos.groupby("date")["nav"].first()
    print(f"\n  {lab}: single-name weight (% of NAV) across {len(mx)} days")
    print(f"    all position-days: med {pos.w.median():.1%} p90 {pos.w.quantile(.9):.1%} "
          f"p99 {pos.w.quantile(.99):.1%} MAX {pos.w.max():.1%}")
    print(f"    daily LARGEST position: med {mx.median():.1%} p90 {mx.quantile(.9):.1%} "
          f"p99 {mx.quantile(.99):.1%} MAX {mx.max():.1%}")
    for thr in [.10, .125, .15, .18]:
        print(f"    days with a position > {thr:.1%} of NAV: {int((mx>thr).sum())} ({(mx>thr).mean():.1%} of days)")
    print(f"    realized gross/NAV: med {gross.median():.2f} p95 {gross.quantile(.95):.2f} max {gross.max():.2f}")
    w99, wmx = mx.quantile(.99), mx.max()
    print("    GAP SCENARIO (stop cannot act inside a gap):")
    for g_ in [-.30, -.50, -.70, -1.0]:
        print(f"      one name gaps {g_:+.0%}: NAV hit {w99*g_:+.2%} (at the p99 weight {w99:.1%}) / "
              f"{wmx*g_:+.2%} (at the max weight {wmx:.1%})")
    # margin headroom after the worst case
    g_ = -0.50
    nav1 = 1 + wmx * g_
    grs = gross.max() + wmx * g_
    print(f"      after a −50% gap on the largest position: NAV {nav1:.3f}x, gross/NAV {grs/nav1:.2f}x, "
          f"equity/gross {nav1/grs:.1%} (Reg-T maintenance = 25%)")
    # inject the catastrophe at the WORST possible moment and re-measure book MaxDD
    nav_s = pd.read_parquet(f"{D}/tr_{period}_{tag}_nav.parquet")["nav"]
    dd = (nav_s - nav_s.cummax()) / nav_s.cummax()
    trough = dd.idxmin()
    for g_ in [-.30, -.50, -1.0]:
        shock = wmx * g_
        v2 = nav_s.copy()
        v2.loc[trough:] = v2.loc[trough:] * (1 + shock)
        dd2 = ((v2 - v2.cummax()) / v2.cummax()).min()
        print(f"      inject a {g_:+.0%} gap on a max-size ({wmx:.1%}) position at the DD trough "
              f"({trough.date()}): MaxDD {dd.min():+.1%} -> {dd2:+.1%}")

# ───────────────────── (f) BOOK COMPOSITION BY INDEX BUCKET ─────────────────────
print("\n### (f) WHAT THE FIX ACTUALLY PUT IN THE BOOK (share of gross $ by index bucket)")
for tag, lab in [("A", "SP1500 (POST-FIX)"), ("B", "SP500 (PRE-FIX)")]:
    pos = P(tag, "pos")
    tot = pos.groupby("bucket")["value"].sum()
    cnt = pos.groupby("bucket").size()
    nm = pos.groupby("bucket")["sym"].nunique()
    print(f"  {lab}: gross$-share {(tot/tot.sum()).round(3).to_dict()} | "
          f"position-days {cnt.to_dict()} | distinct names {nm.to_dict()} | "
          f"avg positions/day {len(pos)/pos['date'].nunique():.1f}")

# ───────────── (g) IS THE #1-WEIGHTED NAME ALWAYS THE TOP MOMENTUM PICK? ─────────────
print("\n### (g) the served rank-1 name = largest blended weight. Where does it come from?")
for tag, lab in [("A", "SP1500 (POST-FIX)"), ("B", "SP500 (PRE-FIX)")]:
    t = P(tag, "tgt")
    top = t.sort_values("w_book", ascending=False).groupby("date").head(1)
    print(f"  {lab}: rank-1 name is a momentum pick on {top.is_mom.mean():.1%} of rebalances "
          f"({len(top)} rebalances); rank-1 weight med {top.w_book.median():.3f} of book "
          f"= {top.w_book.median()*1.49:.1%} of NAV at 1.49x; bucket mix {top.bucket.value_counts().to_dict()}")
    capped = t[t.w_book >= 0.0999]
    print(f"    rebalances where at least one name hit the cap: "
          f"{capped['date'].nunique()}/{t['date'].nunique()} ({capped['date'].nunique()/t['date'].nunique():.0%})")
print("\nDONE")
