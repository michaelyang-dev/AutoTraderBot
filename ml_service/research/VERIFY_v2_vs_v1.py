"""VERIFY v2 vs v1 — MEMORY-SAFE: `extract <v1|v2> <8yr|26yr>` loads ONE pickle, writes a compact
summary, exits. `compare <8yr|26yr>` reads the two summaries. Never two universes in RAM at once."""
import os, sys, pickle, json, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODE, WHICH, HZ = (sys.argv + [None, None, None])[1:4]
if MODE == "extract": HZ = sys.argv[3]
PKL = {"8yr": "data/wrds/complete_sp1500_universe.pkl", "26yr": "data/wrds/sp1500_universe_2000.pkl"}
OUT = "research/_v2v1"
def members_on(d, dt):
    out = set()
    for k in ("sp500_mem", "sp400_mem", "sp600_mem"):
        m = d[k]; prior = [x for x in m if x <= dt]
        if prior: out |= m[max(prior)]
    return out
def extract(which, hz):
    p = PKL[hz] if which == "v1" else PKL[hz].replace(".pkl", "_v2.pkl")
    with open(p, "rb") as f: d = pickle.load(f)
    px = d["prices_df"]; os.makedirs(OUT, exist_ok=True); tag = f"{which}_{hz}"
    px.astype("float32").to_parquet(f"{OUT}/px_{tag}.parquet")
    p2t = d.get("permno_to_ticker", {}); json.dump(p2t, open(f"{OUT}/p2t_{tag}.json", "w"))
    rows = []
    for y in sorted(set(px.index.year)):
        dt = pd.Timestamp(f"{y}-06-30")
        if dt < px.index.min() or dt > px.index.max(): continue
        mem = members_on(d, dt); sz = [len(d[k][max(x for x in d[k] if x <= dt)]) if any(x <= dt for x in d[k]) else 0 for k in ("sp500_mem","sp400_mem","sp600_mem")]
        last = px.loc[:dt].iloc[-1]; have = sum(1 for s in mem if s in last.index and pd.notna(last[s]))
        fb = d["features_by_date"]; fds = [x for x in fb if x <= dt]; f = fb[max(fds)] if fds else {}; roe = sum(1 for s in mem if s in f and not pd.isna(f[s].get("roe", np.nan)))   # v2 keeps 400d price warm-up before START; no features there
        rows.append({"year": y, "sp500": sz[0], "sp400": sz[1], "sp600": sz[2], "members": len(mem), "with_price": have, "with_roe": roe})
    memlast = members_on(d, pd.Timestamp("2025-06-30")); last = px.loc[:"2025-06-30"].iloc[-1]
    nopx = sorted(s for s in memlast if s not in last.index or pd.isna(last[s]))
    json.dump({"rows": rows, "meta": d.get("build_meta", {}), "shape": list(px.shape), "range": [str(px.index.min().date()), str(px.index.max().date())],
               "nonpos": int((px <= 0).values.sum()), "members_2025_no_price": [p2t.get(s, s) for s in nopx][:40], "n_members_2025_no_price": len(nopx), "n_members_2025": len(memlast),
               "members_2026_08": len(members_on(d, pd.Timestamp("2026-08-31"))) if px.index.max() >= pd.Timestamp("2026-08-31") else None},
              open(f"{OUT}/sum_{tag}.json", "w"), default=str)
    print(f"extracted {tag}: prices {px.shape}, {len(rows)} year rows")
def compare(hz):
    a = json.load(open(f"{OUT}/sum_v1_{hz}.json")); b = json.load(open(f"{OUT}/sum_v2_{hz}.json"))
    pa = pd.read_parquet(f"{OUT}/px_v1_{hz}.parquet"); pb = pd.read_parquet(f"{OUT}/px_v2_{hz}.parquet"); p2t = json.load(open(f"{OUT}/p2t_v2_{hz}.json"))
    print(f"\n{'='*92}\nV2 vs V1 — {hz}\n{'='*92}\nv1 {a['shape']} {a['range']} | v2 {b['shape']} {b['range']}\nv2 meta: {b['meta']}")
    print(f"\n1. MEMBERSHIP SIZES + MEMBER PRICE/ROE COVERAGE by year (D9 test — v2 must be >= v1 everywhere)")
    print(f"  {'year':<6}{'v1 500/400/600':>16}{'v2 500/400/600':>16}{'v1 price cov':>14}{'v2 price cov':>14}{'v1 roe':>9}{'v2 roe':>9}")
    A = {r["year"]: r for r in a["rows"]}; B = {r["year"]: r for r in b["rows"]}
    for y in sorted(set(A) | set(B)):
        ra, rb = A.get(y), B.get(y)
        def sz(r): return "%d/%d/%d" % (r["sp500"], r["sp400"], r["sp600"]) if r else "-"
        def cov(r, k): return "%.1f%%" % (100.0 * r[k] / max(r["members"], 1)) if r else "-"
        print("  %-6d%16s%16s%14s%14s%9s%9s" % (y, sz(ra), sz(rb), cov(ra, "with_price"), cov(rb, "with_price"), cov(ra, "with_roe"), cov(rb, "with_roe")))
    print(f"\n2. NON-POSITIVE PRICES: v1 {a['nonpos']:,}  v2 {b['nonpos']:,}")
    print(f"\n3. MEMBERS ON 2025-06-30 WITHOUT A PRICE (the D9 names): v1 {a['n_members_2025_no_price']}/{a['n_members_2025']}  v2 {b['n_members_2025_no_price']}/{b['n_members_2025']}   v2 e.g. {b['members_2025_no_price'][:12]}")
    r = pb.pct_change(fill_method=None); big = r.stack(); big = big[big.abs() > 3.0]
    print(f"\n4. SPLICE CHECK |1-day ret|>300%: v1 {int((pa.pct_change(fill_method=None).stack().abs()>3.0).sum())}  v2 {len(big)}  v2 worst: {[(str(i[0].date()), p2t.get(i[1], i[1]), round(float(v),2)) for i, v in big.sort_values(ascending=False).head(5).items()]}")
    t2p = {}
    for p, t in p2t.items():
        if p in pb.columns: t2p.setdefault(t, []).append(p)
    shared = [t for t in pa.columns if t in t2p and len(t2p[t]) == 1]; corrs = []
    for t in shared[:800]:
        x = pa[t].loc["2018":"2025"].pct_change(fill_method=None); y = pb[t2p[t][0]].loc["2018":"2025"].pct_change(fill_method=None); j = pd.concat([x, y], axis=1).dropna()
        if len(j) > 250: corrs.append(j.iloc[:,0].corr(j.iloc[:,1]))
    corrs = np.array(corrs); print(f"\n5. RETURN AGREEMENT v1-ticker vs v2-PERMNO on {len(corrs)} shared securities 2018-25: corr median {np.median(corrs):.5f}, min {corrs.min():.4f}, {(corrs>0.999).mean():.1%} >0.999 (below = v1 spliced a reused/renamed ticker)")
    seam = pb.loc["2025-12-31":"2026-01-05"]
    if len(seam) >= 2:
        r0 = (seam.iloc[1]/seam.iloc[0]-1).dropna(); typ = pb.loc["2025-10-01":"2025-12-30"].pct_change(fill_method=None).stack().abs().quantile(.99)
        print(f"\n6. 2026 SEAM (CRSP->Compustat, 2025-12-31->next): n={len(r0)} |ret| median {r0.abs().median():.2%} p99 {r0.abs().quantile(.99):.2%} max {r0.abs().max():.1%} (typical daily p99 {typ:.2%}); >25%: {int((r0.abs()>0.25).sum())}  members 2026-08: {b['members_2026_08']}")
if MODE == "extract": extract(WHICH, HZ)
else: compare(WHICH)
