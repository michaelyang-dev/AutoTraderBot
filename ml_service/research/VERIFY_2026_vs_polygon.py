"""Independent VENDOR cross-check of the 2026 price extension (Compustat Security Daily chained onto CRSP)
against the live system's Polygon close panel (data/polygon_close_panel.parquet, split-adjusted closes).
Uses the light extract research/_v2v1/px_v2_<hz>.parquet, never the pickle. Run: python3 research/VERIFY_2026_vs_polygon.py [8yr|26yr]"""
import os, sys, json, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
pg = pd.read_parquet("data/polygon_close_panel.parquet"); pg.index = pd.to_datetime(pg.index)
px = pd.read_parquet(f"research/_v2v1/px_v2_{HZ}.parquet"); p2t = json.load(open(f"research/_v2v1/p2t_v2_{HZ}.json"))
_full = pg.notna().mean(axis=1); D0, D1 = pd.Timestamp("2025-12-31"), min(_full[_full >= 0.9].index.max(), pd.Timestamp("2026-08-31"))   # Polygon panel tail rows are mostly NaN
print(f"\n2026 VENDOR CROSS-CHECK {HZ}: v2 (Compustat chained on CRSP) vs Polygon, {D0.date()} -> {D1.date()}  (Polygon panel {pg.shape}, ends {pg.index.max().date()})")
t2p = {}
for p, t in p2t.items(): t2p.setdefault(t, []).append(p)
a_v, b_v = px.loc[:D0].iloc[-1], px.loc[:D1].iloc[-1]; a_p, b_p = pg.loc[:D0].iloc[-1], pg.loc[:D1].iloc[-1]
rows = []
for t, ps in t2p.items():
    if t not in pg.columns or pd.isna(a_p[t]) or pd.isna(b_p[t]): continue
    for p in ps:
        if p in px.columns and pd.notna(a_v[p]) and pd.notna(b_v[p]): rows.append((t, p, b_v[p]/a_v[p]-1, b_p[t]/a_p[t]-1)); break
for e in ("SPY", "GLD", "XLK", "XLF", "XLE", "XLV", "VIXM"):
    if e in px.columns and e in pg.columns and pd.notna(a_v.get(e)) and pd.notna(b_v.get(e)): rows.append((e, e, b_v[e]/a_v[e]-1, b_p[e]/a_p[e]-1))
c = pd.DataFrame(rows, columns=["tic", "permno", "v2", "polygon"]).dropna(); c["diff"] = c.v2 - c.polygon
print(f"  n={len(c)} securities | corr {c.v2.corr(c.polygon):.4f} | mean v2 {c.v2.mean():+.2%} vs polygon {c.polygon.mean():+.2%} | median diff {c['diff'].median():+.2%} (v2 includes dividends via trfd; Polygon does not) | p95 |diff| {c['diff'].abs().quantile(.95):.2%} | >10% apart: {(c['diff'].abs()>0.10).sum()}")
print("  largest disagreements:\n" + c.reindex(c['diff'].abs().sort_values(ascending=False).index).head(10).to_string(index=False))
print("  ETFs:\n" + c[c.tic.isin(["SPY","GLD","XLK","XLF","XLE","XLV","VIXM"])].to_string(index=False))
# daily-return agreement on the 30 largest 2026 movers (the names a momentum book would hold)
top = c.sort_values("v2", ascending=False).head(30); cors = []
for r in top.itertuples():
    v = px[r.permno].loc["2026-01-01":D1].pct_change().dropna(); q = pg[r.tic].loc["2026-01-01":D1].pct_change().dropna(); j = v.index.intersection(q.index)
    if len(j) > 40: cors.append((r.tic, float(np.corrcoef(v[j], q[j])[0, 1]), float(r.v2), float(r.polygon)))
print("  top-30 2026 movers, daily-return corr v2 vs Polygon: min %.3f median %.3f" % (min(x[1] for x in cors), np.median([x[1] for x in cors])))
print("   " + "  ".join(f"{t}:{cr:.3f}({a:+.0%}/{b:+.0%})" for t, cr, a, b in cors[:15]))
ok = c.v2.corr(c.polygon) > 0.99 and (c['diff'].abs() > 0.10).sum() <= 0.01 * len(c) and min(x[1] for x in cors) > 0.95
print(f"  RESULT: {'PASS' if ok else 'FAIL'} (criteria: corr>0.99, <=1% of names >10% apart, top-mover daily corr >0.95)")
