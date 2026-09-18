"""EXP-060: two more point-in-time feature tables keyed by (asof, cusip8), same layout as ibes_pit for the sleeve bridge.
  onm  — overnight-return momentum (Lou-Polk-Skouras): trailing 252-session sum of log(open_t / close_{t-1}) minus the
         same for intraday; a return-decomposition signal, not close-to-close momentum. Monthly asof (month end).
  inst — 13F institutional breadth (Chen-Hong-Stein): change in the number of managers holding the stock and in aggregate
         shares held, quarter over quarter; asof = report quarter end + 45 days (the filing deadline).
Built from crsp_daily_stock_full (110M rows; PERMNO, date, open, close) and s34_holdings_type3 (125M rows)."""
import os, sys, time, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); W = "data/wrds"; OUT = "research/_exp060"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
# ---------- PERMNO -> cusip8 eras ----------
ci = pd.read_parquet(f"{W}/crsp_security_info.parquet", columns=["PERMNO", "CUSIP", "SecInfoStartDt", "SecInfoEndDt"])
ci["c8"] = ci.CUSIP.astype(str).str[:8]; ci = ci[ci.c8.str.len() == 8]
ci["s"] = pd.to_datetime(ci.SecInfoStartDt); ci["e"] = pd.to_datetime(ci.SecInfoEndDt)
# ---------- overnight momentum ----------
if not os.path.exists(f"{OUT}/onm_pit.parquet"):
    pf = pq.ParquetFile(f"{W}/crsp_daily_stock_full.parquet"); parts = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["PERMNO", "DlyCalDt", "DlyOpen", "DlyPrc", "DlyPrevPrc"])
        d = t.to_pandas(); d = d.dropna(subset=["DlyOpen", "DlyPrc", "DlyPrevPrc"])
        d = d[(d.DlyOpen > 0) & (d.DlyPrc > 0) & (d.DlyPrevPrc > 0)]
        d["on"] = np.log(d.DlyOpen / d.DlyPrevPrc); d["intra"] = np.log(d.DlyPrc / d.DlyOpen)
        parts.append(d[["PERMNO", "DlyCalDt", "on", "intra"]])
        if i % 20 == 0: print(f"  rg {i}/{pf.num_row_groups} ({time.time()-t0:.0f}s)", flush=True)
    d = pd.concat(parts, ignore_index=True); del parts
    d["DlyCalDt"] = pd.to_datetime(d.DlyCalDt); d = d[d.DlyCalDt >= "1999-01-01"]
    print(f"daily rows {len(d):,} permnos {d.PERMNO.nunique():,} ({time.time()-t0:.0f}s)", flush=True)
    d = d.sort_values(["PERMNO", "DlyCalDt"])
    g = d.groupby("PERMNO")
    d["on252"] = g["on"].transform(lambda x: x.rolling(252, min_periods=200).sum())
    d["in252"] = g["intra"].transform(lambda x: x.rolling(252, min_periods=200).sum())
    d["on21"] = g["on"].transform(lambda x: x.rolling(21, min_periods=15).sum())
    m = d.groupby(["PERMNO", d.DlyCalDt.dt.to_period("M")]).tail(1)                 # month-end rows
    m = m.dropna(subset=["on252"]); m["asof"] = m.DlyCalDt
    # PERMNO -> cusip8 as of asof
    m = m.sort_values("asof"); ci2 = ci.sort_values("s")[["PERMNO", "s", "e", "c8"]]
    mm = pd.merge_asof(m, ci2.rename(columns={"s": "asof"}), on="asof", by="PERMNO", direction="backward")
    mm = mm[mm["asof"] <= mm.e]
    out = mm[["asof", "c8", "PERMNO", "on252", "in252", "on21"]].rename(columns={"c8": "cusip8"})
    out["onm"] = out.on252 - out.in252            # overnight minus intraday component
    out.to_parquet(f"{OUT}/onm_pit.parquet", index=False)
    print(f"onm written {len(out):,} rows {out['asof'].min().date()}..{out['asof'].max().date()} ({time.time()-t0:.0f}s)", flush=True)
    del d, m, mm, out
# ---------- 13F breadth ----------
if not os.path.exists(f"{OUT}/inst_pit.parquet"):
    t = pq.read_table(f"{W}/s34_holdings_type3.parquet", columns=["mgrno", "fdate", "cusip", "shares"])
    t = t.filter(pc.greater_equal(t["fdate"], pa.scalar(pd.Timestamp("1998-01-01")))) if pa.types.is_timestamp(t.schema.field("fdate").type) else t
    h = t.to_pandas(); del t
    h["fdate"] = pd.to_datetime(h.fdate); h = h[h.fdate >= "1998-01-01"]; h["cusip8"] = h.cusip.astype(str).str[:8]
    h = h[h.shares > 0]
    print(f"13F rows {len(h):,} ({time.time()-t0:.0f}s)", flush=True)
    q = h.groupby(["cusip8", "fdate"]).agg(nmgr=("mgrno", "nunique"), shr=("shares", "sum")).reset_index().sort_values(["cusip8", "fdate"])
    del h
    gq = q.groupby("cusip8"); pn = gq.nmgr.shift(1); ps = gq.shr.shift(1); pdt = gq.fdate.shift(1)
    ok = (q.fdate - pdt).dt.days.between(80, 100)
    q["d_nmgr"] = np.where(ok, (q.nmgr - pn) / pn.clip(lower=5), np.nan)
    q["d_shr"] = np.where(ok, (q.shr - ps) / ps.clip(lower=1), np.nan)
    q["asof"] = q.fdate + pd.Timedelta(days=45)
    out = q[["asof", "cusip8", "nmgr", "shr", "d_nmgr", "d_shr"]].dropna(subset=["d_nmgr"])
    out["d_nmgr"] = out.d_nmgr.clip(-1, 1); out["d_shr"] = out.d_shr.clip(-1, 1)
    out.to_parquet(f"{OUT}/inst_pit.parquet", index=False)
    print(f"inst written {len(out):,} rows, cusips {out.cusip8.nunique():,} {out['asof'].min().date()}..{out['asof'].max().date()} ({time.time()-t0:.0f}s)", flush=True)
print("done", flush=True)
