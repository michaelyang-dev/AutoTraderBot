"""EXP-060: (1) net share issuance PIT table (asof month-end, cusip8): iss12 = log(cap_t / cap_{t-252}) - sum log(1+retx) over the
same window (split- and dividend-proof: market-cap growth not explained by price return = net issuance; negative = buybacks).
(2) synthetic 10-year Treasury daily total-return series from CRSP constant-maturity yields: r_t ~= y_{t-1}/252 - D * dy (D = modified
duration of a par 10y bond at y). Written to research/_exp060/{iss_pit.parquet, ust10_tr.csv}."""
import os, time, numpy as np, pandas as pd, pyarrow.parquet as pq
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); W = "data/wrds"; OUT = "research/_exp060"; os.makedirs(OUT, exist_ok=True); t0 = time.time()
ci = pd.read_parquet(f"{W}/crsp_security_info.parquet", columns=["PERMNO", "CUSIP", "SecInfoStartDt", "SecInfoEndDt"])
ci["c8"] = ci.CUSIP.astype(str).str[:8]; ci = ci[ci.c8.str.len() == 8]; ci["s"] = pd.to_datetime(ci.SecInfoStartDt); ci["e"] = pd.to_datetime(ci.SecInfoEndDt)
if not os.path.exists(f"{OUT}/iss_pit.parquet"):
    pf = pq.ParquetFile(f"{W}/crsp_daily_stock_full.parquet"); parts = []
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=["PERMNO", "DlyCalDt", "DlyCap", "DlyRetx"]).to_pandas()
        d = d.dropna(subset=["DlyCap"]); d = d[d.DlyCap > 0]; parts.append(d)
    d = pd.concat(parts, ignore_index=True); del parts
    d["DlyCalDt"] = pd.to_datetime(d.DlyCalDt); d = d[d.DlyCalDt >= "1998-01-01"].sort_values(["PERMNO", "DlyCalDt"])
    d["lcap"] = np.log(d.DlyCap); d["lr"] = np.log1p(d.DlyRetx.fillna(0.0).clip(lower=-0.99))
    g = d.groupby("PERMNO")
    d["lcap12"] = g.lcap.shift(252); d["cumlr"] = g.lr.cumsum(); d["cumlr12"] = g.cumlr.shift(252)
    d["iss12"] = (d.lcap - d.lcap12) - (d.cumlr - d.cumlr12)
    m = d.groupby(["PERMNO", d.DlyCalDt.dt.to_period("M")]).tail(1).dropna(subset=["iss12"])
    m["asof"] = m.DlyCalDt + pd.offsets.MonthEnd(0)
    m = m.sort_values("asof"); ci2 = ci.sort_values("s")[["PERMNO", "s", "e", "c8"]]
    mm = pd.merge_asof(m, ci2.rename(columns={"s": "asof"}), on="asof", by="PERMNO", direction="backward"); mm = mm[mm["asof"] <= mm.e]
    out = mm[["asof", "c8", "PERMNO", "iss12"]].rename(columns={"c8": "cusip8"}); out["iss12"] = out.iss12.clip(-1, 1)
    out = out.sort_values(["asof", "PERMNO"]).drop_duplicates(["asof", "cusip8"], keep="last")
    out.to_parquet(f"{OUT}/iss_pit.parquet", index=False)
    print(f"iss written {len(out):,} rows, cusips {out.cusip8.nunique():,} {out['asof'].min().date()}..{out['asof'].max().date()}; iss12 mean {out.iss12.mean():+.3f} p10 {out.iss12.quantile(.1):+.3f} p90 {out.iss12.quantile(.9):+.3f} ({time.time()-t0:.0f}s)", flush=True)
    del d, m, mm
if not os.path.exists(f"{OUT}/ust10_tr.csv"):
    # CRSP daily yield files in this pull are descriptor tables only (no values) -> FRED DGS10 (10y constant maturity)
    f = pd.read_parquet(f"{W}/fred_interest_rates_spreads_daily.parquet", columns=["date", "dgs10"]); f["date"] = pd.to_datetime(f.date)
    yl = f.set_index("date").dgs10.astype(float).dropna().sort_index() / 100.0
    D = (1 - (1 + yl / 2) ** (-20)) / (yl / 2) / (1 + yl / 2) / 2
    r = yl.shift(1) / 252 - D.shift(1) * yl.diff(); tr = (1 + r.fillna(0)).cumprod() * 100.0
    ext = pd.bdate_range(tr.index[-1] + pd.Timedelta(days=1), "2026-08-31")     # past the FRED end: carry only (stated)
    tr = pd.concat([tr, pd.Series((1 + yl.iloc[-1] / 252) ** np.arange(1, len(ext) + 1) * tr.iloc[-1], index=ext)])
    tr.to_frame("UST10").to_csv(f"{OUT}/ust10_tr.csv"); print(f"ust10 written {len(tr):,} days to {tr.index[-1].date()} ({time.time()-t0:.0f}s)", flush=True)
print("done", flush=True)
