"""EXP-060: point-in-time accounting-risk events from Audit Analytics (WRDS) -> research/_exp060/acct_events.parquet
(asof = public filing date, cusip8, kind). NOTE the two source files are named the wrong way round in data/wrds:
'audit_restatements.parquet' holds the SOX-404 ICFR opinions (ic_*), 'audit_sox404_internal_controls.parquet' the restatements (res_*)."""
import os, pandas as pd, pyarrow.parquet as pq
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); W = "data/wrds"; OUT = "research/_exp060"
ic = pq.read_table(f"{W}/audit_restatements.parquet", columns=["file_date", "cusip_number", "ic_is_effective"]).to_pandas()
ic["file_date"] = pd.to_datetime(ic.file_date, errors="coerce"); ic["cusip8"] = ic.cusip_number.astype(str).str[:8]
ic = ic[(ic.cusip8.str.len() == 8) & (ic.ic_is_effective == "N")][["file_date", "cusip8"]].assign(kind="icfr_ineffective")
rs = pq.read_table(f"{W}/audit_sox404_internal_controls.parquet", columns=["file_date", "cusip_number", "res_adverse", "res_fraud", "res_sec_investigation"]).to_pandas()
rs["file_date"] = pd.to_datetime(rs.file_date, errors="coerce"); rs["cusip8"] = rs.cusip_number.astype(str).str[:8]; rs = rs[rs.cusip8.str.len() == 8]
adv = rs[rs.res_adverse == 1][["file_date", "cusip8"]].assign(kind="restatement_adverse")
fr = rs[(rs.res_fraud == 1) | (rs.res_sec_investigation == 1)][["file_date", "cusip8"]].assign(kind="restatement_fraud_or_sec")
ev = pd.concat([ic, adv, fr]).dropna().sort_values("file_date").rename(columns={"file_date": "asof"}); ev.to_parquet(f"{OUT}/acct_events.parquet", index=False)
print("events", len(ev), ev.kind.value_counts().to_dict(), ev["asof"].min().date(), ev["asof"].max().date())

# ---- batch 13 additions: CFO/CEO changes (Audit Analytics director/officer changes, PIT by filing date) and dividend cuts (CRSP) ----
import numpy as np
ev = pd.read_parquet(f"{OUT}/acct_events.parquet"); ev = ev[~ev["kind"].isin(["cfo_change", "ceo_change", "div_cut"])]
do = pq.read_table(f"{W}/audit_director_officer_changes.parquet", columns=["opinion_file_date", "cusip_number", "ceo_change_severity", "cfo_change_severity"]).to_pandas()
do["asof"] = pd.to_datetime(do.opinion_file_date, errors="coerce"); do["cusip8"] = do.cusip_number.astype(str).str[:8]; do = do[do.cusip8.str.len() == 8]
cfo = do[do.cfo_change_severity.notna()][["asof", "cusip8"]].assign(kind="cfo_change"); ceo = do[do.ceo_change_severity.notna()][["asof", "cusip8"]].assign(kind="ceo_change")
dv = pq.read_table(f"{W}/crsp_dividends_distributions.parquet", columns=["PERMNO", "DisDeclareDt", "DisOrdinaryFlg", "DisType", "DisFreqType", "DisDivAmt"]).to_pandas()
dv = dv[(dv.DisOrdinaryFlg == "Y") & (dv.DisType == "CD") & (dv.DisFreqType.isin(["Q", "M"]))].dropna(subset=["DisDivAmt"])
dv["asof"] = pd.to_datetime(dv.DisDeclareDt, errors="coerce"); dv = dv.dropna(subset=["asof"]).sort_values(["PERMNO", "asof"])
dv["prev"] = dv.groupby("PERMNO").DisDivAmt.shift(1); cut = dv[(dv.prev > 0) & (dv.DisDivAmt < dv.prev * 0.999) & (dv["asof"] >= "1998-01-01")]
ci = pd.read_parquet(f"{W}/crsp_security_info.parquet", columns=["PERMNO", "CUSIP", "SecInfoStartDt", "SecInfoEndDt"]); ci["c8"] = ci.CUSIP.astype(str).str[:8]; ci = ci[ci.c8.str.len() == 8]
ci["s"] = pd.to_datetime(ci.SecInfoStartDt); ci["e"] = pd.to_datetime(ci.SecInfoEndDt)
cut = cut.sort_values("asof"); m = pd.merge_asof(cut, ci.sort_values("s")[["PERMNO", "s", "e", "c8"]].rename(columns={"s": "asof"}), on="asof", by="PERMNO", direction="backward"); m = m[m["asof"] <= m.e]
dcut = m[["asof", "c8"]].rename(columns={"c8": "cusip8"}).assign(kind="div_cut")
allev = pd.concat([ev, cfo, ceo, dcut]).dropna().sort_values("asof"); allev.to_parquet(f"{OUT}/acct_events.parquet", index=False)
print("events by kind:", allev["kind"].value_counts().to_dict())
