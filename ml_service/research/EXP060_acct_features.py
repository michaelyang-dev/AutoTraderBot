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
