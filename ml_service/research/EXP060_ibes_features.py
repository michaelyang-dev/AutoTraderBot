"""EXP-060: point-in-time analyst features from IBES (WRDS), keyed by OFTIC ticker, for a 'second engine' sleeve.
Outputs research/_exp060/ibes_pit.parquet with one row per (asof_date, ticker):
  sue       latest quarterly EPS SUE score (anndats <= asof, <= 120 days old)
  sue_days  days since that announcement
  rev1m/rev3m  change in FY1 mean EPS estimate over 1 / 3 statistical periods, scaled by |prior mean| (clipped)
  updown    (NUMUP - NUMDOWN) / NUMEST at the latest STATPERS
  rec1m     change in mean recommendation (lower = better) over 1 period, sign-flipped so + = upgrades
asof_date = each IBES STATPERS (monthly, ~3rd Thursday); SUE rows are joined as-of. PIT by construction: every field
uses only data stamped on/before asof."""
import os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); W = "data/wrds"; OUT = "research/_exp060"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
mem = pd.concat([pd.read_parquet(f"{W}/{f}_membership_history.parquet", columns=["symbol"]) for f in ("sp500", "sp400", "sp600")])
tics = set(mem.symbol.astype(str).str.split("-").str[0].str.upper())
print(f"universe tickers {len(tics):,}", flush=True)
# --- summary history: EPS, FY1, US firms, monthly
tb = pq.read_table(f"{W}/ibes_summary_history.parquet", columns=["OFTIC", "STATPERS", "MEASURE", "FPI", "NUMEST", "NUMUP", "NUMDOWN", "MEANEST", "USFIRM"],
                   filters=[("MEASURE", "==", "EPS"), ("FPI", "==", "1"), ("USFIRM", "==", 1)])
sh = tb.to_pandas(); del tb
sh["OFTIC"] = sh.OFTIC.astype(str).str.upper(); sh = sh[sh.OFTIC.isin(tics)]
sh["STATPERS"] = pd.to_datetime(sh.STATPERS); sh = sh.sort_values(["OFTIC", "STATPERS"]).drop_duplicates(["OFTIC", "STATPERS"], keep="last")
print(f"summary rows {len(sh):,} tickers {sh.OFTIC.nunique():,} {sh.STATPERS.min().date()}..{sh.STATPERS.max().date()} ({time.time()-t0:.0f}s)", flush=True)
g = sh.groupby("OFTIC")
prev1 = g.MEANEST.shift(1); prev3 = g.MEANEST.shift(3); d1 = g.STATPERS.shift(1); d3 = g.STATPERS.shift(3)
ok1 = (sh.STATPERS - d1).dt.days.between(20, 45); ok3 = (sh.STATPERS - d3).dt.days.between(75, 110)
sh["rev1m"] = np.where(ok1, (sh.MEANEST - prev1) / prev1.abs().clip(lower=0.05), np.nan)
sh["rev3m"] = np.where(ok3, (sh.MEANEST - prev3) / prev3.abs().clip(lower=0.05), np.nan)
sh["rev1m"] = sh.rev1m.clip(-1, 1); sh["rev3m"] = sh.rev3m.clip(-1, 1)
sh["updown"] = np.where(sh.NUMEST > 0, (sh.NUMUP.fillna(0) - sh.NUMDOWN.fillna(0)) / sh.NUMEST, np.nan)
# --- recommendations (monthly STATPERS too, own calendar): mean rec 1..5 (1 = strong buy)
rc = pd.read_parquet(f"{W}/ibes_recommendations_summary.parquet", columns=["OFTIC", "STATPERS", "MEANREC", "NUMREC", "USFIRM"])
rc = rc[rc.USFIRM == 1]; rc["OFTIC"] = rc.OFTIC.astype(str).str.upper(); rc = rc[rc.OFTIC.isin(tics)]
rc["STATPERS"] = pd.to_datetime(rc.STATPERS); rc = rc.sort_values(["OFTIC", "STATPERS"]).drop_duplicates(["OFTIC", "STATPERS"], keep="last")
gr = rc.groupby("OFTIC"); pr = gr.MEANREC.shift(1); dr = (rc.STATPERS - gr.STATPERS.shift(1)).dt.days.between(20, 45)
rc["rec1m"] = np.where(dr, -(rc.MEANREC - pr), np.nan); rc["rec1m"] = rc.rec1m.clip(-2, 2)
rc = rc[["OFTIC", "STATPERS", "MEANREC", "NUMREC", "rec1m"]].rename(columns={"MEANREC": "meanrec", "NUMREC": "numrec"})
# --- surprises: quarterly EPS
su = pq.read_table(f"{W}/ibes_surprise.parquet", columns=["OFTIC", "MEASURE", "FISCALP", "anndats", "suescore", "surpmean", "actual", "USFIRM"],
                   filters=[("MEASURE", "==", "EPS"), ("FISCALP", "==", "QTR"), ("USFIRM", "==", 1)]).to_pandas()
su["OFTIC"] = su.OFTIC.astype(str).str.upper(); su = su[su.OFTIC.isin(tics)]; su["anndats"] = pd.to_datetime(su.anndats)
su = su.dropna(subset=["suescore"]).sort_values(["OFTIC", "anndats"]).drop_duplicates(["OFTIC", "anndats"], keep="last")
su["sue"] = su.suescore.clip(-10, 10); su["beat"] = (su.actual > su.surpmean).astype(float)
su = su[["OFTIC", "anndats", "sue", "beat"]]
print(f"rec rows {len(rc):,}; surprise rows {len(su):,} ({time.time()-t0:.0f}s)", flush=True)
# --- as-of join everything onto the summary calendar
base = sh[["OFTIC", "STATPERS", "MEANEST", "NUMEST", "rev1m", "rev3m", "updown"]].rename(columns={"STATPERS": "asof"}).sort_values("asof")
rc2 = rc.rename(columns={"STATPERS": "asof"}).sort_values("asof"); su2 = su.rename(columns={"anndats": "asof"}).sort_values("asof")
out = pd.merge_asof(base, rc2, on="asof", by="OFTIC", direction="backward", tolerance=pd.Timedelta(days=45))
out = pd.merge_asof(out, su2, on="asof", by="OFTIC", direction="backward", tolerance=pd.Timedelta(days=120))
# days since the surprise (for decay analysis) — recompute via a second as-of carrying the announcement date
su3 = su2.assign(ann=su2["asof"])[["OFTIC", "asof", "ann"]]
out = pd.merge_asof(out, su3, on="asof", by="OFTIC", direction="backward", tolerance=pd.Timedelta(days=120))
out["sue_days"] = (out["asof"] - out["ann"]).dt.days; out = out.drop(columns=["ann"]).rename(columns={"OFTIC": "ticker"})
out.to_parquet(f"{OUT}/ibes_pit.parquet", index=False)
print(f"written {len(out):,} rows, {out.ticker.nunique():,} tickers, {out['asof'].min().date()}..{out['asof'].max().date()}; "
      f"coverage sue {out.sue.notna().mean():.0%} rev3m {out.rev3m.notna().mean():.0%} rec1m {out.rec1m.notna().mean():.0%} ({time.time()-t0:.0f}s)", flush=True)
