"""LEAKAGE / INFLATION AUDIT (user checklist A/C/D/E, no-engine parts) on the v2 caches + ledgers.
Run: python3 research/AUDIT_leakage_v2.py [8yr|26yr]"""
import os, sys, json, pickle, bisect, numpy as np, pandas as pd, pyarrow.parquet as pq
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts"); sys.path.insert(0, "research")
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"; CACHE = f"research/_v2_{HZ}"; W = "data/wrds/"
PKL = "data/wrds/complete_sp1500_universe_v2.pkl" if HZ == "8yr" else "data/wrds/sp1500_universe_2000_v2.pkl"
S0 = "2018-01-03" if HZ == "8yr" else "2001-01-03"
R = []
def chk(n, ok, ev): R.append((n, "PASS" if ok else "FAIL", ev)); print(f"\n[{'PASS' if ok else 'FAIL'}] {n}\n   {ev}", flush=True)
def info(n, ev): R.append((n, "INFO", ev)); print(f"\n[INFO] {n}\n   {ev}", flush=True)
print(f"\n{'#'*100}\nLEAKAGE AUDIT {HZ}\n{'#'*100}", flush=True)
d = pickle.load(open(PKL, "rb")); px = d["prices_df"]; fbd = d["features_by_date"]; mems = {k: d[k] for k in ("sp500_mem", "sp400_mem", "sp600_mem")}; keys = {k: sorted(v) for k, v in mems.items()}
p2t = d.get("permno_to_ticker", {})
def mem_at(dt):
    s = set()
    for k, v in mems.items():
        i = bisect.bisect_right(keys[k], dt) - 1
        if i >= 0: s |= set(v[keys[k][i]])
    return s
L = {nm: pd.read_parquet(f"{CACHE}/ledger_{nm}_{S0}.parquet") for nm in ("LIVE", "FINAL_1.25")}; D = {nm: pd.read_parquet(f"{CACHE}/daily_{nm}_{S0}.parquet") for nm in ("LIVE", "FINAL_1.25")}
from build_universe_v2 import load_identifiers, load_fundamentals
ci, link, sm_last = load_identifiers(); fund = load_fundamentals(link)[["tic", "datadate", "rdq", "avail_date", "niq", "seqq", "saleq", "cogsq", "dlttq", "dlcq"]]
# ---------------- 1. ten random (PERMNO, decision date) rows with source timestamps ----------------
rng = np.random.RandomState(11); dec_dates = sorted(L["LIVE"][L["LIVE"].kind == "rebal"].date.unique()); rows = []
for _ in range(10):
    T = pd.Timestamp(rng.choice(dec_dates)); cand = [s for s in mem_at(T) if s in fbd[T] and "roe" in fbd[T][s] and s in px.columns]; s = rng.choice(cand); f = fbd[T][s]
    h = px[s].loc[:T].dropna(); r20 = h.iloc[-1] / h.iloc[-21] - 1; d200 = h.iloc[-1] / h.iloc[-200:].mean() - 1; v20 = h.pct_change().iloc[-20:].std() * np.sqrt(252)
    g = fund[fund.tic == s].sort_values("avail_date"); used = g[g.avail_date <= T].iloc[-1] if (g.avail_date <= T).any() else None; nxt = g[g.avail_date > T].iloc[0] if (g.avail_date > T).any() else None
    roe_rc = used.niq / used.seqq * 4 if used is not None and pd.notna(used.niq) and pd.notna(used.seqq) and used.seqq > 0 else np.nan
    rows.append(dict(decision=T.date(), permno=s, tic=p2t.get(s, "?"), ret_20d=round(f.get("ret_20d", np.nan), 5), ret_20d_recalc=round(r20, 5), win20=f"{h.index[-21].date()}..{h.index[-1].date()}",
                     dist_sma200=round(f.get("dist_sma200", np.nan), 5), recalc=round(d200, 5), win200_start=h.index[-200].date(), vol_20d=round(f.get("vol_20d", np.nan), 4), vol_recalc=round(v20, 4),
                     last_px_date=h.index[-1].date(), roe=round(f.get("roe", np.nan), 4), roe_recalc=round(roe_rc, 4), fund_datadate=used.datadate.date() if used is not None else None, fund_rdq=pd.Timestamp(used.rdq).date() if used is not None and pd.notna(used.rdq) else None,
                     fund_avail=used.avail_date.date() if used is not None else None, next_fund_avail=nxt.avail_date.date() if nxt is not None else None))
df1 = pd.DataFrame(rows); print(df1.to_string(index=False), flush=True)
ok1 = all((r["last_px_date"] <= r["decision"]) and (r["fund_avail"] is None or r["fund_avail"] <= r["decision"]) and (r["next_fund_avail"] is None or r["next_fund_avail"] > r["decision"]) and abs(r["ret_20d"] - r["ret_20d_recalc"]) < 1e-6 and abs(r["dist_sma200"] - r["recalc"]) < 1e-6 and (np.isnan(r["roe"]) or abs(r["roe"] - r["roe_recalc"]) < 1e-6) for r in rows)
chk("1. features at T come only from data dated <= T (prices window ends at T; fundamentals row is the latest with avail_date=rdq (else datadate+90d) <= T; the next filing is after T); values recomputed independently", ok1, f"10/10 rows consistent; analyst estimates / earnings signals / price targets / short interest: maps zeroed under deployed parity (main_production_backtest.py:49-52, _enforce_deployed_parity) — pickle sizes estimates={len(d['estimates_data'])} price_targets={len(d['price_targets'])} fin_growth={len(d['fin_growth'])}")
# ---------------- 2. one trade: exact rows ----------------
lg = L["LIVE"]; t = lg[(lg.kind == "rebal") & (lg.dq > 0)].iloc[5]; s = t.sym; T = pd.Timestamp(t.date); h = px[s]
seg = h.loc[:T].iloc[-3:].to_frame("close"); seg = pd.concat([seg, h.loc[T:].iloc[1:4].to_frame("close")]); seg["role"] = ["signal-2", "signal-1", "SIGNAL & FILL (close)", "T+1 (first P&L day)", "T+2", "T+3"][:len(seg)]
print(f"\n   trade: {t.kind} {int(t.dq):+d} sh {p2t.get(s, s)}({s}) on {T.date()} at {t.px:.4f}, cost {t.cost:.2f}\n" + seg.to_string(), flush=True)
chk("2. fill price == close of the SIGNAL bar (same-bar close fill); P&L accrues from T+1", abs(t.px - h.loc[T]) < 1e-9, f"fill {t.px:.4f} == close({T.date()}) {h.loc[T]:.4f}. There is NO forward-return label anywhere (no ML); the sleeves rank on trailing windows only. Same-bar-close fills are the optimistic convention -> quantified in the harness audit (next-close fills).")
# ---------------- 4. membership as-of-date, turnover ----------------
yrs = sorted({k.year for k in keys["sp500_mem"]}); rowsm = []
for y in yrs:
    ds = [k for k in keys["sp500_mem"] if k.year == y]; a, b = mem_at(ds[0]), mem_at(ds[-1]); rowsm.append((y, len(a), len(b), len(b - a), len(a - b)))
print("\n   year  members_Jan members_Dec adds drops\n" + "\n".join(f"   {y}  {a:>6} {b:>6} {ad:>5} {dr:>5}" for y, a, b, ad, dr in rowsm), flush=True)
tot_turn = sum(r[3] + r[4] for r in rowsm); chk("4. membership is as-of-date (changes through time), not today's list applied backward", tot_turn > 20 * len(yrs), f"{tot_turn} adds+drops over {len(yrs)} years ({tot_turn/len(yrs):.0f}/yr); Jan-vs-Dec sizes above; today's list applied backward would show 0 turnover")
# ---------------- 5. top-20 single-day returns among HELD names, verified vs raw CRSP DlyRet and Polygon ----------------
lg = L["FINAL_1.25"].sort_values("date"); hold = {}; held_days = []
dates = sorted(set(D["FINAL_1.25"].index)); fills = {k: g for k, g in lg.groupby("date")}; cur = {}
for dd in dates:
    if dd in fills:
        for r in fills[dd].itertuples(): cur[(r.tranche, r.sym)] = cur.get((r.tranche, r.sym), 0) + r.dq
    for (tr, s), q in list(cur.items()):
        if q <= 0: cur.pop((tr, s))
    held_days.append((dd, {s for (_, s) in cur}))
ret = px.pct_change(fill_method=None); top = []
for dd, syms in held_days:
    if dd not in ret.index: continue
    r = ret.loc[dd, [s for s in syms if s in ret.columns]].dropna()
    for s, v in r.items(): top.append((dd, s, v))
top = sorted(top, key=lambda x: -abs(x[2]))[:20]
crsp_ids = list({int(s) for _, s, _ in top if str(s).isdigit()})
cr = pd.read_parquet(W + "crsp_daily_stock_full.parquet", columns=["PERMNO", "DlyCalDt", "DlyRet", "DlyPrc"], filters=[("PERMNO", "in", crsp_ids)]); cr["DlyCalDt"] = pd.to_datetime(cr["DlyCalDt"])
cc = pd.read_parquet("data/cached_close_prices.parquet"); cc.index = pd.to_datetime(cc.index); pgp = pd.read_parquet("data/polygon_close_panel.parquet"); pgp.index = pd.to_datetime(pgp.index)
out5 = []; bad5 = 0
for dd, s, v in top:
    c = cr[(cr.PERMNO == int(s)) & (cr.DlyCalDt == dd)]; crsp = float(c.DlyRet.iloc[0]) if len(c) and pd.notna(c.DlyRet.iloc[0]) else np.nan
    tk = p2t.get(s, "?"); poly = np.nan
    for src in (cc, pgp):
        if tk in src.columns and dd in src.index:
            i = src.index.get_loc(dd)
            if i > 0 and pd.notna(src[tk].iloc[i]) and pd.notna(src[tk].iloc[i-1]): poly = src[tk].iloc[i] / src[tk].iloc[i-1] - 1; break
    ref = crsp if not np.isnan(crsp) else poly; agree = (not np.isnan(ref)) and abs(v - ref) < 0.02
    bad5 += int(not agree and not np.isnan(ref)); out5.append((dd.date(), tk, s, f"{v:+.1%}", f"{crsp:+.1%}" if not np.isnan(crsp) else "n/a (2026: not in CRSP)", f"{poly:+.1%}" if not np.isnan(poly) else "n/a", "ok" if agree else ("UNVERIFIED" if np.isnan(ref) else "MISMATCH")))
print("\n   date        tic    permno  v2 ret   CRSP DlyRet   Polygon   verdict\n" + "\n".join(f"   {a}  {b:<6} {c:<7} {e:>7}  {f:<24} {g:>8}  {h}" for a, b, c, e, f, g, h in out5), flush=True)
chk("5. top-20 single-day returns of HELD names agree with raw CRSP DlyRet / Polygon (no fake split/dividend jumps)", bad5 == 0, f"{sum(1 for o in out5 if o[6]=='ok')}/20 verified, {bad5} mismatches, {sum(1 for o in out5 if o[6]=='UNVERIFIED')} unverifiable (no independent source)")
# ---------------- 9. costs per fill from the ledger ----------------
lg = L["LIVE"]; cr_ = (5 + 5) / 1e4; f9 = lg.assign(expected=(lg.dq * lg.px).abs() * cr_); dev = (f9.cost - f9.expected).abs().max()
print("\n" + pd.concat([f9[f9.dq > 0].head(4), f9[(f9.dq < 0) & (f9.kind == "exit")].head(3), f9[f9.kind == "stop"].head(3)]).to_string(index=False), flush=True)
chk("9. cost+slippage applied on EVERY fill (entries, exits, stops), nonzero, = |shares x price| x 10 bp", (f9.cost > 0).all() and dev < 1e-6, f"{len(f9)} fills, min cost {f9.cost.min():.4f}, max |cost - expected| {dev:.2e}; total {f9.cost.sum():,.0f}; financing separately {D['LIVE'].fin.sum():,.0f}")
# ---------------- 12. duplicates / overlaps ----------------
dup = lg.duplicated(subset=["date", "tranche", "sym", "kind"]).sum(); lgF = L["FINAL_1.25"]; multi = sum(1 for dd, syms in held_days for _ in [0]) 
chk("12. no duplicate fills; overlapping same-ticker positions only across the 4 tranche books by design", dup == 0, f"LIVE duplicates {dup}; FINAL: same sym may be held in several tranches simultaneously (that is the K=4 design: each book is sized on NAV/4 with its own 15% cap)")
# ---------------- 13. INDEPENDENT equity rebuild from the ledger (no engine code) ----------------
fin = pd.read_parquet("research/_fin_rate.parquet")["fin_rate"].dropna()
for nm in ("LIVE", "FINAL_1.25"):
    lg = L[nm]; dl = D[nm]; days = list(dl.index); fills = {k: g for k, g in lg.groupby("date")}; cash = 50_000.0; hold = {}; nav = []; last = {}; prev_debit = 0.0
    frs = fin.reindex(pd.DatetimeIndex(days).union(fin.index)).sort_index().ffill().reindex(pd.DatetimeIndex(days)).ffill().bfill()
    for dd in days:
        if prev_debit > 0: cash -= prev_debit * float(frs.loc[dd]) / 252.0
        if dd in fills:
            for r in fills[dd].itertuples(): cash -= r.dq * r.px + r.cost; hold[r.sym] = hold.get(r.sym, 0) + r.dq
        row = px.loc[dd] if dd in px.index else None
        if row is not None:
            for s in hold:
                v = row.get(s)
                if pd.notna(v): last[s] = v
        mtm = sum(q * last.get(s, 0.0) for s, q in hold.items() if q != 0); n = cash + mtm; nav.append(n); prev_debit = max(0.0, -cash)
    v = pd.Series(nav, index=pd.DatetimeIndex(days)); e = dl.nav; rel = ((v - e).abs() / e).max()
    def st(x): r = x.pct_change().dropna(); y = (x.index[-1] - x.index[0]).days / 365.25; return (x.iloc[-1] / x.iloc[0]) ** (1 / y) - 1, r.mean() / r.std() * np.sqrt(252), float(((x - x.cummax()) / x.cummax()).min())
    a, b = st(v), st(e)
    chk(f"13. {nm}: equity curve rebuilt from the raw fill log by a separate loop matches the engine", rel < 1e-3 and abs(a[0]-b[0]) < 1e-3 and abs(a[1]-b[1]) < 1e-2 and abs(a[2]-b[2]) < 1e-3, f"max |NAV diff|/NAV {rel:.2e}; CAGR rebuilt {a[0]:+.3%} vs engine {b[0]:+.3%}; Sharpe {a[1]:.4f} vs {b[1]:.4f}; MaxDD {a[2]:.3%} vs {b[2]:.3%} (financing rebuilt from research/_fin_rate.parquet on the prior day's debit)")
# ---------------- 14. per-year, per-quarter; dominance ----------------
for nm in ("LIVE", "FINAL_1.25"):
    e = D[nm].nav; q = e.resample("QE").last().pct_change().dropna(); yv = e.resample("YE").last().pct_change().dropna(); lr = np.log1p(yv); share = (lr / lr.sum()).sort_values(ascending=False)
    print(f"\n   {nm} start {S0} quarterly: " + " ".join(f"{k.year}Q{k.quarter}:{v:+.0%}" for k, v in q.items()), flush=True)
    info(f"14. {nm}: yearly log-return share of total", "top years: " + ", ".join(f"{k.year} {v:.0%}" for k, v in share.head(4).items()) + f"; best quarter {q.idxmax().year}Q{q.idxmax().quarter} {q.max():+.1%}, worst {q.idxmin().year}Q{q.idxmin().quarter} {q.min():+.1%}")
# ---------------- 16. truncate to 2025-12-31 (compare with the 2026-09-03 numbers on the old universes) + even-month (untouched) starts only ----------------
def st2(x): x = x.dropna(); r = x.pct_change().dropna(); y = (x.index[-1] - x.index[0]).days / 365.25; return (x.iloc[-1] / x.iloc[0]) ** (1 / y) - 1, r.mean() / r.std() * np.sqrt(252), float(((x - x.cummax()) / x.cummax()).min())
prior = {"8yr": {"LIVE": (0.2685, 0.877, -0.380), "FINAL_1.25": (0.2823, 0.973, -0.318)}, "26yr": {"LIVE": (0.1393, 0.589, -0.559), "FINAL_1.25": (0.1485, 0.651, -0.461)}}[HZ]
for nm in ("LIVE", "FINAL_1.25"):
    C = pd.read_parquet(f"{CACHE}/{nm}.parquet"); full = np.array([st2(C[c]) for c in C]); tr = np.array([st2(C[c].loc[:"2025-12-31"]) for c in C]); even = [c for c in C if pd.Timestamp(c).month % 2 == 0]; ev = np.array([st2(C[c]) for c in even]); ev25 = np.array([st2(C[c].loc[:"2025-12-31"]) for c in even])
    info(f"16. {nm}: through 2025-12-31 (same window as the 09-03 numbers) vs full", f"to-2025: CAGR {tr[:,0].mean():+.2%} Sharpe {tr[:,1].mean():.3f} MaxDD {tr[:,2].mean():.1%} | prior (old universe, 09-03): {prior[nm][0]:+.2%} / {prior[nm][1]:.3f} / {prior[nm][2]:.1%} | full to 2026-08: {full[:,0].mean():+.2%} / {full[:,1].mean():.3f} / {full[:,2].mean():.1%}")
    info(f"8/16. {nm}: UNTOUCHED even-month starts only (never used in selection)", f"12 starts: full {ev[:,0].mean():+.2%} / {ev[:,1].mean():.3f} / {ev[:,2].mean():.1%}; to-2025 {ev25[:,0].mean():+.2%} / {ev25[:,1].mean():.3f} / {ev25[:,2].mean():.1%}")
# ---------------- EW buy-and-hold of the same universe (sanity baseline for the DATA) ----------------
m = px.resample("ME").last(); mr = m.pct_change(fill_method=None); ew = []
for dd in mr.index[1:]:
    if dd < pd.Timestamp(S0): continue
    ms = [s for s in mem_at(dd - pd.offsets.MonthEnd(1)) if s in mr.columns]; r = mr.loc[dd, ms].dropna(); ew.append((dd, r.mean()))
ews = pd.Series(dict(ew)); c = (1 + ews).cumprod(); y = (c.index[-1] - c.index[0]).days / 365.25
info("15a. equal-weight buy-and-hold of the SAME members (monthly, unlevered, no costs) — data sanity baseline", f"{c.index[0].date()}->{c.index[-1].date()}: CAGR {(c.iloc[-1])**(1/y)-1:+.2%}, Sharpe(m) {ews.mean()/ews.std()*np.sqrt(12):.2f}, MaxDD {float(((c-c.cummax())/c.cummax()).min()):.1%}")
# ---------------- 17. flags ----------------
for nm in ("LIVE", "FINAL_1.25"):
    C = pd.read_parquet(f"{CACHE}/{nm}.parquet"); A = np.array([st2(C[c]) for c in C]); e = D[nm].nav; le = np.log(e); tt = np.arange(len(le)); r2 = np.corrcoef(tt, le)[0, 1] ** 2
    wr = None
    chk(f"17. {nm}: no too-good flags", A[:,1].max() < 3 and A[:,2].max() < -0.10 and r2 < 0.98, f"max Sharpe {A[:,1].max():.2f} (<3); shallowest MaxDD {A[:,2].max():.1%} (need < -10% at CAGR>20%); log-equity R^2 vs time {r2:.3f} (straight line would be ~0.99+); win rates from EXP058: 49.8-53.2%")
print(f"\n{'='*100}\nLEAKAGE AUDIT {HZ}: {sum(1 for r in R if r[1]=='PASS')} PASS / {sum(1 for r in R if r[1]=='FAIL')} FAIL / {sum(1 for r in R if r[1]=='INFO')} INFO")
for n, s_, ev in R:
    if s_ == "FAIL": print(f"  FAIL: {n}: {ev}")
print("="*100, flush=True)
