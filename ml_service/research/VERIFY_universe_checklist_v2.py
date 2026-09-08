"""Universe CHECKLIST (user spec, section 1) on ONE rebuilt v2 pickle. Numbers, PASS/FAIL, evidence.
Run: python3 research/VERIFY_universe_checklist_v2.py [8yr|26yr]"""
import os, sys, pickle, bisect, time, json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd, pyarrow.parquet as pq
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PKL = "data/wrds/complete_sp1500_universe_v2.pkl" if HZ == "8yr" else "data/wrds/sp1500_universe_2000_v2.pkl"
START = pd.Timestamp("2016-06-01" if HZ == "8yr" else "2000-01-01"); END = pd.Timestamp("2026-09-04")
W = "data/wrds/"; R = []            # (check, PASS/FAIL/INFO, evidence)
def chk(name, ok, ev): R.append((name, "PASS" if ok else "FAIL", ev)); print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {ev}", flush=True)
def info(name, ev): R.append((name, "INFO", ev)); print(f"  [INFO] {name}: {ev}", flush=True)
t0 = time.time(); print(f"\n{'='*100}\nUNIVERSE CHECKLIST {HZ}: {PKL}  (mtime {time.ctime(os.path.getmtime(PKL))}, {os.path.getsize(PKL)/1e9:.2f} GB)\n{'='*100}", flush=True)
d = pickle.load(open(PKL, "rb")); px = d["prices_df"]; fbd = d["features_by_date"]; meta = d.get("build_meta", {})
mems = {k: d[k] for k in ("sp500_mem", "sp400_mem", "sp600_mem")}; keys = {k: sorted(v) for k, v in mems.items()}
pxf = px.loc[START:]   # feature window; prices carry a 400-day warm-up before START (needed for 252d returns / SMA200 on day 1)
p2t = d.get("permno_to_ticker", {}); t2p = {}
for p, t in p2t.items(): t2p.setdefault(t, []).append(p)
info("build_meta", json.dumps({k: v for k, v in meta.items() if k != "unresolved_symbols"})[:600])
info("unresolved membership symbols", f"{len(meta.get('unresolved_symbols', []))}: {meta.get('unresolved_symbols', [])[:20]}")
def mem_at(dt):
    s = set()
    for k, v in mems.items():
        i = bisect.bisect_right(keys[k], dt) - 1
        if i >= 0: s |= set(v[keys[k][i]])
    return s
# ---- 1. row count vs expected -------------------------------------------------------------
print("\n[1] ROW COUNT vs EXPECTED", flush=True)
cal = pd.DatetimeIndex(pd.to_datetime(pq.read_table(W + "crsp_daily_stock_full.parquet", columns=["DlyCalDt"]).column(0).unique().to_pylist()))
cal = cal[(cal >= START) & (cal <= pd.Timestamp("2025-12-31"))]
sd = pd.read_parquet(W + "compustat_security_daily/secd_2026.parquet", columns=["datadate", "prccd", "fic"]); sd = sd[sd["fic"] == "USA"].dropna(subset=["prccd"])
cnt = sd.groupby("datadate").size(); cal26 = pd.DatetimeIndex(cnt[cnt >= 0.5 * cnt.median()].index); cal26 = cal26[(cal26 >= pd.Timestamp("2026-01-01")) & (cal26 <= END)]
exp_dates = cal.union(cal26); exp_dates = exp_dates[(exp_dates >= START) & (exp_dates <= END)]
all_members = set().union(*[set().union(*[set(v[k]) for k in keys[n] if START - pd.Timedelta(days=400) <= k <= END]) for n, v in mems.items()])
etfs = [c for c in px.columns if not str(c).isdigit()]
info("expected dates", f"{len(exp_dates)} trading days {exp_dates[0].date()}..{exp_dates[-1].date()} = CRSP daily calendar (<=2025-12-31) + Compustat Security Daily US days (2026, >=50% of median names priced)")
info("expected columns", f"{len(all_members)} distinct member PERMNOs (union of sp500/sp400/sp600 membership within window) + {len(etfs)} ETFs {etfs}")
chk("prices_df rows (>= START) == expected trading days", len(pxf.index) == len(exp_dates), f"rows {len(pxf.index)} (+{len(px.index)-len(pxf.index)} warm-up) vs expected {len(exp_dates)}; missing {sorted(set(exp_dates)-set(pxf.index))[:10]} extra {sorted(set(pxf.index)-set(exp_dates))[:10]}")
cover = len(all_members & set(map(str, px.columns))); _miss = sorted(all_members - set(map(str, px.columns)))
_md = []
for _p in _miss:
    _ds = [k for n, v in mems.items() for k in keys[n] if _p in v[k]]; _md.append(f"{p2t.get(_p, '?')}({_p}) member {min(_ds).date()}..{max(_ds).date()} {len(_ds)}d" if _ds else f"{_p}: no membership dates")
chk("member PERMNOs present as price columns", cover >= 0.995 * len(all_members), f"{cover}/{len(all_members)} = {cover/len(all_members):.2%}; cells {px.shape[0]*px.shape[1]:,}; missing {len(_miss)}: " + "; ".join(_md))
chk("features_by_date has one entry per price date >= START", set(fbd) == set(pxf.index), f"{len(fbd)} feature dates vs {len(pxf.index)} price dates >= START ({len(px.index)-len(pxf.index)} warm-up dates before START carry prices only)")
# ---- 2. duplicates -----------------------------------------------------------------------
print("\n[2] DUPLICATES", flush=True)
chk("no duplicate dates in prices index", px.index.is_unique and px.index.is_monotonic_increasing, f"unique={px.index.is_unique} monotonic={px.index.is_monotonic_increasing}")
chk("no duplicate PERMNO columns", px.columns.is_unique, f"{px.shape[1]} columns, {px.columns.nunique()} unique")
dupm = sum(len(v[k]) - len(set(v[k])) for n, v in mems.items() for k in v); chk("no duplicate members within a membership date", dupm == 0, f"{dupm} duplicates across {sum(len(v) for v in mems.values())} membership dates")
ov = 0; ovd = []
for dt in (pxf.index[5], pxf.index[len(pxf)//2], pxf.index[-5]):
    a = [set(mems[k][keys[k][bisect.bisect_right(keys[k], dt)-1]]) for k in mems]
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        for p in a[i] & a[j]: ov += 1; ovd.append((dt.date(), p2t.get(p, p), p, ("sp500", "sp400", "sp600")[i], ("sp500", "sp400", "sp600")[j]))
# a name can legitimately sit in two lists on the day it MIGRATES (the vendor's daily file lists it in both); count all-date overlaps too
allov = 0; migr = 0
for dt in keys["sp500_mem"]:
    a = [set(mems[k][keys[k][bisect.bisect_right(keys[k], dt)-1]]) for k in mems]; o = (a[0] & a[1]) | (a[0] & a[2]) | (a[1] & a[2]); allov += len(o)
chk("no PERMNO in two indices at once (3 probe dates)", ov == 0, f"{ov} overlaps {ovd[:6]}; over ALL {len(keys['sp500_mem'])} dates: {allov} PERMNO-days in two lists ({allov/len(keys['sp500_mem']):.2f}/day)")
# ---- 3. NaN / inf --------------------------------------------------------------------------
print("\n[3] NaN / INF in required columns", flush=True)
vals = px.values; ninf = int(np.isinf(vals).sum()); nneg = int(np.nansum(vals <= 0)); chk("prices: no inf, no non-positive", ninf == 0 and nneg == 0, f"inf={ninf} nonpositive={nneg} NaN cells={int(np.isnan(vals).sum()):,} of {vals.size:,} (NaN = not listed that day; legitimate)")
FEATS = ["ret_5d", "ret_20d", "ret_60d", "ret_126d", "ret_252d", "vol_20d", "vol_60d", "dist_sma50", "dist_sma200", "rsi_14", "roe", "gross_margin", "debt_to_equity"]
rows = []; finf = 0
for dt in pxf.index[::max(1, len(pxf.index)//12)]:
    m = mem_at(dt); fd = fbd[dt]; r = {"date": dt.date(), "members": len(m), "has_feat": sum(1 for s in m if s in fd)}
    for f in FEATS:
        v = np.array([fd[s].get(f, np.nan) for s in m if s in fd], dtype=float); finf += int(np.isinf(v).sum()); r[f] = f"{np.isnan(v).mean():.1%}" if len(v) else "n/a"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False), flush=True); print("  (cells = NaN fraction among index members on that date; ret_252d/dist_sma200 need 252/200 prior sessions, so NaN is expected in the first year of the window and for recent listings)", flush=True)
chk("features: no inf", finf == 0, f"inf={finf}")
late = rows[-1]; chk("late-window feature coverage (members with a feature dict)", late["has_feat"] >= 0.97 * late["members"], f"{late['date']}: {late['has_feat']}/{late['members']}")
chk("roe NaN fraction late-window <= 10%", float(late["roe"].rstrip("%")) <= 10, f"{late['date']}: roe NaN {late['roe']}")
# ---- 4. date gaps -------------------------------------------------------------------------
print("\n[4] DATE RANGE / GAPS", flush=True)
_closures = {"2001-09-10": "9/11 (NYSE closed 9/11-9/14)", "2006-12-29": "Ford national day of mourning 2007-01-02", "2012-10-26": "Hurricane Sandy 2012-10-29/30", "2018-12-04": "G.H.W. Bush mourning 2018-12-05", "2025-01-08": "Carter mourning 2025-01-09"}
gaps = [(a.date(), b.date(), (b-a).days, _closures.get(str(a.date()), "UNEXPLAINED")) for a, b in zip(px.index[:-1], px.index[1:]) if (b-a).days > 4]
chk("covers full window", px.index[0] <= START + pd.Timedelta(days=3) and px.index[-1] >= pd.Timestamp("2026-09-01"), f"{px.index[0].date()} .. {px.index[-1].date()} (window {START.date()}..{END.date()})")
chk("no UNEXPLAINED gap > 4 calendar days (known NYSE closures whitelisted)", all(g[3] != "UNEXPLAINED" for g in gaps), f"{len(gaps)} gaps: {gaps[:8]}")
seam = px.loc["2025-12-15":"2026-01-20"].notna().sum(axis=1); info("2025/2026 seam priced-name count", " ".join(f"{k.date()}:{v}" for k, v in seam.items()))
# ---- 5. count per date stability ---------------------------------------------------------
print("\n[5] TICKER COUNT PER DATE", flush=True)
npx = px.notna().sum(axis=1); jump = npx.pct_change().abs(); bad = jump[jump > 0.05]
chk("priced-name count changes <=5% day-over-day", len(bad) == 0, f"min {npx.min()} median {int(npx.median())} max {npx.max()}; >5% jumps: {[(k.date(), round(v,3)) for k, v in bad.items()][:8]}")
msz = pd.Series({k: len(mem_at(k)) for k in px.index}); mj = msz.pct_change().abs(); mbad = mj[mj > 0.05]
chk("SP1500 membership size changes <=5% day-over-day", len(mbad) == 0, f"min {msz.min()} median {int(msz.median())} max {msz.max()}; >5% jumps {[(k.date(), round(v,3)) for k, v in mbad.items()][:8]}")
# ---- 6. spot checks -----------------------------------------------------------------------
print("\n[6] SPOT CHECKS", flush=True)
last_mem = mem_at(px.index[-1])
def find(tk):   # several PERMNOs can END on the same ticker (J.P. Morgan & Co 48071 vs JPMorgan Chase 47896): take the one priced latest
    ps = [p for p in t2p.get(tk, []) if p in px.columns]; return sorted(ps, key=lambda p: px[p].last_valid_index() or pd.Timestamp("1900-01-01"), reverse=True)
present = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]; gone = ["SIVB", "FRC", "BBBY", "TWTR", "ATVI"] if HZ == "8yr" else ["ENE", "WCOM", "LEH", "BSC", "SIVB"]
okp = []
for tk in present:
    ps = find(tk); p = ps[0] if ps else None; s = px[p].dropna() if p else pd.Series(dtype=float)
    okp.append(p is not None and p in last_mem and s.index[-1] >= pd.Timestamp("2026-09-01")); print(f"    {tk:<5} permno={p} in_final_membership={p in last_mem} last_price_date={s.index[-1].date() if len(s) else None} last={s.iloc[-1] if len(s) else None:.2f}")
chk("5 known tickers present, members, priced to 2026-09", all(okp), f"{present} -> {okp}")
okg = []
for tk in gone:
    ps = find(tk); p = ps[0] if ps else None
    if p is None:
        # ticker may map to a PERMNO whose LAST ticker differs; search membership history by any permno whose ticker era matched
        print(f"    {tk:<5} no PERMNO with final ticker {tk} in this universe (ok only if never a member in window)"); okg.append(tk not in {p2t.get(x) for x in all_members}); continue
    s = px[p].dropna(); ever = any(p in mems[k][dt] for k in mems for dt in keys[k][::20]); okg.append((p not in last_mem) and len(s) and s.index[-1] < pd.Timestamp("2025-06-01"))
    print(f"    {tk:<5} permno={p} ever_member={ever} in_final_membership={p in last_mem} price_ends={s.index[-1].date() if len(s) else None} last_bar_ret={s.iloc[-1]/s.iloc[-2]-1 if len(s)>1 else float('nan'):+.1%}")
chk("5 known delisted names absent from final membership and price series terminated", all(okg), f"{gone} -> {okg}")
# ---- 7. PIT fundamentals + trailing features (look-ahead evidence for the merge) -----------
print("\n[7] POINT-IN-TIME EVIDENCE (independent recompute)", flush=True)
rng = np.random.RandomState(7); dates_s = [pxf.index[i] for i in rng.choice(np.arange(5, len(pxf.index)), 6, replace=False)]; mism = 0; n = 0
for dt in dates_s:
    cand = [s for s in mem_at(dt) if s in fbd[dt] and "ret_20d" in fbd[dt][s] and s in px.columns][:3]
    for s in cand:
        h = px[s].loc[:dt].dropna(); r20 = h.iloc[-1]/h.iloc[-21]-1 if len(h) > 21 else np.nan; d200 = h.iloc[-1]/h.iloc[-200:].mean()-1 if len(h) >= 200 else np.nan
        f = fbd[dt][s]; a = abs(r20 - f.get("ret_20d", np.nan)); b = abs(d200 - f.get("dist_sma200", np.nan)) if "dist_sma200" in f and not np.isnan(d200) else 0.0
        n += 1; mism += int(not (a < 1e-6 or np.isnan(a)) or not (b < 1e-6 or np.isnan(b)))
chk("features recomputed from prices<=T match stored (ret_20d, dist_sma200)", mism == 0, f"{n-mism}/{n} sample (date,PERMNO) match to 1e-6; windows use only bars <= T")
try:
    from build_universe_v2 import load_identifiers, load_fundamentals
    ci, link, sm_last = load_identifiers(); fund = load_fundamentals(link); fund = fund[["tic", "datadate", "avail_date", "rdq", "niq", "seqq"]]
    bad = 0; n = 0; ex = []
    for dt in dates_s[:4]:
        for s in [x for x in mem_at(dt) if x in fbd[dt] and "roe" in fbd[dt][x]][:3]:
            g = fund[(fund["tic"] == s) & (fund["avail_date"] <= dt)].sort_values("avail_date")
            if g.empty: continue
            row = g.iloc[-1]; roe = row.niq/row.seqq*4 if pd.notna(row.niq) and pd.notna(row.seqq) and row.seqq > 0 else np.nan; st = fbd[dt][s]["roe"]; n += 1
            ok = (np.isnan(roe) and np.isnan(st)) or abs(roe-st) < 1e-9; bad += int(not ok); ex.append(f"{dt.date()} {s}: datadate {row.datadate.date()} rdq {pd.Timestamp(row.rdq).date() if pd.notna(row.rdq) else 'NaT(+90d)'} avail {row.avail_date.date()} lag {(dt-row.avail_date).days}d roe {st:.3f} {'ok' if ok else 'MISMATCH'}")
    print("\n".join("    " + e for e in ex[:8]), flush=True)
    chk("fundamentals used at T have avail_date (rdq, else datadate+90d) <= T and equal an independent recompute", bad == 0, f"{n-bad}/{n} match; fund datadate max {fund.datadate.max().date()}")
except Exception as e:
    chk("fundamentals PIT recompute", False, f"could not run: {e!r}")
print(f"\n{'='*100}\nCHECKLIST {HZ}: {sum(1 for r in R if r[1]=='PASS')} PASS / {sum(1 for r in R if r[1]=='FAIL')} FAIL / {sum(1 for r in R if r[1]=='INFO')} INFO   ({time.time()-t0:.0f}s)")
for nm, s_, ev in R:
    if s_ == "FAIL": print(f"  FAIL: {nm}: {ev}")
print("="*100, flush=True)
