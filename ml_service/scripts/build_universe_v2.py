"""build_universe_v2 — PERMNO-KEYED universe with date-aware membership resolution and 2026 prices.

WHY A V2 (2026-09-07). The 2026-07-27 builder was PERMNO-keyed internally but exposed every series
under a single membership TICKER, resolved by "the PERMNO that traded under that ticker for the most
days in the window". That is not date-aware, and CRSP reuses tickers: `ACI` has belonged to four
different companies (Atlas Chemical 1966, American Controlled 1974, Ashland Coal 1988, Albertsons
2020). Whoever had more in-window days won the column; the modern member lost. Measured as BUGS D9:
the 26yr file was missing ~10% of the modern universe (179 real names with zero prices), growing
monotonically with time. Ticker changes (BK->BNY, IAC->PPLI, ...) and splices (WW/WTW) are the same
defect from other angles (D10).

THE FIX. The universe key IS the PERMNO (as a string). Membership symbols are resolved to a PERMNO
PER YEAR against CRSP's ticker-era table, with fallbacks through the CCM link, Compustat Security
Monthly and CUSIP. A ticker can therefore point to different PERMNOs in different eras with no
splicing, and a company keeps one column across renames. Prices run on CRSP to 2025-12-31 and are
extended to 2026-09-04 from Compustat Security Daily (verified identical to CRSP on the 2025
overlap) chained by returns. Fundamentals join on LPERMNO directly (no ticker relabel), with the
2026-04..08 quarters added from the new pull.

ETFs (SPY, GLD, VIXM, SH, XL*) keep their TICKER as the key because the engine looks them up by
name. Nothing in the sleeves or the backtester parses an equity key.

Output dict keys are identical to the v1 builder so FastBacktester and the clean-room engine load it
unchanged, plus `permno_to_ticker` and `build_meta` for reporting.

Env: BUILD_UNIVERSE_START (default 2000-01-01), BUILD_UNIVERSE_END (2026-09-04), BUILD_UNIVERSE_OUT.
"""
import os, re, sys, time, pickle, logging, glob
import numpy as np, pandas as pd
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts")
START = os.environ.get("BUILD_UNIVERSE_START", "2000-01-01")
END = os.environ.get("BUILD_UNIVERSE_END", "2026-09-04")
OUTPUT = Path(os.environ.get("BUILD_UNIVERSE_OUT", "data/wrds/sp1500_universe_2000_v2.pkl"))
os.environ["BUILD_UNIVERSE_START"] = START; os.environ["BUILD_UNIVERSE_END"] = END
from build_universe_2000 import compute_features   # unchanged, proven feature code
W = Path("data/wrds")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"); log = logging.getLogger("v2")
ETFS = ["GLD", "VIXM", "SH", "SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]
CRSP_END = pd.Timestamp("2025-12-31")
_strip = lambda s: re.sub(r"-\d+$", "", str(s))
META = {}

def load_identifiers():
    ci = pd.read_parquet(W / "crsp_security_info.parquet", columns=["PERMNO", "Ticker", "CUSIP", "SecInfoStartDt", "SecInfoEndDt"])
    for c in ("SecInfoStartDt", "SecInfoEndDt"): ci[c] = pd.to_datetime(ci[c], errors="coerce")
    ci = ci.dropna(subset=["PERMNO"]); ci["PERMNO"] = ci["PERMNO"].astype(int); ci["cusip8"] = ci["CUSIP"].astype(str).str[:8]
    link = pd.read_parquet(W / "compustat_crsp_link.parquet", columns=["gvkey", "tic", "LIID", "LINKTYPE", "LINKPRIM", "LPERMNO", "LINKDT", "LINKENDDT"])
    link = link.dropna(subset=["LPERMNO"]); link["LPERMNO"] = link["LPERMNO"].astype(int); link["gvkey"] = link["gvkey"].astype(str).str.zfill(6)
    link = link[link["LINKTYPE"].isin(["LC", "LU", "LS"])]
    for c in ("LINKDT", "LINKENDDT"): link[c] = pd.to_datetime(link[c], errors="coerce")
    link["LINKENDDT"] = link["LINKENDDT"].fillna(pd.Timestamp("2099-12-31"))
    sm = pd.read_parquet(W / "compustat_security_monthly.parquet", columns=["tic", "gvkey", "iid", "cusip", "datadate"]).dropna(subset=["tic"])
    sm["gvkey"] = sm["gvkey"].astype(str).str.zfill(6); sm["cusip8"] = sm["cusip"].astype(str).str[:8]
    sm_last = sm.sort_values("datadate").drop_duplicates("tic", keep="last")
    # per (tic, gvkey): first/last month with data. A delisted membership symbol "X-YYYYMM" is the gvkey with tic X whose
    # LAST month is YYYYMM; a bare symbol is the gvkey with tic X that is still reporting. This is the identity the
    # membership files were generated from, so it separates twins that share a tic (JCI / JCI-201609 = Tyco).
    sm["datadate"] = pd.to_datetime(sm["datadate"]); span = sm.groupby(["tic", "gvkey"])["datadate"].agg(["min", "max"]).reset_index()
    resolve._sm_by_tic = {}
    for r in span.itertuples(index=False): resolve._sm_by_tic.setdefault(r.tic, []).append((r.gvkey, r.min, r.max))
    return ci, link, sm_last

def resolve(sym, y0, y1, ci_by_tic, link_by_tic, sm_tic, link_by_g, ci_by_cusip):
    """Return (PERMNO, stage) for a membership symbol active in [y0, y1].

    WHAT A MEMBERSHIP SYMBOL MEANS (established 2026-09-07 from the twins in the files): it names a SECURITY, not a
    ticker-at-the-time.  A bare symbol is the security that is alive today under that ticker, over its whole history
    ("JCI" = Tyco's PERMNO 45356, which trades as JCI plc since 2016; "T" = SBC's PERMNO, "COR" = AmerisourceBergen's,
    "KDP" = Dr Pepper Snapple's).  "X-YYYYMM" is the security that ENDED in month YYYYMM under ticker X ("JCI-201609" =
    Johnson Controls Inc's PERMNO 42534, "T-200511" = the old AT&T, "GMCR-201603" = Keurig).  So a symbol maps to ONE
    PERMNO for all years; the year only matters for fallbacks.
    Two earlier resolvers were wrong: (v1) CRSP ticker era by DATE mapped 197 renamed symbols to whichever company held
    the ticker that year; (v2) Compustat identity first collapsed the twins (JCI/JCI-201609 -> one PERMNO) because both
    are the same Compustat company.  Neither survived the collapse/overlap audit in load_membership."""
    lo, hi = pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y1}-12-31")
    m = re.match(r"^(.+)-(\d{6})$", sym)
    base = m.group(1) if m else sym
    cands = [base] + ([base.replace(".", "")] if "." in base else []) + ([base[:-1]] if len(base) == 5 and base.endswith("Q") else [])
    def best_overlap(rows):
        best, bo = None, pd.Timedelta(0)
        for r in rows:
            ov = min(r[2], hi) - max(r[1], lo)
            if ov > bo: best, bo = r[0], ov
        return best
    ALIVE = pd.Timestamp("2025-12-01")                           # CRSP file ends 2025-12-31; link file uses 2099 for open links
    if m:                                                        # ---- ENDED security: match the END month ----
        dl = pd.Timestamp(year=int(m.group(2)[:4]), month=int(m.group(2)[4:]), day=1) + pd.offsets.MonthEnd(0)
        best, bd = None, pd.Timedelta(days=400)
        for s in cands:                                          # a. CRSP: PERMNO whose last era under this ticker ends nearest YYYYMM
            ends = {}
            for p, a, b in ci_by_tic.get(s, []): ends[p] = max(ends.get(p, b), b)
            for p, e in ends.items():
                if e < ALIVE and abs(e - dl) < bd: best, bd = p, abs(e - dl)
        if best is not None: return best, "crsp_delist"
        best, bd = None, pd.Timedelta(days=400)
        for s in cands:                                          # b. CCM link (tic base, any .N) whose LINKENDDT is nearest YYYYMM
            for r in link_by_tic.get(s, []):
                if r[2] < ALIVE and abs(r[2] - dl) < bd: best, bd = r[0], abs(r[2] - dl)
        if best is not None: return best, "ccm_delist"
        for s in cands:                                          # c. CCM link by tic, best overlap of the window (OTC tails: AMR -> AAMRQ)
            p = best_overlap(link_by_tic.get(s, []))
            if p is not None: return p, "ccm_tic"
        for s in cands:                                          # d. CRSP era, best overlap of the window
            p = best_overlap(ci_by_tic.get(s, []))
            if p is not None: return p, "crsp_era"
    else:                                                        # ---- ALIVE security ----
        for s in cands:                                          # a. CRSP: the PERMNO whose era under this ticker is alive at file end
            eras = [e for e in ci_by_tic.get(s, []) if e[2] >= ALIVE]
            if eras: return sorted(eras, key=lambda e: e[2])[-1][0], "crsp_latest"
        for s in cands:                                          # b. CCM: open link whose tic is this bare tic (ticker changed in 2026, or CRSP writes it differently)
            rows = [r for r in link_by_tic.get(s, []) if r[4] == s and r[2] >= ALIVE]
            if rows: return sorted(rows, key=lambda r: r[1])[-1][0], "ccm_current"
        for s in cands:                                          # c. Security Monthly: still-reporting gvkey with this tic -> open CCM link
            for g, a, b in sorted(getattr(resolve, "_sm_by_tic", {}).get(s, []), key=lambda x: x[2], reverse=True):
                if b >= pd.Timestamp("2025-10-31") and link_by_g.get(g):
                    rows = sorted(link_by_g[g], key=lambda r: r[2]); return rows[-1][0], "secm_current"
        for s in cands:                                          # d. CRSP era latest even if ended (Compustat alive, CRSP ended: rare)
            eras = ci_by_tic.get(s)
            if eras: return sorted(eras, key=lambda e: e[2])[-1][0], "crsp_latest_ended"
    # ---- shared fallbacks: Compustat Security Monthly keyed by the BASE tic (the file's "AAMRQ-201312" is Compustat's
    # frozen last tic AAMRQ; Security Monthly carries it with the CUSIP, and CRSP's security info carries the same CUSIP).
    # This is what recovers the bankruptcies -- AMR, Kodak, Frontier, Chesapeake, Delta, Delphi, Lear, Peabody, Calpine, Dana --
    # whose OTC-phase tic never existed in CRSP. Earlier versions looked these up by the full suffixed symbol and found nothing.
    sm_cusip = getattr(resolve, "_sm_cusip", None) or {}
    for s in cands:
        c8 = sm_cusip.get(s)
        if c8 and c8 in ci_by_cusip:
            eras = ci_by_cusip[c8]
            if m:                                                # ended security: the PERMNO whose last era ends nearest YYYYMM
                ends = {}
                for p, a, b in eras: ends[p] = max(ends.get(p, b), b)
                return sorted(ends.items(), key=lambda kv: abs(kv[1] - dl))[0][0], "cusip_delist"
            p = best_overlap(eras); return (p if p is not None else sorted(eras, key=lambda e: e[2])[-1][0]), "cusip"
    for s in cands:
        g = sm_tic.get(s)
        if g is not None and link_by_g.get(g):
            rows = link_by_g[g]; p = best_overlap(rows); return (p if p is not None else sorted(rows, key=lambda r: r[2])[-1][0]), "secm_gvkey"
    for s in cands:                                              # last resort: CRSP era ignoring dates
        eras = ci_by_tic.get(s)
        if eras: return sorted(eras, key=lambda e: e[1])[-1][0], "crsp_anydate"
    return None, "unresolved"

def load_membership(ci, link, sm_last):
    ci_by_tic, ci_by_cusip = {}, {}
    for r in ci.dropna(subset=["Ticker"]).itertuples(index=False):
        ci_by_tic.setdefault(r.Ticker, []).append((r.PERMNO, r.SecInfoStartDt, r.SecInfoEndDt))
    for r in ci.itertuples(index=False):
        ci_by_cusip.setdefault(r.cusip8, []).append((r.PERMNO, r.SecInfoStartDt, r.SecInfoEndDt))
    link_by_tic, link_by_g = {}, {}
    for r in link.itertuples(index=False):
        if isinstance(r.tic, str): link_by_tic.setdefault(re.sub(r"\.\d+$", "", r.tic), []).append((r.LPERMNO, r.LINKDT, r.LINKENDDT, r.LINKPRIM, r.tic))
        link_by_g.setdefault(r.gvkey, []).append((r.LPERMNO, r.LINKDT, r.LINKENDDT, r.LINKPRIM))
    sm_tic = dict(zip(sm_last.tic, sm_last.gvkey)); resolve._sm_cusip = dict(zip(sm_last.tic, sm_last.cusip8))
    mems, stages, unresolved = {}, {}, set()
    for name, fn in (("sp500", "sp500_membership_history.parquet"), ("sp400", "sp400_membership_history.parquet"), ("sp600", "sp600_membership_history.parquet")):
        df = pd.read_parquet(W / fn).reset_index(); dcol = df.columns[0]; df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        flag = next(c for c in df.columns if c.strip().lower() == "index constituent")
        df = df[(pd.to_numeric(df[flag], errors="coerce") == 1) & (df[dcol] >= pd.Timestamp(START) - pd.Timedelta(days=400)) & (df[dcol] <= pd.Timestamp(END))]
        df["sym"] = df["symbol"].astype(str); df["year"] = df[dcol].dt.year      # RAW symbol: resolve() reads the -YYYYMM delist suffix
        keymap = {}
        for (s, y), _ in df.groupby(["sym", "year"]):
            p, st = resolve(s, y, y, ci_by_tic, link_by_tic, sm_tic, link_by_g, ci_by_cusip)
            stages[st] = stages.get(st, 0) + 1
            if p is None: unresolved.add(s)
            keymap[(s, y)] = None if p is None else str(p)
        df["key"] = [keymap[(s, y)] for s, y in zip(df["sym"], df["year"])]
        m = {}; collapsed = 0
        for d, g in df.dropna(subset=["key"]).groupby(dcol): m[d] = set(g["key"].unique()); collapsed += len(g) - len(m[d])
        mems[name] = m; META[f"{name}_symbol_days_collapsed"] = collapsed
        log.info("  %s: %d symbol-days where two symbols map to one PERMNO (must be ~0)", name, collapsed)
        log.info("  %s: %d dates, members on last date %d (%s)", name, len(m), len(m[max(m)]), max(m).date())
    META["resolution_stages"] = stages; META["unresolved_symbols"] = sorted(unresolved)
    # cross-index overlap audit: a PERMNO in two of the three lists on one date is a mapping error (S&P indices are disjoint)
    ov = 0; k5 = sorted(mems["sp500"])
    for d in k5[::5]:
        a5 = mems["sp500"][d]; a4 = mems["sp400"].get(d, set()); a6 = mems["sp600"].get(d, set()); ov += len(a5 & a4) + len(a5 & a6) + len(a4 & a6)
    META["cross_index_overlap_permno_days_sampled"] = ov; log.info("  cross-index overlaps on every-5th date: %d PERMNO-days (was ~1.7/day before the resolver fix)", ov)
    log.info("  resolution stages: %s ; unresolved symbols: %d %s", stages, len(unresolved), sorted(unresolved)[:15])
    return mems["sp500"], mems["sp400"], mems["sp600"], ci_by_tic, link

def chain_adjusted(g):
    ret = pd.to_numeric(g["DlyRet"], errors="coerce").values; raw = pd.to_numeric(g["DlyPrc"], errors="coerce").abs().values
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan); n = len(ret); pos = np.flatnonzero(np.isfinite(raw))
    if n == 0 or pos.size == 0: return None
    anchor = pos[-1]; adj = np.full(n, np.nan); adj[anchor] = raw[anchor]
    for i in range(anchor + 1, n):
        r = ret[i]; adj[i] = adj[i - 1] * (1.0 + r) if (np.isfinite(r) and r > -1.0) else np.nan
    for i in range(anchor - 1, -1, -1):
        r = ret[i + 1]
        if np.isfinite(r) and r != -1.0: adj[i] = adj[i + 1] / (1.0 + r)
        else:
            pn, pc = raw[i + 1], raw[i]
            adj[i] = adj[i + 1] * (pc / pn) if (np.isfinite(pn) and np.isfinite(pc) and pn > 0 and pc > 0) else adj[i + 1]
    return pd.Series(adj, index=g["DlyCalDt"].values)

def load_prices(needed_permnos, ci_by_tic, link):
    t0 = time.time(); start = pd.Timestamp(START) - pd.Timedelta(days=400)
    etf_permno = {}
    for e in ETFS:
        eras = ci_by_tic.get(e)
        if eras: etf_permno[sorted(eras, key=lambda x: x[1])[-1][0]] = e
    want = set(needed_permnos) | set(etf_permno)
    df = pd.read_parquet(W / "crsp_daily_stock_full.parquet", columns=["PERMNO", "DlyCalDt", "DlyRet", "DlyPrc"], filters=[("DlyCalDt", ">=", start.strftime("%Y-%m-%d"))])   # column is stored as string
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"]); df = df[df["PERMNO"].isin(want)].dropna(subset=["DlyPrc"])
    log.info("  CRSP rows for %d securities: %d (%.0fs)", len(want), len(df), time.time() - t0)
    prices = {}
    for p, g in df.groupby("PERMNO"):
        s = chain_adjusted(g.sort_values("DlyCalDt").drop_duplicates("DlyCalDt", keep="last"))
        if s is not None: prices[etf_permno.get(int(p), str(int(p)))] = s
    pdf = pd.DataFrame(prices); pdf.index = pd.to_datetime(pdf.index); pdf = pdf.sort_index()
    log.info("  CRSP price matrix %s (%.0fs)", pdf.shape, time.time() - t0)
    # ---- 2026 extension from Compustat Security Daily, chained by returns from 2025-12-31 ----
    sd = pd.concat([pd.read_parquet(f, columns=["gvkey", "iid", "tic", "datadate", "prccd", "ajexdi", "trfd", "tpci"]) for f in sorted(glob.glob(str(W / "compustat_security_daily/secd_202[56].parquet")))])
    sd = sd[sd["datadate"] >= pd.Timestamp("2025-12-15")]; sd["adj"] = sd["prccd"] / sd["ajexdi"].replace(0, np.nan) * sd["trfd"].fillna(1.0)
    lk = link[(link["LINKDT"] <= CRSP_END) & (link["LINKENDDT"] >= CRSP_END - pd.Timedelta(days=400))].sort_values(["LPERMNO", "LINKPRIM"]).drop_duplicates("LPERMNO", keep="first")
    key_of = {(r.gvkey, str(r.LIID).zfill(2) if isinstance(r.LIID, str) or not pd.isna(r.LIID) else "01"): str(r.LPERMNO) for r in lk.itertuples(index=False)}
    sd["key"] = [key_of.get((g, str(i).zfill(2))) for g, i in zip(sd["gvkey"], sd["iid"])]
    # Compustat tags ETFs tpci="%" (SPY, GLD, XLK...), NOT "F" (found 2026-09-07: every ETF had ZERO 2026 prices, so the
    # momentum sleeve's SPY regime and sector-ETF tilt ran on frozen 2025 data). Accept both codes and assert coverage below.
    etf_rows = sd[sd["tic"].isin(ETFS) & (sd["tpci"].isin(["F", "%"]))]; sd.loc[etf_rows.index, "key"] = etf_rows["tic"]
    assert set(etf_rows["tic"]) >= set(ETFS), ("ETFs missing from Compustat Security Daily", set(ETFS) - set(etf_rows["tic"]))
    sd = sd.dropna(subset=["key", "adj"]); ext, n_ext = {}, 0
    for k, g in sd.groupby("key"):
        if k not in pdf.columns: continue
        base = pdf[k].loc[:CRSP_END].dropna()
        if base.empty or base.index[-1] < CRSP_END - pd.Timedelta(days=10): continue    # delisted before year-end: no extension
        g = g.sort_values("datadate").drop_duplicates("datadate", keep="last").set_index("datadate")["adj"]
        g = g[g.index >= base.index[-1]]
        if len(g) < 2 or g.index[0] != base.index[-1]: 
            # need the anchor day in secd; fall back to nearest prior secd day
            gg = sd[sd["key"] == k].sort_values("datadate"); prior = gg[gg["datadate"] <= base.index[-1]]
            if prior.empty: continue
            g = pd.concat([pd.Series([prior["adj"].iloc[-1]], index=[base.index[-1]]), g[g.index > base.index[-1]]])
        rel = (g / g.iloc[0]).iloc[1:]; ext[k] = base.iloc[-1] * rel; n_ext += 1
    if ext:
        edf = pd.DataFrame(ext); pdf = pdf.combine_first(edf) if False else pd.concat([pdf, edf[~edf.index.isin(pdf.index)]]).sort_index()
        for k, s in ext.items(): pdf.loc[s.index, k] = s.values
    pdf = pdf[(pdf.index <= pd.Timestamp(END))]
    log.info("  extended %d series into 2026 via Security Daily; matrix now %s, last date %s (%.0fs)", n_ext, pdf.shape, pdf.index.max().date(), time.time() - t0)
    META["n_extended_2026"] = n_ext
    return pdf

def load_fundamentals(link):
    cq = pd.read_parquet(W / "compustat_quarterly.parquet")
    cq["datadate"] = pd.to_datetime(cq["datadate"]); cq["gvkey"] = cq["GVKEY"].astype(str).str.zfill(6)
    import pyarrow.parquet as _pq
    _rawcols = set(_pq.ParquetFile(W / "compustat_fundamentals_quarterly_2026-09_RAW.parquet").schema.names)
    _want = [c for c in cq.columns if c in _rawcols and c not in ("GVKEY", "LPERMNO", "LINKPRIM", "LIID", "LINKTYPE", "LPERMCO", "LINKDT", "LINKENDDT")]
    raw = pd.read_parquet(W / "compustat_fundamentals_quarterly_2026-09_RAW.parquet", columns=_want)
    # keep the frame shape identical so concat is clean -- but NEVER pre-create the link columns:
    # a NaN LPERMNO here survived the merge as LPERMNO (merge produced _x/_y) and dropna() then
    # deleted EVERY new row (found 2026-09-07: fund rows == cq rows, datadate max stuck 2026-03-31).
    _linkcols = ("LPERMNO", "LINKPRIM", "LIID", "LINKTYPE", "LPERMCO", "LINKDT", "LINKENDDT")
    for c in cq.columns:
        if c not in raw.columns and c not in _linkcols: raw[c] = np.nan
    raw["gvkey"] = raw["gvkey"].astype(str).str.zfill(6)
    have = set(zip(cq["gvkey"], cq["datadate"])); new = raw[[(g, d) not in have for g, d in zip(raw["gvkey"], raw["datadate"])]]
    lk = link.sort_values(["gvkey", "LINKPRIM"]).drop_duplicates("gvkey", keep="first")[["gvkey", "LPERMNO"]]
    new = new.merge(lk, on="gvkey", how="left")
    assert "LPERMNO" in new.columns and new.columns.tolist().count("LPERMNO") == 1, new.columns.tolist()
    kept = int(new["LPERMNO"].notna().sum()); unlinked = int(new["LPERMNO"].isna().sum())
    fund = pd.concat([cq, new], ignore_index=True); fund = fund.dropna(subset=["LPERMNO"]); fund["tic"] = fund["LPERMNO"].astype(int).astype(str)
    assert len(fund) == int(cq["LPERMNO"].notna().sum()) + kept, (len(fund), kept)
    assert kept > 0 and fund["datadate"].max() >= pd.Timestamp("2026-06-30"), ("new fundamentals rows were dropped", kept, fund["datadate"].max())
    log.info("  fundamentals: new rows linked to PERMNO %d, unlinkable (no CCM link) %d", kept, unlinked)
    META["fund_new_kept"] = kept; META["fund_new_unlinked"] = unlinked
    fund["avail_date"] = pd.to_datetime(fund["rdq"], errors="coerce").fillna(fund["datadate"] + pd.Timedelta(days=90))
    fund = fund.sort_values(["tic", "avail_date"])
    log.info("  fundamentals: %d rows (%d added from 2026-09 pull), %d PERMNOs, datadate max %s", len(fund), len(new), fund["tic"].nunique(), fund["datadate"].max().date())
    META["fund_rows_added"] = int(len(new))
    return fund

def main():
    t = time.time(); log.info("BUILD v2  START=%s END=%s -> %s", START, END, OUTPUT)
    ci, link, sm_last = load_identifiers()
    sp500, sp400, sp600, ci_by_tic, link = load_membership(ci, link, sm_last)
    needed = set().union(*sp500.values(), *sp400.values(), *sp600.values()); needed = {int(x) for x in needed}
    prices_df = load_prices(needed, ci_by_tic, link)
    features_by_date = compute_features(prices_df)
    fund = load_fundamentals(link)
    # ---- point-in-time fundamentals merge (v1 logic, keyed by PERMNO) ----
    fund_lookup = {}
    for tic, grp in fund.groupby("tic"):
        recs = []; ps = pe = pa = pc = np.nan
        for _, row in grp.sort_values("avail_date").iterrows():
            ad = row["avail_date"]
            if pd.isna(ad): continue
            seqq, niq, saleq, cogsq, dlttq, dlcq, oibdpq, atq, epsfxq = (row.get(k, np.nan) for k in ("seqq", "niq", "saleq", "cogsq", "dlttq", "dlcq", "oibdpq", "atq", "epsfxq"))
            ok = lambda *xs: all(not pd.isna(x) for x in xs)
            fd = {"roe": (niq / seqq * 4) if ok(seqq, niq) and seqq > 0 else np.nan,
                  "gross_margin": ((saleq - cogsq) / saleq) if ok(saleq, cogsq) and saleq > 0 else np.nan,
                  "debt_to_equity": ((float(dlttq if ok(dlttq) else 0) + float(dlcq if ok(dlcq) else 0)) / seqq) if ok(seqq) and seqq > 0 else np.nan,
                  "operating_margin": (oibdpq / saleq) if ok(oibdpq, saleq) and saleq > 0 else np.nan,
                  "net_margin": (niq / saleq) if ok(niq, saleq) and saleq > 0 else np.nan,
                  "revenue_growth_yoy": ((saleq / ps) - 1) if ok(saleq, ps) and ps > 0 else np.nan,
                  "eps_growth_yoy": ((epsfxq - pe) / abs(pe)) if ok(epsfxq, pe) and abs(pe) > 0.01 else np.nan,
                  "asset_growth": ((atq / pa) - 1) if ok(atq, pa) and pa > 0 else np.nan,
                  "gp_assets": ((saleq - cogsq) / atq) if ok(saleq, cogsq, atq) and atq > 0 else np.nan,
                  "net_issuance": ((seqq - pc) / pc) if ok(seqq, pc) and pc > 0 else np.nan}
            recs.append((ad, fd))
            if len(recs) >= 4: ps, pe, pa, pc = (recs[-4][1].get(k, np.nan) for k in ("_saleq", "_epsfxq", "_atq", "_seqq"))
            fd.update({"_saleq": saleq, "_epsfxq": epsfxq, "_atq": atq, "_seqq": seqq})
        if recs: fund_lookup[tic] = sorted(recs, key=lambda x: x[0])
    dates = sorted(features_by_date); cov = []
    for i, d in enumerate(dates):
        n = m = 0
        for sym, feats in features_by_date[d].items():
            n += 1
            if sym in fund_lookup:
                latest = None
                for ad, fd in fund_lookup[sym]:
                    if ad <= d: latest = fd
                    else: break
                if latest: feats.update(latest); m += 1
        if i % 1000 == 0: cov.append((str(d.date()), round(m / max(n, 1), 3)))
    log.info("  fundamentals merged; roe coverage samples %s", cov)
    # earnings signals (zeroed under deployed-parity; kept for key compatibility)
    earnings_signals, revenue_surprise, beat_streak = {}, {}, {}
    for tic, grp in fund.groupby("tic"):
        cs = grp.sort_values("datadate")
        if len(cs) < 5: continue
        bc = 0
        for i in range(4, len(cs)):
            row, prev = cs.iloc[i], cs.iloc[i - 4]; rdq = row.get("rdq")
            if pd.isna(rdq): continue
            ec, ep, rc, rp = row.get("epsfxq"), prev.get("epsfxq"), row.get("saleq"), prev.get("saleq")
            es = (ec - ep) / abs(ep) if pd.notna(ec) and pd.notna(ep) and abs(ep) > 0.01 else np.nan
            rs = (rc - rp) / rp if pd.notna(rc) and pd.notna(rp) and rp > 0 else np.nan
            bc = bc + 1 if (pd.notna(ec) and pd.notna(ep) and ec > ep) else 0
            earnings_signals.setdefault(pd.Timestamp(rdq), {})[tic] = {"eps_surprise": es, "rev_surprise": rs, "beat_streak": bc}
            if not np.isnan(rs): revenue_surprise[tic] = rs
            beat_streak[tic] = bc
    ff = pd.read_parquet(W / "fama_french_5factors_momentum_daily.parquet"); ff["date"] = pd.to_datetime(ff["date"]); ff = ff.set_index("date").sort_index()
    fred = None; ffile = W / "fred_interest_rates_spreads_daily.parquet"
    if ffile.exists(): fred = pd.read_parquet(ffile); fred["date"] = pd.to_datetime(fred["date"]); fred = fred.set_index("date").sort_index()
    etf_df = pd.DataFrame({e: prices_df[e] for e in ETFS if e in prices_df.columns})
    _etf26 = {e: int(prices_df[e].loc["2026-01-01":].notna().sum()) for e in ETFS if e in prices_df.columns}
    assert len(_etf26) == len(ETFS) and min(_etf26.values()) >= 150, ("ETF 2026 coverage", _etf26)
    log.info("  ETF 2026 price counts: %s", _etf26)
    permno_to_ticker = {}
    ci2 = ci.dropna(subset=["Ticker"]).sort_values("SecInfoEndDt").drop_duplicates("PERMNO", keep="last")
    permno_to_ticker = {str(p): t for p, t in zip(ci2.PERMNO, ci2.Ticker)}
    META.update({"start": START, "end": END, "prices_shape": list(prices_df.shape), "feature_dates": len(features_by_date), "built": time.strftime("%Y-%m-%d %H:%M")})
    data = {"prices_df": prices_df, "features_by_date": features_by_date, "sp500_mem": sp500, "sp400_mem": sp400, "sp600_mem": sp600,
            "fin_growth": {}, "ev_data": {}, "estimates_data": {}, "earnings_signals": earnings_signals, "revenue_surprise": revenue_surprise,
            "beat_streak": beat_streak, "price_targets": {}, "fama_french": ff, "fred_rates": fred, "etf_df": etf_df,
            "permno_to_ticker": permno_to_ticker, "build_meta": META}
    with open(OUTPUT, "wb") as f: pickle.dump(data, f, protocol=4)
    log.info("SAVED %s  %.1f GB  total %.0f min  meta=%s", OUTPUT, OUTPUT.stat().st_size / 1e9, (time.time() - t) / 60, META)

if __name__ == "__main__": main()
