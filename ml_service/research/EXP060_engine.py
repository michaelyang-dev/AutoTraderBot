"""EXP-060 (2026-09-17): a SECOND ENGINE — analyst sleeve (IBES PIT) added to the EXP-059 package. Derived from EXP059_frontier.py.
EXP-059 — FRONTIER SEARCH on top of the deployed structure (FINAL @1.49x on the v2 universes).
Switches patched into the clean room (all default OFF = identical to the deployed engine):
  exit_all      : at every tranche day, EVERY book sells names no sleeve wants today (mid-cycle exit, I-29, now cheap: the stride is 5)
  overlay_all   : at every tranche day, ALL books are rescaled to today's vol_scale x gate (prompt overlay; I-03 spirit, cycle 8/13 finding)
  waterfill     : I-02 — cap at 0.10 with the excess redistributed to uncapped names until convergence (instead of clip-then-renormalise)
  min_trade     : no-trade band for resizes as a fraction of book NAV (deployed 0.003)
  slow_vl       : value + lowvol targets refreshed only every N-th rebuild of a book (momentum every rebuild)
  vol_stop      : I-05 — per-name stop = k x annualised vol_60d, clamped [0.25, 0.55] (deployed flat 0.40)
  excl_adds     : I-06 — exclude names that joined the SP1500 within the last N sessions from the momentum sleeve
Run: python3 research/EXP059_frontier.py <8yr|26yr> <stage1|stage2> [arm ...]"""
import os, sys, inspect, textwrap, json, time, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research")
HZ = sys.argv[1]; STAGE = sys.argv[2]; ONLY = sys.argv[3:]; sys.argv = ["x", HZ]
import EXP057_final_on_v2 as E
from main_production_backtest import FastBacktester
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
CACHE = f"research/_v2_{HZ}/exp060"; os.makedirs(CACHE, exist_ok=True)
STARTS = (E.STARTS[0::3] if STAGE == "stage1" else E.STARTS)        # stage1: 8 starts incl. odd months mostly; stage2: all 24
BASE = dict(E.FIN, leverage=1.49)
ARMS = {
    "base":            dict(BASE),
    "P":               dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05),
    # the package + an ANALYST sleeve (IBES, point-in-time) at 15% of the book, other sleeves scaled by 0.85
    "P_x_sue15":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="sue"),
    "P_x_rev15":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="rev3m"),
    "P_x_updown15":    dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="updown"),
    "P_x_rec15":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="rec1m"),
    "P_x_combo15":     dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="combo"),
    # each analyst signal ALONE (100% of the book, same stops / vol overlay / gate / tranches) = raw sleeve quality
    "x_sue100":        dict(BASE, x_w=1.0, x_kind="sue"),
    "x_rev100":        dict(BASE, x_w=1.0, x_kind="rev3m"),
    "x_updown100":     dict(BASE, x_w=1.0, x_kind="updown"),
    "x_rec100":        dict(BASE, x_w=1.0, x_kind="rec1m"),
    "x_combo100":      dict(BASE, x_w=1.0, x_kind="combo"),
    # batch 2: price-based second engines (short-term reversal in uptrends, calendar seasonality, long-term reversal)
    "x_str100":        dict(BASE, x_w=1.0, x_kind="str"),
    "x_season100":     dict(BASE, x_w=1.0, x_kind="season"),
    "x_ltr100":        dict(BASE, x_w=1.0, x_kind="ltr"),
    "P_x_str15":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="str"),
    "P_x_season15":    dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="season"),
    "P_x_ltr15":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="ltr"),
    # batch 3: return decomposition (overnight momentum) and institutional breadth (13F)
    "x_onm100":        dict(BASE, x_w=1.0, x_kind="onm"),
    "x_onmraw100":     dict(BASE, x_w=1.0, x_kind="onm_raw"),
    "x_inst100":       dict(BASE, x_w=1.0, x_kind="inst"),
    "x_instshr100":    dict(BASE, x_w=1.0, x_kind="inst_shr"),
    "P_x_onm15":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="onm"),
    "P_x_inst15":      dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="inst"),
    # batch 3b: a different ASSET as the sleeve (gold trend / always; sector-ETF trend) — 10% and 20% of the book
    "P_x_gldtrend10":  dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.10, x_kind="gld_trend"),
    "P_x_gldtrend20":  dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.20, x_kind="gld_trend"),
    "P_x_gldalways10": dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.10, x_kind="gld_always"),
    "P_x_secttrend15": dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="sect_trend"),
    # batch 4: long-term reversal sleeve — weight, name count, uptrend filter, and cost 2x
    "P_x_ltr10":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.10, x_kind="ltr"),
    "P_x_ltr20":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.20, x_kind="ltr"),
    "P_x_ltr25":       dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.25, x_kind="ltr"),
    "P_x_ltr15_n5":    dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="ltr", x_n=5),
    "P_x_ltr15_n15":   dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="ltr", x_n=15),
    "P_x_ltr15_trend": dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="ltr", x_trend=True),
    "P_x_ltr15_cost2": dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, x_w=0.15, x_kind="ltr", cost_mult=2.0),
    "P_cost2":         dict(BASE, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05, cost_mult=2.0),
}

# ───────────────────────── EXP-060: analyst sleeve from IBES point-in-time features ─────────────────────────
_IBES = pd.read_parquet("research/_exp060/ibes_pit.parquet")
_IBES_DATES = np.array(sorted(_IBES["asof"].unique()))
_IBES_BY = {d: g.set_index("cusip8") for d, g in _IBES.groupby("asof")}
# PERMNO -> [(start, end, cusip8)] from CRSP security-info eras (the universe's members are PERMNO strings)
_ci = pd.read_parquet("data/wrds/crsp_security_info.parquet", columns=["PERMNO", "CUSIP", "SecInfoStartDt", "SecInfoEndDt"])
_ci["c8"] = _ci.CUSIP.astype(str).str[:8]; _ci = _ci[_ci.c8.str.len() == 8]
_ci["s"] = pd.to_datetime(_ci.SecInfoStartDt).values.astype("datetime64[D]"); _ci["e"] = pd.to_datetime(_ci.SecInfoEndDt).values.astype("datetime64[D]")
_P2C = {}
# generic (asof, cusip8)-keyed PIT tables for other second-engine kinds; loaded lazily
_PIT = {}
def _pit(name):
    if name not in _PIT:
        f = f"research/_exp060/{name}_pit.parquet"
        if not os.path.exists(f): _PIT[name] = None
        else:
            t = pd.read_parquet(f); t["asof"] = pd.to_datetime(t["asof"])
            _PIT[name] = (np.array(sorted(t["asof"].unique())), {d: g.set_index("cusip8") for d, g in t.groupby("asof")})
    return _PIT[name]
def _pit_scores(name, col, d64, members, max_age_days):
    """Latest table row on/before d (<= max_age_days old), mapped PERMNO -> cusip8 -> row[col]; returns Series keyed by PERMNO."""
    tab = _pit(name)
    if tab is None: return pd.Series(dtype=float)
    dates, by = tab
    j = np.searchsorted(dates, d64, side="right") - 1
    if j < 0 or (d64 - dates[j]) / np.timedelta64(1, "D") > max_age_days: return pd.Series(dtype=float)
    g = by[dates[j]]; dD = d64.astype("datetime64[D]"); out = {}
    for s_ in members:
        c_ = _cusip_at(s_, dD)
        if c_ is not None and c_ in g.index:
            v = g.at[c_, col] if not isinstance(g.at[c_, col], pd.Series) else g.at[c_, col].iloc[-1]
            if v == v: out[s_] = float(v)
    return pd.Series(out, dtype=float)
for r in _ci.itertuples(index=False):
    _P2C.setdefault(str(int(r.PERMNO)), []).append((r.s, r.e, r.c8))
def _cusip_at(permno, d64):
    eras = _P2C.get(str(permno))
    if not eras: return None
    for s_, e_, c_ in eras:
        if s_ <= d64 <= e_: return c_
    return eras[-1][2] if d64 > eras[-1][1] else None
_PX_CACHE = {}
def _price_kind(d, members, kind, cfg, features, uni):
    """Price-based second-engine candidates (all from the universe's own price panel, PIT by construction):
      str    short-term reversal: score = -ret_20d, restricted to names above their SMA200 (buy the dip in an uptrend)
      season calendar seasonality (Heston-Sadka): mean return of the SAME calendar month over the prior 10 years
      ltr    long-term reversal (DeBondt-Thaler): score = -(return over months 13..60)
    Vectorised over the price panel; monthly panel cached per universe object."""
    fd = features or {}
    if kind == "str":
        return pd.Series({s_: -fd[s_]["ret_20d"] for s_ in members if s_ in fd and fd[s_].get("ret_20d") is not None
                          and fd[s_].get("dist_sma200", -1) > 0}, dtype=float)
    if uni is None: return pd.Series(dtype=float)
    px = uni.prices; dts = pd.Timestamp(d)
    cols = [s_ for s_ in members if s_ in px.columns]
    if not cols: return pd.Series(dtype=float)
    if kind == "ltr":
        hist = px.loc[:dts]
        if len(hist) < 252 * 5 + 1: return pd.Series(dtype=float)
        p12 = hist.iloc[-252][cols]; p60 = hist.iloc[-252 * 5][cols]
        r = (p12 / p60 - 1.0); r = r[(p60 > 0) & p12.notna() & p60.notna()]
        return -r
    # season
    key = id(uni)
    if key not in _PX_CACHE:
        _PX_CACHE[key] = px.resample("ME").last().pct_change()
    m = _PX_CACHE[key]
    cutoff = dts - pd.Timedelta(days=40)
    rows = m[(m.index.month == dts.month) & (m.index < cutoff) & (m.index >= cutoff - pd.Timedelta(days=366 * 10))]
    if len(rows) < 5: return pd.Series(dtype=float)
    sub = rows[cols]; cnt = sub.notna().sum()
    sc = sub.mean(); sc = sc[cnt >= 5]
    return sc.dropna()


def sleeve_x(d, members, cfg, features=None, uni=None):
    """Top-N (equal weight) of the universe members by one analyst signal as of the latest IBES statistical
    period on/before d (must be <= 45 days old). kinds: sue (quarterly EPS surprise score, <= 90 days old),
    rev3m / rev1m (FY1 mean-estimate revision), updown ((#up - #down)/#estimates), rec1m (recommendation
    upgrades), combo (mean rank of sue, rev3m, updown). Needs >= 3 estimates. Positive score only."""
    d64 = np.datetime64(pd.Timestamp(d))
    j = np.searchsorted(_IBES_DATES, d64, side="right") - 1
    if os.getenv("X_DEBUG") and sleeve_x._n < 6:
        sleeve_x._n += 1
        g0 = _IBES_BY[_IBES_DATES[j]] if j >= 0 else None
        ms = list(members)[:6]
        mt = sum(1 for s_ in members if g0 is not None and (_cusip_at(s_, d64.astype("datetime64[D]")) in g0.index)) if g0 is not None else 0
        print(f"[x_debug] d={pd.Timestamp(d).date()} kind={cfg.get('x_kind')} x_w={cfg.get('x_w')} members={len(members)} sample={ms} "
              f"asof={_IBES_DATES[j] if j>=0 else None} matched={mt}", flush=True)
    if j < 0: return {}
    asof = _IBES_DATES[j]
    if (d64 - asof) / np.timedelta64(1, "D") > 45 and cfg.get("x_kind", "combo") in ("sue", "rev3m", "rev1m", "updown", "rec1m", "combo"): return {}
    g = _IBES_BY[asof]; kind = cfg.get("x_kind", "combo"); n = int(cfg.get("x_n", 10))
    dD = d64.astype("datetime64[D]")
    c2p = {}
    for s_ in members:
        c_ = _cusip_at(s_, dD)
        if c_ is not None and c_ in g.index and c_ not in c2p: c2p[c_] = s_
    if not c2p and kind not in ("str", "season", "ltr", "onm", "onm_raw", "inst", "inst_shr", "gld_trend", "gld_always", "sect_trend"): return {}
    sub = g.loc[list(c2p)] if c2p else g.iloc[0:0]
    sub = sub[sub["NUMEST"].fillna(0) >= 3]
    sub.index = [c2p[c_] for c_ in sub.index]          # back to PERMNO keys
    if kind == "sue":
        sc = sub["sue"].where(sub["sue_days"].fillna(999) <= 90)
    elif kind == "combo":
        parts = []
        for c, cond in (("sue", sub["sue_days"].fillna(999) <= 90), ("rev3m", None), ("updown", None)):
            v = sub[c] if cond is None else sub[c].where(cond)
            parts.append(v.rank(pct=True))
        sc = pd.concat(parts, axis=1).mean(axis=1, skipna=False) - 0.5
    elif kind in ("str", "season", "ltr"):
        sc = _price_kind(d, members, kind, cfg, features, uni)
    elif kind in ("gld_trend", "gld_always", "sect_trend"):
        # a DIFFERENT ASSET as the sleeve: gold (GLD, from 2004-11) held when above its 200-session SMA (else the sleeve
        # sits in cash), or always; sect_trend = the 3 sector ETFs with the highest 6m return that are above SMA200
        px = uni.prices if uni is not None else None
        if px is None: return {}
        hist = px.loc[:pd.Timestamp(d)]
        if kind.startswith("gld"):
            if "GLD" not in hist.columns: return {}
            g_ = hist["GLD"].dropna()
            if len(g_) < 200: return {}
            if kind == "gld_always" or g_.iloc[-1] > g_.tail(200).mean(): return {"GLD": 1.0}
            return {}
        sects = [e for e in ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC") if e in hist.columns]
        sc_ = {}
        for e in sects:
            h_ = hist[e].dropna()
            if len(h_) >= 200 and h_.iloc[-1] > h_.tail(200).mean() and len(h_) > 126: sc_[e] = float(h_.iloc[-1] / h_.iloc[-126] - 1)
        top_ = sorted(sc_, key=sc_.get, reverse=True)[:3]
        return {e: 1.0 / len(top_) for e in top_} if top_ else {}
    elif kind == "onm":       # overnight-minus-intraday 12m momentum (monthly table, <= 45d old)
        sc = _pit_scores("onm", "onm", d64, members, 45)
    elif kind == "onm_raw":   # overnight 12m sum alone
        sc = _pit_scores("onm", "on252", d64, members, 45)
    elif kind == "inst":      # 13F breadth: q/q change in number of holders (asof = quarter end + 45d, <= 120d old)
        sc = _pit_scores("inst", "d_nmgr", d64, members, 120)
    elif kind == "inst_shr":  # q/q change in aggregate institutional shares
        sc = _pit_scores("inst", "d_shr", d64, members, 120)
    else:
        sc = sub[kind]
    if isinstance(sc, dict): return sc
    sc = sc.dropna(); sc = sc[sc > 0]
    if cfg.get("x_trend") and features is not None:
        sc = sc[[features.get(s, {}).get("dist_sma200", -1) > -0.15 for s in sc.index]]
    top = sc.sort_values(ascending=False).head(n)
    if os.getenv("X_DEBUG") and sleeve_x._n <= 6:
        print(f"[x_debug]   scored={len(sc)} picks={list(top.index)[:6]}", flush=True)
    if len(top) == 0: return {}
    return {s: 1.0 / len(top) for s in top.index}

sleeve_x._n = 0

def _engine():
    src = inspect.getsource(CleanRoom.run)
    reps = [
        ("dd=float(((v - v.cummax()) / v.cummax()).min()))", "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"),
        ("cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0", 'cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0 * float(cfg.get("cost_mult", 1.0))'),
        # per-name stop level (vol_stop) + stop check uses it
        ("        stop = 0.40\n", "        stop = 0.40; vol_stop_k = cfg.get('vol_stop'); stop_of = {}\n"),
        ("                        if p <= peaks[t][s] * (1.0 - stop):\n", "                        if p <= peaks[t][s] * (1.0 - stop_of.get(s, stop)):\n"),
        # remember last sleeve targets per tranche for slow_vl, and expose today's 'wanted' set for exit_all
        ("        navs, last = [], {}\n", "        navs, last = [], {}; vl_cache = {}; vl_count = {}; adds_seen = {}; prev_mem = None\n"),
        ("                mem = self.uni.get_sp500(d)\n                mv = strategy_value(self.uni, d, mem, top_n=10)\n",
         "                mem = self.uni.get_sp500(d)\n                if cfg.get('excl_adds'):\n                    if prev_mem is not None:\n                        for s_ in mem - prev_mem: adds_seen[s_] = i\n                    prev_mem = set(mem)\n                    m1 = {s_: w_ for s_, w_ in (m1 or {}).items() if i - adds_seen.get(s_, -10**9) > int(cfg['excl_adds'])}\n                mv = strategy_value(self.uni, d, mem, top_n=10)\n                if cfg.get('slow_vl'):\n                    vl_count[t] = vl_count.get(t, 0) + 1\n                    if (vl_count[t] - 1) % int(cfg['slow_vl']) == 0 or t not in vl_cache: vl_cache[t] = (mv, m5)\n                    else: mv, m5 = vl_cache[t]\n"),
        # I-09 / I-10: re-rank the momentum sleeve's candidates (sleeve called with top_n=30) and keep 5 equal-weight
        ("                if m1 is None:\n                    m1 = last.get(\"m\", {})\n",
         "                if m1 is None:\n                    m1 = last.get(\"m\", {})\n                if m1 and (cfg.get('mom_ens') or cfg.get('mom_52wh')):\n                    fd0 = self.bt.features_by_date.get(d, {}); sc = {}\n                    if cfg.get('mom_ens'):\n                        import scipy.stats as _ss\n                        cands = [s_ for s_ in m1 if s_ in fd0]\n                        cols_ = [('ret_252d', 'ret_20d'), ('ret_126d', 'ret_20d'), ('ret_60d', 'ret_20d')]\n                        ranks = np.zeros(len(cands))\n                        for a_, b_ in cols_:\n                            v_ = np.array([fd0[s_].get(a_, np.nan) - fd0[s_].get(b_, 0.0) for s_ in cands], dtype=float); v_ = np.where(np.isnan(v_), -9, v_); ranks += _ss.rankdata(v_)\n                        sc = dict(zip(cands, ranks))\n                    else:\n                        for s_ in m1:\n                            h_ = px[s_].loc[:d].dropna().tail(252)\n                            if len(h_) >= 200 and h_.max() > 0: sc[s_] = float(h_.iloc[-1] / h_.max())\n                    top_ = sorted(sc, key=sc.get, reverse=True)[:5]\n                    m1 = {s_: 1.0 / len(top_) for s_ in top_} if top_ else m1\n"),
        # I-14 VIX term structure gate
        ("                dr = dr * vs\n", "                dr = dr * vs\n                if cfg.get('vix_gate') and getattr(self, '_vix', None) is not None:\n                    vv = self._vix.loc[:d]\n                    if len(vv) > 1 and vv.iloc[-2, 0] > vv.iloc[-2, 1]: dr = dr * float(cfg['vix_gate'])\n"),
        # batch 3: equal-weight momentum picks; min-weight filter; stop level; overlay lookback
        ("                comb = {s: v for s, v in comb.items() if v >= 0.005}\n", "                comb = {s: v for s, v in comb.items() if v >= float(cfg.get('min_weight', 0.005))}\n"),
        ("                m5 = strategy5_lowvol_quality(d, self.uni, di)\n", "                if cfg.get('mom_equal') and m1: m1 = {s_: 1.0 / len(m1) for s_ in m1}\n                if cfg.get('mom_ivol') and m1:\n                    fd_ = self.bt.features_by_date.get(d, {}); iv_ = {}\n                    for s_ in m1:\n                        v_ = fd_.get(s_, {}).get('vol_60d', np.nan)\n                        if v_ == v_ and v_ > 0: iv_[s_] = 1.0 / v_\n                    if len(iv_) == len(m1):\n                        t_ = sum(iv_.values()); m1 = {s_: x_ / t_ for s_, x_ in iv_.items()}\n                    else: m1 = {s_: 1.0 / len(m1) for s_ in m1}\n                m5 = strategy5_lowvol_quality(d, self.uni, di)\n"),
        ("        stop = 0.40; vol_stop_k", "        stop = float(cfg.get('stop', 0.40)); vol_stop_k"),
        # batch 6: value/lowvol sleeves equal-weighted
        ("                m3 = strategy3_sector_rotation(d, self.uni, di)\n",
         "                m3 = strategy3_sector_rotation(d, self.uni, di)\n                if cfg.get('vl_equal'):\n                    if mv: mv = {s_: 1.0 / len(mv) for s_ in mv}\n                    if m5: m5 = {s_: 1.0 / len(m5) for s_ in m5}\n"),
        # batch 6: DAILY de-risk-only overlay on non-tranche days (same maths as the weekly one: today's vol x gate target per book)
        ("            nav = cash + mtm()\n\n            if i in sched and nav > 0:\n",
         "            nav = cash + mtm()\n            if cfg.get('overlay_daily') and i not in sched and nav > 0 and len(navhist) >= 41:\n                dr_ = derisk_v if (gate_pct and gate.get(d, 0.5) >= gate_pct) else 1.0\n                rr_ = np.diff(np.array(navhist[-41:])) / np.array(navhist[-41:-1]); rv_ = float(np.std(rr_)) * np.sqrt(252)\n                vs_ = min(1.0, max(0.30, VOL_TARGET / rv_)) if (use_ov and rv_ > 0.01) else 1.0\n                tgt_ = (nav / K) * lev * dr_ * vs_\n                for t2 in range(K):\n                    if not books[t2]: continue\n                    gross2 = sum(q2 * prc.get(s2, lastpx.get(s2, 0.0)) for s2, q2 in books[t2].items())\n                    if gross2 <= 0: continue\n                    f2 = tgt_ / gross2\n                    if f2 < float(cfg.get('overlay_thr', 0.95)):\n                        for s2 in list(books[t2]):\n                            p2 = prc.get(s2)\n                            if not p2: continue\n                            q_old = books[t2][s2]; q_new = int(q_old * f2); dq2 = q_new - q_old\n                            if dq2 == 0 or abs(dq2 * p2) < (nav / K) * 0.003: continue\n                            cash -= dq2 * p2; cash -= abs(dq2 * p2) * cost_r\n                            if q_new > 0: books[t2][s2] = q_new\n                            else: books[t2].pop(s2, None); peaks[t2].pop(s2, None)\n                nav = cash + mtm()\n\n            if i in sched and nav > 0:\n"),
        # water-filling cap (I-02)
        ("                comb = {s: min(v, 0.10) for s, v in comb.items() if v > 0}\n                g = sum(comb.values())\n                if g > 1.0:\n                    comb = {s: v / g for s, v in comb.items()}\n",
         "                comb = {s: v for s, v in comb.items() if v > 0}\n                if cfg.get('waterfill'):\n                    g0 = sum(comb.values()); target = min(g0, 1.0); comb = {s: v / g0 * target for s, v in comb.items()} if g0 > 0 else {}\n                    for _ in range(20):\n                        over = {s for s, v in comb.items() if v > 0.10 + 1e-12}\n                        if not over: break\n                        excess = sum(comb[s] - 0.10 for s in over); free = {s: v for s, v in comb.items() if s not in over}; fs = sum(free.values())\n                        for s in over: comb[s] = 0.10\n                        if fs <= 0: break\n                        for s in free: comb[s] += excess * free[s] / fs\n                else:\n                    comb = {s: min(v, 0.10) for s, v in comb.items()}\n                    g = sum(comb.values())\n                    if g > 1.0:\n                        comb = {s: v / g for s, v in comb.items()}\n"),
        # vol-normalised stop levels for today's targets
        ("                dr = derisk_v if (gate_pct and gate.get(d, 0.5) >= gate_pct) else 1.0\n",
         "                dr = derisk_v if (gate_pct and gate.get(d, 0.5) >= gate_pct) else 1.0\n                if vol_stop_k:\n                    fd_ = self.bt.features_by_date.get(d, {})\n                    for s_ in comb:\n                        v60 = fd_.get(s_, {}).get('vol_60d', np.nan)\n                        if v60 == v60 and v60 > 0: stop_of[s_] = min(0.55, max(0.25, vol_stop_k * v60))\n"),
        # min-trade band
        ("                    if abs(dq * p) < tnav * 0.003:\n", "                    if abs(dq * p) < tnav * float(cfg.get('min_trade', 0.003)):\n"),
        # I-03 ex-ante (holdings-based) vol instead of / in addition to 40d realised NAV vol
        ("                vs = 1.0\n                if use_ov and len(navhist) >= 41:\n                    rr = np.diff(np.array(navhist[-41:])) / np.array(navhist[-41:-1])\n                    rv = float(np.std(rr)) * np.sqrt(252)\n                    if rv > 0.01:\n                        vs = min(1.0, max(0.30, VOL_TARGET / rv))\n",
         "                vs = 1.0\n                LB_ = int(cfg.get('vol_lookback', 40))\n                if use_ov and len(navhist) >= LB_ + 1:\n                    rr = np.diff(np.array(navhist[-(LB_+1):])) / np.array(navhist[-(LB_+1):-1])\n                    rv = float(np.std(rr)) * np.sqrt(252)\n                    if cfg.get('exante_vol') or cfg.get('exante_max'):\n                        hold = {}\n                        for b_ in books:\n                            for s_, q_ in b_.items(): hold[s_] = hold.get(s_, 0) + q_\n                        syms_ = [s_ for s_ in hold if s_ in px.columns]\n                        if syms_ and nav > 0:\n                            R_ = px[syms_].loc[:d].tail(61).pct_change().dropna(how='all').fillna(0.0)\n                            w_ = np.array([hold[s_] * prc.get(s_, lastpx.get(s_, 0.0)) / nav for s_ in syms_])\n                            if len(R_) >= 30:\n                                pr_ = R_.values @ w_; ex_ = float(np.std(pr_)) * np.sqrt(252)\n                                rv = ex_ if cfg.get('exante_vol') else max(rv, ex_)\n                    if rv > 0.01:\n                        vs = min(1.0, max(0.30, VOL_TARGET / rv))\n"),
        # exit_all + overlay_all: after this book's rebuild, touch the OTHER books
        ("                nav = cash + mtm()\n\n            prev_debit = max(0.0, -cash)\n",
         "                nav = cash + mtm()\n                wanted = set(comb)\n                if cfg.get('exit_all') or cfg.get('overlay_all') or cfg.get('overlay_down'):\n                    for t2 in range(K):\n                        if t2 == t: continue\n                        if cfg.get('exit_all'):\n                            for s2 in list(books[t2]):\n                                if s2 not in wanted:\n                                    p2 = prc.get(s2, lastpx.get(s2, 0.0)); q2 = books[t2].pop(s2); peaks[t2].pop(s2, None)\n                                    cash += q2 * p2; cash -= abs(q2 * p2) * cost_r\n                        if (cfg.get('overlay_all') or cfg.get('overlay_down')) and books[t2]:\n                            gross2 = sum(q2 * prc.get(s2, lastpx.get(s2, 0.0)) for s2, q2 in books[t2].items()); tgt2 = tnav * lev * dr\n                            f2 = (tgt2 / gross2) if gross2 > 0 else None\n                            if f2 is not None and (tgt2 <= 0 or abs(f2 - 1) > 0.05) and (not cfg.get('overlay_down') or f2 < float(cfg.get('overlay_thr', 0.95))):\n                                for s2 in list(books[t2]):\n                                    p2 = prc.get(s2)\n                                    if not p2: continue\n                                    q_old = books[t2][s2]; q_new = int(q_old * f2); dq2 = q_new - q_old\n                                    if dq2 == 0 or abs(dq2 * p2) < tnav * 0.003: continue\n                                    cash -= dq2 * p2; cash -= abs(dq2 * p2) * cost_r\n                                    if q_new > 0: books[t2][s2] = q_new\n                                    else: books[t2].pop(s2, None); peaks[t2].pop(s2, None)\n                    nav = cash + mtm()\n\n            prev_debit = max(0.0, -cash)\n"),
        # batch 7: combiner cap as a switch (default 0.10 = clean-room convention; 0.15 = live signal server)
        ("                    comb = {s: min(v, 0.10) for s, v in comb.items()}\n", "                    comb = {s: min(v, float(cfg.get('cap', 0.10))) for s, v in comb.items()}\n"),
        # batch 8: vol-clamp floor and vol target as switches (weekly path only; the daily-overlay block keeps the constants)
        ("                        vs = min(1.0, max(0.30, VOL_TARGET / rv))\n", "                        vs = min(1.0, max(float(cfg.get('vs_floor', 0.30)), float(cfg.get('vol_target', VOL_TARGET)) / rv))\n"),
        # batch 12: value / lowvol name counts as switches
        ("                mv = strategy_value(self.uni, d, mem, top_n=10)\n", "                mv = strategy_value(self.uni, d, mem, top_n=int(cfg.get('val_n', 10)))\n"),
        ("                m5 = strategy5_lowvol_quality(d, self.uni, di)\n", "                m5 = strategy5_lowvol_quality(d, self.uni, di, top_n=int(cfg.get('lv_n', 10)))\n"),
        # EXP-060: analyst sleeve "x" at cfg x_w, other bull sleeves scaled by (1 - x_w)
        ("                last = {\"m\": m1, \"l\": m5, \"s\": m3}\n",
         "                mx = sleeve_x(d, mem, cfg, self.bt.features_by_date.get(d, {}), self.uni) if cfg.get('x_w') else {}\n                last = {\"m\": m1, \"l\": m5, \"s\": m3}\n"),
        ("                     else {\"mom\": cfg.get(\"mom_w\", .5), \"val\": cfg.get(\"val_w\", .35),\n                           \"s5\": cfg.get(\"lv_w\", .15), \"s3\": 0.0})\n",
         "                     else {\"mom\": cfg.get(\"mom_w\", .5) * (1 - cfg.get('x_w', 0.0)), \"val\": cfg.get(\"val_w\", .35) * (1 - cfg.get('x_w', 0.0)),\n                           \"s5\": cfg.get(\"lv_w\", .15) * (1 - cfg.get('x_w', 0.0)), \"s3\": 0.0, \"x\": cfg.get('x_w', 0.0)})\n"),
        ("                w = {k: w[k] * bl + bw.get(k, 0) * (1 - bl) for k in w}\n",
         "                w = {k: w[k] * bl + bw.get(k, 0) * (1 - bl) for k in w}\n                if 'x' not in w: w['x'] = 0.0\n"),
        ("                for key, src in ((\"mom\", m1), (\"val\", mv), (\"s5\", m5), (\"s3\", m3)):\n",
         "                for key, src in ((\"mom\", m1), (\"val\", mv), (\"s5\", m5), (\"s3\", m3), (\"x\", mx)):\n"),
    ]
    for a, b in reps: assert src.count(a) == 1, a[:80]; src = src.replace(a, b)
    V.END = E.END; ns = dict(V.__dict__); ns['sleeve_x'] = sleeve_x; assert ns["END"] == E.END; exec(compile(textwrap.dedent(src), "<exp059>", "exec"), ns); CleanRoom.run = ns["run"]
def st(v): return E.st(v)
def main():
    t0 = time.time(); _engine(); cr = None
    names = [a for a in ARMS if (not ONLY or a in ONLY)]
    for nm in names:
        f = f"{CACHE}/{STAGE}_{nm}.parquet"
        if os.path.exists(f): continue
        if nm == "base" and STAGE == "stage2" and os.path.exists(f"{E.CACHE}/FINAL_1.49.parquet"):
            pd.read_parquet(f"{E.CACHE}/FINAL_1.49.parquet").to_parquet(f); continue
        if cr is None:
            cr = CleanRoom(FastBacktester(universe_path=E.PATH))
            try:
                vx = pd.read_parquet("data/enhanced_data/vix_cache.parquet"); vx.index = pd.to_datetime(vx.index); cr._vix = vx[["^VIX", "^VIX3M"]].dropna().sort_index()
            except Exception as e:
                cr._vix = None; print("  (vix cache unavailable:", e, ")")
        curves = {}
        for s_ in STARTS:
            t1 = time.time(); curves[s_] = cr.run(s_, ARMS[nm])["curve"]; dt = time.time() - t1
            print(f"    {nm} start {s_} {dt:5.0f}s", file=sys.stderr, flush=True)
            if dt > 600: print(f"    !!! {nm} start {s_} took {dt:.0f}s — pathological run", file=sys.stderr, flush=True)
        pd.DataFrame(curves).to_parquet(f); print(f"  {nm:<16} {time.time()-t0:6.0f}s", flush=True)
    # analyse
    B = pd.read_parquet(f"{CACHE}/{STAGE}_base.parquet"); cols = [c for c in B.columns if c in STARTS]; b = np.array([st(B[c]) for c in cols])
    print(f"\n{HZ} {STAGE} ({len(cols)} starts) — FINAL@1.49 base {b[:,0].mean():+.2%} / {b[:,1].mean():.3f} / {b[:,2].mean():.1%}")
    print(f"  {'arm':<16}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dShrp':>8}{'+Shrp':>7}{'dMaxDD':>8}{'+DD':>6}{'yrs>':>6}")
    for nm in names:
        f = f"{CACHE}/{STAGE}_{nm}.parquet"
        if nm == "base" or not os.path.exists(f): continue
        A = pd.read_parquet(f); a = np.array([st(A[c]) for c in cols]); d = a - b
        if np.abs(d).max() < 1e-12: print(f"  !! {nm}: IDENTICAL to base on every start — the switch never fired (harness bug); cache removed"); os.remove(f); continue
        yl = [E.yearly(B[c]) for c in cols]; ya = [E.yearly(A[c]) for c in cols]; ys = sorted(set().union(*[set(x) for x in yl])); wins = sum(1 for y in ys if np.mean([xa[y] - xb[y] for xa, xb in zip(ya, yl) if y in xa and y in xb]) > 0)
        print(f"  {nm:<16}{a[:,0].mean():>+9.2%}{a[:,1].mean():>8.3f}{a[:,2].mean():>8.1%}{d[:,0].mean()*100:>+7.2f}p{d[:,1].mean():>+8.3f}{int((d[:,1]>0).sum()):>4}/{len(cols)}{d[:,2].mean()*100:>+7.2f}p{int((d[:,2]>0).sum()):>3}/{len(cols)}{wins:>4}/{len(ys)}")
    print(f"  total {time.time()-t0:.0f}s")
if __name__ == "__main__": main()
