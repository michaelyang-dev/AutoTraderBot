"""EXP-063 — WHOLE-SHARE ROUNDING at the live account size: per-book truncation (live) vs account-level rounding.

QUESTION (owner, 2026-10-01). LITE and SNDK were BUY signals at every tranche rebuild, yet IBKR held 0 shares: each of
the 4 books sizes its own slice and truncates to whole shares, and a bear-weight momentum slot (~$567 of a NAV/4 book)
is 0.58 of a ~$1,000 share -> int() -> 0, in every book. Would rounding the SUM of the books' fractional targets at the
ACCOUNT level (4 x 0.58 = 2 shares) help — and is it free of harm? Nothing changes live until this is answered.

WHY THE EXISTING NUMBERS CANNOT ANSWER IT. The clean room (a) compounds from $50k, so within a few years it sizes an
account several times the live one, where truncation is negligible (EXP-059 batch 11: $60k vs $1M only -0.22pp because
the $60k run "leaves the small regime quickly"); and (b) counts shares on the total-return-ADJUSTED price, which
scripts/build_universe_v2.chain_adjusted anchors at the security's LAST price and chains back — a 2005 share of a stock
that later split 56:1 is priced at ~1/56 of what it cost. Both understate truncation. Here:
  raw_px    : shares are counted at the REAL close of that day (CRSP DlyClose/|DlyPrc| to 2025-12-31; Compustat
              Security Daily prccd for 2026, linked to PERMNO exactly as build_universe_v2.load_prices does); P&L stays
              on the total-return series (book quantities are kept in total-return units). Verified: 99%+ coverage,
              == adjusted at the 2025-12-31 anchor, == the live vendor's 2026 closes for LITE/SNDK/MU.
CAPITAL MODELS (R is the primary evidence; C and F must agree in sign):
  R<k> ref_cap  : the account compounds exactly like the validated engine, but whole shares are counted as a $k
                  account would count them (real price x NAV / ref_cap) — every year traded with that size's rounding.
  C62           : start at $62k at real prices and compound — the realistic forward path (rounding fades with growth).
  F<k> fixed_cap: swept back to $k at every close. STRESS VIEW ONLY: the sweep keeps positions while withdrawing
                  profits, so leverage drifts up after gains and down after losses (F62_frac != compounding frac).
`sanity` asserts: R62_frac == compounding frac (exact); F-sweep scale-consistent at 2^14 x (exact); the account layer
is lossless without rounding (exact); book and account rounding converge to fractional at a $1e12 reference size.
ROUNDING RULES (books are virtual ledgers; the broker position is what is actually held, traded and charged):
  book_floor : every book holds int(target / real price)                    <- THE LIVE RULE (ibkr_engine, clean room)
  book_round : every book rounds to nearest, never rounding up through the 15% cap
  acct_round : books hold fractional virtual shares; the account holds round(sum of books) — never rounding up through
               15% of NAV — and trades only when that whole number changes, one net order per name per day
  acct_floor : as acct_round with floor(sum)
  frac       : fractional shares everywhere (unattainable at IBKR): the ceiling for any rounding rule
  acct_frac  : frac with account-level net execution — separates the netting effect from the rounding effect
COSTS: 10 bps on actual trades (book rules: each book order, as live; account rules: the account's net order);
min_comm adds a per-order minimum (IBKR fixed plan $1); cost_mult scales. Overlay trims keep floor semantics.
PARITY GUARD: with every new switch off the patched engine must reproduce the EXP-059-patched clean room exactly
(`validate` runs both engines on the same start in one process and also compares the cached stage-2 curves).

Run from ml_service/:
  python3 research/EXP063_account_rounding.py <8yr|26yr> validate                 parity guard
  python3 research/EXP063_account_rounding.py <8yr|26yr> raw                      build + sanity-check real prices
  python3 research/EXP063_account_rounding.py <8yr|26yr> sanity                   plumbing identities (must PASS)
  python3 research/EXP063_account_rounding.py <8yr|26yr> <stage1|stage2> run <arm> [arm ...]   (EXP063_SHARD=k/n)
  python3 research/EXP063_account_rounding.py <8yr|26yr> <stage1|stage2> analyse [baseline]
"""
import ast
import inspect
import json
import os
import sys
import textwrap
import time

import numpy as np
import pandas as pd

HZ = sys.argv[1]
STAGE = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else ""
ONLY = sys.argv[4:]
sys.argv = ["x", HZ]
import EXP057_final_on_v2 as E  # noqa: E402  (chdirs to ml_service; PATH / STARTS / END / FIN for the horizon)
import VERIFY2_cleanroom as V  # noqa: E402
from VERIFY2_cleanroom import CleanRoom  # noqa: E402

CACHE = f"research/_v2_{HZ}/exp063"
os.makedirs(CACHE, exist_ok=True)
STARTS = E.STARTS[0::3] if STAGE == "stage1" else E.STARTS
PKG = dict(E.FIN, leverage=1.49, cap=0.15, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05)


def R(cap, rule, **kw):
    """PRIMARY model. The account compounds exactly like the validated engine, but whole shares are counted as a
    `cap`-sized account would count them (real price x NAV / cap): every year is traded with that size's rounding."""
    return dict(PKG, raw_px=True, ref_cap=float(cap), initial_capital=62_000.0, rounding=rule, **kw)


def F(cap, rule, **kw):
    """STRESS VIEW ONLY. The account is swept back to `cap` at every close at real prices. The sweep withdraws profits
    but keeps the positions, so leverage drifts UP after gains and DOWN after losses — not how the live account
    behaves (found 2026-10-02: F62_frac != compounding frac on the same starts by up to 2pp CAGR, by construction)."""
    return dict(PKG, raw_px=True, fixed_cap=float(cap), initial_capital=float(cap), rounding=rule, **kw)


def C(rule, **kw):
    """Realistic forward path: start at $62k at real prices and compound; rounding matters less as the account grows."""
    return dict(PKG, raw_px=True, initial_capital=62_000.0, rounding=rule, **kw)


RULES = ("book_floor", "book_round", "acct_round", "acct_floor", "frac", "acct_frac")
ARMS = {"VAL_pkg": dict(PKG)}                              # every new switch off: must equal EXP-059 cap0.15_package
for _rule in RULES:                                        # *_book_floor = THE LIVE RULE (baseline)
    ARMS[f"R62_{_rule}"] = R(62_000, _rule)
    ARMS[f"F62_{_rule}"] = F(62_000, _rule)
    ARMS[f"C62_{_rule}"] = C(_rule)
for _cap in (31_000, 125_000, 250_000):                    # account-size line
    for _rule in ("book_floor", "acct_round", "acct_floor", "frac"):
        ARMS[f"R{_cap // 1000}_{_rule}"] = R(_cap, _rule)
for _rule in ("book_floor", "acct_round", "acct_floor"):   # cost stress: IBKR fixed-plan $1 minimum per order; 2x costs
    ARMS[f"R62_{_rule}_minc1"] = R(62_000, _rule, min_comm=1.0)
    ARMS[f"R62_{_rule}_cost2"] = R(62_000, _rule, cost_mult=2.0)
# PLUMBING CHECKS (mode `sanity`): identities / limits the harness must satisfy before any arm is believed
SANITY = {
    "S_C62_frac_adj": dict(PKG, rounding="frac", initial_capital=62_000.0),        # == R62_frac exactly (frac is scale-free)
    "S_F62x16384_frac": F(62_000 * 2 ** 14, "frac"),                               # == F62_frac exactly (power-of-2 scale)
    "S_frac_cost0": dict(PKG, rounding="frac", cost_mult=0.0, initial_capital=62_000.0),
    "S_acct_frac_cost0": dict(PKG, rounding="acct_frac", cost_mult=0.0, initial_capital=62_000.0),  # == S_frac_cost0 exactly
    "S_R1e12_book_floor": R(1e12, "book_floor"),                                   # -> R62_frac as rounding vanishes
    "S_R1e12_acct_round": R(1e12, "acct_round"),                                   # -> R62_acct_frac as rounding vanishes
}
ARMS.update(SANITY)

# ---- patches on top of the EXP-059 frontier engine (its validated `reps`, parsed verbatim from its source) ----------
HELPERS = """        stop = float(cfg.get('stop', 0.40)); vol_stop_k = cfg.get('vol_stop'); stop_of = {}
        RMODE = cfg.get('rounding', 'book_floor'); ACCT = RMODE in ('acct_round', 'acct_floor', 'acct_frac')
        assert RMODE in ('book_floor', 'book_round', 'frac', 'acct_round', 'acct_floor', 'acct_frac'), RMODE
        assert not (cfg.get('overlay_daily') or cfg.get('exit_all') or cfg.get('overlay_all')), 'EXP-063 patches cover the package paths only'
        _RAW = getattr(self, '_raw', None) if cfg.get('raw_px') else None
        assert not cfg.get('raw_px') or _RAW is not None, 'raw_px arm without a real-price matrix'
        FIXCAP = float(cfg['fixed_cap']) if cfg.get('fixed_cap') else None
        assert not FIXCAP or abs(float(cfg.get('initial_capital', 50_000.0)) - FIXCAP) < 1e-6
        REFCAP = float(cfg['ref_cap']) if cfg.get('ref_cap') else None
        assert not (FIXCAP and REFCAP), 'fixed_cap and ref_cap are alternative capital models'
        MINC = float(cfg.get('min_comm', 0.0))
        resid = {}; _vprev = {}
        _st = dict(rebuilds=0, names=0, zero=0, trades=0, cost=0.0, cap_hold=0, raw_hits=0, raw_miss=0, gross_sum=0.0, days=0, maxw=0.0)
        self._r_stats = _st
        def _rawp(s_, p_):
            # the price whole shares are counted at: the real close (raw_px), and with ref_cap scaled by NAV / ref_cap so
            # the account rounds exactly as a ref_cap-sized account would while its P&L compounds normally
            v_ = p_
            if _RAW is not None:
                try:
                    r_ = _RAW.at[d, s_]
                except KeyError:
                    r_ = float('nan')
                if r_ == r_ and r_ > 0:
                    _st['raw_hits'] += 1
                    v_ = float(r_)
                else:
                    _st['raw_miss'] += 1
            if REFCAP:
                v_ = v_ * nav / REFCAP
            return v_
        def _vcost(v_):
            if ACCT:
                return 0.0
            c_ = v_ * cost_r
            if MINC and v_ > 0 and c_ < MINC:
                c_ = MINC
            _st['trades'] += 1; _st['cost'] += c_
            return c_
        def _trim(s_, q_old, f_, p_):
            if RMODE in ('frac', 'acct_round', 'acct_floor', 'acct_frac'):
                return q_old * f_
            _pr = _rawp(s_, p_)
            if _pr == p_:
                return int(q_old * f_)
            n_old = round(q_old * p_ / _pr)
            return int(n_old * f_ + 1e-9) * _pr / p_
"""

ENDDAY = """            if ACCT:
                _tot = {}
                for b_ in books:
                    for s_, q_ in b_.items():
                        _tot[s_] = _tot.get(s_, 0.0) + q_
                for s_ in set(_tot) | set(_vprev) | set(resid):
                    V_ = _tot.get(s_, 0.0); Vp_ = _vprev.get(s_, 0.0); r0_ = resid.get(s_, 0.0)
                    if V_ == Vp_:
                        continue
                    p_ = prc.get(s_, lastpx.get(s_))
                    if not p_ or p_ <= 0:
                        continue
                    _pr = _rawp(s_, p_)
                    H0_ = Vp_ + r0_
                    n0_ = round(H0_ * p_ / _pr)
                    fr_ = V_ * p_ / _pr
                    if RMODE == 'acct_frac':
                        H1_ = V_
                    else:
                        n1_ = int(fr_ + 0.5) if RMODE == 'acct_round' else int(fr_ + 1e-9)
                        if RMODE == 'acct_round' and n1_ > fr_ and n1_ * _pr > 0.15 * nav * 1.0000001:
                            n1_ -= 1; _st['cap_hold'] += 1
                        H1_ = H0_ if n1_ == n0_ else n1_ * _pr / p_
                    dH_ = H1_ - H0_
                    cash -= (dH_ - (V_ - Vp_)) * p_
                    if abs(dH_) > 1e-12:
                        c_ = abs(dH_ * p_) * cost_r
                        if MINC and c_ < MINC:
                            c_ = MINC
                        cash -= c_; _st['trades'] += 1; _st['cost'] += c_
                    r1_ = H1_ - V_
                    if abs(r1_) > 1e-12:
                        resid[s_] = r1_
                    else:
                        resid.pop(s_, None)
                    if V_ != 0.0:
                        _vprev[s_] = V_
                    else:
                        _vprev.pop(s_, None)
                nav = cash + mtm()
            _g = mtm(); _nv = cash + _g
            if _nv > 0:
                _st['gross_sum'] += _g / _nv; _st['days'] += 1
            if _rb_base is not None and _nv > 0:
                _hold = {}
                for b_ in books:
                    for s_, q_ in b_.items():
                        _hold[s_] = _hold.get(s_, 0.0) + q_
                for s_, r_ in resid.items():
                    _hold[s_] = _hold.get(s_, 0.0) + r_
                _st['rebuilds'] += 1; _st['names'] += len(_rb_base)
                _st['zero'] += sum(1 for s_ in _rb_base if _hold.get(s_, 0.0) <= 1e-9)
                _w = [q_ * prc.get(s_, lastpx.get(s_, 0.0)) / _nv for s_, q_ in _hold.items()]
                if _w:
                    _st['maxw'] = max(_st['maxw'], max(_w))
            if FIXCAP:
                nav = cash + mtm()
                _syn = _syn * (nav / _sod)
                cash -= nav - FIXCAP; nav = FIXCAP; _sod = FIXCAP
            prev_debit = max(0.0, -cash)
            navhist.append(max(_syn if FIXCAP else nav, 1e-9))
            navs.append((d, max(_syn if FIXCAP else nav, 1e-9)))
"""

EXTRA = [
    ("        stop = float(cfg.get('stop', 0.40)); vol_stop_k = cfg.get('vol_stop'); stop_of = {}\n", HELPERS),
    ('        cash = float(cfg.get("initial_capital", 50_000.0))\n',
     '        cash = float(cfg.get("initial_capital", 50_000.0))\n        _syn = cash; _sod = cash\n'),
    ("            for _s, _p in prc.items():\n                lastpx[_s] = _p\n",
     "            for _s, _p in prc.items():\n                lastpx[_s] = _p\n            _rb_base = None\n"),
    ("                            cash -= gross_ * cost_r          # DIFFERENCE 2: separate deduction\n",
     "                            cash -= _vcost(gross_)          # DIFFERENCE 2: separate deduction\n"),
    ("                        tot += q * prc.get(s, lastpx.get(s, 0.0))\n                return tot\n",
     "                        tot += q * prc.get(s, lastpx.get(s, 0.0))\n                for s, r_ in resid.items():\n"
     "                    tot += r_ * prc.get(s, lastpx.get(s, 0.0))\n                return tot\n"),
    ("                        q = int(tnav * min(w0 * mult, cap_pos) / p)   # integer shares\n",
     "                        _tv = tnav * min(w0 * mult, cap_pos)\n"
     "                        if RMODE in ('book_floor', 'book_round'):\n"
     "                            _pr = _rawp(s, p)\n"
     "                            if RMODE == 'book_floor':\n"
     "                                _n = int(_tv / _pr)\n"
     "                            else:\n"
     "                                _n = int(_tv / _pr + 0.5)\n"
     "                                if _n * _pr > tnav * cap_pos * 1.0000001:\n"
     "                                    _n -= 1\n"
     "                            q = _n if _pr == p else _n * _pr / p   # books are kept in total-return units\n"
     "                        else:\n"
     "                            q = _tv / p                             # fractional (virtual) book: scale-free\n"),
    ("                for s in list(books[t]):\n                    if s not in tgt:\n"
     "                        p = prc.get(s, lastpx.get(s, 0.0))\n                        q = books[t].pop(s); peaks[t].pop(s, None)\n"
     "                        cash += q * p\n                        cash -= abs(q * p) * cost_r\n",
     "                _rb_base = list(base)\n"
     "                for s in list(books[t]):\n                    if s not in tgt:\n"
     "                        p = prc.get(s, lastpx.get(s, 0.0))\n                        q = books[t].pop(s); peaks[t].pop(s, None)\n"
     "                        cash += q * p\n                        cash -= _vcost(abs(q * p))\n"),
    ("                    cash -= abs(dq * p) * cost_r\n", "                    cash -= _vcost(abs(dq * p))\n"),
    ("                                    q_old = books[t2][s2]; q_new = int(q_old * f2); dq2 = q_new - q_old\n"
     "                                    if dq2 == 0 or abs(dq2 * p2) < tnav * 0.003: continue\n"
     "                                    cash -= dq2 * p2; cash -= abs(dq2 * p2) * cost_r\n",
     "                                    q_old = books[t2][s2]; q_new = _trim(s2, q_old, f2, p2); dq2 = q_new - q_old\n"
     "                                    if dq2 == 0 or abs(dq2 * p2) < tnav * 0.003: continue\n"
     "                                    cash -= dq2 * p2; cash -= _vcost(abs(dq2 * p2))\n"),
    ("            prev_debit = max(0.0, -cash)\n            navhist.append(max(nav, 1e-9))\n"
     "            navs.append((d, max(nav, 1e-9)))\n", ENDDAY),
]


def _reps():
    tree = ast.parse(open("research/EXP059_frontier.py").read())
    eng = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_engine")
    node = next(n for n in ast.walk(eng) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "reps")
    return ast.literal_eval(node.value)


_ORIG_RUN = CleanRoom.run


def install(extra=True):
    src = inspect.getsource(_ORIG_RUN)
    for a, b in _reps() + (EXTRA if extra else []):
        assert src.count(a) == 1, a[:100]
        src = src.replace(a, b)
    V.END = E.END
    ns = dict(V.__dict__)
    assert ns["END"] == E.END
    exec(compile(textwrap.dedent(src), "<exp063>" if extra else "<exp059>", "exec"), ns)
    CleanRoom.run = ns["run"]


def build_raw(px):
    """Real (unadjusted) close per universe column and date. Cached."""
    f = f"{CACHE}/raw_close.parquet"
    if os.path.exists(f):
        R = pd.read_parquet(f)
        R.index = pd.to_datetime(R.index)
        return R.reindex(index=px.index, columns=px.columns)
    import pyarrow.dataset as ds
    t0 = time.time()
    cols = [c for c in px.columns if str(c).isdigit()]
    flt = ((ds.field("PERMNO").isin([int(c) for c in cols]))
           & (ds.field("DlyCalDt") >= px.index.min().strftime("%Y-%m-%d")) & (ds.field("DlyCalDt") <= "2025-12-31"))
    raw = ds.dataset("data/wrds/crsp_daily_stock_full.parquet").to_table(
        columns=["PERMNO", "DlyCalDt", "DlyClose", "DlyPrc"], filter=flt).to_pandas()
    raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"])
    raw["c"] = raw["DlyClose"].where(raw["DlyClose"] > 0, raw["DlyPrc"].abs())
    raw = raw.dropna(subset=["c"]).drop_duplicates(["PERMNO", "DlyCalDt"], keep="last")
    R1 = raw.pivot(index="DlyCalDt", columns="PERMNO", values="c")
    R1.columns = [str(c) for c in R1.columns]
    print(f"  CRSP raw closes {R1.shape} ({time.time() - t0:.0f}s)", flush=True)
    # 2026: Compustat Security Daily prccd, linked to PERMNO exactly as scripts/build_universe_v2.load_prices links it
    crsp_end = pd.Timestamp("2025-12-31")
    link = pd.read_parquet("data/wrds/compustat_crsp_link.parquet",
                           columns=["gvkey", "LIID", "LINKTYPE", "LINKPRIM", "LPERMNO", "LINKDT", "LINKENDDT"])
    link = link.dropna(subset=["LPERMNO"])
    link["LPERMNO"] = link["LPERMNO"].astype(int)
    link["gvkey"] = link["gvkey"].astype(str).str.zfill(6)
    link = link[link["LINKTYPE"].isin(["LC", "LU", "LS"])].copy()
    for c in ("LINKDT", "LINKENDDT"):
        link[c] = pd.to_datetime(link[c], errors="coerce")
    link["LINKENDDT"] = link["LINKENDDT"].fillna(pd.Timestamp("2099-12-31"))
    lk = (link[(link["LINKDT"] <= crsp_end) & (link["LINKENDDT"] >= crsp_end - pd.Timedelta(days=400))]
          .sort_values(["LPERMNO", "LINKPRIM"]).drop_duplicates("LPERMNO", keep="first"))
    key_of = {(r.gvkey, str(r.LIID).zfill(2) if isinstance(r.LIID, str) or not pd.isna(r.LIID) else "01"): str(r.LPERMNO)
              for r in lk.itertuples(index=False)}
    sd = pd.read_parquet("data/wrds/compustat_security_daily/secd_2026.parquet", columns=["gvkey", "iid", "datadate", "prccd"])
    sd["datadate"] = pd.to_datetime(sd["datadate"])
    sd = sd[(sd["datadate"] > crsp_end) & (sd["prccd"] > 0)].copy()
    sd["key"] = [key_of.get((str(g).zfill(6), str(i).zfill(2))) for g, i in zip(sd["gvkey"], sd["iid"])]
    sd = sd.dropna(subset=["key"]).drop_duplicates(["key", "datadate"], keep="last")
    R2 = sd.pivot(index="datadate", columns="key", values="prccd")
    print(f"  Compustat 2026 raw closes {R2.shape}", flush=True)
    R = pd.concat([R1, R2]).sort_index()
    R = R[~R.index.duplicated(keep="first")].reindex(index=px.index, columns=px.columns)
    R.to_parquet(f)
    print(f"  real-price matrix {R.shape} cached ({time.time() - t0:.0f}s)", flush=True)
    return R


def _bt():
    from main_production_backtest import FastBacktester
    return FastBacktester(universe_path=E.PATH)


def mode_raw():
    cr = CleanRoom(_bt())
    px = cr.px
    R = build_raw(px)
    have = px.notna()
    cov = (R.notna() & have).sum(axis=1) / have.sum(axis=1).replace(0, np.nan)
    print("\nreal-price coverage of priced cells, by year:")
    print(cov.groupby(cov.index.year).mean().round(3).to_string())
    ratio = (px / R).where(have & R.notna())
    for lab, day in (("last CRSP day 2025-12-31", "2025-12-31"), ("2026-08-31", "2026-08-31")):
        dd = ratio.index[ratio.index <= pd.Timestamp(day)][-1]
        r = ratio.loc[dd].dropna()
        print(f"  adjusted/real at {dd.date()} ({lab}): n={len(r)} median {r.median():.4f} within 0.5% {float(((r - 1).abs() < 0.005).mean()):.1%}"
              f"  <0.5 {int((r < 0.5).sum())}  >2 {int((r > 2).sum())}")
    for y in (2001, 2005, 2010, 2015, 2020, 2024):
        if y < ratio.index.min().year:
            continue
        dd = ratio.index[ratio.index.year == y][0]
        r = ratio.loc[dd].dropna()
        print(f"  adjusted/real {dd.date()}: median {r.median():.3f}  p10 {r.quantile(.1):.3f}  share < 0.5: {float((r < 0.5).mean()):.1%}")
    # median REAL price of names over time (what whole shares cost)
    med = R.where(have).median(axis=1)
    print("  median real share price by year:", {int(k): round(float(v), 1) for k, v in med.groupby(med.index.year).median().items()})
    # spot checks against known prints
    for perm, day, nm in (("14593", "2005-01-03", "AAPL (pre-2005/2014/2020 splits)"), ("14593", "2025-12-31", "AAPL"),
                          ("86580", "2015-01-02", "NVDA")):
        if perm in R.columns:
            dd = R.index[R.index >= pd.Timestamp(day)][0]
            print(f"  {nm} {dd.date()}: real {R.at[dd, perm]:.2f}  adjusted {px.at[dd, perm]:.2f}")
    for t in ("LITE", "SNDK", "MU"):
        f = f"data/massive_cache/{t}_adj.parquet"
        if not os.path.exists(f):
            continue
        m = pd.read_parquet(f)
        m.index = pd.to_datetime(m.index)
        ccols = [x for x in m.columns if str(x).lower() in ("close", "c", "adj_close")]
        if not ccols:
            print(f"  {t}: vendor cache has no close column {list(m.columns)[:8]}")
            continue
        c = m[ccols[0]]
        if getattr(c.index, "tz", None) is not None:
            c.index = c.index.tz_localize(None)
        c.index = c.index.normalize()
        # find the universe column whose 2026 real closes match this ticker's vendor closes best
        j = c.index.intersection(R.index[R.index >= pd.Timestamp("2026-01-02")])
        if len(j) < 20:
            continue
        sub = R.loc[j]
        err = (sub.div(c.loc[j], axis=0) - 1).abs().median()
        best = err.idxmin()
        print(f"  {t}: universe column {best}, median |real/vendor - 1| over {len(j)} 2026 days = {err.min():.4%}")


def mode_validate():
    """Parity guard: EXP-059 engine vs EXP-063 engine with every new switch off, same start, same process."""
    bt = _bt()
    s_ = E.STARTS[0]
    install(extra=False)
    a = CleanRoom(bt).run(s_, ARMS["VAL_pkg"])["curve"]
    install(extra=True)
    b = CleanRoom(bt).run(s_, ARMS["VAL_pkg"])["curve"]
    diff = float((a - b).abs().max())
    print(f"  EXP-059 engine vs EXP-063 engine (switches off), start {s_}: {len(a)} days, max |NAV diff| = {diff:.6g}")
    f = f"research/_v2_{HZ}/exp059/stage2_cap0.15_package.parquet"
    if os.path.exists(f):
        c = pd.read_parquet(f)[s_].dropna()
        j = c.index.intersection(b.index)
        print(f"  vs cached EXP-059 stage-2 cap0.15_package curve: {len(j)} common days, max |rel diff| = {float((b.loc[j] / c.loc[j] - 1).abs().max()):.3g}")
    assert diff == 0.0, "PARITY BROKEN — do not use EXP-063 results"
    print("  PARITY OK")


def mode_sanity():
    """Plumbing identities. Any failure means no EXP-063 number may be quoted."""
    install(extra=True)
    cr = CleanRoom(_bt())
    cr._raw = build_raw(cr.px)
    starts = [E.STARTS[0], E.STARTS[7]]
    cache = {}

    def curve(nm, s):
        if (nm, s) not in cache:
            cache[(nm, s)] = cr.run(s, ARMS[nm])["curve"]
        return cache[(nm, s)]

    ok = True
    for a, b, kind in (("R62_frac", "S_C62_frac_adj", "exact"),          # real prices + ref size are inert for frac
                       ("F62_frac", "S_F62x16384_frac", "exact"),        # the fixed-capital sweep is scale-consistent
                       ("S_frac_cost0", "S_acct_frac_cost0", "exact"),   # the account layer is lossless without rounding
                       ("R62_frac", "S_R1e12_book_floor", "limit"),      # book rounding vanishes at a huge size
                       ("R62_acct_frac", "S_R1e12_acct_round", "limit")):  # account rounding vanishes at a huge size
        for s in starts:
            va, vb = curve(a, s), curve(b, s)
            ra, rb = va.pct_change().dropna(), vb.pct_change().dropna()
            j = ra.index.intersection(rb.index)
            mx = float((ra[j] - rb[j]).abs().max()) if len(j) else float("nan")
            ca, cb = E.st(va)[0], E.st(vb)[0]
            good = (mx < 1e-12) if kind == "exact" else (abs(ca - cb) < 0.001)
            ok = ok and good
            print(f"  {'OK ' if good else 'BAD'} {kind:<5} {a} vs {b} {s}: max |daily return diff| {mx:.3g}"
                  f"  CAGR {ca:+.3%} vs {cb:+.3%}", flush=True)
    print("  SANITY " + ("PASSED" if ok else "FAILED — do not use EXP-063 results"))
    assert ok


def mode_run(names):
    for nm in names:
        assert nm in ARMS, nm
    shard = os.environ.get("EXP063_SHARD")
    starts = STARTS
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        starts = [s for i, s in enumerate(STARTS) if i % n == k]
    install(extra=True)
    cr = CleanRoom(_bt())
    if any(ARMS[nm].get("raw_px") for nm in names):
        cr._raw = build_raw(cr.px)
    t0 = time.time()
    for nm in names:
        d_ = f"{CACHE}/{STAGE}_{nm}"
        os.makedirs(d_, exist_ok=True)
        for s_ in starts:
            f_ = f"{d_}/{s_}.parquet"
            if os.path.exists(f_):
                continue
            t1 = time.time()
            r = cr.run(s_, ARMS[nm])
            pd.DataFrame({s_: r["curve"]}).to_parquet(f_ + ".tmp")
            os.replace(f_ + ".tmp", f_)
            with open(f"{d_}/{s_}.json", "w") as fh:
                json.dump(cr._r_stats, fh)
            cagr, sh, dd = E.st(r["curve"])
            print(f"  {nm:<22} {s_}  {cagr:+.2%} / {sh:.3f} / {dd:.1%}  ({time.time() - t1:.0f}s, total {time.time() - t0:.0f}s)", flush=True)


def load(nm):
    d_ = f"{CACHE}/{STAGE}_{nm}"
    cur = {}
    for s_ in STARTS:
        f_ = f"{d_}/{s_}.parquet"
        if os.path.exists(f_):
            cur[s_] = pd.read_parquet(f_)[s_]
    return pd.DataFrame(cur)


def mech(nm, cols):
    agg = {}
    for s_ in cols:
        f_ = f"{CACHE}/{STAGE}_{nm}/{s_}.json"
        if os.path.exists(f_):
            for k, v in json.load(open(f_)).items():
                agg.setdefault(k, []).append(v)
    if not agg:
        return ""
    curves = {s_: pd.read_parquet(f"{CACHE}/{STAGE}_{nm}/{s_}.parquet")[s_].dropna() for s_ in cols}
    yrs = np.mean([(c.index[-1] - pd.Timestamp(s_)).days / 365.25 for s_, c in curves.items()])
    cap = ARMS[nm].get("fixed_cap")
    z = sum(agg["zero"]) / max(sum(agg["names"]), 1)
    out = (f"zeroed {z:.1%} of target names | trades/yr {np.mean(agg['trades']) / yrs:.0f} | gross {sum(agg['gross_sum']) / max(sum(agg['days']), 1):.2f}x"
           f" | max name {np.max(agg['maxw']):.1%}")
    # costs per year as a share of the capital actually at work (fixed cap, or the run's average NAV when compounding)
    denom = [cap if cap else float(curves[s_].mean()) for s_ in cols]
    out += f" | costs {np.mean([c_ / yrs / dn for c_, dn in zip(agg['cost'], denom)]):.2%}/yr of NAV"
    if sum(agg["raw_hits"]) + sum(agg["raw_miss"]):
        out += f" | real-price hits {sum(agg['raw_hits']) / (sum(agg['raw_hits']) + sum(agg['raw_miss'])):.1%}"
    if sum(agg["cap_hold"]):
        out += f" | cap-held round-ups {int(np.mean(agg['cap_hold']))}/run"
    return out


def mode_analyse(base):
    rng = np.random.default_rng(1)
    B = load(base)
    print(f"\n{HZ} {STAGE} — baseline {base} ({B.shape[1]} starts)")
    if B.shape[1]:
        b = np.array([E.st(B[c]) for c in B.columns])
        print(f"  {base:<22} {b[:, 0].mean():+.2%} / {b[:, 1].mean():.3f} / {b[:, 2].mean():.1%}   {mech(base, list(B.columns))}")
    for nm in ARMS:
        if nm in (base, "VAL_pkg"):
            continue
        A = load(nm)
        cols = [c for c in A.columns if c in B.columns]
        if not cols:
            continue
        a = np.array([E.st(A[c]) for c in cols])
        bb = np.array([E.st(B[c]) for c in cols])
        d = a - bb
        yb = [E.yearly(B[c]) for c in cols]
        ya = [E.yearly(A[c]) for c in cols]
        ys = sorted(set().union(*[set(x) for x in yb]))
        dy = {y: np.mean([xa[y] - xb[y] for xa, xb in zip(ya, yb) if y in xa and y in xb]) for y in ys}
        wins = sum(1 for y in ys if dy[y] > 0)
        worst = min(dy, key=dy.get) if dy else None
        print(f"  {nm:<22} {a[:, 0].mean():+.2%} / {a[:, 1].mean():.3f} / {a[:, 2].mean():.1%}"
              f"  dCAGR {d[:, 0].mean() * 100:+.2f}pp ({int((d[:, 0] > 0).sum())}/{len(cols)})"
              f"  dSharpe {d[:, 1].mean():+.3f} ({int((d[:, 1] > 0).sum())}/{len(cols)})"
              f"  dMaxDD {d[:, 2].mean() * 100:+.2f}pp ({int((d[:, 2] > 0).sum())}/{len(cols)})"
              f"  years better {wins}/{len(ys)}" + (f"  worst year {worst} {dy[worst] * 100:+.2f}pp" if worst else ""))
        m = mech(nm, cols)
        if m:
            print(f"  {'':<22} {m}")
        # GATE059 check (0): is the gain just more exposure? Vol-matched CAGR and drawdown per unit of vol.
        vb_ = np.mean([B[c].dropna().pct_change().std() * np.sqrt(252) for c in cols])
        va_ = np.mean([A[c].dropna().pct_change().std() * np.sqrt(252) for c in cols])
        print(f"  {'':<22} risk: vol {vb_:.2%} -> {va_:.2%} | CAGR vol-matched to baseline {a[:, 0].mean() * vb_ / va_:+.2%}"
              f" vs {bb[:, 0].mean():+.2%} ({(a[:, 0].mean() * vb_ / va_ - bb[:, 0].mean()) * 100:+.2f}pp)"
              f" | MaxDD/vol {bb[:, 2].mean() / vb_:.2f} -> {a[:, 2].mean() / va_:.2f}")
        if STAGE == "stage2" and len(cols) >= 12:
            # GATE059-style checks: bootstrap dSharpe, sub-periods, untouched starts, best-5 days removed
            shd = []
            for _ in range(1000):
                c = cols[rng.integers(0, len(cols))]
                rb_ = B[c].dropna().pct_change().dropna()
                ra_ = A[c].dropna().pct_change().dropna()
                j = rb_.index.intersection(ra_.index)
                x_, y_ = rb_[j].values, ra_[j].values
                Bk = 60
                idx = rng.integers(0, len(x_) - Bk, size=len(x_) // Bk + 1)
                sel = np.concatenate([np.arange(i, i + Bk) for i in idx])[:len(x_)]
                shd.append(y_[sel].mean() / y_[sel].std() * np.sqrt(252) - x_[sel].mean() / x_[sel].std() * np.sqrt(252))
            shd = np.array(shd)
            subs = []
            for lab, y0, y1 in E.SUBS:
                ds_ = [E.st(A[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"])[1] - E.st(B[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"])[1]
                       for c in cols if len(A[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"]) > 60]
                subs.append(f"{lab} {np.mean(ds_):+.3f}" if ds_ else f"{lab} n/a")
            ev = [c for c in cols if pd.Timestamp(c).month % 2 == 0]
            de = np.mean([E.st(A[c])[1] - E.st(B[c])[1] for c in ev]) if ev else float("nan")

            def drop_best(v, k=5):
                r = v.dropna().pct_change().dropna()
                return (1 + r.drop(r.nlargest(k).index)).cumprod()

            d5 = np.mean([E.st(drop_best(A[c]))[1] - E.st(drop_best(B[c]))[1] for c in cols])
            print(f"  {'':<22} gate: bootstrap dSharpe {shd.mean():+.3f} 90% CI [{np.percentile(shd, 5):+.3f}, {np.percentile(shd, 95):+.3f}]"
                  f" P(>0) {np.mean(shd > 0):.0%} | sub-periods {' '.join(subs)} | even-month starts {de:+.3f} | best-5 removed {d5:+.3f}")
            print(f"  {'':<22} years: " + " ".join(f"{y}:{v * 100:+.2f}" for y, v in dy.items()))


if __name__ == "__main__":
    if STAGE == "validate":
        mode_validate()
    elif STAGE == "raw":
        mode_raw()
    elif STAGE == "sanity":
        mode_sanity()
    elif MODE == "run":
        mode_run(ONLY)
    elif MODE == "analyse":
        mode_analyse(ONLY[0] if ONLY else "R62_book_floor")
    else:
        print(__doc__)
