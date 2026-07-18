"""
THREAD A — micro-futures feasibility gate at $50K (A1) + crisis convexity (A3) +
equity integration (A2 overlay / A4 realloc).

Prior: FULL 98-market book -> −0.34 net Sharpe at $30k (0 markets survive rounding; bonds,
the key hedge, have NO micro and ~$100k notional). A1 asks the narrower question: a book of
ONLY micro-capable markets (MES/MNQ/M2K/MYM/M6E + MGC/MCL/MSI/MHG + 6J/6B/6A micros),
capacity-realized with INTEGER micro contracts at $50k — is net Sharpe > 0?  A3: with NO
bonds (no micro exists), does 2008/2020/2022 crisis convexity survive?

Reuses futures_capacity.py's construction (trend+xsmom+carry, vol-target sleeves, 12% book
vol overlay, integer-contract rounding, tiered costs). Modern era (>=2010) for realism.
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd  # noqa: E402
import futures_lib as fl  # noqa: E402

# (full_mult, micro_mult_or_None, cost_bps_per_side) — from futures_capacity.SPEC
SPEC = {
 "ES":(50,5,1.5),"NQ":(20,2,1.5),"YM":(5,0.5,2),"RTY":(50,5,2),
 "6E":(125000,12500,1.5),"6J":(12500000,1250000,2),"6B":(62500,6250,2),"6A":(100000,10000,2),
 "CL":(1000,100,2),"GC":(100,10,2),"SI":(5000,1000,3),"HG":(25000,2500,3),
}
MICRO_MKTS = list(SPEC.keys())  # only markets with a real micro contract

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
vol = R.ewm(span=60, min_periods=20).std()


def conv(L):
    mom = CCB.diff(L) / NON.shift(L).abs()
    return np.tanh(mom / (vol * np.sqrt(L)))


tr = sum(conv(L) for L in [21, 63, 252]) / 3
ramom = (CCB.diff(252) / NON.shift(252).abs()) / vol
xs = np.tanh(ramom.sub(ramom.median(axis=1), axis=0) / ramom.std())

keep = [m for m in MICRO_MKTS if m in R.columns]
print("micro-capable markets available:", keep, flush=True)
R = R[keep]; NON = NON[keep]; CCB = CCB[keep]
div = [m for m in keep if m not in fl.EQUITY]     # non-equity diversifiers for trend
carry = pd.read_parquet("_sleeve_carry_sig.parquet").reindex(index=CCB.index, columns=keep)
pos_tr = fl.vol_target(tr[div], R[div], lev_cap=7).reindex(columns=keep)
pos_xs = fl.vol_target(xs[keep], R, lev_cap=7)
pos_ca = fl.vol_target(carry, R, lev_cap=7)
eff = pd.DataFrame(np.nanmean(np.stack([pos_tr.values, pos_xs.values, pos_ca.values]), axis=0),
                   index=pos_xs.index, columns=keep)
act = eff.notna() & R.notna()
book_raw = (eff * R).where(act).mean(axis=1)
sc = (0.12 / (book_raw.rolling(252, min_periods=60).std().shift(1) * np.sqrt(252))).clip(upper=3).fillna(1)
Nact = act.sum(axis=1).clip(lower=1)
omega = eff.mul(sc, axis=0).div(Nact, axis=0)

micro = pd.Series({m: (SPEC[m][1] or SPEC[m][0]) for m in keep})
cost = pd.Series({m: SPEC[m][2] for m in keep}) / 1e4
notm = NON.mul(micro, axis=1)
wk = omega.resample("W-FRI").last()
notm_wk = notm.resample("W-FRI").last()


def realized(A, rounding=True, cost_mult=1.0):
    con = (wk * A) / notm_wk
    if rounding:
        con = np.round(con).fillna(0)
    rw = (con * notm_wk / A).reindex(R.index).ffill()
    turn = (con.diff().abs() * notm_wk / A)
    cst = (turn * cost).sum(axis=1).reindex(R.index).fillna(0) * cost_mult
    ret = (rw.shift(1) * R).sum(axis=1) - cst
    held = (con.abs() >= 1).sum(axis=1)
    return ret.dropna(), held, rw.abs().sum(axis=1)


def sh(r, y0=2010):
    r = r[r.index.year >= y0].dropna()
    return r.mean() / r.std() * np.sqrt(252) if r.std() else 0


def cagr_dd(r, y0=2010):
    r = r[r.index.year >= y0].dropna()
    c = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return c.iloc[-1] ** (1 / yrs) - 1, ((c - c.cummax()) / c.cummax()).min()


print("\n=== A1: micro-only book realized Sharpe vs AUM (2010-26, integer micro contracts) ===", flush=True)
ig, _, g0 = realized(1e9, rounding=False, cost_mult=0)
print(f"  IDEAL gross (no cost/round)  Sharpe {sh(ig):5.2f}  gross {g0[g0.index.year>=2010].mean():.1f}x", flush=True)
for A in [30e3, 50e3, 100e3, 300e3, 1e6, 3e6]:
    ret, held, gross = realized(A)
    h = held[held.index.year >= 2010].mean()
    c, d = cagr_dd(ret)
    print(f"  AUM ${A/1e3:>7.0f}k   Sharpe {sh(ret):5.2f}   held ~{h:4.1f} mkts   gross {gross[gross.index.year>=2010].mean():.2f}x"
          f"   CAGR {c:+.1%}  MaxDD {d:+.1%}", flush=True)

print("\n=== A3: crisis convexity of the micro book at $50k (NO bonds available) ===", flush=True)
ret50, _, _ = realized(50e3)
for cname, a, z in [("2008 GFC", "2008-06-01", "2009-03-31"), ("2020 COVID", "2020-02-15", "2020-04-15"),
                    ("2022 bear", "2022-01-01", "2022-10-31"), ("2011 EU", "2011-07-01", "2011-10-31"),
                    ("2018 Q4", "2018-09-01", "2018-12-31")]:
    m = (ret50.index >= a) & (ret50.index <= z)
    if m.sum() > 3:
        wr = (1 + ret50[m]).prod() - 1
        print(f"  {cname:<11} micro-book window return {wr:+.1%}  ({m.sum()}d)", flush=True)

# save the $50k micro-book return series for integration (A2/A4)
ret50.to_frame("micro50k").to_parquet("_threadA_micro50k.parquet")
print("\nsaved _threadA_micro50k.parquet for equity-integration step.", flush=True)
print("\nGATE: A1 pass only if net Sharpe > 0 at $50k AND A3 crisis convexity survives losing bonds.", flush=True)
