"""
Diversified time-series-momentum (trend / managed-futures) sleeve on the Norgate
gold-standard continuous-futures panel. The question that decides whether this is
worth building: does a broad TSMOM book generate CONVEX crisis-alpha — i.e. make
money in 2008 / 2020 / 2022 while equities crash — which the 3-instrument
SPY/TLT/GLD version failed to do?

Method (classic Hurst-Ooi-Pedersen "Century of Evidence" construction):
  - return per market: ccb.diff()/nonadj.shift(1)   [validated: _CCB is difference
    back-adjusted, so ccb.diff() = true gap-free point change; nonadj = real price level]
  - signal: average sign of trend over {21,63,252}d lookbacks  (value in [-1,1])
  - per-market vol-target: position scaled to constant risk (EWMA-60d vol)
  - portfolio: equal-risk-weight (mean across active markets), 1-day execution lag
  - costs charged on turnover; vol-scaled overlay for CAGR/DD (Sharpe is scale-free)
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

PANEL = "data/norgate/norgate_continuous_futures.parquet"
LOOKBACKS = [21, 63, 252]        # 1m / 3m / 12m trend
VOL_SPAN = 60                    # EWMA span for ex-ante per-market vol
TGT_MKT_VOL = 0.15               # annualized vol target per market leg
TGT_PORT_VOL = 0.12              # annualized portfolio vol target (overlay)
COST_BPS = 1.5                   # per-unit-turnover cost (bps of notional) — futures are cheap
EXCLUDE = {  # roll-variants + micros (dupe full-size, shorter history) + thin
    "FDAX9","FESX9","FOAT9","LEU9","LFT9","YAP4","YAP10",
    "MES","MNQ","MYM","M2K","MBT","MET","MHI",
}

df = pd.read_parquet(PANEL)
df["date"] = pd.to_datetime(df["date"])

# ---- build wide CCB (back-adj) and NONADJ close matrices -------------------
ccb = df[df["symbol"].str.endswith("_CCB")].copy()
ccb["mkt"] = ccb["symbol"].str.replace("_CCB$", "", regex=True).str.lstrip("&")
non = df[~df["symbol"].str.endswith("_CCB")].copy()
non["mkt"] = non["symbol"].str.lstrip("&")

CCB = ccb.pivot_table(index="date", columns="mkt", values="Close")
NON = non.pivot_table(index="date", columns="mkt", values="Close")
mkts = [m for m in CCB.columns if m in NON.columns and m not in EXCLUDE]
CCB, NON = CCB[mkts].sort_index(), NON[mkts].sort_index()
print(f"[{len(mkts)} markets, {CCB.index.min().date()}..{CCB.index.max().date()}]")

# ---- per-market returns: ccb.diff()/nonadj.shift(1), winsorized -------------
R = (CCB.diff() / NON.shift(1)).clip(-0.5, 0.5)
R = R.where(NON.shift(1) > 0)            # guard denom

# ---- ex-ante vol (EWMA), lagged --------------------------------------------
vol = R.ewm(span=VOL_SPAN, min_periods=20).std().shift(1)
vol_ann = vol * np.sqrt(252)

# ---- trend signal: mean sign over lookbacks (uses gap-free CCB point change) -
sig = sum(np.sign(CCB.diff(L)) for L in LOOKBACKS) / len(LOOKBACKS)

# ---- vol-targeted position, 1-day execution lag ----------------------------
lev = (TGT_MKT_VOL / vol_ann).clip(upper=20)        # cap per-market leverage
pos = (sig * lev).shift(1)                          # trade on next bar
pos = pos.where(R.notna())                          # only hold where tradable

# ---- gross pnl + costs ------------------------------------------------------
gross = pos * R
turnover = pos.diff().abs()
cost = turnover * (COST_BPS / 1e4)
pnl_m = gross - cost                                 # per-market net

# portfolio = mean across active markets each day (equal risk weight)
active = pos.notna() & R.notna()
port = pnl_m.where(active).mean(axis=1)
n_active = active.sum(axis=1)
port = port[n_active >= 5]                           # need >=5 markets for a real book

# ---- vol-scaled overlay (rolling, no lookahead) for CAGR/DD -----------------
roll_vol = port.rolling(252, min_periods=60).std().shift(1) * np.sqrt(252)
scalar = (TGT_PORT_VOL / roll_vol).clip(upper=3).fillna(1.0)
port_s = (port * scalar).dropna()

def stats(r, label):
    r = r.dropna()
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol_ = r.std() * np.sqrt(252)
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    print(f"  {label:<22} CAGR {cagr:+6.1%}  vol {vol_:5.1%}  Sharpe {sh:5.2f}  MaxDD {dd:6.1%}  ({r.index.min().date()}..{r.index.max().date()})")
    return r

print("\n=== Diversified TSMOM trend sleeve ===")
stats(port, "raw (equal-risk)")
ps = stats(port_s, "vol-scaled (12% tgt)")
print("\n  by decade (vol-scaled):")
for dec in [1980,1990,2000,2010,2020]:
    seg = ps[(ps.index.year>=dec)&(ps.index.year<dec+10)]
    if len(seg) > 60: stats(seg, f"  {dec}s")

# ---- equity benchmark + correlation ----------------------------------------
eq_ret = R["ES"].dropna() if "ES" in R.columns else None
if eq_ret is not None:
    j = ps.index.intersection(eq_ret.index)
    corr = np.corrcoef(ps[j], eq_ret[j])[0,1]
    print(f"\n  corr(trend sleeve, ES equity returns): {corr:+.3f}")

# ---- THE crisis test: sleeve vs equity in each crash -----------------------
print("\n=== CRISIS-ALPHA TEST (sleeve return vs equity crash) ===")
CRISES = {
    "1987 crash":      ("1987-09-01","1987-12-31"),
    "2000-02 dotcom":  ("2000-03-01","2002-10-31"),
    "2008 GFC":        ("2007-10-01","2009-03-31"),
    "2008 Sep-Nov":    ("2008-09-01","2008-11-30"),
    "2020 COVID":      ("2020-02-19","2020-03-23"),
    "2022 bear":       ("2022-01-01","2022-12-31"),
}
def cum(r, a, b):
    s = r[(r.index>=a)&(r.index<=b)]
    return (1+s).prod()-1 if len(s) else np.nan
for name,(a,b) in CRISES.items():
    sl = cum(ps, a, b)
    eqr = cum(eq_ret, a, b) if eq_ret is not None else np.nan
    eqs = f"{eqr:+6.1%}" if pd.notna(eqr) else "  n/a "
    print(f"  {name:<16} equity {eqs}   trend sleeve {sl:+6.1%}")

# ---- left-tail (crisis) convexity: sleeve return on worst equity months ----
if eq_ret is not None:
    m_sl = (1+ps).resample("ME").prod()-1
    m_eq = (1+eq_ret).resample("ME").prod()-1
    j = m_sl.index.intersection(m_eq.index)
    m_sl, m_eq = m_sl[j], m_eq[j]
    worst = m_eq <= m_eq.quantile(0.10)         # worst-decile equity months
    print(f"\n  worst-decile equity months (n={worst.sum()}): "
          f"equity avg {m_eq[worst].mean():+.1%}  ->  trend sleeve avg {m_sl[worst].mean():+.1%}")
    print(f"  all months: equity avg {m_eq.mean():+.1%}  trend sleeve avg {m_sl.mean():+.1%}")
