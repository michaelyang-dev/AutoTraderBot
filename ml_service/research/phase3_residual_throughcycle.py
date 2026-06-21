"""
Phase 3 (through-cycle) — residual vs total-return momentum INCLUDING the GFC.
============================================================================
The prior residual-momentum test ran 2018-2025 (NO GFC) and found it worse. But
the document's central claim is that residual momentum kills the 2008-09 beta-
driven crash. This re-runs the SAME long-only top-5 sleeve comparison on the
25-year universe (sp1500_universe_2000.pkl), with explicit GFC/COVID/2022 crash
windows. Long-only (not long-short), because that is what is actually traded —
the academic crash mechanism is a long-short effect and may not transfer.

5 FF factors (mktrf smb hml rmw cma), loadings over 504d, momentum over recent
231d sub-window, standardized residual (BHM).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester

EST, FORM, SKIP, TOPN = 504, 252, 21, 5
MEAS = FORM - SKIP

t0 = time.time()
bt = FastBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl")
print(f"[loaded {time.time()-t0:.0f}s]")
prices = bt.prices
rets = prices.pct_change()
bt.uni.get_sp500 = bt._get_sp1500

ff = pd.read_parquet("data/wrds/fama_french_5factors_momentum_daily.parquet")
ff["date"] = pd.to_datetime(ff["date"]); ff = ff.set_index("date").sort_index()
FCOLS = [c for c in ["mktrf", "smb", "hml", "rmw", "cma"] if c in ff.columns]
F = ff[FCOLS].reindex(prices.index).fillna(0.0)
RF = ff["rf"].reindex(prices.index).fillna(0.0)
print(f"[FF factors: {FCOLS}]")

START = "2006-01-01"
dates = [d for d in prices.index if pd.Timestamp(START) <= d <= pd.Timestamp("2025-12-31")]
allidx = list(prices.index)
print(f"[dates {dates[0].date()}..{dates[-1].date()}, n={len(dates)}]")


def port_fwd(picks, d, fwd):
    r, n = 0.0, 0
    p0, p1 = prices.loc[d], prices.loc[fwd]
    for s in picks:
        a, b = p0.get(s), p1.get(s)
        if a and b and a > 0 and not np.isnan(a) and not np.isnan(b):
            r += (b / a - 1); n += 1
    return r / n if n else np.nan


tot, res, leg = [], [], []
for i in range(0, len(dates) - 21, 20):
    d = dates[i]; loc = allidx.index(d)
    if loc < EST + 5:
        continue
    members = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
    if len(members) < 50:
        continue
    est_win = allidx[loc - EST: loc - SKIP]
    if len(est_win) < 300:
        continue
    Y = rets.loc[est_win, members].values
    Xf = F.loc[est_win].values
    rfw = RF.loc[est_win].values
    valid = np.isfinite(Y).sum(axis=0) > 0.9 * len(est_win)
    cols = [members[j] for j in range(len(members)) if valid[j]]
    if len(cols) < 50:
        continue
    Yv = np.nan_to_num(Y[:, valid])
    exc = Yv - rfw[:, None]
    X = np.column_stack([np.ones(len(est_win)), Xf])
    beta, *_ = np.linalg.lstsq(X, exc, rcond=None)
    resid = exc - X @ beta
    rm = resid[-MEAS:]
    res_mom = rm.sum(axis=0) / (rm.std(axis=0) + 1e-9)
    tot_mom = Yv[-MEAS:].sum(axis=0)
    fwd = dates[i + 20]
    tot.append(port_fwd([cols[j] for j in np.argsort(-tot_mom)[:TOPN]], d, fwd))
    res.append(port_fwd([cols[j] for j in np.argsort(-res_mom)[:TOPN]], d, fwd))
    leg.append(fwd)

idx = pd.Index(leg)
s_tot = pd.Series(tot, index=idx).dropna()
s_res = pd.Series(res, index=idx).dropna()


def metrics(s):
    eq = (1 + s).cumprod()
    yrs = len(s) * 20 / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sh = (s.mean() / s.std()) * np.sqrt(252 / 20)
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    return cagr, sh, mdd, eq


print("=" * 70)
print(f"RESIDUAL vs TOTAL-RETURN MOMENTUM, LONG-ONLY top-5, {idx.min().year}-{idx.max().year}")
print("=" * 70)
for label, s in [("Total-return momentum (current)", s_tot),
                 ("Residual momentum (BHM standardized, 5-factor)", s_res)]:
    cagr, sh, mdd, eq = metrics(s)
    print(f"\n  {label}")
    print(f"     CAGR {cagr*100:5.1f}%  Sharpe {sh:.2f}  MaxDD {mdd*100:.1f}%")
    for wl, y0, y1 in [("GFC 2008", "2008-01-01", "2008-12-31"),
                       ("GFC rebound 2009", "2009-01-01", "2009-12-31"),
                       ("GFC full 07-09", "2007-07-01", "2009-06-30"),
                       ("COVID 2020", "2020-02-01", "2020-04-30"),
                       ("2022 reversal", "2022-01-01", "2022-12-31")]:
        w = eq[(eq.index >= pd.Timestamp(y0)) & (eq.index <= pd.Timestamp(y1))]
        if len(w) > 1:
            wdd = ((w - w.cummax()) / w.cummax()).min()
            print(f"       {wl:18s}: ret {(w.iloc[-1]/w.iloc[0]-1)*100:+7.1f}%  intra-DD {wdd*100:6.1f}%")
print(f"\n[total {time.time()-t0:.0f}s]")
