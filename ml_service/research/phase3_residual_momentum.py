"""
Phase 3 — Residual (idiosyncratic) momentum, Blitz-Huij-Martens (2011).
Strip factor exposure from each stock's returns, compute momentum on the
RESIDUAL (stock-specific) return. Documented: similar return, LOWER vol and
crash risk -> a tail tool, not just an alpha hunt.

CORRECT BHM STRUCTURE (the subtlety): factor loadings are fit over a LONG
window (504d), but residual momentum is the cumulative residual over a RECENT
sub-window (t-252..t-21). If you instead fit and cumulate over the SAME window,
the intercept forces sum(residuals)=0 for every stock -> the rank is pure noise.

Compares the momentum SLEEVE only (top-5), residual vs total-return momentum,
on CAGR / Sharpe / MaxDD AND crash-window behavior (the tail claim).
Efficient: one batched OLS per rebalance date (all stocks at once).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

EST = 504    # estimation window for factor loadings (~2y)
FORM = 252   # momentum measurement window
SKIP = 21    # skip last ~1 month
TOPN = 5
MEAS = FORM - SKIP   # recent sub-window length (231), tail of the est window

bt = FastBacktester()
prices = bt.prices
rets = prices.pct_change()
bt.uni.get_sp500 = bt._get_sp1500

ff = pd.read_parquet("data/wrds/fama_french_5factors_momentum_daily.parquet")
ff["date"] = pd.to_datetime(ff["date"]); ff = ff.set_index("date").sort_index()
F = ff[["mktrf", "smb", "hml"]].reindex(prices.index).fillna(0.0)
RF = ff["rf"].reindex(prices.index).fillna(0.0)

dates = [d for d in prices.index if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")]
allidx = list(prices.index)


def port_fwd(picks, d, fwd):
    r = 0.0; n = 0
    p0 = prices.loc[d]; p1 = prices.loc[fwd]
    for s in picks:
        a, b = p0.get(s), p1.get(s)
        if a and b and a > 0 and not np.isnan(a) and not np.isnan(b):
            r += (b / a - 1); n += 1
    return r / n if n else np.nan


tot_rets, resS_rets, resU_rets, leg_dates = [], [], [], []
for i in range(0, len(dates) - 21, 20):
    d = dates[i]
    loc = allidx.index(d)
    if loc < EST + 5:
        continue
    members = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
    est_win = allidx[loc - EST: loc - SKIP]        # long window to FIT loadings (ends t-21)
    if len(est_win) < 300:
        continue
    Y = rets.loc[est_win, members].values          # (T x N)
    Xf = F.loc[est_win].values
    rfw = RF.loc[est_win].values
    valid = np.isfinite(Y).sum(axis=0) > 0.9 * len(est_win)
    cols = [members[j] for j in range(len(members)) if valid[j]]
    Yv = np.nan_to_num(Y[:, valid])
    exc = Yv - rfw[:, None]
    X = np.column_stack([np.ones(len(est_win)), Xf])
    beta, *_ = np.linalg.lstsq(X, exc, rcond=None)
    resid = exc - X @ beta                          # residuals over the FULL est window
    # residual momentum = cumulate residuals over the RECENT sub-window (tail MEAS rows)
    rm = resid[-MEAS:]                               # (MEAS x Nv), within est window -> not forced to 0
    res_cum = rm.sum(axis=0)
    res_std = rm.std(axis=0) + 1e-9
    res_mom_std = res_cum / res_std                  # BHM standardized
    res_mom_unstd = res_cum                          # cross-check: raw cumulative residual
    tot_mom = Yv[-MEAS:].sum(axis=0)                 # total-return momentum over same recent window
    fwd = dates[i + 20]
    top_tot = [cols[j] for j in np.argsort(-tot_mom)[:TOPN]]
    top_rS = [cols[j] for j in np.argsort(-res_mom_std)[:TOPN]]
    top_rU = [cols[j] for j in np.argsort(-res_mom_unstd)[:TOPN]]
    tot_rets.append(port_fwd(top_tot, d, fwd))
    resS_rets.append(port_fwd(top_rS, d, fwd))
    resU_rets.append(port_fwd(top_rU, d, fwd))
    leg_dates.append(fwd)

idx = pd.Index(leg_dates)
s_tot = pd.Series(tot_rets, index=idx).dropna()
s_rS = pd.Series(resS_rets, index=idx).dropna()
s_rU = pd.Series(resU_rets, index=idx).dropna()


def metrics(s):
    eq = (1 + s).cumprod()
    yrs = len(s) * 20 / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sh = (s.mean() / s.std()) * np.sqrt(252 / 20)
    peak = eq.cummax(); mdd = ((eq - peak) / peak).min()
    return cagr, sh, mdd, eq


print("=" * 66, flush=True)
print(f"PHASE 3: RESIDUAL vs TOTAL-RETURN MOMENTUM (top-5, {idx.min().year}-{idx.max().year})")
print("  fit loadings over 504d, momentum over recent 231d sub-window")
print("=" * 66)
for label, s in [("Total-return momentum (current)", s_tot),
                 ("Residual momentum  STANDARDIZED (BHM)", s_rS),
                 ("Residual momentum  raw-cumulative", s_rU)]:
    cagr, sh, mdd, eq = metrics(s)
    print(f"\n  {label}")
    print(f"     CAGR {cagr*100:5.1f}%  Sharpe {sh:.2f}  MaxDD {mdd*100:.1f}%")
    for wl, y0, y1 in [("COVID 2020", "2020-02-01", "2020-04-30"), ("2022 reversal", "2022-01-01", "2022-12-31")]:
        w = eq[(eq.index >= pd.Timestamp(y0)) & (eq.index <= pd.Timestamp(y1))]
        if len(w) > 1:
            wdd = ((w - w.cummax()) / w.cummax()).min()
            print(f"       {wl}: ret {(w.iloc[-1]/w.iloc[0]-1)*100:+6.1f}%  intra-DD {wdd*100:.1f}%")
print("\nResidual wins if Sharpe/MaxDD improve (esp. crash-window DD) at acceptable CAGR cost.")
