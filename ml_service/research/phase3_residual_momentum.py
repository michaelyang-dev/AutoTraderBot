"""
Phase 3 — Residual (idiosyncratic) momentum, Blitz-Huij-Martens (2011).
Strip factor exposure from each stock's returns, compute momentum on the
RESIDUAL (stock-specific) return. Documented: similar/higher returns, LOWER
vol and crash risk -> a tail tool, not just an alpha hunt.

Compares the momentum SLEEVE only (top-5), residual vs total-return momentum,
on CAGR / Sharpe / MaxDD AND crash-window drawdowns (the tail claim).
Efficient: one batched OLS per rebalance date (all stocks at once).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

FORM = 252   # formation window (trading days)
SKIP = 21    # skip last ~1 month
TOPN = 5

bt = FastBacktester()
prices = bt.prices
rets = prices.pct_change()
bt.uni.get_sp500 = bt._get_sp1500

# Fama-French daily factors aligned to price dates
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


tot_rets, res_rets, leg_dates = [], [], []
for i in range(0, len(dates) - 21, 20):
    d = dates[i]
    loc = allidx.index(d)
    if loc < FORM + 5:
        continue
    members = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
    win = allidx[loc - FORM: loc - SKIP]          # formation window, skip last month
    if len(win) < 100:
        continue
    Y = rets.loc[win, members].values             # (T x N) stock returns
    Xf = F.loc[win].values                         # (T x 3) factors
    rfw = RF.loc[win].values
    # valid stocks: enough non-nan
    valid = np.isfinite(Y).sum(axis=0) > 0.8 * len(win)
    cols = [members[j] for j in range(len(members)) if valid[j]]
    Yv = np.nan_to_num(Y[:, valid])
    exc = Yv - rfw[:, None]                         # excess returns
    X = np.column_stack([np.ones(len(win)), Xf])   # intercept + 3 factors
    # batched OLS: beta = (X'X)^-1 X'Y  -> residuals = Y - X beta
    beta, *_ = np.linalg.lstsq(X, exc, rcond=None)
    resid = exc - X @ beta                          # (T x Nv) idiosyncratic returns
    # signals
    tot_mom = np.nansum(Yv, axis=0)                 # total-return momentum (cum return over window)
    res_cum = resid.sum(axis=0)
    res_std = resid.std(axis=0) + 1e-9
    res_mom = res_cum / res_std                      # standardized residual momentum
    fwd = dates[i + 20]
    top_tot = [cols[j] for j in np.argsort(-tot_mom)[:TOPN]]
    top_res = [cols[j] for j in np.argsort(-res_mom)[:TOPN]]
    tot_rets.append(port_fwd(top_tot, d, fwd))
    res_rets.append(port_fwd(top_res, d, fwd))
    leg_dates.append(fwd)

s_tot = pd.Series(tot_rets, index=leg_dates).dropna()
s_res = pd.Series(res_rets, index=leg_dates).dropna()


def metrics(s):
    eq = (1 + s).cumprod()
    yrs = len(s) * 20 / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sh = (s.mean() / s.std()) * np.sqrt(252 / 20)
    peak = eq.cummax(); mdd = ((eq - peak) / peak).min()
    return cagr, sh, mdd, eq


print("=" * 64, flush=True)
print("PHASE 3: RESIDUAL vs TOTAL-RETURN MOMENTUM (top-5 sleeve, 2018-2025)")
print("=" * 64)
for label, s in [("Total-return momentum (current)", s_tot), ("Residual momentum (BHM)", s_res)]:
    cagr, sh, mdd, eq = metrics(s)
    print(f"\n  {label}")
    print(f"     CAGR {cagr*100:5.1f}%  Sharpe {sh:.2f}  MaxDD {mdd*100:.1f}%")
    for wl, y0, y1 in [("COVID 2020", "2020-02-01", "2020-04-30"), ("2022 reversal", "2022-01-01", "2022-12-31")]:
        w = eq[(eq.index >= pd.Timestamp(y0)) & (eq.index <= pd.Timestamp(y1))]
        if len(w) > 1:
            wdd = ((w - w.cummax()) / w.cummax()).min()
            print(f"       {wl}: return {(w.iloc[-1]/w.iloc[0]-1)*100:+.1f}%  intra-DD {wdd*100:.1f}%")
print("\nResidual wins if Sharpe/MaxDD improve (esp. crash-window DD) — the tail claim.")
