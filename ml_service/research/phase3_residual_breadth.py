"""
Phase 3b — Is residual momentum's CRASH protection real in THIS universe/period?
BHM documented lower crash risk on BROAD (decile) portfolios, not a top-5.
So separate two questions:
  (1) does our top-5 sleeve benefit?  (already: no)
  (2) does the BHM crash-protection claim hold AT BREADTH here?  <- this script

For N in {5, 20, 50} (long-only equal-weight), compare total-return vs
standardized-residual momentum on CAGR/Sharpe/MaxDD and the COVID-2020 crash.
If residual lowers crash DD only at breadth -> "needs breadth, not our sleeve".
If it never lowers crash DD -> the claim doesn't transfer to 2018-2025 mega-caps.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd

EST, FORM, SKIP = 504, 252, 21
MEAS = FORM - SKIP

bt = FastBacktester(); prices = bt.prices; rets = prices.pct_change()
bt.uni.get_sp500 = bt._get_sp1500
ff = pd.read_parquet("data/wrds/fama_french_5factors_momentum_daily.parquet")
ff["date"] = pd.to_datetime(ff["date"]); ff = ff.set_index("date").sort_index()
F = ff[["mktrf", "smb", "hml"]].reindex(prices.index).fillna(0.0)
RF = ff["rf"].reindex(prices.index).fillna(0.0)
dates = [d for d in prices.index if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")]
allidx = list(prices.index)

# Per-rebalance: store signals + each stock's forward return vector
recs = []
for i in range(0, len(dates) - 21, 20):
    d = dates[i]; loc = allidx.index(d)
    if loc < EST + 5:
        continue
    members = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
    est_win = allidx[loc - EST: loc - SKIP]
    if len(est_win) < 300:
        continue
    Y = rets.loc[est_win, members].values
    valid = np.isfinite(Y).sum(axis=0) > 0.9 * len(est_win)
    cols = [members[j] for j in range(len(members)) if valid[j]]
    Yv = np.nan_to_num(Y[:, valid])
    exc = Yv - RF.loc[est_win].values[:, None]
    X = np.column_stack([np.ones(len(est_win)), F.loc[est_win].values])
    beta, *_ = np.linalg.lstsq(X, exc, rcond=None)
    resid = (exc - X @ beta)[-MEAS:]
    res_mom = resid.sum(axis=0) / (resid.std(axis=0) + 1e-9)
    tot_mom = Yv[-MEAS:].sum(axis=0)
    fwd = dates[i + 20]
    p0 = prices.loc[d, cols].values; p1 = prices.loc[fwd, cols].values
    fr = np.where((p0 > 0) & np.isfinite(p0) & np.isfinite(p1), p1 / p0 - 1, np.nan)
    recs.append((fwd, np.array(cols), tot_mom, res_mom, fr))


def curve(signal_key, N):
    out = []
    for fwd, cols, tm, rm, fr in recs:
        sig = tm if signal_key == "tot" else rm
        order = np.argsort(-sig)[:N]
        r = np.nanmean(fr[order])
        out.append((fwd, r))
    s = pd.Series([r for _, r in out], index=[f for f, _ in out]).dropna()
    eq = (1 + s).cumprod()
    yrs = len(s) * 20 / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sh = (s.mean() / s.std()) * np.sqrt(252 / 20)
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    w = eq[(eq.index >= "2020-02-01") & (eq.index <= "2020-04-30")]
    covid = (w.iloc[-1] / w.iloc[0] - 1) if len(w) > 1 else np.nan
    coviddd = ((w - w.cummax()) / w.cummax()).min() if len(w) > 1 else np.nan
    return cagr, sh, mdd, covid, coviddd


print("=" * 72)
print(f"PHASE 3b: residual crash-protection at BREADTH ({recs[0][0].year}-{recs[-1][0].year})")
print("=" * 72)
print(f"\n{'N':>4} {'signal':<10}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'COVID ret':>11}{'COVID DD':>10}")
for N in [5, 20, 50]:
    for key, lab in [("tot", "total"), ("res", "residual")]:
        cg, sh, md, cv, cvd = curve(key, N)
        print(f"{N:>4} {lab:<10}{cg*100:>7.1f}%{sh:>8.2f}{md*100:>7.1f}%{cv*100:>10.1f}%{cvd*100:>9.1f}%")
    print()
print("Claim holds only if residual's COVID DD < total's COVID DD (lower crash risk).")
