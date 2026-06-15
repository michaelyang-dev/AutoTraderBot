"""
Phase 7 — Earnings estimate REVISIONS as a within-winners dispersion signal.
The one unexplored idea prior research flagged as highest-EV: revisions are
momentum-adjacent (information diffusion), so unlike valuation factors they may
COMPLEMENT momentum rather than dilute it, and may create dispersion AMONG
winners (the only thing that can help, per the lazy-prices 3-gate lesson).

Signal: net_rev = (NUMUP - NUMDOWN) / NUMEST on FY1 EPS, latest IBES monthly
snapshot (STATPERS) <= rebalance date. Point-in-time, no look-ahead.

Three gates:
  1. Predictive?  rank-IC vs forward 20d return.
  2. Orthogonal?  correlation with skip-month momentum.
  3. Binds on winners?  within the momentum top-quintile, does sorting by
     revisions separate forward returns? (else it's redundant with momentum)
Plus practical: top-5 momentum vs top-5 momentum filtered to positive revisions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

bt = FastBacktester()
prices = bt.prices
bt.uni.get_sp500 = bt._get_sp1500
allidx = list(prices.index)

# ── Load IBES FY1 EPS revisions ──
print("loading IBES estimate revisions...", flush=True)
ib = pd.read_parquet("data/wrds/ibes_summary_history.parquet",
                     columns=["OFTIC", "STATPERS", "MEASURE", "FPI", "NUMEST", "NUMUP", "NUMDOWN", "MEDEST"],
                     filters=[("MEASURE", "=", "EPS"), ("FPI", "=", "1")])
ib["STATPERS"] = pd.to_datetime(ib["STATPERS"])
ib = ib[ib["STATPERS"] >= "2016-06-01"].dropna(subset=["OFTIC", "NUMEST"])
ib = ib[ib["NUMEST"] > 0]
ib["net_rev"] = (ib["NUMUP"].fillna(0) - ib["NUMDOWN"].fillna(0)) / ib["NUMEST"]
ib = ib.sort_values("STATPERS")
# per-ticker arrays for fast as-of lookup
REV = {}
for tic, g in ib.groupby("OFTIC"):
    if tic in prices.columns:
        REV[tic] = (g["STATPERS"].values, g["net_rev"].values)
print(f"tickers with revisions data in price panel: {len(REV)}", flush=True)


def rev_asof(tic, d):
    arr = REV.get(tic)
    if arr is None:
        return np.nan
    sp, nr = arr
    i = np.searchsorted(sp, np.datetime64(d), "right") - 1
    if i < 0 or (np.datetime64(d) - sp[i]) > np.timedelta64(100, "D"):
        return np.nan
    return nr[i]


dates = [d for d in prices.index if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")]
ics, corrs = [], []
g3_hi, g3_lo = [], []            # gate-3: within momentum winners, top vs bottom revisions
plain5, filt5 = [], []           # practical: top-5 mom vs top-5 mom w/ positive revisions

for k in range(0, len(dates) - 21, 20):
    d = dates[k]; loc = allidx.index(d)
    if loc < 260:
        continue
    members = [m for m in bt.uni.get_sp500(d) if m in prices.columns]
    p0 = prices.loc[d]; p20 = prices.loc[allidx[loc - 20]]; p252 = prices.loc[allidx[loc - 252]]
    fwd = dates[k + 20]; pf = prices.loc[fwd]
    rows = []
    for m in members:
        a, a20, a252, af = p0.get(m), p20.get(m), p252.get(m), pf.get(m)
        if not (a and a252 and af) or np.isnan(a) or np.isnan(a252) or np.isnan(af):
            continue
        mom = (a / a252 - 1) - (a / a20 - 1) if a20 and not np.isnan(a20) else (a / a252 - 1)
        r = rev_asof(m, d)
        rows.append((m, mom, r, af / a - 1))
    if len(rows) < 50:
        continue
    df = pd.DataFrame(rows, columns=["sym", "mom", "rev", "fwd"])
    valid = df.dropna(subset=["rev"])
    if len(valid) > 30:
        ics.append(spearmanr(valid["rev"], valid["fwd"]).correlation)
        corrs.append(spearmanr(valid["rev"], valid["mom"]).correlation)
        # Gate 3: top momentum quintile, split by revisions
        topq = valid[valid["mom"] >= valid["mom"].quantile(0.80)]
        if len(topq) >= 10:
            med = topq["rev"].median()
            g3_hi.append(topq[topq["rev"] > med]["fwd"].mean())
            g3_lo.append(topq[topq["rev"] <= med]["fwd"].mean())
    # Practical: top-5 by momentum vs top-5 by momentum among positive-revision names
    top5 = df.nlargest(5, "mom")["fwd"].mean()
    posrev = df[df["rev"] > 0]
    f5 = posrev.nlargest(5, "mom")["fwd"].mean() if len(posrev) >= 5 else top5
    plain5.append(top5); filt5.append(f5)


def ann(series):
    s = pd.Series(series).dropna()
    eq = (1 + s).cumprod(); yrs = len(s) * 20 / 252
    return eq.iloc[-1] ** (1 / yrs) - 1, (s.mean() / s.std()) * np.sqrt(252 / 20)


print("\n" + "=" * 64)
print("PHASE 7: ESTIMATE REVISIONS (FY1 EPS net-up ratio), 2018-2025")
print("=" * 64)
print(f"\n  GATE 1 — Predictive?   mean rank-IC vs fwd 20d = {np.nanmean(ics):+.4f}  (>0.02 = useful)")
print(f"  GATE 2 — Orthogonal?   corr with momentum      = {np.nanmean(corrs):+.3f}  (near 0 = independent)")
hi, lo = np.nanmean(g3_hi), np.nanmean(g3_lo)
print(f"  GATE 3 — Binds on winners? within top-mom quintile, per-20d fwd return:")
print(f"             high-revision half {hi*100:+.2f}%  vs  low-revision half {lo*100:+.2f}%  (spread {(hi-lo)*100:+.2f}pp)")
cp, sp = ann(plain5); cf, sf = ann(filt5)
print(f"\n  PRACTICAL — top-5 momentum sleeve:")
print(f"     plain                {cp*100:5.1f}% CAGR  Sharpe {sp:.2f}")
print(f"     + positive-revisions {cf*100:5.1f}% CAGR  Sharpe {sf:.2f}")
print("\n  Verdict: useful only if IC>0.02, corr near 0, AND gate-3 spread is clearly positive.")
