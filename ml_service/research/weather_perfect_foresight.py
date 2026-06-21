"""
Decisive test of the 'weather forecast model' idea: grant PERFECT FORESIGHT.
Realized future weather = a flawless forecast. If trading on perfectly-known future
weather earns ~0, then no real (imperfect) forecast model can help. Tests the
perfect-foresight relationship corr(weather anomaly over [t,t+N], market-relative
basket return over [t,t+N]) across horizons N (incl. sub-seasonal), for each link.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASK = {
 "natgas_E&P": ["APA","CHK","CNX","CTRA","DVN","EQT","RRC","SWN"],
 "utilities": ["AEE","AEP","CMS","CNP","D","DTE","DUK","ED","ES","ETR","EXC","FE","NEE","NI","PCG","PEG","SO","SRE","WEC","XEL"],
 "retail": ["BBY","COST","DG","DLTR","HD","KSS","LOW","M","ROST","TGT","TJX","WMT"],
}
HOR = [5, 10, 20, 40, 60]   # trading-day horizons (incl. sub-seasonal / seasonal)

w = pd.read_parquet("research/_weather.parquet"); w.index = pd.to_datetime(w.index)
cl = pd.read_parquet("data/cached_close_prices.parquet"); cl.index = pd.to_datetime(cl.index)
dr = cl.pct_change()
# daily seasonal-normal anomaly (by day-of-year, full-sample normal)
w["doy"] = w.index.dayofyear
for c in ["hdd", "cdd", "temp"]:
    w[c + "_an"] = w[c] - w.groupby("doy")[c].transform("mean")

cal = cl.index  # trading days
pos = {d: i for i, d in enumerate(cal)}


def basket_ret(tickers, i0, i1):
    have = [t for t in tickers if t in dr.columns]
    seg = dr[have].iloc[i0+1:i1+1].mean(axis=1)
    spy = dr["SPY"].iloc[i0+1:i1+1]
    return (1 + seg).prod() - (1 + spy).prod()   # market-relative


def perfect_foresight(anom_col, tickers, N):
    rows = []
    for k in range(0, len(cal) - N, N):   # non-overlapping windows
        d0, d1 = cal[k], cal[k+N]
        wseg = w.loc[(w.index > d0) & (w.index <= d1), anom_col]
        if len(wseg) < N * 0.6:
            continue
        an = wseg.sum() if anom_col != "temp_an" else wseg.mean()
        rows.append((an, basket_ret(tickers, k, k+N)))
    a = pd.DataFrame(rows, columns=["an", "ret"]).dropna()
    if len(a) < 12:
        return None, 0
    return spearmanr(a["an"], a["ret"]).statistic, len(a)


print("=== PERFECT-FORESIGHT weather -> market-relative basket return (corr by horizon) ===")
print("  (if perfect foresight ~0, no forecast model can help)\n")
tests = [("HDD_an  -> natgas", "hdd_an", "natgas_E&P"),
         ("HDD_an  -> utilities", "hdd_an", "utilities"),
         ("CDD_an  -> utilities", "cdd_an", "utilities"),
         ("temp_an -> retail", "temp_an", "retail")]
print(f"  {'signal':<24} " + "  ".join(f"N={n}" for n in HOR))
for name, col, bk in tests:
    cells = []
    for N in HOR:
        c, n = perfect_foresight(col, BASK[bk], N)
        cells.append(f"{c:+.2f}" if c is not None else "  -- ")
    print(f"  {name:<24} " + "    ".join(cells))
print("\n  (corr; sample = non-overlapping windows, 2016-2024 ~ 2200/N windows)")
