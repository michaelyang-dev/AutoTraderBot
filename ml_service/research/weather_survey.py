"""
Shallow weather->equity survey (energy + retail). Weekly weather anomalies (vs ISO-week
seasonal normal) vs MARKET-RELATIVE forward 1-week basket returns. Reports both
CONTEMPORANEOUS corr (does weather move these stocks at all = sanity) and FORWARD IC
(tradeable lag/underreaction). Insurance handled separately (hurricane events).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASKETS = {
 "natgas_E&P": ["APA", "CHK", "CNX", "CTRA", "DVN", "EQT", "RRC", "SWN"],
 "utilities": ["AEE","AEP","CMS","CNP","D","DTE","DUK","ED","ES","ETR","EXC","FE","NEE","NI","PCG","PEG","SO","SRE","WEC","XEL"],
 "retail": ["BBY","COST","DG","DLTR","HD","KSS","LOW","M","ROST","TGT","TJX","WMT"],
 "restaurants": ["CMG","DPZ","DRI","MCD","SBUX","YUM"],
 "homebuilders": ["DHI","LEN","NVR","PHM"],
}

w = pd.read_parquet("research/_weather.parquet"); w.index = pd.to_datetime(w.index)
cl = pd.read_parquet("data/cached_close_prices.parquet"); cl.index = pd.to_datetime(cl.index)
dr = cl.pct_change()

# weekly weather (W-FRI): sum degree days, mean temp
wk = pd.DataFrame({
    "hdd": w["hdd"].resample("W-FRI").sum(),
    "cdd": w["cdd"].resample("W-FRI").sum(),
    "temp": w["temp"].resample("W-FRI").mean(),
})
wk["iso"] = wk.index.isocalendar().week.astype(int)
for c in ["hdd", "cdd", "temp"]:
    norm = wk.groupby("iso")[c].transform("mean")   # ISO-week seasonal normal (full-sample)
    wk[c + "_anom"] = wk[c] - norm

spy_wk = (1 + dr["SPY"]).resample("W-FRI").prod() - 1


def basket_weekly(tickers):
    have = [t for t in tickers if t in dr.columns]
    d = dr[have].mean(axis=1)               # equal-weight daily basket return
    return (1 + d).resample("W-FRI").prod() - 1


def test(name, anom_col, tickers, season=None):
    bw = basket_weekly(tickers)
    rel = (bw - spy_wk).dropna()            # market-relative weekly basket return
    df = pd.concat([wk[anom_col], rel.rename("ret")], axis=1).dropna()
    if season == "winter":
        df = df[df.index.month.isin([11, 12, 1, 2, 3])]
    elif season == "summer":
        df = df[df.index.month.isin([6, 7, 8, 9])]
    if len(df) < 30:
        print(f"  {name}: n<30"); return
    contemp = spearmanr(df[anom_col], df["ret"]).statistic
    fwd = df.copy(); fwd["fret"] = fwd["ret"].shift(-1)
    fwd = fwd.dropna()
    fic = spearmanr(fwd[anom_col], fwd["fret"]).statistic
    # quintile fwd spread
    try:
        fwd["q"] = pd.qcut(fwd[anom_col].rank(method="first"), 5, labels=False)
        spread = (fwd[fwd.q == 4]["fret"].mean() - fwd[fwd.q == 0]["fret"].mean()) * 100
    except Exception:
        spread = np.nan
    print(f"  {name:<34} contemp corr {contemp:+.3f}  |  FORWARD IC {fic:+.3f}  "
          f"top-bot fwd spread {spread:+.2f}%  n={len(fwd)}")


print("=== ENERGY: degree-day anomalies vs market-relative forward weekly returns ===")
test("HDD_anom -> natgas (winter)", "hdd_anom", BASKETS["natgas_E&P"], "winter")
test("HDD_anom -> natgas (all yr)", "hdd_anom", BASKETS["natgas_E&P"])
test("HDD_anom -> utilities (winter)", "hdd_anom", BASKETS["utilities"], "winter")
test("CDD_anom -> utilities (summer)", "cdd_anom", BASKETS["utilities"], "summer")
test("CDD_anom -> natgas (summer)", "cdd_anom", BASKETS["natgas_E&P"], "summer")

print("\n=== RETAIL/CONSUMER: temperature anomaly vs market-relative forward weekly returns ===")
test("temp_anom -> retail (all yr)", "temp_anom", BASKETS["retail"])
test("temp_anom -> retail (winter)", "temp_anom", BASKETS["retail"], "winter")
test("temp_anom -> restaurants (all)", "temp_anom", BASKETS["restaurants"])
test("temp_anom -> homebuilders (all)", "temp_anom", BASKETS["homebuilders"])
print("\n(contemp = does weather move them at all; FORWARD IC = tradeable underreaction)")
