"""Liquidity profile of SP500 vs SP400 vs SP600 from CRSP daily (2018-2025).
ADV20 dollar volume + quoted spread (DlyBid/DlyAsk) -> realistic cost for a $X trade."""
import os, pickle, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd

import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from costreal_cache import SP, ensure_cache  # noqa: E402
ensure_cache()
TRADE = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0

d = pd.read_parquet(SP + "liq.parquet")
d["date"] = pd.to_datetime(d["YYYYMMDD"], format="%Y%m%d")
d = d.dropna(subset=["Ticker"])
d = d.sort_values(["PERMNO", "date"])
# dollar volume: prefer DlyPrcVol, fall back to |close|*vol
dv = d["DlyPrcVol"].where(d["DlyPrcVol"] > 0, d["DlyClose"].abs() * d["DlyVol"])
d["dvol"] = dv
d["adv20"] = d.groupby("PERMNO")["dvol"].transform(lambda s: s.rolling(20, min_periods=10).mean())
mid = (d["DlyBid"] + d["DlyAsk"]) / 2
d["spr_bps"] = ((d["DlyAsk"] - d["DlyBid"]) / mid * 1e4).where((d["DlyAsk"] > d["DlyBid"]) & (mid > 0))
d.loc[d["spr_bps"] > 2000, "spr_bps"] = np.nan   # bad quotes
d["px"] = d["DlyClose"].abs()

mem = pickle.load(open(SP + "mem.pkl", "rb"))
mem_keys = {k: sorted(v.keys()) for k, v in mem.items()}


def members(tier, dt):
    ks = mem_keys[tier]
    i = np.searchsorted(ks, dt, side="right") - 1
    return set(mem[tier][ks[i]]) if i >= 0 else set()


dates = sorted(d["date"].unique())
dates = [x for x in dates if pd.Timestamp("2018-01-01") <= x <= pd.Timestamp("2025-12-31")]
# month-end sample
samp = pd.Series(dates, index=pd.DatetimeIndex(dates)).resample("ME").last().dropna().tolist()
print(f"sample dates: {len(samp)}  {samp[0].date()} -> {samp[-1].date()}   trade size ${TRADE:,.0f}")

by = {t: [] for t in ["sp500_mem", "sp400_mem", "sp600_mem"]}
d = d.set_index("date").sort_index()
rows = []
for dt in samp:
    day = d.loc[[dt]] if dt in d.index else None
    if day is None or len(day) == 0:
        continue
    day = day.dropna(subset=["adv20"])
    day = day[~day["Ticker"].duplicated(keep="last")].set_index("Ticker")
    for tier in by:
        ms = members(tier, dt) & set(day.index)
        if not ms:
            continue
        sub = day.loc[sorted(ms)]
        rows.append(pd.DataFrame({"date": dt, "tier": tier, "ticker": sub.index,
                                  "adv20": sub["adv20"].values, "spr": sub["spr_bps"].values,
                                  "px": sub["px"].values}))
R = pd.concat(rows, ignore_index=True)
R.to_parquet(SP + "liq_panel.parquet")

print(f"\n{'tier':<10}{'n/mo':>7}{'ADV20 p5':>12}{'p25':>12}{'median':>12}{'p75':>12}"
      f"{'  |  spread bps p50':>20}{'p75':>8}{'p95':>8}{'  px p50':>9}")
for tier in ["sp500_mem", "sp400_mem", "sp600_mem"]:
    s = R[R.tier == tier]
    a = s["adv20"]
    sp = s["spr"].dropna()
    print(f"{tier:<10}{len(s)/len(samp):>7.0f}{a.quantile(.05):>12,.0f}{a.quantile(.25):>12,.0f}"
          f"{a.median():>12,.0f}{a.quantile(.75):>12,.0f}{sp.median():>20.1f}{sp.quantile(.75):>8.1f}"
          f"{sp.quantile(.95):>8.1f}{s['px'].median():>9.2f}")

print(f"\nPARTICIPATION of a ${TRADE:,.0f} order (trade/ADV20), percent:")
print(f"{'tier':<10}{'p50':>10}{'p75':>10}{'p90':>10}{'p95':>10}{'p99':>10}{'max':>10}"
      f"{'  %names >1% ADV':>18}{'  %>0.1%':>10}")
for tier in ["sp500_mem", "sp400_mem", "sp600_mem"]:
    s = R[R.tier == tier]
    p = TRADE / s["adv20"] * 100
    print(f"{tier:<10}{p.quantile(.5):>10.4f}{p.quantile(.75):>10.4f}{p.quantile(.9):>10.4f}"
          f"{p.quantile(.95):>10.4f}{p.quantile(.99):>10.4f}{p.max():>10.4f}"
          f"{(p > 1).mean()*100:>18.3f}{(p > 0.1).mean()*100:>10.3f}")

# IBKR Pro commission: $0.005/sh, min $1, max 1% of value
sh = TRADE / R["px"]
comm = np.clip(sh * 0.005, 1.0, TRADE * 0.01)
R["comm_bps"] = comm / TRADE * 1e4
print(f"\nIBKR Pro fixed commission on a ${TRADE:,.0f} order ($0.005/sh, $1 min, 1% max), bps:")
print(f"{'tier':<10}{'p25':>8}{'p50':>8}{'p75':>8}{'p95':>8}{'mean':>8}")
for tier in ["sp500_mem", "sp400_mem", "sp600_mem"]:
    c = R[R.tier == tier]["comm_bps"]
    print(f"{tier:<10}{c.quantile(.25):>8.2f}{c.quantile(.5):>8.2f}{c.quantile(.75):>8.2f}"
          f"{c.quantile(.95):>8.2f}{c.mean():>8.2f}")
