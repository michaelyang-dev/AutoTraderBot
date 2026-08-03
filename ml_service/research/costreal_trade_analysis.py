"""What does the SP1500 book ACTUALLY trade? tier mix, size, ADV participation, spread."""
import os, pickle, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from costreal_cache import SP, ensure_cache  # noqa: E402
ensure_cache()

TR = pickle.load(open(SP + "trades.pkl", "rb"))
mem = pickle.load(open(SP + "mem.pkl", "rb"))
mk = {k: np.array(sorted(v.keys())) for k, v in mem.items()}


def tier_of(sym, dt):
    for t, lbl in [("sp500_mem", "SP500"), ("sp400_mem", "SP400"), ("sp600_mem", "SP600")]:
        ks = mk[t]
        i = np.searchsorted(ks, dt, side="right") - 1
        if i >= 0 and sym in mem[t][ks[i]]:
            return lbl
    return "other"


liq = pd.read_parquet(SP + "liq.parquet")
liq["date"] = pd.to_datetime(liq["YYYYMMDD"], format="%Y%m%d")
dv = liq["DlyPrcVol"].where(liq["DlyPrcVol"] > 0, liq["DlyClose"].abs() * liq["DlyVol"])
liq["dvol"] = dv
mid = (liq["DlyBid"] + liq["DlyAsk"]) / 2
liq["spr"] = ((liq["DlyAsk"] - liq["DlyBid"]) / mid * 1e4).where((liq["DlyAsk"] > liq["DlyBid"]) & (mid > 0))
liq.loc[liq["spr"] > 2000, "spr"] = np.nan
liq = liq.sort_values("dvol").drop_duplicates(["Ticker", "date"], keep="last")
ADV = liq.pivot(index="date", columns="Ticker", values="dvol").sort_index().rolling(20, min_periods=5).mean()
SPRD = liq.pivot(index="date", columns="Ticker", values="spr").sort_index().ffill(limit=10)
PX = liq.pivot(index="date", columns="Ticker", values="DlyClose").abs().sort_index()

KEY_B = "B realistic: half-quoted-spread+impact+comm"
for pool in ["SP1500 (fixed)", "SP500-only (bug)"]:
    t = TR[(KEY_B, pool)].copy()
    t = t[t.dollars > 0]
    t["tier"] = [tier_of(s, d) for s, d in zip(t["sym"], t["date"])]
    t["adv"] = [ADV.at[d, s] if (d in ADV.index and s in ADV.columns) else np.nan
                for d, s in zip(t["date"], t["sym"])]
    t["spr"] = [SPRD.at[d, s] if (d in SPRD.index and s in SPRD.columns) else np.nan
                for d, s in zip(t["date"], t["sym"])]
    t["px"] = [PX.at[d, s] if (d in PX.index and s in PX.columns) else np.nan
               for d, s in zip(t["date"], t["sym"])]
    t["part"] = t.dollars / t.adv * 100
    tot = t.dollars.sum()
    print("\n" + "=" * 118)
    print(f"POOL = {pool}   (3 starts pooled; {len(t)} trade legs, ${tot:,.0f} traded)")
    print(f"{'tier':<8}{'legs':>7}{'%legs':>7}{'%$traded':>10}{'med $sz':>10}{'p95 $sz':>10}"
          f"{'med ADV$':>13}{'med part%':>11}{'p95 part%':>11}{'med spr bp':>12}{'med px':>9}"
          f"{'$w cost bp':>11}")
    for tier in ["SP500", "SP400", "SP600", "other"]:
        s = t[t.tier == tier]
        if len(s) == 0:
            continue
        w = (s.cf * s.dollars).sum() / s.dollars.sum() * 1e4
        print(f"{tier:<8}{len(s):>7}{len(s)/len(t)*100:>7.1f}{s.dollars.sum()/tot*100:>10.1f}"
              f"{s.dollars.median():>10,.0f}{s.dollars.quantile(.95):>10,.0f}"
              f"{s.adv.median():>13,.0f}{s.part.median():>11.4f}{s.part.quantile(.95):>11.4f}"
              f"{s.spr.median():>12.1f}{s.px.median():>9.2f}{w:>11.2f}")
    allw = (t.cf * t.dollars).sum() / tot * 1e4
    print(f"{'ALL':<8}{len(t):>7}{100.0:>7.1f}{100.0:>10.1f}{t.dollars.median():>10,.0f}"
          f"{t.dollars.quantile(.95):>10,.0f}{t.adv.median():>13,.0f}{t.part.median():>11.4f}"
          f"{t.part.quantile(.95):>11.4f}{t.spr.median():>12.1f}{t.px.median():>9.2f}{allw:>11.2f}")
    print(f"  worst-case participation: max={t.part.max():.3f}%  "
          f"legs>0.5%ADV: {(t.part > 0.5).sum()} ({(t.part>0.5).mean()*100:.2f}%)  "
          f"legs>1%ADV: {(t.part > 1).sum()}")
    print(f"  cost decomposition ($-wtd bps): half-spread {(t.half_spread*t.dollars).sum()/tot*1e4:.2f}"
          f"   impact+commission {(t.impact_comm*t.dollars).sum()/tot*1e4:.2f}")
    # turnover
    yrs = 8.0
    print(f"  $ traded/yr per start: ${tot/3/yrs:,.0f}")
