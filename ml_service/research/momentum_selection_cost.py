"""
MOMENTUM SELECTION COST — the return impact of Polygon (live) vs WRDS (backtest) data.

An exact multi-year pp backtest is impossible: Massive/Polygon hard-caps history at ~376
trading days (2025-01-08 on), so the overlap with WRDS (ends 2025-12-31) is ~6 months.
Within that window this measures the DIRECT return cost of the selection difference:
at each 20-day rebalance, pick the momentum top-N by WRDS vs by Polygon, then score BOTH
baskets' forward 20-day return using the SAME clean WRDS prices. The gap = the pp cost of
selecting on Polygon data instead of WRDS, isolated from any P&L-pricing difference.

Both sources use the SAME adaptive lookback (longest Polygon supports at each date).
Small sample (~6 rebalances) -> wide error bars; reported as total + annualized with caveat.

Run local (pkl + Polygon panel local): python3 research/momentum_selection_cost.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))

TOPN = 5
HOLD = 20


def mom(panel, date, s, lb):
    if s not in panel:
        return None
    px = panel[s].dropna()
    px = px[px.index <= date]
    if len(px) < lb + 1:
        return None
    try:
        p0, p20, plb = float(px.iloc[-1]), float(px.iloc[-21]), float(px.iloc[-(lb + 1)])
    except (TypeError, ValueError):
        return None
    return (p0 / plb - 1) - (p0 / p20 - 1) if p20 > 0 and plb > 0 else None


def fwd_ret(wp, d, s, hold):
    """WRDS forward `hold`-trading-day return from date d (clean P&L reference)."""
    px = wp[s].dropna()
    idx = px.index[px.index <= d]
    if not len(idx):
        return None
    i = px.index.get_loc(idx[-1])
    if i + hold >= len(px):
        return None
    p0, p1 = float(px.iloc[i]), float(px.iloc[i + hold])
    return p1 / p0 - 1 if p0 > 0 else None


def main():
    d = pickle.load(open(ML / "data/wrds/complete_sp1500_universe.pkl", "rb"))
    wp = d["prices_df"]
    pp = pd.read_parquet(ML / "data/polygon_close_panel.parquet")
    pp.index = pd.to_datetime(pp.index)
    syms = [s for s in pp.columns if s in wp.columns]

    # rebal dates: every 20 td, 2025-07 .. where fwd return still fits in WRDS (ends 12-31)
    wd = [dt for dt in wp.index if pd.Timestamp("2025-07-01") <= dt <= pd.Timestamp("2025-12-01")]
    rebals = [wd[i] for i in range(0, len(wd), HOLD)]
    print(f"rebalances: {len(rebals)} ({rebals[0].date()} .. {rebals[-1].date()}), "
          f"top-{TOPN}, {HOLD}d hold, common syms {len(syms)}\n", flush=True)

    print(f"{'rebal':<12}{'lb':>4}{'WRDS ret':>10}{'Poly ret':>10}{'diff pp':>9}  swapped names", flush=True)
    print("-" * 70, flush=True)
    wr, pr = [], []
    for d0 in rebals:
        pdt = pp.index[pp.index <= d0][-1]
        ref = pp["AAPL"].dropna()                 # real non-NaN Polygon history (not index len)
        lb = min(200, (ref.index <= pdt).sum() - 25)
        if lb < 60:
            continue
        wm = {s: mom(wp, d0, s, lb) for s in syms}
        pm = {s: mom(pp, pdt, s, lb) for s in syms}
        wm = {s: v for s, v in wm.items() if v is not None}
        pm = {s: v for s, v in pm.items() if v is not None}
        common = [s for s in wm if s in pm]
        w5 = pd.Series({s: wm[s] for s in common}, dtype=float).nlargest(TOPN).index
        p5 = pd.Series({s: pm[s] for s in common}, dtype=float).nlargest(TOPN).index
        # forward return of each basket from CLEAN WRDS prices
        wf = [fwd_ret(wp, d0, s, HOLD) for s in w5]
        pf = [fwd_ret(wp, d0, s, HOLD) for s in p5]
        wf = [x for x in wf if x is not None]
        pf = [x for x in pf if x is not None]
        if not wf or not pf:
            continue
        wret, pret = float(np.mean(wf)), float(np.mean(pf))
        wr.append(wret); pr.append(pret)
        swap = sorted(set(w5) ^ set(p5))
        print(f"{str(d0.date()):<12}{lb:>4}{wret*100:>9.2f}%{pret*100:>9.2f}%"
              f"{(wret-pret)*100:>8.2f}  {swap if swap else 'identical'}", flush=True)

    wcum = float(np.prod([1 + r for r in wr]) - 1)
    pcum = float(np.prod([1 + r for r in pr]) - 1)
    n = len(wr)
    yrs = n * HOLD / 252
    print("\n" + "=" * 70, flush=True)
    print(f"over {n} rebalances (~{yrs:.2f} yr):", flush=True)
    print(f"  WRDS-selected momentum basket:    {wcum*100:+.2f}% cumulative", flush=True)
    print(f"  Polygon-selected momentum basket: {pcum*100:+.2f}% cumulative", flush=True)
    print(f"  SELECTION COST (WRDS - Polygon):  {(wcum-pcum)*100:+.2f}pp over the window", flush=True)
    if yrs > 0:
        ann = (1 + wcum) ** (1 / yrs) - (1 + pcum) ** (1 / yrs)
        print(f"  annualized gap (WIDE error bars, n={n}): {ann*100:+.2f}pp/yr", flush=True)
    print("  NOTE: momentum-sleeve selection only (the dominant, most price-sensitive", flush=True)
    print("  sleeve); value/low-vol add little. 6-mo window = indicative, not a clean CAGR.", flush=True)


if __name__ == "__main__":
    main()
