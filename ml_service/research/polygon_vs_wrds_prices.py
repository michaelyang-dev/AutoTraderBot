"""
POLYGON (live) vs WRDS/CRSP (backtest) PRICE-DATA PARITY.

Live momentum is computed from Polygon/Massive bars; the +28.3% backtest uses WRDS/CRSP
prices (prices_df in the universe pkl). If the two disagree enough to reorder momentum
ranks, live would pick different stocks than the backtest. This test computes momentum
IDENTICALLY from both sources on matched historical dates and reports:
  - price_corr    : log-close level agreement (should be ~1.0)
  - mom_rankcorr  : Spearman rank corr of the 12-1 momentum score (the thing that ranks)
  - top5 / top20  : overlap of the momentum SELECTION (what actually gets traded)
  - med_px_diff%  : median |Polygon-WRDS|/WRDS close (root-cause magnitude)

High rank corr + high top-5 overlap => the data source does NOT change selection.
Run on AWS (Polygon access): venv/bin/python research/polygon_vs_wrds_prices.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
from massive_data_provider import fetch_bars_batch_massive  # noqa: E402


def mom_at(panel, date, s, lb):
    """Skip-20 momentum with lookback `lb` trading days: ret_lb - ret_20, using the
    panel's own trading-day grid. Adaptive lb because this Polygon plan only returns
    ~2025-onward history (capped ~1.5yr), so 252-1 has no WRDS overlap — but if prices
    agree, any-horizon momentum agrees, and a ~200d momentum is a strong proxy."""
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
    if p20 > 0 and plb > 0:
        return (p0 / plb - 1) - (p0 / p20 - 1)
    return None


def last_le(panel, date, s):
    px = panel[s].dropna()
    px = px[px.index <= date]
    try:
        return float(px.iloc[-1]) if len(px) else None
    except (TypeError, ValueError):
        return None


PANEL = ML / "data/polygon_close_panel.parquet"


def fetch_mode():
    """AWS (reliable Polygon connection): fetch current members, save close panel."""
    members = json.load(open(ML / "data/sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    print(f"fetching {len(syms)} current members from Polygon...", flush=True)
    bars = fetch_bars_batch_massive(syms, warmup_days=900)
    poly = {s: bars[s]["close"] for s in syms if bars.get(s) is not None and len(bars[s])}
    pp = pd.DataFrame(poly)
    pp.to_parquet(PANEL)
    print(f"saved Polygon panel: {pp.shape[1]} symbols, {pp.shape[0]} days "
          f"({pp.index.min().date()} -> {pp.index.max().date()}) -> {PANEL}", flush=True)


def main():
    d = pickle.load(open(ML / "data/wrds/complete_sp1500_universe.pkl", "rb"))
    wp = d["prices_df"]                       # WRDS/CRSP close, dates x symbols
    pp = pd.read_parquet(PANEL)               # Polygon close, fetched on AWS
    pp.index = pd.to_datetime(pp.index)
    syms = [s for s in pp.columns if s in wp.columns]
    print(f"current members in BOTH WRDS + Polygon: {len(syms)}", flush=True)
    print(f"Polygon panel: {pp.shape[1]} symbols, {pp.shape[0]} days "
          f"({pp.index.min().date()} -> {pp.index.max().date()})", flush=True)

    # overlap where Polygon has data: 2025 (WRDS ends 2025-12; Polygon dense from 2025-01)
    poly_start = pp.apply(lambda c: c.first_valid_index()).min()
    wdates = [dt for dt in wp.index if dt.year == 2025 and dt >= pd.Timestamp("2025-07-01")]
    test_dates = [wdates[i] for i in range(0, len(wdates), 21)][-5:]  # ~monthly, H2 2025

    print("\nPART A — DIRECT PRICE AGREEMENT (root cause: if prices match, all price "
          "signals match)", flush=True)
    hdr = f"{'date':<12}{'n':>5}{'logpx_corr':>11}{'medΔpx%':>9}{'p90Δpx%':>9}{'>2%diff':>9}"
    print(hdr); print("-" * len(hdr), flush=True)
    for td in test_dates:
        pdt = pp.index[pp.index <= td][-1]
        wl, pl, pxdiff = {}, {}, []
        for s in syms:
            wpx, ppx = last_le(wp, td, s), last_le(pp, pdt, s)
            if wpx and ppx and wpx > 0:
                wl[s], pl[s] = wpx, ppx
                pxdiff.append(abs(ppx - wpx) / wpx * 100)
        pc = np.log(pd.Series(wl, dtype=float)).corr(np.log(pd.Series(pl, dtype=float)))
        pxd = np.array(pxdiff)
        print(f"{str(td.date()):<12}{len(pxd):>5}{pc:>11.5f}{np.median(pxd):>9.3f}"
              f"{np.percentile(pxd, 90):>9.3f}{(pxd > 2).mean() * 100:>8.1f}%", flush=True)

    print("\nPART B — MOMENTUM RANK + TOP-N SELECTION (skip-20, adaptive lookback to "
          "Polygon's ~2025 history)", flush=True)
    hdr = f"{'date':<12}{'lb':>5}{'n':>5}{'mom_rankcorr':>13}{'top5':>8}{'top20':>8}"
    print(hdr); print("-" * len(hdr), flush=True)
    for td in test_dates:
        pdt = pp.index[pp.index <= td][-1]
        lb = min(220, (pp.index <= pdt).sum() - 25)  # longest lookback Polygon supports
        if lb < 60:
            continue
        wmom, pmom = {}, {}
        for s in syms:
            wm, pm = mom_at(wp, td, s, lb), mom_at(pp, pdt, s, lb)
            if wm is not None and pm is not None:
                wmom[s], pmom[s] = wm, pm
        w, p = pd.Series(wmom, dtype=float), pd.Series(pmom, dtype=float)
        rc = w.rank().corr(p.rank())
        w5, p5 = set(w.nlargest(5).index), set(p.nlargest(5).index)
        w20, p20 = set(w.nlargest(20).index), set(p.nlargest(20).index)
        print(f"{str(td.date()):<12}{lb:>5}{len(w):>5}{rc:>13.4f}"
              f"{len(w5 & p5):>6}/5{len(w20 & p20):>6}/20", flush=True)
        if len(w5 & p5) < 5:
            print(f"    WRDS top5:    {sorted(w5)} | Polygon top5: {sorted(p5)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch_mode()
    else:
        main()
