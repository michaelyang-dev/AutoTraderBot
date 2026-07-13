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


def mom_at(panel, date, s):
    """12-1 momentum from a close panel: ret_252d - ret_20d, using the panel's own
    trading-day grid (each source uses its own 20/252 trading days back)."""
    if s not in panel:
        return None
    px = panel[s].dropna()
    px = px[px.index <= date]
    if len(px) < 253:
        return None
    p0, p20, p252 = px.iloc[-1], px.iloc[-21], px.iloc[-253]
    if p20 > 0 and p252 > 0:
        return (p0 / p252 - 1) - (p0 / p20 - 1)
    return None


def last_le(panel, date, s):
    px = panel[s].dropna()
    px = px[px.index <= date]
    return px.iloc[-1] if len(px) else None


def main():
    d = pickle.load(open(ML / "data/wrds/complete_sp1500_universe.pkl", "rb"))
    wp = d["prices_df"]                       # WRDS/CRSP close, dates x symbols
    members = json.load(open(ML / "data/sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    syms = [s for s in syms if s in wp.columns]
    print(f"current members present in WRDS: {len(syms)}", flush=True)

    bars = fetch_bars_batch_massive(syms, warmup_days=900)
    poly = {s: bars[s]["close"] for s in syms if bars.get(s) is not None and len(bars[s])}
    pp = pd.DataFrame(poly)
    print(f"Polygon panel: {pp.shape[1]} symbols, {pp.shape[0]} days "
          f"({pp.index.min().date()} -> {pp.index.max().date()})", flush=True)

    wdates = [dt for dt in wp.index if dt.year == 2025]
    test_dates = [wdates[i] for i in range(0, len(wdates), 21)][-4:]  # ~monthly, last 4 of 2025

    hdr = f"{'date':<12}{'n':>5}{'price_corr':>11}{'mom_rankcorr':>13}{'top5':>8}{'top20':>8}{'medΔpx%':>9}"
    print(hdr); print("-" * len(hdr), flush=True)
    for td in test_dates:
        pdts = pp.index[pp.index <= td]
        if not len(pdts):
            continue
        pdt = pdts[-1]
        wmom, pmom, wl, pl, pxdiff = {}, {}, {}, {}, []
        for s in syms:
            wm, pm = mom_at(wp, td, s), mom_at(pp, pdt, s)
            if wm is None or pm is None:
                continue
            wmom[s], pmom[s] = wm, pm
            wpx, ppx = last_le(wp, td, s), last_le(pp, pdt, s)
            wl[s], pl[s] = wpx, ppx
            if wpx and wpx > 0:
                pxdiff.append(abs(ppx - wpx) / wpx * 100)
        common = list(wmom)
        w, p = pd.Series(wmom), pd.Series(pmom)
        rank_corr = w.rank().corr(p.rank())
        price_corr = np.log(pd.Series(wl)).corr(np.log(pd.Series(pl)))
        w5, p5 = set(w.nlargest(5).index), set(p.nlargest(5).index)
        w20, p20 = set(w.nlargest(20).index), set(p.nlargest(20).index)
        med = np.median(pxdiff) if pxdiff else float("nan")
        print(f"{str(td.date()):<12}{len(common):>5}{price_corr:>11.4f}{rank_corr:>13.4f}"
              f"{len(w5 & p5):>6}/5{len(w20 & p20):>6}/20{med:>9.2f}", flush=True)
        if len(w5 & p5) < 5:
            print(f"    WRDS top5:    {sorted(w5)}", flush=True)
            print(f"    Polygon top5: {sorted(p5)}", flush=True)


if __name__ == "__main__":
    main()
