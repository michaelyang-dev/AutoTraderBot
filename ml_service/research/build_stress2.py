"""Build _macro_stress2.parquet — extra stress signals from data we already own but never
used for leverage.

The deployed gate uses ONE signal (hy_oas). Sitting unused in the universe pickle:
  fama_french  mktrf/smb/hml/rmw/cma/rf/umd   1963-07 -> 2026-02
  fred_rates   82 columns                     1954-01 -> 2025-02

Signals built here (all oriented so HIGH = STRESS, because _stress_pctile_map gates on
`percentile >= gate_pct`):

  baa_aaa    dbaa - daaa. Credit QUALITY spread, independent of hy_oas: it measures
             investment-grade risk aversion rather than junk. History from 1986.
  term_inv   -(dgs10 - dgs3mo). Negated so curve INVERSION reads as stress.
  umd_crash  -(20-day sum of the UMD factor). The strategy is 50% momentum, and UMD already
             drives a WEIGHT switch (mom .50 -> .167) but has NEVER been allowed to touch
             LEVERAGE. When the momentum factor itself is crashing, the book is at its most
             fragile and is still fully levered. This is the most promising of the four.
  mkt_vol    20d realized vol of mktrf. A MARKET-wide vol estimate, independent of the book's
             own 40d shadow vol -- it moves before the book's own P&L reflects the regime.

Point-in-time honesty: these are raw daily series. _stress_pctile_map applies an expanding
percentile with .shift(1), so only past data is ever used at any date.

NOTE fred_rates ends 2025-02 and fama_french 2026-02, so baa_aaa/term_inv are unusable for a
LIVE gate as-is. That is a research limitation, not a fatal one -- both are FRED series the
existing credit-gate cron could pull. umd_crash needs no external feed at all: it is already
computed live from prices (compute_price_umd_20d).

Run:  python3 research/build_stress2.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_macro_stress2.parquet")


def main():
    src = "data/wrds/sp1500_universe_2000.pkl"
    print(f"loading {src} ...", flush=True)
    u = pickle.load(open(src, "rb"))
    fr, ff = u["fred_rates"], u["fama_french"]
    fr.index = pd.to_datetime(fr.index)
    ff.index = pd.to_datetime(ff.index)

    cols = {}

    if {"dbaa", "daaa"} <= set(fr.columns):
        s = (fr["dbaa"] - fr["daaa"]).dropna()
        cols["baa_aaa"] = s
        print(f"  baa_aaa   n={len(s):>6} {s.index.min().date()} -> {s.index.max().date()}")

    long_c = "dgs10" if "dgs10" in fr.columns else None
    short_c = "dgs3mo" if "dgs3mo" in fr.columns else ("dgs2" if "dgs2" in fr.columns else None)
    if long_c and short_c:
        s = -(fr[long_c] - fr[short_c]).dropna()      # negate: inversion = stress
        cols["term_inv"] = s
        print(f"  term_inv  n={len(s):>6} {s.index.min().date()} -> {s.index.max().date()} "
              f"(-({long_c}-{short_c}))")

    if "umd" in ff.columns:
        s = -(ff["umd"].dropna().rolling(20).sum()).dropna()   # negate: UMD crash = stress
        cols["umd_crash"] = s
        print(f"  umd_crash n={len(s):>6} {s.index.min().date()} -> {s.index.max().date()}")

    if "mktrf" in ff.columns:
        s = (ff["mktrf"].dropna().rolling(20).std() * np.sqrt(252)).dropna()
        cols["mkt_vol"] = s
        print(f"  mkt_vol   n={len(s):>6} {s.index.min().date()} -> {s.index.max().date()}")

    if not cols:
        raise SystemExit("no signals built — check the source columns")

    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df.to_parquet(OUT)
    print(f"\nwrote {OUT}  shape={df.shape}")
    print(df.tail(3))
    print("\ncoverage since 2001:")
    print((df.loc["2001-01-01":].notna().mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
