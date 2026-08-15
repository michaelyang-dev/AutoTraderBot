"""Build a TIME-VARYING broker financing curve. Cached to research/_fin_rate.parquet.

WHY THIS EXISTS
  Every backtest in this repo charges a FLAT 6.3%/yr on the margin debit across the entire
  2001-2025 period. 6.3% is roughly right TODAY (IBKR Pro on a small balance is benchmark
  + ~1.5%, and fed funds is ~4.3%). It is badly wrong historically: fed funds was 0-0.25%
  for most of 2009-2015 and again through 2020-21.

  That biases every leverage conclusion AGAINST leverage on the 26yr horizon, because the
  strategy is charged ~5pp/yr too much on its debit for roughly half the sample. EXP-005
  measured leverage costing -0.044 Sharpe over 26 years using the flat rate; that number
  cannot be trusted until the rate is right.

CONSTRUCTION
  financing_t = DFF_t (daily effective fed funds, FRED) + spread, default 1.50pp, which is
  IBKR Pro's published tier for balances under $100k. Forward-filled across non-business days.
  DFF is a published daily rate, known same-day, so there is no look-ahead in charging
  financing at date t using DFF at date t.

Run:  python3 research/build_financing_curve.py
"""
import os
import sys
import io

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import urllib.request  # noqa: E402

OUT = "research/_fin_rate.parquet"
SPREAD = 0.015


def main():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
    print(f"fetching {url} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read().decode()
        df = pd.read_csv(io.StringIO(raw))
        dcol = df.columns[0]
        vcol = [c for c in df.columns if c.lower() != dcol.lower()][0]
        df[dcol] = pd.to_datetime(df[dcol])
        s = pd.to_numeric(df[vcol], errors="coerce")
        s.index = df[dcol]
        s = s.dropna() / 100.0
        src = "FRED live"
    except Exception as e:
        print(f"  FRED fetch failed ({e}); falling back to the local WRDS mirror", flush=True)
        w = pd.read_parquet("data/wrds/fred_interest_rates_spreads_daily.parquet",
                            columns=["date", "dff"])
        w["date"] = pd.to_datetime(w["date"])
        s = pd.to_numeric(w["dff"], errors="coerce")
        s.index = w["date"]
        s = s.dropna() / 100.0
        src = "WRDS mirror (ends 2025-02)"

    fin = (s + SPREAD).sort_index()
    fin = fin[~fin.index.duplicated(keep="last")]
    fin.name = "fin_rate"
    fin.to_frame().to_parquet(OUT)

    print(f"  source: {src}", flush=True)
    print(f"  {len(fin):,} rows  {fin.index[0].date()} -> {fin.index[-1].date()}", flush=True)
    for lo, hi in [("2001-01-01", "2025-12-31"), ("2001-01-01", "2007-12-31"),
                   ("2009-01-01", "2015-12-31"), ("2018-01-01", "2025-12-31"),
                   ("2020-03-01", "2022-02-28"), ("2024-01-01", "2025-12-31")]:
        w = fin.loc[lo:hi]
        if len(w):
            print(f"  {lo[:7]}..{hi[:7]}  mean {w.mean():.2%}  min {w.min():.2%}  "
                  f"max {w.max():.2%}", flush=True)
    print(f"\n  FLAT ASSUMPTION IN USE EVERYWHERE: 6.30%", flush=True)
    print(f"  actual mean 2001-2025: {fin.loc['2001-01-01':'2025-12-31'].mean():.2%}  "
          f"-> the flat rate overcharges by "
          f"{(0.063 - fin.loc['2001-01-01':'2025-12-31'].mean())*100:.2f}pp/yr on the debit",
          flush=True)
    print(f"  actual mean 2018-2025: {fin.loc['2018-01-01':'2025-12-31'].mean():.2%}",
          flush=True)
    print(f"\n  wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
