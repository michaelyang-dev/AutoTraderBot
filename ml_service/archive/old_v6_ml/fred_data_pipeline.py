"""
FRED Macro Data Pipeline
========================
Fetches key macro economic series from the FRED API and caches
them to data/macro_fred.parquet for use in feature engineering.

Series fetched:
  DGS10           - 10-year Treasury yield
  DGS2            - 2-year Treasury yield
  T10Y2Y          - 10Y-2Y yield spread (pre-calculated)
  BAMLH0A0HYM2    - ICE BofA High Yield spread
  DTWEXBGS        - Trade-weighted Dollar Index
  VIXCLS          - CBOE VIX close

Run with:
    python3 fred_data_pipeline.py
"""

import os
import json
import ssl
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "macro_fred.parquet"

SERIES = [
    "DGS10",           # 10-year Treasury yield
    "DGS2",            # 2-year Treasury yield
    "T10Y2Y",          # Yield curve (10Y - 2Y)
    "BAMLH0A0HYM2",   # High yield spread
    "DTWEXBGS",        # Dollar index (broad)
    "VIXCLS",          # VIX
]

# 15 years + buffer
START_DATE = (datetime.today() - timedelta(days=365 * 15 + 400)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a single FRED series via the API."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&observation_start={start}"
        f"&observation_end={end}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        data = json.loads(resp.read().decode())

    rows = []
    for obs in data.get("observations", []):
        val = obs.get("value", ".")
        if val == "." or val == "":
            continue
        try:
            rows.append({"date": pd.Timestamp(obs["date"]), "value": float(val)})
        except (ValueError, KeyError):
            continue

    if not rows:
        return pd.Series(dtype=float, name=series_id)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
    return df.set_index("date")["value"].rename(series_id)


def main():
    if not FRED_API_KEY:
        print("ERROR: FRED_API_KEY not set in .env")
        return

    print("=" * 60)
    print("FRED Macro Data Pipeline")
    print(f"Date range: {START_DATE} -> {END_DATE}")
    print(f"Series: {', '.join(SERIES)}")
    print("=" * 60)

    all_series = {}
    for sid in SERIES:
        print(f"  Fetching {sid} ...", end=" ", flush=True)
        try:
            s = fetch_fred_series(sid, START_DATE, END_DATE)
            all_series[sid] = s
            print(f"{len(s):,} observations ({s.index.min().date()} -> {s.index.max().date()})")
        except Exception as e:
            print(f"FAILED: {e}")
            all_series[sid] = pd.Series(dtype=float, name=sid)

    # Combine into a single DataFrame
    df = pd.DataFrame(all_series)
    df.index.name = "date"

    if df.empty or df.dropna(how="all").empty:
        print("\nERROR: No data fetched from FRED.")
        return

    # Forward-fill missing dates (weekends, holidays)
    # First, create a business-day index spanning the range
    valid = df.dropna(how="all")
    full_idx = pd.bdate_range(start=valid.index.min(), end=valid.index.max())
    df = df.reindex(full_idx)
    df.index.name = "date"
    df = df.ffill()

    # Drop rows where all values are NaN (before first observation)
    df = df.dropna(how="all")

    print(f"\nCombined: {len(df):,} rows x {len(df.columns)} columns")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Missing values per column:")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        print(f"  {col:<20} {n_miss:>5} NaN")

    df.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy")
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
