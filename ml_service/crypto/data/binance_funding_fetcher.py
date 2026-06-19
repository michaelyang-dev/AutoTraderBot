"""
Through-cycle funding fetcher — Binance public data CDN (free, US-accessible).

Pulls monthly funding-rate history back to 2020 for the top USD-M perps → the
through-cycle dataset (2021 mania + 2022 LUNA/FTX blowups) that Hyperliquid (2023+)
can't provide. Daily funding = sum of the three 8h rates. Survivorship-aware by
construction: a coin simply has no data before its perp listed.

Run:  python crypto/data/binance_funding_fetcher.py
"""
import sys, os, io, time, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd
import numpy as np
from datetime import datetime

BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "ADA", "LINK", "DOT", "LTC", "MATIC"]


def months(start="2020-01"):
    s = pd.Period(start, "M")
    e = pd.Period(datetime.now().strftime("%Y-%m"), "M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]


def grab(sym, ym):
    url = f"{BASE}/{sym}/{sym}-fundingRate-{ym}.zip"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
        t = pd.to_numeric(df[0], errors="coerce")
        f = pd.to_numeric(df[df.columns[-1]], errors="coerce")
        out = pd.DataFrame({"t": t, "f": f}).dropna()              # drops the header row automatically
        out["date"] = pd.to_datetime(out["t"], unit="ms").dt.normalize()
        return out.groupby("date")["f"].sum()                      # daily funding
    except Exception:
        return None


def fetch():
    mlist = months()
    panel = {}
    for base in COINS:
        sym = base + "USDT"
        series = []
        for ym in mlist:
            s = grab(sym, ym)
            if s is not None and len(s):
                series.append(s)
            time.sleep(0.02)
        if series:
            panel[base] = pd.concat(series).sort_index()
            ann = panel[base].mean() * 365 * 100
            print("  %-6s %4d days %s→%s | mean ann funding %+.1f%%" % (
                base, len(panel[base]), panel[base].index.min().date(),
                panel[base].index.max().date(), ann), flush=True)
    df = pd.DataFrame(panel).sort_index()
    os.makedirs(DATA, exist_ok=True)
    df.to_parquet(os.path.join(DATA, "binance_funding.parquet"))
    print("saved %s → binance_funding.parquet" % str(df.shape))
    return df


if __name__ == "__main__":
    print("Fetching through-cycle funding (Binance CDN, 2020→now)...")
    fetch()
