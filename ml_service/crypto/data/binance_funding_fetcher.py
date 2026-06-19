"""
Through-cycle funding fetcher — Binance public data CDN (free, US-accessible).

Pulls monthly funding-rate history back to 2020 for the top USD-M perps → the
through-cycle dataset (2021 mania + 2022 LUNA/FTX blowups) that Hyperliquid (2023+)
can't provide. Daily funding = sum of the three 8h rates. Survivorship-aware by
construction: a coin simply has no data before its perp listed.

Run:  python crypto/data/binance_funding_fetcher.py
"""
import sys, os, io, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

SESSION = requests.Session()

BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
COINS = [  # broad liquid Binance USDT-perp set spanning the cycle (point-in-time: no data pre-listing)
    "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "DOT", "LTC", "LINK", "MATIC", "AVAX",
    "UNI", "ATOM", "ETC", "BCH", "XLM", "TRX", "FIL", "AAVE", "ALGO", "ICP", "VET", "FTM",
    "SAND", "MANA", "AXS", "THETA", "EGLD", "NEAR", "GRT", "ENJ", "CHZ", "ZEC", "DASH",
    "COMP", "SUSHI", "YFI", "SNX", "CRV", "RUNE", "HBAR", "IOTA", "WAVES", "ZIL", "ONE",
    "APT", "ARB", "OP", "SUI", "SEI", "TIA", "INJ", "WLD", "ORDI", "STX", "FET", "PEPE",
]


def months(start="2020-01"):
    s = pd.Period(start, "M")
    e = pd.Period(datetime.now().strftime("%Y-%m"), "M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]


def grab(task):
    sym, ym = task
    url = f"{BASE}/{sym}/{sym}-fundingRate-{ym}.zip"
    try:
        r = SESSION.get(url, timeout=20)
        if r.status_code != 200:
            return sym, None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
        t = pd.to_numeric(df[0], errors="coerce")
        f = pd.to_numeric(df[df.columns[-1]], errors="coerce")
        out = pd.DataFrame({"t": t, "f": f}).dropna()              # drops the header row automatically
        out["date"] = pd.to_datetime(out["t"], unit="ms").dt.normalize()
        return sym, out.groupby("date")["f"].sum()                 # daily funding
    except Exception:
        return sym, None


def fetch():
    mlist = months()
    tasks = [(base + "USDT", ym) for base in COINS for ym in mlist]
    parts = defaultdict(list)
    with ThreadPoolExecutor(max_workers=14) as ex:                 # parallel downloads (CDN handles it)
        for sym, s in ex.map(grab, tasks):
            if s is not None and len(s):
                parts[sym].append(s)
    panel = {}
    for base in COINS:
        sym = base + "USDT"
        if parts[sym]:
            panel[base] = pd.concat(parts[sym]).sort_index()
            panel[base] = panel[base][~panel[base].index.duplicated()]
    print("fetched %d coins" % len(panel))
    df = pd.DataFrame(panel).sort_index()
    os.makedirs(DATA, exist_ok=True)
    df.to_parquet(os.path.join(DATA, "binance_funding.parquet"))
    print("saved %s → binance_funding.parquet" % str(df.shape))
    return df


if __name__ == "__main__":
    print("Fetching through-cycle funding (Binance CDN, 2020→now)...")
    fetch()
