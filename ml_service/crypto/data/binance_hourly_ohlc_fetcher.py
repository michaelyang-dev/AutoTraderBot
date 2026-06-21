"""
Hourly OHLC for liquid majors — needed for the liquidation-WICK test (enter near the intra-hour
low a resting bid would catch, not the close). Saves open/high/low/close + qvol panels. Free CDN.

Run:  python crypto/data/binance_hourly_ohlc_fetcher.py
"""
import sys, os, io, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

SESSION = requests.Session()
KLINE = "https://data.binance.vision/data/futures/um/monthly/klines"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
SYMS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC",
        "DOT", "BCH", "TRX", "MATIC", "ATOM", "UNI", "ETC", "XLM", "FIL", "NEAR",
        "APT", "ARB", "OP", "INJ", "TIA", "SUI", "SEI", "AAVE", "RUNE", "FTM",
        "PEPE", "WLD", "ORDI", "STX", "FET", "GRT", "SAND", "AXS", "ICP", "ALGO"]


def months(start="2020-01"):
    s = pd.Period(start, "M"); e = pd.Period(datetime.now().strftime("%Y-%m"), "M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]


def grab(task):
    sym, ym = task
    url = f"{KLINE}/{sym}/1h/{sym}-1h-{ym}.zip"
    try:
        r = SESSION.get(url, timeout=25)
        if r.status_code != 200:
            return sym, None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
        out = pd.DataFrame({
            "t": pd.to_numeric(df[0], errors="coerce"),
            "o": pd.to_numeric(df[1], errors="coerce"), "h": pd.to_numeric(df[2], errors="coerce"),
            "l": pd.to_numeric(df[3], errors="coerce"), "c": pd.to_numeric(df[4], errors="coerce"),
            "qv": pd.to_numeric(df[7], errors="coerce")}).dropna()
        out["ts"] = pd.to_datetime(out["t"], unit="ms")
        return sym, out.set_index("ts")[["o", "h", "l", "c", "qv"]]
    except Exception:
        return sym, None


def fetch():
    tasks = [(s + "USDT", ym) for s in SYMS for ym in months()]
    print("pulling %d symbol-months (hourly OHLC) ..." % len(tasks), flush=True)
    parts = defaultdict(list); done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for sym, fr in ex.map(grab, tasks):
            done += 1
            if fr is not None and len(fr):
                parts[sym].append(fr)
            if done % 1000 == 0:
                print("  %d/%d ..." % (done, len(tasks)), flush=True)
    panels = {f: {} for f in ["o", "h", "l", "c", "qv"]}
    for sym, frames in parts.items():
        base = sym[:-4]
        df = pd.concat(frames).sort_index(); df = df[~df.index.duplicated()]
        for f in panels:
            panels[f][base] = df[f]
    for f in panels:
        pd.DataFrame(panels[f]).sort_index().to_parquet(os.path.join(DATA, f"binance_1h_{f}.parquet"))
    print("saved binance_1h_{o,h,l,c,qv}.parquet")


if __name__ == "__main__":
    print("Fetching hourly OHLC majors...")
    fetch()
