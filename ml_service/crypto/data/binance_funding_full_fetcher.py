"""
SURVIVORSHIP-COMPLETE Binance funding — ALL USD-M perps ever listed (incl. delisted), 2020→now.

Solves the carry's two data limitations at once:
  - LONGER history: 2020-2026 (6+ yrs) through LUNA (2022-05) + FTX (2022-11), vs Hyperliquid's 3yr.
  - COMPLETE survivorship: symbols discovered from the CDN bucket listing (delisted coins retained),
    so the dead/squeezed coins that would hurt a short-perp book ARE in the data.

Binance is the VALIDATION venue (not US-tradable); HL stays execution. The funding-carry mechanism
is venue-agnostic, so this is the honest long/complete-data test of the EDGE.

Run:  python crypto/data/binance_funding_full_fetcher.py
"""
import sys, os, io, zipfile, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

SESSION = requests.Session()
BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PREFIX = "data/futures/um/monthly/fundingRate/"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")


def all_symbols():
    syms, marker = [], ""
    while True:
        url = f"{LIST}?delimiter=/&prefix={PREFIX}" + (f"&marker={marker}" if marker else "")
        r = SESSION.get(url, timeout=30); r.raise_for_status()
        got = re.findall(r"<Prefix>" + re.escape(PREFIX) + r"([^/<]+)/</Prefix>", r.text)
        syms += got
        if "<IsTruncated>true</IsTruncated>" in r.text:
            marker = PREFIX + got[-1] + "/"
        else:
            break
    usdt = sorted({s for s in syms if s.endswith("USDT")})
    print("discovered %d USDT perp symbols with funding (incl. delisted)" % len(usdt))
    return usdt


def months(start="2020-01"):
    s = pd.Period(start, "M"); e = pd.Period(datetime.now().strftime("%Y-%m"), "M")
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
        out = pd.DataFrame({"t": t, "f": f}).dropna()
        out["date"] = pd.to_datetime(out["t"], unit="ms").dt.normalize()
        return sym, out.groupby("date")["f"].sum()
    except Exception:
        return sym, None


def fetch():
    syms = all_symbols()
    tasks = [(s, ym) for s in syms for ym in months()]
    print("pulling %d symbol-months ..." % len(tasks), flush=True)
    parts = defaultdict(list); done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for sym, s in ex.map(grab, tasks):
            done += 1
            if s is not None and len(s):
                parts[sym].append(s)
            if done % 5000 == 0:
                print("  %d/%d ..." % (done, len(tasks)), flush=True)
    panel = {}
    for sym, ser in parts.items():
        base = sym[:-4]
        x = pd.concat(ser).sort_index()
        x = x[~x.index.duplicated()]
        if x.notna().sum() > 30:
            panel[base] = x
    df = pd.DataFrame(panel).sort_index()
    df.to_parquet(os.path.join(DATA, "binance_funding_full.parquet"))
    print("saved %s → binance_funding_full.parquet (%s→%s)" % (df.shape, df.index.min().date(), df.index.max().date()))


if __name__ == "__main__":
    print("Fetching survivorship-complete Binance funding (all perps incl. delisted, 2020→now)...")
    fetch()
