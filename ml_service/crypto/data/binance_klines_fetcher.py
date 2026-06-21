"""
SURVIVORSHIP-COMPLETE price/volume fetcher — Binance public data CDN (free, US-accessible).

The honest fix for the survivorship-sensitivity the robustness sweep flagged: instead of
CoinMarketCap (whose dead-coin tail crypto2 can't fetch), pull the price history of EVERY
USD-M perp Binance ever listed — INCLUDING DELISTED ONES. The CDN keeps a delisted symbol's
klines forever, so this is gold-standard survivorship-clean *for the exact universe we can
trade*. Symbols are discovered from the S3 bucket listing (includes dead coins), not from
the live exchangeInfo (which would be survivorship-biased to currently-listed only).

Ranking will use trailing dollar-VOLUME (capacity-aware + execution-realistic), not mcap.

Run:  python crypto/data/binance_klines_fetcher.py
"""
import sys, os, io, zipfile, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

SESSION = requests.Session()
KLINE = "https://data.binance.vision/data/futures/um/monthly/klines"
LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
PREFIX = "data/futures/um/monthly/klines/"


def all_symbols():
    """List every USD-M perp symbol dir in the CDN bucket (incl. delisted) via paginated S3 XML."""
    syms, marker = [], ""
    while True:
        url = f"{LIST}?delimiter=/&prefix={PREFIX}"
        if marker:
            url += f"&marker={marker}"
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        prefixes = re.findall(r"<Prefix>" + re.escape(PREFIX) + r"([^/<]+)/</Prefix>", r.text)
        syms.extend(prefixes)
        if "<IsTruncated>true</IsTruncated>" in r.text:
            marker = PREFIX + prefixes[-1] + "/"          # next page starts after last seen
        else:
            break
    usdt = sorted({s for s in syms if s.endswith("USDT")})  # USDT-margined perps only
    print("discovered %d USDT perp symbols (incl. delisted)" % len(usdt))
    return usdt


def months(start="2020-01"):
    s = pd.Period(start, "M")
    e = pd.Period(datetime.now().strftime("%Y-%m"), "M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]


def grab(task):
    sym, ym = task
    url = f"{KLINE}/{sym}/1d/{sym}-1d-{ym}.zip"
    try:
        r = SESSION.get(url, timeout=25)
        if r.status_code != 200:
            return sym, None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]), header=None)
        # futures kline cols: 0 open_time,4 close,7 quote_volume
        t = pd.to_numeric(df[0], errors="coerce")
        close = pd.to_numeric(df[4], errors="coerce")
        qv = pd.to_numeric(df[7], errors="coerce")
        out = pd.DataFrame({"t": t, "close": close, "qv": qv}).dropna()
        out["date"] = pd.to_datetime(out["t"], unit="ms").dt.normalize()
        return sym, out.set_index("date")[["close", "qv"]]
    except Exception:
        return sym, None


def fetch():
    syms = all_symbols()
    mlist = months()
    tasks = [(s, ym) for s in syms for ym in mlist]
    print("pulling %d symbol-months ..." % len(tasks), flush=True)
    parts = defaultdict(list)
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for sym, frame in ex.map(grab, tasks):
            done += 1
            if frame is not None and len(frame):
                parts[sym].append(frame)
            if done % 2000 == 0:
                print("  %d/%d ..." % (done, len(tasks)), flush=True)
    closep, qvp = {}, {}
    for sym, frames in parts.items():
        base = sym[:-4]                                   # strip USDT
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated()]
        if df["close"].notna().sum() > 60:                # need a usable history
            closep[base] = df["close"]
            qvp[base] = df["qv"]
    close = pd.DataFrame(closep).sort_index()
    qv = pd.DataFrame(qvp).sort_index()
    os.makedirs(DATA, exist_ok=True)
    close.to_parquet(os.path.join(DATA, "binance_close.parquet"))
    qv.to_parquet(os.path.join(DATA, "binance_qvol.parquet"))
    print("saved close %s + qvol %s → binance_close/qvol.parquet (%s→%s)"
          % (close.shape, qv.shape, close.index.min().date(), close.index.max().date()))
    return close, qv


if __name__ == "__main__":
    print("Fetching survivorship-complete perp klines (Binance CDN, 2020→now, incl. delisted)...")
    fetch()
