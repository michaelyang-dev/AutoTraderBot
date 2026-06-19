"""
Hyperliquid funding + price history fetcher.

Pulls full per-coin hourly funding history (paginated) + daily prices for the most
liquid perps, the data foundation for the cross-sectional funding-carry backtest.
Hyperliquid = the actual US-accessible (self-custody) execution venue.

Daily funding = sum of the 24 hourly funding rates (what a delta-neutral short
collects that day). Annualized = daily * 365.

NOTE: only currently-listed coins are returned → survivorship caveat (delisted
high-funding coins are missing). The production version (Tardis) handles this.

Run:  python crypto/data/funding_fetcher.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd
import numpy as np

HL = "https://api.hyperliquid.xyz/info"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")


def post(body, retries=4):
    for _ in range(retries):
        try:
            r = requests.post(HL, json=body, timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(0.6)
    return None


def universe_by_oi(top):
    r = post({"type": "metaAndAssetCtxs"})
    meta, ctxs = r[0]["universe"], r[1]
    rows = []
    for u, c in zip(meta, ctxs):
        try:
            oi = float(c.get("openInterest", 0)) * float(c.get("markPx", 0))
            rows.append((u["name"], oi))
        except Exception:
            pass
    rows.sort(key=lambda x: -x[1])
    return [n for n, _ in rows[:top]]


def funding_history(coin, start_ms, end_ms):
    out, t = [], start_ms
    while t < end_ms:
        chunk = post({"type": "fundingHistory", "coin": coin, "startTime": t, "endTime": end_ms})
        if not chunk:
            break
        out += chunk
        if len(chunk) < 500:
            break
        t = chunk[-1]["time"] + 1
        time.sleep(0.04)
    return out


def daily_prices(coin, start_ms, end_ms):
    out, t = [], start_ms
    while t < end_ms:
        chunk = post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1d",
                                                         "startTime": t, "endTime": end_ms}})
        if not chunk:
            break
        out += chunk
        if len(chunk) < 500:
            break
        t = chunk[-1]["t"] + 1
        time.sleep(0.04)
    return out


def fetch(top=30, start="2023-06-01"):
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    coins = universe_by_oi(top)
    print("universe (top %d by OI): %s" % (top, ", ".join(coins)))
    fund, price = {}, {}
    for coin in coins:
        fh = funding_history(coin, start_ms, end_ms)
        if fh:
            f = pd.DataFrame(fh)
            f["date"] = pd.to_datetime(f["time"], unit="ms").dt.normalize()
            f["fundingRate"] = f["fundingRate"].astype(float)
            fund[coin] = f.groupby("date")["fundingRate"].sum()
        cd = daily_prices(coin, start_ms, end_ms)
        if cd:
            c = pd.DataFrame(cd)
            c["date"] = pd.to_datetime(c["t"], unit="ms").dt.normalize()
            price[coin] = c.drop_duplicates("date").set_index("date")["c"].astype(float)
        print("  %-8s funding=%4d days  price=%4d days" % (coin, len(fund.get(coin, [])), len(price.get(coin, []))), flush=True)
        time.sleep(0.04)
    fdf = pd.DataFrame(fund).sort_index()
    pdf = pd.DataFrame(price).sort_index()
    os.makedirs(DATA, exist_ok=True)
    fdf.to_parquet(os.path.join(DATA, "hl_funding.parquet"))
    pdf.to_parquet(os.path.join(DATA, "hl_price.parquet"))
    print("saved funding %s, price %s → %s" % (fdf.shape, pdf.shape, DATA))
    ann = fdf.mean() * 365 * 100
    print("per-coin mean annualized funding (top by |mean|):")
    print(ann.reindex(ann.abs().sort_values(ascending=False).index).head(12).round(1).to_string())
    return fdf, pdf


if __name__ == "__main__":
    fetch(top=30)
