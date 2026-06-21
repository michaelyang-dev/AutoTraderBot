"""
dYdX v4 funding fetcher — FREE public indexer. A second US-accessible DeFi perp venue (besides
Hyperliquid), so we can test cross-venue funding dispersion that a US person can ACTUALLY trade
(long dYdX / short HL, or vice versa — a clean perp-perp hedge) and diversify HL counterparty risk.

dYdX funding is hourly → aggregated to daily (sum of 24). v4 launched ~Nov 2023.
Run:  python crypto/data/dydx_funding_fetcher.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

SESSION = requests.Session()
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
URL = "https://indexer.dydx.trade/v4/historicalFunding"
COINS = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "NEAR", "SUI", "AAVE", "XRP", "ADA", "LINK", "WLD"]


def fetch_coin(base):
    ticker = f"{base}-USD"
    rows, before = [], None
    for _ in range(400):                                       # safety cap on pages
        p = {"limit": 100}
        if before:
            p["effectiveBeforeOrAt"] = before
        try:
            r = SESSION.get(f"{URL}/{ticker}", params=p, timeout=25)
            d = r.json().get("historicalFunding", [])
        except Exception:
            break
        if not d:
            break
        rows += d
        before = d[-1]["effectiveAt"]
        if pd.to_datetime(before) < pd.Timestamp("2023-10-01", tz="UTC"):
            break
    if not rows:
        return base, None
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["effectiveAt"]).dt.tz_localize(None)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["date"] = df["t"].dt.normalize()
    return base, df.groupby("date")["rate"].sum()              # hourly → daily funding


if __name__ == "__main__":
    print("Fetching dYdX v4 funding (US-accessible DeFi perp)...")
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for base, s in ex.map(fetch_coin, COINS):
            if s is not None and len(s):
                out[base] = s
                print("  %s: %d days %s→%s | ann %.1f%%" % (base, len(s), s.index.min().date(), s.index.max().date(), s.mean() * 365 * 100), flush=True)
    df = pd.DataFrame(out).sort_index()
    df.to_parquet(os.path.join(DATA, "dydx_funding.parquet"))
    print("saved %s → dydx_funding.parquet" % str(df.shape))
