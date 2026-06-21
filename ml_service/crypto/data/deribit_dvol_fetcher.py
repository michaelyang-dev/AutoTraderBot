"""
Deribit DVOL (implied-vol index) fetcher — FREE, for the vol-risk-premium pilot.

DVOL is Deribit's 30-day implied volatility index (crypto VIX) for BTC and ETH. Comparing it to
subsequently REALIZED vol tests whether options are systematically overpriced (the structural VRP
edge: people overpay for crypto optionality → selling vol harvests IV − RV). Free public API.

Run:  python crypto/data/deribit_dvol_fetcher.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def fetch_dvol(currency):
    end = int(time.time() * 1000)
    start = int(pd.Timestamp("2021-01-01").timestamp() * 1000)
    rows = []
    cur = start
    while cur < end:
        chunk_end = min(cur + 300 * 86400 * 1000, end)
        r = requests.get(URL, params={"currency": currency, "start_timestamp": cur,
                                      "end_timestamp": chunk_end, "resolution": "1D"}, timeout=25)
        data = r.json().get("result", {}).get("data", [])
        if not data:
            cur = chunk_end + 86400 * 1000
            continue
        rows += data
        cur = data[-1][0] + 86400 * 1000
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).drop_duplicates("ts")
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    return df.set_index("date")["close"].sort_index() / 100.0      # DVOL in vol points → fraction


if __name__ == "__main__":
    print("Fetching Deribit DVOL (BTC, ETH)...")
    out = {}
    for ccy in ["BTC", "ETH"]:
        s = fetch_dvol(ccy)
        out[ccy] = s
        print("  %s: %d days %s→%s | latest IV %.1f%%" % (ccy, len(s), s.index.min().date(), s.index.max().date(), s.iloc[-1] * 100))
    df = pd.DataFrame(out)
    df.to_parquet(os.path.join(DATA, "deribit_dvol.parquet"))
    print("saved %s → deribit_dvol.parquet" % str(df.shape))
