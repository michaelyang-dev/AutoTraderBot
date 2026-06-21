"""
Coin Metrics Community (FREE) on-chain fetcher — for the on-chain-flow pilot.

Free-tier metrics that are genuinely DIFFERENT from price (real network/economic activity):
  FlowInExUSD / FlowOutExUSD  exchange in/outflows (classic accumulate/distribute signal)
  AdrActCnt                   active addresses (adoption/usage)
  TxCnt                       transaction count
  CapMrktCurUSD               market cap (for valuation ratios)
  SplyCur                     circulating supply
  HashRate                    miner commitment (BTC)
History back to 2016+. Run:  python crypto/data/coinmetrics_fetcher.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ["PriceUSD", "CapMrktCurUSD", "AdrActCnt", "TxCnt", "SplyCur", "HashRate",
           "FlowInExUSD", "FlowOutExUSD"]


def fetch(asset):
    rows, token = [], None
    params = {"assets": asset, "metrics": ",".join(METRICS), "frequency": "1d",
              "start_time": "2017-01-01", "page_size": 10000}
    while True:
        if token:
            params["next_page_token"] = token
        r = requests.get(URL, params=params, timeout=40)
        j = r.json()
        rows += j.get("data", [])
        token = j.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    for m in METRICS:
        if m in df:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df.set_index("date")[[m for m in METRICS if m in df]].sort_index()


if __name__ == "__main__":
    print("Fetching Coin Metrics on-chain (BTC, ETH)...")
    panels = {}
    for a in ["btc", "eth"]:
        df = fetch(a)
        panels[a.upper()] = df
        print("  %s: %d days %s→%s | metrics: %s" % (a.upper(), len(df), df.index.min().date(), df.index.max().date(), list(df.columns)))
    # save as a wide multi-asset parquet (columns: ASSET_metric)
    wide = pd.concat({k: v for k, v in panels.items()}, axis=1)
    wide.columns = ["%s_%s" % (a, m) for a, m in wide.columns]
    wide.to_parquet(os.path.join(DATA, "coinmetrics_onchain.parquet"))
    print("saved %s → coinmetrics_onchain.parquet" % str(wide.shape))
