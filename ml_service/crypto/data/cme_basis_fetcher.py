"""
CME basis fetcher — the data foundation for the cash-and-carry backtest.

Pulls CME front-month futures (BTC=F, ETH=F) + spot, computes the true basis and
its annualized form (using days-to-expiry from the CME calendar: contracts expire
the LAST FRIDAY of the contract month). Saves a clean daily panel to
data/crypto/cme_basis.parquet.

Sources (all free / already keyed): yfinance for CME continuous front-month +
spot. For LIVE execution the actual CME contracts trade in the IBKR account; this
historical series is for the backtest.

Run:  python crypto/data/cme_basis_fetcher.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "data", "crypto", "cme_basis.parquet")

PAIRS = {"BTC": ("BTC=F", "BTC-USD"), "ETH": ("ETH=F", "ETH-USD")}


def last_friday(year, month):
    """CME crypto futures expire the last Friday of the contract month."""
    import calendar
    cal = calendar.monthcalendar(year, month)
    fridays = [w[calendar.FRIDAY] for w in cal if w[calendar.FRIDAY] != 0]
    return pd.Timestamp(year, month, fridays[-1])


def front_month_dte(dates):
    """Days to the front-month expiry for each date (>=1)."""
    out = []
    for d in dates:
        exp = last_friday(d.year, d.month)
        if d > exp:  # rolled to next month
            ny, nm = (d.year + (d.month == 12), d.month % 12 + 1)
            exp = last_friday(ny, nm)
        out.append(max((exp - d).days, 1))
    return pd.Series(out, index=dates), pd.Series(
        [last_friday(d.year, d.month) if d <= last_friday(d.year, d.month)
         else last_friday(d.year + (d.month == 12), d.month % 12 + 1) for d in dates], index=dates)


def fetch():
    import yfinance as yf
    frames = {}
    for asset, (fut_t, spot_t) in PAIRS.items():
        fut = yf.download(fut_t, start="2018-01-01", progress=False)["Close"].squeeze()
        spot = yf.download(spot_t, start="2018-01-01", progress=False)["Close"].squeeze()
        df = pd.concat([fut, spot], axis=1, keys=["fut", "spot"]).dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        dte, expiry = front_month_dte(df.index)
        df["dte"] = dte
        df["expiry"] = expiry
        df["roll"] = df["expiry"].ne(df["expiry"].shift(1))   # True on the day the front-month rolls
        df["basis"] = df["fut"] / df["spot"] - 1               # raw front-month basis
        df["basis_ann"] = df["basis"] * (365.0 / df["dte"])    # annualized
        frames[asset] = df
        print(f"  {asset}: {len(df)} days {df.index.min().date()}→{df.index.max().date()} | "
              f"ann basis mean {df['basis_ann'].mean()*100:5.1f}% | "
              f"2021 {df['basis_ann'].loc['2021'].mean()*100:5.1f}% | "
              f"2024+ {df['basis_ann'].loc['2024':].mean()*100:5.1f}%")
    panel = pd.concat(frames, axis=1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    panel.to_parquet(OUT)
    print(f"saved → {OUT}")
    return panel


if __name__ == "__main__":
    print("Fetching CME basis (BTC, ETH)...")
    fetch()
