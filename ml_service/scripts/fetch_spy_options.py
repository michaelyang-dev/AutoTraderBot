#!/usr/bin/env python3
"""
Fetch historical SPY options data from Polygon for VRP backtest.
Monthly expirations, ATM strangles, 2016-2026.
"""
import os, sys, time, json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from calendar import monthcalendar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

API_KEY = os.environ.get("MASSIVE_API_KEY")
BASE = "https://api.polygon.io"
SAVE_PATH = Path(__file__).resolve().parent.parent / "data" / "spy_options_vrp_history.parquet"

def api_get(endpoint, params=None, retries=3):
    params = params or {}
    params["apiKey"] = API_KEY
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE}{endpoint}", params=params, timeout=20)
            time.sleep(0.1)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:  # rate limit
                time.sleep(2)
                continue
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            continue
    return None

def third_friday(year, month):
    cal = monthcalendar(year, month)
    fridays = [week[4] for week in cal if week[4] != 0]
    return datetime(year, month, fridays[2])

def main():
    print("=" * 60)
    print("FETCHING SPY OPTIONS FOR VRP BACKTEST")
    print("=" * 60)

    # Get SPY price history
    print("Loading SPY prices...")
    spy_prices = {}
    for start_year in range(2015, 2027):
        data = api_get(f"/v2/aggs/ticker/SPY/range/1/day/{start_year}-01-01/{start_year}-12-31", {"limit": 400})
        if data and "results" in data:
            for bar in data["results"]:
                dt = datetime.fromtimestamp(bar["t"]/1000).strftime("%Y-%m-%d")
                spy_prices[dt] = bar["c"]
    print(f"  SPY prices: {len(spy_prices)} days")

    # Generate monthly expirations
    expirations = []
    for year in range(2016, 2027):
        for month in range(1, 13):
            try:
                exp = third_friday(year, month)
                if exp <= datetime(2026, 5, 12):
                    expirations.append(exp)
            except:
                pass
    print(f"  Expirations: {len(expirations)}")

    all_trades = []

    for idx, exp_date in enumerate(expirations):
        exp_str = exp_date.strftime("%Y-%m-%d")
        entry_date = exp_date - timedelta(days=30)

        # Find entry trading day
        entry_str = None; spy_entry = None
        for offset in range(7):
            check = (entry_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            if check in spy_prices:
                entry_str = check; spy_entry = spy_prices[check]; break
        if not entry_str: continue

        # Find expiry trading day
        exp_actual = None; spy_expiry = None
        for offset in range(-2, 5):
            check = (exp_date + timedelta(days=offset)).strftime("%Y-%m-%d")
            if check in spy_prices:
                exp_actual = check; spy_expiry = spy_prices[check]; break
        if not exp_actual: continue

        # Strikes: 5% OTM put, 5% OTM call
        put_strike = round((spy_entry * 0.95) / 1) * 1  # round to $1
        call_strike = round((spy_entry * 1.05) / 1) * 1

        # Polygon option ticker: O:SPY{YYMMDD}{C/P}{strike*1000:08d}
        exp_compact = exp_date.strftime("%y%m%d")
        put_ticker = f"O:SPY{exp_compact}P{int(put_strike * 1000):08d}"
        call_ticker = f"O:SPY{exp_compact}C{int(call_strike * 1000):08d}"

        # Fetch entry-day prices
        put_entry = api_get(f"/v2/aggs/ticker/{put_ticker}/range/1/day/{entry_str}/{entry_str}", {"limit": 1})
        call_entry = api_get(f"/v2/aggs/ticker/{call_ticker}/range/1/day/{entry_str}/{entry_str}", {"limit": 1})

        put_price = None; call_price = None
        if put_entry and put_entry.get("results"):
            put_price = put_entry["results"][0].get("c")
        if call_entry and call_entry.get("results"):
            call_price = call_entry["results"][0].get("c")

        if put_price is None or call_price is None:
            # Try nearby strikes
            for adj in [-2, -1, 1, 2, -3, 3]:
                if put_price is None:
                    alt_strike = put_strike + adj
                    alt_ticker = f"O:SPY{exp_compact}P{int(alt_strike * 1000):08d}"
                    alt_data = api_get(f"/v2/aggs/ticker/{alt_ticker}/range/1/day/{entry_str}/{entry_str}", {"limit": 1})
                    if alt_data and alt_data.get("results"):
                        put_price = alt_data["results"][0].get("c")
                        put_strike = alt_strike
                if call_price is None:
                    alt_strike = call_strike + adj
                    alt_ticker = f"O:SPY{exp_compact}C{int(alt_strike * 1000):08d}"
                    alt_data = api_get(f"/v2/aggs/ticker/{alt_ticker}/range/1/day/{entry_str}/{entry_str}", {"limit": 1})
                    if alt_data and alt_data.get("results"):
                        call_price = alt_data["results"][0].get("c")
                        call_strike = alt_strike
                if put_price and call_price:
                    break

        if put_price and call_price:
            premium = (put_price + call_price) * 100
            put_intrinsic = max(0, put_strike - spy_expiry) * 100
            call_intrinsic = max(0, spy_expiry - call_strike) * 100
            pnl = premium - put_intrinsic - call_intrinsic
            margin = max(put_strike, call_strike) * 100 * 0.20

            all_trades.append({
                "entry_date": entry_str,
                "expiry_date": exp_actual,
                "spy_entry": spy_entry,
                "spy_expiry": spy_expiry,
                "spy_move_pct": (spy_expiry - spy_entry) / spy_entry,
                "put_strike": put_strike,
                "call_strike": call_strike,
                "put_premium": put_price,
                "call_premium": call_price,
                "total_premium": put_price + call_price,
                "put_intrinsic": put_intrinsic / 100,
                "call_intrinsic": call_intrinsic / 100,
                "pnl_per_contract": pnl,
                "pnl_pct_margin": pnl / margin if margin > 0 else 0,
                "margin_per_contract": margin,
            })

        if (idx + 1) % 5 == 0:
            print(f"  {idx+1}/{len(expirations)}: {len(all_trades)} trades collected")
            # Save intermediate
            if all_trades:
                pd.DataFrame(all_trades).to_parquet(SAVE_PATH, index=False)

    # Final save
    df = pd.DataFrame(all_trades)
    df.to_parquet(SAVE_PATH, index=False)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(df)} monthly strangle trades")
    print(f"Date range: {df['entry_date'].min()} to {df['entry_date'].max()}")
    print(f"Avg premium: ${df['total_premium'].mean():.2f} per share")
    print(f"Avg P&L: ${df['pnl_per_contract'].mean():.0f} per contract")
    print(f"Avg P&L %: {df['pnl_pct_margin'].mean():.1%} on margin")
    print(f"Win rate: {(df['pnl_per_contract'] > 0).mean():.0%}")
    print(f"Saved to: {SAVE_PATH}")

    # Yearly summary
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    print(f"\nYearly:")
    for yr, grp in df.groupby("year"):
        avg_pnl = grp["pnl_pct_margin"].mean()
        wr = (grp["pnl_per_contract"] > 0).mean()
        n = len(grp)
        ann = avg_pnl * 12  # monthly to annual
        print(f"  {yr}: {n} trades, avg={avg_pnl:+.1%}/trade, WR={wr:.0%}, ann≈{ann:+.1%}")

if __name__ == "__main__":
    main()
