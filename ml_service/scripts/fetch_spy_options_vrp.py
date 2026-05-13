#!/usr/bin/env python3
"""
Fetch historical SPY options data from Polygon for VRP backtest.
Monthly expirations, ~5% OTM strangles, 2016-2026.
"""
import os, sys, time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from calendar import monthcalendar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT.parent / ".env")

API_KEY = os.environ.get("MASSIVE_API_KEY")
BASE = "https://api.polygon.io"
SAVE_PATH = ROOT / "data" / "spy_options_vrp_history.parquet"


def api_get(endpoint, params=None, retries=3):
    params = params or {}
    params["apiKey"] = API_KEY
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE}{endpoint}", params=params, timeout=30)
            time.sleep(0.1)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                time.sleep(3)
                continue
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
    return None


def third_friday(year, month):
    cal = monthcalendar(year, month)
    fridays = [week[4] for week in cal if week[4] != 0]
    return datetime(year, month, fridays[2])


def main():
    print("=" * 60)
    print("FETCHING SPY OPTIONS FOR VRP BACKTEST")
    print("=" * 60)

    # ── SPY price history ──
    print("Loading SPY prices...")
    spy_prices = {}
    for yr in range(2015, 2027):
        data = api_get(f"/v2/aggs/ticker/SPY/range/1/day/{yr}-01-01/{yr}-12-31",
                       {"limit": 400, "adjusted": "true"})
        if data and "results" in data:
            for bar in data["results"]:
                dt = datetime.fromtimestamp(bar["t"] / 1000).strftime("%Y-%m-%d")
                spy_prices[dt] = bar["c"]
    print(f"  SPY prices: {len(spy_prices)} days")

    # ── Monthly expirations 2016-2026 ──
    expirations = []
    for year in range(2016, 2027):
        for month in range(1, 13):
            try:
                exp = third_friday(year, month)
                if exp <= datetime(2026, 5, 12):
                    expirations.append(exp)
            except Exception:
                pass
    print(f"  Expirations to process: {len(expirations)}")

    # ── Fetch option prices for each cycle ──
    all_trades = []
    misses = 0

    for idx, exp_date in enumerate(expirations):
        entry_target = exp_date - timedelta(days=30)

        # Find entry trading day
        entry_str = spy_entry = None
        for off in range(7):
            d = (entry_target + timedelta(days=off)).strftime("%Y-%m-%d")
            if d in spy_prices:
                entry_str, spy_entry = d, spy_prices[d]
                break
        if not entry_str:
            misses += 1
            continue

        # Find expiry trading day
        exp_str = spy_expiry = None
        for off in range(-2, 5):
            d = (exp_date + timedelta(days=off)).strftime("%Y-%m-%d")
            if d in spy_prices:
                exp_str, spy_expiry = d, spy_prices[d]
                break
        if not exp_str:
            misses += 1
            continue

        # Strikes: 5% OTM strangle
        put_strike = round(spy_entry * 0.95)
        call_strike = round(spy_entry * 1.05)

        # Polygon option ticker: O:SPY{YYMMDD}{C/P}{strike*1000:08d}
        ec = exp_date.strftime("%y%m%d")
        put_tk = f"O:SPY{ec}P{int(put_strike * 1000):08d}"
        call_tk = f"O:SPY{ec}C{int(call_strike * 1000):08d}"

        # Fetch entry-day close prices
        pd_entry = api_get(f"/v2/aggs/ticker/{put_tk}/range/1/day/{entry_str}/{entry_str}")
        cd_entry = api_get(f"/v2/aggs/ticker/{call_tk}/range/1/day/{entry_str}/{entry_str}")

        pp = pd_entry["results"][0]["c"] if pd_entry and pd_entry.get("results") else None
        cp = cd_entry["results"][0]["c"] if cd_entry and cd_entry.get("results") else None

        if pp is None or cp is None:
            # Try ATM instead of 5% OTM
            atm = round(spy_entry)
            put_tk2 = f"O:SPY{ec}P{int(atm * 1000):08d}"
            call_tk2 = f"O:SPY{ec}C{int(atm * 1000):08d}"
            if pp is None:
                pd2 = api_get(f"/v2/aggs/ticker/{put_tk2}/range/1/day/{entry_str}/{entry_str}")
                if pd2 and pd2.get("results"):
                    pp = pd2["results"][0]["c"]
                    put_strike = atm
                    put_tk = put_tk2
            if cp is None:
                cd2 = api_get(f"/v2/aggs/ticker/{call_tk2}/range/1/day/{entry_str}/{entry_str}")
                if cd2 and cd2.get("results"):
                    cp = cd2["results"][0]["c"]
                    call_strike = atm
                    call_tk = call_tk2

        if pp is None or cp is None:
            misses += 1
            continue

        # Compute P&L at expiry (intrinsic only — options expire)
        put_intrinsic = max(0, put_strike - spy_expiry)
        call_intrinsic = max(0, spy_expiry - call_strike)
        premium = pp + cp
        pnl = premium - put_intrinsic - call_intrinsic  # per share
        margin = max(put_strike, call_strike) * 0.20  # ~20% margin requirement

        all_trades.append({
            "entry_date": entry_str,
            "expiry_date": exp_str,
            "dte": (exp_date - datetime.strptime(entry_str, "%Y-%m-%d")).days,
            "spy_entry": spy_entry,
            "spy_expiry": spy_expiry,
            "spy_move_pct": (spy_expiry - spy_entry) / spy_entry,
            "put_strike": put_strike,
            "call_strike": call_strike,
            "put_ticker": put_tk,
            "call_ticker": call_tk,
            "put_premium": pp,
            "call_premium": cp,
            "total_premium": premium,
            "put_intrinsic": put_intrinsic,
            "call_intrinsic": call_intrinsic,
            "pnl_per_share": pnl,
            "pnl_per_contract": pnl * 100,
            "pnl_pct_margin": pnl / margin if margin else 0,
            "margin_per_contract": margin * 100,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(expirations)}] fetched={len(all_trades)} misses={misses}")

    # ── Save ──
    df = pd.DataFrame(all_trades)
    df.to_parquet(SAVE_PATH, index=False)
    print(f"\nSaved {len(df)} trades to {SAVE_PATH}")
    print(f"Misses: {misses}")

    if len(df) > 0:
        print(f"\n{'='*60}")
        print("VRP BACKTEST SUMMARY (raw strangle selling)")
        print(f"{'='*60}")
        print(f"Period:         {df['entry_date'].min()} to {df['entry_date'].max()}")
        print(f"Trades:         {len(df)}")
        print(f"Avg premium:    ${df['total_premium'].mean():.2f}/share")
        print(f"Avg P&L:        ${df['pnl_per_contract'].mean():.0f}/contract")
        print(f"Avg return:     {df['pnl_pct_margin'].mean():.1%} per cycle")
        print(f"Win rate:       {(df['pnl_per_contract'] > 0).mean():.0%}")
        print(f"Best trade:     ${df['pnl_per_contract'].max():.0f}")
        print(f"Worst trade:    ${df['pnl_per_contract'].min():.0f}")
        print(f"Sharpe (ann):   {df['pnl_pct_margin'].mean() / df['pnl_pct_margin'].std() * np.sqrt(12):.2f}")


if __name__ == "__main__":
    main()
