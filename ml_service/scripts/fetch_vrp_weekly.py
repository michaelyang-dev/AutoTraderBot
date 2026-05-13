#!/usr/bin/env python3
"""
Fetch weekly historical options for SPY, QQQ, IWM from Alpha Vantage.
Extracts 30-delta strangles for VRP backtest.
"""
import os, sys, time, json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "data"
AV_KEY = "ODTCLRYD9DC93IDM"
BASE = "https://www.alphavantage.co/query"
RATE_DELAY = 0.85  # 75 req/min


def av_get(symbol, date_str, retries=3):
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "date": date_str,
        "apikey": AV_KEY,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(BASE, params=params, timeout=30)
            time.sleep(RATE_DELAY)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("message") == "success" and data.get("data"):
                    return data["data"]
                if "rate limit" in str(data).lower() or "premium" in str(data).lower():
                    print(f"    Rate limited, waiting 15s...")
                    time.sleep(15)
                    continue
                return []
            elif resp.status_code == 429:
                time.sleep(15)
                continue
        except Exception:
            if attempt < retries - 1:
                time.sleep(3)
    return []


def find_30delta_strangle(contracts):
    """Extract ~30-delta strangle from options chain."""
    if not contracts:
        return None

    calls, puts = [], []
    for c in contracts:
        try:
            entry = {
                "expiration": c.get("expiration", ""),
                "strike": float(c.get("strike", 0)),
                "delta": float(c.get("delta", 0)),
                "mark": float(c.get("mark", 0)),
                "bid": float(c.get("bid", 0)),
                "ask": float(c.get("ask", 0)),
                "iv": float(c.get("implied_volatility", 0)),
                "volume": int(c.get("volume", 0)),
                "oi": int(c.get("open_interest", 0)),
                "contract_id": c.get("contractID", ""),
            }
            entry["mid"] = (entry["bid"] + entry["ask"]) / 2 if entry["bid"] > 0 and entry["ask"] > 0 else entry["mark"]
            if c.get("type") == "call":
                calls.append(entry)
            else:
                puts.append(entry)
        except (ValueError, TypeError):
            continue

    if not calls or not puts:
        return None

    chain_date = contracts[0].get("date", "")
    if not chain_date:
        return None
    chain_dt = datetime.strptime(chain_date, "%Y-%m-%d")

    # Find all expirations and pick ~30-45 DTE
    all_exps = sorted(set(c["expiration"] for c in calls))
    best_exp = None
    best_dte = 999
    for exp in all_exps:
        try:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - chain_dt).days
            if 25 <= dte <= 50 and abs(dte - 35) < abs(best_dte - 35):
                best_exp, best_dte = exp, dte
        except ValueError:
            continue
    # Fallback
    if not best_exp:
        for exp in all_exps:
            try:
                dte = (datetime.strptime(exp, "%Y-%m-%d") - chain_dt).days
                if dte > 14 and abs(dte - 30) < abs(best_dte - 30):
                    best_exp, best_dte = exp, dte
            except ValueError:
                continue
    if not best_exp:
        return None

    ec = [c for c in calls if c["expiration"] == best_exp]
    ep = [c for c in puts if c["expiration"] == best_exp]
    if not ec or not ep:
        return None

    bc = min(ec, key=lambda c: abs(c["delta"] - 0.30))
    bp = min(ep, key=lambda p: abs(p["delta"] - (-0.30)))

    if bc["mid"] <= 0.01 or bp["mid"] <= 0.01:
        return None

    return {
        "chain_date": chain_date,
        "expiration": best_exp,
        "dte": best_dte,
        "call_strike": bc["strike"],
        "call_delta": bc["delta"],
        "call_mid": bc["mid"],
        "call_bid": bc["bid"],
        "call_ask": bc["ask"],
        "call_iv": bc["iv"],
        "put_strike": bp["strike"],
        "put_delta": bp["delta"],
        "put_mid": bp["mid"],
        "put_bid": bp["bid"],
        "put_ask": bp["ask"],
        "put_iv": bp["iv"],
        "total_premium_mid": bc["mid"] + bp["mid"],
        "total_premium_bid": bc["bid"] + bp["bid"],
    }


def generate_weekly_dates(start_year=2008, end_date="2026-05-12"):
    """Generate every Wednesday from start_year to end_date."""
    dates = []
    dt = datetime(start_year, 1, 1)
    end = datetime.strptime(end_date, "%Y-%m-%d")
    # Find first Wednesday
    while dt.weekday() != 2:  # Wednesday
        dt += timedelta(days=1)
    while dt <= end:
        dates.append(dt.strftime("%Y-%m-%d"))
        dt += timedelta(days=7)
    return dates


def main():
    symbols = ["SPY", "QQQ", "IWM"]
    weekly_dates = generate_weekly_dates(2008, "2026-05-12")
    total_calls = len(symbols) * len(weekly_dates)

    print("=" * 60)
    print("FETCH WEEKLY VRP DATA: SPY + QQQ + IWM")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Weekly dates: {len(weekly_dates)} ({weekly_dates[0]} to {weekly_dates[-1]})")
    print(f"Total API calls: {total_calls}")
    print(f"Estimated time: {total_calls * RATE_DELAY / 60:.0f} minutes")
    print()

    # Check for partial download
    save_path = SAVE_DIR / "vrp_weekly_strangles.parquet"
    existing_keys = set()
    if save_path.exists():
        existing = pd.read_parquet(save_path)
        existing_keys = set(zip(existing["symbol"], existing["chain_date"]))
        print(f"Resuming: {len(existing_keys)} entries already fetched")

    all_strangles = []
    fetched = 0
    errors = 0
    skipped = 0
    call_count = 0

    for sym in symbols:
        print(f"\n{'─'*40}")
        print(f"  Fetching {sym}...")
        print(f"{'─'*40}")
        sym_fetched = 0
        sym_errors = 0

        for date_str in weekly_dates:
            call_count += 1

            if (sym, date_str) in existing_keys:
                skipped += 1
                continue

            contracts = av_get(sym, date_str)

            if contracts:
                strangle = find_30delta_strangle(contracts)
                if strangle:
                    strangle["symbol"] = sym
                    all_strangles.append(strangle)
                    sym_fetched += 1
                    fetched += 1
                else:
                    sym_errors += 1
                    errors += 1
            else:
                sym_errors += 1
                errors += 1

            if call_count % 50 == 0:
                pct = call_count / total_calls * 100
                print(f"  [{call_count}/{total_calls}] ({pct:.0f}%) "
                      f"fetched={fetched} errors={errors} skipped={skipped}")

            # Checkpoint every 200 fetches
            if fetched > 0 and fetched % 200 == 0:
                df_s = pd.DataFrame(all_strangles)
                if save_path.exists():
                    old = pd.read_parquet(save_path)
                    df_s = pd.concat([old, df_s], ignore_index=True)
                    df_s = df_s.drop_duplicates(subset=["symbol", "chain_date"])
                df_s.to_parquet(save_path, index=False)
                print(f"    [checkpoint: {len(df_s)} total strangles saved]")

        print(f"  {sym}: {sym_fetched} strangles, {sym_errors} errors")

    # Final save — merge with any existing
    if all_strangles:
        df_new = pd.DataFrame(all_strangles)
        if save_path.exists():
            old = pd.read_parquet(save_path)
            df_all = pd.concat([old, df_new], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=["symbol", "chain_date"])
        else:
            df_all = df_new
        df_all.to_parquet(save_path, index=False)
        print(f"\nSaved {len(df_all)} total strangles → {save_path.name}")

    # Summary per symbol
    df_all = pd.read_parquet(save_path) if save_path.exists() else pd.DataFrame(all_strangles)
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for sym in symbols:
        sdf = df_all[df_all["symbol"] == sym]
        if len(sdf) > 0:
            print(f"  {sym}: {len(sdf)} strangles, "
                  f"{sdf['chain_date'].min()} to {sdf['chain_date'].max()}, "
                  f"avg premium ${sdf['total_premium_mid'].mean():.2f}")

    print(f"\nDone: {fetched} new, {errors} errors, {skipped} resumed")


if __name__ == "__main__":
    main()
