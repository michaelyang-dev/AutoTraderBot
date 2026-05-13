#!/usr/bin/env python3
"""
Fetch historical SPY options from Alpha Vantage for VRP backtest.
Downloads full chain for ~monthly entry dates from 2008 to 2026.
Extracts 30-delta strangles for VRP P&L computation.
"""
import os, sys, time, json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from calendar import monthcalendar

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "data"
AV_KEY = "ODTCLRYD9DC93IDM"
BASE = "https://www.alphavantage.co/query"

# Rate limit: 75 req/min = 1 req every 0.8s, use 0.85s to be safe
RATE_DELAY = 0.85


def av_get(symbol, date_str, retries=3):
    """Fetch historical options chain for a symbol on a given date."""
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
                # Rate limit hit
                if "rate limit" in str(data).lower() or "premium" in str(data).lower():
                    print(f"    Rate limited, waiting 10s...")
                    time.sleep(10)
                    continue
                return []
            elif resp.status_code == 429:
                time.sleep(10)
                continue
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
    return []


def third_friday(year, month):
    """Get 3rd Friday of a month (standard monthly expiration)."""
    cal = monthcalendar(year, month)
    fridays = [week[4] for week in cal if week[4] != 0]
    return datetime(year, month, fridays[2])


def find_30delta_strangle(contracts, target_expiry=None):
    """
    From a full options chain, find the ~30-delta put and ~30-delta call
    for the nearest monthly expiration ~30 days out.
    Returns dict with entry prices and strikes, or None.
    """
    if not contracts:
        return None

    # Parse contracts into structured data
    calls = []
    puts = []
    for c in contracts:
        try:
            exp = c.get("expiration", "")
            strike = float(c.get("strike", 0))
            delta = float(c.get("delta", 0))
            mark = float(c.get("mark", 0))
            bid = float(c.get("bid", 0))
            ask = float(c.get("ask", 0))
            iv = float(c.get("implied_volatility", 0))
            volume = int(c.get("volume", 0))
            oi = int(c.get("open_interest", 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else mark

            entry = {
                "expiration": exp, "strike": strike, "delta": delta,
                "mark": mark, "bid": bid, "ask": ask, "mid": mid,
                "iv": iv, "volume": volume, "oi": oi,
                "contract_id": c.get("contractID", ""),
            }

            if c.get("type") == "call":
                calls.append(entry)
            else:
                puts.append(entry)
        except (ValueError, TypeError):
            continue

    if not calls or not puts:
        return None

    # Get available expirations
    all_exps = sorted(set(c["expiration"] for c in calls))
    date_of_chain = contracts[0].get("date", "")

    # Find expiration ~30-45 days out
    if target_expiry:
        best_exp = target_expiry
    else:
        chain_date = datetime.strptime(date_of_chain, "%Y-%m-%d") if date_of_chain else datetime.now()
        best_exp = None
        best_dte = 999
        for exp in all_exps:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                dte = (exp_dt - chain_date).days
                if 25 <= dte <= 50:  # target 30-45 DTE window
                    if abs(dte - 35) < abs(best_dte - 35):
                        best_exp = exp
                        best_dte = dte
            except ValueError:
                continue
        # Fallback: closest to 30 DTE
        if not best_exp:
            for exp in all_exps:
                try:
                    exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                    dte = (exp_dt - chain_date).days
                    if dte > 14:  # at least 2 weeks out
                        if abs(dte - 30) < abs(best_dte - 30):
                            best_exp = exp
                            best_dte = dte
                except ValueError:
                    continue

    if not best_exp:
        return None

    # Filter to target expiration
    exp_calls = [c for c in calls if c["expiration"] == best_exp]
    exp_puts = [c for c in puts if c["expiration"] == best_exp]

    if not exp_calls or not exp_puts:
        return None

    # Find ~30-delta call (delta closest to 0.30)
    best_call = min(exp_calls, key=lambda c: abs(c["delta"] - 0.30))
    # Find ~30-delta put (delta closest to -0.30)
    best_put = min(exp_puts, key=lambda p: abs(p["delta"] - (-0.30)))

    # Validate: skip if no real prices
    if best_call["mid"] <= 0.01 or best_put["mid"] <= 0.01:
        return None

    chain_date_str = date_of_chain if date_of_chain else "unknown"

    return {
        "chain_date": chain_date_str,
        "expiration": best_exp,
        "dte": best_dte if not target_expiry else None,
        "call_strike": best_call["strike"],
        "call_delta": best_call["delta"],
        "call_mid": best_call["mid"],
        "call_bid": best_call["bid"],
        "call_ask": best_call["ask"],
        "call_iv": best_call["iv"],
        "call_oi": best_call["oi"],
        "put_strike": best_put["strike"],
        "put_delta": best_put["delta"],
        "put_mid": best_put["mid"],
        "put_bid": best_put["bid"],
        "put_ask": best_put["ask"],
        "put_iv": best_put["iv"],
        "put_oi": best_put["oi"],
        "total_premium_mid": best_call["mid"] + best_put["mid"],
        "total_premium_bid": best_call["bid"] + best_put["bid"],
    }


def main():
    print("=" * 60)
    print("ALPHA VANTAGE: FETCH SPY HISTORICAL OPTIONS")
    print("=" * 60)

    # Generate entry dates: ~30 days before each monthly expiration
    # from 2008 to 2026
    entry_dates = []
    for year in range(2008, 2027):
        for month in range(1, 13):
            try:
                exp = third_friday(year, month)
                if exp > datetime(2026, 5, 12):
                    continue
                # Entry = ~30 days before expiration
                entry = exp - timedelta(days=30)
                # Shift to weekday
                while entry.weekday() >= 5:
                    entry += timedelta(days=1)
                entry_dates.append({
                    "entry": entry.strftime("%Y-%m-%d"),
                    "target_expiry": exp.strftime("%Y-%m-%d"),
                })
            except Exception:
                pass

    print(f"Entry dates to fetch: {len(entry_dates)}")
    print(f"Range: {entry_dates[0]['entry']} to {entry_dates[-1]['entry']}")
    print(f"Estimated time: {len(entry_dates) * RATE_DELAY / 60:.1f} minutes")
    print()

    # Check for existing partial download
    partial_path = SAVE_DIR / "spy_options_av_raw.parquet"
    existing_dates = set()
    if partial_path.exists():
        existing = pd.read_parquet(partial_path)
        existing_dates = set(existing["chain_date"].unique())
        print(f"Resuming: {len(existing_dates)} dates already fetched")

    all_strangles = []
    all_raw_chains = []  # store full chains for flexibility
    fetched = 0
    skipped = 0
    errors = 0

    for idx, ed in enumerate(entry_dates):
        date_str = ed["entry"]

        if date_str in existing_dates:
            skipped += 1
            continue

        contracts = av_get("SPY", date_str)

        if contracts:
            # Extract 30-delta strangle
            strangle = find_30delta_strangle(contracts)
            if strangle:
                strangle["target_expiry"] = ed["target_expiry"]
                all_strangles.append(strangle)

            # Also save summary stats per expiration for the chain
            # (keep it lean — don't store 11K contracts per date)
            exps_in_chain = set(c.get("expiration", "") for c in contracts)
            for exp in sorted(exps_in_chain):
                exp_contracts = [c for c in contracts if c.get("expiration") == exp]
                exp_calls = [c for c in exp_contracts if c.get("type") == "call"]
                exp_puts = [c for c in exp_contracts if c.get("type") == "put"]
                try:
                    avg_call_iv = np.mean([float(c.get("implied_volatility", 0))
                                           for c in exp_calls if float(c.get("implied_volatility", 0)) > 0])
                    avg_put_iv = np.mean([float(c.get("implied_volatility", 0))
                                          for c in exp_puts if float(c.get("implied_volatility", 0)) > 0])
                except (ValueError, ZeroDivisionError):
                    avg_call_iv = avg_put_iv = 0

                all_raw_chains.append({
                    "chain_date": date_str,
                    "expiration": exp,
                    "n_calls": len(exp_calls),
                    "n_puts": len(exp_puts),
                    "avg_call_iv": avg_call_iv,
                    "avg_put_iv": avg_put_iv,
                    "total_call_vol": sum(int(c.get("volume", 0)) for c in exp_calls),
                    "total_put_vol": sum(int(c.get("volume", 0)) for c in exp_puts),
                    "total_call_oi": sum(int(c.get("open_interest", 0)) for c in exp_calls),
                    "total_put_oi": sum(int(c.get("open_interest", 0)) for c in exp_puts),
                })

            fetched += 1
        else:
            errors += 1

        if (idx + 1) % 10 == 0:
            pct = (idx + 1) / len(entry_dates) * 100
            print(f"  [{idx+1}/{len(entry_dates)}] ({pct:.0f}%) fetched={fetched} "
                  f"strangles={len(all_strangles)} errors={errors} skipped={skipped}")

        # Save checkpoint every 50 fetches
        if fetched > 0 and fetched % 50 == 0:
            df_s = pd.DataFrame(all_strangles)
            df_s.to_parquet(SAVE_DIR / "spy_vrp_strangles.parquet", index=False)
            df_r = pd.DataFrame(all_raw_chains)
            df_r.to_parquet(SAVE_DIR / "spy_options_av_raw.parquet", index=False)
            print(f"    [checkpoint saved: {len(df_s)} strangles, {len(df_r)} chain summaries]")

    # Final save
    if all_strangles:
        df_strangles = pd.DataFrame(all_strangles)
        df_strangles.to_parquet(SAVE_DIR / "spy_vrp_strangles.parquet", index=False)
        print(f"\nSaved {len(df_strangles)} strangles → spy_vrp_strangles.parquet")
    if all_raw_chains:
        df_raw = pd.DataFrame(all_raw_chains)
        df_raw.to_parquet(SAVE_DIR / "spy_options_av_raw.parquet", index=False)
        print(f"Saved {len(df_raw)} chain summaries → spy_options_av_raw.parquet")

    # Summary
    if all_strangles:
        df = pd.DataFrame(all_strangles)
        print(f"\n{'='*60}")
        print("30-DELTA STRANGLE DATA SUMMARY")
        print(f"{'='*60}")
        print(f"Period:          {df['chain_date'].min()} to {df['chain_date'].max()}")
        print(f"Entries:         {len(df)}")
        print(f"Avg premium:     ${df['total_premium_mid'].mean():.2f}/share")
        print(f"Avg call delta:  {df['call_delta'].mean():.3f}")
        print(f"Avg put delta:   {df['put_delta'].mean():.3f}")
        print(f"Avg call IV:     {df['call_iv'].mean():.3f}")
        print(f"Avg put IV:      {df['put_iv'].mean():.3f}")
    else:
        print("\nNo strangles extracted!")

    print(f"\nDone: {fetched} fetched, {errors} errors, {skipped} skipped")


if __name__ == "__main__":
    main()
