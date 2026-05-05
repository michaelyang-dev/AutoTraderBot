#!/usr/bin/env python3
"""
FINRA Short Interest Auto-Fetcher
===================================
Fetches short interest data from FINRA's free API biweekly.
Computes SI change rankings and saves for the signal server.

FINRA reports short interest twice monthly (settlement dates ~15th and ~end of month).
Data is released ~8 days after settlement date.

Schedule: Cron on 1st and 15th of each month.

Usage:
    cd ml_service && python3 scripts/fetch_finra_si.py
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Setup paths
ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))
os.chdir(ML_DIR)

from dotenv import load_dotenv
load_dotenv(ML_DIR.parent / ".env")

DATA_DIR = ML_DIR / "data"
WRDS_DIR = DATA_DIR / "wrds"
SI_FILE = WRDS_DIR / "compustat_short_interest.parquet"
SI_CHANGE_FILE = WRDS_DIR / "si_change_ranks.parquet"

# FINRA API
FINRA_API = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
# No auth needed for basic queries (public access tier)
# If rate-limited, register at https://gateway.finra.org/app/dfo-console


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_latest_settlement_date():
    """Find the most recent settlement date with data on FINRA."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Try recent dates (FINRA has ~8 day delay, reports on ~15th and ~end of month)
    from datetime import date
    today = date.today()
    candidates = []
    for days_ago in range(0, 45):
        d = today - timedelta(days=days_ago)
        candidates.append(d.strftime("%Y-%m-%d"))

    for test_date in candidates:
        payload = {
            "limit": 1,
            "fields": ["symbolCode", "settlementDate"],
            "domainFilters": [{"fieldName": "settlementDate", "values": [test_date]}],
        }
        try:
            resp = requests.post(FINRA_API, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200 and resp.json():
                log(f"Latest settlement date: {test_date}")
                return test_date
        except:
            continue
    return None


def fetch_finra_short_interest():
    """Fetch current short interest from FINRA API (free, no auth needed)."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    settlement_date = find_latest_settlement_date()
    if not settlement_date:
        log("Could not find any recent settlement date")
        return None

    # FINRA provides both current AND previous SI in each record
    # so we only need one date to compute the change
    all_records = []
    offset = 0
    while True:
        payload = {
            "limit": 5000,
            "offset": offset,
            "fields": ["symbolCode", "issueName", "currentShortPositionQuantity",
                       "previousShortPositionQuantity", "settlementDate",
                       "averageDailyVolumeQuantity", "daysToCoverQuantity"],
            "domainFilters": [{"fieldName": "settlementDate", "values": [settlement_date]}],
        }

        try:
            resp = requests.post(FINRA_API, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                batch = resp.json()
                if not batch:
                    break
                all_records.extend(batch)
                log(f"  Fetched {len(batch)} records (total: {len(all_records)})")
                if len(batch) < 5000:
                    break
                offset += 5000
                time.sleep(0.5)  # rate limit courtesy
            else:
                log(f"  FINRA API error at offset {offset}: {resp.status_code}")
                break
        except Exception as e:
            log(f"  Request failed at offset {offset}: {e}")
            break

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    log(f"Total records: {len(df)} tickers for {settlement_date}")
    return df


def compute_si_change_ranks(df):
    """Compute SI change rankings from FINRA data.
    FINRA provides both current and previous SI in each record."""
    if df is None or len(df) == 0:
        return None

    df = df.rename(columns={
        "symbolCode": "tic",
        "currentShortPositionQuantity": "shortint",
        "previousShortPositionQuantity": "prev_shortint",
        "settlementDate": "datadate",
    })

    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df.dropna(subset=["shortint", "prev_shortint"])
    df = df[(df["shortint"] > 0) & (df["prev_shortint"] > 0)]

    # SI change = (current - previous) / previous
    df["si_change"] = (df["shortint"] - df["prev_shortint"]) / df["prev_shortint"]

    # Rank: negative change (covering) = high rank = bullish
    ranks = (-df.set_index("tic")["si_change"]).rank(pct=True)

    log(f"SI change ranks computed: {len(ranks)} tickers")
    covering = df.nsmallest(5, "si_change")[["tic", "si_change"]]
    building = df.nlargest(5, "si_change")[["tic", "si_change"]]
    log(f"  Top covering: {dict(zip(covering['tic'], covering['si_change'].round(3)))}")
    log(f"  Top building: {dict(zip(building['tic'], building['si_change'].round(3)))}")

    return ranks


def update_wrds_file(finra_df):
    """Append FINRA data to existing WRDS parquet for continuity."""
    if finra_df is None:
        return

    # Convert to WRDS-compatible format
    new_data = finra_df.rename(columns={
        "symbolCode": "tic",
        "currentShortPositionQuantity": "shortintadj",
        "settlementDate": "datadate",
    })[["tic", "datadate", "shortintadj"]].copy()
    new_data["datadate"] = pd.to_datetime(new_data["datadate"])
    new_data = new_data.dropna(subset=["shortintadj"])
    new_data = new_data[new_data["shortintadj"] > 0]

    if SI_FILE.exists():
        existing = pd.read_parquet(SI_FILE, columns=["tic", "datadate", "shortintadj"])
        existing["datadate"] = pd.to_datetime(existing["datadate"])

        # Only append dates we don't already have
        existing_dates = set(existing["datadate"].dt.date.unique())
        new_dates = set(new_data["datadate"].dt.date.unique())
        truly_new = new_dates - existing_dates

        if truly_new:
            new_rows = new_data[new_data["datadate"].dt.date.isin(truly_new)]
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.sort_values(["tic", "datadate"])
            combined.to_parquet(SI_FILE, index=False)
            log(f"Appended {len(new_rows)} new rows ({len(truly_new)} new dates) to SI file")
        else:
            log("No new dates to append — SI file already up to date")
    else:
        WRDS_DIR.mkdir(parents=True, exist_ok=True)
        new_data.to_parquet(SI_FILE, index=False)
        log(f"Created new SI file with {len(new_data)} rows")


def save_si_change_ranks(ranks):
    """Save SI change ranks for signal server to pick up."""
    if ranks is None:
        return

    # Save as simple parquet
    df = pd.DataFrame({"tic": ranks.index, "si_change_rank": ranks.values})
    df.to_parquet(SI_CHANGE_FILE, index=False)
    log(f"Saved SI change ranks to {SI_CHANGE_FILE}")


if __name__ == "__main__":
    log("=" * 60)
    log("FINRA Short Interest Fetch — Biweekly Update")
    log("=" * 60)

    # Fetch from FINRA
    df = fetch_finra_short_interest()

    if df is not None and len(df) > 0:
        # Compute ranks
        ranks = compute_si_change_ranks(df)

        # Update the WRDS file (append new data)
        update_wrds_file(df)

        # Save ranks for signal server
        save_si_change_ranks(ranks)

        log("Done — SI data updated successfully")
    else:
        log("FAILED — could not fetch FINRA data")
        # Try alternative: use existing WRDS file to recompute ranks
        if SI_FILE.exists():
            log("Falling back to existing WRDS file for rank computation")
            si = pd.read_parquet(SI_FILE, columns=["tic", "datadate", "shortintadj"])
            si["datadate"] = pd.to_datetime(si["datadate"])
            si = si.dropna(subset=["shortintadj"])
            si = si[si["shortintadj"] > 0].sort_values(["tic", "datadate"])
            si["si_prev"] = si.groupby("tic")["shortintadj"].shift(2)
            si["si_change"] = (si["shortintadj"] - si["si_prev"]) / si["si_prev"]
            si = si.dropna(subset=["si_change"])
            latest = si.sort_values("datadate").groupby("tic")["si_change"].last()
            ranks = (-latest).rank(pct=True)
            save_si_change_ranks(ranks)
            log(f"Fallback: recomputed ranks from existing file ({len(ranks)} tickers)")

    log("=" * 60)
