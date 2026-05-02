"""
Norgate SP1500 Constituent Export (runs on Windows/Parallels)
=============================================================
Run this weekly after NDU updates. It exports current SP500/SP400/SP600
membership to a JSON file on the shared Mac folder.

Setup:
  1. pip install norgatedata  (on the Windows/Parallels side)
  2. Set EXPORT_PATH below to your shared folder path
  3. Schedule via NDU "Post-Update Command" or Windows Task Scheduler

Usage:
  python norgate_export_sp1500.py
"""

import json
import os
from datetime import datetime

import norgatedata

# ── Configure this path ─────────────────────────────────────────
# Parallels shared folder — adjust if your Mac username or repo path differs
EXPORT_PATH = r"\\Mac\Home\Downloads\auto-trader 2\ml_service\data\sp1500_members.json"

# Fallback: local export (copy manually)
EXPORT_PATH_LOCAL = "sp1500_members.json"


def export():
    print("Fetching index constituents from Norgate...")

    sp500 = norgatedata.index_constituent_tickers("S&P 500")
    sp400 = norgatedata.index_constituent_tickers("S&P MidCap 400")
    sp600 = norgatedata.index_constituent_tickers("S&P SmallCap 600")

    data = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sp500": sorted(sp500),
        "sp400": sorted(sp400),
        "sp600": sorted(sp600),
    }

    total = len(sp500) + len(sp400) + len(sp600)
    print(f"  SP500: {len(sp500)}, SP400: {len(sp400)}, SP600: {len(sp600)} → total {total}")

    # Try shared folder first, fall back to local
    path = EXPORT_PATH
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported to {path}")
    except OSError:
        path = EXPORT_PATH_LOCAL
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Shared folder unavailable — exported to {path}")
        print("Copy this file to: ml_service/data/sp1500_members.json on your Mac")

    return data


if __name__ == "__main__":
    export()
