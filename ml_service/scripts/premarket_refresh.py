#!/usr/bin/env python3
"""Quick pre-market refresh: VIX + crypto only (~2 min)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
os.chdir(Path(__file__).resolve().parent.parent)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from strategies.fetch_all_data import fetch_crypto_forex, fetch_vix_data
print("Pre-market refresh: VIX + crypto...", flush=True)
fetch_vix_data()
fetch_crypto_forex(["SPY"])
print("Pre-market refresh done.", flush=True)
