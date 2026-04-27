#!/usr/bin/env python3
"""
Weekly Data Refresh
===================
Refreshes FMP enhanced data (price targets, DCF, growth, etc.)
and VIX cache so the v9.5 strategy has current fundamentals.

Run via PM2 cron: every Sunday at 5:00 PM ET
After completion, restarts ml-server to pick up new data.

Usage:
    cd ml_service && python3 scripts/refresh_data.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure ml_service is on path
ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))
os.chdir(ML_DIR)

from dotenv import load_dotenv
load_dotenv(ML_DIR.parent / ".env")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def refresh_enhanced_data():
    """Run fetch_all_data.py to refresh FMP caches."""
    log("Refreshing FMP enhanced data (price targets, DCF, growth, profiles, crypto/forex)...")
    t0 = time.time()
    try:
        from strategies.fetch_all_data import main as fetch_main
        fetch_main()
        log(f"Enhanced data refreshed in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        log(f"ERROR refreshing enhanced data: {e}")
        return False


def refresh_vix_cache():
    """Refresh VIX cache from yfinance."""
    log("Refreshing VIX cache...")
    t0 = time.time()
    try:
        import pandas as pd
        import yfinance as yf

        DATA_DIR = ML_DIR / "data" / "enhanced_data"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        vix_raw = yf.download(
            ["^VIX", "^VIX3M"],
            start="2016-01-01",
            end=(datetime.now().strftime("%Y-%m-%d")),
            progress=False,
            auto_adjust=True,
        )
        if len(vix_raw) > 0:
            vix_data = vix_raw["Close"]
            vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)
            vix_data.to_parquet(DATA_DIR / "vix_cache.parquet")
            log(f"VIX cache refreshed: {len(vix_data)} rows, latest={vix_data.index[-1].date()}")
        else:
            log("WARNING: No VIX data returned from yfinance")
        log(f"VIX refresh done in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        log(f"ERROR refreshing VIX: {e}")
        return False


def refresh_fundamentals():
    """Refresh FMP fundamentals parquets (ratios, income, metrics, earnings)."""
    log("Refreshing FMP fundamentals parquets...")
    t0 = time.time()
    try:
        from fmp_fundamentals_pipeline import main as fmp_main
        fmp_main()
        log(f"Fundamentals refreshed in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        log(f"ERROR refreshing fundamentals: {e}")
        return False


def restart_ml_server():
    """Restart ml-server via PM2 to pick up fresh data."""
    log("Restarting ml-server to load fresh data...")
    try:
        result = subprocess.run(
            ["pm2", "restart", "ml-server"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log("ml-server restarted successfully")
        else:
            log(f"WARNING: pm2 restart returned code {result.returncode}: {result.stderr}")
    except Exception as e:
        log(f"ERROR restarting ml-server: {e}")


def main():
    log("=" * 60)
    log("  WEEKLY DATA REFRESH")
    log("=" * 60)

    ok1 = refresh_enhanced_data()
    ok2 = refresh_vix_cache()
    ok3 = refresh_fundamentals()

    if ok1 or ok2 or ok3:
        restart_ml_server()

    status = "OK" if (ok1 and ok2 and ok3) else "PARTIAL"
    log(f"Refresh complete: {status} (enhanced={ok1}, vix={ok2}, fundamentals={ok3})")


if __name__ == "__main__":
    main()
