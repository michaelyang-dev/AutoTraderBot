#!/usr/bin/env python3
"""
Weekly Data Refresh
===================
Refreshes FMP enhanced data (price targets, DCF, growth, etc.)
and VIX cache so the v9.6 strategy has current fundamentals.

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


def refresh_options():
    """Refresh options snapshots (put/call ratio, IV) from Polygon."""
    log("Refreshing options snapshots...")
    t0 = time.time()
    try:
        from strategies.fetch_all_data import fetch_options_snapshots
        from sp500_universe import get_stock_symbols
        symbols = get_stock_symbols()
        fetch_options_snapshots(symbols)
        log(f"Options refreshed in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        log(f"ERROR refreshing options: {e}")
        return False


def refresh_ortex():
    """Refresh Ortex short interest data (bulk SP500)."""
    log("Refreshing Ortex short interest...")
    t0 = time.time()
    try:
        import requests
        import urllib.parse
        import pandas as pd
        from pathlib import Path

        ORTEX_KEY = os.environ.get("ORTEX_API_KEY", "")
        if not ORTEX_KEY:
            log("  ORTEX_API_KEY not set — skipping")
            return False

        base = "https://api.ortex.com/api/v1"
        headers = {"Ortex-Api-Key": ORTEX_KEY, "accept": "application/json"}
        DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "enhanced_data"

        for endpoint, filename in [
            ("short_interest", "ortex_short_interest.parquet"),
            ("short_dtc", "ortex_short_dtc.parquet"),
            ("short_ctb", "ortex_short_ctb.parquet"),
            ("short_availability", "ortex_short_availability.parquet"),
        ]:
            all_rows = []
            url = f"{base}/index/{endpoint}?format=json&index={urllib.parse.quote('US-S 500')}"
            while url:
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                all_rows.extend(data.get("rows", []))
                nxt = data.get("paginationLinks", {}).get("next")
                url = nxt if nxt and nxt.startswith("http") else None
                time.sleep(0.3)

            if all_rows:
                df = pd.DataFrame(all_rows)
                df.to_parquet(DATA_DIR / filename, index=False)
                log(f"  {endpoint}: {len(df)} stocks")

        log(f"Ortex refreshed in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        log(f"ERROR refreshing Ortex: {e}")
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


def refresh_snapshot_history():
    """Accumulate daily snapshots of financial scores and analyst grades for future backtesting."""
    log("Accumulating daily snapshot history (scores, grades)...")
    t0 = time.time()
    try:
        import requests
        import pandas as pd
        from pathlib import Path

        FMP_KEY = os.environ.get("FMP_API_KEY", "")
        DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "enhanced_data"
        today = datetime.now().strftime("%Y-%m-%d")

        for endpoint, snapshot_file, history_file, symbol_col in [
            ("financial-scores", "financial_scores.parquet", "financial_scores_history.parquet", "symbol"),
            ("grades-consensus", "analyst_grades_consensus.parquet", "analyst_grades_history.parquet", "symbol"),
        ]:
            # Read current snapshot
            snap_path = DATA_DIR / snapshot_file
            if not snap_path.exists():
                continue
            df = pd.read_parquet(snap_path)
            df["date"] = today

            # Append to history
            hist_path = DATA_DIR / history_file
            if hist_path.exists():
                try:
                    existing = pd.read_parquet(hist_path)
                    existing = existing[existing["date"] != today]  # remove today if re-running
                    df = pd.concat([existing, df], ignore_index=True)
                except Exception:
                    pass
            df.to_parquet(hist_path, index=False)
            log(f"  {history_file}: {len(df)} total rows ({df['date'].nunique()} dates)")

        log(f"Snapshot history updated in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        log(f"ERROR updating snapshot history: {e}")
        return False


def main():
    log("=" * 60)
    log("  DAILY DATA REFRESH")
    log("=" * 60)

    ok1 = refresh_enhanced_data()
    ok2 = refresh_vix_cache()
    ok3 = refresh_fundamentals()
    ok4 = refresh_options()
    ok5 = refresh_ortex()
    ok6 = refresh_snapshot_history()

    if ok1 or ok2 or ok3 or ok4 or ok5:
        restart_ml_server()

    status = "OK" if (ok1 and ok2 and ok3 and ok4 and ok5 and ok6) else "PARTIAL"
    log(f"Refresh complete: {status} (enhanced={ok1}, vix={ok2}, fundamentals={ok3}, options={ok4}, ortex={ok5}, snapshots={ok6})")


if __name__ == "__main__":
    main()
