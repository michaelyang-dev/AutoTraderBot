#!/usr/bin/env python3
"""
Daily Data Refresh
==================
Refreshes all data sources and accumulates daily snapshots for
future backtesting. Runs Mon-Fri at 5:00 PM ET via PM2 cron.

What it does:
  1. Refreshes FMP enhanced data (price targets, DCF, growth, profiles)
  2. Refreshes VIX cache from yfinance
  3. Refreshes FMP fundamentals (income, ratios, earnings)
  4. Refreshes options snapshots from Massive/Polygon (accumulates history)
  5. Refreshes Ortex short interest data
  6. Accumulates ALL snapshots to _history.parquet files (point-in-time database)

The history accumulation (#6) is critical: it builds a point-in-time record
of signals that change daily (price targets, DCF, short interest, etc.).
Without this, backtests use current values applied to past dates (look-ahead bias).

After completion, restarts ml-server to pick up new data.

Usage:
    cd ml_service && python3 scripts/refresh_data.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
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
    """Refresh VIX cache from Polygon/Massive (primary) or yfinance (fallback)."""
    log("Refreshing VIX cache...")
    t0 = time.time()
    try:
        import pandas as pd
        import requests

        DATA_DIR = ML_DIR / "data" / "enhanced_data"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("MASSIVE_API_KEY", "")
        success = False

        # Primary: Polygon/Massive
        if api_key:
            try:
                vix_frames = {}
                for ticker, col_name in [("I:VIX", "^VIX"), ("I:VIX3M", "^VIX3M")]:
                    resp = requests.get(
                        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2016-01-01/{datetime.now().strftime('%Y-%m-%d')}",
                        params={"apiKey": api_key, "limit": 50000, "adjusted": "true"},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        if results:
                            df = pd.DataFrame(results)
                            df["date"] = pd.to_datetime(df["t"], unit="ms")
                            df = df.set_index("date")["c"].rename(col_name)
                            vix_frames[col_name] = df
                    time.sleep(0.15)

                if vix_frames:
                    vix_data = pd.DataFrame(vix_frames)
                    vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)
                    vix_data.to_parquet(DATA_DIR / "vix_cache.parquet")
                    log(f"VIX cache refreshed from Polygon: {len(vix_data)} rows, latest={vix_data.index[-1].date()}")
                    success = True
            except Exception as e:
                log(f"Polygon VIX failed: {e}, trying yfinance fallback...")

        # Fallback: yfinance
        if not success:
            try:
                import yfinance as yf
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
                    log(f"VIX cache refreshed from yfinance: {len(vix_data)} rows, latest={vix_data.index[-1].date()}")
                    success = True
                else:
                    log("WARNING: No VIX data returned from yfinance")
            except Exception as e:
                log(f"yfinance VIX also failed: {e}")

        log(f"VIX refresh done in {time.time() - t0:.0f}s")
        return success
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
        from sp500_universe import get_all_symbols, get_etf_symbols
        etfs = set(get_etf_symbols())
        symbols = [s for s in get_all_symbols() if s not in etfs]
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
    """Restart signal-server via PM2 to pick up fresh data."""
    log("Restarting signal-server to load fresh data...")
    try:
        result = subprocess.run(
            ["pm2", "restart", "signal-server"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log("signal-server restarted successfully")
        else:
            log(f"WARNING: pm2 restart returned code {result.returncode}: {result.stderr}")
    except Exception as e:
        log(f"ERROR restarting signal-server: {e}")


def _accumulate_snapshot(snapshot_file, history_file, data_dir, today):
    """Append today's snapshot to a history parquet file.

    Reads the current snapshot, stamps it with today's date,
    and appends it to the history file (deduplicating if re-run same day).
    """
    import pandas as pd

    snap_path = data_dir / snapshot_file
    if not snap_path.exists():
        return None

    df = pd.read_parquet(snap_path)
    df["date"] = today

    hist_path = data_dir / history_file
    if hist_path.exists():
        try:
            existing = pd.read_parquet(hist_path)
            existing = existing[existing["date"] != today]  # remove today if re-running
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass

    df.to_parquet(hist_path, index=False)
    n_dates = df["date"].nunique()
    log(f"  {history_file}: {len(df)} total rows ({n_dates} dates)")
    return len(df)


def refresh_snapshot_history():
    """Accumulate daily snapshots of ALL enhanced data for future backtesting.

    Every snapshot file gets stamped with today's date and appended to a
    corresponding _history.parquet file. Over time this builds a point-in-time
    database of signals that can't be reconstructed from other sources.

    Data accumulated:
      - Financial scores (Piotroski, Altman Z)
      - Analyst grades consensus (buy/hold/sell)
      - Price targets (analyst consensus targets)
      - DCF values (intrinsic value estimates)
      - Financial growth (revenue/EPS/FCF growth rates)
      - Enterprise values (EV/Revenue, market cap)
      - Ortex short interest (SI%, days-to-cover, cost-to-borrow)
      - Options snapshots (put/call ratio, IV) — already accumulated by fetch_options_snapshots
      - Company profiles (beta, sector, market cap)
      - Transcript sentiment (earnings call sentiment)
    """
    log("Accumulating daily snapshot history (ALL enhanced data)...")
    t0 = time.time()
    try:
        import pandas as pd

        DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "enhanced_data"
        today = datetime.now().strftime("%Y-%m-%d")
        accumulated = 0

        # All snapshot → history pairs to accumulate
        SNAPSHOT_PAIRS = [
            # (snapshot_file, history_file)
            # Financial scores & analyst grades (already existed)
            ("financial_scores.parquet", "financial_scores_history.parquet"),
            ("analyst_grades_consensus.parquet", "analyst_grades_history.parquet"),
            # Price targets & DCF (snapshot-only before, NOW accumulating)
            ("price_targets.parquet", "price_targets_history.parquet"),
            ("dcf_values.parquet", "dcf_values_history.parquet"),
            # Financial growth & enterprise values
            ("financial_growth.parquet", "financial_growth_history.parquet"),
            ("enterprise_values.parquet", "enterprise_values_history.parquet"),
            # Ortex short interest
            ("ortex_short_interest.parquet", "ortex_short_interest_history.parquet"),
            ("ortex_short_dtc.parquet", "ortex_short_dtc_history.parquet"),
            ("ortex_short_ctb.parquet", "ortex_short_ctb_history.parquet"),
            ("ortex_short_availability.parquet", "ortex_short_availability_history.parquet"),
            # Company profiles & sentiment
            ("company_profiles.parquet", "company_profiles_history.parquet"),
            ("transcript_sentiment.parquet", "transcript_sentiment_history.parquet"),
        ]

        for snapshot_file, history_file in SNAPSHOT_PAIRS:
            result = _accumulate_snapshot(snapshot_file, history_file, DATA_DIR, today)
            if result is not None:
                accumulated += 1

        # Also accumulate fundamentals (stored in parent data/ dir, not enhanced_data/)
        FUND_DIR = DATA_DIR.parent
        FUND_PAIRS = [
            ("fundamentals_earnings.parquet", "fundamentals_earnings_history.parquet"),
            ("fundamentals_ratios.parquet", "fundamentals_ratios_history.parquet"),
            ("fundamentals_estimates.parquet", "fundamentals_estimates_history.parquet"),
        ]
        for snapshot_file, history_file in FUND_PAIRS:
            result = _accumulate_snapshot(snapshot_file, history_file, FUND_DIR, today)
            if result is not None:
                accumulated += 1

        # Version SP1500 membership (copy with date stamp)
        sp1500_file = FUND_DIR / "sp1500_members.json"
        sp1500_archive = FUND_DIR / "sp1500_archive"
        if sp1500_file.exists():
            sp1500_archive.mkdir(parents=True, exist_ok=True)
            dated = sp1500_archive / f"sp1500_{today}.json"
            if not dated.exists():
                import shutil
                shutil.copy2(sp1500_file, dated)
                log(f"  SP1500 membership archived: {dated.name}")
                accumulated += 1

        log(f"Snapshot history updated: {accumulated} datasets accumulated in {time.time() - t0:.0f}s")
        return True
    except Exception as e:
        log(f"ERROR updating snapshot history: {e}")
        return False


def _run_with_timeout(func, label, timeout_sec=600):
    """Run a refresh function with a hard timeout."""
    import signal as _sig

    def _handler(signum, frame):
        raise TimeoutError(f"{label} exceeded {timeout_sec}s timeout")

    old = _sig.signal(_sig.SIGALRM, _handler)
    _sig.alarm(timeout_sec)
    try:
        result = func()
    except TimeoutError as e:
        log(f"TIMEOUT: {e}")
        result = False
    finally:
        _sig.alarm(0)
        _sig.signal(_sig.SIGALRM, old)
    return result


def archive_daily_prices():
    """Archive today's closing prices from Massive cache for future backtesting.
    This builds a point-in-time price database that doesn't rely on WRDS."""
    import pandas as pd
    CACHE_DIR = ML_DIR / "data" / "massive_cache"
    ARCHIVE_DIR = ML_DIR / "data" / "price_archive"
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    archive_file = ARCHIVE_DIR / f"closes_{today}.parquet"

    if archive_file.exists():
        log(f"Price archive for {today} already exists — skipping")
        return True

    try:
        closes = {}
        parquets = list(CACHE_DIR.glob("*_adj.parquet"))
        for f in parquets:
            try:
                df = pd.read_parquet(f)
                if "close" in df.columns and len(df) > 0:
                    sym = f.stem.replace("_adj", "")
                    closes[sym] = df["close"].iloc[-1]
            except:
                continue

        if closes:
            pd.DataFrame([{"date": today, **closes}]).to_parquet(archive_file, index=False)
            log(f"Archived {len(closes)} closing prices for {today}")

            # Cleanup: keep last 90 days of daily files, consolidate older into monthly
            cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            old_files = [f for f in ARCHIVE_DIR.glob("closes_*.parquet")
                         if f.stem.split("_")[1] < cutoff]
            if len(old_files) > 30:
                log(f"Consolidating {len(old_files)} old price archives...")
                dfs = [pd.read_parquet(f) for f in old_files]
                combined = pd.concat(dfs, ignore_index=True)
                combined.to_parquet(ARCHIVE_DIR / "closes_consolidated.parquet", index=False)
                for f in old_files:
                    f.unlink()
                log(f"Consolidated into closes_consolidated.parquet ({len(combined)} rows)")
        return True
    except Exception as e:
        log(f"Price archiving failed: {e}")
        return False


def cleanup_journal_db():
    """Prevent journal.db from bloating by cleaning old signals/events."""
    import sqlite3
    DB_PATH = ML_DIR.parent / "data" / "journal.db"
    if not DB_PATH.exists():
        return True
    try:
        conn = sqlite3.connect(str(DB_PATH))
        # Keep only last 7 days of signals (this table bloats fastest)
        conn.execute("DELETE FROM signals WHERE timestamp < datetime('now', '-7 days')")
        conn.execute("DELETE FROM events WHERE timestamp < datetime('now', '-30 days')")
        deleted = conn.total_changes
        conn.execute("VACUUM")
        conn.commit()
        conn.close()
        if deleted > 0:
            log(f"Journal cleanup: deleted {deleted} old rows, vacuumed")
        return True
    except Exception as e:
        log(f"Journal cleanup failed: {e}")
        return False


def main():
    log("=" * 60)
    log("  DAILY DATA REFRESH")
    log("=" * 60)

    ok1 = _run_with_timeout(refresh_enhanced_data, "enhanced_data", 300)
    ok2 = _run_with_timeout(refresh_vix_cache, "VIX", 60)
    ok3 = _run_with_timeout(refresh_fundamentals, "fundamentals", 600)
    ok4 = _run_with_timeout(refresh_options, "options", 300)
    ok5 = _run_with_timeout(refresh_ortex, "ortex", 600)
    ok6 = _run_with_timeout(refresh_snapshot_history, "snapshots", 120)
    ok7 = _run_with_timeout(archive_daily_prices, "price_archive", 120)
    ok8 = _run_with_timeout(cleanup_journal_db, "journal_cleanup", 60)

    ok9 = _run_with_timeout(check_data_gaps, "data_gaps", 120)

    if ok1 or ok2 or ok3 or ok4 or ok5:
        restart_ml_server()

    status = "OK" if (ok1 and ok2 and ok3 and ok4 and ok5 and ok6) else "PARTIAL"
    log(f"Refresh complete: {status} (enhanced={ok1}, vix={ok2}, fundamentals={ok3}, options={ok4}, ortex={ok5}, snapshots={ok6}, prices={ok7}, journal={ok8})")

    # Alert on failure via Telegram
    if status != "OK":
        failures = []
        if not ok1: failures.append("enhanced_data")
        if not ok2: failures.append("VIX")
        if not ok3: failures.append("fundamentals")
        if not ok4: failures.append("options")
        if not ok5: failures.append("ortex")
        if not ok6: failures.append("snapshots")
        _send_telegram_alert(
            f"⚠️ DATA REFRESH {status}\n"
            f"Failed: {', '.join(failures)}\n"
            f"System will use last known good data."
        )
    else:
        log("All refreshes succeeded — no alerts needed")


def check_data_gaps():
    """Scan Massive cache for stocks with insufficient price history.
    Stocks with <252 bars will produce NaN features and get rejected
    by signal_builder. This check identifies them proactively.
    """
    import json
    import pandas as pd
    from pathlib import Path

    data_root = Path(__file__).resolve().parent.parent / "data"
    cache_dir = data_root / "massive_cache"
    members_file = data_root / "sp1500_members.json"

    if not cache_dir.exists():
        log("No massive_cache directory — skipping data gap check")
        return

    # Load SP1500 members
    sp1500 = set()
    if members_file.exists():
        with open(members_file) as f:
            data = json.load(f)
        for key in ["sp500", "sp400", "sp600"]:
            sp1500.update(data.get(key, []))

    short_bars = []
    missing = []
    total = 0

    for sym in sorted(sp1500):
        f = cache_dir / f"{sym}_adj.parquet"
        if not f.exists():
            missing.append(sym)
            continue
        total += 1
        try:
            df = pd.read_parquet(f)
            n = len(df)
            if n < 252:
                short_bars.append((sym, n))
        except Exception:
            short_bars.append((sym, 0))

    log(f"Data gap check: {total} cached, {len(missing)} missing, {len(short_bars)} with <252 bars")

    if short_bars:
        log(f"  Short bars: {short_bars[:20]}")

    if missing and len(missing) < 50:
        log(f"  Missing: {missing[:20]}")

    if len(short_bars) > 30 or len(missing) > 100:
        _send_telegram_alert(
            f"⚠️ DATA GAPS: {len(short_bars)} stocks with <252 bars, "
            f"{len(missing)} missing from cache"
        )


def _send_telegram_alert(message):
    """Send alert via Telegram (same bot as tradingEngine)."""
    import urllib.request
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        log("Telegram not configured — alert not sent")
        return
    try:
        import urllib.parse
        url = (f"https://api.telegram.org/bot{bot_token}/sendMessage"
               f"?chat_id={chat_id}&text={urllib.parse.quote(message)}")
        urllib.request.urlopen(url, timeout=10)
        log(f"Telegram alert sent: {message[:80]}...")
    except Exception as e:
        log(f"Failed to send Telegram alert: {e}")


if __name__ == "__main__":
    main()
