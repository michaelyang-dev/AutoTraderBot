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

After completion, restarts signal-server to pick up new data.

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
    """Refresh the FMP 'enhanced' caches (price targets, DCF, growth, EV, profiles, economic calendar).

    2026-09-30: FMP ONLY. It used to call fetch_all_data.main(), which ALSO swept options snapshots for all
    1,502 names — the same ~4-minute sweep refresh_options() runs again later in this job. On the weekly
    day when the 7-day FMP caches expire (~7.5 min of FMP calls) the duplicate sweep pushed the step past
    its 600s budget (2026-09-29 17:40 "enhanced_data exceeded 600s timeout" -> false 'DATA REFRESH
    PARTIAL' alert; every file had in fact been written). Each fetcher keeps its own TTL cache and saves
    its own file, so a partial run keeps everything already fetched."""
    log("Refreshing FMP enhanced data (price targets, DCF, growth, EV, profiles, economic calendar)...")
    t0 = time.time()
    from strategies import fetch_all_data as F
    from sp500_universe import get_all_symbols, get_etf_symbols
    etfs = set(get_etf_symbols())
    symbols = sorted(s for s in get_all_symbols() if s not in etfs)
    F.fetch_price_targets(symbols)
    F.fetch_dcf_values(symbols)
    F.fetch_financial_growth(symbols)
    F.fetch_enterprise_values(symbols)
    F.fetch_company_profiles(symbols)
    F.fetch_economic_calendar()
    log(f"Enhanced (FMP) data refreshed in {time.time() - t0:.0f}s")
    return True


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
        # Clear stale JSON cache first (rebuilds from API, prevents 900MB bloat)
        from pathlib import Path
        cache_dir = Path(__file__).resolve().parent.parent / "data" / "fundamentals_cache"
        if cache_dir.exists():
            import glob
            old_files = glob.glob(str(cache_dir / "*.json"))
            if len(old_files) > 100:
                for f in old_files:
                    Path(f).unlink(missing_ok=True)
                log(f"  Cleared {len(old_files)} stale JSON cache files")

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


def refresh_fama_french():
    """Update Fama-French factors from Ken French's website (free, daily)."""
    log("Refreshing Fama-French factors...")
    try:
        import pandas as pd
        from pathlib import Path
        import io, zipfile, urllib.request

        data_dir = Path(__file__).resolve().parent.parent / "data" / "wrds"

        # Download from Ken French's website (canonical source, free)
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

        # 5 factors
        resp = urllib.request.urlopen(url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
        raw = z.read(csv_name).decode('utf-8')
        # Skip header rows (find first line that starts with a date)
        lines = raw.strip().split('\n')
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit() and len(line.strip().split(',')[0]) == 8:
                data_start = i
                break
        data_lines = []
        for line in lines[data_start:]:
            parts = line.strip().split(',')
            if len(parts) >= 6 and parts[0].strip().isdigit():
                data_lines.append(parts)
            else:
                break

        ff5 = pd.DataFrame(data_lines, columns=['date', 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'rf'])
        ff5['date'] = pd.to_datetime(ff5['date'].str.strip(), format='%Y%m%d')
        for col in ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'rf']:
            ff5[col] = pd.to_numeric(ff5[col].str.strip(), errors='coerce') / 100

        # Momentum factor
        resp2 = urllib.request.urlopen(mom_url, timeout=30)
        z2 = zipfile.ZipFile(io.BytesIO(resp2.read()))
        csv2 = [n for n in z2.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
        raw2 = z2.read(csv2).decode('utf-8')
        lines2 = raw2.strip().split('\n')
        data_start2 = 0
        for i, line in enumerate(lines2):
            if line.strip() and line.strip()[0].isdigit() and len(line.strip().split(',')[0]) == 8:
                data_start2 = i
                break
        mom_lines = []
        for line in lines2[data_start2:]:
            parts = line.strip().split(',')
            if len(parts) >= 2 and parts[0].strip().isdigit():
                mom_lines.append(parts[:2])
            else:
                break

        mom = pd.DataFrame(mom_lines, columns=['date', 'umd'])
        mom['date'] = pd.to_datetime(mom['date'].str.strip(), format='%Y%m%d')
        mom['umd'] = pd.to_numeric(mom['umd'].str.strip(), errors='coerce') / 100

        # Merge
        ff = ff5.merge(mom, on='date', how='outer').sort_values('date').dropna(subset=['date'])
        ff.to_parquet(data_dir / "fama_french_5factors_momentum_daily.parquet", index=False)

        log(f"Fama-French updated: {len(ff)} rows, latest={ff['date'].max().date()}")
        return True
    except Exception as e:
        log(f"ERROR refreshing Fama-French: {e}")
        return False


def flush_old_fundamentals():
    """Remove fundamentals cache files older than 1 day to save disk.
    The daily refresh rebuilds the full cache, so old files are just wasting space.
    886MB/day × 7 days = 6GB — disk only has 2GB free."""
    log("Flushing old fundamentals cache...")
    try:
        from pathlib import Path
        cache_dir = Path(__file__).resolve().parent.parent / "data" / "fundamentals_cache"
        if not cache_dir.exists():
            return True

        cutoff = time.time() - 1 * 86400  # 1 day — keep only today's data
        removed = 0
        for f in cache_dir.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1

        if removed > 0:
            log(f"  Removed {removed} stale cache files (>1 day old)")
        else:
            log(f"  No stale cache files found")
        return True
    except Exception as e:
        log(f"ERROR flushing cache: {e}")
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


def _step_child(func):
    """Body of a forked step process. Exit code is the ONLY result channel: 0 = func returned truthy,
    3 = func returned falsy, 4 = func raised. os._exit skips inherited atexit handlers."""
    code = 4
    try:
        code = 0 if func() else 3
    except BaseException as e:            # noqa: BLE001 — report everything, the parent decides
        log(f"  step raised {type(e).__name__}: {e}")
        code = 4
    finally:
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        os._exit(code)


def _stale_outputs(outputs, now=None):
    """[(path, max_age_hours)] -> list of human-readable problems (missing / too old)."""
    now = now or time.time()
    bad = []
    for rel, max_h in outputs:
        pth = ML_DIR / rel
        if not pth.exists():
            bad.append(f"{rel} missing")
            continue
        age_h = (now - pth.stat().st_mtime) / 3600
        if age_h > max_h:
            bad.append(f"{rel} {age_h:.0f}h old (max {max_h}h)")
    return bad


def _run_step(func, label, timeout_sec, retries=1, outputs=()):
    """Run one refresh step in a FORKED CHILD PROCESS with a hard kill at `timeout_sec`.

    Replaces _run_with_timeout (SIGALRM), which had three failure modes (2026-09-29 incident):
      * the alarm raises TimeoutError INSIDE the step, where any broad `except Exception` (every
        fetcher has one) can swallow it -> the timeout silently never fires, or fires at a random
        point and throws away unsaved work;
      * steps catch their own errors and RETURN False, and the old wrapper only retried on an
        exception -> retries never ran;
      * a step could 'succeed' without writing anything.
    Now: timeout = SIGTERM/SIGKILL of the child (cannot be swallowed); retry on timeout, crash,
    exception or a falsy return; after a clean exit the step's declared outputs must exist and be
    within their max age, otherwise the attempt counts as failed.
    Returns (ok: bool, detail: str)."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    detail = "not run"
    for attempt in range(1 + retries):
        t0 = time.time()
        proc = ctx.Process(target=_step_child, args=(func,), name=f"refresh-{label}", daemon=False)
        proc.start()
        proc.join(timeout_sec)
        if proc.is_alive():
            proc.terminate(); proc.join(15)
            if proc.is_alive():
                proc.kill(); proc.join(5)
            detail = f"timed out after {timeout_sec}s (child killed)"
        elif proc.exitcode == 0:
            bad = _stale_outputs(outputs)
            if not bad:
                return True, f"ok in {time.time() - t0:.0f}s" + (f" (attempt {attempt + 1})" if attempt else "")
            detail = "finished but outputs not fresh: " + "; ".join(bad)
        elif proc.exitcode == 3:
            detail = "step reported failure (returned False)"
        elif proc.exitcode == 4:
            detail = "step raised an exception (see log above)"
        else:
            detail = f"child died (exit code {proc.exitcode})"
        log(f"STEP {label}: attempt {attempt + 1}/{retries + 1} FAILED — {detail}")
        if attempt < retries:
            log(f"  Retrying {label} in 10s...")
            time.sleep(10)
    return False, detail


ARCHIVE_SETTLE_HOUR_ET = 17   # == massive_data_provider.SETTLE_HOUR_ET (a bar is final from 17:00 ET on its date)


def _final_bar(df, mtime, settle_hour=ARCHIVE_SETTLE_HOUR_ET):
    """(session_date, close) of the LAST bar in `df` whose session had settled when the file was written
    (epoch `mtime`), else None. Looks back at most 3 rows. Pure — unit-tested."""
    from zoneinfo import ZoneInfo
    import pandas as pd
    et = ZoneInfo("America/New_York")
    idx = pd.to_datetime(df.index)
    for pos in range(len(df) - 1, max(len(df) - 4, -1), -1):
        d = idx[pos]
        if datetime(d.year, d.month, d.day, settle_hour, tzinfo=et).timestamp() <= mtime:
            c = df["close"].iloc[pos]
            if pd.notna(c):
                return pd.Timestamp(d).normalize(), float(c)
    return None


def archive_daily_prices():
    """Archive each symbol's FINAL daily close from the Massive cache, labelled by the bar's OWN session.

    Until 2026-09-30 this took every cache file's LAST row and labelled it with the wall-clock date. The
    cache was refreshed in the morning, so for most names that row was the PREVIOUS session (and on Mondays
    a ~15:05 intraday snapshot): closes_D held mostly D-1's closes. Now: a bar counts only if the file was
    written after that session settled (17:00 ET), and only the session most files agree on is archived,
    under its own date. If that session is already archived this is a no-op."""
    import pandas as pd
    CACHE_DIR = ML_DIR / "data" / "massive_cache"
    ARCHIVE_DIR = ML_DIR / "data" / "price_archive"
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        finals = {}
        for f in CACHE_DIR.glob("*_adj.parquet"):
            try:
                df = pd.read_parquet(f)
                if "close" in df.columns and len(df) > 0:
                    fb = _final_bar(df, f.stat().st_mtime)
                    if fb is not None:
                        finals[f.stem.replace("_adj", "")] = fb
            except Exception:
                continue
        if not finals:
            log("Price archive: no settled bars in the cache — nothing to archive")
            return True
        session = pd.Series([d for d, _ in finals.values()]).mode().max()
        closes = {s: c for s, (d, c) in finals.items() if d == session}
        label = session.strftime("%Y-%m-%d")
        archive_file = ARCHIVE_DIR / f"closes_{label}.parquet"
        if archive_file.exists():
            log(f"Price archive for session {label} already exists — skipping")
            return True
        pd.DataFrame([{"date": label, **closes}]).to_parquet(archive_file, index=False)
        log(f"Archived {len(closes)} final closes for session {label} "
            f"({len(finals) - len(closes)} files on other sessions not archived)")

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
        # Get actual table/column names to avoid errors
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        deleted = 0
        for table in tables:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            # Find a date column
            date_col = None
            for c in ['timestamp', 'created_at', 'date', 'time']:
                if c in cols:
                    date_col = c
                    break
            if date_col:
                days = 7 if table == 'signals' else 30
                conn.execute(f"DELETE FROM {table} WHERE {date_col} < datetime('now', '-{days} days')")
                deleted += conn.total_changes
        conn.commit()  # MUST commit the DELETEs first: VACUUM inside the implicit
        # transaction raised "cannot VACUUM from within a transaction" every run,
        # and the exception path then skipped commit — so cleanup NEVER applied.
        if deleted > 0:
            conn.execute("VACUUM")
            log(f"Journal cleanup: deleted {deleted} old rows, vacuumed")
        conn.close()
        return True
    except Exception as e:
        log(f"Journal cleanup failed: {e}")
        return False


# Step table (2026-09-30). `live` = does a failure change what the LIVE engine trades?
#   The live signal path is build_signals_v9(raw, enhanced_data=None, ...) (signal_server.py) — the FMP
#   "enhanced" caches are deliberately NOT fed to live selection (parity with the backtest, whose
#   FastBacktester.DEPLOYED_EMPTY_MAPS clears the same maps). tests/test_refresh_runner.py asserts that
#   call signature: if someone wires enhanced data into live signals, the test fails and this table must
#   be re-classified. Budgets are ~2x the slowest run seen in logs/refresh_data.log.
STEPS = [
    # label,            func-name,                  budget_s, retries, live,  what it feeds,                                             outputs [(path, max_age_h)]
    ("enhanced_fmp",    "refresh_enhanced_data",    1500,     1,       False, "FMP price targets/DCF/growth/EV/profiles/econ calendar — research + snapshot history only",
        [("data/enhanced_data/price_targets.parquet", 96), ("data/enhanced_data/dcf_values.parquet", 96),
         ("data/enhanced_data/financial_growth.parquet", 192), ("data/enhanced_data/enterprise_values.parquet", 192),
         ("data/enhanced_data/company_profiles.parquet", 192), ("data/enhanced_data/economic_calendar.parquet", 48)]),
    ("VIX",             "refresh_vix_cache",        180,      2,       True,  "VIX / VIX3M in the live universe's regime dict",
        [("data/enhanced_data/vix_cache.parquet", 2)]),
    ("fundamentals",    "refresh_fundamentals",     3600,     1,       False, "FMP fundamentals — live FALLBACK only (WRDS Compustat + EDGAR are primary)",
        [("data/fundamentals_ratios.parquet", 48), ("data/fundamentals_income.parquet", 48),
         ("data/fundamentals_metrics.parquet", 48), ("data/fundamentals_earnings.parquet", 48)]),
    ("options",         "refresh_options",          900,      1,       False, "options P/C + IV snapshots — research history only",
        [("data/enhanced_data/options_snapshots.parquet", 2)]),
    ("snapshots",       "refresh_snapshot_history", 300,      1,       False, "point-in-time history files — research only", []),
    ("price_archive",   "archive_daily_prices",     300,      1,       False, "daily close archive — research only", []),
    ("journal_cleanup", "cleanup_journal_db",       120,      0,       False, "journal DB housekeeping", []),
    ("fama_french",     "refresh_fama_french",      300,      2,       False, "FF factors — research; live momentum-crash detector is price-based",
        [("data/wrds/fama_french_5factors_momentum_daily.parquet", 2)]),
    ("data_gaps",       "check_data_gaps",          600,      0,       False, "Massive cache gap scan + yfinance patch", []),
    ("cache_flush",     "flush_old_fundamentals",   120,      0,       False, "cache housekeeping", []),
]


def main():
    log("=" * 60)
    log("  DAILY DATA REFRESH")
    log("=" * 60)
    t_all = time.time()
    results = {}
    for label, fname, budget, retries, live, feeds, outputs in STEPS:
        ok, detail = _run_step(globals()[fname], label, budget, retries=retries, outputs=outputs)
        results[label] = (ok, detail, live, feeds)
        log(f"STEP {label}: {'OK' if ok else 'FAILED'} — {detail}")

    if any(results[k][0] for k in ("enhanced_fmp", "VIX", "fundamentals", "options")):
        restart_ml_server()

    failed = [(k, v) for k, v in results.items() if not v[0]]
    live_failed = [(k, v) for k, v in failed if v[2]]
    status = "OK" if not failed else ("LIVE-INPUT FAILURE" if live_failed else "PARTIAL (research data only)")
    log(f"Refresh complete in {time.time() - t_all:.0f}s: {status} — " +
        ", ".join(f"{k}={'ok' if v[0] else 'FAIL'}" for k, v in results.items()))

    # Housekeeping-only failures are logged, not paged.
    paged = [(k, v) for k, v in failed if k not in ("journal_cleanup", "data_gaps", "cache_flush")]
    if paged:
        head = ("🚨 DATA REFRESH — LIVE INPUT FAILED" if live_failed else
                "ℹ️ DATA REFRESH — research data only (live trading NOT affected)")
        lines = [head]
        for k, (ok, detail, live, feeds) in paged:
            lines.append(f"• {k}: {detail}\n  feeds: {feeds}{' [LIVE]' if live else ''}")
        lines.append("Last good files stay in place; the next scheduled run retries.")
        _send_telegram_alert("\n".join(lines))
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

    data_root = ML_DIR / "data"          # == Path(__file__).parent.parent / "data"; ML_DIR so tests can redirect it
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

    # Auto-fix: patch short-bar stocks from yfinance
    to_fix = [sym for sym, n in short_bars if n > 0] + missing[:50]
    if to_fix:
        log(f"  Auto-patching {len(to_fix)} symbols from yfinance...")
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            start = (datetime.today() - timedelta(days=550)).strftime("%Y-%m-%d")
            # yfinance `end` is EXCLUSIVE. With end=today this step (runs ~18:30 ET) wrote history ending
            # the PREVIOUS session into the live cache every evening; fresh by mtime, those files were
            # reused by the 18:33 build and the next pre-open build, so ~15 renamed tickers ranked with a
            # missing last bar (2026-09-30). end=tomorrow includes the session that just closed.
            end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

            def _write_if_not_older(sym, df):
                """Never replace a cache file with history that ends EARLIER than what it holds."""
                f = cache_dir / f"{sym}_adj.parquet"
                try:
                    if f.exists():
                        cur_last = pd.to_datetime(pd.read_parquet(f).index).max()
                        if pd.to_datetime(df.index).max() < cur_last:
                            log(f"  {sym}: yfinance ends {str(pd.to_datetime(df.index).max())[:10]} < cached "
                                f"{str(cur_last)[:10]} — NOT overwriting")
                            return False
                except Exception:
                    pass
                df[["open", "high", "low", "close", "volume"]].to_parquet(f)
                return True

            CHUNK = 50
            patched = 0
            for i in range(0, len(to_fix), CHUNK):
                chunk = to_fix[i:i + CHUNK]
                try:
                    data = yf.download(chunk, start=start, end=end,
                                       auto_adjust=True, progress=False, threads=True)
                    if isinstance(data.columns, pd.MultiIndex):
                        for sym in chunk:
                            try:
                                df = data.xs(sym, level=1, axis=1).dropna(how="all")
                                df.columns = [c.lower() for c in df.columns]
                                if len(df) >= 252 and _write_if_not_older(sym, df):
                                    patched += 1
                            except Exception:
                                pass
                    elif len(chunk) == 1:
                        data.columns = [c.lower() for c in data.columns]
                        _d1 = data.dropna(how="all")
                        if len(_d1) >= 252 and _write_if_not_older(chunk[0], _d1):
                            patched += 1
                except Exception as e:
                    log(f"  yfinance chunk failed: {e}")

            log(f"  Patched {patched}/{len(to_fix)} symbols from yfinance")
        except ImportError:
            log("  yfinance not available — skipping auto-patch")

    if len(short_bars) > 30 or len(missing) > 100:
        _send_telegram_alert(
            f"⚠️ DATA GAPS: {len(short_bars)} stocks with <252 bars, "
            f"{len(missing)} missing from cache"
        )
    # The scan/patch ran; short or missing histories are reported above, not a step failure. Without this the
    # forked runner read the implicit None as "returned False" and marked the step FAILED every night (2026-09-30).
    return True


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
