#!/usr/bin/env python3
"""
Sentiment Collector — builds a point-in-time news+sentiment archive going forward.

WHY: there is no historical PIT sentiment dataset to backtest on (RavenPack etc.
are institutional-priced), so we accumulate our own from Alpaca's news feed
(Benzinga-sourced, ticker-tagged, timestamped). The RAW news is archived first —
that's the irreplaceable point-in-time data; the sentiment SCORE is a convenience
that can be recomputed later from the raw text with a better model (FinBERT/LLM).

Design:
  - Fetches news since the last checkpoint (self-heals missed runs).
  - Idempotent: dedupes by article id, so it can run as often as you like.
  - Filters to the SP1500 universe to control disk.
  - Raw -> data/sentiment_archive/raw/news_YYYY-MM.jsonl (append, deduped).
  - Daily per-ticker aggregate -> data/sentiment_archive/daily/YYYY-MM-DD.json
    (rebuilt from raw each run, so always correct regardless of run frequency).

Run daily via cron (after close). VADER score if installed; otherwise raw-only
(scores backfillable later). Never throws on a single bad article.
"""
import os
import re
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parent
ARCH = BASE / "data" / "sentiment_archive"
RAW_DIR = ARCH / "raw"
DAILY_DIR = ARCH / "daily"
STATE = ARCH / "state.json"
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
LOOKBACK_DAYS_COLD = 5   # first run: how far back to seed

# ── optional finance-aware-ish sentiment (VADER if available; else raw-only) ──
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()

    def score_text(text):
        return round(_vader.polarity_scores(text)["compound"], 4) if text else None
    SCORER = "vader"
except Exception:
    def score_text(text):
        return None
    SCORER = "none"


def load_env():
    env = dict(os.environ)
    # check ml_service/.env and the repo-root .env (keys live at repo root here)
    for f in (BASE / ".env", BASE.parent / ".env"):
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env.setdefault(k.strip(), v.strip())
    return env


def load_universe():
    """SP1500 tickers to filter on; None = keep everything."""
    for p in [BASE / "data" / "sp1500_members.json", BASE / "sp1500_members.json",
              BASE / "data" / "sp1500_universe.json"]:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if isinstance(d, list):
                    return set(d)
                if isinstance(d, dict):
                    # union every list-of-tickers value (handles {sp500:[...],sp400:[...],sp600:[...]})
                    tickers = set()
                    for v in d.values():
                        if isinstance(v, list):
                            tickers.update(x for x in v if isinstance(x, str))
                    if tickers:
                        return tickers
            except Exception:
                pass
    return None


def fetch_news(keys, start_iso, end_iso):
    headers = {"APCA-API-KEY-ID": keys["ALPACA_API_KEY"],
               "APCA-API-SECRET-KEY": keys["ALPACA_SECRET_KEY"]}
    params = {"start": start_iso, "end": end_iso, "limit": 50,
              "sort": "asc", "include_content": "false"}
    out, token = [], None
    while True:
        if token:
            params["page_token"] = token
        r = requests.get(NEWS_URL, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("news", []))
        token = data.get("next_page_token")
        if not token or len(out) > 50000:
            break
        time.sleep(0.25)
    return out


def existing_ids(month):
    f = RAW_DIR / f"news_{month}.jsonl"
    ids = set()
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                continue
    return ids


def rebuild_daily(days_touched):
    """Recompute per-ticker daily aggregates from raw for the affected days."""
    months = sorted({d[:7] for d in days_touched})
    by_day = {}
    for m in months:
        f = RAW_DIR / f"news_{m}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            day = (rec.get("created_at") or "")[:10]
            if day not in days_touched or rec.get("sentiment") is None:
                continue
            for s in rec.get("symbols", []):
                by_day.setdefault(day, {}).setdefault(s, []).append(rec["sentiment"])
    for day, tickmap in by_day.items():
        agg = {t: {"mean_sent": round(sum(v) / len(v), 4), "n": len(v)}
               for t, v in tickmap.items()}
        (DAILY_DIR / f"{day}.json").write_text(json.dumps(agg))
    return len(by_day)


def main():
    for d in (RAW_DIR, DAILY_DIR):
        d.mkdir(parents=True, exist_ok=True)
    keys = load_env()
    if not keys.get("ALPACA_API_KEY") or not keys.get("ALPACA_SECRET_KEY"):
        print("ERROR: no Alpaca API keys in env/.env")
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    now = datetime.now(timezone.utc)
    start = state.get("last_iso") or (now - timedelta(days=LOOKBACK_DAYS_COLD)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[sentiment] fetch {start} -> {end} | scorer={SCORER}")

    arts = fetch_news(keys, start, end)
    print(f"[sentiment] fetched {len(arts)} raw articles")
    universe = load_universe()
    print(f"[sentiment] universe filter: {'SP1500 ('+str(len(universe))+')' if universe else 'none (keep all)'}")

    # dedupe against what's already archived this/last month
    month_ids = {}
    new_by_month = {}
    days_touched = set()
    new_count = 0
    for a in arts:
        created = a.get("created_at") or ""
        month = created[:7] or now.strftime("%Y-%m")
        if month not in month_ids:
            month_ids[month] = existing_ids(month)
        aid = a.get("id")
        if aid in month_ids[month]:
            continue
        syms = [s for s in (a.get("symbols") or []) if (universe is None or s in universe)]
        if not syms:
            continue
        text = ((a.get("headline") or "") + ". " + (a.get("summary") or "")).strip()
        rec = {"id": aid, "created_at": created, "symbols": syms,
               "headline": a.get("headline"), "summary": a.get("summary"),
               "source": a.get("source"), "url": a.get("url"),
               "sentiment": score_text(text)}
        new_by_month.setdefault(month, []).append(rec)
        month_ids[month].add(aid)
        if created[:10]:
            days_touched.add(created[:10])
        new_count += 1

    for month, recs in new_by_month.items():
        with open(RAW_DIR / f"news_{month}.jsonl", "a") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")

    n_days = rebuild_daily(days_touched) if days_touched else 0
    state["last_iso"] = end
    state["last_run"] = now.isoformat()
    state["total_archived"] = state.get("total_archived", 0) + new_count
    state["scorer"] = SCORER
    STATE.write_text(json.dumps(state))
    print(f"[sentiment] archived {new_count} new ticker-relevant articles; "
          f"daily files updated: {n_days}; total_archived={state['total_archived']}")


if __name__ == "__main__":
    main()
