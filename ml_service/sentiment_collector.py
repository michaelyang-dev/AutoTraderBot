#!/usr/bin/env python3
"""
Sentiment Collector — builds a point-in-time news+sentiment archive going forward.

WHY: there is no historical PIT sentiment dataset to backtest on (RavenPack etc.
are institutional-priced), so we accumulate our own. The RAW news is archived
first — that's the irreplaceable point-in-time data; numeric scores can be
recomputed later from raw text with a better model (FinBERT/LLM).

Sources (multi-source for coverage + redundancy; dedup by native id):
  - Alpaca (Benzinga) news — VADER-scored headline+summary.
  - Polygon news — ships PER-TICKER sentiment insights (pos/neu/neg), used
    directly when present (higher quality than VADER), else VADER fallback.
  (FMP news endpoints are empty on the current plan tier — not used.)

Each raw record carries scores={ticker: float}. Daily per-ticker aggregates are
rebuilt from raw, so correct regardless of run frequency. Idempotent (dedupe by
id), self-heals missed runs. Run via cron. Keys in repo-root .env.
"""
import os
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
ALPACA_URL = "https://data.alpaca.markets/v1beta1/news"
POLYGON_URL = "https://api.polygon.io/v2/reference/news"
LOOKBACK_DAYS_COLD = 5
POLY_SENT = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()

    def vader(text):
        return round(_vader.polarity_scores(text)["compound"], 4) if text else None
    SCORER = "vader+polygon-insights"
except Exception:
    def vader(text):
        return None
    SCORER = "polygon-insights-only"


def load_env():
    env = dict(os.environ)
    for f in (BASE / ".env", BASE.parent / ".env"):
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env.setdefault(k.strip(), v.strip())
    return env


def load_universe():
    for p in [BASE / "data" / "sp1500_members.json", BASE / "sp1500_members.json"]:
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if isinstance(d, list):
                    return set(d)
                if isinstance(d, dict):
                    t = set()
                    for v in d.values():
                        if isinstance(v, list):
                            t.update(x for x in v if isinstance(x, str))
                    if t:
                        return t
            except Exception:
                pass
    return None


def fetch_alpaca(keys, start_iso, end_iso, universe):
    if not keys.get("ALPACA_API_KEY") or not keys.get("ALPACA_SECRET_KEY"):
        print("[alpaca] no keys, skipping")
        return []
    headers = {"APCA-API-KEY-ID": keys["ALPACA_API_KEY"],
               "APCA-API-SECRET-KEY": keys["ALPACA_SECRET_KEY"]}
    params = {"start": start_iso, "end": end_iso, "limit": 50,
              "sort": "asc", "include_content": "false"}
    raw, token = [], None
    while True:
        if token:
            params["page_token"] = token
        r = requests.get(ALPACA_URL, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        raw.extend(data.get("news", []))
        token = data.get("next_page_token")
        if not token or len(raw) > 50000:
            break
        time.sleep(0.25)
    recs = []
    for a in raw:
        syms = [s for s in (a.get("symbols") or []) if (universe is None or s in universe)]
        if not syms:
            continue
        v = vader(((a.get("headline") or "") + ". " + (a.get("summary") or "")).strip())
        recs.append({"id": "alpaca:" + str(a.get("id")), "created_at": a.get("created_at") or "",
                     "symbols": syms, "headline": a.get("headline"), "summary": a.get("summary"),
                     "source": "alpaca/" + (a.get("source") or ""), "url": a.get("url"),
                     "scores": {s: v for s in syms}})
    print(f"[alpaca] {len(raw)} fetched -> {len(recs)} universe-relevant")
    return recs


def fetch_polygon(keys, start_iso, end_iso, universe):
    key = keys.get("MASSIVE_API_KEY") or keys.get("POLYGON_API_KEY")
    if not key:
        print("[polygon] no key, skipping")
        return []
    params = {"published_utc.gte": start_iso, "published_utc.lte": end_iso,
              "order": "asc", "sort": "published_utc", "limit": 1000, "apiKey": key}
    recs, raw_n, url, pages = [], 0, POLYGON_URL, 0
    while url and pages < 25:
        r = requests.get(url, params=params if url == POLYGON_URL else {"apiKey": key}, timeout=30)
        if r.status_code == 429:
            time.sleep(13)
            continue
        r.raise_for_status()
        data = r.json()
        for a in data.get("results", []):
            raw_n += 1
            tickers = [t for t in (a.get("tickers") or []) if (universe is None or t in universe)]
            if not tickers:
                continue
            insights = {i.get("ticker"): POLY_SENT.get(i.get("sentiment"))
                        for i in (a.get("insights") or []) if i.get("ticker")}
            vfb = vader(((a.get("title") or "") + ". " + (a.get("description") or "")).strip())
            scores = {t: (insights[t] if insights.get(t) is not None else vfb) for t in tickers}
            recs.append({"id": "polygon:" + str(a.get("id")), "created_at": a.get("published_utc") or "",
                         "symbols": tickers, "headline": a.get("title"), "summary": a.get("description"),
                         "source": "polygon/" + (a.get("publisher", {}).get("name", "")),
                         "url": a.get("article_url"), "scores": scores})
        url = data.get("next_url")
        pages += 1
        if url:
            time.sleep(0.3)
    print(f"[polygon] {raw_n} fetched -> {len(recs)} universe-relevant ({pages} pages)")
    return recs


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
            if day not in days_touched:
                continue
            # support both new (scores dict) and legacy (sentiment + symbols) records
            scores = rec.get("scores")
            if scores is None and rec.get("sentiment") is not None:
                scores = {s: rec["sentiment"] for s in rec.get("symbols", [])}
            if not scores:
                continue
            for sym, sc in scores.items():
                if sc is not None:
                    by_day.setdefault(day, {}).setdefault(sym, []).append(sc)
    for day, tickmap in by_day.items():
        agg = {t: {"mean_sent": round(sum(v) / len(v), 4), "n": len(v)}
               for t, v in tickmap.items()}
        (DAILY_DIR / f"{day}.json").write_text(json.dumps(agg))
    return len(by_day)


def main():
    for d in (RAW_DIR, DAILY_DIR):
        d.mkdir(parents=True, exist_ok=True)
    keys = load_env()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    now = datetime.now(timezone.utc)
    start = state.get("last_iso") or (now - timedelta(days=LOOKBACK_DAYS_COLD)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    universe = load_universe()
    print(f"[sentiment] {start} -> {end} | scorer={SCORER} | "
          f"universe={'SP1500('+str(len(universe))+')' if universe else 'all'}")

    articles = []
    for fetch in (fetch_alpaca, fetch_polygon):
        try:
            articles += fetch(keys, start, end, universe)
        except Exception as e:
            print(f"[sentiment] {fetch.__name__} error: {e}")

    month_ids, new_by_month, days_touched, new_count = {}, {}, set(), 0
    for rec in articles:
        month = (rec["created_at"][:7]) or now.strftime("%Y-%m")
        if month not in month_ids:
            month_ids[month] = existing_ids(month)
        if rec["id"] in month_ids[month]:
            continue
        new_by_month.setdefault(month, []).append(rec)
        month_ids[month].add(rec["id"])
        if rec["created_at"][:10]:
            days_touched.add(rec["created_at"][:10])
        new_count += 1

    for month, recs in new_by_month.items():
        with open(RAW_DIR / f"news_{month}.jsonl", "a") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")

    n_days = rebuild_daily(days_touched) if days_touched else 0
    state.update({"last_iso": end, "last_run": now.isoformat(),
                  "total_archived": state.get("total_archived", 0) + new_count, "scorer": SCORER})
    STATE.write_text(json.dumps(state))
    print(f"[sentiment] archived {new_count} new; daily files updated: {n_days}; "
          f"total_archived={state['total_archived']}")


if __name__ == "__main__":
    main()
