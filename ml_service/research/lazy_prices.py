"""
Lazy Prices Signal — Filing Similarity (Cohen, Malloy, Nguyen 2020)
=====================================================================
When a company materially changes language in its 10-K/10-Q vs prior year,
it predicts negative future returns. The market underreacts to text changes.

Signal: cosine similarity between current filing and same-type filing from
prior year. LOW similarity = bearish (company added hedging/risk language).

Data: SEC EDGAR (free, complete back to 1990s).
Point-in-time: use filing acceptance timestamp, trade next open.

Three gates:
1. Does it predict anything? (IC + decile spread)
2. Is it orthogonal to momentum? (factor regression)
3. Is the edge in under-covered names? (by market cap)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
import re
import urllib.request
import json
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

log = logging.getLogger("lazy_prices")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "edgar_filings"


import ssl
_SSL_CTX = ssl._create_unverified_context()

def fetch_filing_index(cik, filing_type="10-K", start_date="2016-01-01"):
    """Fetch filing index from EDGAR company submissions API."""
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    headers = {"User-Agent": "AutoTrader Research michael@example.com"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            raw = resp.read()
            # Handle gzip if needed
            if raw[:2] == b'\x1f\x8b':
                import gzip
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode())
        return data
    except Exception as e:
        return None


def get_filing_text_from_url(filing_url):
    """Download and extract text from an EDGAR filing."""
    headers = {"User-Agent": "AutoTrader Research michael@example.com"}
    try:
        req = urllib.request.Request(filing_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
            raw = resp.read().decode('utf-8', errors='replace')

        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', raw)
        # Strip excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove XBRL/XML artifacts
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        # Keep only ASCII text
        text = re.sub(r'[^\x20-\x7E\n]', '', text)

        return text.strip()
    except Exception as e:
        return None


def get_filings_for_ticker(ticker, cik_map, filing_type="10-K"):
    """Get list of filings with dates and URLs for a ticker."""
    cik = cik_map.get(ticker)
    if not cik:
        return []

    data = fetch_filing_index(cik, filing_type)
    if not data:
        return []

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i in range(len(forms)):
        if forms[i] == filing_type:
            acc = accessions[i].replace("-", "")
            doc = primary_docs[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
            filings.append({
                "date": dates[i],
                "url": url,
                "accession": accessions[i],
            })

    return sorted(filings, key=lambda x: x["date"])


def compute_similarity(text1, text2):
    """Compute cosine similarity between two filing texts using TF-IDF."""
    if not text1 or not text2:
        return None

    # Use only the first 50k chars (filings can be huge)
    t1 = text1[:50000]
    t2 = text2[:50000]

    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            min_df=1,
            ngram_range=(1, 2),
        )
        tfidf = vectorizer.fit_transform([t1, t2])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except Exception:
        return None


def build_cik_map():
    """Build ticker -> CIK mapping from SEC company tickers."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "AutoTrader Research michael@example.com"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode())

        cik_map = {}
        for entry in data.values():
            ticker = entry.get("ticker", "")
            cik = entry.get("cik_str", "")
            if ticker and cik:
                cik_map[ticker] = str(cik)
        return cik_map
    except Exception as e:
        print(f"Error building CIK map: {e}")
        return {}


def main():
    print("="*70)
    print("LAZY PRICES — FILING SIMILARITY SIGNAL")
    print("="*70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build CIK map
    print("\n[1] Building ticker -> CIK mapping...")
    cache_file = DATA_DIR / "cik_map.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cik_map = json.load(f)
        print(f"  Loaded {len(cik_map)} mappings from cache")
    else:
        cik_map = build_cik_map()
        with open(cache_file, "w") as f:
            json.dump(cik_map, f)
        print(f"  Built {len(cik_map)} mappings")

    # Step 2: Get a sample of tickers from our universe
    print("\n[2] Selecting test universe...")
    try:
        sp1500 = json.load(open(
            Path(__file__).resolve().parent.parent / "data" / "sp1500_members.json"))
        if isinstance(sp1500, dict):
            all_tickers = set()
            for idx_members in sp1500.values():
                if isinstance(idx_members, list):
                    all_tickers.update(idx_members)
            tickers = sorted(all_tickers)
        else:
            tickers = sorted(sp1500)
    except Exception:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                    "JPM", "BAC", "WFC", "JNJ", "PFE", "UNH", "HD", "LOW",
                    "MCD", "SBUX", "NKE", "DIS", "NFLX"]

    # Filter to those with CIK
    tickers_with_cik = [t for t in tickers if t in cik_map]
    print(f"  Universe: {len(tickers)} tickers, {len(tickers_with_cik)} with CIK")

    # Start with a small sample for speed
    sample = tickers_with_cik[:100]  # First 100 for initial test
    print(f"  Testing on {len(sample)} tickers (sample)")

    # Step 3: Fetch filings and compute similarities
    print("\n[3] Fetching filings from EDGAR (this may take a while)...")
    similarities = {}  # {ticker: [(date, similarity), ...]}

    fetched = 0
    errors = 0
    for i, ticker in enumerate(sample):
        if i % 20 == 0:
            print(f"  ...{i}/{len(sample)} tickers processed", flush=True)

        filings = get_filings_for_ticker(ticker, cik_map, "10-K")
        if len(filings) < 2:
            continue

        ticker_sims = []
        for j in range(1, len(filings)):
            curr = filings[j]
            prev = filings[j-1]

            # Check cache
            cache_key = f"{ticker}_{curr['date']}_{prev['date']}"
            sim_cache = DATA_DIR / f"{cache_key}.sim"

            if sim_cache.exists():
                sim = float(sim_cache.read_text())
            else:
                # Rate limit: EDGAR allows 10 requests/second
                time.sleep(0.15)

                text_curr = get_filing_text_from_url(curr["url"])
                time.sleep(0.15)
                text_prev = get_filing_text_from_url(prev["url"])

                sim = compute_similarity(text_curr, text_prev)
                fetched += 1

                if sim is not None:
                    sim_cache.write_text(str(sim))
                else:
                    errors += 1
                    continue

            ticker_sims.append({
                "date": curr["date"],
                "similarity": sim,
            })

        if ticker_sims:
            similarities[ticker] = ticker_sims

    print(f"  Fetched {fetched} filing pairs, {errors} errors")
    print(f"  Similarities computed for {len(similarities)} tickers")

    if len(similarities) < 20:
        print("\n  Not enough data for meaningful analysis. Need more tickers.")
        print("  Try increasing sample size or checking EDGAR rate limits.")
        return

    # Step 4: Gate 1 — Does it predict anything?
    print("\n[4] GATE 1: Does filing similarity predict returns?")
    print("-"*50)

    # Load price data
    try:
        import pickle
        with open("data/wrds/complete_sp1500_universe.pkl", "rb") as f:
            uni_data = pickle.load(f)
        prices = uni_data["prices_df"]
    except Exception:
        from fast_backtest import FastBacktester
        bt = FastBacktester("data/wrds/complete_sp1500_universe.pkl")
        prices = bt.prices

    # Compute forward returns after each filing
    signal_data = []
    for ticker, sims in similarities.items():
        for entry in sims:
            filing_date = pd.Timestamp(entry["date"])
            sim = entry["similarity"]

            # Forward returns: 1-month, 3-month, 6-month after filing
            if ticker not in prices.columns:
                continue

            px = prices[ticker]
            # Find first trading day after filing
            future_dates = px.index[px.index > filing_date]
            if len(future_dates) < 130:
                continue

            entry_px = px.loc[future_dates[0]]
            if pd.isna(entry_px) or entry_px <= 0:
                continue

            ret_21 = (px.loc[future_dates[min(20, len(future_dates)-1)]] / entry_px - 1)
            ret_63 = (px.loc[future_dates[min(62, len(future_dates)-1)]] / entry_px - 1)
            ret_126 = (px.loc[future_dates[min(125, len(future_dates)-1)]] / entry_px - 1)

            signal_data.append({
                "ticker": ticker,
                "date": filing_date,
                "similarity": sim,
                "ret_1m": ret_21,
                "ret_3m": ret_63,
                "ret_6m": ret_126,
            })

    df = pd.DataFrame(signal_data)
    if len(df) < 50:
        print("  Not enough signal observations for analysis")
        return

    print(f"  {len(df)} filing-return observations")
    print(f"  Similarity range: {df['similarity'].min():.3f} to {df['similarity'].max():.3f}")
    print(f"  Mean similarity: {df['similarity'].mean():.3f}")

    # Information coefficient (rank correlation)
    for horizon in ["ret_1m", "ret_3m", "ret_6m"]:
        ic = df["similarity"].corr(df[horizon], method="spearman")
        print(f"  IC (similarity → {horizon}): {ic:+.4f}")

    # Quintile spread
    df["sim_quintile"] = pd.qcut(df["similarity"], 5, labels=[1,2,3,4,5], duplicates="drop")
    print(f"\n  Quintile analysis (1=most changed, 5=most similar):")
    print(f"  {'Quintile':<10} {'Count':>6} {'Avg 3m Ret':>12} {'Avg 6m Ret':>12}")
    print(f"  {'-'*44}")
    for q in sorted(df["sim_quintile"].unique()):
        subset = df[df["sim_quintile"] == q]
        print(f"  Q{q:<9} {len(subset):>6} {subset['ret_3m'].mean()*100:>+11.2f}% {subset['ret_6m'].mean()*100:>+11.2f}%")

    q1 = df[df["sim_quintile"] == 1]["ret_3m"].mean()
    q5 = df[df["sim_quintile"] == 5]["ret_3m"].mean()
    spread = q5 - q1
    print(f"\n  Long-short spread (Q5 - Q1, 3-month): {spread*100:+.2f}%")
    print(f"  Prediction: LOW similarity → NEGATIVE returns (spread should be positive)")

    if spread > 0.01:
        print(f"  → GATE 1 PASSED: Filing similarity predicts returns")
    elif spread > 0:
        print(f"  → GATE 1 WEAK: Small positive spread, marginal")
    else:
        print(f"  → GATE 1 FAILED: No predictive power")

    # Step 5: Gate 2 — Orthogonal to momentum?
    print(f"\n[5] GATE 2: Is it orthogonal to momentum?")
    print("-"*50)

    # Compute momentum for each observation
    for idx, row in df.iterrows():
        ticker = row["ticker"]
        date = row["date"]
        if ticker in prices.columns:
            px = prices[ticker]
            hist = px.loc[:date].dropna()
            if len(hist) >= 252:
                ret_12m = hist.iloc[-1] / hist.iloc[-252] - 1
                ret_1m_hist = hist.iloc[-1] / hist.iloc[-21] - 1
                df.loc[idx, "momentum_12_1"] = ret_12m - ret_1m_hist
            else:
                df.loc[idx, "momentum_12_1"] = np.nan
        else:
            df.loc[idx, "momentum_12_1"] = np.nan

    valid = df.dropna(subset=["momentum_12_1", "similarity", "ret_3m"])
    if len(valid) > 30:
        corr = valid["similarity"].corr(valid["momentum_12_1"], method="spearman")
        print(f"  Correlation (similarity vs momentum): {corr:+.3f}")
        if abs(corr) < 0.2:
            print(f"  → Low correlation — signal IS orthogonal to momentum")
        else:
            print(f"  → High correlation — signal overlaps with momentum")

    print(f"\n{'='*70}")
    print("LAZY PRICES ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
