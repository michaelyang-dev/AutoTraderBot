"""
Stream FNSPID All_external.csv (5.7GB Benzinga headlines, ticker-tagged) and keep
only SP1500-universe tickers, 2016-2023, columns (date, symbol, title). Discards
the empty article bodies. Saves a compact parquet for NLP feature building.
"""
import ssl, urllib.request, certifi, csv, io, time
import pandas as pd

CTX = ssl.create_default_context(cafile=certifi.where())
URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"

# universe tickers = columns of the cached FMP price matrix (SP1500-ish), fast to load
uni = set(pd.read_parquet("data/cached_close_prices.parquet").columns.astype(str))
print(f"[universe filter: {len(uni)} tickers]", flush=True)

t0 = time.time()
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
rows = []
kept = 0
with urllib.request.urlopen(req, context=CTX, timeout=120) as resp:
    text = io.TextIOWrapper(resp, encoding="utf-8", errors="replace", newline="")
    rdr = csv.reader(text)
    header = next(rdr)
    di = {c: i for i, c in enumerate(header)}
    iD, iT, iS = di["Date"], di["Article_title"], di["Stock_symbol"]
    n = 0
    for row in rdr:
        n += 1
        if n % 2_000_000 == 0:
            print(f"  scanned {n/1e6:.0f}M rows, kept {kept} ({time.time()-t0:.0f}s)", flush=True)
        if len(row) <= iS:
            continue
        sym = row[iS]
        if sym not in uni:
            continue
        d = row[iD][:10]
        if d < "2016-01-01" or d > "2023-12-31":
            continue
        title = row[iT]
        if not title:
            continue
        rows.append((d, sym, title))
        kept += 1
df = pd.DataFrame(rows, columns=["date", "symbol", "title"])
df["date"] = pd.to_datetime(df["date"])
df.to_parquet("research/_fnspid_headlines.parquet")
print(f"\n[DONE: {len(df)} headlines, {df['symbol'].nunique()} tickers, "
      f"{df['date'].min().date()}..{df['date'].max().date()}, {time.time()-t0:.0f}s]", flush=True)
