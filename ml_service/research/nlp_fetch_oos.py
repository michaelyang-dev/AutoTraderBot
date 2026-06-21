"""
Fetch a TRUE OOS news slice (2020-2023) from FNSPID nasdaq_exteral_data.csv (23GB,
full-text; we keep only date/title/symbol). Same universe filter as the 2016-2020
in-sample. -> research/_fnspid_oos.parquet for an out-of-sample news-momentum test.
"""
import ssl, urllib.request, certifi, csv, io, time, sys
import pandas as pd

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # full-text Article fields are huge
CTX = ssl.create_default_context(cafile=certifi.where())
URL = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
uni = set(pd.read_parquet("data/cached_close_prices.parquet").columns.astype(str))
print(f"[universe {len(uni)} tickers; streaming 23GB nasdaq file, keeping 2020-2023]", flush=True)

t0 = time.time()
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
rows, kept, n = [], 0, 0
with urllib.request.urlopen(req, context=CTX, timeout=180) as resp:
    text = io.TextIOWrapper(resp, encoding="utf-8", errors="replace", newline="")
    rdr = csv.reader(text)
    header = next(rdr)
    di = {c: i for i, c in enumerate(header)}
    iD, iT, iS = di["Date"], di["Article_title"], di["Stock_symbol"]
    for row in rdr:
        n += 1
        if n % 2_000_000 == 0:
            print(f"  scanned {n/1e6:.0f}M, kept {kept} ({time.time()-t0:.0f}s)", flush=True)
        if len(row) <= max(iD, iT, iS):
            continue
        sym = row[iS]
        if sym not in uni:
            continue
        d = row[iD][:10]
        if d < "2020-01-01" or d > "2023-12-31":
            continue
        title = row[iT]
        if title:
            rows.append((d, sym, title)); kept += 1
df = pd.DataFrame(rows, columns=["date", "symbol", "title"]).drop_duplicates()
df["date"] = pd.to_datetime(df["date"])
df.to_parquet("research/_fnspid_oos.parquet")
print(f"\n[DONE: {len(df)} headlines, {df['symbol'].nunique()} tickers, "
      f"{df['date'].min().date()}..{df['date'].max().date()}, {time.time()-t0:.0f}s]", flush=True)
