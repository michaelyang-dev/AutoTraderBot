"""
RUN THIS ON THE WINDOWS MACHINE that has Norgate Data Updater (NDU) running.
  pip install norgatedata pandas pyarrow
  python norgate_export.py

Exports EVERY futures/commodities database Norgate gives you (Continuous Futures,
Cash Commodities, and the individual-contract Futures DB if present) — full history,
all OHLCV columns — one parquet + one meta csv per database. Small/important ones
(continuous, cash) are done first; the big individual-contracts DB (if any) runs last.

Hand back every norgate_*.parquet and norgate_*_meta.csv this writes.
"""
import os
import norgatedata
import pandas as pd

OUT = os.getcwd()                       # files land in the folder you run this from
avail = list(norgatedata.databases())
print("Databases available:")
for d in avail:
    print("   ", d)

# auto-pick every futures/commodities database (no name-guessing); small ones first
targets = [d for d in avail if ("futures" in d.lower() or "commodit" in d.lower())]


def _prio(name):
    n = name.lower()
    if "continuous" in n:                # the must-have, small
        return 0
    if "cash" in n or "commodit" in n:   # small
        return 1
    return 2                             # individual contracts (big) last


targets.sort(key=_prio)
print("\nWill export:", targets)


def export_db(db):
    syms = norgatedata.database_symbols(db)
    print(f"\n=== {db}: {len(syms)} symbols ===")
    if len(syms) > 1000:
        print("   (large database — may take several minutes and a fair bit of RAM)")
    frames, meta = [], []
    for i, s in enumerate(syms, 1):
        try:
            df = norgatedata.price_timeseries(s, timeseriesformat="pandas-dataframe")
            if df is not None and len(df):
                df = df.copy()
                df.insert(0, "symbol", s)
                df.index.name = "date"
                frames.append(df.reset_index())
                try:
                    nm = norgatedata.security_name(s)
                except Exception:
                    nm = ""
                meta.append((s, nm, str(df.index.min().date()),
                             str(df.index.max().date()), len(df)))
        except Exception as e:
            print("  skip", s, str(e)[:60])
        if i % 50 == 0:
            print(f"   ...{i}/{len(syms)}")
    if not frames:
        print("   no data."); return
    out = pd.concat(frames, ignore_index=True)
    tag = db.lower().replace(" ", "_")
    out.to_parquet(os.path.join(OUT, f"norgate_{tag}.parquet"))
    pd.DataFrame(meta, columns=["symbol", "name", "start", "end", "rows"]).to_csv(
        os.path.join(OUT, f"norgate_{tag}_meta.csv"), index=False)
    print(f"   WROTE norgate_{tag}.parquet  ({len(out):,} rows, "
          f"{out['symbol'].nunique()} symbols)  + norgate_{tag}_meta.csv")


for db in targets:
    export_db(db)

print("\nDONE. Send me every norgate_*.parquet and norgate_*_meta.csv in this folder.")
