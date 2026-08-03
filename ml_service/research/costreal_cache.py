"""Shared cache builder for the REALISTIC transaction-cost study (costreal_*).

Builds two gitignored caches under research/ from the WRDS parquets:
  _costreal_liq.parquet  — CRSP daily Ticker/PERMNO/date/close/vol/$vol/bid/ask, 2017+
  _costreal_mem.pkl      — PIT sp500_mem / sp400_mem / sp600_mem from the SP1500 pickle

Regenerating takes ~20s. Both files are large (~440MB) — do NOT commit them.
"""
import os
import pickle

SP = os.path.dirname(os.path.abspath(__file__)) + "/_costreal_"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LIQ = SP + "liq.parquet"
_MEM = SP + "mem.pkl"


def ensure_cache():
    import pandas as pd
    if not os.path.exists(_LIQ):
        print("[costreal_cache] building liquidity cache from CRSP ...", flush=True)
        d = pd.read_parquet(
            os.path.join(ML, "data/wrds/crsp_daily_stock_full.parquet"),
            columns=["Ticker", "PERMNO", "YYYYMMDD", "DlyClose", "DlyVol",
                     "DlyPrcVol", "DlyBid", "DlyAsk"],
            filters=[("YYYYMMDD", ">=", 20170101)])
        d.to_parquet(_LIQ)
        print(f"[costreal_cache]   {len(d):,} rows -> {_LIQ}", flush=True)
    if not os.path.exists(_MEM):
        print("[costreal_cache] building membership cache ...", flush=True)
        u = pickle.load(open(os.path.join(ML, "data/wrds/complete_sp1500_universe.pkl"), "rb"))
        pickle.dump({k: u[k] for k in ("sp500_mem", "sp400_mem", "sp600_mem")},
                    open(_MEM, "wb"))
        del u
    return SP
