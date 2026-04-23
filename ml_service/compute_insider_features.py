#!/usr/bin/env python3
"""
Compute 5 insider-trade features from FMP altdata cache,
merge into existing features.parquet, and save as features_v5.parquet.

Features (all keyed on filingDate for point-in-time integrity):
  1. insider_net_buy_count_30d   — #buys minus #sells in trailing 30 calendar days
  2. insider_cluster_score_10d   — #distinct insiders who filed in trailing 10 days
  3. insider_buy_dollar_30d      — log1p of total buy dollar value in trailing 30 days
  4. insider_ceo_buy_90d         — binary: CEO filed a purchase in trailing 90 days
  5. insider_director_buy_90d    — binary: any director filed a purchase in trailing 90 days

Usage:
    python3 compute_insider_features.py          # compute and save
    python3 compute_insider_features.py --stats  # also print feature statistics
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "altdata_cache"
FEATURES_IN = DATA_DIR / "features.parquet"
FEATURES_OUT = DATA_DIR / "features_v5.parquet"


def _is_ceo(owner_type: str) -> bool:
    """Check if typeOfOwner indicates CEO."""
    if not isinstance(owner_type, str):
        return False
    return "chief executive officer" in owner_type.lower()


def _is_director(owner_type: str) -> bool:
    """Check if typeOfOwner indicates a board director."""
    if not isinstance(owner_type, str):
        return False
    return owner_type.lower().startswith("director")


def compute_insider_features_for_symbol(symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute 5 insider features for one symbol across all dates.

    Uses filingDate as point-in-time availability date.
    Returns DataFrame indexed by dates with 5 feature columns.
    """
    cache_file = CACHE_DIR / f"{symbol}_insider.parquet"

    n = len(dates)
    result = pd.DataFrame({
        "insider_net_buy_count_30d": np.zeros(n, dtype=np.float32),
        "insider_cluster_score_10d": np.zeros(n, dtype=np.float32),
        "insider_buy_dollar_30d": np.zeros(n, dtype=np.float32),
        "insider_ceo_buy_90d": np.zeros(n, dtype=np.float32),
        "insider_director_buy_90d": np.zeros(n, dtype=np.float32),
    }, index=dates)

    if not cache_file.exists():
        return result

    try:
        ins = pd.read_parquet(cache_file)
    except Exception:
        return result

    if ins.empty or "filingDate" not in ins.columns:
        return result

    ins = ins.dropna(subset=["filingDate"]).sort_values("filingDate")
    ins["filingDate"] = pd.to_datetime(ins["filingDate"])

    # Pre-compute per-transaction attributes
    is_buy = (ins["transactionType"] == "P-Purchase").values
    is_sell = (ins["transactionType"] == "S-Sale").values
    filing_dates = ins["filingDate"].values  # numpy datetime64
    prices = ins["price"].values.astype(np.float64)
    shares = ins["securitiesTransacted"].values.astype(np.float64)
    dollar_values = prices * shares

    # Owner type flags
    owner_types = ins["typeOfOwner"].values
    is_ceo_arr = np.array([_is_ceo(o) for o in owner_types])
    is_dir_arr = np.array([_is_director(o) for o in owner_types])

    # Reporting names for cluster score
    reporter_names = ins["reportingName"].values

    date_vals = dates.values  # numpy datetime64

    # Vectorized: for each date, compute features using boolean masking
    # Use numpy timedelta for window lookbacks
    td_30 = np.timedelta64(30, "D")
    td_10 = np.timedelta64(10, "D")
    td_90 = np.timedelta64(90, "D")
    td_1 = np.timedelta64(1, "D")

    net_buy_count = np.zeros(n, dtype=np.float32)
    cluster_score = np.zeros(n, dtype=np.float32)
    buy_dollar = np.zeros(n, dtype=np.float32)
    ceo_buy = np.zeros(n, dtype=np.float32)
    director_buy = np.zeros(n, dtype=np.float32)

    for d in range(n):
        dt = date_vals[d]
        # Point-in-time: only use filings strictly before this date
        # (filingDate < date, i.e. available by market close the day before)
        cutoff = dt - td_1

        # 30-day window for net_buy_count and buy_dollar
        mask_30 = (filing_dates >= (cutoff - td_30)) & (filing_dates <= cutoff)
        if mask_30.any():
            buys_30 = (is_buy & mask_30).sum()
            sells_30 = (is_sell & mask_30).sum()
            net_buy_count[d] = buys_30 - sells_30

            buy_mask_30 = is_buy & mask_30
            if buy_mask_30.any():
                buy_dollar[d] = np.log1p(dollar_values[buy_mask_30].sum())

        # 10-day window for cluster score
        mask_10 = (filing_dates >= (cutoff - td_10)) & (filing_dates <= cutoff)
        if mask_10.any():
            cluster_score[d] = len(set(reporter_names[mask_10]))

        # 90-day window for CEO buy and director buy
        mask_90 = (filing_dates >= (cutoff - td_90)) & (filing_dates <= cutoff)
        if mask_90.any():
            ceo_buy_mask = is_buy & mask_90 & is_ceo_arr
            if ceo_buy_mask.any():
                ceo_buy[d] = 1.0

            dir_buy_mask = is_buy & mask_90 & is_dir_arr
            if dir_buy_mask.any():
                director_buy[d] = 1.0

    result["insider_net_buy_count_30d"] = net_buy_count
    result["insider_cluster_score_10d"] = cluster_score
    result["insider_buy_dollar_30d"] = buy_dollar
    result["insider_ceo_buy_90d"] = ceo_buy
    result["insider_director_buy_90d"] = director_buy

    return result


def main():
    t0 = time.perf_counter()
    show_stats = "--stats" in sys.argv

    print("=" * 60)
    print("Compute Insider Features (v5)")
    print("=" * 60)

    # Load existing features
    if not FEATURES_IN.exists():
        sys.exit(f"ERROR: {FEATURES_IN} not found. Run data_pipeline.py first.")

    print(f"Loading {FEATURES_IN} ...")
    master = pd.read_parquet(FEATURES_IN)
    master["date"] = pd.to_datetime(master["date"])
    print(f"  Shape: {master.shape}")

    symbols = sorted(master["symbol"].unique())
    all_dates = pd.DatetimeIndex(sorted(master["date"].unique()))
    print(f"  Symbols: {len(symbols)} | Dates: {len(all_dates)}")

    # Check which symbols have insider cache files
    cached_symbols = set()
    for sym in symbols:
        if (CACHE_DIR / f"{sym}_insider.parquet").exists():
            cached_symbols.add(sym)
    print(f"  Symbols with insider cache: {len(cached_symbols)}/{len(symbols)}")

    # Compute features per symbol
    print("\nComputing insider features ...")
    feature_frames = []
    symbols_with_data = 0

    for i, sym in enumerate(symbols, 1):
        # Get dates for this symbol from the master DataFrame
        sym_dates = pd.DatetimeIndex(
            master.loc[master["symbol"] == sym, "date"].values
        )

        feat = compute_insider_features_for_symbol(sym, sym_dates)

        # Check if any non-zero features (before adding string column)
        if sym in cached_symbols:
            nonzero = (feat.abs() > 0).any(axis=None)

        feat["symbol"] = sym
        feat.index.name = "date"
        feature_frames.append(feat.reset_index())

        if sym in cached_symbols:
            if nonzero:
                symbols_with_data += 1

        if i % 50 == 0 or i == len(symbols):
            print(f"  [{i}/{len(symbols)}] processed ({symbols_with_data} with non-zero features)")

    # Combine all feature frames
    print("\nMerging features ...")
    insider_df = pd.concat(feature_frames, ignore_index=True)
    insider_df["date"] = pd.to_datetime(insider_df["date"])

    # Merge into master on (date, symbol)
    new_cols = [
        "insider_net_buy_count_30d",
        "insider_cluster_score_10d",
        "insider_buy_dollar_30d",
        "insider_ceo_buy_90d",
        "insider_director_buy_90d",
    ]

    # Drop old insider columns if they exist in master (from fundamentals pipeline)
    # Keep the old ones — they use different data sources. Just add new columns.
    for col in new_cols:
        if col in master.columns:
            master = master.drop(columns=[col])

    master = master.merge(
        insider_df[["date", "symbol"] + new_cols],
        on=["date", "symbol"],
        how="left",
    )

    # Fill NaN with 0 for these features (no insider activity = 0)
    for col in new_cols:
        master[col] = master[col].fillna(0)

    print(f"\nFinal shape: {master.shape}")

    # Save
    print(f"Saving to {FEATURES_OUT} ...")
    master.to_parquet(FEATURES_OUT, index=False, engine="pyarrow", compression="snappy")
    print(f"  File size: {FEATURES_OUT.stat().st_size / 1_048_576:.1f} MB")

    # Feature statistics
    print(f"\n{'='*60}")
    print("Feature Statistics")
    print(f"{'='*60}")
    for col in new_cols:
        s = master[col]
        nonzero_pct = (s != 0).mean() * 100
        print(f"\n  {col}:")
        print(f"    mean={s.mean():.4f}  std={s.std():.4f}  "
              f"min={s.min():.1f}  max={s.max():.1f}")
        print(f"    non-zero: {nonzero_pct:.2f}%  ({(s != 0).sum():,} / {len(s):,})")
        if col in ("insider_ceo_buy_90d", "insider_director_buy_90d"):
            print(f"    positive: {(s > 0).sum():,} rows ({(s > 0).mean()*100:.2f}%)")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
