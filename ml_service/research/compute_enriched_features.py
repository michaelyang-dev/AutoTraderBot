#!/usr/bin/env python3
"""
Compute Enriched Features
=========================
Loads existing features.parquet, adds new features from FMP fundamental data
that is already fetched but not fully utilized, and saves to
features_enriched.parquet.

New features (all point-in-time correct using reportedDate):
  1. earnings_beat_streak    — consecutive quarters beating EPS estimate
  2. revenue_surprise_last   — most recent revenue surprise %
  3. earnings_surprise_3q_avg — avg EPS surprise over last 3 reported quarters
  4. revenue_surprise_3q_avg  — avg revenue surprise over last 3 reported quarters
  5. insider_buy_cluster      — binary: 3+ insider buys in last 90 days
  6. insider_net_shares_180d  — net insider shares over 180-day window

Does NOT overwrite features.parquet.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "features.parquet"
OUTPUT_FILE = DATA_DIR / "features_enriched.parquet"

EARNINGS_FILE = DATA_DIR / "fundamentals_earnings.parquet"
INSIDERS_FILE = DATA_DIR / "fundamentals_insiders.parquet"


def compute_earnings_features(features_df):
    """Add earnings beat streak, revenue surprise, and multi-quarter averages."""
    print("[Enrichment] Computing earnings features ...")

    if not EARNINGS_FILE.exists():
        print("  WARNING: fundamentals_earnings.parquet not found, skipping")
        features_df["earnings_beat_streak"] = np.nan
        features_df["revenue_surprise_last"] = np.nan
        features_df["earnings_surprise_3q_avg"] = np.nan
        features_df["revenue_surprise_3q_avg"] = np.nan
        return features_df

    earn = pd.read_parquet(EARNINGS_FILE)
    earn["date"] = pd.to_datetime(earn["date"])

    # Only use rows where actual is reported (not future estimates)
    earn = earn[earn["eps_actual"].notna()].copy()
    earn = earn.sort_values(["symbol", "date"])

    # Pre-compute per-symbol earnings data
    sym_earn_data = {}
    for sym, grp in earn.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # EPS surprise per quarter
        eps_surprises = []
        for _, row in grp.iterrows():
            est = row.get("eps_estimated")
            actual = row["eps_actual"]
            if pd.notna(est) and abs(est) > 1e-9:
                eps_surprises.append({
                    "date": row["date"],
                    "surprise": (actual - est) / abs(est),
                    "beat": 1 if actual > est else 0,
                })
            else:
                eps_surprises.append({
                    "date": row["date"],
                    "surprise": np.nan,
                    "beat": np.nan,
                })

        # Revenue surprise per quarter
        rev_surprises = []
        for _, row in grp.iterrows():
            rev_actual = row.get("revenue_actual")
            rev_est = row.get("revenue_estimated")
            if pd.notna(rev_actual) and pd.notna(rev_est) and abs(rev_est) > 1e-6:
                rev_surprises.append({
                    "date": row["date"],
                    "surprise": (rev_actual - rev_est) / abs(rev_est),
                })
            else:
                rev_surprises.append({
                    "date": row["date"],
                    "surprise": np.nan,
                })

        sym_earn_data[sym] = {
            "dates": np.array([e["date"] for e in eps_surprises]),
            "eps_surprises": np.array([e["surprise"] for e in eps_surprises]),
            "eps_beats": np.array([e["beat"] for e in eps_surprises]),
            "rev_surprises": np.array([e["surprise"] for e in rev_surprises]),
        }

    # Now map to daily features
    n = len(features_df)
    beat_streak = np.full(n, np.nan)
    rev_surp_last = np.full(n, np.nan)
    eps_surp_3q = np.full(n, np.nan)
    rev_surp_3q = np.full(n, np.nan)

    symbols = features_df["symbol"].values
    dates = features_df["date"].values

    # Group by symbol for vectorized processing
    sym_groups = features_df.groupby("symbol").indices

    for sym, indices in sym_groups.items():
        if sym not in sym_earn_data:
            continue

        edata = sym_earn_data[sym]
        earn_dates = edata["dates"]
        eps_surps = edata["eps_surprises"]
        eps_beats = edata["eps_beats"]
        rev_surps = edata["rev_surprises"]

        if len(earn_dates) == 0:
            continue

        feat_dates = dates[indices]

        for i, idx in enumerate(indices):
            dt = feat_dates[i]
            # Find most recent earnings with date <= dt
            e_idx = np.searchsorted(earn_dates, dt, side="right") - 1
            if e_idx < 0:
                continue

            # Revenue surprise (most recent)
            rev_surp_last[idx] = np.clip(rev_surps[e_idx], -2.0, 2.0)

            # Earnings beat streak: count consecutive beats going backwards
            streak = 0
            for j in range(e_idx, -1, -1):
                if np.isnan(eps_beats[j]):
                    break
                if eps_beats[j] == 1:
                    streak += 1
                else:
                    break
            beat_streak[idx] = streak

            # 3-quarter average EPS surprise
            start_q = max(0, e_idx - 2)
            recent_eps = eps_surps[start_q:e_idx + 1]
            valid_eps = recent_eps[~np.isnan(recent_eps)]
            if len(valid_eps) > 0:
                eps_surp_3q[idx] = np.clip(np.mean(valid_eps), -2.0, 2.0)

            # 3-quarter average revenue surprise
            recent_rev = rev_surps[start_q:e_idx + 1]
            valid_rev = recent_rev[~np.isnan(recent_rev)]
            if len(valid_rev) > 0:
                rev_surp_3q[idx] = np.clip(np.mean(valid_rev), -2.0, 2.0)

    features_df["earnings_beat_streak"] = beat_streak
    features_df["revenue_surprise_last"] = rev_surp_last
    features_df["earnings_surprise_3q_avg"] = eps_surp_3q
    features_df["revenue_surprise_3q_avg"] = rev_surp_3q

    # Coverage stats
    for col in ["earnings_beat_streak", "revenue_surprise_last",
                "earnings_surprise_3q_avg", "revenue_surprise_3q_avg"]:
        pct = features_df[col].notna().mean() * 100
        print(f"  {col}: {pct:.1f}% coverage")

    return features_df


def compute_insider_features(features_df):
    """Add insider buy clustering and 180-day net shares."""
    print("[Enrichment] Computing insider features ...")

    if not INSIDERS_FILE.exists():
        print("  WARNING: fundamentals_insiders.parquet not found, skipping")
        features_df["insider_buy_cluster"] = np.nan
        features_df["insider_net_shares_180d"] = np.nan
        return features_df

    ins = pd.read_parquet(INSIDERS_FILE)
    ins["date"] = pd.to_datetime(ins["date"])
    ins = ins.sort_values(["symbol", "date"])

    # Pre-index insider data by symbol
    sym_ins_data = {}
    for sym, grp in ins.groupby("symbol"):
        grp = grp.sort_values("date")
        sym_ins_data[sym] = {
            "dates": grp["date"].values,
            "is_buy": grp["is_buy"].values,
            "shares": grp["shares"].values,
        }

    n = len(features_df)
    buy_cluster = np.full(n, np.nan)
    net_shares_180 = np.full(n, np.nan)

    symbols = features_df["symbol"].values
    dates = features_df["date"].values

    sym_groups = features_df.groupby("symbol").indices

    for sym, indices in sym_groups.items():
        if sym not in sym_ins_data:
            continue

        idata = sym_ins_data[sym]
        ins_dates = idata["dates"]
        is_buy = idata["is_buy"]
        shares = idata["shares"]

        if len(ins_dates) == 0:
            continue

        feat_dates = dates[indices]

        for i, idx in enumerate(indices):
            dt = feat_dates[i]

            # 90-day lookback for buy clustering
            cutoff_90 = dt - np.timedelta64(90, "D")
            mask_90 = (ins_dates >= cutoff_90) & (ins_dates <= dt)
            if mask_90.any():
                n_buys_90 = is_buy[mask_90].sum()
                buy_cluster[idx] = 1.0 if n_buys_90 >= 3 else 0.0

            # 180-day lookback for net shares
            cutoff_180 = dt - np.timedelta64(180, "D")
            mask_180 = (ins_dates >= cutoff_180) & (ins_dates <= dt)
            if mask_180.any():
                buy_mask = mask_180 & (is_buy == 1)
                sell_mask = mask_180 & (is_buy == 0)
                buy_shares = shares[buy_mask].sum() if buy_mask.any() else 0
                sell_shares = shares[sell_mask].sum() if sell_mask.any() else 0
                net_shares_180[idx] = buy_shares - sell_shares

    features_df["insider_buy_cluster"] = buy_cluster
    features_df["insider_net_shares_180d"] = net_shares_180

    for col in ["insider_buy_cluster", "insider_net_shares_180d"]:
        pct = features_df[col].notna().mean() * 100
        print(f"  {col}: {pct:.1f}% coverage")

    return features_df


def main():
    print("=" * 60)
    print("Feature Enrichment Pipeline")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"ERROR: {INPUT_FILE} not found")
        sys.exit(1)

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Shape: {df.shape}")
    orig_cols = set(df.columns)

    t0 = time.time()

    df = compute_earnings_features(df)
    df = compute_insider_features(df)

    new_cols = sorted(set(df.columns) - orig_cols)
    print(f"\n[Enrichment] Added {len(new_cols)} new features: {new_cols}")
    print(f"[Enrichment] Final shape: {df.shape}")
    print(f"[Enrichment] Time: {time.time() - t0:.0f}s")

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"[Enrichment] Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
