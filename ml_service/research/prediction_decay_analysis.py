#!/usr/bin/env python3
"""
Prediction Decay Analysis
==========================
For every LambdaRank prediction across the 11-year walk-forward:
  1. Compute realized return at horizons 1, 2, 3, 5, 7, 10, 15, 20 days
  2. Bucket predictions into score deciles
  3. For each decile, compute mean forward return at each horizon
  4. Compute Information Coefficient (Spearman rank-corr between score and
     realized return) at each horizon
  5. Identify the alpha-peak horizon
  6. Run backtests at the optimal hold period vs current 10-day

Also tests whether the current 10-day hold is optimal by running backtests
at each candidate hold period.

Run:
    cd ml_service && python3 research/prediction_decay_analysis.py
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WF_DIR = DATA_DIR / "walkforward"

INITIAL_CASH = 100_000.0
HORIZONS = [1, 2, 3, 5, 7, 10, 15, 20]
N_DECILES = 10
EW_TOP_N = 5
SW_TOP_N = 10
MAX_SINGLE_NAME_PCT = 0.25
MAX_SECTOR_PCT = 0.35
MAX_BETA = 1.2


def log(msg: str):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Decay Analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_forward_returns(preds: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each prediction row, compute realized forward returns at multiple horizons.
    """
    preds = preds.copy()
    preds = preds[preds["in_sp500"] == True]

    # Get trading dates
    trading_dates = sorted(prices.index.tolist())
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    results = []
    # Process by date for efficiency
    unique_dates = sorted(preds["date"].unique())

    for date in unique_dates:
        if date not in date_to_idx:
            continue
        idx = date_to_idx[date]
        day_preds = preds[preds["date"] == date]

        for _, row in day_preds.iterrows():
            sym = row["symbol"]
            score = row["prob_ensemble"]

            if sym not in prices.columns:
                continue

            entry_px = prices.loc[date].get(sym)
            if entry_px is None or pd.isna(entry_px) or entry_px <= 0:
                continue

            fwd_rets = {"date": date, "symbol": sym, "score": score}

            for h in HORIZONS:
                future_idx = idx + h
                if future_idx >= len(trading_dates):
                    fwd_rets[f"ret_{h}d"] = np.nan
                    continue
                future_date = trading_dates[future_idx]
                future_px = prices.loc[future_date].get(sym)
                if future_px is None or pd.isna(future_px):
                    fwd_rets[f"ret_{h}d"] = np.nan
                else:
                    fwd_rets[f"ret_{h}d"] = (future_px / entry_px) - 1.0

            results.append(fwd_rets)

    return pd.DataFrame(results)


def analyze_decay(decay_df: pd.DataFrame):
    """
    Compute:
    1. Mean return by score decile at each horizon
    2. Top-decile excess return (vs median decile) at each horizon
    3. Information Coefficient at each horizon
    """
    # Assign score deciles
    decay_df["score_decile"] = pd.qcut(decay_df["score"], N_DECILES,
                                        labels=False, duplicates="drop")

    # Mean return by decile × horizon
    decile_returns = {}
    for h in HORIZONS:
        col = f"ret_{h}d"
        if col not in decay_df.columns:
            continue
        valid = decay_df.dropna(subset=[col])
        grouped = valid.groupby("score_decile")[col].mean()
        decile_returns[h] = grouped

    # Top decile (9) excess return over median decile (4-5 avg)
    top_excess = {}
    for h, grouped in decile_returns.items():
        top = grouped.get(N_DECILES - 1, np.nan)
        mid = (grouped.get(4, 0) + grouped.get(5, 0)) / 2
        top_excess[h] = top - mid

    # Information Coefficient (Spearman rank correlation)
    ic_by_horizon = {}
    for h in HORIZONS:
        col = f"ret_{h}d"
        if col not in decay_df.columns:
            continue
        valid = decay_df.dropna(subset=[col, "score"])
        # Compute IC per date, then average (more robust)
        daily_ics = []
        for date, grp in valid.groupby("date"):
            if len(grp) < 20:
                continue
            corr, _ = sp_stats.spearmanr(grp["score"], grp[col])
            if not np.isnan(corr):
                daily_ics.append(corr)
        ic_by_horizon[h] = {
            "mean_ic": np.mean(daily_ics) if daily_ics else 0,
            "ic_std": np.std(daily_ics) if daily_ics else 0,
            "ic_ir": (np.mean(daily_ics) / np.std(daily_ics) * np.sqrt(252 / max(h, 1))
                      if daily_ics and np.std(daily_ics) > 0 else 0),
            "n_days": len(daily_ics),
        }

    return decile_returns, top_excess, ic_by_horizon


# ══════════════════════════════════════════════════════════════════════════════
#  Hold-Period Backtest
# ══════════════════════════════════════════════════════════════════════════════

def signal_weighted_allocation(candidates, sector_map, betas_today, n=10):
    top = candidates[:n]
    if not top: return {}
    scores = np.array([s for _, s, _, _ in top])
    vols = np.array([max(v, 0.05) for _, _, v, _ in top])
    excess = scores - scores.min() + 1e-6
    raw_w = excess / vols
    raw_w = raw_w / raw_w.sum()
    syms = [s for s, _, _, _ in top]
    weights = dict(zip(syms, raw_w))
    for _ in range(20):
        changed = False
        for sym in list(weights.keys()):
            if weights[sym] > MAX_SINGLE_NAME_PCT:
                weights[sym] = MAX_SINGLE_NAME_PCT; changed = True
        sector_totals = {}
        for sym, w in weights.items():
            sec = sector_map.get(sym, "Unknown")
            sector_totals[sec] = sector_totals.get(sec, 0) + w
        for sec, total in sector_totals.items():
            if total > MAX_SECTOR_PCT:
                scale = MAX_SECTOR_PCT / total
                for sym in list(weights.keys()):
                    if sector_map.get(sym, "Unknown") == sec:
                        weights[sym] *= scale
                changed = True
        port_beta = sum(weights[s] * betas_today.get(s, 1.0) for s in weights)
        if port_beta > MAX_BETA:
            scale = MAX_BETA / port_beta
            for sym in weights: weights[sym] *= scale
            changed = True
        total_w = sum(weights.values())
        if total_w > 1.0:
            for sym in weights: weights[sym] /= total_w
            changed = True
        if not changed: break
    return {s: w for s, w in weights.items() if w > 0.01}


def run_backtest_hold(preds, prices, sector_map, vol_matrix, betas,
                       hold_days, mode="equal_weight"):
    preds_sp500 = preds[preds["in_sp500"] == True].copy()
    rebal_dates = sorted(preds_sp500["date"].unique())
    rebal_dates_set = set(d for i, d in enumerate(rebal_dates) if i % hold_days == 0)

    trading_dates = sorted(prices.index)
    wf_start, wf_end = preds_sp500["date"].min(), preds_sp500["date"].max()
    trading_dates = [d for d in trading_dates if wf_start <= d <= wf_end]

    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    prev_value = INITIAL_CASH

    for date in trading_dates:
        day_prices = prices.loc[date] if date in prices.index else pd.Series(dtype=float)

        if date in rebal_dates_set:
            day_preds = preds_sp500[preds_sp500["date"] == date].sort_values(
                "prob_ensemble", ascending=False)
            if len(day_preds) == 0:
                port_values.append((date, prev_value)); continue

            vol_today = vol_matrix.loc[date] if date in vol_matrix.index else pd.Series(dtype=float)
            candidates = []
            for _, row in day_preds.iterrows():
                sym, score = row["symbol"], row["prob_ensemble"]
                v = vol_today.get(sym, 0.20) if len(vol_today) > 0 else 0.20
                if pd.isna(v) or v <= 0: v = 0.20
                candidates.append((sym, score, v, sector_map.get(sym, "Unknown")))

            betas_today = {}
            if date in betas.index:
                for sym, _, _, _ in candidates:
                    b = betas.loc[date].get(sym, 1.0)
                    betas_today[sym] = b if not pd.isna(b) else 1.0

            if mode == "equal_weight":
                top = candidates[:EW_TOP_N]
                tw = {s: 1.0/len(top) for s, _, _, _ in top} if top else {}
            else:
                tw = signal_weighted_allocation(candidates, sector_map, betas_today, n=SW_TOP_N)

            total_value = cash
            for sym, h in holdings.items():
                px = day_prices.get(sym, h.get("ep", 0))
                if pd.isna(px): px = h.get("ep", 0)
                total_value += h["sh"] * px
            cash = total_value
            holdings = {}
            for sym, w in tw.items():
                px = day_prices.get(sym)
                if px is None or pd.isna(px) or px <= 0: continue
                alloc = total_value * w
                holdings[sym] = {"sh": alloc / px, "ep": px}
                cash -= alloc

        total_value = cash
        for sym, h in holdings.items():
            px = day_prices.get(sym, h.get("ep", 0))
            if pd.isna(px): px = h.get("ep", 0)
            total_value += h["sh"] * px
        port_values.append((date, total_value))
        prev_value = total_value

    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0: years = 1.0
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd_s = dr[dr < 0].std()
    sortino = dr.mean() / dd_s * np.sqrt(252) if dd_s > 0 else np.nan
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    vol = dr.std() * np.sqrt(252)
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "vol": vol}


def run_per_year_hold(preds, prices, sector_map, vol_matrix, betas, hold_days, mode):
    results = []
    for year in range(2015, 2026):
        yp = preds[(preds["date"] >= f"{year}-01-01") & (preds["date"] <= f"{year}-12-31")]
        if len(yp) < 50: continue
        m = run_backtest_hold(yp, prices, sector_map, vol_matrix, betas, hold_days, mode)
        m["year"] = year
        results.append(m)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  PREDICTION DECAY ANALYSIS")
    log("  Finding the optimal alpha horizon for LambdaRank predictions")
    log("=" * 80)

    # Load data
    log("\n1. Loading data ...")
    lr_preds = pd.read_parquet(WF_DIR / "predictions_walkforward_lambdarank.parquet")
    lr_preds["date"] = pd.to_datetime(lr_preds["date"])

    prices = pd.read_parquet(DATA_DIR / "cache_prices_costmodel.parquet")
    prices.index = pd.to_datetime(prices.index)

    with open(DATA_DIR / "cache_sectors.json") as f:
        sector_map = json.load(f)

    feat_vol = pd.read_parquet(DATA_DIR / "features.parquet", columns=["date", "symbol", "vol_20d"])
    feat_vol["date"] = pd.to_datetime(feat_vol["date"])
    vol_matrix = feat_vol.pivot_table(index="date", columns="symbol", values="vol_20d", aggfunc="last")

    returns = prices.pct_change()
    spy_ret = returns["SPY"]
    spy_var = spy_ret.rolling(60, min_periods=30).var()
    xy = returns.multiply(spy_ret, axis=0)
    cov = xy.rolling(60, min_periods=30).mean() - returns.rolling(60, min_periods=30).mean().multiply(
        spy_ret.rolling(60, min_periods=30).mean(), axis=0)
    betas = cov.divide(spy_var, axis=0)
    betas["SPY"] = 1.0

    log(f"  Predictions: {len(lr_preds):,} rows")
    log(f"  Prices: {prices.shape}")

    # Compute forward returns at all horizons
    log("\n2. Computing forward returns at horizons {HORIZONS} ...")
    t_fwd = time.perf_counter()
    decay_df = compute_forward_returns(lr_preds, prices)
    log(f"  Computed {len(decay_df):,} prediction-horizon pairs in {time.perf_counter()-t_fwd:.0f}s")

    # Analyze decay
    log("\n3. Analyzing prediction decay ...")
    decile_returns, top_excess, ic_by_horizon = analyze_decay(decay_df)

    # Print IC table
    log(f"\n{'='*80}")
    log("  INFORMATION COEFFICIENT BY HORIZON")
    log(f"{'='*80}")
    log(f"\n  {'Horizon':>8} {'Mean IC':>8} {'IC Std':>7} {'IC IR':>7} {'Top-Mid':>9} {'N days':>7}")
    log(f"  {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*9} {'─'*7}")

    best_ic_horizon = 0
    best_ic = -999
    for h in HORIZONS:
        ic = ic_by_horizon.get(h, {})
        te = top_excess.get(h, 0)
        mean_ic = ic.get("mean_ic", 0)
        log(f"  {h:>5}d   {mean_ic:>8.4f} {ic.get('ic_std', 0):>7.4f} "
            f"{ic.get('ic_ir', 0):>7.2f} {te:>+8.2%} {ic.get('n_days', 0):>7}")
        if mean_ic > best_ic:
            best_ic = mean_ic
            best_ic_horizon = h

    log(f"\n  Peak IC at {best_ic_horizon}-day horizon (IC = {best_ic:.4f})")

    # Print decile return table
    log(f"\n{'='*80}")
    log("  MEAN RETURN BY SCORE DECILE × HORIZON")
    log(f"{'='*80}")

    header = f"  {'Decile':>6}"
    for h in HORIZONS:
        header += f" {h:>6}d"
    log(header)
    log(f"  {'─'*6}" + "".join(f" {'─'*6}" for _ in HORIZONS))

    for d in range(N_DECILES):
        line = f"  {d:>6}"
        for h in HORIZONS:
            ret = decile_returns.get(h, pd.Series()).get(d, np.nan)
            line += f" {ret:>+5.2%}" if not np.isnan(ret) else f" {'N/A':>6}"
        label = " ← bottom" if d == 0 else (" ← TOP" if d == N_DECILES - 1 else "")
        log(line + label)

    # Top-decile spread
    log(f"\n  Top-decile excess return (top minus middle):")
    for h in HORIZONS:
        te = top_excess.get(h, 0)
        marker = " ◀ PEAK" if h == best_ic_horizon else ""
        log(f"    {h:>3}d: {te:>+.2%}{marker}")

    # Hold period backtest
    log(f"\n{'='*80}")
    log("  HOLD PERIOD OPTIMIZATION (backtest at each horizon)")
    log(f"{'='*80}")

    test_holds = [3, 5, 7, 10, 15, 20]
    log(f"\n  Testing hold periods: {test_holds}")

    for mode_name, mode in [("EW", "equal_weight"), ("SW", "signal_weighted")]:
        log(f"\n  {mode_name} mode:")
        log(f"    {'Hold':>6} {'C.CAGR':>7} {'C.Shp':>6} {'C.Sort':>7} {'C.DD':>6} {'C.Vol':>6}"
            f" {'M.CAGR':>7} {'M.Shp':>6}")
        log(f"    {'─'*6} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*6}")

        best_sharpe = -999
        best_hold = 10

        for hold in test_holds:
            cont = run_backtest_hold(lr_preds, prices, sector_map, vol_matrix,
                                      betas, hold, mode)
            yearly = run_per_year_hold(lr_preds, prices, sector_map, vol_matrix,
                                        betas, hold, mode)
            med_cagr = np.median([y["cagr"] for y in yearly])
            med_sharpe = np.median([y["sharpe"] for y in yearly])

            marker = ""
            if cont["sharpe"] > best_sharpe:
                best_sharpe = cont["sharpe"]
                best_hold = hold

            if hold == 10:
                marker = " ← current"

            log(f"    {hold:>4}d {cont['cagr']:>+6.1%} {cont['sharpe']:>6.2f} "
                f"{cont['sortino']:>7.2f} {cont['max_dd']:>5.1%} {cont['vol']:>5.1%}"
                f" {med_cagr:>+6.1%} {med_sharpe:>6.2f}{marker}")

        log(f"    Best continuous Sharpe: {best_hold}d ({best_sharpe:.2f})")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
