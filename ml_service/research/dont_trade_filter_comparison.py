#!/usr/bin/env python3
"""
Don't-Trade Filter Comparison
==============================
Tests post-ranking filters on LambdaRank predictions:

  1. Earnings filter: exclude stocks with earnings in next 2 trading days
  2. Liquidity filter: exclude bottom quintile by 20-day dollar volume
  3. Spread proxy: exclude top 5% by vol_20d (high vol ≈ wide spread)

4 combinations (continuous + per-year):
  1. LR + EW (no filters)
  2. LR + EW + filters
  3. LR + SW (no filters)
  4. LR + SW + filters

Also tests filters + HMM (best config: 3st/min50%/lin) for 2 more combos.

Run:
    cd ml_service && python3 research/dont_trade_filter_comparison.py
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WF_DIR = DATA_DIR / "walkforward"

INITIAL_CASH = 100_000.0
HOLD_DAYS = 10

EW_TOP_N = 5
SW_TOP_N = 10
MAX_SINGLE_NAME_PCT = 0.25
MAX_SECTOR_PCT = 0.35
MAX_BETA = 1.2

# Filter params
EARNINGS_EXCLUSION_DAYS = 2  # exclude if earnings within N trading days
LIQUIDITY_BOTTOM_QUINTILE = 0.20  # exclude bottom 20% by dollar volume
VOL_TOP_PERCENTILE = 0.95  # exclude top 5% by vol (spread proxy)


def log(msg: str):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Build Earnings Calendar
# ══════════════════════════════════════════════════════════════════════════════

def build_earnings_blackout(earnings_df: pd.DataFrame,
                             trading_dates: list) -> dict:
    """
    Build {date: set(symbols)} of stocks that should NOT be traded on each date
    because they have earnings within EARNINGS_EXCLUSION_DAYS trading days.
    """
    earnings_df = earnings_df.copy()
    earnings_df["date"] = pd.to_datetime(earnings_df["date"])

    # Build a set of (symbol, earnings_date) pairs
    earnings_by_sym = {}
    for _, row in earnings_df[["symbol", "date"]].iterrows():
        sym = row["symbol"]
        if sym not in earnings_by_sym:
            earnings_by_sym[sym] = []
        earnings_by_sym[sym].append(row["date"])

    # Sort trading dates
    td = sorted(trading_dates)
    td_set = set(td)

    blackout = {d: set() for d in td}

    for sym, edates in earnings_by_sym.items():
        for edate in edates:
            # Block EARNINGS_EXCLUSION_DAYS trading days before earnings
            # Find trading days just before and on earnings date
            for i, td_date in enumerate(td):
                if td_date > edate:
                    break
                # Check if this trading day is within N days before earnings
                days_until = (edate - td_date).days
                if 0 <= days_until <= EARNINGS_EXCLUSION_DAYS * 2:  # calendar days buffer
                    # Count actual trading days between
                    trading_days_between = sum(1 for d in td if td_date <= d < edate)
                    if trading_days_between <= EARNINGS_EXCLUSION_DAYS:
                        blackout[td_date].add(sym)

    return blackout


def build_earnings_blackout_fast(earnings_df: pd.DataFrame,
                                  trading_dates: list) -> dict:
    """
    Fast version: for each trading date, find stocks with earnings in next
    N trading days.
    """
    earnings_df = earnings_df.copy()
    earnings_df["date"] = pd.to_datetime(earnings_df["date"])

    td = sorted(trading_dates)
    td_arr = np.array(td, dtype="datetime64[ns]")

    # Build earnings lookup: symbol -> sorted list of earnings dates
    earnings_by_sym = {}
    for sym, grp in earnings_df.groupby("symbol"):
        earnings_by_sym[sym] = sorted(grp["date"].tolist())

    blackout = {d: set() for d in td}

    for sym, edates in earnings_by_sym.items():
        edate_arr = np.array(edates, dtype="datetime64[ns]")
        for edate in edate_arr:
            # Find trading dates within EXCLUSION_DAYS before this earnings
            # Use calendar day approximation: 2 trading days ≈ 3-4 calendar days
            window_start = edate - np.timedelta64(EARNINGS_EXCLUSION_DAYS * 2 + 1, "D")
            mask = (td_arr >= window_start) & (td_arr <= edate)
            blocked_dates = td_arr[mask]
            # Keep only the last EARNINGS_EXCLUSION_DAYS trading dates before earnings
            if len(blocked_dates) > EARNINGS_EXCLUSION_DAYS + 1:
                blocked_dates = blocked_dates[-(EARNINGS_EXCLUSION_DAYS + 1):]
            for bd in blocked_dates:
                bd_ts = pd.Timestamp(bd)
                if bd_ts in blackout:
                    blackout[bd_ts].add(sym)

    return blackout


# ══════════════════════════════════════════════════════════════════════════════
#  Liquidity & Spread Filters
# ══════════════════════════════════════════════════════════════════════════════

def build_liquidity_blackout(prices: pd.DataFrame, volume: pd.DataFrame,
                              trading_dates: list) -> dict:
    """
    For each date, find stocks in the bottom quintile of 20-day avg dollar volume.
    """
    dollar_vol = (prices * volume).rolling(20, min_periods=10).mean()

    blackout = {}
    for date in trading_dates:
        if date not in dollar_vol.index:
            blackout[date] = set()
            continue
        dv = dollar_vol.loc[date].dropna()
        if len(dv) < 20:
            blackout[date] = set()
            continue
        threshold = dv.quantile(LIQUIDITY_BOTTOM_QUINTILE)
        illiquid = set(dv[dv <= threshold].index.tolist())
        blackout[date] = illiquid

    return blackout


def build_vol_blackout(vol_matrix: pd.DataFrame, trading_dates: list) -> dict:
    """
    For each date, find stocks in the top 5% by vol_20d (spread proxy).
    """
    blackout = {}
    for date in trading_dates:
        if date not in vol_matrix.index:
            blackout[date] = set()
            continue
        v = vol_matrix.loc[date].dropna()
        if len(v) < 20:
            blackout[date] = set()
            continue
        threshold = v.quantile(VOL_TOP_PERCENTILE)
        high_vol = set(v[v >= threshold].index.tolist())
        blackout[date] = high_vol

    return blackout


# ══════════════════════════════════════════════════════════════════════════════
#  Position Sizing
# ══════════════════════════════════════════════════════════════════════════════

def equal_weight_allocation(candidates, n=5):
    top = candidates[:n]
    if not top: return {}
    w = 1.0 / len(top)
    return {sym: w for sym, _, _, _ in top}


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


# ══════════════════════════════════════════════════════════════════════════════
#  Backtester
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(preds, prices, sector_map, vol_matrix, betas,
                 mode="equal_weight",
                 earnings_blackout=None, liquidity_blackout=None,
                 vol_blackout=None, hmm_exposures=None):
    preds_sp500 = preds[preds["in_sp500"] == True].copy()
    rebal_dates = sorted(preds_sp500["date"].unique())
    rebal_dates_set = set(d for i, d in enumerate(rebal_dates) if i % HOLD_DAYS == 0)

    trading_dates = sorted(prices.index)
    wf_start, wf_end = preds_sp500["date"].min(), preds_sp500["date"].max()
    trading_dates = [d for d in trading_dates if wf_start <= d <= wf_end]

    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    prev_value = INITIAL_CASH
    filtered_count = 0
    total_candidates = 0

    for date in trading_dates:
        day_prices = prices.loc[date] if date in prices.index else pd.Series(dtype=float)

        if date in rebal_dates_set:
            day_preds = preds_sp500[preds_sp500["date"] == date].sort_values(
                "prob_ensemble", ascending=False)
            if len(day_preds) == 0:
                port_values.append((date, prev_value)); continue

            vol_today = vol_matrix.loc[date] if date in vol_matrix.index else pd.Series(dtype=float)

            # Build candidate list
            candidates = []
            for _, row in day_preds.iterrows():
                sym, score = row["symbol"], row["prob_ensemble"]

                # Apply filters
                blocked = False
                if earnings_blackout and sym in earnings_blackout.get(date, set()):
                    blocked = True
                if liquidity_blackout and sym in liquidity_blackout.get(date, set()):
                    blocked = True
                if vol_blackout and sym in vol_blackout.get(date, set()):
                    blocked = True

                total_candidates += 1
                if blocked:
                    filtered_count += 1
                    continue

                v = vol_today.get(sym, 0.20) if len(vol_today) > 0 else 0.20
                if pd.isna(v) or v <= 0: v = 0.20
                candidates.append((sym, score, v, sector_map.get(sym, "Unknown")))

            betas_today = {}
            if date in betas.index:
                for sym, _, _, _ in candidates:
                    b = betas.loc[date].get(sym, 1.0)
                    betas_today[sym] = b if not pd.isna(b) else 1.0

            if mode == "equal_weight":
                tw = equal_weight_allocation(candidates, n=EW_TOP_N)
            else:
                tw = signal_weighted_allocation(candidates, sector_map, betas_today, n=SW_TOP_N)

            # Apply HMM if provided
            if hmm_exposures is not None:
                exp = hmm_exposures.get(date, 1.0)
                tw = {s: w * exp for s, w in tw.items()}

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
    filter_pct = filtered_count / total_candidates * 100 if total_candidates > 0 else 0

    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "vol": vol, "filter_pct": filter_pct}


def run_per_year(preds, prices, sector_map, vol_matrix, betas, mode,
                 earnings_blackout=None, liquidity_blackout=None,
                 vol_blackout=None, hmm_exposures=None):
    results = []
    for year in range(2015, 2026):
        yp = preds[(preds["date"] >= f"{year}-01-01") & (preds["date"] <= f"{year}-12-31")]
        if len(yp) < 50: continue
        m = run_backtest(yp, prices, sector_map, vol_matrix, betas, mode=mode,
                         earnings_blackout=earnings_blackout,
                         liquidity_blackout=liquidity_blackout,
                         vol_blackout=vol_blackout, hmm_exposures=hmm_exposures)
        m["year"] = year
        results.append(m)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  HMM (best config from tuning: 3st/min50%/lin)
# ══════════════════════════════════════════════════════════════════════════════

def load_hmm_exposures(prediction_dates):
    """Load HMM features and compute exposures with best config."""
    from hmmlearn.hmm import GaussianHMM

    hmm_features = pd.read_parquet(DATA_DIR / "cache_hmm_features.parquet")
    hmm_features.index = pd.to_datetime(hmm_features.index)

    feature_cols = list(hmm_features.columns)
    all_dates = hmm_features.index.sort_values()
    exposures = {}
    last_model = None
    last_fit_month = None

    for date in sorted(set(prediction_dates)):
        if date not in hmm_features.index:
            valid = all_dates[all_dates <= date]
            if len(valid) == 0: exposures[date] = 1.0; continue
            date_lookup = valid[-1]
        else:
            date_lookup = date

        current_month = pd.Timestamp(date).to_period("M")
        if last_model is None or current_month != last_fit_month:
            train_end = date_lookup
            train_start = train_end - pd.Timedelta(days=12 * 365)
            train_mask = (all_dates >= train_start) & (all_dates <= train_end)
            train_data = hmm_features.loc[all_dates[train_mask], feature_cols].values
            if len(train_data) < 252: exposures[date] = 1.0; continue

            mu = train_data.mean(axis=0)
            sigma = train_data.std(axis=0); sigma[sigma == 0] = 1.0
            train_scaled = (train_data - mu) / sigma
            try:
                model = GaussianHMM(n_components=3, covariance_type="full",
                                     n_iter=100, random_state=42, verbose=False)
                model.fit(train_scaled)
                last_model, last_fit_month = model, current_month
                last_mu, last_sigma = mu, sigma
            except Exception:
                exposures[date] = 1.0; continue

        history = hmm_features.loc[all_dates[all_dates <= date_lookup], feature_cols].values
        history_scaled = (history - last_mu) / last_sigma
        try:
            posteriors = last_model.predict_proba(history_scaled)
            bull_state = np.argmax(last_model.means_[:, 0])
            p_bull = posteriors[-1, bull_state]
        except Exception:
            p_bull = 0.5
        exposures[date] = 0.50 + 0.50 * p_bull  # min50%/lin

    return exposures


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  DON'T-TRADE FILTER COMPARISON")
    log("  Earnings + Liquidity + Spread filters on LambdaRank predictions")
    log("=" * 80)

    # Load data
    log("\n1. Loading data ...")
    lr_preds = pd.read_parquet(WF_DIR / "predictions_walkforward_lambdarank.parquet")
    lr_preds["date"] = pd.to_datetime(lr_preds["date"])

    prices = pd.read_parquet(DATA_DIR / "cache_prices_costmodel.parquet")
    prices.index = pd.to_datetime(prices.index)

    volume = pd.read_parquet(DATA_DIR / "cache_volume_costmodel.parquet")
    volume.index = pd.to_datetime(volume.index)

    with open(DATA_DIR / "cache_sectors.json") as f:
        sector_map = json.load(f)

    feat_vol = pd.read_parquet(DATA_DIR / "features.parquet", columns=["date", "symbol", "vol_20d"])
    feat_vol["date"] = pd.to_datetime(feat_vol["date"])
    vol_matrix = feat_vol.pivot_table(index="date", columns="symbol", values="vol_20d", aggfunc="last")

    earnings = pd.read_parquet(DATA_DIR / "fundamentals_earnings.parquet")

    # Betas
    log("  Computing betas ...")
    returns = prices.pct_change()
    spy_ret = returns["SPY"]
    spy_var = spy_ret.rolling(60, min_periods=30).var()
    xy = returns.multiply(spy_ret, axis=0)
    cov = xy.rolling(60, min_periods=30).mean() - returns.rolling(60, min_periods=30).mean().multiply(
        spy_ret.rolling(60, min_periods=30).mean(), axis=0)
    betas = cov.divide(spy_var, axis=0)
    betas["SPY"] = 1.0

    # Build filters
    log("\n2. Building filters ...")

    all_trading_dates = sorted(set(lr_preds["date"].unique().tolist()) &
                                set(prices.index.tolist()))

    log("  Building earnings blackout ...")
    t_f = time.perf_counter()
    earn_blackout = build_earnings_blackout_fast(earnings, all_trading_dates)
    earn_blocked = sum(len(v) for v in earn_blackout.values())
    log(f"    Earnings: {earn_blocked:,} stock-days blocked ({time.perf_counter()-t_f:.0f}s)")

    log("  Building liquidity blackout ...")
    liq_blackout = build_liquidity_blackout(prices, volume, all_trading_dates)
    liq_blocked = sum(len(v) for v in liq_blackout.values())
    log(f"    Liquidity: {liq_blocked:,} stock-days blocked")

    log("  Building vol/spread blackout ...")
    vol_blackout = build_vol_blackout(vol_matrix, all_trading_dates)
    vol_blocked = sum(len(v) for v in vol_blackout.values())
    log(f"    Vol/spread: {vol_blocked:,} stock-days blocked")

    # HMM exposures (best config)
    log("\n3. Computing HMM exposures (3st/min50%/lin) ...")
    hmm_exposures = load_hmm_exposures(all_trading_dates)
    log(f"  Mean exposure: {np.mean(list(hmm_exposures.values())):.2f}")

    # Run all combos
    log("\n4. Running backtests ...")

    combos = [
        ("LR + EW",                  "equal_weight",    False, False),
        ("LR + EW + Filters",        "equal_weight",    True,  False),
        ("LR + SW",                  "signal_weighted",  False, False),
        ("LR + SW + Filters",        "signal_weighted",  True,  False),
        ("LR + EW + HMM",           "equal_weight",    False, True),
        ("LR + EW + Filters + HMM", "equal_weight",    True,  True),
        ("LR + SW + HMM",           "signal_weighted",  False, True),
        ("LR + SW + Filters + HMM", "signal_weighted",  True,  True),
    ]

    results = []
    for name, mode, use_filters, use_hmm in combos:
        log(f"  {name} ...")
        eb = earn_blackout if use_filters else None
        lb = liq_blackout if use_filters else None
        vb = vol_blackout if use_filters else None
        hm = hmm_exposures if use_hmm else None

        cont = run_backtest(lr_preds, prices, sector_map, vol_matrix, betas,
                            mode=mode, earnings_blackout=eb, liquidity_blackout=lb,
                            vol_blackout=vb, hmm_exposures=hm)
        yearly = run_per_year(lr_preds, prices, sector_map, vol_matrix, betas,
                              mode=mode, earnings_blackout=eb, liquidity_blackout=lb,
                              vol_blackout=vb, hmm_exposures=hm)

        med_cagr = np.median([y["cagr"] for y in yearly])
        med_sharpe = np.median([y["sharpe"] for y in yearly])
        worst_dd = min(y["max_dd"] for y in yearly)
        pos = sum(1 for y in yearly if y["cagr"] > 0)

        results.append({
            "name": name, "cont": cont, "med_cagr": med_cagr,
            "med_sharpe": med_sharpe, "worst_yr_dd": worst_dd,
            "pos_years": pos, "n_years": len(yearly), "yearly": yearly,
        })

        log(f"    Cont: CAGR={cont['cagr']:+.1%} Sharpe={cont['sharpe']:.2f} "
            f"DD={cont['max_dd']:.1%} Filtered={cont['filter_pct']:.1f}%")
        log(f"    Med:  CAGR={med_cagr:+.1%} Sharpe={med_sharpe:.2f} Pos={pos}/{len(yearly)}")

    # Summary table
    log(f"\n{'='*80}")
    log("  RESULTS SUMMARY")
    log(f"{'='*80}")

    log(f"\n  {'Combination':<27} {'C.CAGR':>7} {'C.Shp':>6} {'C.Sort':>7} {'C.DD':>6} "
        f"{'M.CAGR':>7} {'M.Shp':>6} {'W.DD':>6} {'Pos':>4} {'Filt%':>5}")
    log(f"  {'─'*27} {'─'*7} {'─'*6} {'─'*7} {'─'*6} "
        f"{'─'*7} {'─'*6} {'─'*6} {'─'*4} {'─'*5}")

    for r in results:
        c = r["cont"]
        log(f"  {r['name']:<27} "
            f"{c['cagr']:>+6.1%} {c['sharpe']:>6.2f} {c['sortino']:>7.2f} {c['max_dd']:>5.1%} "
            f"{r['med_cagr']:>+6.1%} {r['med_sharpe']:>6.2f} {r['worst_yr_dd']:>5.1%} "
            f"{r['pos_years']:>2}/{r['n_years']} {c['filter_pct']:>4.1f}")

    # Filter impact
    log(f"\n{'='*80}")
    log("  FILTER IMPACT (filters only, no HMM)")
    log(f"{'='*80}")
    for i in [0, 2]:
        base, filt = results[i], results[i+1]
        log(f"\n  {base['name']} → {filt['name']}:")
        log(f"    Cont Sharpe: {base['cont']['sharpe']:.2f} → {filt['cont']['sharpe']:.2f} ({filt['cont']['sharpe']-base['cont']['sharpe']:+.2f})")
        log(f"    Cont DD:     {base['cont']['max_dd']:.1%} → {filt['cont']['max_dd']:.1%} ({filt['cont']['max_dd']-base['cont']['max_dd']:+.1%})")
        log(f"    Cont CAGR:   {base['cont']['cagr']:+.1%} → {filt['cont']['cagr']:+.1%}")
        log(f"    Med Sharpe:  {base['med_sharpe']:.2f} → {filt['med_sharpe']:.2f}")

    log(f"\n{'='*80}")
    log("  BEST COMBO: FILTERS + HMM")
    log(f"{'='*80}")
    for i in [0, 2]:
        base = results[i]
        best = results[i+3]  # filters + HMM
        log(f"\n  {base['name']} → {best['name']}:")
        log(f"    Cont Sharpe: {base['cont']['sharpe']:.2f} → {best['cont']['sharpe']:.2f}")
        log(f"    Cont DD:     {base['cont']['max_dd']:.1%} → {best['cont']['max_dd']:.1%}")
        log(f"    Cont CAGR:   {base['cont']['cagr']:+.1%} → {best['cont']['cagr']:+.1%}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
