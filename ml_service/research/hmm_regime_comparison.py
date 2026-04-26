#!/usr/bin/env python3
"""
HMM Regime Detector Comparison
===============================
Tests a 3-state Gaussian HMM regime detector as a continuous gross-exposure
moderator on top of LambdaRank predictions.

4 combinations (continuous + per-year median):
  1. LambdaRank + EW top-5              (baseline 1)
  2. LambdaRank + EW top-5 + HMM        (baseline 1 + regime)
  3. LambdaRank + Signal-weighted        (baseline 2)
  4. LambdaRank + Signal-weighted + HMM  (baseline 2 + regime)

HMM features (daily):
  - SPY 5-day return
  - SPY 20-day realized vol (annualized)
  - VIX level
  - VIX3M/VIX ratio (term structure)
  - HYG/LQD ratio 20-day change
  - Breadth (% of SP500 above 60-day return > 0)
  - SPY drawdown from 6-month peak

Gross exposure = 0.30 + 0.70 × P(bull)

Run:
    cd ml_service && python3 research/hmm_regime_comparison.py
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
from hmmlearn.hmm import GaussianHMM

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WF_DIR = DATA_DIR / "walkforward"

INITIAL_CASH = 100_000.0
HOLD_DAYS = 10
TRADING_DAYS = 252

# Position sizing configs
EW_TOP_N = 5
SW_TOP_N = 10
MAX_SINGLE_NAME_PCT = 0.25
MAX_SECTOR_PCT = 0.35
MAX_BETA = 1.2

# HMM config
HMM_N_STATES = 3
HMM_MIN_EXPOSURE = 0.30
HMM_MAX_EXPOSURE = 1.00
HMM_TRAIN_YEARS = 12  # rolling window for HMM fitting


def log(msg: str):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HMM Regime Detector
# ══════════════════════════════════════════════════════════════════════════════

def download_hmm_features(start: str, end: str) -> pd.DataFrame:
    """Download and compute all HMM input features.
    Uses Massive for HYG/LQD/SPY, yfinance only for ^VIX/^VIX3M (indices)."""
    import yfinance as yf
    from massive_data_provider import MassiveDataProvider

    cache_file = DATA_DIR / "cache_hmm_features.parquet"
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=5):
            return cached

    provider = MassiveDataProvider(validate_vs_yfinance=False)

    # Fetch HYG, LQD, SPY from Massive
    log("  Fetching HYG, LQD, SPY from Massive ...")
    etf_bars = provider.fetch_bars_batch(["HYG", "LQD", "SPY"], warmup_days=
        (pd.Timestamp(end) - pd.Timestamp(start)).days + 30)

    # Fetch ^VIX, ^VIX3M from yfinance (indices not in Massive plan)
    log("  Fetching ^VIX, ^VIX3M from yfinance (indices) ...")
    vix_raw = yf.download(["^VIX", "^VIX3M"], start=start, end=end,
                           progress=False, auto_adjust=True)
    vix_close = vix_raw["Close"]
    vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None)

    # Build close price DataFrame
    spy_close = etf_bars["SPY"]["close"] if "SPY" in etf_bars and len(etf_bars["SPY"]) > 0 else pd.Series(dtype=float)
    hyg_close = etf_bars["HYG"]["close"] if "HYG" in etf_bars and len(etf_bars["HYG"]) > 0 else pd.Series(dtype=float)
    lqd_close = etf_bars["LQD"]["close"] if "LQD" in etf_bars and len(etf_bars["LQD"]) > 0 else pd.Series(dtype=float)

    # Align all on common index
    all_close = pd.DataFrame({
        "SPY": spy_close, "HYG": hyg_close, "LQD": lqd_close,
        "^VIX": vix_close["^VIX"], "^VIX3M": vix_close["^VIX3M"],
    }).dropna()

    hmm_df = pd.DataFrame(index=all_close.index)

    # SPY features
    spy = all_close["SPY"]
    hmm_df["spy_ret_5d"] = spy.pct_change(5)
    hmm_df["spy_vol_20d"] = spy.pct_change().rolling(20).std() * np.sqrt(252)
    spy_peak_6m = spy.rolling(126, min_periods=20).max()
    hmm_df["spy_dd_6m"] = (spy - spy_peak_6m) / spy_peak_6m

    # VIX features
    hmm_df["vix"] = all_close["^VIX"]
    hmm_df["vix_term_structure"] = all_close["^VIX3M"] / all_close["^VIX"]

    # Credit
    hmm_df["hyg_lqd_change_20d"] = (all_close["HYG"] / all_close["LQD"]).pct_change(20)

    # Breadth — compute from SP500 price data
    price_file = DATA_DIR / "cache_prices_costmodel.parquet"
    if price_file.exists():
        all_prices = pd.read_parquet(price_file)
        all_prices.index = pd.to_datetime(all_prices.index)
        ret_60d = all_prices.pct_change(60)
        breadth = (ret_60d > 0).mean(axis=1)
        hmm_df["breadth"] = breadth.reindex(hmm_df.index, method="ffill")
    else:
        hmm_df["breadth"] = (spy.pct_change(60) > 0).astype(float).rolling(20).mean()

    hmm_df = hmm_df.dropna()
    hmm_df.to_parquet(cache_file)
    log(f"  HMM features: {hmm_df.shape}, {hmm_df.index.min().date()} -> {hmm_df.index.max().date()}")
    return hmm_df


def fit_hmm_and_get_exposures(hmm_features: pd.DataFrame,
                               prediction_dates: list) -> dict:
    """
    Fit rolling HMM and compute P(bull) for each prediction date.
    Returns {date: gross_exposure} where exposure = 0.30 + 0.70 * P(bull).
    """
    feature_cols = list(hmm_features.columns)
    all_dates = hmm_features.index.sort_values()
    pred_dates_set = set(prediction_dates)

    # Standardize features globally for initial reference
    exposures = {}
    last_model = None
    last_fit_month = None

    for date in sorted(pred_dates_set):
        if date not in hmm_features.index:
            # Use nearest previous date
            valid = all_dates[all_dates <= date]
            if len(valid) == 0:
                exposures[date] = HMM_MAX_EXPOSURE
                continue
            date_lookup = valid[-1]
        else:
            date_lookup = date

        # Refit monthly
        current_month = pd.Timestamp(date).to_period("M")
        if last_model is None or current_month != last_fit_month:
            # Rolling window
            train_end = date_lookup
            train_start = train_end - pd.Timedelta(days=HMM_TRAIN_YEARS * 365)
            train_mask = (all_dates >= train_start) & (all_dates <= train_end)
            train_data = hmm_features.loc[all_dates[train_mask], feature_cols].values

            if len(train_data) < 252:  # need at least 1 year
                exposures[date] = HMM_MAX_EXPOSURE
                continue

            # Standardize
            mu = train_data.mean(axis=0)
            sigma = train_data.std(axis=0)
            sigma[sigma == 0] = 1.0
            train_scaled = (train_data - mu) / sigma

            # Fit HMM
            try:
                model = GaussianHMM(
                    n_components=HMM_N_STATES,
                    covariance_type="full",
                    n_iter=100,
                    random_state=42,
                    verbose=False,
                )
                model.fit(train_scaled)
                last_model = model
                last_fit_month = current_month
                last_mu = mu
                last_sigma = sigma

                # Label states by mean SPY return (column 0 = spy_ret_5d)
                state_means = model.means_[:, 0]  # spy_ret_5d mean per state
                bull_state = np.argmax(state_means)
            except Exception:
                exposures[date] = HMM_MAX_EXPOSURE
                continue
        else:
            model = last_model
            mu = last_mu
            sigma = last_sigma
            state_means = model.means_[:, 0]
            bull_state = np.argmax(state_means)

        # Get P(bull) for current date using all history up to date
        history_mask = all_dates <= date_lookup
        history = hmm_features.loc[all_dates[history_mask], feature_cols].values
        history_scaled = (history - mu) / sigma

        try:
            posteriors = model.predict_proba(history_scaled)
            p_bull = posteriors[-1, bull_state]
        except Exception:
            p_bull = 0.5

        exposure = HMM_MIN_EXPOSURE + (HMM_MAX_EXPOSURE - HMM_MIN_EXPOSURE) * p_bull
        exposures[date] = exposure

    return exposures


# ══════════════════════════════════════════════════════════════════════════════
#  Position Sizing (same as comprehensive_comparison.py)
# ══════════════════════════════════════════════════════════════════════════════

def equal_weight_allocation(candidates, n=5):
    top = candidates[:n]
    if not top:
        return {}
    w = 1.0 / len(top)
    return {sym: w for sym, _, _, _ in top}


def signal_weighted_allocation(candidates, sector_map, betas_today, n=10):
    top = candidates[:n]
    if not top:
        return {}

    scores = np.array([s for _, s, _, _ in top])
    vols = np.array([max(v, 0.05) for _, _, v, _ in top])
    min_score = scores.min()
    excess = scores - min_score + 1e-6
    raw_w = excess / vols
    raw_w = raw_w / raw_w.sum()

    syms = [s for s, _, _, _ in top]
    weights = dict(zip(syms, raw_w))

    for _ in range(20):
        changed = False
        for sym in list(weights.keys()):
            if weights[sym] > MAX_SINGLE_NAME_PCT:
                weights[sym] = MAX_SINGLE_NAME_PCT
                changed = True

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
            for sym in weights:
                weights[sym] *= scale
            changed = True

        total_w = sum(weights.values())
        if total_w > 1.0:
            for sym in weights:
                weights[sym] /= total_w
            changed = True

        if not changed:
            break

    return {s: w for s, w in weights.items() if w > 0.01}


# ══════════════════════════════════════════════════════════════════════════════
#  Backtester with HMM exposure scaling
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(preds, prices, sector_map, vol_matrix, betas,
                 mode="equal_weight", hmm_exposures=None):
    """
    Run position-level backtest. If hmm_exposures is provided, scale all
    weights by the exposure for each rebalance date.
    """
    preds_sp500 = preds[preds["in_sp500"] == True].copy()
    rebal_dates = sorted(preds_sp500["date"].unique())
    rebal_dates = [d for i, d in enumerate(rebal_dates) if i % HOLD_DAYS == 0]

    trading_dates = sorted(prices.index)
    wf_start, wf_end = preds_sp500["date"].min(), preds_sp500["date"].max()
    trading_dates = [d for d in trading_dates if wf_start <= d <= wf_end]

    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    prev_value = INITIAL_CASH

    for date in trading_dates:
        day_prices = prices.loc[date] if date in prices.index else pd.Series(dtype=float)

        if date in rebal_dates:
            day_preds = preds_sp500[preds_sp500["date"] == date].sort_values(
                "prob_ensemble", ascending=False)
            if len(day_preds) == 0:
                port_values.append((date, prev_value))
                continue

            vol_today = vol_matrix.loc[date] if date in vol_matrix.index else pd.Series(dtype=float)
            candidates = []
            for _, row in day_preds.iterrows():
                sym = row["symbol"]
                score = row["prob_ensemble"]
                vol_20d = vol_today.get(sym, 0.20) if len(vol_today) > 0 else 0.20
                if pd.isna(vol_20d) or vol_20d <= 0:
                    vol_20d = 0.20
                candidates.append((sym, score, vol_20d, sector_map.get(sym, "Unknown")))

            betas_today = {}
            if date in betas.index:
                for sym, _, _, _ in candidates:
                    b = betas.loc[date].get(sym, 1.0)
                    betas_today[sym] = b if not pd.isna(b) else 1.0

            if mode == "equal_weight":
                target_weights = equal_weight_allocation(candidates, n=EW_TOP_N)
            elif mode == "signal_weighted":
                target_weights = signal_weighted_allocation(
                    candidates, sector_map, betas_today, n=SW_TOP_N)
            else:
                raise ValueError(mode)

            # Apply HMM exposure scaling
            if hmm_exposures is not None:
                exposure = hmm_exposures.get(date, 1.0)
                target_weights = {s: w * exposure for s, w in target_weights.items()}

            # Rebalance
            total_value = cash
            for sym, h in holdings.items():
                px = day_prices.get(sym, h.get("entry_px", 0))
                if pd.isna(px): px = h.get("entry_px", 0)
                total_value += h["shares"] * px

            cash = total_value
            holdings = {}
            for sym, w in target_weights.items():
                px = day_prices.get(sym)
                if px is None or pd.isna(px) or px <= 0:
                    continue
                alloc = total_value * w
                holdings[sym] = {"shares": alloc / px, "entry_px": px}
                cash -= alloc

        # Mark-to-market
        total_value = cash
        for sym, h in holdings.items():
            px = day_prices.get(sym, h.get("entry_px", 0))
            if pd.isna(px): px = h.get("entry_px", 0)
            total_value += h["shares"] * px

        port_values.append((date, total_value))
        prev_value = total_value

    # Metrics
    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0: years = 1.0

    daily_ret = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    sortino_d = daily_ret[daily_ret < 0].std()
    sortino = daily_ret.mean() / sortino_d * np.sqrt(252) if sortino_d > 0 else np.nan
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    realized_vol = daily_ret.std() * np.sqrt(252)

    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "vol": realized_vol, "years": years,
    }


def run_per_year(preds, prices, sector_map, vol_matrix, betas,
                 mode, hmm_exposures=None):
    results = []
    for year in range(2015, 2026):
        ys = pd.Timestamp(f"{year}-01-01")
        ye = pd.Timestamp(f"{year}-12-31")
        yp = preds[(preds["date"] >= ys) & (preds["date"] <= ye)]
        if len(yp) < 50:
            continue
        m = run_backtest(yp, prices, sector_map, vol_matrix, betas,
                         mode=mode, hmm_exposures=hmm_exposures)
        m["year"] = year
        results.append(m)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()

    log("=" * 80)
    log("  HMM REGIME DETECTOR COMPARISON")
    log("  LambdaRank × (EW / Signal-weighted) × (No HMM / HMM)")
    log("=" * 80)

    # ── Load data ──
    log("\n1. Loading data ...")
    lr_preds = pd.read_parquet(WF_DIR / "predictions_walkforward_lambdarank.parquet")
    lr_preds["date"] = pd.to_datetime(lr_preds["date"])
    log(f"  Predictions: {len(lr_preds):,} rows")

    prices = pd.read_parquet(DATA_DIR / "cache_prices_costmodel.parquet")
    prices.index = pd.to_datetime(prices.index)

    with open(DATA_DIR / "cache_sectors.json") as f:
        sector_map = json.load(f)

    feat_vol = pd.read_parquet(DATA_DIR / "features.parquet", columns=["date", "symbol", "vol_20d"])
    feat_vol["date"] = pd.to_datetime(feat_vol["date"])
    vol_matrix = feat_vol.pivot_table(index="date", columns="symbol", values="vol_20d", aggfunc="last")

    # Betas
    log("  Computing betas ...")
    returns = prices.pct_change()
    spy_ret = returns["SPY"]
    spy_var = spy_ret.rolling(60, min_periods=30).var()
    xy = returns.multiply(spy_ret, axis=0)
    mean_xy = xy.rolling(60, min_periods=30).mean()
    mean_x = returns.rolling(60, min_periods=30).mean()
    mean_spy = spy_ret.rolling(60, min_periods=30).mean()
    cov = mean_xy.subtract(mean_x.multiply(mean_spy, axis=0))
    betas = cov.divide(spy_var, axis=0)
    betas["SPY"] = 1.0

    # ── Download HMM features ──
    log("\n2. Building HMM features ...")
    hmm_start = "2011-01-01"
    hmm_end = "2026-05-01"
    hmm_features = download_hmm_features(hmm_start, hmm_end)

    # ── Fit HMM and compute exposures ──
    log("\n3. Fitting HMM and computing exposures ...")
    prediction_dates = sorted(lr_preds["date"].unique().tolist())
    t_hmm = time.perf_counter()
    hmm_exposures = fit_hmm_and_get_exposures(hmm_features, prediction_dates)
    log(f"  HMM fitted in {time.perf_counter() - t_hmm:.0f}s")

    # Exposure stats
    exp_vals = list(hmm_exposures.values())
    log(f"  Exposure range: [{min(exp_vals):.2f}, {max(exp_vals):.2f}]")
    log(f"  Mean exposure: {np.mean(exp_vals):.2f}")
    log(f"  Days at min (<0.35): {sum(1 for e in exp_vals if e < 0.35)}/{len(exp_vals)}")
    log(f"  Days at max (>0.95): {sum(1 for e in exp_vals if e > 0.95)}/{len(exp_vals)}")

    # Show regime during known events
    log("\n  Regime check (exposure on key dates):")
    check_dates = [
        ("2018-12-24", "Dec 2018 crash"),
        ("2020-03-23", "COVID bottom"),
        ("2020-03-16", "COVID crash week"),
        ("2022-06-16", "2022 bear"),
        ("2022-01-24", "2022 selloff start"),
        ("2021-11-19", "2021 peak"),
        ("2023-10-27", "2023 Oct low"),
        ("2024-07-16", "2024 bull"),
    ]
    for date_str, label in check_dates:
        d = pd.Timestamp(date_str)
        exp = hmm_exposures.get(d, None)
        if exp is None:
            # Find nearest
            nearest = min(hmm_exposures.keys(), key=lambda x: abs(x - d))
            exp = hmm_exposures[nearest]
            date_str = str(nearest.date())
        log(f"    {date_str} ({label}): exposure={exp:.2f}")

    # ── Run 4 combinations ──
    log("\n4. Running backtests ...")

    combos = [
        ("LR + EW (no HMM)",    "equal_weight",    None),
        ("LR + EW + HMM",       "equal_weight",    hmm_exposures),
        ("LR + SW (no HMM)",    "signal_weighted",  None),
        ("LR + SW + HMM",       "signal_weighted",  hmm_exposures),
    ]

    results = []
    for name, mode, exposures in combos:
        log(f"\n  {name} ...")

        cont = run_backtest(lr_preds, prices, sector_map, vol_matrix, betas,
                            mode=mode, hmm_exposures=exposures)
        yearly = run_per_year(lr_preds, prices, sector_map, vol_matrix, betas,
                              mode=mode, hmm_exposures=exposures)

        med_cagr = np.median([y["cagr"] for y in yearly])
        med_sharpe = np.median([y["sharpe"] for y in yearly])
        worst_dd = min(y["max_dd"] for y in yearly)
        pos_years = sum(1 for y in yearly if y["cagr"] > 0)

        results.append({
            "name": name,
            "cont_cagr": cont["cagr"], "cont_sharpe": cont["sharpe"],
            "cont_sortino": cont["sortino"],
            "cont_dd": cont["max_dd"], "cont_vol": cont["vol"],
            "med_cagr": med_cagr, "med_sharpe": med_sharpe,
            "worst_yr_dd": worst_dd, "pos_years": pos_years,
            "n_years": len(yearly), "yearly": yearly,
        })

        log(f"    Cont: CAGR={cont['cagr']:+.1%}  Sharpe={cont['sharpe']:.2f}  "
            f"DD={cont['max_dd']:.1%}  Vol={cont['vol']:.1%}")
        log(f"    Med:  CAGR={med_cagr:+.1%}  Sharpe={med_sharpe:.2f}  "
            f"Pos={pos_years}/{len(yearly)}")

    # ── Summary Table ──
    log(f"\n{'='*80}")
    log("  RESULTS SUMMARY")
    log(f"{'='*80}")

    log(f"\n  {'Combination':<25} {'C.CAGR':>7} {'C.Shp':>6} {'C.Sort':>7} {'C.DD':>6} {'C.Vol':>6}"
        f" {'M.CAGR':>7} {'M.Shp':>6} {'W.DD':>6} {'Pos':>4}")
    log(f"  {'─'*25} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*6}"
        f" {'─'*7} {'─'*6} {'─'*6} {'─'*4}")

    for r in results:
        log(f"  {r['name']:<25} "
            f"{r['cont_cagr']:>+6.1%} {r['cont_sharpe']:>6.2f} {r['cont_sortino']:>7.2f} "
            f"{r['cont_dd']:>5.1%} {r['cont_vol']:>5.1%} "
            f"{r['med_cagr']:>+6.1%} {r['med_sharpe']:>6.2f} {r['worst_yr_dd']:>5.1%} "
            f"{r['pos_years']:>2}/{r['n_years']}")

    # ── Per-Year Detail ──
    log(f"\n{'='*80}")
    log("  PER-YEAR DETAIL")
    log(f"{'='*80}")

    header = f"  {'Year':<6}"
    for r in results:
        header += f" {r['name'][:12]:>13}"
    log(header)
    sep = f"  {'─'*6}"
    for _ in results:
        sep += f" {'─'*13}"
    log(sep)

    for i in range(len(results[0]["yearly"])):
        year = results[0]["yearly"][i]["year"]
        line = f"  {year:<6}"
        for r in results:
            y = r["yearly"][i]
            line += f" {y['cagr']:>+5.0%}/{y['sharpe']:>4.1f}"
        log(line)

    # ── HMM Impact ──
    log(f"\n{'='*80}")
    log("  HMM IMPACT")
    log(f"{'='*80}")

    for i in [0, 2]:  # EW and SW baselines
        base = results[i]
        hmm = results[i + 1]
        log(f"\n  {base['name']} → {hmm['name']}:")
        log(f"    Continuous Sharpe: {base['cont_sharpe']:.2f} → {hmm['cont_sharpe']:.2f} ({hmm['cont_sharpe'] - base['cont_sharpe']:+.2f})")
        log(f"    Continuous DD:    {base['cont_dd']:.1%} → {hmm['cont_dd']:.1%} ({hmm['cont_dd'] - base['cont_dd']:+.1%})")
        log(f"    Continuous Vol:   {base['cont_vol']:.1%} → {hmm['cont_vol']:.1%}")
        log(f"    Median Sharpe:    {base['med_sharpe']:.2f} → {hmm['med_sharpe']:.2f} ({hmm['med_sharpe'] - base['med_sharpe']:+.2f})")
        log(f"    Worst year DD:    {base['worst_yr_dd']:.1%} → {hmm['worst_yr_dd']:.1%}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
