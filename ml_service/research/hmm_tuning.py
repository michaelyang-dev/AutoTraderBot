#!/usr/bin/env python3
"""
HMM Tuning — find the best exposure config to maximize Sharpe while keeping DD low.

Tests a grid of:
  - n_states: 2 vs 3
  - min_exposure: 0.30, 0.40, 0.50, 0.60
  - exposure formula: linear vs sqrt (sqrt is more aggressive in bull)

Runs on LR + SW (best Sharpe combo) — continuous + per-year.
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
SW_TOP_N = 10
EW_TOP_N = 5
MAX_SINGLE_NAME_PCT = 0.25
MAX_SECTOR_PCT = 0.35
MAX_BETA = 1.2
HMM_TRAIN_YEARS = 12


def log(msg: str):
    print(msg, flush=True)


def load_hmm_features():
    cache = DATA_DIR / "cache_hmm_features.parquet"
    df = pd.read_parquet(cache)
    df.index = pd.to_datetime(df.index)
    return df


def fit_hmm_exposures(hmm_features, prediction_dates, n_states=3,
                       min_exp=0.30, use_sqrt=False):
    feature_cols = list(hmm_features.columns)
    all_dates = hmm_features.index.sort_values()
    pred_dates_set = set(prediction_dates)
    exposures = {}
    last_model = None
    last_fit_month = None

    for date in sorted(pred_dates_set):
        if date not in hmm_features.index:
            valid = all_dates[all_dates <= date]
            if len(valid) == 0:
                exposures[date] = 1.0
                continue
            date_lookup = valid[-1]
        else:
            date_lookup = date

        current_month = pd.Timestamp(date).to_period("M")
        if last_model is None or current_month != last_fit_month:
            train_end = date_lookup
            train_start = train_end - pd.Timedelta(days=HMM_TRAIN_YEARS * 365)
            train_mask = (all_dates >= train_start) & (all_dates <= train_end)
            train_data = hmm_features.loc[all_dates[train_mask], feature_cols].values

            if len(train_data) < 252:
                exposures[date] = 1.0
                continue

            mu = train_data.mean(axis=0)
            sigma = train_data.std(axis=0)
            sigma[sigma == 0] = 1.0
            train_scaled = (train_data - mu) / sigma

            try:
                model = GaussianHMM(n_components=n_states, covariance_type="full",
                                     n_iter=100, random_state=42, verbose=False)
                model.fit(train_scaled)
                last_model = model
                last_fit_month = current_month
                last_mu, last_sigma = mu, sigma
                state_means = model.means_[:, 0]
                bull_state = np.argmax(state_means)
            except Exception:
                exposures[date] = 1.0
                continue
        else:
            model = last_model
            mu, sigma = last_mu, last_sigma
            state_means = model.means_[:, 0]
            bull_state = np.argmax(state_means)

        history_mask = all_dates <= date_lookup
        history = hmm_features.loc[all_dates[history_mask], feature_cols].values
        history_scaled = (history - mu) / sigma

        try:
            posteriors = model.predict_proba(history_scaled)
            p_bull = posteriors[-1, bull_state]
        except Exception:
            p_bull = 0.5

        if use_sqrt:
            # sqrt formula: more aggressive in bull, still protective in bear
            exposure = min_exp + (1.0 - min_exp) * np.sqrt(p_bull)
        else:
            exposure = min_exp + (1.0 - min_exp) * p_bull

        exposures[date] = exposure

    return exposures


# ── Position sizing (signal-weighted) ──

def signal_weighted_allocation(candidates, sector_map, betas_today, n=10):
    top = candidates[:n]
    if not top:
        return {}
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


def equal_weight_allocation(candidates, n=5):
    top = candidates[:n]
    if not top: return {}
    w = 1.0 / len(top)
    return {sym: w for sym, _, _, _ in top}


# ── Backtester ──

def run_backtest(preds, prices, sector_map, vol_matrix, betas,
                 mode="signal_weighted", hmm_exposures=None):
    preds_sp500 = preds[preds["in_sp500"] == True].copy()
    rebal_dates = sorted(preds_sp500["date"].unique())
    rebal_dates = set(d for i, d in enumerate(rebal_dates) if i % HOLD_DAYS == 0)

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

            if mode == "signal_weighted":
                tw = signal_weighted_allocation(candidates, sector_map, betas_today, n=SW_TOP_N)
            else:
                tw = equal_weight_allocation(candidates, n=EW_TOP_N)

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
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "vol": vol, "years": years}


def run_per_year(preds, prices, sector_map, vol_matrix, betas, mode, hmm_exposures=None):
    results = []
    for year in range(2015, 2026):
        yp = preds[(preds["date"] >= f"{year}-01-01") & (preds["date"] <= f"{year}-12-31")]
        if len(yp) < 50: continue
        m = run_backtest(yp, prices, sector_map, vol_matrix, betas, mode=mode, hmm_exposures=hmm_exposures)
        m["year"] = year
        results.append(m)
    return results


def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  HMM TUNING GRID")
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
    cov = xy.rolling(60, min_periods=30).mean() - returns.rolling(60, min_periods=30).mean().multiply(spy_ret.rolling(60, min_periods=30).mean(), axis=0)
    betas = cov.divide(spy_var, axis=0)
    betas["SPY"] = 1.0

    hmm_features = load_hmm_features()
    prediction_dates = sorted(lr_preds["date"].unique().tolist())

    # Grid search
    configs = []
    for n_states in [2, 3]:
        for min_exp in [0.30, 0.40, 0.50, 0.60]:
            for use_sqrt in [False, True]:
                configs.append({
                    "n_states": n_states, "min_exp": min_exp,
                    "use_sqrt": use_sqrt,
                    "label": f"{n_states}st/min{min_exp:.0%}/{'sqrt' if use_sqrt else 'lin'}",
                })

    # Add no-HMM baselines
    log(f"\n2. Running baselines ...")
    baselines = {}
    for mode_name, mode in [("EW", "equal_weight"), ("SW", "signal_weighted")]:
        cont = run_backtest(lr_preds, prices, sector_map, vol_matrix, betas, mode=mode)
        yearly = run_per_year(lr_preds, prices, sector_map, vol_matrix, betas, mode=mode)
        baselines[mode_name] = {
            "cont": cont,
            "med_sharpe": np.median([y["sharpe"] for y in yearly]),
            "med_cagr": np.median([y["cagr"] for y in yearly]),
        }
        log(f"  {mode_name} baseline: C.CAGR={cont['cagr']:+.1%} C.Sharpe={cont['sharpe']:.2f} "
            f"C.DD={cont['max_dd']:.1%} M.Sharpe={baselines[mode_name]['med_sharpe']:.2f}")

    log(f"\n3. Testing {len(configs)} HMM configs × 2 sizing modes = {len(configs)*2} combos ...")

    all_results = []
    for i, cfg in enumerate(configs):
        log(f"  [{i+1}/{len(configs)}] {cfg['label']} ...", )
        exposures = fit_hmm_exposures(
            hmm_features, prediction_dates,
            n_states=cfg["n_states"], min_exp=cfg["min_exp"],
            use_sqrt=cfg["use_sqrt"])

        mean_exp = np.mean(list(exposures.values()))

        for mode_name, mode in [("EW", "equal_weight"), ("SW", "signal_weighted")]:
            cont = run_backtest(lr_preds, prices, sector_map, vol_matrix, betas,
                                mode=mode, hmm_exposures=exposures)
            yearly = run_per_year(lr_preds, prices, sector_map, vol_matrix, betas,
                                  mode=mode, hmm_exposures=exposures)
            med_sharpe = np.median([y["sharpe"] for y in yearly])
            med_cagr = np.median([y["cagr"] for y in yearly])
            pos = sum(1 for y in yearly if y["cagr"] > 0)

            all_results.append({
                **cfg, "mode": mode_name, "mean_exp": mean_exp,
                "cont_cagr": cont["cagr"], "cont_sharpe": cont["sharpe"],
                "cont_sortino": cont["sortino"], "cont_dd": cont["max_dd"],
                "cont_vol": cont["vol"],
                "med_cagr": med_cagr, "med_sharpe": med_sharpe,
                "pos_years": pos,
            })

    # ── Results ──
    log(f"\n{'='*80}")
    log("  ALL RESULTS (sorted by continuous Sharpe)")
    log(f"{'='*80}")

    log(f"\n  {'Config':<22} {'Mode':<4} {'MnExp':>5} {'C.CAGR':>7} {'C.Shp':>6} {'C.Sort':>7} "
        f"{'C.DD':>6} {'M.CAGR':>7} {'M.Shp':>6} {'Pos':>4}")
    log(f"  {'─'*22} {'─'*4} {'─'*5} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*4}")

    # Add baselines to results for sorting
    for mode_name in ["EW", "SW"]:
        b = baselines[mode_name]
        all_results.append({
            "label": "NO HMM", "mode": mode_name, "mean_exp": 1.0,
            "cont_cagr": b["cont"]["cagr"], "cont_sharpe": b["cont"]["sharpe"],
            "cont_sortino": b["cont"]["sortino"], "cont_dd": b["cont"]["max_dd"],
            "cont_vol": b["cont"]["vol"],
            "med_cagr": b["med_cagr"], "med_sharpe": b["med_sharpe"],
            "pos_years": 0,
        })

    all_results.sort(key=lambda r: r["cont_sharpe"], reverse=True)

    for r in all_results:
        log(f"  {r['label']:<22} {r['mode']:<4} {r['mean_exp']:>5.2f} "
            f"{r['cont_cagr']:>+6.1%} {r['cont_sharpe']:>6.2f} {r['cont_sortino']:>7.2f} "
            f"{r['cont_dd']:>5.1%} {r['med_cagr']:>+6.1%} {r['med_sharpe']:>6.2f} "
            f"{r['pos_years']:>3}")

    # ── Best configs ──
    log(f"\n{'='*80}")
    log("  TOP 5 BY CONTINUOUS SHARPE")
    log(f"{'='*80}")
    for r in all_results[:5]:
        log(f"  {r['label']} + {r['mode']}: Sharpe={r['cont_sharpe']:.2f} "
            f"CAGR={r['cont_cagr']:+.1%} DD={r['cont_dd']:.1%} "
            f"Sortino={r['cont_sortino']:.2f} MeanExp={r['mean_exp']:.2f}")

    log(f"\n  TOP 5 BY CONTINUOUS SHARPE WITH DD > -35%")
    good_dd = [r for r in all_results if r["cont_dd"] > -0.35]
    for r in good_dd[:5]:
        log(f"  {r['label']} + {r['mode']}: Sharpe={r['cont_sharpe']:.2f} "
            f"CAGR={r['cont_cagr']:+.1%} DD={r['cont_dd']:.1%} "
            f"Sortino={r['cont_sortino']:.2f} MeanExp={r['mean_exp']:.2f}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
