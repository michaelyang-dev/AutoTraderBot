#!/usr/bin/env python3
"""
Full Retrain Report — Survivorship-Fixed LambdaRank Model
=========================================================
Comprehensive comparison of the new survivorship-fixed model against
the previous V6 baseline. Runs all configs and produces full metrics.

Run after walk-forward and production retrain complete:
    cd ml_service && python3 research/full_retrain_report.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WF_DIR = DATA_DIR / "walkforward"

INITIAL_CASH = 100_000.0
HOLD_DAYS = 10
EW_TOP_N = 5
SW_TOP_N = 10
MAX_SINGLE_NAME_PCT = 0.25
MAX_SECTOR_PCT = 0.35
MAX_BETA = 1.2


def log(msg: str):
    print(msg, flush=True)


# ── Position Sizing ──────────────────────────────────────────────────────────

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


# ── HMM ──────────────────────────────────────────────────────────────────────

def load_hmm_exposures(prediction_dates):
    from hmmlearn.hmm import GaussianHMM
    import yfinance as yf
    from massive_data_provider import MassiveDataProvider

    # Build HMM features
    provider = MassiveDataProvider(validate_vs_yfinance=False)
    etf_bars = provider.fetch_bars_batch(["HYG", "LQD", "SPY"], warmup_days=4000)
    vix_raw = yf.download(["^VIX", "^VIX3M"], start="2013-01-01", end="2027-01-01",
                           progress=False, auto_adjust=True)
    vix_close = vix_raw["Close"]
    vix_close.index = pd.to_datetime(vix_close.index).tz_localize(None)

    spy_close = etf_bars["SPY"]["close"]
    hyg_close = etf_bars["HYG"]["close"]
    lqd_close = etf_bars["LQD"]["close"]

    all_close = pd.DataFrame({
        "SPY": spy_close, "HYG": hyg_close, "LQD": lqd_close,
        "^VIX": vix_close["^VIX"], "^VIX3M": vix_close["^VIX3M"],
    }).dropna()

    hmm_df = pd.DataFrame(index=all_close.index)
    spy = all_close["SPY"]
    hmm_df["spy_ret_5d"] = spy.pct_change(5)
    hmm_df["spy_vol_20d"] = spy.pct_change().rolling(20).std() * np.sqrt(252)
    hmm_df["spy_dd_6m"] = (spy - spy.rolling(126, min_periods=20).max()) / spy.rolling(126, min_periods=20).max()
    hmm_df["vix"] = all_close["^VIX"]
    hmm_df["vix_term_structure"] = all_close["^VIX3M"] / all_close["^VIX"]
    hmm_df["hyg_lqd_change_20d"] = (all_close["HYG"] / all_close["LQD"]).pct_change(20)

    price_file = DATA_DIR / "cache_prices_costmodel.parquet"
    if price_file.exists():
        all_prices = pd.read_parquet(price_file)
        all_prices.index = pd.to_datetime(all_prices.index)
        breadth = (all_prices.pct_change(60) > 0).mean(axis=1)
        hmm_df["breadth"] = breadth.reindex(hmm_df.index, method="ffill")
    else:
        hmm_df["breadth"] = (spy.pct_change(60) > 0).astype(float).rolling(20).mean()

    hmm_df = hmm_df.dropna()

    # Fit HMM and get exposures (3st/min50%/lin)
    feature_cols = list(hmm_df.columns)
    all_dates = hmm_df.index.sort_values()
    exposures = {}
    last_model = None
    last_fit_month = None

    for date in sorted(set(prediction_dates)):
        if date not in hmm_df.index:
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
            train_data = hmm_df.loc[all_dates[train_mask], feature_cols].values
            if len(train_data) < 252: exposures[date] = 1.0; continue
            mu = train_data.mean(axis=0)
            sigma = train_data.std(axis=0); sigma[sigma == 0] = 1.0
            try:
                model = GaussianHMM(n_components=3, covariance_type="full",
                                     n_iter=100, random_state=42, verbose=False)
                model.fit((train_data - mu) / sigma)
                last_model, last_fit_month = model, current_month
                last_mu, last_sigma = mu, sigma
            except Exception:
                exposures[date] = 1.0; continue

        history = hmm_df.loc[all_dates[all_dates <= date_lookup], feature_cols].values
        try:
            posteriors = last_model.predict_proba((history - last_mu) / last_sigma)
            bull_state = np.argmax(last_model.means_[:, 0])
            p_bull = posteriors[-1, bull_state]
        except Exception:
            p_bull = 0.5
        exposures[date] = 0.50 + 0.50 * p_bull

    return exposures


# ── Backtester ───────────────────────────────────────────────────────────────

def run_backtest(preds, prices, sector_map, vol_matrix, betas,
                 mode="equal_weight", hmm_exposures=None):
    preds_sp500 = preds[preds["in_sp500"] == True].copy()
    rebal_dates = sorted(preds_sp500["date"].unique())
    rebal_dates_set = set(d for i, d in enumerate(rebal_dates) if i % HOLD_DAYS == 0)
    trading_dates = [d for d in sorted(prices.index)
                     if preds_sp500["date"].min() <= d <= preds_sp500["date"].max()]

    cash = INITIAL_CASH; holdings = {}; port_values = []; prev_value = INITIAL_CASH

    for date in trading_dates:
        day_prices = prices.loc[date] if date in prices.index else pd.Series(dtype=float)
        if date in rebal_dates_set:
            day_preds = preds_sp500[preds_sp500["date"] == date].sort_values("prob_ensemble", ascending=False)
            if len(day_preds) == 0: port_values.append((date, prev_value)); continue
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
                tw = equal_weight_allocation(candidates, n=EW_TOP_N)
            else:
                tw = signal_weighted_allocation(candidates, sector_map, betas_today, n=SW_TOP_N)
            if hmm_exposures is not None:
                exp = hmm_exposures.get(date, 1.0)
                tw = {s: w * exp for s, w in tw.items()}
            total_value = cash
            for sym, h in holdings.items():
                px = day_prices.get(sym, h.get("ep", 0))
                if pd.isna(px): px = h.get("ep", 0)
                total_value += h["sh"] * px
            cash = total_value; holdings = {}
            for sym, w in tw.items():
                px = day_prices.get(sym)
                if px is None or pd.isna(px) or px <= 0: continue
                alloc = total_value * w
                holdings[sym] = {"sh": alloc / px, "ep": px}; cash -= alloc
        total_value = cash
        for sym, h in holdings.items():
            px = day_prices.get(sym, h.get("ep", 0))
            if pd.isna(px): px = h.get("ep", 0)
            total_value += h["sh"] * px
        port_values.append((date, total_value)); prev_value = total_value

    vals = pd.Series([v for _, v in port_values], index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0: years = 1.0
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd_s = dr[dr < 0].std()
    sortino = dr.mean() / dd_s * np.sqrt(252) if dd_s > 0 else np.nan
    peak = vals.cummax(); max_dd = ((vals - peak) / peak).min()
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
            "vol": dr.std() * np.sqrt(252), "years": years}


def run_per_year(preds, prices, sector_map, vol_matrix, betas, mode, hmm_exposures=None):
    results = []
    for year in range(2020, 2026):
        yp = preds[(preds["date"] >= f"{year}-01-01") & (preds["date"] <= f"{year}-12-31")]
        if len(yp) < 50: continue
        m = run_backtest(yp, prices, sector_map, vol_matrix, betas, mode=mode, hmm_exposures=hmm_exposures)
        m["year"] = year; results.append(m)
    return results


# ── Feature Importance ───────────────────────────────────────────────────────

def get_feature_importance():
    import joblib
    model_file = DATA_DIR / "model_base_lgbm.pkl"
    if not model_file.exists():
        return None
    wrapper = joblib.load(str(model_file))
    ranker = wrapper.ranker
    importance = ranker.feature_importances_

    from train_production_model import get_feature_cols, SECTOR_FEATURE_COLS
    feat = pd.read_parquet(DATA_DIR / "features.parquet", columns=["date"])
    # Approximate feature names from training
    feat_full = pd.read_parquet(DATA_DIR / "features.parquet")
    base_cols = get_feature_cols(feat_full, exclude_cols=set(SECTOR_FEATURE_COLS))

    if len(importance) == len(base_cols):
        return pd.Series(importance, index=base_cols).sort_values(ascending=False)
    return pd.Series(importance).sort_values(ascending=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  FULL RETRAIN REPORT — Survivorship-Fixed LambdaRank Model")
    log("=" * 80)

    # Load data
    log("\n1. Loading data ...")
    lr_preds = pd.read_parquet(WF_DIR / "predictions_walkforward_all.parquet")
    lr_preds["date"] = pd.to_datetime(lr_preds["date"])
    log(f"  Predictions: {len(lr_preds):,} rows, {lr_preds['symbol'].nunique()} symbols")
    log(f"  Date range: {lr_preds['date'].min().date()} -> {lr_preds['date'].max().date()}")

    # Load previous predictions if available
    old_preds_file = WF_DIR / "predictions_walkforward_lambdarank.parquet"
    has_old = old_preds_file.exists()
    if has_old:
        old_preds = pd.read_parquet(old_preds_file)
        old_preds["date"] = pd.to_datetime(old_preds["date"])
        log(f"  Old predictions: {len(old_preds):,} rows")

    # Prices
    from massive_data_provider import MassiveDataProvider
    provider = MassiveDataProvider(validate_vs_yfinance=False)
    all_syms = sorted(lr_preds["symbol"].unique().tolist())
    log("  Loading prices ...")
    bars = provider.fetch_bars_batch(list(set(all_syms + ["SPY"])), warmup_days=4000)
    close_frames = {sym: df["close"] for sym, df in bars.items() if len(df) > 0}
    prices = pd.DataFrame(close_frames)
    prices.index = pd.to_datetime(prices.index)

    # Sectors
    sector_map = provider.build_sector_map(all_syms)

    # Vol matrix
    feat_vol = pd.read_parquet(DATA_DIR / "features.parquet", columns=["date", "symbol", "vol_20d"])
    feat_vol["date"] = pd.to_datetime(feat_vol["date"])
    vol_matrix = feat_vol.pivot_table(index="date", columns="symbol", values="vol_20d", aggfunc="last")

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

    # HMM exposures
    log("\n2. Computing HMM exposures ...")
    prediction_dates = sorted(lr_preds["date"].unique().tolist())
    hmm_exposures = load_hmm_exposures(prediction_dates)
    log(f"  Mean exposure: {np.mean(list(hmm_exposures.values())):.2f}")

    # ── Run all configs ──
    log("\n3. Running all configs ...")
    combos = [
        ("LR + EW",          "equal_weight",   None),
        ("LR + EW + HMM",   "equal_weight",   hmm_exposures),
        ("LR + SW",          "signal_weighted", None),
        ("LR + SW + HMM",   "signal_weighted", hmm_exposures),
    ]

    results = []
    for name, mode, hm in combos:
        log(f"  {name} ...")
        cont = run_backtest(lr_preds, prices, sector_map, vol_matrix, betas, mode=mode, hmm_exposures=hm)
        yearly = run_per_year(lr_preds, prices, sector_map, vol_matrix, betas, mode=mode, hmm_exposures=hm)
        med_cagr = np.median([y["cagr"] for y in yearly])
        med_sharpe = np.median([y["sharpe"] for y in yearly])
        pos = sum(1 for y in yearly if y["cagr"] > 0)
        results.append({"name": name, "cont": cont, "yearly": yearly,
                        "med_cagr": med_cagr, "med_sharpe": med_sharpe, "pos": pos, "n": len(yearly)})
        log(f"    Cont: CAGR={cont['cagr']:+.1%} Sharpe={cont['sharpe']:.2f} DD={cont['max_dd']:.1%}")
        log(f"    Med:  CAGR={med_cagr:+.1%} Sharpe={med_sharpe:.2f} Pos={pos}/{len(yearly)}")

    # ── Per-year detail ──
    log(f"\n{'='*80}")
    log("  PER-YEAR RESULTS (NEW - Survivorship Fixed)")
    log(f"{'='*80}")
    for r in results:
        log(f"\n  {r['name']}:")
        log(f"    {'Year':<6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7}")
        for y in r["yearly"]:
            log(f"    {y['year']:<6} {y['cagr']:>+7.1%} {y['sharpe']:>7.2f} {y['max_dd']:>6.1%}")

    # ── Summary table ──
    log(f"\n{'='*80}")
    log("  FULL COMPARISON TABLE")
    log(f"{'='*80}")
    log(f"\n  {'Config':<20} {'C.CAGR':>7} {'C.Shp':>6} {'C.Sort':>7} {'C.DD':>6} {'M.CAGR':>7} {'M.Shp':>6} {'Pos':>4}")
    log(f"  {'─'*20} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*4}")
    for r in results:
        c = r["cont"]
        log(f"  {r['name']:<20} {c['cagr']:>+6.1%} {c['sharpe']:>6.2f} {c['sortino']:>7.2f} "
            f"{c['max_dd']:>5.1%} {r['med_cagr']:>+6.1%} {r['med_sharpe']:>6.2f} {r['pos']:>2}/{r['n']}")

    # ── Side-by-side vs previous (if available) ──
    if has_old:
        log(f"\n{'='*80}")
        log("  COMPARISON: NEW (survivorship fixed) vs OLD (yfinance, no fix)")
        log(f"{'='*80}")
        log("\n  Note: Old model had 11 years (2015-2025), new has 6 years (2020-2025)")
        log("  Direct comparison only on overlapping years 2020-2025")

        # Run old predictions through same backtester for fair comparison
        old_2020 = old_preds[(old_preds["date"] >= "2020-01-01")]
        new_2020 = lr_preds  # already 2020+

        for mode_name, mode in [("EW", "equal_weight"), ("SW", "signal_weighted")]:
            log(f"\n  {mode_name} mode (2020-2025):")
            old_cont = run_backtest(old_2020, prices, sector_map, vol_matrix, betas, mode=mode)
            new_cont = run_backtest(new_2020, prices, sector_map, vol_matrix, betas, mode=mode)
            log(f"    {'':15} {'Old':>8} {'New':>8} {'Delta':>8}")
            log(f"    {'CAGR':15} {old_cont['cagr']:>+7.1%} {new_cont['cagr']:>+7.1%} {new_cont['cagr']-old_cont['cagr']:>+7.1%}")
            log(f"    {'Sharpe':15} {old_cont['sharpe']:>8.2f} {new_cont['sharpe']:>8.2f} {new_cont['sharpe']-old_cont['sharpe']:>+8.2f}")
            log(f"    {'Max DD':15} {old_cont['max_dd']:>7.1%} {new_cont['max_dd']:>7.1%} {new_cont['max_dd']-old_cont['max_dd']:>+7.1%}")

    # ── Feature importance ──
    log(f"\n{'='*80}")
    log("  FEATURE IMPORTANCE (Top 20)")
    log(f"{'='*80}")
    fi = get_feature_importance()
    if fi is not None:
        log(f"\n  {'Rank':<5} {'Feature':<30} {'Importance':>10}")
        log(f"  {'─'*5} {'─'*30} {'─'*10}")
        for i, (feat, imp) in enumerate(fi.head(20).items(), 1):
            log(f"  {i:<5} {feat:<30} {imp:>10.0f}")

        # Check fundamental features
        fund_feats = ['revenue_growth_yoy', 'eps_growth_yoy', 'gross_margin',
                      'operating_margin', 'net_margin', 'pe_ratio', 'debt_to_equity',
                      'roe', 'roa', 'days_since_earnings', 'eps_surprise_last']
        log(f"\n  Fundamental feature rankings:")
        for feat in fund_feats:
            if feat in fi.index:
                rank = (fi.index.tolist().index(feat)) + 1
                log(f"    {feat:<30} rank #{rank}")

    # ── Deploy gate ──
    log(f"\n{'='*80}")
    log("  DEPLOY GATE")
    log(f"{'='*80}")

    # Simple deploy gate: check 90-day backtest
    recent_preds = lr_preds[lr_preds["date"] >= lr_preds["date"].max() - pd.Timedelta(days=120)]
    if len(recent_preds) > 50:
        gate = run_backtest(recent_preds, prices, sector_map, vol_matrix, betas, mode="equal_weight")
        gate_sharpe = gate["sharpe"]
        gate_dd = gate["max_dd"]
        gate_cagr = gate["cagr"]

        passed = True
        reasons = []
        if gate_sharpe < 0:
            passed = False; reasons.append(f"Sharpe {gate_sharpe:.2f} < 0")
        if gate_dd < -0.30:
            passed = False; reasons.append(f"DD {gate_dd:.1%} < -30%")

        log(f"\n  90-day backtest:")
        log(f"    CAGR:   {gate_cagr:+.1%}")
        log(f"    Sharpe: {gate_sharpe:.2f}")
        log(f"    Max DD: {gate_dd:.1%}")
        log(f"    Verdict: {'PASS' if passed else 'FAIL'}")
        if not passed:
            for r in reasons:
                log(f"    Failed: {r}")
    else:
        log("  Insufficient recent data for deploy gate")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
