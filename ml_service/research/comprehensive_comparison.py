#!/usr/bin/env python3
"""
Comprehensive Comparison: Binary vs LambdaRank × 3 Position Sizing Modes
========================================================================
Generates per-year median AND continuous metrics for all 6 combinations:

  1. Binary + equal-weight top-5
  2. Binary + signal-weighted + caps
  3. Binary + signal-weighted + vol target
  4. LambdaRank + equal-weight top-5
  5. LambdaRank + signal-weighted + caps
  6. LambdaRank + signal-weighted + vol target

Step 1: Re-train binary walk-forward (saves predictions separately)
Step 2: Load both prediction sets
Step 3: Run all 6 combos through per-year and continuous backtests
Step 4: Print summary table

Run:
    cd ml_service && python3 research/comprehensive_comparison.py
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
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified_backtester import load_bars_cached

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WF_DIR = DATA_DIR / "walkforward"

INITIAL_CASH = 100_000.0
HOLD_DAYS = 10
PURGE_TRADING_DAYS = 10

# Position sizing configs
EW_TOP_N = 5
SW_TOP_N = 10
MAX_SINGLE_NAME_PCT = 0.25
MAX_SECTOR_PCT = 0.35
MAX_BETA = 1.2
VOL_TARGET_ANN = 0.15
TRADING_DAYS = 252

# Binary model params
BINARY_LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, objective="binary", metric="auc",
    random_state=42, n_jobs=-1, verbose=-1,
)
BINARY_XGB_PARAMS = dict(
    n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
    n_jobs=1, random_state=42, eval_metric="auc",
)

EARLY_STOP_ROUNDS = 100
N_DECILES = 10
YEARS = list(range(2015, 2026))

FUNDAMENTAL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy", "revenue_growth_qoq",
    "gross_margin", "operating_margin", "net_margin", "margin_trend_4q",
    "pe_ratio", "ps_ratio", "pe_vs_universe_median", "ps_vs_universe_median",
    "debt_to_equity", "current_ratio", "roe", "roa",
    "days_since_earnings", "eps_surprise_last",
    "eps_revision_30d", "revenue_revision_30d",
    "insider_buy_ratio_90d", "insider_net_shares_90d",
]

SECTOR_FEATURE_COLS = [
    "ret_10d_vs_sector", "ret_20d_vs_sector",
    "rsi_14_vs_sector", "vol_20d_vs_sector",
]

RANK_FEATURES = [
    ("vol_20d", "vol_rank_20d"), ("ret_60d", "momentum_rank_60d"),
    ("rsi_14", "rsi_rank"), ("dist_sma50", "dist_sma50_rank"),
]

BLEND_WEIGHT_BASE = 0.4
BLEND_WEIGHT_SECTOR = 0.6


def log(msg: str):
    print(msg, flush=True)


def get_feature_cols(df, exclude_cols=None):
    exclude = {"date", "symbol", "target", "target_v5", "target_rank",
               "in_sp500", "pct_rank"}
    if exclude_cols:
        exclude.update(exclude_cols)
    fwd = {"fwd", "forward", "future"}
    return [c for c in df.columns
            if c not in exclude and not any(k in c.lower() for k in fwd)]


# ══════════════════════════════════════════════════════════════════════════════
#  Binary Walk-Forward (inline, saves to separate file)
# ══════════════════════════════════════════════════════════════════════════════

def train_binary_walkforward(df, base_feature_cols, all_feature_cols):
    """Train binary classification walk-forward, return all predictions."""
    all_preds = []

    for year in YEARS:
        train_end = pd.Timestamp(f"{year - 1}-12-31")
        calib_start = pd.Timestamp(f"{year - 2}-01-01")
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")

        train_all = df[df["date"] <= train_end].copy()
        sp500_train = train_all[train_all["in_sp500"] == True].copy()
        sp500_train = sp500_train.sort_values("date").reset_index(drop=True)

        train_dates_arr = np.sort(sp500_train["date"].unique())
        calib_boundary_idx = np.searchsorted(train_dates_arr, calib_start)
        purge_train_end = pd.Timestamp(train_dates_arr[max(0, calib_boundary_idx - PURGE_TRADING_DAYS)])
        purge_calib_start = pd.Timestamp(train_dates_arr[min(len(train_dates_arr) - 1, calib_boundary_idx + PURGE_TRADING_DAYS)])
        calib_end_idx = len(train_dates_arr) - 1
        purge_calib_end = pd.Timestamp(train_dates_arr[max(0, calib_end_idx - PURGE_TRADING_DAYS)])

        pure_train_mask = sp500_train["date"] < purge_train_end
        calib_mask = (sp500_train["date"] >= purge_calib_start) & (sp500_train["date"] <= purge_calib_end)

        year_mask = (df["date"] >= year_start) & (df["date"] <= year_end)
        df_year = df[year_mask].copy()

        if pure_train_mask.sum() < 1000 or calib_mask.sum() < 100 or len(df_year) < 50:
            continue

        y_train = sp500_train.loc[pure_train_mask, "target_v5"].values.astype(int)
        y_calib = sp500_train.loc[calib_mask, "target_v5"].values.astype(int)
        scale = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

        log(f"    {year}: train={pure_train_mask.sum():,} calib={calib_mask.sum():,} pred={len(df_year):,}")

        def _train_binary(feat_cols, label):
            X_tr = sp500_train.loc[pure_train_mask, feat_cols].values
            X_cal = sp500_train.loc[calib_mask, feat_cols].values
            imp = SimpleImputer(strategy="median")
            X_tr_imp = imp.fit_transform(X_tr)
            X_cal_imp = imp.transform(X_cal)

            m_lgb = lgb.LGBMClassifier(**BINARY_LGB_PARAMS, scale_pos_weight=scale)
            m_lgb.fit(X_tr_imp, y_train, eval_set=[(X_cal_imp, y_calib)],
                      callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                                 lgb.log_evaluation(period=-1)])
            cal_lgbm = CalibratedClassifierCV(m_lgb, method="isotonic", cv="prefit")
            cal_lgbm.fit(X_cal_imp, y_calib)

            m_xgb = xgb.XGBClassifier(**BINARY_XGB_PARAMS, scale_pos_weight=scale)
            m_xgb.fit(X_tr_imp, y_train)
            cal_xgb = CalibratedClassifierCV(m_xgb, method="isotonic", cv="prefit")
            cal_xgb.fit(X_cal_imp, y_calib)

            return cal_lgbm, cal_xgb, imp

        base_lgbm, base_xgb, imp_base = _train_binary(base_feature_cols, "Base")
        sect_lgbm, sect_xgb, imp_sect = _train_binary(all_feature_cols, "Sector")

        X_year_base = imp_base.transform(df_year[base_feature_cols].values)
        X_year_sect = imp_sect.transform(df_year[all_feature_cols].values)

        base_probs = 0.5 * base_lgbm.predict_proba(X_year_base)[:, 1] + \
                     0.5 * base_xgb.predict_proba(X_year_base)[:, 1]
        sect_probs = 0.5 * sect_lgbm.predict_proba(X_year_sect)[:, 1] + \
                     0.5 * sect_xgb.predict_proba(X_year_sect)[:, 1]
        ensemble = BLEND_WEIGHT_BASE * base_probs + BLEND_WEIGHT_SECTOR * sect_probs

        df_year["prob_ensemble"] = ensemble
        df_year["prob_base"] = base_probs
        df_year["prob_sector"] = sect_probs

        save_cols = ["date", "symbol", "target_v5", "prob_base", "prob_sector",
                     "prob_ensemble", "fwd_ret", "in_sp500"]
        all_preds.append(df_year[save_cols].copy())

    return pd.concat(all_preds, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Position Sizing (copied from position_sizing_backtest.py)
# ══════════════════════════════════════════════════════════════════════════════

def equal_weight_allocation(candidates, n=5):
    top = candidates[:n]
    if not top:
        return {}
    w = 1.0 / len(top)
    return {sym: w for sym, _, _, _ in top}


def signal_weighted_allocation(candidates, sector_map, betas_today, n=10,
                                apply_vol_target=False, port_vol_20d=None):
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

    if apply_vol_target and port_vol_20d is not None and port_vol_20d > 0:
        target_daily_vol = VOL_TARGET_ANN / np.sqrt(TRADING_DAYS)
        vol_scale = min(target_daily_vol / port_vol_20d, 1.0)
        for sym in weights:
            weights[sym] *= vol_scale

    return {s: w for s, w in weights.items() if w > 0.01}


# ══════════════════════════════════════════════════════════════════════════════
#  Backtester
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(preds, prices, sector_map, vol_matrix, betas, mode="equal_weight"):
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
    recent_port_returns = []

    for date in trading_dates:
        day_prices = prices.loc[date] if date in prices.index else pd.Series(dtype=float)

        if date in rebal_dates:
            day_preds = preds_sp500[preds_sp500["date"] == date].sort_values("prob_ensemble", ascending=False)
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
                target_weights = signal_weighted_allocation(candidates, sector_map, betas_today, n=SW_TOP_N)
            elif mode == "signal_weighted_voltarget":
                port_vol = np.std(recent_port_returns[-20:]) if len(recent_port_returns) >= 10 else 0.01
                target_weights = signal_weighted_allocation(candidates, sector_map, betas_today, n=SW_TOP_N,
                                                            apply_vol_target=True, port_vol_20d=port_vol)
            else:
                raise ValueError(mode)

            total_value = cash
            for sym, h in holdings.items():
                px = day_prices.get(sym, h.get("entry_px", 0))
                if pd.isna(px): px = h.get("entry_px", 0)
                total_value += h["shares"] * px

            cash = total_value
            holdings = {}
            for sym, w in target_weights.items():
                px = day_prices.get(sym)
                if px is None or pd.isna(px) or px <= 0: continue
                alloc = total_value * w
                holdings[sym] = {"shares": alloc / px, "entry_px": px}
                cash -= alloc

        total_value = cash
        for sym, h in holdings.items():
            px = day_prices.get(sym, h.get("entry_px", 0))
            if pd.isna(px): px = h.get("entry_px", 0)
            total_value += h["shares"] * px

        port_values.append((date, total_value))
        daily_ret = (total_value / prev_value) - 1.0 if prev_value > 0 else 0.0
        recent_port_returns.append(daily_ret)
        if len(recent_port_returns) > 60:
            recent_port_returns = recent_port_returns[-60:]
        prev_value = total_value

    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0: years = 1.0

    daily_ret = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    realized_vol = daily_ret.std() * np.sqrt(252)

    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "vol": realized_vol, "years": years}


def run_per_year(preds, prices, sector_map, vol_matrix, betas, mode):
    results = []
    for year in YEARS:
        ys, ye = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        yp = preds[(preds["date"] >= ys) & (preds["date"] <= ye)]
        if len(yp) < 50: continue
        m = run_backtest(yp, prices, sector_map, vol_matrix, betas, mode=mode)
        m["year"] = year
        results.append(m)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  COMPREHENSIVE COMPARISON")
    log("  Binary vs LambdaRank × 3 Position Sizing = 6 combinations")
    log("  Per-year median + continuous metrics")
    log("=" * 80)

    # ── Load and prepare features ──
    log("\n1. Loading features ...")
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    for base_col, rank_col in RANK_FEATURES:
        if base_col in df.columns:
            df[rank_col] = df.groupby("date")[base_col].rank(pct=True)

    df["fwd_10d_ret"] = df.groupby("symbol")["ret_10d"].shift(-10)
    sp500_mask = df["in_sp500"] == True
    has_fwd = df["fwd_10d_ret"].notna()
    df["target_v5"] = np.nan
    df["target_rank"] = np.nan
    valid = df[sp500_mask & has_fwd].copy()
    pct = valid.groupby("date")["fwd_10d_ret"].rank(pct=True)
    valid["target_v5"] = (pct >= 0.80).astype(int)
    valid["target_rank"] = np.clip((pct * N_DECILES).astype(int), 0, N_DECILES - 1)
    df.loc[valid.index, "target_v5"] = valid["target_v5"]
    df.loc[valid.index, "target_rank"] = valid["target_rank"]
    df["fwd_ret"] = df["fwd_10d_ret"]

    all_feature_cols = get_feature_cols(df)
    base_feature_cols = get_feature_cols(df, exclude_cols=set(SECTOR_FEATURE_COLS))
    non_fund = [c for c in all_feature_cols if c not in FUNDAMENTAL_FEATURE_COLS]
    df = df.dropna(subset=non_fund + ["target_v5"])
    log(f"  {len(df):,} rows, {df['symbol'].nunique()} symbols")

    # ── Load LambdaRank predictions ──
    log("\n2. Loading LambdaRank predictions ...")
    lr_file = WF_DIR / "predictions_walkforward_lambdarank.parquet"
    lr_preds = pd.read_parquet(lr_file)
    lr_preds["date"] = pd.to_datetime(lr_preds["date"])
    log(f"  LambdaRank: {len(lr_preds):,} rows")

    # ── Train binary walk-forward ──
    log("\n3. Training binary walk-forward (11 years) ...")
    t_bin = time.perf_counter()
    bin_preds = train_binary_walkforward(df, base_feature_cols, all_feature_cols)
    log(f"  Binary: {len(bin_preds):,} rows in {time.perf_counter() - t_bin:.0f}s")

    # Save binary predictions
    bin_file = WF_DIR / "predictions_walkforward_binary.parquet"
    bin_preds.to_parquet(bin_file, index=False)

    # ── Load support data ──
    log("\n4. Loading prices, sectors, vol, betas ...")
    prices = pd.read_parquet(DATA_DIR / "cache_prices_costmodel.parquet")
    prices.index = pd.to_datetime(prices.index)

    with open(DATA_DIR / "cache_sectors.json") as f:
        sector_map = json.load(f)

    feat_vol = pd.read_parquet(DATA_DIR / "features.parquet", columns=["date", "symbol", "vol_20d"])
    feat_vol["date"] = pd.to_datetime(feat_vol["date"])
    vol_matrix = feat_vol.pivot_table(index="date", columns="symbol", values="vol_20d", aggfunc="last")

    # Betas
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
    log(f"  Betas: {betas.shape}")

    # ── Run all 6 combinations ──
    log("\n5. Running all 6 combinations ...")

    combos = [
        ("Binary + EW top-5",           bin_preds, "equal_weight"),
        ("Binary + Signal-weighted",     bin_preds, "signal_weighted"),
        ("Binary + SW + Vol target",     bin_preds, "signal_weighted_voltarget"),
        ("LambdaRank + EW top-5",        lr_preds,  "equal_weight"),
        ("LambdaRank + Signal-weighted", lr_preds,  "signal_weighted"),
        ("LambdaRank + SW + Vol target", lr_preds,  "signal_weighted_voltarget"),
    ]

    results = []
    for name, preds, mode in combos:
        log(f"  {name} ...")

        # Continuous
        cont = run_backtest(preds, prices, sector_map, vol_matrix, betas, mode=mode)

        # Per-year
        yearly = run_per_year(preds, prices, sector_map, vol_matrix, betas, mode=mode)
        med_cagr = np.median([y["cagr"] for y in yearly])
        med_sharpe = np.median([y["sharpe"] for y in yearly])
        worst_dd = min(y["max_dd"] for y in yearly)
        pos_years = sum(1 for y in yearly if y["cagr"] > 0)

        results.append({
            "name": name,
            "cont_cagr": cont["cagr"],
            "cont_sharpe": cont["sharpe"],
            "cont_dd": cont["max_dd"],
            "cont_vol": cont["vol"],
            "med_cagr": med_cagr,
            "med_sharpe": med_sharpe,
            "worst_yr_dd": worst_dd,
            "pos_years": pos_years,
            "n_years": len(yearly),
            "yearly": yearly,
        })

        log(f"    Continuous: CAGR={cont['cagr']:+.1%}  Sharpe={cont['sharpe']:.2f}  DD={cont['max_dd']:.1%}")
        log(f"    Per-year:   Med CAGR={med_cagr:+.1%}  Med Sharpe={med_sharpe:.2f}  Pos={pos_years}/{len(yearly)}")

    # ── Summary Table ──
    log(f"\n{'='*80}")
    log("  FINAL COMPARISON TABLE")
    log(f"{'='*80}")

    log(f"\n  {'Combination':<35} {'Cont':>5} {'Cont':>6} {'Cont':>6} {'Med':>6} {'Med':>6} {'Worst':>6} {'Pos':>4}")
    log(f"  {'':35} {'CAGR':>5} {'Sharpe':>6} {'DD':>6} {'CAGR':>6} {'Sharpe':>6} {'YrDD':>6} {'Yrs':>4}")
    log(f"  {'─'*35} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*4}")

    for r in results:
        log(f"  {r['name']:<35} "
            f"{r['cont_cagr']:>+4.0%} {r['cont_sharpe']:>6.2f} {r['cont_dd']:>5.0%} "
            f"{r['med_cagr']:>+5.0%} {r['med_sharpe']:>6.2f} {r['worst_yr_dd']:>5.0%} "
            f"{r['pos_years']:>2}/{r['n_years']}")

    # ── Per-year detail for best combos ──
    log(f"\n{'='*80}")
    log("  PER-YEAR DETAIL (top combos)")
    log(f"{'='*80}")

    # Find best continuous Sharpe and best median Sharpe
    best_cont = max(results, key=lambda r: r["cont_sharpe"])
    best_med = max(results, key=lambda r: r["med_sharpe"])

    for r in [results[0], results[3], best_cont, best_med]:
        if r in [best_cont, best_med] and r in [results[0], results[3]]:
            continue
        log(f"\n  {r['name']}:")
        log(f"    {'Year':<6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7}")
        for y in r["yearly"]:
            log(f"    {y['year']:<6} {y['cagr']:>+7.1%} {y['sharpe']:>7.2f} {y['max_dd']:>6.1%}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
