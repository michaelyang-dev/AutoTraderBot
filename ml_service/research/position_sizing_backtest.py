#!/usr/bin/env python3
"""
Position Sizing Backtest — Vol-Targeted, Signal-Weighted, Constrained
=====================================================================
Compares three position sizing approaches on the same LambdaRank walk-forward
predictions:

  A) Equal-weight top-5 (current baseline)
  B) Signal-weighted, vol-inverse, with sector/name/beta caps, top-10 candidates
  C) Same as B + portfolio-level vol target (15% annualized)

Uses the walk-forward predictions from walk_forward_validation.py.

Run:
    cd ml_service && python3 research/position_sizing_backtest.py
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

# ── Position sizing configs ──────────────────────────────────────────────────
HOLD_DAYS = 10

# Strategy A: equal-weight top-5 (baseline)
EW_TOP_N = 5

# Strategy B/C: signal-weighted with constraints
SW_TOP_N = 10              # wider candidate net
MAX_SINGLE_NAME_PCT = 0.25  # max 25% in any one stock
MAX_SECTOR_PCT = 0.35       # max 35% in any GICS sector
MAX_BETA = 1.2              # portfolio-level beta cap
VOL_TARGET_ANN = 0.15       # 15% annualized vol target (strategy C only)
TRADING_DAYS = 252


def log(msg: str):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load predictions, prices, features, sectors."""
    # Walk-forward predictions
    wf_file = WF_DIR / "predictions_walkforward_all.parquet"
    if not wf_file.exists():
        sys.exit(f"ERROR: {wf_file} not found")
    preds = pd.read_parquet(wf_file)
    preds["date"] = pd.to_datetime(preds["date"])
    log(f"  Predictions: {len(preds):,} rows, {preds['date'].min().date()} -> {preds['date'].max().date()}")

    # Prices (cached from cost model run)
    price_file = DATA_DIR / "cache_prices_costmodel.parquet"
    if not price_file.exists():
        sys.exit("ERROR: cache_prices_costmodel.parquet not found — run cost_model_comparison.py first")
    prices = pd.read_parquet(price_file)
    prices.index = pd.to_datetime(prices.index)

    # Sectors
    sector_file = DATA_DIR / "cache_sectors.json"
    if not sector_file.exists():
        sys.exit("ERROR: cache_sectors.json not found — run cost_model_comparison.py first")
    with open(sector_file) as f:
        sector_map = json.load(f)

    # Features for vol_20d — pivot to date×symbol DataFrame for fast .loc lookup
    features = pd.read_parquet(DATA_DIR / "features.parquet", columns=["date", "symbol", "vol_20d"])
    features["date"] = pd.to_datetime(features["date"])
    vol_matrix = features.pivot_table(index="date", columns="symbol", values="vol_20d", aggfunc="last")
    log(f"  Vol matrix: {vol_matrix.shape}")

    return preds, prices, sector_map, vol_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  Position Sizing Logic
# ══════════════════════════════════════════════════════════════════════════════

def compute_stock_betas(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Compute rolling 60-day beta to SPY for each stock (vectorized)."""
    returns = prices.pct_change()
    spy_ret = returns["SPY"]
    spy_var = spy_ret.rolling(window, min_periods=30).var()

    # Vectorized: multiply each stock's return by SPY return, then rolling mean
    # cov(X, SPY) = E[X*SPY] - E[X]*E[SPY]
    xy = returns.multiply(spy_ret, axis=0)
    mean_xy = xy.rolling(window, min_periods=30).mean()
    mean_x = returns.rolling(window, min_periods=30).mean()
    mean_spy = spy_ret.rolling(window, min_periods=30).mean()

    cov = mean_xy.subtract(mean_x.multiply(mean_spy, axis=0), axis=0)
    betas = cov.divide(spy_var, axis=0)
    betas["SPY"] = 1.0

    return betas


def equal_weight_allocation(candidates: list, n: int = 5) -> dict:
    """Simple equal-weight top-N."""
    top = candidates[:n]
    if not top:
        return {}
    w = 1.0 / len(top)
    return {sym: w for sym, _, _, _ in top}


def signal_weighted_allocation(candidates: list, sector_map: dict,
                                betas_today: dict, n: int = 10,
                                apply_vol_target: bool = False,
                                port_vol_20d: float = None) -> dict:
    """
    Signal-weighted, vol-inverse allocation with constraints.

    candidates: [(symbol, score, vol_20d, sector), ...] sorted by score desc
    Returns: {symbol: weight} summing to <= 1.0
    """
    top = candidates[:n]
    if not top:
        return {}

    # Step 1: compute raw weights = score_strength / vol
    # For ranking models, score_strength = score relative to candidate pool
    scores = np.array([s for _, s, _, _ in top])
    vols = np.array([max(v, 0.05) for _, _, v, _ in top])  # floor vol at 5%

    # Signal strength: excess over minimum score in pool
    min_score = scores.min()
    excess = scores - min_score + 1e-6  # small epsilon to avoid zero
    raw_w = excess / vols
    raw_w = raw_w / raw_w.sum()  # normalize to sum=1

    syms = [s for s, _, _, _ in top]
    sectors = [sec for _, _, _, sec in top]
    weights = dict(zip(syms, raw_w))

    # Step 2: apply constraints (iterate until stable)
    for _ in range(20):
        changed = False

        # Single-name cap
        for sym in list(weights.keys()):
            if weights[sym] > MAX_SINGLE_NAME_PCT:
                weights[sym] = MAX_SINGLE_NAME_PCT
                changed = True

        # Sector cap
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

        # Beta cap
        port_beta = sum(weights[s] * betas_today.get(s, 1.0) for s in weights)
        if port_beta > MAX_BETA:
            scale = MAX_BETA / port_beta
            for sym in weights:
                weights[sym] *= scale
            changed = True

        # Renormalize so gross exposure <= 1.0
        total_w = sum(weights.values())
        if total_w > 1.0:
            for sym in weights:
                weights[sym] /= total_w
            changed = True

        if not changed:
            break

    # Step 3: vol target (strategy C only)
    if apply_vol_target and port_vol_20d is not None and port_vol_20d > 0:
        target_daily_vol = VOL_TARGET_ANN / np.sqrt(TRADING_DAYS)
        vol_scale = min(target_daily_vol / port_vol_20d, 1.0)  # no leverage
        for sym in weights:
            weights[sym] *= vol_scale

    # Remove tiny weights
    weights = {s: w for s, w in weights.items() if w > 0.01}

    return weights


# ══════════════════════════════════════════════════════════════════════════════
#  Backtester
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(preds: pd.DataFrame, prices: pd.DataFrame,
                 sector_map: dict, vol_matrix: pd.DataFrame,
                 betas: pd.DataFrame, mode: str = "equal_weight") -> dict:
    """
    Run position-level backtest with specified allocation mode.

    mode: "equal_weight", "signal_weighted", "signal_weighted_voltarget"
    """
    preds_sp500 = preds[preds["in_sp500"] == True].copy()
    rebal_dates = sorted(preds_sp500["date"].unique())

    # Filter to every HOLD_DAYS
    rebal_dates = [d for i, d in enumerate(rebal_dates) if i % HOLD_DAYS == 0]

    trading_dates = sorted(prices.index)
    wf_start = preds_sp500["date"].min()
    wf_end = preds_sp500["date"].max()
    trading_dates = [d for d in trading_dates if wf_start <= d <= wf_end]

    cash = INITIAL_CASH
    holdings = {}  # sym -> {"shares": float, "weight": float}
    port_values = []
    daily_returns_list = []
    prev_value = INITIAL_CASH

    # Track portfolio vol for vol-targeting
    recent_port_returns = []

    for date in trading_dates:
        day_prices = prices.loc[date] if date in prices.index else pd.Series(dtype=float)

        # Rebalance?
        if date in rebal_dates:
            # Get candidates
            day_preds = preds_sp500[preds_sp500["date"] == date].copy()
            if len(day_preds) == 0:
                port_values.append((date, prev_value))
                continue

            day_preds = day_preds.sort_values("prob_ensemble", ascending=False)

            # Build candidate list: (symbol, score, vol_20d, sector)
            # Get vol row for this date
            vol_today = vol_matrix.loc[date] if date in vol_matrix.index else pd.Series(dtype=float)

            candidates = []
            for _, row in day_preds.iterrows():
                sym = row["symbol"]
                score = row["prob_ensemble"]
                vol_20d = vol_today.get(sym, 0.20) if len(vol_today) > 0 else 0.20
                if pd.isna(vol_20d) or vol_20d <= 0:
                    vol_20d = 0.20
                sec = sector_map.get(sym, "Unknown")
                candidates.append((sym, score, vol_20d, sec))

            # Get betas for today
            betas_today = {}
            if date in betas.index:
                for sym, _, _, _ in candidates:
                    b = betas.loc[date].get(sym, 1.0)
                    betas_today[sym] = b if not pd.isna(b) else 1.0

            # Compute target weights
            if mode == "equal_weight":
                target_weights = equal_weight_allocation(candidates, n=EW_TOP_N)
            elif mode == "signal_weighted":
                target_weights = signal_weighted_allocation(
                    candidates, sector_map, betas_today, n=SW_TOP_N,
                    apply_vol_target=False)
            elif mode == "signal_weighted_voltarget":
                # Estimate portfolio vol from recent returns
                port_vol = np.std(recent_port_returns[-20:]) if len(recent_port_returns) >= 10 else 0.01
                target_weights = signal_weighted_allocation(
                    candidates, sector_map, betas_today, n=SW_TOP_N,
                    apply_vol_target=True, port_vol_20d=port_vol)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Liquidate all positions and rebuild
            # (simplified: full rebalance each period)
            total_value = cash
            for sym, h in holdings.items():
                px = day_prices.get(sym, h.get("entry_px", 0))
                if pd.isna(px):
                    px = h.get("entry_px", 0)
                total_value += h["shares"] * px

            cash = total_value
            holdings = {}

            # Buy new positions
            for sym, w in target_weights.items():
                px = day_prices.get(sym)
                if px is None or pd.isna(px) or px <= 0:
                    continue
                alloc = total_value * w
                shares = alloc / px
                cash -= alloc
                holdings[sym] = {"shares": shares, "entry_px": px}

        # Mark-to-market
        total_value = cash
        for sym, h in holdings.items():
            px = day_prices.get(sym, h.get("entry_px", 0))
            if pd.isna(px):
                px = h.get("entry_px", 0)
            total_value += h["shares"] * px

        port_values.append((date, total_value))

        # Track daily return for vol targeting
        daily_ret = (total_value / prev_value) - 1.0 if prev_value > 0 else 0.0
        recent_port_returns.append(daily_ret)
        if len(recent_port_returns) > 60:
            recent_port_returns = recent_port_returns[-60:]
        prev_value = total_value

    # Compute metrics
    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0:
        years = 1.0

    daily_ret = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    sortino_denom = daily_ret[daily_ret < 0].std()
    sortino = daily_ret.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else np.nan
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    realized_vol = daily_ret.std() * np.sqrt(252)

    # SPY comparison
    spy_vals = prices["SPY"].reindex(vals.index, method="ffill").dropna()
    if len(spy_vals) > 20:
        spy_vals = spy_vals / spy_vals.iloc[0] * INITIAL_CASH
        from scipy import stats
        strat_ret = vals.pct_change().dropna()
        spy_ret = spy_vals.pct_change().dropna()
        combined = pd.concat([strat_ret, spy_ret], axis=1, join="inner").dropna()
        combined.columns = ["strat", "spy"]
        if len(combined) > 20:
            slope, intercept, *_ = stats.linregress(combined["spy"], combined["strat"])
            alpha = (1 + intercept) ** 252 - 1
            beta = slope
        else:
            alpha, beta = np.nan, np.nan
    else:
        alpha, beta = np.nan, np.nan

    return {
        "mode": mode,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "realized_vol": realized_vol,
        "alpha": alpha,
        "beta": beta,
        "final_value": vals.iloc[-1],
        "years": years,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Per-Year Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_per_year(preds, prices, sector_map, vol_matrix, betas, mode):
    """Run backtest year-by-year for detailed comparison."""
    results = []
    for year in range(2015, 2026):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        year_preds = preds[(preds["date"] >= year_start) & (preds["date"] <= year_end)]
        if len(year_preds) < 50:
            continue

        m = run_backtest(year_preds, prices, sector_map, vol_matrix, betas, mode=mode)
        m["year"] = year
        results.append(m)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()

    log("=" * 75)
    log("  POSITION SIZING COMPARISON")
    log("  Equal-weight vs Signal-weighted vs Vol-targeted")
    log("=" * 75)

    # Load data
    log("\n1. Loading data ...")
    preds, prices, sector_map, vol_matrix = load_all_data()

    # Compute betas
    log("\n2. Computing 60-day rolling betas ...")
    betas = compute_stock_betas(prices, window=60)
    log(f"   Betas computed: {betas.shape}")

    # Run full-period backtests
    log("\n3. Running full-period backtests ...")

    modes = [
        ("equal_weight", "A) Equal-weight top-5"),
        ("signal_weighted", "B) Signal-weighted + constraints"),
        ("signal_weighted_voltarget", "C) Signal-weighted + vol target"),
    ]

    full_results = []
    for mode, label in modes:
        log(f"\n   {label} ...")
        m = run_backtest(preds, prices, sector_map, vol_matrix, betas, mode=mode)
        full_results.append((label, m))
        log(f"     CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:.2f}  "
            f"MaxDD={m['max_dd']:.1%}  Vol={m['realized_vol']:.1%}  "
            f"Beta={m['beta']:.2f}  Alpha={m['alpha']:+.1%}")

    # Per-year comparison
    log(f"\n{'='*75}")
    log("  PER-YEAR COMPARISON")
    log(f"{'='*75}")

    yearly_data = {}
    for mode, label in modes:
        yearly_data[label] = run_per_year(preds, prices, sector_map, vol_matrix, betas, mode)

    # Print per-year table
    log(f"\n  {'Year':<6} {'EW CAGR':>8} {'EW Shp':>7} {'SW CAGR':>8} {'SW Shp':>7} {'VT CAGR':>8} {'VT Shp':>7} {'EW DD':>7} {'SW DD':>7} {'VT DD':>7}")
    log(f"  {'─'*6} {'─'*8} {'─'*7} {'─'*8} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    labels = [l for _, l in modes]
    for i in range(len(yearly_data[labels[0]])):
        ew = yearly_data[labels[0]][i]
        sw = yearly_data[labels[1]][i]
        vt = yearly_data[labels[2]][i]
        log(f"  {ew['year']:<6} "
            f"{ew['cagr']:>+7.1%} {ew['sharpe']:>7.2f} "
            f"{sw['cagr']:>+7.1%} {sw['sharpe']:>7.2f} "
            f"{vt['cagr']:>+7.1%} {vt['sharpe']:>7.2f} "
            f"{ew['max_dd']:>6.1%} {sw['max_dd']:>6.1%} {vt['max_dd']:>6.1%}")

    # Summary
    log(f"\n{'='*75}")
    log("  FULL-PERIOD SUMMARY")
    log(f"{'='*75}")

    log(f"\n  {'Strategy':<40} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Vol':>6} {'Beta':>5} {'Alpha':>7}")
    log(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*6} {'─'*5} {'─'*7}")

    for label, m in full_results:
        log(f"  {label:<40} {m['cagr']:>+6.1%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
            f"{m['max_dd']:>6.1%} {m['realized_vol']:>5.1%} {m['beta']:>5.2f} {m['alpha']:>+6.1%}")

    # Improvement analysis
    ew = full_results[0][1]
    sw = full_results[1][1]
    vt = full_results[2][1]

    log(f"\n  Sharpe improvement:")
    log(f"    Signal-weighted vs equal-weight:   {sw['sharpe'] - ew['sharpe']:+.2f}")
    log(f"    Vol-targeted vs equal-weight:      {vt['sharpe'] - ew['sharpe']:+.2f}")
    log(f"    Vol-targeted vs signal-weighted:   {vt['sharpe'] - sw['sharpe']:+.2f}")

    log(f"\n  Max drawdown improvement:")
    log(f"    Signal-weighted vs equal-weight:   {sw['max_dd'] - ew['max_dd']:+.1%}")
    log(f"    Vol-targeted vs equal-weight:      {vt['max_dd'] - ew['max_dd']:+.1%}")

    # Per-year medians
    for label in labels:
        yearly = yearly_data[label]
        med_cagr = np.median([y["cagr"] for y in yearly])
        med_sharpe = np.median([y["sharpe"] for y in yearly])
        pos_years = sum(1 for y in yearly if y["cagr"] > 0)
        log(f"\n  {label}: Median CAGR={med_cagr:+.1%}, Median Sharpe={med_sharpe:.2f}, "
            f"Positive years={pos_years}/{len(yearly)}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
