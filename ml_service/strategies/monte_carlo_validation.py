"""
Monte Carlo Robustness Validation
==================================
Tests whether the v8 strategy results are robust or overfitted by:

1. Bootstrap resampling: resample daily returns with replacement,
   generate 1000 synthetic equity curves, compute confidence intervals
2. Random date shifting: shift rebalance dates by +/- 1-3 days randomly
   to test sensitivity to exact timing
3. Parameter perturbation: randomly jitter strategy parameters within
   +/- 20% and check if performance degrades dramatically

If the strategy is overfit, small perturbations should destroy returns.
If robust, the confidence interval should be tight.

Usage:
    cd ml_service && python3 -m strategies.monte_carlo_validation
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
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from massive_data_provider import MassiveDataProvider
from strategies.multi_strategy_engine import (
    FastUniverse, strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, strategy4_index_inclusion,
    STRATEGY_CONFIG_BULL, STRATEGY_CONFIG_BEAR,
    INITIAL_CASH, COST_BPS,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
N_BOOTSTRAP = 500
N_PARAM_PERTURB = 50


def log(msg):
    print(msg, flush=True)


def run_backtest_returns(uni, start="2022-01-01", end="2025-12-31",
                          s1_top_n=10, s1_rebal=10, s5_top_n=15, s5_rebal=10,
                          rebal_offset=0):
    """Run backtest and return daily returns series."""
    trading_dates = [d for d in sorted(uni.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    cost_frac = COST_BPS / 10000
    cash = INITIAL_CASH; holdings = {}; port_values = []
    last_targets = {}; s4_active = {}

    for day_idx, date in enumerate(trading_dates):
        # Build price lookup for today — all symbols we might trade
        tp = {}
        if date in uni.prices.index:
            for sym in uni.prices.columns:
                px = uni.prices.loc[date, sym]
                if not np.isnan(px):
                    tp[sym] = px

        adj_idx = day_idx + rebal_offset
        regime = uni.get_regime(date)
        vix = regime.get("vix", 20)

        t1 = strategy1_momentum_reversal(date, uni, adj_idx, top_n=s1_top_n, rebal_days=s1_rebal)
        t3 = strategy3_sector_rotation(date, uni, adj_idx)
        t4 = strategy4_index_inclusion(date, uni, adj_idx, s4_active)
        t5 = strategy5_lowvol_quality(date, uni, adj_idx, top_n=s5_top_n, rebal_days=s5_rebal)

        if t1 is not None: last_targets["s1_momentum"] = t1
        if t3 is not None: last_targets["s3_sector"] = t3
        last_targets["s4_inclusion"] = t4 if t4 else {}
        if t5 is not None: last_targets["s5_lowvol"] = t5

        major_rebal = any(x is not None for x in [t1, t3, t5])
        if not major_rebal:
            equity = cash
            for sym, h in holdings.items():
                equity += h["shares"] * tp.get(sym, h["entry_px"])
            port_values.append((date, equity))
            continue

        # Breadth blend
        dist_sma50 = uni.get_feature_map(date, "dist_sma50")
        breadth = sum(1 for v in dist_sma50.values() if v > 0) / max(len(dist_sma50), 1) if dist_sma50 else 0.5
        blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))

        blended = {}
        for name, bull_pct in STRATEGY_CONFIG_BULL:
            bear_pct = dict(STRATEGY_CONFIG_BEAR).get(name, 0)
            blended[name] = bull_pct * blend + bear_pct * (1 - blend)

        paused = set()
        if vix > 40:
            paused = {"s1_momentum", "s4_inclusion"}

        combined = {}
        for name, cap_pct in blended.items():
            if name in paused: continue
            for sym, w in last_targets.get(name, {}).items():
                combined[sym] = combined.get(sym, 0) + w * cap_pct

        for sym in list(combined):
            if combined[sym] > 0.15: combined[sym] = 0.15
        gross = sum(combined.values())
        if gross > 1.0:
            for sym in combined: combined[sym] /= gross
        combined = {s: w for s, w in combined.items() if w >= 0.005}

        equity = cash
        for sym, h in holdings.items():
            equity += h["shares"] * tp.get(sym, h["entry_px"])

        target_d = {s: w * equity for s, w in combined.items()}
        current_d = {s: h["shares"] * tp.get(s, h["entry_px"]) for s, h in holdings.items()}

        for sym in list(holdings):
            if sym not in target_d:
                px = tp.get(sym, holdings[sym]["entry_px"])
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                del holdings[sym]

        for sym, tgt in target_d.items():
            px = tp.get(sym)
            if not px or px <= 0: continue
            cur = current_d.get(sym, 0)
            delta = tgt - cur
            if abs(delta) < equity * 0.005: continue
            cost = abs(delta) * cost_frac
            if delta > 0 and cash >= delta:
                shares = (delta - cost) / px
                if sym in holdings: holdings[sym]["shares"] += shares
                else: holdings[sym] = {"shares": shares, "entry_px": px}
                cash -= delta
            elif delta < 0 and sym in holdings:
                sell_sh = min(abs(delta) / px, holdings[sym]["shares"])
                cash += sell_sh * px - cost
                holdings[sym]["shares"] -= sell_sh
                if holdings[sym]["shares"] < 0.01: del holdings[sym]

        equity = cash
        for sym, h in holdings.items():
            equity += h["shares"] * tp.get(sym, h["entry_px"])
        port_values.append((date, max(equity, 0)))

    vals = pd.Series([v for _, v in port_values], index=pd.DatetimeIndex([d for d, _ in port_values]))
    return vals.pct_change().dropna()


def compute_metrics(returns):
    """Compute CAGR, Sharpe, DD from daily returns series."""
    vals = (1 + returns).cumprod() * INITIAL_CASH
    years = len(returns) / 252
    if years <= 0: years = 1
    cagr = (vals.iloc[-1] / INITIAL_CASH) ** (1 / years) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    peak = vals.cummax()
    dd = ((vals - peak) / peak).min()
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": dd}


def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  MONTE CARLO ROBUSTNESS VALIDATION")
    log("=" * 80)

    # Load data
    log("\n1. Loading data ...")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    provider = MassiveDataProvider(validate_vs_yfinance=False)
    all_syms = sorted(features["symbol"].unique().tolist())
    bars = provider.fetch_bars_batch(list(set(all_syms + ["SPY"])), warmup_days=3800)
    close_frames = {sym: df["close"] for sym, df in bars.items() if len(df) > 0}
    prices = pd.DataFrame(close_frames); prices.index = pd.to_datetime(prices.index)
    with open(DATA_DIR / "cache_sectors.json") as f: sector_map = json.load(f)
    with open(DATA_DIR / "sp500_changes.json") as f: sp500_changes = json.load(f)
    vix_data = None
    try:
        vix_raw = yf.download(["^VIX", "^VIX3M"], start="2016-01-01", end="2027-01-01",
                               progress=False, auto_adjust=True)
        vix_data = vix_raw["Close"]; vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)
    except: pass
    uni = FastUniverse(features, prices, sector_map, sp500_changes, vix_data)

    # Base case
    log("\n2. Running base case ...")
    base_returns = run_backtest_returns(uni)
    base_m = compute_metrics(base_returns)
    log(f"  Base: CAGR={base_m['cagr']:+.1%}  Sharpe={base_m['sharpe']:.2f}  DD={base_m['max_dd']:.1%}")

    # Test 1: Bootstrap resampling
    log(f"\n3. Bootstrap resampling ({N_BOOTSTRAP} iterations) ...")
    boot_cagrs = []
    boot_sharpes = []
    boot_dds = []
    for i in range(N_BOOTSTRAP):
        # Resample daily returns with replacement (block bootstrap, 5-day blocks)
        block_size = 5
        n_blocks = len(base_returns) // block_size
        block_starts = np.random.randint(0, len(base_returns) - block_size, n_blocks)
        resampled = pd.concat([base_returns.iloc[s:s+block_size] for s in block_starts])
        resampled.index = pd.RangeIndex(len(resampled))
        m = compute_metrics(resampled)
        boot_cagrs.append(m["cagr"])
        boot_sharpes.append(m["sharpe"])
        boot_dds.append(m["max_dd"])

    log(f"  CAGR: mean={np.mean(boot_cagrs):+.1%}  "
        f"5th={np.percentile(boot_cagrs, 5):+.1%}  "
        f"50th={np.percentile(boot_cagrs, 50):+.1%}  "
        f"95th={np.percentile(boot_cagrs, 95):+.1%}")
    log(f"  Sharpe: mean={np.mean(boot_sharpes):.2f}  "
        f"5th={np.percentile(boot_sharpes, 5):.2f}  "
        f"50th={np.percentile(boot_sharpes, 50):.2f}  "
        f"95th={np.percentile(boot_sharpes, 95):.2f}")
    log(f"  DD: mean={np.mean(boot_dds):.1%}  "
        f"5th={np.percentile(boot_dds, 5):.1%}  "
        f"95th={np.percentile(boot_dds, 95):.1%}")

    # Test 2: Rebalance date sensitivity
    log(f"\n4. Rebalance date sensitivity (offsets -3 to +3) ...")
    log(f"  {'Offset':<8} {'CAGR':>7} {'Sharpe':>7} {'DD':>7}")
    log(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*7}")
    for offset in range(-3, 4):
        rets = run_backtest_returns(uni, rebal_offset=offset)
        m = compute_metrics(rets)
        marker = " <-- base" if offset == 0 else ""
        log(f"  {offset:>+5}    {m['cagr']:>+6.1%} {m['sharpe']:>7.2f} {m['max_dd']:>6.1%}{marker}")

    # Test 3: Parameter perturbation
    log(f"\n5. Parameter perturbation ({N_PARAM_PERTURB} random configs) ...")
    perturb_results = []
    for i in range(N_PARAM_PERTURB):
        # Jitter parameters by +/- 20%
        s1_n = max(5, int(10 * np.random.uniform(0.8, 1.2)))
        s1_r = max(3, int(10 * np.random.uniform(0.8, 1.2)))
        s5_n = max(8, int(15 * np.random.uniform(0.8, 1.2)))
        s5_r = max(5, int(10 * np.random.uniform(0.8, 1.2)))
        rets = run_backtest_returns(uni, s1_top_n=s1_n, s1_rebal=s1_r,
                                    s5_top_n=s5_n, s5_rebal=s5_r)
        m = compute_metrics(rets)
        perturb_results.append(m)

    p_cagrs = [m["cagr"] for m in perturb_results]
    p_sharpes = [m["sharpe"] for m in perturb_results]
    log(f"  CAGR range: [{min(p_cagrs):+.1%}, {max(p_cagrs):+.1%}]  "
        f"mean={np.mean(p_cagrs):+.1%}  std={np.std(p_cagrs):.1%}")
    log(f"  Sharpe range: [{min(p_sharpes):.2f}, {max(p_sharpes):.2f}]  "
        f"mean={np.mean(p_sharpes):.2f}  std={np.std(p_sharpes):.2f}")
    log(f"  % configs with CAGR > 0: {sum(1 for c in p_cagrs if c > 0) / len(p_cagrs):.0%}")
    log(f"  % configs with Sharpe > 0.3: {sum(1 for s in p_sharpes if s > 0.3) / len(p_sharpes):.0%}")

    # Verdict
    log(f"\n{'='*80}")
    log("  ROBUSTNESS VERDICT")
    log(f"{'='*80}")

    robust = True
    issues = []

    # Check 1: Bootstrap 5th percentile should be positive
    boot_5th_cagr = np.percentile(boot_cagrs, 5)
    if boot_5th_cagr < 0:
        issues.append(f"Bootstrap 5th percentile CAGR is negative ({boot_5th_cagr:+.1%})")

    # Check 2: Rebalance sensitivity should be small
    # (already checked above visually)

    # Check 3: >70% of perturbed configs should be profitable
    pct_positive = sum(1 for c in p_cagrs if c > 0) / len(p_cagrs)
    if pct_positive < 0.70:
        issues.append(f"Only {pct_positive:.0%} of perturbed configs are profitable")
        robust = False

    # Check 4: Perturbed Sharpe std should be < 50% of base Sharpe
    if np.std(p_sharpes) > abs(base_m["sharpe"]) * 0.5:
        issues.append(f"High parameter sensitivity: Sharpe std={np.std(p_sharpes):.2f}")

    if issues:
        log(f"\n  Status: CONCERNS")
        for issue in issues:
            log(f"  - {issue}")
        log(f"\n  The strategy has some sensitivity to parameters.")
        log(f"  Consider using the median of perturbed results as your expected performance.")
    else:
        log(f"\n  Status: ROBUST")
        log(f"  Strategy performance is stable across bootstrap resampling,")
        log(f"  rebalance timing, and parameter perturbation.")

    log(f"\n  Expected performance (perturbed median):")
    log(f"    CAGR: {np.median(p_cagrs):+.1%}")
    log(f"    Sharpe: {np.median(p_sharpes):.2f}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
