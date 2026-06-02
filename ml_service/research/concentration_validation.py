"""
Concentration Validation — Testing the Expert's Statistical Concerns
=====================================================================
1. Bootstrap Sharpe CI: do n=5 and n=30 confidence intervals overlap?
2. Contributor-drop test: remove top 3 lifetime winners, does CAGR collapse?
3. Rolling 5-year windows: does n=5 edge exist in ALL eras or just recent?
4. Deflated Sharpe: adjust for multiple testing across all configs searched
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from fast_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, strategy5_lowvol_quality,
    INITIAL_CASH, COST_BPS,
)
import numpy as np, pandas as pd, time


def run_and_get_monthly(bt, start, end, cfg):
    """Run backtest and return monthly returns series."""
    r = bt.run(start, end, cfg)
    if not r:
        return None, None

    # Re-run to capture daily values
    trading_dates = [d for d in sorted(bt.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    last_targets = {}
    top_n = cfg.get('top_n', 5)
    rebal_days = cfg.get('rebal_days', 20)
    mom_w = cfg.get('mom_w', 0.50)
    val_w = cfg.get('val_w', 0.35)
    lv_w = cfg.get('lv_w', 0.15)
    cap = cfg.get('cap', 0.15)
    trailing_stop = cfg.get('trailing_stop', 0.40)

    # Track which stocks contributed most
    stock_pnl = {}  # {sym: total_pnl}

    for day_idx, date in enumerate(trading_dates):
        today = {}
        if date in bt.prices.index:
            row = bt.prices.loc[date]
            for sym in list(holdings.keys()):
                v = row.get(sym)
                if v is not None and not np.isnan(v):
                    today[sym] = v
            for sym in row.dropna().index:
                today[sym] = row[sym]

        if trailing_stop:
            for sym in list(holdings):
                px = today.get(sym)
                if px:
                    if "peak_px" not in holdings[sym]:
                        holdings[sym]["peak_px"] = px
                    if px > holdings[sym]["peak_px"]:
                        holdings[sym]["peak_px"] = px
                    dd = (px - holdings[sym]["peak_px"]) / holdings[sym]["peak_px"]
                    if dd < -abs(trailing_stop):
                        exit_val = holdings[sym]["shares"] * px * (1 - cost_frac)
                        entry_val = holdings[sym]["shares"] * holdings[sym]["entry_px"]
                        pnl = exit_val - entry_val
                        stock_pnl[sym] = stock_pnl.get(sym, 0) + pnl
                        cash += exit_val
                        del holdings[sym]

        total_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                for s, h in holdings.items())

        if day_idx % rebal_days != 0:
            port_values.append((date, total_val))
            continue

        t1 = strategy1_momentum_reversal(date, bt.uni, day_idx,
                                          top_n=top_n, rebal_days=rebal_days)
        if t1 is None:
            t1 = last_targets.get("mom", {})
        members = bt.uni.get_sp500(date)
        t_val = bt._strategy_value(date, members, top_n=10)
        t5 = strategy5_lowvol_quality(date, bt.uni, day_idx)
        if t5 is None:
            t5 = last_targets.get("s5", {})
        last_targets.update({"mom": t1, "val": t_val, "s5": t5})

        nu = bt.umd_20d.loc[:date]
        in_crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < -0.05
        if in_crash:
            ew = {"mom": 0.15, "val": 0.45, "s5": 0.30}
        else:
            ew = {"mom": mom_w, "val": val_w, "s5": lv_w}

        fdate = bt.features_by_date.get(date, {})
        above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
        total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
        breadth = above / max(total_f, 1)
        blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))
        bear = {"mom": 0.10, "val": 0.20, "s5": 0.60}
        blended = {n: ew[n] * blend + bear.get(n, 0) * (1 - blend) for n in ew}

        combined = {}
        for name, cap_pct in blended.items():
            tgt = last_targets.get(name, {})
            for sym, w in tgt.items():
                if w > 0:
                    combined[sym] = combined.get(sym, 0) + w * cap_pct

        longs = {s: w for s, w in combined.items() if w > 0}
        for sym in list(longs):
            if longs[sym] > cap:
                longs[sym] = cap
        gross = sum(longs.values())
        if gross > 1.0:
            for sym in longs:
                longs[sym] /= gross
        combined = {s: w for s, w in longs.items() if w >= 0.005}

        eq_pct = 1.0
        trend_scale = cfg.get("trend_scale", None)
        if trend_scale and "SPY" in bt.prices.columns:
            spy_px = today.get("SPY", 0)
            spy_hist = bt.prices["SPY"].loc[:date].dropna()
            if len(spy_hist) >= 200:
                spy_sma200 = spy_hist.tail(200).mean()
                if spy_px < spy_sma200:
                    eq_pct *= trend_scale.get("bear", 0.5)

        target_d = {s: w * total_val * eq_pct for s, w in combined.items()}

        for sym in list(holdings):
            if sym not in target_d:
                px = today.get(sym, holdings[sym]["entry_px"])
                exit_val = holdings[sym]["shares"] * px * (1 - cost_frac)
                entry_val = holdings[sym]["shares"] * holdings[sym]["entry_px"]
                pnl = exit_val - entry_val
                stock_pnl[sym] = stock_pnl.get(sym, 0) + pnl
                cash += exit_val
                del holdings[sym]

        for sym, tgt in target_d.items():
            px = today.get(sym)
            if not px or px <= 0:
                continue
            cur_val = holdings[sym]["shares"] * px if sym in holdings else 0
            diff = tgt - cur_val
            if abs(diff) < total_val * 0.01:
                continue
            shares_delta = diff / px
            if sym in holdings:
                holdings[sym]["shares"] += shares_delta
                cost = abs(shares_delta * px) * cost_frac
                cash -= shares_delta * px + cost
            else:
                if shares_delta > 0:
                    cost = shares_delta * px * cost_frac
                    holdings[sym] = {"shares": shares_delta, "entry_px": px}
                    cash -= shares_delta * px + cost

        port_values.append((date, total_val))

    # Convert to monthly returns
    if len(port_values) < 2:
        return None, None
    val_series = pd.Series([v[1] for v in port_values],
                            index=pd.DatetimeIndex([v[0] for v in port_values]))
    monthly = val_series.resample('ME').last().pct_change().dropna()

    return monthly, stock_pnl


def main():
    print("="*70)
    print("CONCENTRATION VALIDATION — EXPERT'S STATISTICAL TESTS")
    print("="*70)

    bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
    bt.uni.get_sp500 = bt._get_sp1500

    cfg5 = {
        'mom_w': 0.50, 'val_w': 0.35, 'lv_w': 0.15, 'sec_w': 0.0,
        'top_n': 5, 'rebal_days': 20, 'trailing_stop': 0.40,
        'cap': 0.15, 'use_rp': False,
        'trend_scale': {'bear': 0.40, 'caution': 0.65},
    }
    cfg30 = {**cfg5, 'top_n': 30}

    # ═══════════════════════════════════════════════════════════
    # TEST 1: Bootstrap Sharpe CI
    # ═══════════════════════════════════════════════════════════
    print("\n[TEST 1] BOOTSTRAP SHARPE CONFIDENCE INTERVALS")
    print("-"*50)

    monthly5, pnl5 = run_and_get_monthly(bt, "2018-01-01", "2025-12-31", cfg5)
    monthly30, pnl30 = run_and_get_monthly(bt, "2018-01-01", "2025-12-31", cfg30)

    if monthly5 is not None and monthly30 is not None:
        n_bootstrap = 10000
        block_size = 3  # 3-month blocks for stationary bootstrap

        def block_bootstrap_sharpe(returns, n_boot=10000, block=3):
            """Stationary block bootstrap for Sharpe ratio."""
            n = len(returns)
            sharpes = []
            vals = returns.values
            for _ in range(n_boot):
                # Generate block bootstrap sample
                sample = []
                while len(sample) < n:
                    start = np.random.randint(0, n)
                    length = np.random.geometric(1.0 / block)
                    for j in range(length):
                        if len(sample) >= n:
                            break
                        sample.append(vals[(start + j) % n])
                sample = np.array(sample[:n])
                s = np.mean(sample) / np.std(sample) * np.sqrt(12) if np.std(sample) > 0 else 0
                sharpes.append(s)
            return np.array(sharpes)

        print("  Running 10,000 block bootstraps (3-month blocks)...")
        boot5 = block_bootstrap_sharpe(monthly5, n_bootstrap, block_size)
        boot30 = block_bootstrap_sharpe(monthly30, n_bootstrap, block_size)

        ci5_lo, ci5_hi = np.percentile(boot5, [2.5, 97.5])
        ci30_lo, ci30_hi = np.percentile(boot30, [2.5, 97.5])
        sharpe5 = monthly5.mean() / monthly5.std() * np.sqrt(12)
        sharpe30 = monthly30.mean() / monthly30.std() * np.sqrt(12)

        overlap = ci5_lo < ci30_hi and ci30_lo < ci5_hi

        print(f"  n=5:  Sharpe = {sharpe5:.2f}  95% CI: [{ci5_lo:.2f}, {ci5_hi:.2f}]")
        print(f"  n=30: Sharpe = {sharpe30:.2f}  95% CI: [{ci30_lo:.2f}, {ci30_hi:.2f}]")
        print(f"  CIs overlap: {'YES' if overlap else 'NO'}")
        if overlap:
            print(f"  → Expert is RIGHT: the Sharpe difference is within statistical noise")
        else:
            print(f"  → n=5 Sharpe is statistically significantly higher")

        # Also bootstrap the DIFFERENCE
        diff_boot = boot5 - boot30
        diff_lo, diff_hi = np.percentile(diff_boot, [2.5, 97.5])
        print(f"  Sharpe difference: {sharpe5 - sharpe30:.2f}  95% CI: [{diff_lo:.2f}, {diff_hi:.2f}]")
        print(f"  Zero in CI: {'YES (not significant)' if diff_lo < 0 else 'NO (significant)'}")

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Contributor-drop test
    # ═══════════════════════════════════════════════════════════
    print(f"\n[TEST 2] CONTRIBUTOR-DROP TEST")
    print("-"*50)

    if pnl5:
        sorted_pnl = sorted(pnl5.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 10 lifetime P&L contributors (n=5 book):")
        total_pnl = sum(v for v in pnl5.values())
        for sym, pnl in sorted_pnl[:10]:
            print(f"    {sym:<8} ${pnl:>12,.0f}  ({pnl/total_pnl*100:.1f}% of total)")

        top3 = [sym for sym, _ in sorted_pnl[:3]]
        top3_pnl = sum(pnl for _, pnl in sorted_pnl[:3])
        print(f"\n  Top 3 total: ${top3_pnl:,.0f} ({top3_pnl/total_pnl*100:.1f}% of all P&L)")
        print(f"  If we remove top 3 ({', '.join(top3)}), remaining P&L: ${total_pnl - top3_pnl:,.0f}")

        # What % of total return came from top 3?
        final_val = INITIAL_CASH + total_pnl
        final_without_top3 = INITIAL_CASH + total_pnl - top3_pnl
        n_years = 8
        cagr_with = (final_val / INITIAL_CASH) ** (1/n_years) - 1
        cagr_without = (final_without_top3 / INITIAL_CASH) ** (1/n_years) - 1
        print(f"\n  CAGR with all stocks:    {cagr_with*100:.1f}%")
        print(f"  CAGR without top 3:      {cagr_without*100:.1f}%")
        print(f"  CAGR collapse:           {(cagr_with - cagr_without)*100:.1f}pp")

        if cagr_without < cagr_with * 0.5:
            print(f"  → Expert is RIGHT: strategy IS three stocks, not a system")
        else:
            print(f"  → Strategy survives top-3 removal (still {cagr_without*100:.1f}% CAGR)")

    # ═══════════════════════════════════════════════════════════
    # TEST 3: Rolling 5-year windows
    # ═══════════════════════════════════════════════════════════
    print(f"\n[TEST 3] ROLLING 5-YEAR WINDOWS")
    print("-"*50)
    print(f"  {'Window':<15} {'n=5':>10} {'n=30':>10} {'Diff':>10} {'Winner':>8}")
    print(f"  {'-'*55}")

    n5_wins = 0
    n30_wins = 0
    windows = []
    for start_year in range(2016, 2022):
        end_year = start_year + 4
        s = f"{start_year}-07-01"
        e = f"{end_year}-12-31"
        r5 = bt.run(s, e, cfg5)
        r30 = bt.run(s, e, cfg30)
        c5 = r5['cagr']*100 if r5 else 0
        c30 = r30['cagr']*100 if r30 else 0
        winner = "n=5" if c5 > c30 else "n=30"
        if c5 > c30: n5_wins += 1
        else: n30_wins += 1
        windows.append((c5, c30))
        print(f"  {start_year}-{end_year}{'':>5} {c5:>+9.1f}% {c30:>+9.1f}% {c5-c30:>+9.1f}% {winner:>8}")

    print(f"\n  n=5 wins: {n5_wins}/{n5_wins+n30_wins} windows")
    if n5_wins <= n30_wins:
        print(f"  → Expert is RIGHT: n=5 edge doesn't persist across eras")
    elif all(w[0] > w[1] for w in windows):
        print(f"  → n=5 wins EVERY window — edge is persistent")
    else:
        print(f"  → Mixed: n=5 wins most but not all windows")

    # ═══════════════════════════════════════════════════════════
    # TEST 4: Deflated Sharpe Ratio
    # ═══════════════════════════════════════════════════════════
    print(f"\n[TEST 4] DEFLATED SHARPE RATIO")
    print("-"*50)

    # How many configs did we search?
    # From grid_search.py: 96 configs tested
    # From earlier parameter sweeps: ~200+ configs total
    n_trials = 200  # conservative estimate
    T = len(monthly5) if monthly5 is not None else 96  # months

    if monthly5 is not None:
        observed_sharpe = sharpe5

        # Expected max Sharpe under null (Harvey & Liu 2015)
        # E[max(SR)] ≈ sqrt(2 * ln(N)) where N = number of trials
        expected_max_sr = np.sqrt(2 * np.log(n_trials))

        # Deflated Sharpe (simplified Bailey & Lopez de Prado 2014)
        # DSR = Prob(SR* > 0 | SR_hat, T, N)
        # Approximate: SR_deflated = SR_observed - E[max(SR)] * sqrt(1/T)
        sr_deflated = observed_sharpe - expected_max_sr * np.sqrt(1/T)

        # Also compute the haircut ratio
        haircut = expected_max_sr * np.sqrt(1/T) / observed_sharpe * 100

        print(f"  Observed Sharpe (n=5): {observed_sharpe:.2f}")
        print(f"  Trials searched: {n_trials}")
        print(f"  Sample months: {T}")
        print(f"  Expected max Sharpe under null: {expected_max_sr:.2f}")
        print(f"  Deflated Sharpe: {sr_deflated:.2f}")
        print(f"  Haircut: {haircut:.0f}%")
        print()
        if sr_deflated > 0:
            print(f"  → Deflated Sharpe > 0: signal survives multiple testing adjustment")
        else:
            print(f"  → Deflated Sharpe < 0: COULD be multiple testing artifact")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("""
    Test 1 (Bootstrap): Do the Sharpe CIs overlap?
    Test 2 (Contributors): Does removing top 3 stocks collapse CAGR?
    Test 3 (Rolling windows): Does n=5 win in ALL eras?
    Test 4 (Deflated Sharpe): Does the Sharpe survive multiple testing?

    If Tests 1+2+3 all favor the expert → we should diversify
    If Tests 1+2+3 all favor n=5 → concentration is a real edge
    If mixed → it's ambiguous and we should be cautious
    """)


if __name__ == "__main__":
    main()
