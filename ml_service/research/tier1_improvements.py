"""
Tier 1 Improvements — Expert Recommendations
==============================================
1. No-trade band: only rebalance when drift > 20-30% (cut turnover)
2. Z-scored value sleeve with real valuation metric
3. Re-test short interest as filter

All tested against v12 baseline with proper start-day averaging.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, strategy5_lowvol_quality,
    INITIAL_CASH, COST_BPS,
)
import numpy as np, pandas as pd, time


def run_with_mods(bt, start, end, cfg, no_trade_band=0.01, use_si=False):
    """Run backtest with configurable no-trade band and SI option."""
    trading_dates = [d for d in sorted(bt.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

    cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
    mom_w = cfg.get("mom_w", 0.50)
    val_w = cfg.get("val_w", 0.35)
    lv_w = cfg.get("lv_w", 0.15)
    top_n = cfg.get("top_n", 5)
    rebal_days = cfg.get("rebal_days", 20)
    trailing_stop = cfg.get("trailing_stop", 0.40)
    cap = cfg.get("cap", 0.15)

    # Enable SI if requested
    if use_si and hasattr(bt, '_si_months_backup'):
        bt._si_months = bt._si_months_backup
        bt._si_ranks_by_month = bt._si_ranks_backup
        bt._si_change_ranks_by_month = bt._si_change_ranks_backup

    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    last_targets = {}
    total_turnover = 0
    total_costs = 0

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
                        sell_val = holdings[sym]["shares"] * px
                        total_turnover += sell_val
                        total_costs += sell_val * cost_frac
                        cash += sell_val * (1 - cost_frac)
                        del holdings[sym]

        total_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                for s, h in holdings.items())

        if day_idx % rebal_days != 0:
            port_values.append((date, total_val))
            continue

        # Update SI if enabled
        if use_si and bt._si_months:
            midx = np.searchsorted(bt._si_months, date, side="right") - 1
            if midx >= 0:
                bt.uni._short_interest_rank = bt._si_ranks_by_month.get(
                    bt._si_months[midx], {})
                bt.uni._si_change_rank = bt._si_change_ranks_by_month.get(
                    bt._si_months[midx], {})

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

        # SELLS — only sell if position dropped out entirely
        for sym in list(holdings):
            if sym not in target_d:
                px = today.get(sym, holdings[sym]["entry_px"])
                sell_val = holdings[sym]["shares"] * px
                total_turnover += sell_val
                total_costs += sell_val * cost_frac
                cash += sell_val * (1 - cost_frac)
                del holdings[sym]

        # BUYS/RESIZES — apply no-trade band
        for sym, tgt in target_d.items():
            px = today.get(sym)
            if not px or px <= 0:
                continue
            cur_val = holdings[sym]["shares"] * px if sym in holdings else 0

            # No-trade band: skip if current is within band % of target
            if cur_val > 0:
                drift = abs(tgt - cur_val) / tgt if tgt > 0 else 1.0
                if drift < no_trade_band:
                    continue  # within band, don't trade

            delta = tgt - cur_val
            if abs(delta) < 100:  # min trade size $100
                continue

            trade_val = abs(delta)
            cost = trade_val * cost_frac
            total_turnover += trade_val
            total_costs += cost

            shares_delta = delta / px
            if sym in holdings:
                holdings[sym]["shares"] += shares_delta
                cash -= delta + cost if delta > 0 else delta - cost
            else:
                if shares_delta > 0:
                    holdings[sym] = {"shares": shares_delta, "entry_px": px, "peak_px": px}
                    cash -= delta + cost

        port_values.append((date, total_val))

    if len(port_values) < 2:
        return None

    dates = [v[0] for v in port_values]
    vals = [v[1] for v in port_values]
    n_years = (dates[-1] - dates[0]).days / 365.25
    cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    rets = pd.Series(vals).pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    peak = pd.Series(vals).cummax()
    max_dd = ((pd.Series(vals) - peak) / peak).min()
    avg_port = np.mean(vals)
    annual_turnover = total_turnover / n_years / avg_port
    annual_cost_drag = total_costs / n_years / avg_port

    return {
        "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
        "annual_turnover": annual_turnover,
        "annual_cost_drag": annual_cost_drag,
    }


def main():
    print("="*70)
    print("TIER 1 IMPROVEMENTS — TESTING")
    print("="*70)

    bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')

    # Save SI data before clearing (for re-testing)
    bt._si_months_backup = list(bt._si_months)
    bt._si_ranks_backup = dict(bt._si_ranks_by_month)
    bt._si_change_ranks_backup = dict(bt._si_change_ranks_by_month)

    # Clear everything
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
    bt.uni.get_sp500 = bt._get_sp1500

    cfg = {
        'mom_w': 0.50, 'val_w': 0.35, 'lv_w': 0.15, 'sec_w': 0.0,
        'top_n': 5, 'rebal_days': 20, 'trailing_stop': 0.40,
        'cap': 0.15, 'use_rp': False,
        'trend_scale': {'bear': 0.40, 'caution': 0.65},
    }

    # ═══════════════════════════════════════════════════════════
    # TEST 1: No-trade bands (cut turnover)
    # ═══════════════════════════════════════════════════════════
    print(f"\n[1] NO-TRADE BAND — CUT TURNOVER")
    print(f"{'Band':<15} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Turnover':>10} {'Cost Drag':>10}")
    print("-"*65)

    for band in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        r = run_with_mods(bt, "2018-01-01", "2025-12-31", cfg, no_trade_band=band)
        if r:
            label = f"{int(band*100)}% band"
            if band == 0.01:
                label = "1% (current)"
            print(f"{label:<15} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}% {r['annual_turnover']*100:>9.0f}% {r['annual_cost_drag']*100:>9.2f}%")

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Short interest re-test
    # ═══════════════════════════════════════════════════════════
    print(f"\n[2] SHORT INTEREST — RE-TEST")
    print(f"{'Config':<20} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*48)

    # Without SI (current)
    r_no_si = run_with_mods(bt, "2018-01-01", "2025-12-31", cfg, use_si=False)
    if r_no_si:
        print(f"{'No SI (current)':<20} {r_no_si['cagr']*100:>7.1f}% {r_no_si['sharpe']:>8.2f} {r_no_si['max_dd']*100:>7.1f}%")

    # With SI
    r_si = run_with_mods(bt, "2018-01-01", "2025-12-31", cfg, use_si=True)
    if r_si:
        print(f"{'With SI':<20} {r_si['cagr']*100:>7.1f}% {r_si['sharpe']:>8.2f} {r_si['max_dd']*100:>7.1f}%")

    # Clear SI again for other tests
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

    # ═══════════════════════════════════════════════════════════
    # START-DAY AVERAGED for best band + baseline
    # ═══════════════════════════════════════════════════════════
    print(f"\n[3] START-DAY AVERAGED (7 offsets)")
    print(f"{'Config':<25} {'CAGR':>10} {'Std':>8} {'Sharpe':>8} {'MaxDD':>8} {'Turnover':>10} {'Cost':>8}")
    print("-"*82)

    for label, band, si in [
        ("Baseline (1% band)", 0.01, False),
        ("10% no-trade band", 0.10, False),
        ("20% no-trade band", 0.20, False),
        ("30% no-trade band", 0.30, False),
        ("With SI", 0.01, True),
        ("20% band + SI", 0.20, True),
    ]:
        cagrs, sharpes, dds, turnovers, costs = [], [], [], [], []
        for offset in range(7):
            start = pd.Timestamp('2018-01-01') + pd.Timedelta(days=offset)
            r = run_with_mods(bt, str(start.date()), '2025-12-31', cfg,
                               no_trade_band=band, use_si=si)
            if r and r['cagr']:
                cagrs.append(r['cagr'])
                sharpes.append(r['sharpe'])
                dds.append(r['max_dd'])
                turnovers.append(r['annual_turnover'])
                costs.append(r['annual_cost_drag'])
        if cagrs:
            print(f"{label:<25} {np.mean(cagrs)*100:>9.1f}% {np.std(cagrs)*100:>7.1f}% {np.mean(sharpes):>8.2f} {np.mean(dds)*100:>7.1f}% {np.mean(turnovers)*100:>9.0f}% {np.mean(costs)*100:>7.2f}%")

    # Year by year: baseline vs best improvement
    print(f"\n[4] YEAR-BY-YEAR: Baseline vs 20% band")
    print(f"{'Year':<6} {'Baseline':>10} {'20% band':>10} {'Diff':>10}")
    print("-"*40)
    for year in range(2018, 2026):
        r1 = run_with_mods(bt, f'{year}-01-01', f'{year}-12-31', cfg, no_trade_band=0.01)
        r2 = run_with_mods(bt, f'{year}-01-01', f'{year}-12-31', cfg, no_trade_band=0.20)
        c1 = r1['cagr']*100 if r1 else 0
        c2 = r2['cagr']*100 if r2 else 0
        print(f"{year:<6} {c1:>+9.1f}% {c2:>+9.1f}% {c2-c1:>+9.1f}%")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
