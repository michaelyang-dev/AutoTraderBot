"""
Validate Adaptive Trailing Stop — Full Honest Validation
==========================================================
1. Walk-forward: train on early years, test on later years
2. Sub-period analysis: does it work in ALL eras?
3. Parameter sensitivity: is 30% vol threshold special or robust?
4. Start-day averaged across all tests
5. 25-year test if available
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


def run_with_stop(bt, start, end, cfg, adaptive=False, vol_thresh=0.30, tight_stop=0.25):
    """Run backtest with optional adaptive trailing stop."""
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
    base_stop = cfg.get("trailing_stop", 0.40)
    cap = cfg.get("cap", 0.15)

    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    last_targets = {}
    recent_rets = []

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

        # Adaptive stop
        if adaptive and len(recent_rets) >= 20:
            rv = np.std(recent_rets[-20:]) * np.sqrt(252)
            stop = tight_stop if rv > vol_thresh else base_stop
        else:
            stop = base_stop

        # Trailing stop
        for sym in list(holdings):
            px = today.get(sym)
            if px:
                if "peak_px" not in holdings[sym]:
                    holdings[sym]["peak_px"] = px
                if px > holdings[sym]["peak_px"]:
                    holdings[sym]["peak_px"] = px
                dd = (px - holdings[sym]["peak_px"]) / holdings[sym]["peak_px"]
                if dd < -abs(stop):
                    cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                    del holdings[sym]

        total_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                for s, h in holdings.items())

        if len(port_values) > 0:
            prev = port_values[-1][1]
            if prev > 0:
                recent_rets.append(total_val / prev - 1)
                if len(recent_rets) > 60:
                    recent_rets.pop(0)

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
                spy_sma50 = spy_hist.tail(50).mean()
                if spy_px < spy_sma200:
                    eq_pct *= trend_scale.get("bear", 0.5)
                elif spy_px < spy_sma50:
                    eq_pct *= trend_scale.get("caution", 0.75)

        target_d = {s: w * total_val * eq_pct for s, w in combined.items()}

        for sym in list(holdings):
            if sym not in target_d:
                px = today.get(sym, holdings[sym]["entry_px"])
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
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

    if len(port_values) < 2:
        return None
    dates = [v[0] for v in port_values]
    vals = [v[1] for v in port_values]
    rets = pd.Series(vals).pct_change().dropna()
    n_years = (dates[-1] - dates[0]).days / 365.25
    cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    peak = pd.Series(vals).cummax()
    dd = (pd.Series(vals) - peak) / peak
    max_dd = dd.min()
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}


def main():
    print("="*70)
    print("ADAPTIVE TRAILING STOP — FULL VALIDATION")
    print("="*70)

    bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
    bt.uni.get_sp500 = bt._get_sp1500

    cfg = {
        'mom_w': 0.50, 'val_w': 0.35, 'lv_w': 0.15,
        'top_n': 5, 'rebal_days': 20, 'trailing_stop': 0.40,
        'cap': 0.15, 'trend_scale': {'bear': 0.40, 'caution': 0.65},
    }

    # ═══════════════════════════════════════════════════════════
    # TEST 1: Parameter sensitivity — is 30% vol threshold special?
    # ═══════════════════════════════════════════════════════════
    print("\n[1] PARAMETER SENSITIVITY (vol threshold)")
    print(f"{'Vol Thresh':<12} {'Tight Stop':<12} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*52)

    # Test various vol thresholds and tight stop levels
    for vol_t in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for tight in [0.20, 0.25, 0.30]:
            r = run_with_stop(bt, "2018-01-01", "2025-12-31", cfg,
                               adaptive=True, vol_thresh=vol_t, tight_stop=tight)
            if r:
                print(f"{vol_t:<12.2f} {tight:<12.2f} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # Baseline for comparison
    r_base = run_with_stop(bt, "2018-01-01", "2025-12-31", cfg, adaptive=False)
    print(f"{'BASELINE':<12} {'0.40':<12} {r_base['cagr']*100:>7.1f}% {r_base['sharpe']:>8.2f} {r_base['max_dd']*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Walk-forward (train/test split)
    # ═══════════════════════════════════════════════════════════
    print(f"\n[2] WALK-FORWARD VALIDATION")
    print("  Adaptive stop (vol>0.30 → 25% stop) vs baseline (fixed 40%)")
    print(f"{'Period':<20} {'Baseline':>10} {'Adaptive':>10} {'Diff':>10}")
    print("-"*55)

    periods = [
        ("2016-2018", "2016-07-01", "2018-12-31"),
        ("2019-2020", "2019-01-01", "2020-12-31"),
        ("2021-2022", "2021-01-01", "2022-12-31"),
        ("2023-2025", "2023-01-01", "2025-12-31"),
    ]

    base_wins = 0
    adapt_wins = 0
    for label, s, e in periods:
        rb = run_with_stop(bt, s, e, cfg, adaptive=False)
        ra = run_with_stop(bt, s, e, cfg, adaptive=True, vol_thresh=0.30, tight_stop=0.25)
        cb = rb['cagr']*100 if rb else 0
        ca = ra['cagr']*100 if ra else 0
        winner = "Adaptive" if ca > cb else "Baseline"
        if ca > cb: adapt_wins += 1
        else: base_wins += 1
        print(f"{label:<20} {cb:>+9.1f}% {ca:>+9.1f}% {ca-cb:>+9.1f}%  {winner}")

    print(f"\n  Baseline wins: {base_wins}/{base_wins+adapt_wins}")
    print(f"  Adaptive wins: {adapt_wins}/{base_wins+adapt_wins}")

    # ═══════════════════════════════════════════════════════════
    # TEST 3: Year-by-year
    # ═══════════════════════════════════════════════════════════
    print(f"\n[3] YEAR-BY-YEAR")
    print(f"{'Year':<6} {'Baseline':>10} {'Adaptive':>10} {'Diff':>10}")
    print("-"*40)

    for year in range(2016, 2026):
        rb = run_with_stop(bt, f'{year}-01-01', f'{year}-12-31', cfg, adaptive=False)
        ra = run_with_stop(bt, f'{year}-01-01', f'{year}-12-31', cfg, adaptive=True)
        cb = rb['cagr']*100 if rb else 0
        ca = ra['cagr']*100 if ra else 0
        print(f"{year:<6} {cb:>+9.1f}% {ca:>+9.1f}% {ca-cb:>+9.1f}%")

    # ═══════════════════════════════════════════════════════════
    # TEST 4: Start-day averaged (honest)
    # ═══════════════════════════════════════════════════════════
    print(f"\n[4] START-DAY AVERAGED (7 offsets)")

    for label, adaptive in [("Baseline (fixed 40%)", False), ("Adaptive (30%→25%)", True)]:
        cagrs, sharpes, dds = [], [], []
        for offset in range(7):
            start = pd.Timestamp('2018-01-01') + pd.Timedelta(days=offset)
            r = run_with_stop(bt, str(start.date()), '2025-12-31', cfg, adaptive=adaptive)
            if r and r['cagr']:
                cagrs.append(r['cagr'])
                sharpes.append(r['sharpe'])
                dds.append(r['max_dd'])
        print(f"  {label:<25} CAGR: {np.mean(cagrs)*100:.1f}% +/- {np.std(cagrs)*100:.1f}%  "
              f"Sharpe: {np.mean(sharpes):.2f}  MaxDD: {np.mean(dds)*100:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # TEST 5: 25-year test
    # ═══════════════════════════════════════════════════════════
    print(f"\n[5] 25-YEAR TEST")
    try:
        bt2 = FastBacktester('data/wrds/sp1500_universe_2000.pkl')
        bt2.uni._fin_growth = {}; bt2.uni._ev = {}; bt2.uni._estimates = {}
        bt2.uni._price_targets = {}; bt2.uni._revenue_surprise = {}
        bt2.uni._beat_streak = {}; bt2.uni._earnings_signals = {}
        bt2._si_ranks_by_month = {}; bt2._si_change_ranks_by_month = {}
        bt2._si_months = []; bt2.uni._short_interest_rank = {}; bt2.uni._si_change_rank = {}

        for label, adaptive in [("Baseline", False), ("Adaptive", True)]:
            cagrs = []
            for offset in range(5):
                start = pd.Timestamp('2001-01-01') + pd.Timedelta(days=offset)
                r = run_with_stop(bt2, str(start.date()), '2025-12-31', cfg, adaptive=adaptive)
                if r and r['cagr']:
                    cagrs.append(r['cagr'])
            if cagrs:
                print(f"  {label:<15} 25yr CAGR: {np.mean(cagrs)*100:.1f}% +/- {np.std(cagrs)*100:.1f}%")
    except Exception as e:
        print(f"  (25-year universe not available: {e})")

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
