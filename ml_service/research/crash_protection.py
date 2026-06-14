"""
Crash Protection Research
==========================
The 2019-2020 sub-period shows -8.8% CAGR and -37.9% MaxDD.
Problem: SMA200 is too slow — by the time SPY drops below SMA200,
we've already taken a big hit.

Test faster crash detection:
1. Dual SMA filter (SMA50 + SMA200)
2. Volatility regime scaling (reduce when vol spikes)
3. UMD momentum factor crash detector
4. Drawdown-based dynamic exposure
5. Faster bear trigger (SMA100 instead of SMA200)
6. Combined: multiple signals
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, INITIAL_CASH, COST_BPS,
)
import numpy as np, pandas as pd

def run_with_custom_exposure(bt, start, end, cfg, exposure_fn):
    """Run backtest with custom exposure function instead of fixed trend filter."""
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

        # Trailing stop
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
                        cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                        del holdings[sym]

        total_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                for s, h in holdings.items())

        # Track portfolio returns for vol-based exposure
        if len(port_values) > 0:
            prev = port_values[-1][1]
            if prev > 0:
                recent_rets.append(total_val / prev - 1)
                if len(recent_rets) > 60:
                    recent_rets.pop(0)

        if day_idx % rebal_days != 0:
            port_values.append((date, total_val))
            continue

        # Strategy signals
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

        # UMD regime
        nu = bt.umd_20d.loc[:date]
        in_crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < -0.05
        if in_crash:
            ew = {"mom": 0.15, "val": 0.45, "s5": 0.30}
        else:
            ew = {"mom": mom_w, "val": val_w, "s5": lv_w}

        # Breadth blending
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

        # ── CUSTOM EXPOSURE FUNCTION ──
        eq_pct = exposure_fn(date, bt.prices, today, recent_rets, bt.umd_20d, breadth)

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
    print("CRASH PROTECTION RESEARCH")
    print("="*70)

    bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

    # CRITICAL: use SP1500 universe (bt.run() does this automatically, custom loop doesn't)
    bt.uni.get_sp500 = bt._get_sp1500

    cfg = {
        'mom_w': 0.50, 'val_w': 0.35, 'lv_w': 0.15,
        'top_n': 5, 'rebal_days': 20, 'trailing_stop': 0.40, 'cap': 0.15,
    }

    # ── Exposure functions ──

    def expo_baseline(date, prices, today, recent_rets, umd_20d, breadth):
        """Current: SMA200 only, bear=40%"""
        if "SPY" not in prices.columns: return 1.0
        spy_px = today.get("SPY", 0)
        spy_hist = prices["SPY"].loc[:date].dropna()
        if len(spy_hist) < 200: return 1.0
        sma200 = spy_hist.tail(200).mean()
        sma50 = spy_hist.tail(50).mean()
        if spy_px < sma200: return 0.40
        if spy_px < sma50: return 0.65
        return 1.0

    def expo_dual_sma(date, prices, today, recent_rets, umd_20d, breadth):
        """Dual SMA: SMA50 + SMA200. Faster signal from SMA50."""
        if "SPY" not in prices.columns: return 1.0
        spy_px = today.get("SPY", 0)
        spy_hist = prices["SPY"].loc[:date].dropna()
        if len(spy_hist) < 200: return 1.0
        sma200 = spy_hist.tail(200).mean()
        sma50 = spy_hist.tail(50).mean()
        if spy_px < sma200: return 0.30  # deep bear
        if spy_px < sma50: return 0.50   # caution
        if sma50 < sma200: return 0.70   # death cross
        return 1.0

    def expo_vol_regime(date, prices, today, recent_rets, umd_20d, breadth):
        """Vol regime: reduce when portfolio vol spikes."""
        if "SPY" not in prices.columns: return 1.0
        spy_px = today.get("SPY", 0)
        spy_hist = prices["SPY"].loc[:date].dropna()
        if len(spy_hist) < 200: return 1.0
        sma200 = spy_hist.tail(200).mean()

        # Base: SMA200 filter
        base = 1.0
        if spy_px < sma200: base = 0.40

        # Vol scaling: if recent portfolio vol is high, reduce further
        if len(recent_rets) >= 20:
            realized_vol = np.std(recent_rets[-20:]) * np.sqrt(252)
            if realized_vol > 0.35:  # very high vol
                base *= 0.60
            elif realized_vol > 0.25:  # elevated vol
                base *= 0.80

        return max(base, 0.20)

    def expo_umd_crash(date, prices, today, recent_rets, umd_20d, breadth):
        """UMD crash: reduce when momentum factor is crashing."""
        if "SPY" not in prices.columns: return 1.0
        spy_px = today.get("SPY", 0)
        spy_hist = prices["SPY"].loc[:date].dropna()
        if len(spy_hist) < 200: return 1.0
        sma200 = spy_hist.tail(200).mean()

        base = 1.0
        if spy_px < sma200: base = 0.40

        # UMD crash: if momentum factor down >5% in 20 days, go defensive
        nu = umd_20d.loc[:date]
        if len(nu) > 0 and pd.notna(nu.iloc[-1]):
            if nu.iloc[-1] < -0.08:  # severe momentum crash
                base *= 0.50
            elif nu.iloc[-1] < -0.05:  # moderate crash
                base *= 0.70

        return max(base, 0.20)

    def expo_drawdown(date, prices, today, recent_rets, umd_20d, breadth):
        """Drawdown-based: reduce exposure when strategy is in drawdown."""
        if "SPY" not in prices.columns: return 1.0
        spy_px = today.get("SPY", 0)
        spy_hist = prices["SPY"].loc[:date].dropna()
        if len(spy_hist) < 200: return 1.0
        sma200 = spy_hist.tail(200).mean()

        base = 1.0
        if spy_px < sma200: base = 0.40

        # If portfolio has lost >10% from recent peak, cut exposure
        if len(recent_rets) >= 5:
            cum = np.cumprod(1 + np.array(recent_rets[-40:]))
            peak = np.maximum.accumulate(cum)
            dd = (cum[-1] - peak[-1]) / peak[-1] if peak[-1] > 0 else 0
            if dd < -0.15:  # 15% drawdown
                base *= 0.50
            elif dd < -0.10:  # 10% drawdown
                base *= 0.70

        return max(base, 0.20)

    def expo_breadth_enhanced(date, prices, today, recent_rets, umd_20d, breadth):
        """Enhanced breadth: use market breadth more aggressively."""
        if "SPY" not in prices.columns: return 1.0
        spy_px = today.get("SPY", 0)
        spy_hist = prices["SPY"].loc[:date].dropna()
        if len(spy_hist) < 200: return 1.0
        sma200 = spy_hist.tail(200).mean()

        base = 1.0
        if spy_px < sma200: base = 0.40

        # Use breadth: if <40% stocks above SMA50, reduce
        if breadth < 0.30:
            base *= 0.50
        elif breadth < 0.40:
            base *= 0.70

        return max(base, 0.20)

    def expo_combined(date, prices, today, recent_rets, umd_20d, breadth):
        """Combined: all signals together."""
        if "SPY" not in prices.columns: return 1.0
        spy_px = today.get("SPY", 0)
        spy_hist = prices["SPY"].loc[:date].dropna()
        if len(spy_hist) < 200: return 1.0
        sma200 = spy_hist.tail(200).mean()
        sma50 = spy_hist.tail(50).mean()

        base = 1.0

        # Signal 1: SMA200
        if spy_px < sma200: base *= 0.50

        # Signal 2: SMA50 caution
        if spy_px < sma50: base *= 0.80

        # Signal 3: UMD crash
        nu = umd_20d.loc[:date]
        if len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < -0.05:
            base *= 0.70

        # Signal 4: Breadth
        if breadth < 0.30:
            base *= 0.70

        # Signal 5: Vol spike
        if len(recent_rets) >= 20:
            rv = np.std(recent_rets[-20:]) * np.sqrt(252)
            if rv > 0.30:
                base *= 0.80

        return max(base, 0.15)

    # ── Test all exposure functions ──
    strategies = {
        "Baseline (SMA200)":       expo_baseline,
        "Dual SMA (50+200)":       expo_dual_sma,
        "Vol regime":              expo_vol_regime,
        "UMD crash":               expo_umd_crash,
        "Drawdown-based":          expo_drawdown,
        "Breadth enhanced":        expo_breadth_enhanced,
        "Combined (all signals)":  expo_combined,
    }

    # Quick single-run screen
    print(f"\n[1] SINGLE-RUN SCREEN (2018-2025):")
    print(f"{'Strategy':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*58)
    for name, fn in strategies.items():
        r = run_with_custom_exposure(bt, "2018-01-01", "2025-12-31", cfg, fn)
        if r:
            print(f"{name:<30} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # 2019-2020 specifically (the problem period)
    print(f"\n[2] 2019-2020 SUB-PERIOD (the problem):")
    print(f"{'Strategy':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*58)
    for name, fn in strategies.items():
        r = run_with_custom_exposure(bt, "2019-01-01", "2020-12-31", cfg, fn)
        if r:
            print(f"{name:<30} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # Start-day averaged for the most promising
    print(f"\n[3] START-DAY AVERAGED (7 offsets, top strategies):")
    for name, fn in strategies.items():
        cagrs, sharpes, dds = [], [], []
        for offset in range(7):
            start = pd.Timestamp('2018-01-01') + pd.Timedelta(days=offset)
            r = run_with_custom_exposure(bt, str(start.date()), '2025-12-31', cfg, fn)
            if r and r['cagr']:
                cagrs.append(r['cagr'])
                sharpes.append(r['sharpe'])
                dds.append(r['max_dd'])
        if cagrs:
            print(f"  {name:<28} CAGR: {np.mean(cagrs)*100:.1f}% +/- {np.std(cagrs)*100:.1f}%  Sharpe: {np.mean(sharpes):.2f}  MaxDD: {np.mean(dds)*100:.1f}%")

    # Year-by-year for top 2
    print(f"\n[4] YEAR-BY-YEAR:")
    best_two = list(strategies.keys())[:2]  # Will update after seeing results
    for name in strategies:
        fn = strategies[name]
        years_str = f"  {name:<28}"
        for year in range(2018, 2026):
            r = run_with_custom_exposure(bt, f'{year}-01-01', f'{year}-12-31', cfg, fn)
            if r:
                years_str += f" {year}:{r['cagr']*100:+.0f}%"
        print(years_str)

    print("\nDONE")

if __name__ == "__main__":
    main()
