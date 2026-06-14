"""
V12 Improvement Testing
========================
Tests improvements that DON'T require new data:
1. Equal-weight within momentum sleeve (reduce concentration)
2. Adaptive trailing stop (tighter during high vol)
3. VIX-based crash detector (reduce exposure on VIX spike)
4. Combined: all improvements together

All compared to the v12 baseline (25.4% CAGR honest).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, INITIAL_CASH, COST_BPS,
)
import numpy as np, pandas as pd, time


def run_custom(bt, start, end, cfg, modifications=None):
    """Run backtest with custom modifications to the standard loop."""
    trading_dates = [d for d in sorted(bt.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

    mods = modifications or {}
    cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
    mom_w = cfg.get("mom_w", 0.50)
    val_w = cfg.get("val_w", 0.35)
    lv_w = cfg.get("lv_w", 0.15)
    top_n = cfg.get("top_n", 5)
    rebal_days = cfg.get("rebal_days", 20)
    base_trailing_stop = cfg.get("trailing_stop", 0.40)
    cap = cfg.get("cap", 0.15)

    # Modifications
    equal_weight_mom = mods.get("equal_weight_mom", False)
    adaptive_stop = mods.get("adaptive_stop", False)
    vix_crash = mods.get("vix_crash", False)

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

        # ── Adaptive trailing stop ──
        if adaptive_stop and len(recent_rets) >= 20:
            realized_vol = np.std(recent_rets[-20:]) * np.sqrt(252)
            if realized_vol > 0.30:
                trailing_stop = 0.25  # tighten during high vol
            elif realized_vol > 0.20:
                trailing_stop = 0.30
            else:
                trailing_stop = base_trailing_stop  # normal: 0.40
        else:
            trailing_stop = base_trailing_stop

        # ── Trailing stop check ──
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

        # Track portfolio returns
        if len(port_values) > 0:
            prev = port_values[-1][1]
            if prev > 0:
                recent_rets.append(total_val / prev - 1)
                if len(recent_rets) > 60:
                    recent_rets.pop(0)

        if day_idx % rebal_days != 0:
            port_values.append((date, total_val))
            continue

        # ── Strategy signals ──
        t1 = strategy1_momentum_reversal(date, bt.uni, day_idx,
                                          top_n=top_n, rebal_days=rebal_days)
        if t1 is None:
            t1 = last_targets.get("mom", {})

        # Equal-weight modification: make all momentum picks equal weight
        if equal_weight_mom and t1:
            eq_w = 1.0 / len(t1)
            t1 = {s: eq_w for s in t1}

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

        # Breadth
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

        # ── VIX crash detector: reduce exposure on VIX spike ──
        eq_pct = 1.0

        # Standard trend filter
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

        # VIX crash detector
        if vix_crash and date in bt.etf_df.index and "VIXM" in bt.etf_df.columns:
            # Use VIXM as VIX proxy (it's in the ETF data)
            vixm = bt.etf_df["VIXM"].loc[:date].dropna()
            if len(vixm) >= 20:
                vixm_now = vixm.iloc[-1]
                vixm_20d = vixm.tail(20).mean()
                # If VIXM spikes >30% above 20-day average, reduce exposure
                if vixm_now > vixm_20d * 1.30:
                    eq_pct *= 0.50
                elif vixm_now > vixm_20d * 1.15:
                    eq_pct *= 0.75

        target_d = {s: w * total_val * eq_pct for s, w in combined.items()}

        # Execute trades
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
    print("V12 IMPROVEMENTS — TESTING")
    print("="*70)

    t0 = time.time()
    bt = FastBacktester('data/wrds/complete_sp1500_universe.pkl')
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}
    bt._si_months = []; bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}
    bt.uni.get_sp500 = bt._get_sp1500
    print(f"Loaded in {time.time()-t0:.1f}s")

    cfg = {
        'mom_w': 0.50, 'val_w': 0.35, 'lv_w': 0.15,
        'top_n': 5, 'rebal_days': 20, 'trailing_stop': 0.40,
        'cap': 0.15, 'trend_scale': {'bear': 0.40, 'caution': 0.65},
    }

    tests = {
        "Baseline (v12)": {},
        "Equal-weight momentum": {"equal_weight_mom": True},
        "Adaptive trailing stop": {"adaptive_stop": True},
        "VIX crash detector": {"vix_crash": True},
        "Equal-wt + adaptive stop": {"equal_weight_mom": True, "adaptive_stop": True},
        "Equal-wt + VIX": {"equal_weight_mom": True, "vix_crash": True},
        "Adaptive stop + VIX": {"adaptive_stop": True, "vix_crash": True},
        "ALL combined": {"equal_weight_mom": True, "adaptive_stop": True, "vix_crash": True},
    }

    # Single run screen
    print(f"\n{'Test':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*58)

    for name, mods in tests.items():
        r = run_custom(bt, "2018-01-01", "2025-12-31", cfg, mods)
        if r:
            print(f"{name:<30} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # Start-day averaged for top performers
    print(f"\nSTART-DAY AVERAGED (7 offsets):")
    print(f"{'Test':<30} {'CAGR':>10} {'Std':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*68)

    for name, mods in tests.items():
        cagrs, sharpes, dds = [], [], []
        for offset in range(7):
            start = pd.Timestamp('2018-01-01') + pd.Timedelta(days=offset)
            r = run_custom(bt, str(start.date()), '2025-12-31', cfg, mods)
            if r and r['cagr']:
                cagrs.append(r['cagr'])
                sharpes.append(r['sharpe'])
                dds.append(r['max_dd'])
        if cagrs:
            print(f"{name:<30} {np.mean(cagrs)*100:>9.1f}% {np.std(cagrs)*100:>7.1f}% {np.mean(sharpes):>8.2f} {np.mean(dds)*100:>7.1f}%")

    # Year by year for baseline vs best improvement
    print(f"\nYEAR-BY-YEAR: Baseline vs ALL combined:")
    print(f"{'Year':<6} {'Baseline':>10} {'Combined':>10} {'Diff':>10}")
    print("-"*40)

    for year in range(2018, 2026):
        r1 = run_custom(bt, f'{year}-01-01', f'{year}-12-31', cfg, {})
        r2 = run_custom(bt, f'{year}-01-01', f'{year}-12-31', cfg,
                         {"equal_weight_mom": True, "adaptive_stop": True, "vix_crash": True})
        c1 = r1['cagr']*100 if r1 else 0
        c2 = r2['cagr']*100 if r2 else 0
        print(f"{year:<6} {c1:>+9.1f}% {c2:>+9.1f}% {c2-c1:>+9.1f}%")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
