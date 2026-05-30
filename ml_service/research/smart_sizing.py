"""
Smart Sizing Research
======================
Tests whether we can improve the momentum strategy by:
1. Using fundamentals as FILTERS (not scoring) — avoid momentum traps
2. Using vol for position SIZING — give low-vol momentum stocks bigger positions
3. Using crash risk for position SIZING — reduce position in fragile stocks

Key insight from prior research: blending fundamentals INTO momentum scoring
hurts returns because it dilutes the momentum signal. Instead, use fundamentals
as binary gates or position sizing adjustments.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pickle
import time

def main():
    print("="*70)
    print("SMART SIZING RESEARCH")
    print("="*70)

    # Load
    print("\n[1] Loading data...")
    t0 = time.time()
    with open("data/wrds/complete_sp1500_universe.pkl", "rb") as f:
        data = pickle.load(f)

    prices_df = data["prices_df"]
    features_by_date = data["features_by_date"]
    sp500_mem = data.get("sp500_mem", {})
    sp400_mem = data.get("sp400_mem", {})
    sp600_mem = data.get("sp600_mem", {})

    def get_sp1500(date):
        members = set()
        for mem in [sp500_mem, sp400_mem, sp600_mem]:
            if date in mem:
                members.update(mem[date])
            else:
                prior = [d for d in mem.keys() if d <= date]
                if prior:
                    members.update(mem[max(prior)])
        return members

    trading_dates = sorted(prices_df.index)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Backtest function
    def run_backtest(start, end, strategy_fn, rebal=15, top_n=15):
        test_dates = [d for d in trading_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        cost_frac = 10 / 10000
        cash = 100000.0
        holdings = {}
        port_values = []

        for day_idx, date in enumerate(test_dates):
            today_px = {}
            if date in prices_df.index:
                row = prices_df.loc[date]
                for sym in list(holdings.keys()) + list(row.dropna().index):
                    v = row.get(sym)
                    if v is not None and not np.isnan(v):
                        today_px[sym] = v

            total_val = cash + sum(h["shares"] * today_px.get(s, h["entry_px"])
                                    for s, h in holdings.items())

            if day_idx % rebal != 0:
                port_values.append((date, total_val))
                continue

            members = get_sp1500(date)
            fdate = features_by_date.get(date, {})

            # Get scores and weights from strategy function
            result = strategy_fn(date, members, fdate, prices_df, today_px, top_n)
            if not result:
                port_values.append((date, total_val))
                continue

            target_d = {s: w * total_val for s, w in result.items()}

            for sym in list(holdings):
                if sym not in target_d:
                    px = today_px.get(sym, holdings[sym]["entry_px"])
                    cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                    del holdings[sym]

            for sym, tgt in target_d.items():
                px = today_px.get(sym)
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
        dates_bt = [v[0] for v in port_values]
        vals = [v[1] for v in port_values]
        rets = pd.Series(vals).pct_change().dropna()
        n_years = (dates_bt[-1] - dates_bt[0]).days / 365.25
        cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        peak = pd.Series(vals).cummax()
        dd = (pd.Series(vals) - peak) / peak
        max_dd = dd.min()
        return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd}

    # ═══════════════════════════════════════════════════════════
    # Strategy 1: BASELINE — pure momentum, equal weight
    # ═══════════════════════════════════════════════════════════
    def strat_baseline(date, members, fdate, prices_df, today_px, top_n):
        scores = {}
        for s in members:
            sf = fdate.get(s, {})
            r252 = sf.get('ret_252d'); r20 = sf.get('ret_20d'); d200 = sf.get('dist_sma200')
            if r252 is None or r20 is None or d200 is None: continue
            if np.isnan(r252) or np.isnan(r20) or np.isnan(d200): continue
            if d200 <= 0: continue
            mom = r252 - r20
            if mom <= 0: continue
            roe_v = sf.get('roe')
            if roe_v is not None and not np.isnan(roe_v) and roe_v > 0.15:
                mom *= 1.05
            scores[s] = mom

        # Trend filter
        eq_pct = _trend_filter(date, prices_df, today_px)

        sorted_s = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not sorted_s: return {}
        w = 1.0 / len(sorted_s) * eq_pct
        return {s: w for s in sorted_s}

    # ═══════════════════════════════════════════════════════════
    # Strategy 2: Signal-proportional (current production)
    # ═══════════════════════════════════════════════════════════
    def strat_signal_prop(date, members, fdate, prices_df, today_px, top_n):
        scores = {}
        for s in members:
            sf = fdate.get(s, {})
            r252 = sf.get('ret_252d'); r20 = sf.get('ret_20d'); d200 = sf.get('dist_sma200')
            if r252 is None or r20 is None or d200 is None: continue
            if np.isnan(r252) or np.isnan(r20) or np.isnan(d200): continue
            if d200 <= 0: continue
            mom = r252 - r20
            if mom <= 0: continue
            roe_v = sf.get('roe')
            if roe_v is not None and not np.isnan(roe_v) and roe_v > 0.15:
                mom *= 1.05
            scores[s] = mom

        eq_pct = _trend_filter(date, prices_df, today_px)
        sorted_s = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not sorted_s: return {}
        raw = {s: max(scores[s], 0.001) for s in sorted_s}
        total = sum(raw.values())
        return {s: (v / total) * eq_pct for s, v in raw.items()}

    # ═══════════════════════════════════════════════════════════
    # Strategy 3: Inverse-vol weighted (risk parity within momentum)
    # ═══════════════════════════════════════════════════════════
    def strat_invvol(date, members, fdate, prices_df, today_px, top_n):
        scores = {}
        for s in members:
            sf = fdate.get(s, {})
            r252 = sf.get('ret_252d'); r20 = sf.get('ret_20d'); d200 = sf.get('dist_sma200')
            if r252 is None or r20 is None or d200 is None: continue
            if np.isnan(r252) or np.isnan(r20) or np.isnan(d200): continue
            if d200 <= 0: continue
            mom = r252 - r20
            if mom <= 0: continue
            roe_v = sf.get('roe')
            if roe_v is not None and not np.isnan(roe_v) and roe_v > 0.15:
                mom *= 1.05
            scores[s] = mom

        eq_pct = _trend_filter(date, prices_df, today_px)
        sorted_s = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not sorted_s: return {}

        # Size by inverse volatility
        inv_vol = {}
        for s in sorted_s:
            v = fdate.get(s, {}).get('vol_60d')
            if v and not np.isnan(v) and v > 0.01:
                inv_vol[s] = 1.0 / v
            else:
                inv_vol[s] = 1.0 / 0.25

        total = sum(inv_vol.values())
        return {s: (v / total) * eq_pct for s, v in inv_vol.items()}

    # ═══════════════════════════════════════════════════════════
    # Strategy 4: Blended sizing (60% signal-prop + 40% inv-vol)
    # ═══════════════════════════════════════════════════════════
    def strat_blended(date, members, fdate, prices_df, today_px, top_n):
        scores = {}
        for s in members:
            sf = fdate.get(s, {})
            r252 = sf.get('ret_252d'); r20 = sf.get('ret_20d'); d200 = sf.get('dist_sma200')
            if r252 is None or r20 is None or d200 is None: continue
            if np.isnan(r252) or np.isnan(r20) or np.isnan(d200): continue
            if d200 <= 0: continue
            mom = r252 - r20
            if mom <= 0: continue
            roe_v = sf.get('roe')
            if roe_v is not None and not np.isnan(roe_v) and roe_v > 0.15:
                mom *= 1.05
            scores[s] = mom

        eq_pct = _trend_filter(date, prices_df, today_px)
        sorted_s = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not sorted_s: return {}

        # Signal-proportional
        raw = {s: max(scores[s], 0.001) for s in sorted_s}
        total_sig = sum(raw.values())
        sig_w = {s: v / total_sig for s, v in raw.items()}

        # Inverse vol
        inv_vol = {}
        for s in sorted_s:
            v = fdate.get(s, {}).get('vol_60d')
            if v and not np.isnan(v) and v > 0.01:
                inv_vol[s] = 1.0 / v
            else:
                inv_vol[s] = 1.0 / 0.25
        total_iv = sum(inv_vol.values())
        iv_w = {s: v / total_iv for s, v in inv_vol.items()}

        # Blend
        weights = {}
        for s in sorted_s:
            weights[s] = (0.6 * sig_w[s] + 0.4 * iv_w[s]) * eq_pct

        total_w = sum(weights.values())
        if total_w > 0:
            weights = {s: w / total_w * eq_pct for s, w in weights.items()}
        return weights

    # ═══════════════════════════════════════════════════════════
    # Strategy 5: Quality GATE (not scoring) — reject low-quality momentum
    # ═══════════════════════════════════════════════════════════
    def strat_quality_gate(date, members, fdate, prices_df, today_px, top_n):
        scores = {}
        for s in members:
            sf = fdate.get(s, {})
            r252 = sf.get('ret_252d'); r20 = sf.get('ret_20d'); d200 = sf.get('dist_sma200')
            if r252 is None or r20 is None or d200 is None: continue
            if np.isnan(r252) or np.isnan(r20) or np.isnan(d200): continue
            if d200 <= 0: continue
            mom = r252 - r20
            if mom <= 0: continue

            # Quality GATE: reject stocks with negative ROE or very high debt
            roe_v = sf.get('roe')
            de = sf.get('debt_to_equity')
            gm = sf.get('gross_margin')

            # Must have positive profitability
            if roe_v is not None and not np.isnan(roe_v) and roe_v < 0:
                continue
            # Must not have extreme leverage
            if de is not None and not np.isnan(de) and de > 5.0:
                continue
            # Must have positive gross margin
            if gm is not None and not np.isnan(gm) and gm < 0:
                continue

            scores[s] = mom

        eq_pct = _trend_filter(date, prices_df, today_px)
        sorted_s = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not sorted_s: return {}
        raw = {s: max(scores[s], 0.001) for s in sorted_s}
        total = sum(raw.values())
        return {s: (v / total) * eq_pct for s, v in raw.items()}

    # ═══════════════════════════════════════════════════════════
    # Strategy 6: Quality gate + inv-vol sizing
    # ═══════════════════════════════════════════════════════════
    def strat_gate_invvol(date, members, fdate, prices_df, today_px, top_n):
        scores = {}
        for s in members:
            sf = fdate.get(s, {})
            r252 = sf.get('ret_252d'); r20 = sf.get('ret_20d'); d200 = sf.get('dist_sma200')
            if r252 is None or r20 is None or d200 is None: continue
            if np.isnan(r252) or np.isnan(r20) or np.isnan(d200): continue
            if d200 <= 0: continue
            mom = r252 - r20
            if mom <= 0: continue

            roe_v = sf.get('roe')
            de = sf.get('debt_to_equity')
            gm = sf.get('gross_margin')
            if roe_v is not None and not np.isnan(roe_v) and roe_v < 0: continue
            if de is not None and not np.isnan(de) and de > 5.0: continue
            if gm is not None and not np.isnan(gm) and gm < 0: continue

            scores[s] = mom

        eq_pct = _trend_filter(date, prices_df, today_px)
        sorted_s = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not sorted_s: return {}

        # Blended sizing
        raw = {s: max(scores[s], 0.001) for s in sorted_s}
        total_sig = sum(raw.values())
        sig_w = {s: v / total_sig for s, v in raw.items()}

        inv_vol = {}
        for s in sorted_s:
            v = fdate.get(s, {}).get('vol_60d')
            if v and not np.isnan(v) and v > 0.01:
                inv_vol[s] = 1.0 / v
            else:
                inv_vol[s] = 1.0 / 0.25
        total_iv = sum(inv_vol.values())
        iv_w = {s: v / total_iv for s, v in inv_vol.items()}

        weights = {}
        for s in sorted_s:
            weights[s] = 0.6 * sig_w[s] + 0.4 * iv_w[s]

        total_w = sum(weights.values())
        if total_w > 0:
            weights = {s: w / total_w * eq_pct for s, w in weights.items()}
        return weights

    # ═══════════════════════════════════════════════════════════
    # Strategy 7: Risk-adjusted momentum scoring + equal weight
    # ═══════════════════════════════════════════════════════════
    def strat_risk_adj_mom(date, members, fdate, prices_df, today_px, top_n):
        scores = {}
        for s in members:
            sf = fdate.get(s, {})
            r252 = sf.get('ret_252d'); r20 = sf.get('ret_20d'); d200 = sf.get('dist_sma200')
            v60 = sf.get('vol_60d')
            if r252 is None or r20 is None or d200 is None: continue
            if np.isnan(r252) or np.isnan(r20) or np.isnan(d200): continue
            if d200 <= 0: continue
            if v60 is None or np.isnan(v60) or v60 < 0.01: continue

            # Score by risk-adjusted momentum (Sharpe-like)
            mom = (r252 - r20) / v60
            if mom <= 0: continue
            scores[s] = mom

        eq_pct = _trend_filter(date, prices_df, today_px)
        sorted_s = sorted(scores, key=scores.get, reverse=True)[:top_n]
        if not sorted_s: return {}
        w = 1.0 / len(sorted_s) * eq_pct
        return {s: w for s in sorted_s}

    # Helper
    def _trend_filter(date, prices_df, today_px):
        eq_pct = 1.0
        if "SPY" in prices_df.columns:
            spy_px = today_px.get("SPY", 0)
            spy_hist = prices_df["SPY"].loc[:date].dropna()
            if len(spy_hist) >= 200:
                spy_sma200 = spy_hist.tail(200).mean()
                if spy_px < spy_sma200:
                    eq_pct = 0.50
        return eq_pct

    # ═══════════════════════════════════════════════════════════
    # Run all strategies
    # ═══════════════════════════════════════════════════════════
    strategies = {
        "1. Equal weight": strat_baseline,
        "2. Signal-proportional": strat_signal_prop,
        "3. Inverse-vol sized": strat_invvol,
        "4. Blended (60sig/40vol)": strat_blended,
        "5. Quality gate + sigprop": strat_quality_gate,
        "6. Quality gate + blended": strat_gate_invvol,
        "7. Risk-adj mom + equal": strat_risk_adj_mom,
    }

    # 2018-2025
    print(f"\n{'Strategy':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*60)

    for name, fn in strategies.items():
        r = run_backtest("2018-01-01", "2025-12-31", fn, rebal=15, top_n=15)
        if r:
            print(f"{name:<30} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    # Start-day averaged for best strategies
    print(f"\nStart-day averaged (2018-2025):")
    print(f"{'Strategy':<30} {'Avg CAGR':>10} {'Std':>8}")
    print("-"*55)

    for name, fn in strategies.items():
        cagrs = []
        for offset in range(5):
            start_dt = pd.Timestamp("2018-01-01") + pd.Timedelta(days=offset)
            r = run_backtest(str(start_dt.date()), "2025-12-31", fn, rebal=15, top_n=15)
            if r and r['cagr']:
                cagrs.append(r['cagr'])
        if cagrs:
            print(f"{name:<30} {np.mean(cagrs)*100:>9.1f}% {np.std(cagrs)*100:>7.1f}%")

    # Test with different position counts
    print(f"\nPosition count sensitivity (Strategy 6: Quality gate + blended):")
    print(f"{'Positions':<15} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*45)
    for n in [5, 8, 10, 15, 20, 25]:
        r = run_backtest("2018-01-01", "2025-12-31", strat_gate_invvol, rebal=15, top_n=n)
        if r:
            print(f"{n:<15} {r['cagr']*100:>7.1f}% {r['sharpe']:>8.2f} {r['max_dd']*100:>7.1f}%")

    print("\n" + "="*70)
    print("SMART SIZING RESEARCH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
