"""
Compustat Alpha Factors + Risk-Weighted Portfolio
===================================================
Extract PROVEN academic alpha factors from Compustat quarterly data,
then combine with momentum using risk-aware position sizing.

Academic factors used:
1. Accruals (Sloan 1996): low accruals = higher returns
2. Investment (Titman 2004): low capex growth = higher returns
3. Profitability (Novy-Marx 2013): high gross profit / assets
4. Earnings Quality: cash earnings > accrual earnings
5. Book-to-Market (Fama-French): classic value factor
6. Cash Flow Yield: operating cash flow / market cap

All computed POINT-IN-TIME using rdq (report date).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pickle
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
log = logging.getLogger("compustat_alpha")


class CompustatAlphaFactors:
    """
    Point-in-time alpha factors from Compustat quarterly.
    Uses rdq (report date) to ensure no look-ahead bias.
    """

    def __init__(self, compustat_path="data/wrds/compustat_quarterly.parquet"):
        t0 = time.time()
        cols = [
            'tic', 'rdq', 'datadate', 'fyearq', 'fqtr',
            'atq', 'actq', 'lctq', 'cheq', 'dlcq', 'dpq',  # accruals
            'oancfy', 'ibq', 'niq',  # cash flow, earnings
            'saleq', 'cogsq', 'revtq',  # profitability
            'ppentq', 'capxy',  # investment
            'seqq', 'mkvaltq', 'cshoq',  # book-to-market
            'xrdq',  # R&D
            'invtq', 'rectq',  # working capital
            'dlttq',  # long-term debt
        ]

        df = pd.read_parquet(compustat_path, columns=cols)
        df['rdq'] = pd.to_datetime(df['rdq'])
        df['datadate'] = pd.to_datetime(df['datadate'])
        df = df.dropna(subset=['tic', 'rdq'])
        df = df.sort_values(['tic', 'rdq'])

        # For each ticker, compute factors at each report date
        self._factors_by_ticker = {}
        self._all_report_dates = set()

        for tic, grp in df.groupby('tic'):
            grp = grp.sort_values('rdq').drop_duplicates(subset=['rdq'], keep='last')
            if len(grp) < 4:
                continue

            factors = []
            for i in range(4, len(grp)):
                row = grp.iloc[i]
                prev = grp.iloc[i-4]  # same quarter last year
                prev1 = grp.iloc[i-1]  # previous quarter
                rdq = row['rdq']

                f = {'rdq': rdq}

                # ── 1. Accruals (Sloan 1996) ──
                # Accruals = (ΔCA - ΔCash) - (ΔCL - ΔSTD) - Dep
                # Normalized by total assets
                try:
                    delta_ca = (row['actq'] or 0) - (prev1['actq'] or 0)
                    delta_cash = (row['cheq'] or 0) - (prev1['cheq'] or 0)
                    delta_cl = (row['lctq'] or 0) - (prev1['lctq'] or 0)
                    delta_std = (row['dlcq'] or 0) - (prev1['dlcq'] or 0)
                    dep = row['dpq'] or 0
                    at = row['atq']
                    if at and at > 0:
                        accruals = ((delta_ca - delta_cash) - (delta_cl - delta_std) - dep) / at
                        f['accruals'] = accruals
                except:
                    pass

                # ── 2. Gross Profitability (Novy-Marx) ──
                # GP/Assets = (Revenue - COGS) / Assets
                try:
                    sale = row['saleq'] or row['revtq']
                    cogs = row['cogsq']
                    at = row['atq']
                    if sale and cogs and at and at > 0:
                        f['gp_assets'] = (sale - cogs) / at
                except:
                    pass

                # ── 3. Investment Factor ──
                # Asset growth = (AT_t - AT_t-4) / AT_t-4
                try:
                    at_now = row['atq']
                    at_prev = prev['atq']
                    if at_now and at_prev and at_prev > 0:
                        f['asset_growth'] = (at_now - at_prev) / at_prev
                except:
                    pass

                # Capex growth
                try:
                    capx_now = row['capxy']
                    capx_prev = prev['capxy']
                    if capx_now and capx_prev and capx_prev > 0:
                        f['capex_growth'] = (capx_now - capx_prev) / abs(capx_prev)
                except:
                    pass

                # ── 4. Earnings Quality ──
                # Cash component = operating cash flow / assets
                # If cash earnings >> accrual earnings, quality is high
                try:
                    ocf = row['oancfy']
                    ib = row['ibq']
                    at = row['atq']
                    if ocf is not None and ib is not None and at and at > 0:
                        f['earnings_quality'] = (ocf / 4 - ib) / at  # cash - accrual
                        f['cfo_assets'] = ocf / (4 * at)  # annualized CF/assets
                except:
                    pass

                # ── 5. Book-to-Market ──
                try:
                    bv = row['seqq']
                    mv = row['mkvaltq']
                    if bv and mv and mv > 0:
                        f['book_to_market'] = bv / mv
                except:
                    pass

                # ── 6. Cash Flow Yield ──
                try:
                    ocf = row['oancfy']
                    mv = row['mkvaltq']
                    if ocf is not None and mv and mv > 0:
                        f['cf_yield'] = ocf / mv
                except:
                    pass

                # ── 7. ROE change (earnings momentum) ──
                try:
                    roe_now = row['niq'] / row['seqq'] if row['seqq'] and row['seqq'] > 0 else None
                    roe_prev = prev['niq'] / prev['seqq'] if prev['seqq'] and prev['seqq'] > 0 else None
                    if roe_now is not None and roe_prev is not None:
                        f['roe_change'] = roe_now - roe_prev
                except:
                    pass

                # ── 8. Revenue growth acceleration ──
                try:
                    rev_now = row['saleq'] or row['revtq']
                    rev_prev = prev['saleq'] or prev['revtq']
                    rev_prev1 = prev1['saleq'] or prev1['revtq']
                    if rev_now and rev_prev and rev_prev1 and rev_prev > 0 and rev_prev1 > 0:
                        growth_now = (rev_now - rev_prev) / rev_prev
                        # Previous quarter's YoY growth (approximate)
                        growth_prev = (rev_prev1 - prev['saleq'] if prev['saleq'] and prev['saleq'] > 0 else np.nan)
                        if not np.isnan(growth_prev) and prev['saleq'] > 0:
                            growth_prev = growth_prev / prev['saleq']
                            f['rev_growth_accel'] = growth_now - growth_prev
                except:
                    pass

                # ── 9. Net Issuance (share dilution) ──
                try:
                    shares_now = row['cshoq']
                    shares_prev = prev['cshoq']
                    if shares_now and shares_prev and shares_prev > 0:
                        f['net_issuance'] = (shares_now - shares_prev) / shares_prev
                except:
                    pass

                # ── 10. R&D Intensity ──
                try:
                    rd = row['xrdq']
                    at = row['atq']
                    if rd is not None and at and at > 0:
                        f['rd_intensity'] = rd / at
                except:
                    pass

                if len(f) > 2:  # has rdq + at least 2 factors
                    factors.append(f)
                    self._all_report_dates.add(rdq)

            if factors:
                self._factors_by_ticker[tic] = factors

        log.info(f"Loaded {len(self._factors_by_ticker)} tickers, "
                 f"{sum(len(v) for v in self._factors_by_ticker.values())} factor observations "
                 f"in {time.time()-t0:.1f}s")

    def get_factors(self, date, ticker):
        """Get most recent factors for ticker as of date (point-in-time)."""
        factors = self._factors_by_ticker.get(ticker)
        if not factors:
            return {}

        date_ts = pd.Timestamp(date)
        # Binary search for most recent report before date
        best = None
        for f in factors:
            if f['rdq'] <= date_ts:
                best = f
            else:
                break

        if best is None:
            return {}

        # Don't use stale data (>180 days old)
        if (date_ts - best['rdq']).days > 180:
            return {}

        return {k: v for k, v in best.items() if k != 'rdq' and not np.isnan(v)}


def main():
    print("="*70)
    print("COMPUSTAT ALPHA FACTORS + RISK-WEIGHTED PORTFOLIO")
    print("="*70)

    # ── Load everything ──
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

    print(f"  Universe loaded in {time.time()-t0:.1f}s")

    print("\n[2] Building Compustat alpha factors...")
    t0 = time.time()
    caf = CompustatAlphaFactors()
    print(f"  Factors built in {time.time()-t0:.1f}s")

    # Preview
    trading_dates = sorted(prices_df.index)
    sample_date = trading_dates[len(trading_dates)//2]
    members = get_sp1500(sample_date)
    sample_factors = {}
    for s in list(members)[:5]:
        f = caf.get_factors(sample_date, s)
        if f:
            sample_factors[s] = f
    print(f"\n  Sample factors at {sample_date.date()}:")
    for s, f in sample_factors.items():
        print(f"    {s}: {f}")

    # ── Backtest with Compustat factors ──
    print("\n[3] Backtesting factor combinations (2018-2025)...")

    cost_frac = 10 / 10000
    initial_cash = 100000
    REBAL = 15
    TOP_N = 15

    configs = {
        "Momentum only": {"use_compustat": False, "factors": []},
        "Mom + Accruals": {"use_compustat": True, "factors": ["accruals"]},
        "Mom + GP/Assets": {"use_compustat": True, "factors": ["gp_assets"]},
        "Mom + EarningsQual": {"use_compustat": True, "factors": ["earnings_quality"]},
        "Mom + BookToMkt": {"use_compustat": True, "factors": ["book_to_market"]},
        "Mom + CFYield": {"use_compustat": True, "factors": ["cf_yield"]},
        "Mom + AssetGrowth": {"use_compustat": True, "factors": ["asset_growth"]},
        "Mom + ALL factors": {"use_compustat": True, "factors": [
            "accruals", "gp_assets", "earnings_quality", "book_to_market",
            "cf_yield", "asset_growth", "roe_change", "net_issuance"
        ]},
        "Mom + Quality combo": {"use_compustat": True, "factors": [
            "gp_assets", "earnings_quality", "roe_change"
        ]},
        "Mom + Value combo": {"use_compustat": True, "factors": [
            "book_to_market", "cf_yield", "accruals"
        ]},
    }

    test_dates = [d for d in trading_dates
                  if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")]

    print(f"\n  {'Config':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("  " + "-"*55)

    for cfg_name, cfg in configs.items():
        cash = initial_cash
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

            if day_idx % REBAL != 0:
                port_values.append((date, total_val))
                continue

            members = get_sp1500(date)
            fdate = features_by_date.get(date, {})

            # Score each stock
            scores = {}
            for s in members:
                sf = fdate.get(s, {})
                r252 = sf.get('ret_252d')
                r20 = sf.get('ret_20d')
                d200 = sf.get('dist_sma200')
                roe_v = sf.get('roe')

                if r252 is None or r20 is None or d200 is None:
                    continue
                if np.isnan(r252) or np.isnan(r20) or np.isnan(d200):
                    continue
                if d200 <= 0:
                    continue

                # Base: skip-month momentum
                mom_score = r252 - r20
                if mom_score <= 0:
                    continue

                if not cfg["use_compustat"]:
                    # Quality boost (existing production code)
                    score = mom_score
                    if roe_v is not None and not np.isnan(roe_v) and roe_v > 0.15:
                        score *= 1.05
                    scores[s] = score
                else:
                    # Get Compustat factors
                    cf = caf.get_factors(date, s)
                    if not cf:
                        # No Compustat data — use momentum only
                        scores[s] = mom_score
                        continue

                    # Cross-sectionally rank each factor (will do later in batch)
                    # For now, store raw scores
                    scores[s] = {'mom': mom_score, **{f: cf.get(f) for f in cfg["factors"] if f in cf}}

            if not scores:
                port_values.append((date, total_val))
                continue

            # If using Compustat factors, do cross-sectional ranking
            if cfg["use_compustat"]:
                # Collect all raw values for ranking
                factor_vals = {f: {} for f in cfg["factors"]}
                mom_vals = {}

                for s, sv in scores.items():
                    if isinstance(sv, dict):
                        mom_vals[s] = sv['mom']
                        for f in cfg["factors"]:
                            fv = sv.get(f)
                            if fv is not None and not np.isnan(fv):
                                factor_vals[f][s] = fv
                    else:
                        mom_vals[s] = sv

                # Rank momentum
                mom_series = pd.Series(mom_vals)
                mom_rank = mom_series.rank(pct=True)

                # Rank each factor and combine
                composite_rank = mom_rank.copy() * 0.5  # 50% weight to momentum

                n_factors = len(cfg["factors"])
                factor_weight = 0.5 / max(n_factors, 1)  # split remaining 50%

                for f in cfg["factors"]:
                    if len(factor_vals[f]) < 20:
                        composite_rank += 0.5 / max(n_factors, 1)  # give momentum more weight
                        continue

                    f_series = pd.Series(factor_vals[f])
                    f_rank = f_series.rank(pct=True)

                    # Direction: some factors are "higher is better", some "lower is better"
                    if f in ['accruals', 'asset_growth', 'capex_growth', 'net_issuance']:
                        f_rank = 1 - f_rank  # lower is better (negative = good)

                    for s in composite_rank.index:
                        if s in f_rank.index:
                            composite_rank[s] += f_rank[s] * factor_weight
                        else:
                            composite_rank[s] += 0.5 * factor_weight

                # Use composite rank as score
                final_scores = composite_rank.to_dict()
            else:
                final_scores = scores

            # Trend filter
            eq_pct = 1.0
            if "SPY" in prices_df.columns:
                spy_px = today_px.get("SPY", 0)
                spy_hist = prices_df["SPY"].loc[:date].dropna()
                if len(spy_hist) >= 200:
                    spy_sma200 = spy_hist.tail(200).mean()
                    if spy_px < spy_sma200:
                        eq_pct = 0.50

            # Select top N
            sorted_syms = sorted(final_scores.keys(), key=lambda s: final_scores[s], reverse=True)[:TOP_N]
            raw_s = {s: max(final_scores[s], 0.001) for s in sorted_syms}
            total_score = sum(raw_s.values())
            weights = {s: v / total_score for s, v in raw_s.items()}

            # Risk-weighted sizing: scale DOWN high-vol stocks
            vol_map = {}
            for s in sorted_syms:
                v = fdate.get(s, {}).get('vol_60d')
                if v and not np.isnan(v) and v > 0.01:
                    vol_map[s] = v
                else:
                    vol_map[s] = 0.25  # default

            if vol_map:
                inv_vol = {s: 1.0 / v for s, v in vol_map.items()}
                iv_total = sum(inv_vol.values())
                # Blend: 60% score-proportional + 40% inverse-vol
                for s in weights:
                    iv_w = inv_vol.get(s, 1.0 / 0.25) / iv_total
                    weights[s] = 0.6 * weights[s] + 0.4 * iv_w

                w_total = sum(weights.values())
                if w_total > 0:
                    weights = {s: w / w_total for s, w in weights.items()}

            target_d = {s: w * total_val * eq_pct for s, w in weights.items()}

            # Execute trades
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

        # Compute metrics
        if len(port_values) > 2:
            dates_bt = [v[0] for v in port_values]
            vals = [v[1] for v in port_values]
            rets = pd.Series(vals).pct_change().dropna()
            n_years = (dates_bt[-1] - dates_bt[0]).days / 365.25
            cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
            sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
            peak = pd.Series(vals).cummax()
            dd = (pd.Series(vals) - peak) / peak
            max_dd = dd.min()
            print(f"  {cfg_name:<25} {cagr*100:>7.1f}% {sharpe:>8.2f} {max_dd*100:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Start-day averaged for top configs
    # ═══════════════════════════════════════════════════════════
    print("\n[4] Start-day averaged (momentum only vs best Compustat combo)...")

    for cfg_name in ["Momentum only", "Mom + Quality combo", "Mom + ALL factors"]:
        cfg = configs[cfg_name]
        cagrs = []
        for offset in range(5):
            start_dt = pd.Timestamp("2018-01-01") + pd.Timedelta(days=offset)
            test_dates_offset = [d for d in trading_dates
                                  if start_dt <= d <= pd.Timestamp("2025-12-31")]

            cash = initial_cash
            holdings = {}
            port_values = []

            for day_idx, date in enumerate(test_dates_offset):
                today_px = {}
                if date in prices_df.index:
                    row = prices_df.loc[date]
                    for sym in list(holdings.keys()) + list(row.dropna().index):
                        v = row.get(sym)
                        if v is not None and not np.isnan(v):
                            today_px[sym] = v

                total_val = cash + sum(h["shares"] * today_px.get(s, h["entry_px"])
                                        for s, h in holdings.items())

                if day_idx % REBAL != 0:
                    port_values.append((date, total_val))
                    continue

                members = get_sp1500(date)
                fdate = features_by_date.get(date, {})

                scores = {}
                for s in members:
                    sf = fdate.get(s, {})
                    r252 = sf.get('ret_252d')
                    r20 = sf.get('ret_20d')
                    d200 = sf.get('dist_sma200')
                    roe_v = sf.get('roe')
                    if r252 is None or r20 is None or d200 is None:
                        continue
                    if np.isnan(r252) or np.isnan(r20) or np.isnan(d200):
                        continue
                    if d200 <= 0:
                        continue
                    mom_score = r252 - r20
                    if mom_score <= 0:
                        continue

                    if not cfg["use_compustat"]:
                        score = mom_score
                        if roe_v is not None and not np.isnan(roe_v) and roe_v > 0.15:
                            score *= 1.05
                        scores[s] = score
                    else:
                        cf = caf.get_factors(date, s)
                        scores[s] = {'mom': mom_score, **{f: cf.get(f) for f in cfg["factors"] if f in cf}} if cf else mom_score

                if not scores:
                    port_values.append((date, total_val))
                    continue

                if cfg["use_compustat"]:
                    factor_vals = {f: {} for f in cfg["factors"]}
                    mom_vals = {}
                    for s, sv in scores.items():
                        if isinstance(sv, dict):
                            mom_vals[s] = sv['mom']
                            for f in cfg["factors"]:
                                fv = sv.get(f)
                                if fv is not None and not np.isnan(fv):
                                    factor_vals[f][s] = fv
                        else:
                            mom_vals[s] = sv
                    mom_series = pd.Series(mom_vals)
                    mom_rank = mom_series.rank(pct=True)
                    composite_rank = mom_rank.copy() * 0.5
                    n_factors = len(cfg["factors"])
                    factor_weight = 0.5 / max(n_factors, 1)
                    for f in cfg["factors"]:
                        if len(factor_vals[f]) < 20:
                            composite_rank += 0.5 / max(n_factors, 1)
                            continue
                        f_series = pd.Series(factor_vals[f])
                        f_rank = f_series.rank(pct=True)
                        if f in ['accruals', 'asset_growth', 'capex_growth', 'net_issuance']:
                            f_rank = 1 - f_rank
                        for s in composite_rank.index:
                            if s in f_rank.index:
                                composite_rank[s] += f_rank[s] * factor_weight
                            else:
                                composite_rank[s] += 0.5 * factor_weight
                    final_scores = composite_rank.to_dict()
                else:
                    final_scores = scores

                eq_pct = 1.0
                if "SPY" in prices_df.columns:
                    spy_px = today_px.get("SPY", 0)
                    spy_hist = prices_df["SPY"].loc[:date].dropna()
                    if len(spy_hist) >= 200:
                        spy_sma200 = spy_hist.tail(200).mean()
                        if spy_px < spy_sma200:
                            eq_pct = 0.50

                sorted_syms = sorted(final_scores.keys(), key=lambda s: final_scores[s], reverse=True)[:TOP_N]
                raw_s = {s: max(final_scores[s], 0.001) for s in sorted_syms}
                total_score = sum(raw_s.values())
                weights = {s: v / total_score for s, v in raw_s.items()}

                # Risk-weighted sizing
                vol_map = {}
                for s in sorted_syms:
                    v = fdate.get(s, {}).get('vol_60d')
                    if v and not np.isnan(v) and v > 0.01:
                        vol_map[s] = v
                    else:
                        vol_map[s] = 0.25
                inv_vol = {s: 1.0 / v for s, v in vol_map.items()}
                iv_total = sum(inv_vol.values())
                for s in weights:
                    iv_w = inv_vol.get(s, 1.0 / 0.25) / iv_total
                    weights[s] = 0.6 * weights[s] + 0.4 * iv_w
                w_total = sum(weights.values())
                if w_total > 0:
                    weights = {s: w / w_total for s, w in weights.items()}

                target_d = {s: w * total_val * eq_pct for s, w in weights.items()}

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

            if len(port_values) > 2:
                dates_bt = [v[0] for v in port_values]
                vals = [v[1] for v in port_values]
                n_years = (dates_bt[-1] - dates_bt[0]).days / 365.25
                cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
                cagrs.append(cagr)

        if cagrs:
            avg = np.mean(cagrs)
            std = np.std(cagrs)
            print(f"  {cfg_name:<25} avg CAGR={avg*100:.1f}% +/- {std*100:.1f}%")

    print("\n" + "="*70)
    print("COMPUSTAT FACTOR RESEARCH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
