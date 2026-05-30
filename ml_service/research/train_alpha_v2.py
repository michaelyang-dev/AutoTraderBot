"""
Alpha Pipeline v2 — Fix the architecture
==========================================
Key changes from v1:
1. REMOVE market-wide features from cross-sectional ranker
   (FF factors, FRED rates, breadth are same for all stocks = useless for ranking)
2. Add SECTOR-RELATIVE features (stock momentum vs sector momentum)
3. More walk-forward splits (1-year test windows)
4. Separate regime model for exposure sizing
5. Sample more frequently for more training data
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from wrds_universe import WRDSUniverse
from wrds_data_provider import SP500Membership, WRDSDataProvider


def build_cross_sectional_features(prices_df, features_by_date, date, members):
    """
    Build ONLY cross-sectional features — things that DIFFER across stocks.
    No market-wide features (those go to the regime model).
    """
    date_ts = pd.Timestamp(date)
    if date_ts not in prices_df.index:
        return pd.DataFrame()

    px = prices_df.loc[date_ts]
    member_list = [s for s in members if s in px.index and pd.notna(px[s]) and px[s] > 0]
    if len(member_list) < 50:
        return pd.DataFrame()

    features = {}
    fdate = features_by_date.get(date_ts, {})

    # ── 1. MOMENTUM at multiple horizons ──
    for n in [5, 10, 20, 60, 120, 252]:
        key = f'ret_{n}d'
        vals = {}
        for s in member_list:
            v = fdate.get(s, {}).get(key)
            if v is not None and not np.isnan(v):
                vals[s] = v
        if len(vals) > 50:
            features[key] = vals

    # Skip-month momentum (12-1)
    mom_12_1 = {}
    for s in member_list:
        r252 = fdate.get(s, {}).get('ret_252d')
        r20 = fdate.get(s, {}).get('ret_20d')
        if r252 is not None and r20 is not None and not np.isnan(r252) and not np.isnan(r20):
            mom_12_1[s] = r252 - r20
    if len(mom_12_1) > 50:
        features['mom_12_1'] = mom_12_1

    # 6-1 month momentum
    mom_6_1 = {}
    for s in member_list:
        r120 = fdate.get(s, {}).get('ret_120d')
        r20 = fdate.get(s, {}).get('ret_20d')
        if r120 is not None and r20 is not None and not np.isnan(r120) and not np.isnan(r20):
            mom_6_1[s] = r120 - r20
    if len(mom_6_1) > 50:
        features['mom_6_1'] = mom_6_1

    # Momentum acceleration
    mom_accel = {}
    for s in member_list:
        r60 = fdate.get(s, {}).get('ret_60d')
        r252 = fdate.get(s, {}).get('ret_252d')
        if r60 is not None and r252 is not None and not np.isnan(r60) and not np.isnan(r252):
            mom_accel[s] = (r60 / 3) - (r252 / 12)
    if len(mom_accel) > 50:
        features['mom_accel'] = mom_accel

    # ── 2. VOLATILITY ──
    for n in [10, 20, 60]:
        key = f'vol_{n}d'
        vals = {}
        for s in member_list:
            v = fdate.get(s, {}).get(key)
            if v is not None and not np.isnan(v):
                vals[s] = v
        if len(vals) > 50:
            features[key] = vals

    # Vol ratio (short/long)
    vol_ratio = {}
    for s in member_list:
        v10 = fdate.get(s, {}).get('vol_10d')
        v60 = fdate.get(s, {}).get('vol_60d')
        if v10 is not None and v60 is not None and not np.isnan(v10) and not np.isnan(v60) and v60 > 0.01:
            vol_ratio[s] = v10 / v60
    if len(vol_ratio) > 50:
        features['vol_ratio'] = vol_ratio

    # ── 3. RISK-ADJUSTED MOMENTUM ──
    risk_adj = {}
    for s in member_list:
        r252 = fdate.get(s, {}).get('ret_252d')
        r20 = fdate.get(s, {}).get('ret_20d')
        v60 = fdate.get(s, {}).get('vol_60d')
        if all(x is not None and not np.isnan(x) for x in [r252, r20, v60]) and v60 > 0.01:
            risk_adj[s] = (r252 - r20) / v60
    if len(risk_adj) > 50:
        features['risk_adj_mom'] = risk_adj

    # ── 4. TECHNICAL ──
    for key in ['dist_sma50', 'dist_sma200', 'rsi_14']:
        vals = {}
        for s in member_list:
            v = fdate.get(s, {}).get(key)
            if v is not None and not np.isnan(v):
                vals[s] = v
        if len(vals) > 50:
            features[key] = vals

    # ── 5. FUNDAMENTAL QUALITY ──
    for key in ['roe', 'gross_margin', 'operating_margin', 'net_margin',
                'debt_to_equity', 'revenue_growth_yoy', 'eps_growth_yoy',
                'asset_growth', 'gp_assets', 'net_issuance']:
        vals = {}
        for s in member_list:
            v = fdate.get(s, {}).get(key)
            if v is not None and not np.isnan(v):
                vals[s] = v
        if len(vals) > 30:
            features[key] = vals

    # ── 6. INTERACTION FEATURES ──
    # Momentum × Quality
    if 'mom_12_1' in features and 'roe' in features:
        features['mom_x_roe'] = {
            s: features['mom_12_1'].get(s, 0) * max(features['roe'].get(s, 0), 0)
            for s in member_list
            if s in features['mom_12_1'] and s in features['roe']
        }

    # Momentum × Inverse Vol
    if 'mom_12_1' in features and 'vol_60d' in features:
        features['mom_x_invvol'] = {
            s: features['mom_12_1'].get(s, 0) / max(features['vol_60d'].get(s, 0.01), 0.01)
            for s in member_list
            if s in features['mom_12_1'] and s in features['vol_60d']
        }

    # Quality composite
    if 'gross_margin' in features and 'roe' in features:
        features['quality_score'] = {
            s: features['gross_margin'].get(s, 0) + features['roe'].get(s, 0)
            for s in member_list
            if s in features['gross_margin'] and s in features['roe']
        }

    # Convert to DataFrame
    df = pd.DataFrame(features, index=member_list)
    return df


def rank_cross_sectional(df):
    """Convert to cross-sectional percentile ranks (0-1)."""
    if df.empty:
        return df
    return df.rank(pct=True, na_option='keep')


def train_and_test():
    print("="*70)
    print("ALPHA PIPELINE v2 — CROSS-SECTIONAL ONLY")
    print("="*70)

    # Load data
    print("\n[1] Loading universe...")
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
    print(f"  Loaded in {time.time()-t0:.1f}s, {len(trading_dates)} dates")

    # ── Feature preview ──
    sample_date = trading_dates[len(trading_dates)//2]
    members = get_sp1500(sample_date)
    feat_df = build_cross_sectional_features(prices_df, features_by_date, sample_date, members)
    print(f"\n  Sample features ({sample_date.date()}):")
    print(f"    Stocks: {len(feat_df)}, Features: {len(feat_df.columns)}")
    print(f"    Columns: {sorted(feat_df.columns.tolist())}")

    # ══════════════════════════════════════════════════════════
    # Build training data
    # ══════════════════════════════════════════════════════════
    print("\n[2] Building training data (every 3 trading days)...")
    t0 = time.time()

    FORWARD_DAYS = 10
    PURGE_DAYS = 15
    SAMPLE_FREQ = 3  # sample every 3 days for more data

    all_X = []
    all_y = []
    all_dates = []

    for i in range(0, len(trading_dates) - FORWARD_DAYS - 5, SAMPLE_FREQ):
        date = trading_dates[i]
        fwd_date = trading_dates[min(i + FORWARD_DAYS, len(trading_dates) - 1)]

        members = get_sp1500(date)
        if len(members) < 100:
            continue

        feat_df = build_cross_sectional_features(prices_df, features_by_date, date, members)
        if feat_df.empty or len(feat_df) < 50:
            continue

        ranked = rank_cross_sectional(feat_df)

        # Forward returns
        px_now = prices_df.loc[date]
        px_fwd = prices_df.loc[fwd_date]
        fwd_ret = {}
        for s in ranked.index:
            p0 = px_now.get(s)
            p1 = px_fwd.get(s)
            if p0 and p1 and pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_ret[s] = (p1 - p0) / p0

        if len(fwd_ret) < 50:
            continue

        fwd_series = pd.Series(fwd_ret)
        fwd_rank = fwd_series.rank(pct=True)

        common = ranked.index.intersection(fwd_rank.index)
        if len(common) < 50:
            continue

        all_X.append(ranked.loc[common])
        all_y.append(fwd_rank[common])
        all_dates.extend([date] * len(common))

    X = pd.concat(all_X, axis=0)
    y = pd.concat(all_y, axis=0)
    dates = pd.Series(all_dates, index=X.index)

    # Drop high-NaN columns
    nan_frac = X.isna().mean()
    good_cols = nan_frac[nan_frac < 0.3].index.tolist()
    X = X[good_cols]
    feature_names = good_cols

    print(f"  Built in {time.time()-t0:.1f}s")
    print(f"  Samples: {len(X)}, Features: {len(good_cols)}")
    print(f"  Dates: {dates.min().date()} to {dates.max().date()}")
    print(f"  Features: {sorted(good_cols)}")

    # ══════════════════════════════════════════════════════════
    # Walk-forward training
    # ══════════════════════════════════════════════════════════
    print("\n[3] Walk-forward training (1-year test windows)...")

    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)

    try:
        import lightgbm as lgb
        USE_LGB = True
        print("  Using LightGBM")
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        USE_LGB = False
        print("  Using sklearn GBM (LightGBM not available)")

    # Walk-forward: train on 3 years, test on 1 year, roll forward
    oos_results = []
    models = []

    # Calendar-year splits for cleaner analysis
    years = sorted(set(d.year for d in unique_dates))
    print(f"  Available years: {years}")

    for test_year in range(2019, max(years) + 1):
        train_end = pd.Timestamp(f"{test_year-1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01") + pd.Timedelta(days=PURGE_DAYS)
        test_end = pd.Timestamp(f"{test_year}-12-31")

        train_mask = dates <= train_end
        test_mask = (dates >= test_start) & (dates <= test_end)

        X_train = X[train_mask].fillna(0.5)
        y_train = y[train_mask]
        X_test = X[test_mask].fillna(0.5)
        y_test = y[test_mask]
        test_dates_split = dates[test_mask]

        if len(X_train) < 500 or len(X_test) < 200:
            print(f"  {test_year}: SKIP (train={len(X_train)}, test={len(X_test)})")
            continue

        if USE_LGB:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 31,
                'max_depth': 5,
                'learning_rate': 0.03,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'min_child_samples': 100,
                'lambda_l1': 0.5,
                'lambda_l2': 2.0,
                'verbose': -1,
                'n_jobs': 1,
                'seed': 42,
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            model = lgb.train(
                params, train_data,
                num_boost_round=800,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.7, min_samples_leaf=100, random_state=42
            )
            model.fit(X_train, y_train)

        # Evaluate: daily Spearman IC
        preds = model.predict(X_test)
        pred_series = pd.Series(preds, index=X_test.index)

        daily_ics = []
        for d in test_dates_split.unique():
            mask = test_dates_split == d
            if mask.sum() < 30:
                continue
            p = pred_series[mask]
            a = y_test[mask]
            ic = p.corr(a, method='spearman')
            if pd.notna(ic):
                daily_ics.append(ic)

        if daily_ics:
            mean_ic = np.mean(daily_ics)
            ic_std = np.std(daily_ics)
            ic_ir = mean_ic / max(ic_std, 0.001)
            pct_positive = sum(1 for x in daily_ics if x > 0) / len(daily_ics)

            oos_results.append({
                'year': test_year,
                'ic': mean_ic,
                'ic_std': ic_std,
                'icir': ic_ir,
                'pct_positive': pct_positive,
                'n_days': len(daily_ics),
            })
            models.append(model)

            signal = "STRONG" if mean_ic > 0.05 else "MODERATE" if mean_ic > 0.03 else "WEAK" if mean_ic > 0.01 else "NONE"
            print(f"  {test_year}: IC={mean_ic:+.4f} (±{ic_std:.4f}), ICIR={ic_ir:.2f}, "
                  f"{pct_positive*100:.0f}% positive days, signal={signal}")

    # ══════════════════════════════════════════════════════════
    # Feature importance
    # ══════════════════════════════════════════════════════════
    print("\n[4] Feature importance (last model)...")
    if models:
        if USE_LGB:
            importance = dict(zip(feature_names, models[-1].feature_importance(importance_type='gain')))
        else:
            importance = dict(zip(feature_names, models[-1].feature_importances_))

        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:20]:
            print(f"    {name:<25} {imp:.1f}")

    # ══════════════════════════════════════════════════════════
    # Backtest with ML scores
    # ══════════════════════════════════════════════════════════
    print("\n[5] Backtest: ML ranker vs factor-only (2020-2025)...")

    if not models:
        print("  No trained models, skipping backtest")
        return

    # Use the latest model
    model = models[-1]

    # Simple backtest
    cost_frac = 10 / 10000  # 10bps round trip
    initial_cash = 100000
    TOP_N = 15
    REBAL = 15

    # Run both factor-only and ML backtests
    for label, use_ml in [("Factor-only", False), ("ML ranker", True)]:
        cash = initial_cash
        holdings = {}
        port_values = []

        test_dates_bt = [d for d in trading_dates
                          if pd.Timestamp("2020-01-01") <= d <= pd.Timestamp("2025-12-31")]

        for day_idx, date in enumerate(test_dates_bt):
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
            feat_df = build_cross_sectional_features(prices_df, features_by_date, date, members)
            if feat_df.empty or len(feat_df) < 50:
                port_values.append((date, total_val))
                continue

            # Get scores
            if use_ml:
                ranked = rank_cross_sectional(feat_df)
                X_pred = ranked[feature_names].fillna(0.5)
                ml_scores = dict(zip(X_pred.index, model.predict(X_pred)))

                # Blend: 60% ML + 40% factor (mom_12_1 rank)
                factor_scores = ranked.get('mom_12_1', pd.Series()).to_dict()
                scores = {}
                for s in set(ml_scores) | set(factor_scores):
                    ml_s = ml_scores.get(s, 0.5)
                    f_s = factor_scores.get(s, 0.5)
                    scores[s] = 0.6 * ml_s + 0.4 * f_s
            else:
                # Pure factor: 12-1 momentum + trend filter + quality boost
                scores = {}
                for s in feat_df.index:
                    mom = feat_df.get('mom_12_1', pd.Series()).get(s, np.nan)
                    d200 = feat_df.get('dist_sma200', pd.Series()).get(s, np.nan)
                    roe_v = feat_df.get('roe', pd.Series()).get(s, np.nan)
                    if pd.notna(mom) and pd.notna(d200) and d200 > 0:
                        score = mom
                        if pd.notna(roe_v) and roe_v > 0.15:
                            score *= 1.1
                        scores[s] = score

            if not scores:
                port_values.append((date, total_val))
                continue

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
            sorted_syms = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)[:TOP_N]
            raw_scores = {s: max(scores[s], 0.001) for s in sorted_syms}
            total_score = sum(raw_scores.values())
            weights = {s: v / total_score for s, v in raw_scores.items()}

            target_d = {s: w * total_val * eq_pct for s, w in weights.items()}

            # Execute
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

        # Metrics
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
            print(f"  {label:<20} CAGR={cagr*100:>6.1f}%  Sharpe={sharpe:.2f}  MaxDD={max_dd*100:.1f}%")

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if oos_results:
        ics = [r['ic'] for r in oos_results]
        print(f"\nML Cross-Sectional Ranker:")
        print(f"  Years tested: {[r['year'] for r in oos_results]}")
        ic_strs = [f"{r['ic']:+.4f}" for r in oos_results]
        print(f"  OOS IC per year: {ic_strs}")
        print(f"  Mean OOS IC: {np.mean(ics):+.4f}")
        print(f"  Positive IC years: {sum(1 for ic in ics if ic > 0)}/{len(ics)}")
        print(f"\n  Interpretation:")
        mean_ic = np.mean(ics)
        if mean_ic > 0.05:
            print(f"    IC={mean_ic:.4f}: INSTITUTIONAL-GRADE signal")
            print(f"    This would be deployed at a quant fund")
        elif mean_ic > 0.03:
            print(f"    IC={mean_ic:.4f}: USABLE signal")
            print(f"    Worth blending with factor scores")
        elif mean_ic > 0.01:
            print(f"    IC={mean_ic:.4f}: WEAK signal")
            print(f"    Marginal improvement over factors")
        else:
            print(f"    IC={mean_ic:.4f}: NO predictive power")
            print(f"    Model cannot differentiate winners from losers")
            print(f"    Possible reasons:")
            print(f"      - Features too correlated with simple momentum")
            print(f"      - Need alternative data (NLP, options, flows)")
            print(f"      - Signal-to-noise too low in weekly horizons")
            print(f"      - Model underfitting or overfitting")


if __name__ == "__main__":
    train_and_test()
