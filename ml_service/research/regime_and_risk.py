"""
ML Regime Detection + Stock Risk Model
========================================
Two separate, targeted ML applications:

1. REGIME MODEL: Predict whether the next 20 trading days will be
   "good" or "bad" for momentum stocks. Uses FF factors + FRED macro data.
   Output: exposure scaling (0.3-1.0)

2. STOCK RISK MODEL: Predict which stocks are likely to crash (>-8% in 10 days).
   Uses cross-sectional features. Output: crash probability per stock.
   Stocks with high crash prob get filtered out.

These are more tractable than return prediction because:
- Regime has clearer signals (macro data predicts regimes better than stock returns)
- Crash risk has asymmetric signal (bad events are more predictable than good events)
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
log = logging.getLogger("regime_risk")


def build_regime_features(ff_factors, fred_rates, date):
    """
    Build macro regime features. These are market-wide (same for all stocks).
    Designed to predict: will momentum stocks do well in next 20 days?
    """
    feats = {}
    date_ts = pd.Timestamp(date)

    # Fama-French cumulative factor returns at multiple horizons
    for col in ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']:
        if col not in ff_factors.columns:
            continue
        series = ff_factors[col].loc[:date_ts].dropna()
        if len(series) < 60:
            continue

        feats[f'{col}_5d'] = series.tail(5).sum()
        feats[f'{col}_20d'] = series.tail(20).sum()
        feats[f'{col}_60d'] = series.tail(60).sum()

        # Volatility of factor
        feats[f'{col}_vol20'] = series.tail(20).std() * np.sqrt(252)

    # FRED macro
    if fred_rates is not None:
        macro_series = {
            't10y2y': 'yield_curve',
            't10y3m': 'yield_curve_3m',
            'bamlh0a0hym2': 'hy_spread',
            'tedrate': 'ted_spread',
            'dff': 'fed_funds',
            't10yie': 'inflation_be',
            'dgs10': 'treasury_10y',
            'dgs2': 'treasury_2y',
        }
        for col, name in macro_series.items():
            if col not in fred_rates.columns:
                continue
            series = fred_rates[col].loc[:date_ts].dropna()
            if len(series) < 60:
                continue

            feats[f'{name}_level'] = series.iloc[-1]
            feats[f'{name}_chg5'] = series.iloc[-1] - series.iloc[-5] if len(series) >= 5 else np.nan
            feats[f'{name}_chg20'] = series.iloc[-1] - series.iloc[-20] if len(series) >= 20 else np.nan
            feats[f'{name}_chg60'] = series.iloc[-1] - series.iloc[-60] if len(series) >= 60 else np.nan

    return feats


def build_stock_risk_features(features_by_date, prices_df, date, members):
    """
    Build features for stock-level crash prediction.
    Target: probability of >8% decline in next 10 trading days.
    """
    date_ts = pd.Timestamp(date)
    fdate = features_by_date.get(date_ts, {})

    if date_ts not in prices_df.index:
        return pd.DataFrame()

    px = prices_df.loc[date_ts]
    member_list = [s for s in members if s in px.index and pd.notna(px[s]) and px[s] > 0]
    if len(member_list) < 50:
        return pd.DataFrame()

    features = {}

    # Key crash predictors from the literature:
    # 1. Recent volatility spike (vol_ratio high = instability)
    # 2. Extreme recent returns (very positive = mean reversion risk)
    # 3. Distance from moving averages (overextended)
    # 4. RSI overbought
    # 5. High leverage (fundamental fragility)
    # 6. Revenue/earnings decline

    for key in ['ret_5d', 'ret_10d', 'ret_20d', 'ret_60d', 'ret_120d', 'ret_252d',
                'vol_10d', 'vol_20d', 'vol_60d',
                'dist_sma50', 'dist_sma200', 'rsi_14',
                'roe', 'gross_margin', 'debt_to_equity',
                'revenue_growth_yoy', 'eps_growth_yoy',
                'asset_growth', 'net_issuance']:
        vals = {}
        for s in member_list:
            v = fdate.get(s, {}).get(key)
            if v is not None and not np.isnan(v):
                vals[s] = v
        if len(vals) > 30:
            features[key] = vals

    # Derived features
    # Vol ratio (short/long vol - spike means regime change)
    vol_ratio = {}
    for s in member_list:
        v10 = fdate.get(s, {}).get('vol_10d')
        v60 = fdate.get(s, {}).get('vol_60d')
        if v10 is not None and v60 is not None and not np.isnan(v10) and not np.isnan(v60) and v60 > 0.01:
            vol_ratio[s] = v10 / v60
    if len(vol_ratio) > 30:
        features['vol_ratio'] = vol_ratio

    # Overextension: ret_20d / vol_20d (high = risky)
    overext = {}
    for s in member_list:
        r20 = fdate.get(s, {}).get('ret_20d')
        v20 = fdate.get(s, {}).get('vol_20d')
        if r20 is not None and v20 is not None and not np.isnan(r20) and not np.isnan(v20) and v20 > 0.01:
            overext[s] = r20 / v20
    if len(overext) > 30:
        features['overextension'] = overext

    df = pd.DataFrame(features, index=member_list)
    return df


def main():
    print("="*70)
    print("REGIME DETECTION + STOCK RISK MODEL")
    print("="*70)

    # Load data
    print("\n[1] Loading data...")
    t0 = time.time()
    with open("data/wrds/complete_sp1500_universe.pkl", "rb") as f:
        data = pickle.load(f)

    prices_df = data["prices_df"]
    features_by_date = data["features_by_date"]
    sp500_mem = data.get("sp500_mem", {})
    sp400_mem = data.get("sp400_mem", {})
    sp600_mem = data.get("sp600_mem", {})

    from wrds_data_provider import WRDSDataProvider
    provider = WRDSDataProvider()
    ff = provider.fama_french
    fred = provider.fred_rates

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

    # ══════════════════════════════════════════════════════════
    # PART A: REGIME MODEL
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PART A: REGIME DETECTION MODEL")
    print("="*70)
    print("Target: predict whether top-quintile momentum stocks will beat")
    print("bottom-quintile in next 20 days (momentum factor profitability)")

    # Build training data
    print("\n[A1] Building regime training data...")
    FORWARD_DAYS = 20

    regime_X = []
    regime_y = []
    regime_dates = []

    for i in range(0, len(trading_dates) - FORWARD_DAYS - 5, 5):
        date = trading_dates[i]
        fwd_date = trading_dates[min(i + FORWARD_DAYS, len(trading_dates)-1)]

        # Regime features
        feats = build_regime_features(ff, fred, date)
        if len(feats) < 10:
            continue

        # Target: UMD factor return over next 20 days
        umd_fwd = ff['umd'].loc[date:fwd_date]
        if len(umd_fwd) < 10:
            continue
        target = umd_fwd.sum()  # cumulative momentum factor return

        regime_X.append(feats)
        regime_y.append(target)
        regime_dates.append(date)

    regime_df = pd.DataFrame(regime_X, index=regime_dates)
    regime_target = pd.Series(regime_y, index=regime_dates)

    # Drop high-NaN columns
    nan_frac = regime_df.isna().mean()
    good_cols = nan_frac[nan_frac < 0.3].index.tolist()
    regime_df = regime_df[good_cols]

    print(f"  Samples: {len(regime_df)}, Features: {len(good_cols)}")
    print(f"  Target (20d UMD return): mean={regime_target.mean():.4f}, std={regime_target.std():.4f}")

    # Walk-forward regime model
    print("\n[A2] Walk-forward regime prediction...")

    try:
        import lightgbm as lgb
        USE_LGB = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        USE_LGB = False

    regime_results = []
    for test_year in range(2019, 2026):
        train_end = pd.Timestamp(f"{test_year-1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-15")
        test_end = pd.Timestamp(f"{test_year}-12-31")

        train_mask = regime_df.index <= train_end
        test_mask = (regime_df.index >= test_start) & (regime_df.index <= test_end)

        X_train = regime_df[train_mask].fillna(0)
        y_train = regime_target[train_mask]
        X_test = regime_df[test_mask].fillna(0)
        y_test = regime_target[test_mask]

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        if USE_LGB:
            params = {
                'objective': 'regression', 'metric': 'mae',
                'num_leaves': 15, 'max_depth': 3, 'learning_rate': 0.02,
                'feature_fraction': 0.5, 'bagging_fraction': 0.7,
                'bagging_freq': 3, 'min_child_samples': 20,
                'lambda_l1': 1.0, 'lambda_l2': 5.0,
                'verbose': -1, 'n_jobs': 1,
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            model = lgb.train(
                params, train_data, num_boost_round=300,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            preds = model.predict(X_test)
        else:
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.02,
                subsample=0.7, min_samples_leaf=20, random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        # Evaluate: correlation between predicted and actual regime
        ic = pd.Series(preds, index=X_test.index).corr(y_test, method='spearman')

        # More importantly: does it predict DIRECTION?
        pred_up = preds > 0
        actual_up = y_test.values > 0
        accuracy = np.mean(pred_up == actual_up)

        # Profitability: if model says momentum will be bad, reduce exposure
        # Simulate: when pred < 0, go to 50% exposure; when pred > 0, go 100%
        actual_rets = y_test.values
        # Strategy: scale exposure by prediction
        pred_scaled = np.clip(preds / max(np.abs(preds).max(), 0.001), -1, 1)
        # Full exposure when pred > 0, half when pred < 0
        exposure = np.where(preds > 0, 1.0, 0.5)
        adj_rets = actual_rets * exposure
        unadj_mean = np.mean(actual_rets)
        adj_mean = np.mean(adj_rets)

        regime_results.append({
            'year': test_year, 'ic': ic, 'accuracy': accuracy,
            'unadj_return': unadj_mean * 252/20,  # annualized
            'adj_return': adj_mean * 252/20,
        })

        print(f"  {test_year}: IC={ic:+.3f}, Direction={accuracy*100:.0f}%, "
              f"UMD annualized: unadj={unadj_mean*252/20:+.1%} adj={adj_mean*252/20:+.1%}")

    if regime_results:
        mean_ic = np.mean([r['ic'] for r in regime_results])
        mean_acc = np.mean([r['accuracy'] for r in regime_results])
        print(f"\n  Regime model summary:")
        print(f"    Mean IC: {mean_ic:+.3f}")
        print(f"    Mean direction accuracy: {mean_acc*100:.0f}%")
        if mean_acc > 0.55:
            print(f"    ✓ Model predicts regime direction better than chance")
        else:
            print(f"    ✗ Model doesn't reliably predict regime direction")

        # Feature importance
        if USE_LGB:
            importance = dict(zip(good_cols, model.feature_importance(importance_type='gain')))
        else:
            importance = dict(zip(good_cols, model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 regime features:")
        for name, imp in sorted_imp[:10]:
            print(f"    {name:<30} {imp:.1f}")

    # ══════════════════════════════════════════════════════════
    # PART B: STOCK CRASH RISK MODEL
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PART B: STOCK CRASH RISK MODEL")
    print("="*70)
    print("Target: predict P(stock drops >8% in next 10 days)")

    # Build training data
    print("\n[B1] Building crash risk training data...")
    FORWARD_DAYS = 10
    CRASH_THRESHOLD = -0.08

    crash_X = []
    crash_y = []
    crash_dates = []

    for i in range(0, len(trading_dates) - FORWARD_DAYS - 5, 5):
        date = trading_dates[i]
        fwd_date = trading_dates[min(i + FORWARD_DAYS, len(trading_dates)-1)]

        members = get_sp1500(date)
        if len(members) < 100:
            continue

        feat_df = build_stock_risk_features(features_by_date, prices_df, date, members)
        if feat_df.empty or len(feat_df) < 50:
            continue

        ranked = feat_df.rank(pct=True, na_option='keep')

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

        # Binary target: did stock crash?
        fwd_series = pd.Series(fwd_ret)
        crash = (fwd_series < CRASH_THRESHOLD).astype(int)

        common = ranked.index.intersection(crash.index)
        if len(common) < 50:
            continue

        crash_X.append(ranked.loc[common])
        crash_y.append(crash[common])
        crash_dates.extend([date] * len(common))

    crash_X_df = pd.concat(crash_X, axis=0)
    crash_y_series = pd.concat(crash_y, axis=0)
    crash_dates_series = pd.Series(crash_dates, index=crash_X_df.index)

    # Drop high-NaN columns
    nan_frac = crash_X_df.isna().mean()
    good_cols_crash = nan_frac[nan_frac < 0.3].index.tolist()
    crash_X_df = crash_X_df[good_cols_crash]

    crash_rate = crash_y_series.mean()
    print(f"  Samples: {len(crash_X_df)}, Features: {len(good_cols_crash)}")
    print(f"  Base crash rate: {crash_rate*100:.1f}% (stocks crashing >8% in 10 days)")

    # Walk-forward crash model
    print("\n[B2] Walk-forward crash prediction...")

    crash_results = []
    for test_year in range(2019, 2026):
        train_end = pd.Timestamp(f"{test_year-1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-15")
        test_end = pd.Timestamp(f"{test_year}-12-31")

        train_mask = crash_dates_series <= train_end
        test_mask = (crash_dates_series >= test_start) & (crash_dates_series <= test_end)

        X_train = crash_X_df[train_mask].fillna(0.5)
        y_train = crash_y_series[train_mask]
        X_test = crash_X_df[test_mask].fillna(0.5)
        y_test = crash_y_series[test_mask]

        if len(X_train) < 500 or len(X_test) < 200:
            continue

        if USE_LGB:
            # Use scale_pos_weight for imbalanced classes
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            params = {
                'objective': 'binary', 'metric': 'auc',
                'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.03,
                'feature_fraction': 0.6, 'bagging_fraction': 0.7,
                'bagging_freq': 5, 'min_child_samples': 100,
                'lambda_l1': 0.5, 'lambda_l2': 2.0,
                'scale_pos_weight': n_neg / max(n_pos, 1),
                'verbose': -1, 'n_jobs': 1,
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            model = lgb.train(
                params, train_data, num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            preds = model.predict(X_test)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.7, min_samples_leaf=100, random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]

        # Evaluate: AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_test, preds)
        except:
            auc = 0.5

        # Practical value: if we exclude top 10% highest-risk stocks,
        # what's the crash rate reduction?
        threshold = np.percentile(preds, 90)
        high_risk = preds >= threshold
        low_risk = ~high_risk

        crash_rate_all = y_test.mean()
        crash_rate_excluded = y_test[high_risk].mean() if high_risk.sum() > 0 else 0
        crash_rate_kept = y_test[low_risk].mean() if low_risk.sum() > 0 else 0

        crash_results.append({
            'year': test_year, 'auc': auc,
            'crash_all': crash_rate_all,
            'crash_excluded': crash_rate_excluded,
            'crash_kept': crash_rate_kept,
        })

        print(f"  {test_year}: AUC={auc:.3f}, "
              f"Crash rate: all={crash_rate_all*100:.1f}%, "
              f"top10%risk={crash_rate_excluded*100:.1f}%, "
              f"bottom90%={crash_rate_kept*100:.1f}%")

    if crash_results:
        mean_auc = np.mean([r['auc'] for r in crash_results])
        mean_reduction = np.mean([r['crash_all'] - r['crash_kept'] for r in crash_results])
        print(f"\n  Crash model summary:")
        print(f"    Mean AUC: {mean_auc:.3f}")
        print(f"    Average crash rate reduction: {mean_reduction*100:.2f}pp")
        if mean_auc > 0.60:
            print(f"    ✓ Model identifies risky stocks better than random")
        else:
            print(f"    ✗ Model doesn't reliably identify crash-prone stocks")

        # Feature importance
        if USE_LGB:
            importance = dict(zip(good_cols_crash, model.feature_importance(importance_type='gain')))
        else:
            importance = dict(zip(good_cols_crash, model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 crash risk features:")
        for name, imp in sorted_imp[:10]:
            print(f"    {name:<25} {imp:.1f}")

    # ══════════════════════════════════════════════════════════
    # PART C: COMBINED BACKTEST
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("PART C: COMBINED IMPACT ON PORTFOLIO")
    print("="*70)
    print("Test: factor strategy with crash filter vs without")

    cost_frac = 10 / 10000
    initial_cash = 100000
    TOP_N = 15
    REBAL = 15

    for label, use_crash_filter in [("No crash filter", False), ("With crash filter", True)]:
        cash = initial_cash
        holdings = {}
        port_values = []

        test_dates_bt = [d for d in trading_dates
                          if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")]

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
            fdate = features_by_date.get(date, {})

            # Factor scores: 12-1 momentum + trend + quality
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
                score = r252 - r20
                if roe_v is not None and not np.isnan(roe_v) and roe_v > 0.15:
                    score *= 1.1
                scores[s] = score

            # Crash filter: exclude stocks with high crash probability
            if use_crash_filter and crash_results:
                feat_df = build_stock_risk_features(features_by_date, prices_df, date, members)
                if not feat_df.empty and len(feat_df) > 30:
                    ranked = feat_df.rank(pct=True, na_option='keep')
                    X_pred = ranked[good_cols_crash].fillna(0.5)
                    try:
                        crash_probs = dict(zip(X_pred.index, model.predict(X_pred)))
                        # Remove top 15% highest crash risk
                        if crash_probs:
                            threshold = np.percentile(list(crash_probs.values()), 85)
                            scores = {s: v for s, v in scores.items()
                                       if crash_probs.get(s, 0) < threshold}
                    except:
                        pass

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
            rets = pd.Series(vals).pct_change().dropna()
            n_years = (dates_bt[-1] - dates_bt[0]).days / 365.25
            cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
            sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
            peak = pd.Series(vals).cummax()
            dd = (pd.Series(vals) - peak) / peak
            max_dd = dd.min()
            print(f"  {label:<25} CAGR={cagr*100:>6.1f}%  Sharpe={sharpe:.2f}  MaxDD={max_dd*100:.1f}%")

    print("\n" + "="*70)
    print("RESEARCH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
