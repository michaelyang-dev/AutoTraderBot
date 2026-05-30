"""
Alpha Engine — Institutional-style ML pipeline
================================================
Generates 60+ features per stock, trains cross-sectional ranking model,
and produces portfolio weights with correlation-aware construction.

Pipeline:
  Raw Data → Feature Engine → ML Cross-Sectional Ranker → Portfolio Optimizer

Key design:
  - ALL features are cross-sectional RANKS (percentiles within each date)
  - ML predicts relative ranking, not absolute returns
  - Walk-forward with purge gap prevents leakage
  - No snapshot data — everything is point-in-time
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pickle
import time
import logging
from pathlib import Path
from collections import defaultdict

log = logging.getLogger("alpha_engine")


# ════════════════════════════════════════════════════════════════════
# STEP 1: FEATURE ENGINE
# ════════════════════════════════════════════════════════════════════

class FeatureEngine:
    """
    Generates 60+ cross-sectional features from price + fundamental data.

    All features are expressed as cross-sectional percentile ranks (0-1)
    within each date. This makes them:
    - Comparable across time (no look-ahead from normalization)
    - Robust to outliers
    - Easy to combine
    """

    def __init__(self, prices_df, features_by_date, compustat_path=None,
                 ff_factors=None, fred_rates=None):
        self.prices = prices_df
        self.features_by_date = features_by_date
        self.ff = ff_factors
        self.fred = fred_rates

        # Pre-compute returns at multiple horizons from prices
        self.returns = {}
        for n in [5, 10, 20, 60, 120, 252]:
            self.returns[n] = prices_df.pct_change(n)

        # Pre-compute volatility
        daily_ret = prices_df.pct_change()
        self.vol = {}
        for n in [10, 20, 60]:
            self.vol[n] = daily_ret.rolling(n).std() * np.sqrt(252)

        # Pre-compute SMAs
        self.sma = {}
        for n in [20, 50, 200]:
            self.sma[n] = prices_df.rolling(n).mean()

        # Pre-compute RSI
        delta = prices_df.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.rsi = 100 - (100 / (1 + rs))

        # Pre-compute volume features if available
        self.volume_ratio = None

        # Pre-compute rolling max/min for 52-week high/low
        self.high_252 = prices_df.rolling(252).max()
        self.low_252 = prices_df.rolling(252).min()

        # Pre-compute Fama-French regime features
        self._ff_features = {}
        if ff_factors is not None:
            for col in ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']:
                if col in ff_factors.columns:
                    self._ff_features[f'{col}_20d'] = ff_factors[col].rolling(20).sum()
                    self._ff_features[f'{col}_60d'] = ff_factors[col].rolling(60).sum()

        # Pre-compute FRED macro features
        self._macro_features = {}
        if fred_rates is not None:
            # Yield curve slope
            if 'dgs10' in fred_rates.columns and 'dgs2' in fred_rates.columns:
                self._macro_features['yield_slope'] = fred_rates['dgs10'] - fred_rates['dgs2']
            if 't10y2y' in fred_rates.columns:
                self._macro_features['t10y2y'] = fred_rates['t10y2y']
                self._macro_features['t10y2y_chg20'] = fred_rates['t10y2y'].diff(20)
            if 't10y3m' in fred_rates.columns:
                self._macro_features['t10y3m'] = fred_rates['t10y3m']
            # Credit spreads
            if 'bamlh0a0hym2' in fred_rates.columns:
                hy = fred_rates['bamlh0a0hym2']
                self._macro_features['hy_spread'] = hy
                self._macro_features['hy_spread_chg20'] = hy.diff(20)
                self._macro_features['hy_spread_chg60'] = hy.diff(60)
            if 'tedrate' in fred_rates.columns:
                self._macro_features['ted_spread'] = fred_rates['tedrate']
            # Fed funds
            if 'dff' in fred_rates.columns:
                self._macro_features['fed_funds'] = fred_rates['dff']
                self._macro_features['fed_funds_chg60'] = fred_rates['dff'].diff(60)
            # Breakeven inflation
            if 't10yie' in fred_rates.columns:
                self._macro_features['inflation_be'] = fred_rates['t10yie']
                self._macro_features['inflation_be_chg20'] = fred_rates['t10yie'].diff(20)

        log.info(f"FeatureEngine initialized: {len(self.prices.columns)} stocks, "
                 f"{len(self._ff_features)} FF features, {len(self._macro_features)} macro features")

    def generate_features(self, date, members):
        """
        Generate all features for a given date and member universe.
        Returns DataFrame: rows=symbols, columns=features.
        All values are RAW (not yet ranked).
        """
        features = {}
        date_ts = pd.Timestamp(date)

        # Get current prices
        if date_ts not in self.prices.index:
            return pd.DataFrame()

        px = self.prices.loc[date_ts]
        member_list = [s for s in members if s in px.index and pd.notna(px[s]) and px[s] > 0]
        if len(member_list) < 50:
            return pd.DataFrame()

        # ── MOMENTUM FEATURES ──────────────────────────────
        for n in [5, 10, 20, 60, 120, 252]:
            if date_ts in self.returns[n].index:
                r = self.returns[n].loc[date_ts]
                features[f'ret_{n}d'] = {s: r.get(s, np.nan) for s in member_list}

        # Skip-month momentum (12-1)
        if date_ts in self.returns[252].index and date_ts in self.returns[20].index:
            r252 = self.returns[252].loc[date_ts]
            r20 = self.returns[20].loc[date_ts]
            features['mom_12_1'] = {s: r252.get(s, np.nan) - r20.get(s, np.nan) for s in member_list}

        # 6-1 month momentum
        if date_ts in self.returns[120].index and date_ts in self.returns[20].index:
            r120 = self.returns[120].loc[date_ts]
            r20 = self.returns[20].loc[date_ts]
            features['mom_6_1'] = {s: r120.get(s, np.nan) - r20.get(s, np.nan) for s in member_list}

        # Short-term reversal (5-day)
        if date_ts in self.returns[5].index:
            r5 = self.returns[5].loc[date_ts]
            features['reversal_5d'] = {s: -r5.get(s, np.nan) for s in member_list}

        # Momentum acceleration: (ret_60d/3) vs (ret_252d/12)
        if date_ts in self.returns[60].index and date_ts in self.returns[252].index:
            r60 = self.returns[60].loc[date_ts]
            r252 = self.returns[252].loc[date_ts]
            features['mom_accel'] = {
                s: (r60.get(s, np.nan) / 3) - (r252.get(s, np.nan) / 12)
                for s in member_list
            }

        # ── VOLATILITY FEATURES ────────────────────────────
        for n in [10, 20, 60]:
            if date_ts in self.vol[n].index:
                v = self.vol[n].loc[date_ts]
                features[f'vol_{n}d'] = {s: v.get(s, np.nan) for s in member_list}

        # Vol ratio (short/long) — mean reversion signal
        if date_ts in self.vol[10].index and date_ts in self.vol[60].index:
            v10 = self.vol[10].loc[date_ts]
            v60 = self.vol[60].loc[date_ts]
            features['vol_ratio'] = {
                s: v10.get(s, np.nan) / max(v60.get(s, np.nan), 0.01)
                for s in member_list
            }

        # Risk-adjusted momentum (Sharpe-like)
        if date_ts in self.returns[252].index and date_ts in self.vol[60].index:
            r252 = self.returns[252].loc[date_ts]
            v60 = self.vol[60].loc[date_ts]
            features['risk_adj_mom'] = {
                s: r252.get(s, np.nan) / max(v60.get(s, np.nan), 0.01)
                for s in member_list
            }

        # ── TREND / TECHNICAL FEATURES ─────────────────────
        for n in [50, 200]:
            if date_ts in self.sma[n].index:
                sma_vals = self.sma[n].loc[date_ts]
                features[f'dist_sma{n}'] = {
                    s: (px[s] - sma_vals.get(s, np.nan)) / max(sma_vals.get(s, np.nan), 0.01)
                    for s in member_list if pd.notna(sma_vals.get(s))
                }

        # RSI
        if date_ts in self.rsi.index:
            rsi_vals = self.rsi.loc[date_ts]
            features['rsi_14'] = {s: rsi_vals.get(s, np.nan) for s in member_list}

        # Distance from 52-week high/low
        if date_ts in self.high_252.index:
            h252 = self.high_252.loc[date_ts]
            l252 = self.low_252.loc[date_ts]
            features['dist_52w_high'] = {
                s: (px[s] - h252.get(s, np.nan)) / max(h252.get(s, np.nan), 0.01)
                for s in member_list if pd.notna(h252.get(s))
            }
            features['dist_52w_low'] = {
                s: (px[s] - l252.get(s, np.nan)) / max(l252.get(s, np.nan), 0.01)
                for s in member_list if pd.notna(l252.get(s))
            }

        # ── FUNDAMENTAL FEATURES ───────────────────────────
        fdate = self.features_by_date.get(date_ts, {})

        fund_features = ['roe', 'gross_margin', 'operating_margin', 'net_margin',
                         'debt_to_equity', 'revenue_growth_yoy', 'eps_growth_yoy',
                         'asset_growth', 'gp_assets', 'net_issuance']
        for feat in fund_features:
            vals = {}
            for s in member_list:
                sf = fdate.get(s, {})
                v = sf.get(feat)
                if v is not None and not np.isnan(v):
                    vals[s] = v
            if len(vals) > 20:
                features[feat] = vals

        # ── COMPOSITE / INTERACTION FEATURES ───────────────
        # Momentum × Quality: high momentum + high ROE
        if 'mom_12_1' in features and 'roe' in features:
            features['mom_x_quality'] = {
                s: features['mom_12_1'].get(s, 0) * max(features['roe'].get(s, 0), 0)
                for s in member_list
                if s in features['mom_12_1'] and s in features['roe']
            }

        # Momentum × Low Vol: high momentum + low vol
        if 'mom_12_1' in features and 'vol_60d' in features:
            features['mom_x_invvol'] = {
                s: features['mom_12_1'].get(s, 0) / max(features['vol_60d'].get(s, 0.01), 0.01)
                for s in member_list
                if s in features['mom_12_1'] and s in features['vol_60d']
            }

        # Value × Quality: cheap + profitable
        if 'gross_margin' in features and 'debt_to_equity' in features:
            features['quality_composite'] = {
                s: features['gross_margin'].get(s, 0) - 0.1 * max(features['debt_to_equity'].get(s, 0), 0)
                for s in member_list
                if s in features['gross_margin'] and s in features['debt_to_equity']
            }

        # ── MARKET-WIDE FEATURES (same for all stocks on a given date) ──
        # Market breadth
        dist_sma50 = features.get('dist_sma50', {})
        if dist_sma50:
            breadth = sum(1 for v in dist_sma50.values() if v > 0) / max(len(dist_sma50), 1)
            for s in member_list:
                features.setdefault('mkt_breadth', {})[s] = breadth

        # Fama-French factor momentum (same for all stocks)
        for fname, fseries in self._ff_features.items():
            if date_ts in fseries.index and pd.notna(fseries[date_ts]):
                val = fseries[date_ts]
                for s in member_list:
                    features.setdefault(fname, {})[s] = val

        # Macro features (same for all stocks)
        for mname, mseries in self._macro_features.items():
            # Find most recent non-NaN value
            prior = mseries.loc[:date_ts].dropna()
            if len(prior) > 0:
                val = prior.iloc[-1]
                for s in member_list:
                    features.setdefault(mname, {})[s] = val

        # Convert to DataFrame
        df = pd.DataFrame(features, index=member_list)

        return df

    def rank_features(self, df):
        """
        Convert raw features to cross-sectional percentile ranks (0-1).
        This is point-in-time safe — ranking is within the current date only.
        """
        if df.empty:
            return df
        ranked = df.rank(pct=True, na_option='keep')
        return ranked


# ════════════════════════════════════════════════════════════════════
# STEP 2: ML CROSS-SECTIONAL RANKER
# ════════════════════════════════════════════════════════════════════

class CrossSectionalRanker:
    """
    LightGBM model that predicts relative stock ranking.

    Target: forward N-day cross-sectional return rank (0-1).
    This is a REGRESSION task predicting continuous rank,
    not classification. Evaluated by rank correlation (Spearman IC).

    Walk-forward training with purge gap to prevent leakage.
    """

    def __init__(self, forward_days=10, purge_days=15, n_features=None):
        self.forward_days = forward_days
        self.purge_days = purge_days
        self.model = None
        self.feature_names = None
        self.train_history = []

    def prepare_training_data(self, feature_engine, universe_members_by_date,
                               start_date, end_date, sample_freq=5):
        """
        Build training dataset from historical features.

        Args:
            feature_engine: FeatureEngine instance
            universe_members_by_date: {date: set of symbols}
            start_date, end_date: date range
            sample_freq: sample every N trading days (reduces size)

        Returns:
            X: DataFrame of ranked features
            y: Series of forward return ranks (target)
            dates: Series of dates for each row
        """
        prices = feature_engine.prices
        trading_dates = sorted([d for d in prices.index
                                if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)])

        all_X = []
        all_y = []
        all_dates = []
        all_syms = []

        for i, date in enumerate(trading_dates):
            if i % sample_freq != 0:
                continue

            # Skip if we can't compute forward returns
            fwd_idx = i + self.forward_days
            if fwd_idx >= len(trading_dates):
                continue
            fwd_date = trading_dates[fwd_idx]

            members = universe_members_by_date.get(date, set())
            if len(members) < 50:
                continue

            # Generate features
            feat_df = feature_engine.generate_features(date, members)
            if feat_df.empty or len(feat_df) < 30:
                continue

            # Rank features
            ranked = feature_engine.rank_features(feat_df)

            # Compute forward returns
            px_now = prices.loc[date]
            px_fwd = prices.loc[fwd_date]
            fwd_ret = {}
            for s in ranked.index:
                p0 = px_now.get(s)
                p1 = px_fwd.get(s)
                if p0 and p1 and pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    fwd_ret[s] = (p1 - p0) / p0

            if len(fwd_ret) < 30:
                continue

            # Target: cross-sectional rank of forward returns
            fwd_series = pd.Series(fwd_ret)
            fwd_rank = fwd_series.rank(pct=True)

            # Only keep stocks with both features AND forward returns
            common = ranked.index.intersection(fwd_rank.index)
            if len(common) < 30:
                continue

            all_X.append(ranked.loc[common])
            all_y.append(fwd_rank[common])
            all_dates.extend([date] * len(common))
            all_syms.extend(list(common))

        if not all_X:
            return None, None, None, None

        X = pd.concat(all_X, axis=0)
        y = pd.concat(all_y, axis=0)
        dates = pd.Series(all_dates, index=X.index)
        syms = pd.Series(all_syms, index=X.index)

        log.info(f"Training data: {len(X)} samples, {len(X.columns)} features, "
                 f"{len(set(all_dates))} dates")

        return X, y, dates, syms

    def train_walkforward(self, X, y, dates, n_splits=5, train_years=4):
        """
        Walk-forward cross-validation with purge gap.

        Returns trained model and OOS performance metrics.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            log.warning("LightGBM not available, falling back to sklearn GBM")
            return self._train_walkforward_sklearn(X, y, dates, n_splits, train_years)

        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        # Feature names (drop NaN-heavy columns)
        nan_frac = X.isna().mean()
        good_cols = nan_frac[nan_frac < 0.5].index.tolist()
        X_clean = X[good_cols].copy()
        self.feature_names = good_cols

        oos_ics = []  # Out-of-sample information coefficients
        models = []

        # Walk-forward splits
        dates_per_year = n_dates // ((unique_dates[-1] - unique_dates[0]).days / 365)
        split_size = max(int(dates_per_year * 1), 50)  # ~1 year per test split

        for split_end_idx in range(int(dates_per_year * train_years), n_dates, split_size):
            if split_end_idx + split_size > n_dates:
                break

            train_end = unique_dates[split_end_idx]
            # Purge gap
            test_start_idx = split_end_idx + int(self.purge_days / 5)  # approx trading days
            if test_start_idx >= n_dates:
                break
            test_start = unique_dates[min(test_start_idx, n_dates - 1)]
            test_end_idx = min(split_end_idx + split_size, n_dates - 1)
            test_end = unique_dates[test_end_idx]

            train_mask = dates <= train_end
            test_mask = (dates >= test_start) & (dates <= test_end)

            X_train = X_clean[train_mask].fillna(0.5)  # Fill NaN with median rank
            y_train = y[train_mask]
            X_test = X_clean[test_mask].fillna(0.5)
            y_test = y[test_mask]
            test_dates = dates[test_mask]

            if len(X_train) < 100 or len(X_test) < 50:
                continue

            # Train LightGBM
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 31,
                'max_depth': 5,
                'learning_rate': 0.05,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 50,
                'lambda_l1': 0.1,
                'lambda_l2': 1.0,
                'verbose': -1,
                'n_jobs': 1,
                'seed': 42,
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            model = lgb.train(
                params, train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )

            # Evaluate: daily Spearman IC
            preds = model.predict(X_test)
            pred_series = pd.Series(preds, index=X_test.index)

            daily_ics = []
            for d in test_dates.unique():
                mask = test_dates == d
                if mask.sum() < 20:
                    continue
                p = pred_series[mask]
                a = y_test[mask]
                ic = p.corr(a, method='spearman')
                if pd.notna(ic):
                    daily_ics.append(ic)

            if daily_ics:
                mean_ic = np.mean(daily_ics)
                ic_ir = mean_ic / max(np.std(daily_ics), 0.001)
                oos_ics.append(mean_ic)
                log.info(f"  Split {len(oos_ics)}: train<={train_end.date()}, "
                         f"test {test_start.date()}-{test_end.date()}, "
                         f"IC={mean_ic:.4f}, ICIR={ic_ir:.2f}")
                models.append(model)

        if not models:
            log.warning("No successful walk-forward splits")
            return None

        # Use the last model (most recent data)
        self.model = models[-1]

        mean_oos_ic = np.mean(oos_ics)
        self.train_history = {
            'oos_ics': oos_ics,
            'mean_ic': mean_oos_ic,
            'n_splits': len(oos_ics),
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importance(importance_type='gain')
            )),
        }

        log.info(f"Walk-forward complete: mean OOS IC = {mean_oos_ic:.4f}, "
                 f"{len(oos_ics)} splits")

        return self.train_history

    def _train_walkforward_sklearn(self, X, y, dates, n_splits, train_years):
        """Fallback using sklearn GradientBoostingRegressor."""
        from sklearn.ensemble import GradientBoostingRegressor

        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        nan_frac = X.isna().mean()
        good_cols = nan_frac[nan_frac < 0.5].index.tolist()
        X_clean = X[good_cols].copy()
        self.feature_names = good_cols

        oos_ics = []
        models = []

        dates_per_year = max(n_dates // max(((unique_dates[-1] - unique_dates[0]).days / 365), 1), 50)
        split_size = max(int(dates_per_year * 1), 50)

        for split_end_idx in range(int(dates_per_year * train_years), n_dates, split_size):
            if split_end_idx + split_size > n_dates:
                break

            train_end = unique_dates[split_end_idx]
            test_start_idx = split_end_idx + int(self.purge_days / 5)
            if test_start_idx >= n_dates:
                break
            test_start = unique_dates[min(test_start_idx, n_dates - 1)]
            test_end_idx = min(split_end_idx + split_size, n_dates - 1)
            test_end = unique_dates[test_end_idx]

            train_mask = dates <= train_end
            test_mask = (dates >= test_start) & (dates <= test_end)

            X_train = X_clean[train_mask].fillna(0.5)
            y_train = y[train_mask]
            X_test = X_clean[test_mask].fillna(0.5)
            y_test = y[test_mask]
            test_dates = dates[test_mask]

            if len(X_train) < 100 or len(X_test) < 50:
                continue

            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=50, random_state=42
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            pred_series = pd.Series(preds, index=X_test.index)

            daily_ics = []
            for d in test_dates.unique():
                mask = test_dates == d
                if mask.sum() < 20:
                    continue
                p = pred_series[mask]
                a = y_test[mask]
                ic = p.corr(a, method='spearman')
                if pd.notna(ic):
                    daily_ics.append(ic)

            if daily_ics:
                mean_ic = np.mean(daily_ics)
                oos_ics.append(mean_ic)
                models.append(model)
                log.info(f"  Split {len(oos_ics)}: IC={mean_ic:.4f}")

        if not models:
            return None

        self.model = models[-1]
        mean_oos_ic = np.mean(oos_ics)

        self.train_history = {
            'oos_ics': oos_ics,
            'mean_ic': mean_oos_ic,
            'n_splits': len(oos_ics),
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importances_
            )),
        }

        return self.train_history

    def predict(self, feature_df_ranked):
        """
        Score stocks. Returns {symbol: score} dict.
        Higher score = model thinks stock will rank higher.
        """
        if self.model is None or feature_df_ranked.empty:
            return {}

        X = feature_df_ranked[self.feature_names].fillna(0.5)

        try:
            preds = self.model.predict(X)
        except Exception as e:
            log.warning(f"Prediction failed: {e}")
            return {}

        return dict(zip(X.index, preds))


# ════════════════════════════════════════════════════════════════════
# STEP 3: PORTFOLIO OPTIMIZER
# ════════════════════════════════════════════════════════════════════

class PortfolioOptimizer:
    """
    Correlation-aware portfolio construction.

    Instead of naive signal-proportional sizing, this:
    1. Limits sector concentration
    2. Penalizes correlated positions
    3. Controls turnover
    """

    def __init__(self, max_positions=15, max_sector_pct=0.35,
                 turnover_penalty=0.002):
        self.max_positions = max_positions
        self.max_sector_pct = max_sector_pct
        self.turnover_penalty = turnover_penalty

    def optimize(self, scores, prices_df, date, current_holdings=None,
                 sector_map=None, lookback=60):
        """
        Build portfolio weights from scores.

        Args:
            scores: {symbol: score} from ML model or factor model
            prices_df: price DataFrame for correlation calculation
            date: current date
            current_holdings: set of currently held symbols
            sector_map: {symbol: sector} for sector constraints
            lookback: days for correlation estimation

        Returns:
            {symbol: weight} portfolio weights summing to ~1.0
        """
        if not scores:
            return {}

        # Step 1: Select top candidates by score
        sorted_syms = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
        candidates = sorted_syms[:self.max_positions * 2]  # 2x candidates for filtering

        if not candidates:
            return {}

        # Step 2: Compute recent return correlations
        date_ts = pd.Timestamp(date)
        avail_cols = [s for s in candidates if s in prices_df.columns]
        if len(avail_cols) < 5:
            # Fallback: equal weight top N
            top = sorted_syms[:self.max_positions]
            return {s: 1.0 / len(top) for s in top}

        ret_df = prices_df[avail_cols].pct_change()
        ret_window = ret_df.loc[:date_ts].tail(lookback).dropna(axis=1, how='all')

        if ret_window.empty or len(ret_window.columns) < 5:
            top = sorted_syms[:self.max_positions]
            return {s: 1.0 / len(top) for s in top}

        # Step 3: Greedy selection — pick highest score stocks that aren't
        # too correlated with already-selected stocks
        selected = []
        corr_matrix = ret_window.corr()

        for sym in sorted_syms:
            if sym not in corr_matrix.columns:
                continue

            if len(selected) >= self.max_positions:
                break

            # Check correlation with already selected
            if selected:
                max_corr = max(
                    abs(corr_matrix.loc[sym, s]) if s in corr_matrix.columns else 0
                    for s in selected
                )
                # Skip if too correlated (> 0.7) with any selected stock
                if max_corr > 0.70 and len(selected) >= 5:
                    continue

            # Sector constraint
            if sector_map and len(selected) >= 3:
                sec = sector_map.get(sym, 'Unknown')
                sec_count = sum(1 for s in selected if sector_map.get(s) == sec)
                max_in_sector = max(2, int(self.max_positions * self.max_sector_pct))
                if sec_count >= max_in_sector:
                    continue

            selected.append(sym)

        if not selected:
            selected = sorted_syms[:self.max_positions]

        # Step 4: Score-proportional weights with cap
        raw_scores = {s: max(scores.get(s, 0), 0.001) for s in selected}
        total = sum(raw_scores.values())

        cap_per_stock = 2.0 / len(selected)
        weights = {s: min(v / total, cap_per_stock) for s, v in raw_scores.items()}

        # Renormalize
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {s: w / w_total for s, w in weights.items()}

        # Step 5: Turnover penalty — slight bias toward current holdings
        if current_holdings and self.turnover_penalty > 0:
            for s in weights:
                if s in current_holdings:
                    weights[s] *= (1 + self.turnover_penalty)
            w_total = sum(weights.values())
            if w_total > 0:
                weights = {s: w / w_total for s, w in weights.items()}

        return weights


# ════════════════════════════════════════════════════════════════════
# STEP 4: INTEGRATED BACKTEST
# ════════════════════════════════════════════════════════════════════

class AlphaBacktester:
    """
    Backtest the full alpha pipeline:
    Feature Engine → ML Ranker → Portfolio Optimizer → Returns
    """

    def __init__(self, universe_path="data/wrds/complete_sp1500_universe.pkl"):
        t0 = time.time()

        with open(universe_path, "rb") as f:
            data = pickle.load(f)

        self.prices = data["prices_df"]
        self.features_by_date = data["features_by_date"]
        self.sp500_mem = data.get("sp500_mem", {})
        self.sp400_mem = data.get("sp400_mem", {})
        self.sp600_mem = data.get("sp600_mem", {})

        # Load Fama-French and FRED
        from wrds_data_provider import WRDSDataProvider
        provider = WRDSDataProvider()
        ff = provider.fama_french
        fred = provider.fred_rates

        # Build SP1500 membership cache
        self._sp1500_cache = {}

        # Initialize feature engine
        self.feature_engine = FeatureEngine(
            self.prices, self.features_by_date,
            ff_factors=ff, fred_rates=fred
        )

        log.info(f"AlphaBacktester loaded in {time.time()-t0:.1f}s")

    def _get_sp1500(self, date):
        if date not in self._sp1500_cache:
            members = set()
            for mem in [self.sp500_mem, self.sp400_mem, self.sp600_mem]:
                if date in mem:
                    members.update(mem[date])
                else:
                    prior = [d for d in mem.keys() if d <= date]
                    if prior:
                        members.update(mem[max(prior)])
            self._sp1500_cache[date] = members
        return self._sp1500_cache[date]

    def train_model(self, train_start="2016-07-01", train_end="2022-12-31"):
        """Train ML ranker on historical data."""
        log.info(f"Training ML ranker on {train_start} to {train_end}...")

        # Build universe membership by date
        trading_dates = [d for d in sorted(self.prices.index)
                         if pd.Timestamp(train_start) <= d <= pd.Timestamp(train_end)]

        members_by_date = {}
        for d in trading_dates:
            members_by_date[d] = self._get_sp1500(d)

        self.ranker = CrossSectionalRanker(forward_days=10, purge_days=15)

        X, y, dates, syms = self.ranker.prepare_training_data(
            self.feature_engine, members_by_date,
            train_start, train_end, sample_freq=5
        )

        if X is None:
            log.error("Failed to build training data")
            return None

        result = self.ranker.train_walkforward(X, y, dates)
        return result

    def run(self, start="2018-01-01", end="2025-12-31", config=None):
        """Run backtest using the alpha pipeline."""
        if config is None:
            config = {}

        trading_dates = [d for d in sorted(self.prices.index)
                         if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        if not trading_dates:
            return None

        rebal_days = config.get("rebal_days", 15)
        trailing_stop = config.get("trailing_stop", None)
        max_positions = config.get("max_positions", 15)
        use_trend_filter = config.get("use_trend_filter", True)
        factor_blend = config.get("factor_blend", 0.0)  # 0 = pure ML, 1 = pure factor
        cost_bps = config.get("cost_bps", 10)

        cost_frac = cost_bps / 10000
        optimizer = PortfolioOptimizer(max_positions=max_positions)

        cash = 100000.0
        holdings = {}
        port_values = []

        for day_idx, date in enumerate(trading_dates):
            # Get current prices
            today = {}
            if date in self.prices.index:
                row = self.prices.loc[date]
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

            if day_idx % rebal_days != 0:
                port_values.append((date, total_val))
                continue

            # ── Generate features ──
            members = self._get_sp1500(date)
            feat_df = self.feature_engine.generate_features(date, members)

            if feat_df.empty or len(feat_df) < 30:
                port_values.append((date, total_val))
                continue

            ranked = self.feature_engine.rank_features(feat_df)

            # ── Get ML scores ──
            if self.ranker and self.ranker.model is not None:
                ml_scores = self.ranker.predict(ranked)
            else:
                ml_scores = {}

            # ── Get factor scores (simple momentum + quality) ──
            factor_scores = {}
            if factor_blend > 0 or not ml_scores:
                mom = feat_df.get('mom_12_1', pd.Series())
                roe = feat_df.get('roe', pd.Series())
                for sym in members:
                    if sym in feat_df.index:
                        m = mom.get(sym, np.nan)
                        r = roe.get(sym, np.nan)
                        if pd.notna(m) and pd.notna(r):
                            # Simple composite: momentum + quality boost
                            s = m
                            if r > 0.15:
                                s *= 1.1
                            # Trend filter: dist_sma200 > 0
                            d200 = feat_df.get('dist_sma200', pd.Series()).get(sym, np.nan)
                            if pd.notna(d200) and d200 > 0:
                                factor_scores[sym] = s

            # ── Blend ML + factor scores ──
            if ml_scores and factor_blend < 1.0:
                scores = {}
                all_syms = set(ml_scores.keys()) | set(factor_scores.keys())
                for s in all_syms:
                    ml_s = ml_scores.get(s, 0.5)
                    f_s = factor_scores.get(s, 0.5)
                    scores[s] = (1 - factor_blend) * ml_s + factor_blend * f_s
            elif factor_scores:
                scores = factor_scores
            else:
                scores = ml_scores if ml_scores else {}

            if not scores:
                port_values.append((date, total_val))
                continue

            # ── Trend filter: reduce exposure when SPY < SMA200 ──
            eq_pct = 1.0
            if use_trend_filter and "SPY" in self.prices.columns:
                spy_px = today.get("SPY", 0)
                spy_hist = self.prices["SPY"].loc[:date].dropna()
                if len(spy_hist) >= 200:
                    spy_sma200 = spy_hist.tail(200).mean()
                    if spy_px < spy_sma200:
                        eq_pct = 0.50

            # ── Portfolio optimization ──
            current = set(holdings.keys())
            weights = optimizer.optimize(scores, self.prices, date, current)

            target_d = {s: w * total_val * eq_pct for s, w in weights.items()}

            # ── Execute trades ──
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

        # ── Compute metrics ──
        if len(port_values) < 2:
            return None

        dates_list = [v[0] for v in port_values]
        vals = [v[1] for v in port_values]
        rets = pd.Series(vals).pct_change().dropna()
        n_years = (dates_list[-1] - dates_list[0]).days / 365.25
        cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        peak = pd.Series(vals).cummax()
        dd = (pd.Series(vals) - peak) / peak
        max_dd = dd.min()

        # Annual returns
        val_series = pd.Series(vals, index=dates_list)
        annual = val_series.resample('YE').last().pct_change().dropna()

        return {
            "cagr": cagr,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "final_val": vals[-1],
            "n_years": n_years,
            "annual_returns": annual.to_dict() if len(annual) > 0 else {},
            "avg_positions": max_positions,
        }
