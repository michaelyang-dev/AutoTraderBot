"""
XGBoost LambdaRank Stock Ranker — Built From Scratch
=====================================================
Re-ranks top-20 momentum candidates using XGBoost's LambdaRank objective.
Learns non-linear feature interactions that simple weighted scoring misses.

Optimizes NDCG@8 (ranking quality at position 8, matching our top-8 portfolio).
Blends with factor scores (default 60% factor / 40% ML) to enhance rather than
replace the proven multi-factor strategy.

Usage:
    # Training
    ranker = XGBoostRanker()
    ranker.train_from_universe("data/wrds/complete_sp1500_universe.pkl")
    ranker.save("data/xgboost_ranker_v1.pkl")

    # Prediction (live)
    ranker = XGBoostRanker("data/xgboost_ranker_v1.pkl")
    scores = ranker.predict(date, universe, candidates)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger("xgboost_ranker")

# 12 factor features — economically meaningful, backtest-safe
FEATURES = [
    "momentum_score",     # avg(ret_20d,60d,126d,252d) × consistency²
    "gross_margin",       # profitability
    "eps_surprise_last",  # earnings beat/miss
    "ret_10d_vs_sector",  # sector-relative strength
    "rev_growth",         # revenue acceleration
    "eps_growth",         # EPS acceleration
    "debt_to_equity",     # leverage risk
    "roe",                # management efficiency
    "dist_sma50",         # trend strength
    "market_breadth",     # regime (% above 50d SMA, same for all stocks on date)
    "vol_60d",            # risk
    "ret_126d",           # 6-month momentum
]

PURGE_DAYS = 10  # gap between features and forward label (prevent look-ahead)

XGB_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@8",
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0,
}


class XGBoostRanker:
    """XGBoost ranking model for stock selection."""

    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = FEATURES
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "features": self.feature_names}, f)
        log.info(f"Model saved to {path}")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data.get("features", FEATURES)
        log.info(f"Model loaded from {path}")

    # ── Feature Extraction (works with both WRDSUniverse and FastUniverse) ──

    @staticmethod
    def _momentum_score(rets_dict, sym):
        """avg(multi-horizon returns) × consistency² — mirrors strategy1 scoring."""
        vals = []
        for col in ["ret_20d", "ret_60d", "ret_126d", "ret_252d"]:
            v = rets_dict.get(col, {}).get(sym)
            if v is not None and not np.isnan(v):
                vals.append(v)
        if len(vals) < 2:
            return np.nan
        consistency = sum(1 for r in vals if r > 0) / len(vals)
        return np.mean(vals) * (consistency ** 2)

    def extract_features(self, date, uni, candidates):
        """Extract 12 features from a FastUniverse/WRDSUniverse for candidate symbols.
        Returns {symbol: np.array(12,)} dict."""
        # Pre-fetch feature maps (one call per feature, not per symbol)
        fmaps = {}
        for name in ["gross_margin", "eps_surprise_last", "ret_10d_vs_sector",
                      "debt_to_equity", "roe", "dist_sma50", "vol_60d",
                      "ret_126d", "ret_20d", "ret_60d", "ret_252d"]:
            fmaps[name] = uni.get_feature_map(date, name)

        # Market breadth (same for all stocks)
        d50 = uni.get_feature_map(date, "dist_sma50")
        breadth = sum(1 for v in d50.values() if v > 0) / max(len(d50), 1) if d50 else 0.5

        features = {}
        for sym in candidates:
            mom = self._momentum_score(fmaps, sym)
            if np.isnan(mom):
                continue

            # Growth from enhanced data
            fg = getattr(uni, "_fin_growth", {}).get(sym, {})
            rev_g = fg.get("rev_growth", np.nan)
            eps_g = fg.get("eps_growth", np.nan)
            rev_g = float(rev_g) if rev_g is not None and isinstance(rev_g, (int, float)) else np.nan
            eps_g = float(eps_g) if eps_g is not None and isinstance(eps_g, (int, float)) else np.nan

            row = np.array([
                mom,
                fmaps.get("gross_margin", {}).get(sym, np.nan),
                fmaps.get("eps_surprise_last", {}).get(sym, np.nan),
                fmaps.get("ret_10d_vs_sector", {}).get(sym, np.nan),
                rev_g,
                eps_g,
                fmaps.get("debt_to_equity", {}).get(sym, np.nan),
                fmaps.get("roe", {}).get(sym, np.nan),
                fmaps.get("dist_sma50", {}).get(sym, np.nan),
                breadth,
                fmaps.get("vol_60d", {}).get(sym, np.nan),
                fmaps.get("ret_126d", {}).get(sym, np.nan),
            ], dtype=np.float64)

            features[sym] = row

        return features

    def predict(self, date, uni, candidates):
        """Score candidates. Returns {symbol: score} dict."""
        if self.model is None:
            return {}

        features = self.extract_features(date, uni, candidates)
        if len(features) < 2:
            return {}

        syms = list(features.keys())
        X = np.array([features[s] for s in syms])

        # XGBoost handles NaN natively
        dmat = xgb.DMatrix(X, feature_names=self.feature_names, missing=np.nan)
        scores = self.model.predict(dmat)

        return {s: float(sc) for s, sc in zip(syms, scores)}

    # ── Training ───────────────────────────────────────────────────────

    def build_training_data(self, universe_path):
        """Build training matrix from WRDS universe pickle.
        Returns DataFrame with: date, symbol, 12 features, target_decile, fwd_ret
        """
        log.info(f"Loading universe from {universe_path}...")
        with open(universe_path, "rb") as f:
            data = pickle.load(f)

        prices_df = data["prices_df"]
        features_by_date = data["features_by_date"]
        fin_growth = data.get("fin_growth", {})

        dates = sorted(features_by_date.keys())
        log.info(f"Building training data: {len(dates)} dates, "
                 f"{len(prices_df.columns)} tickers")

        rows = []
        for i, date in enumerate(dates):
            if i % 200 == 0:
                log.info(f"  Processing date {i}/{len(dates)}: {date.date()}")

            date_feats = features_by_date[date]
            if len(date_feats) < 50:
                continue

            # Market breadth for this date
            d50_vals = [f.get("dist_sma50", np.nan)
                        for f in date_feats.values()
                        if "dist_sma50" in f]
            d50_clean = [v for v in d50_vals if not np.isnan(v)]
            breadth = sum(1 for v in d50_clean if v > 0) / max(len(d50_clean), 1) if d50_clean else 0.5

            # Forward 10-day return: price[t+10] / price[t] - 1
            fwd_date_idx = i + PURGE_DAYS
            if fwd_date_idx >= len(dates):
                continue
            fwd_date = dates[fwd_date_idx]

            for sym, feats in date_feats.items():
                # Compute momentum score
                rets = []
                for col in ["ret_20d", "ret_60d", "ret_126d", "ret_252d"]:
                    v = feats.get(col)
                    if v is not None and not np.isnan(v):
                        rets.append(v)
                if len(rets) < 2:
                    continue
                consistency = sum(1 for r in rets if r > 0) / len(rets)
                mom_score = np.mean(rets) * (consistency ** 2)

                # Forward return from prices
                if sym not in prices_df.columns:
                    continue
                px_now = prices_df.loc[date, sym] if date in prices_df.index else np.nan
                px_fwd = prices_df.loc[fwd_date, sym] if fwd_date in prices_df.index else np.nan
                if np.isnan(px_now) or np.isnan(px_fwd) or px_now <= 0:
                    continue
                fwd_ret = px_fwd / px_now - 1

                # Growth data
                fg = fin_growth.get(sym, {})
                rev_g = fg.get("rev_growth", np.nan)
                eps_g = fg.get("eps_growth", np.nan)
                rev_g = float(rev_g) if isinstance(rev_g, (int, float)) else np.nan
                eps_g = float(eps_g) if isinstance(eps_g, (int, float)) else np.nan

                rows.append({
                    "date": date,
                    "symbol": sym,
                    "momentum_score": mom_score,
                    "gross_margin": feats.get("gross_margin", np.nan),
                    "eps_surprise_last": feats.get("eps_surprise_last", np.nan),
                    "ret_10d_vs_sector": feats.get("ret_10d_vs_sector", np.nan),
                    "rev_growth": rev_g,
                    "eps_growth": eps_g,
                    "debt_to_equity": feats.get("debt_to_equity", np.nan),
                    "roe": feats.get("roe", np.nan),
                    "dist_sma50": feats.get("dist_sma50", np.nan),
                    "market_breadth": breadth,
                    "vol_60d": feats.get("vol_60d", np.nan),
                    "ret_126d": feats.get("ret_126d", np.nan),
                    "fwd_ret": fwd_ret,
                })

        df = pd.DataFrame(rows)
        log.info(f"Raw training data: {len(df)} samples")

        # Compute cross-sectional decile labels per date
        def _decile(s):
            try:
                return pd.qcut(s, 10, labels=range(10), duplicates="drop").astype(float)
            except Exception:
                return pd.Series(np.nan, index=s.index)

        df["target_decile"] = df.groupby("date")["fwd_ret"].transform(_decile)
        df = df.dropna(subset=["target_decile"])
        log.info(f"After decile labeling: {len(df)} samples")

        return df

    def train(self, train_df, val_df=None, n_rounds=500, early_stopping=50):
        """Train XGBoost ranker on prepared data.
        train_df/val_df: DataFrames with columns: date, symbol, FEATURES, target_decile
        """
        X_train = train_df[FEATURES].values
        y_train = train_df["target_decile"].values

        # Group sizes: number of stocks per date (required for LambdaRank)
        groups_train = train_df.groupby("date").size().values

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES, missing=np.nan)
        dtrain.set_group(groups_train)

        evals = [(dtrain, "train")]
        params = dict(XGB_PARAMS)

        if val_df is not None and len(val_df) > 0:
            X_val = val_df[FEATURES].values
            y_val = val_df["target_decile"].values
            groups_val = val_df.groupby("date").size().values
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES, missing=np.nan)
            dval.set_group(groups_val)
            evals.append((dval, "val"))

        log.info(f"Training XGBoost: {len(X_train)} samples, "
                 f"{len(groups_train)} dates, {len(FEATURES)} features")

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping if val_df is not None else None,
            verbose_eval=50,
        )

        log.info(f"Training complete. Best iteration: {self.model.best_iteration}")

        # Feature importance
        importance = self.model.get_score(importance_type="gain")
        log.info("Feature importance (gain):")
        for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
            log.info(f"  {feat}: {score:.1f}")

        return self.model

    def train_walkforward(self, df, train_years=4, val_years=1):
        """Walk-forward training: train on N years, validate on next year.
        Returns list of (train_period, val_period, ndcg, model) tuples.
        """
        df["year"] = df["date"].dt.year
        years = sorted(df["year"].unique())
        results = []

        for i in range(train_years, len(years)):
            val_year = years[i]
            train_end_year = years[i - 1]
            train_start_year = years[max(0, i - train_years)]

            train_mask = (df["year"] >= train_start_year) & (df["year"] <= train_end_year)
            val_mask = df["year"] == val_year

            train_data = df[train_mask]
            val_data = df[val_mask]

            if len(train_data) < 1000 or len(val_data) < 100:
                continue

            log.info(f"\n{'='*50}")
            log.info(f"Walk-forward: train {train_start_year}-{train_end_year}, val {val_year}")
            log.info(f"  Train: {len(train_data)} samples, Val: {len(val_data)} samples")

            self.train(train_data, val_data, n_rounds=500, early_stopping=50)

            # Evaluate: compute NDCG@8 on validation set manually
            X_val = val_data[FEATURES].values
            y_val = val_data["target_decile"].values
            groups_val = val_data.groupby("date").size().values
            dval = xgb.DMatrix(X_val, feature_names=FEATURES, missing=np.nan)
            preds = self.model.predict(dval)

            # Compute per-date NDCG@8
            ndcgs = []
            idx = 0
            for g in groups_val:
                if g < 8:
                    idx += g
                    continue
                date_preds = preds[idx:idx + g]
                date_labels = y_val[idx:idx + g]
                # NDCG@8: how well does our ranking match ideal ranking?
                top8_idx = np.argsort(-date_preds)[:8]
                dcg = sum(date_labels[top8_idx[j]] / np.log2(j + 2) for j in range(min(8, len(top8_idx))))
                ideal_top8 = np.sort(date_labels)[::-1][:8]
                idcg = sum(ideal_top8[j] / np.log2(j + 2) for j in range(min(8, len(ideal_top8))))
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
                idx += g

            avg_ndcg = np.mean(ndcgs) if ndcgs else 0
            log.info(f"  Val NDCG@8: {avg_ndcg:.4f} (over {len(ndcgs)} dates)")

            results.append({
                "train_period": f"{train_start_year}-{train_end_year}",
                "val_year": val_year,
                "ndcg_at_8": avg_ndcg,
                "n_train": len(train_data),
                "n_val": len(val_data),
                "best_iter": self.model.best_iteration,
            })

        return results

    def train_from_universe(self, universe_path, save_path=None):
        """End-to-end: load data, train walk-forward, save best model.
        Returns walk-forward results.
        """
        df = self.build_training_data(universe_path)

        # Walk-forward training
        results = self.train_walkforward(df, train_years=4, val_years=1)

        # Final model: train on all data except last year (for production)
        df["year"] = df["date"].dt.year
        years = sorted(df["year"].unique())
        final_train = df[df["year"] < years[-1]]
        final_val = df[df["year"] == years[-1]]

        log.info(f"\nFinal model: train {years[0]}-{years[-2]}, holdout {years[-1]}")
        self.train(final_train, final_val, n_rounds=500, early_stopping=50)

        if save_path:
            self.save(save_path)

        return results
