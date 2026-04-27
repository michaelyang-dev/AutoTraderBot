"""
ML Factor Ranker
================
LightGBM ranking model that re-ranks strategy1's candidate stocks
using 12 clean factor scores as features. Acts as a re-optimizer
on top of the proven factor strategy, not a replacement.

Usage:
    # Training:
    python3 -m strategies.train_factor_ranker

    # In strategy1:
    ranker = MLFactorRanker("data/ml_factor_model.pkl")
    ml_scores = ranker.predict(date, uni, candidates)
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger("ml_factor_ranker")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# 12 factor features — all backtest-safe, economically meaningful
FACTOR_FEATURES = [
    "momentum_score",     # avg(ret_20d,60d,126d,252d) * consistency^2
    "gross_margin",       # profitability quality
    "eps_surprise_last",  # earnings beat/miss
    "ret_10d_vs_sector",  # sector-relative strength
    "rev_growth",         # revenue acceleration (from FMP)
    "eps_growth",         # EPS acceleration (from FMP)
    "debt_to_equity",     # leverage risk
    "roe",                # management efficiency
    "dist_sma50",         # trend strength
    "market_breadth",     # regime context (% above 50d SMA)
    "vol_60d",            # risk/volatility
    "ret_126d",           # 6-month momentum
]

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [8],
    "num_leaves": 16,
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

PURGE_DAYS = 10  # match 10-day forward return horizon


class MLFactorRanker:
    """LightGBM ranking model on factor scores."""

    def __init__(self, model_path=None):
        self.model = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

    # ── Feature Extraction ──────────────────────────────────────────

    @staticmethod
    def compute_momentum_score(row):
        """Replicate strategy1's base score: avg_ret * consistency^2."""
        rets = []
        for col in ["ret_20d", "ret_60d", "ret_126d", "ret_252d"]:
            v = row.get(col)
            if v is not None and not np.isnan(v):
                rets.append(v)
        if len(rets) < 2:
            return np.nan
        consistency = sum(1 for r in rets if r > 0) / len(rets)
        avg_ret = np.mean(rets)
        return avg_ret * (consistency ** 2)

    @staticmethod
    def compute_market_breadth(date_df):
        """% of stocks above 50d SMA on a given date."""
        d50 = date_df["dist_sma50"].dropna()
        if len(d50) < 50:
            return 0.5
        return (d50 > 0).mean()

    def extract_features_from_uni(self, date, uni, candidates):
        """Extract 12 factor features for a list of candidate symbols from FastUniverse.
        Returns dict {symbol: np.array of 12 features}."""
        features = {}
        # Pre-fetch feature maps
        fmaps = {}
        for feat_name in ["gross_margin", "eps_surprise_last", "ret_10d_vs_sector",
                          "debt_to_equity", "roe", "dist_sma50", "vol_60d", "ret_126d",
                          "ret_20d", "ret_60d", "ret_252d"]:
            fmaps[feat_name] = uni.get_feature_map(date, feat_name)

        # Market breadth (same for all stocks on this date)
        dist_sma50_all = uni.get_feature_map(date, "dist_sma50")
        breadth = sum(1 for v in dist_sma50_all.values() if v > 0) / max(len(dist_sma50_all), 1) if dist_sma50_all else 0.5

        for sym in candidates:
            # Momentum score
            rets = []
            for col in ["ret_20d", "ret_60d", "ret_126d", "ret_252d"]:
                v = fmaps.get(col, {}).get(sym)
                if v is not None and not np.isnan(v):
                    rets.append(v)
            if len(rets) < 2:
                continue
            consistency = sum(1 for r in rets if r > 0) / len(rets)
            mom_score = np.mean(rets) * (consistency ** 2)

            # Financial growth from enhanced data
            fg = uni._fin_growth.get(sym, {})
            rev_g = fg.get("rev_growth", np.nan)
            eps_g = fg.get("eps_growth", np.nan)
            if rev_g is not None and isinstance(rev_g, (int, float)):
                rev_g = float(rev_g)
            else:
                rev_g = np.nan
            if eps_g is not None and isinstance(eps_g, (int, float)):
                eps_g = float(eps_g)
            else:
                eps_g = np.nan

            row = np.array([
                mom_score,
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

    # ── Training Data Builder ───────────────────────────────────────

    def build_training_data(self, features_df, fin_growth_df=None,
                            start_date="2017-01-01", end_date="2025-12-31"):
        """Build training matrix from features.parquet + financial_growth.parquet.

        Returns DataFrame with columns: date, symbol, f1..f12, target_decile, fwd_ret
        """
        df = features_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Filter date range and SP500
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        if "in_sp500" in df.columns:
            mask &= df["in_sp500"] == True
        df = df[mask].copy()

        # Compute forward 10-day return per symbol
        df = df.sort_values(["symbol", "date"])
        df["fwd_ret"] = df.groupby("symbol")["ret_10d"].shift(-2)
        # shift(-2) because ret_10d is the PAST 10-day return; we want FUTURE
        # Actually, let's compute it directly from close prices
        # fwd_10d = close(t+10) / close(t) - 1
        # But we don't have close in features.parquet directly.
        # Use the shift approach: for each symbol, shift ret_10d by -1 gives
        # the return that starts on date+1. But that's overlapping.
        # Better: compute from sequential dates
        # Actually the simplest: ret_10d at date t+10 IS the return from t to t+10
        # So fwd_ret for date t = ret_10d at date t+10
        # Which is shift(-10/rebal_period). Since dates are daily, shift(-10) for the symbol.
        # But dates may have gaps. Let's use a direct approach.
        df["fwd_ret"] = df.groupby("symbol").apply(
            lambda g: g["ret_10d"].shift(-10) if len(g) > 10 else pd.Series(np.nan, index=g.index)
        ).reset_index(level=0, drop=True)

        # Drop rows without forward returns
        df = df.dropna(subset=["fwd_ret"])

        # Compute factor features
        # 1. momentum_score
        def _mom_score(row):
            rets = [row.get(c) for c in ["ret_20d", "ret_60d", "ret_126d", "ret_252d"]]
            rets = [r for r in rets if r is not None and not np.isnan(r)]
            if len(rets) < 2:
                return np.nan
            c = sum(1 for r in rets if r > 0) / len(rets)
            return np.mean(rets) * (c ** 2)

        df["momentum_score"] = df.apply(_mom_score, axis=1)

        # 2. market_breadth (per date)
        breadth = df.groupby("date")["dist_sma50"].apply(
            lambda s: (s.dropna() > 0).mean() if len(s.dropna()) > 50 else 0.5
        ).rename("market_breadth")
        df = df.merge(breadth, on="date", how="left")

        # 3. Financial growth (asof merge)
        if fin_growth_df is not None and len(fin_growth_df) > 0:
            fg = fin_growth_df.copy()
            fg["date"] = pd.to_datetime(fg["date"])
            fg = fg.sort_values(["symbol", "date"])
            # Get latest quarterly report before each daily date
            fg_latest = fg.groupby("symbol").apply(
                lambda g: g[["date", "revenue_growth", "eps_growth"]].set_index("date").resample("D").ffill()
            ).reset_index()
            if "level_0" in fg_latest.columns:
                fg_latest = fg_latest.rename(columns={"level_0": "symbol"})
            # Merge
            fg_latest = fg_latest.rename(columns={"revenue_growth": "rev_growth", "eps_growth": "eps_growth_fg"})
            df = df.merge(fg_latest[["symbol", "date", "rev_growth", "eps_growth_fg"]],
                          on=["symbol", "date"], how="left")
            if "eps_growth_fg" in df.columns:
                df["eps_growth"] = df["eps_growth_fg"]
                df.drop(columns=["eps_growth_fg"], inplace=True)
            if "rev_growth" not in df.columns:
                df["rev_growth"] = np.nan
        else:
            df["rev_growth"] = np.nan
            if "eps_growth" not in df.columns:
                df["eps_growth"] = np.nan

        # Rename eps_growth_yoy if needed
        if "eps_growth" not in df.columns and "eps_growth_yoy" in df.columns:
            df["eps_growth"] = df["eps_growth_yoy"]
        if "rev_growth" not in df.columns and "revenue_growth_yoy" in df.columns:
            df["rev_growth"] = df["revenue_growth_yoy"]

        # Compute target: cross-sectional decile of fwd_ret per date
        def _decile(s):
            try:
                return pd.qcut(s, 10, labels=range(10), duplicates="drop").astype(float)
            except Exception:
                return pd.Series(np.nan, index=s.index)

        df["target_decile"] = df.groupby("date")["fwd_ret"].transform(_decile)
        df = df.dropna(subset=["target_decile", "momentum_score"])

        # Select only the columns we need
        out_cols = ["date", "symbol"] + FACTOR_FEATURES + ["target_decile", "fwd_ret"]
        available = [c for c in out_cols if c in df.columns]
        result = df[available].copy()

        # Fill missing factor features with NaN (LightGBM handles natively)
        for c in FACTOR_FEATURES:
            if c not in result.columns:
                result[c] = np.nan

        return result.sort_values(["date", "symbol"]).reset_index(drop=True)

    # ── Training ────────────────────────────────────────────────────

    def train(self, train_df, val_df=None, params=None):
        """Train LGBMRanker on factor features.

        train_df must have columns: date, FACTOR_FEATURES, target_decile
        Returns the trained model.
        """
        if params is None:
            params = LGB_PARAMS.copy()

        X_train = train_df[FACTOR_FEATURES].values
        y_train = train_df["target_decile"].values.astype(int)

        # Group sizes: number of stocks per date
        train_groups = train_df.groupby("date").size().values

        n_est = params.pop("n_estimators", 400)

        train_set = lgb.Dataset(X_train, label=y_train, group=train_groups,
                                feature_name=FACTOR_FEATURES, free_raw_data=False)

        callbacks = [lgb.log_evaluation(100)]
        val_sets = []
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[FACTOR_FEATURES].values
            y_val = val_df["target_decile"].values.astype(int)
            val_groups = val_df.groupby("date").size().values
            val_set = lgb.Dataset(X_val, label=y_val, group=val_groups,
                                  reference=train_set, free_raw_data=False)
            val_sets = [val_set]
            callbacks.append(lgb.early_stopping(50, verbose=True))

        self.model = lgb.train(
            params, train_set,
            num_boost_round=n_est,
            valid_sets=val_sets,
            callbacks=callbacks,
        )

        return self.model

    # ── Prediction ──────────────────────────────────────────────────

    def predict_from_df(self, df):
        """Predict scores from a DataFrame with FACTOR_FEATURES columns.
        Returns numpy array of scores (higher = better).
        """
        if self.model is None:
            return np.zeros(len(df))
        X = df[FACTOR_FEATURES].values
        return self.model.predict(X)

    def predict(self, date, uni, candidates):
        """Predict ML scores for candidates using FastUniverse data.
        Returns {symbol: score}.
        """
        if self.model is None:
            return {}

        features = self.extract_features_from_uni(date, uni, candidates)
        if not features:
            return {}

        syms = list(features.keys())
        X = np.array([features[s] for s in syms])
        scores = self.model.predict(X)

        return {sym: float(score) for sym, score in zip(syms, scores)}

    # ── Evaluation ──────────────────────────────────────────────────

    @staticmethod
    def evaluate_ndcg(df, score_col="ml_score", k=8):
        """Compute NDCG@k per date, return mean."""
        from sklearn.metrics import ndcg_score as _ndcg

        ndcgs = []
        for date, grp in df.groupby("date"):
            if len(grp) < k:
                continue
            true_rel = grp["target_decile"].values.reshape(1, -1)
            pred_scores = grp[score_col].values.reshape(1, -1)
            try:
                ndcgs.append(_ndcg(true_rel, pred_scores, k=k))
            except Exception:
                continue

        return np.mean(ndcgs) if ndcgs else 0.0

    @staticmethod
    def compare_top_k_returns(df, factor_col="momentum_score", ml_col="ml_score",
                              blend_col="blend_score", k=8):
        """Compare average forward returns of top-k picks by different ranking methods."""
        results = {}
        for col, label in [(factor_col, "Factor"), (ml_col, "ML"), (blend_col, "Blend")]:
            if col not in df.columns:
                continue
            fwd_rets = []
            for date, grp in df.groupby("date"):
                if len(grp) < k:
                    continue
                top_k = grp.nlargest(k, col)
                fwd_rets.append(top_k["fwd_ret"].mean())
            results[label] = {
                "avg_10d_ret": np.mean(fwd_rets) if fwd_rets else 0,
                "median_10d_ret": np.median(fwd_rets) if fwd_rets else 0,
                "hit_rate": np.mean([r > 0 for r in fwd_rets]) if fwd_rets else 0,
                "n_periods": len(fwd_rets),
            }
        return results
