# FMP Alt-Data Experiment: Insider Trading Features

**Date**: 2026-04-22
**Branch**: experiment/fmp-alpha-features
**Verdict**: DO NOT DEPLOY — baseline outperforms on all metrics

## Hypothesis

Insider trading signals (Form 4 filings from FMP API) could improve ML model
predictions by capturing information asymmetry. Specifically:

- Net insider buying → bullish signal
- Clustered insider activity → coordinated conviction
- CEO/director purchases → high-conviction rare events
- Dollar-weighted buy volume → magnitude of insider conviction

## Features Added (5 new, keyed on filingDate for point-in-time integrity)

| Feature | Description | Non-zero % | Mean |
|---|---|---|---|
| insider_net_buy_count_30d | #buys - #sells in trailing 30d | 20.6% | -0.94 |
| insider_cluster_score_10d | #distinct insiders filing in 10d | 10.4% | 0.18 |
| insider_buy_dollar_30d | log1p(buy dollar value) in 30d | 2.9% | 0.36 |
| insider_ceo_buy_90d | Binary: CEO bought in 90d | 0.23% | 0.002 |
| insider_director_buy_90d | Binary: director bought in 90d | 6.1% | 0.061 |

## Data Pipeline

1. **fmp_altdata_client.py** — fetched insider trades for 501 S&P 500 symbols via FMP `/stable/insider-trading/search` endpoint. Cached as parquet files in `data/altdata_cache/`.
2. **compute_insider_features.py** — computed 5 features per (symbol, date) using rolling windows with strict point-in-time cutoff (filingDate < date - 1 day). Merged into features_v5.parquet (88 columns).
3. **train_altdata_model.py** — trained LGBM + RF ensemble on features_v5.parquet.

## Training Results

| Metric | Baseline (83 feat) | Alt-Data (88 feat) |
|---|---|---|
| LGBM trees (early stop) | 41 | 41 |
| LGBM Calibration AUC | 0.6152 | 0.6252 |
| RF Calibration AUC | — | 0.6236 |
| Ensemble Calibration AUC | — | 0.6263 |

**Feature importance**: All 7 insider features (including 2 existing from fundamentals pipeline) ranked #79–88 with **importance = 0** in LightGBM. The model found zero predictive signal in insider trading data.

## A/B Backtest (combined_live strategy, full history)

| Metric | A: Baseline | B: Alt-Data | Delta |
|---|---|---|---|
| CAGR | 47.11% | 42.24% | **-4.87%** |
| Sharpe | 2.204 | 2.016 | **-0.187** |
| Sortino | 2.845 | 2.557 | **-0.288** |
| Max Drawdown | -37.6% | -41.3% | **-3.6%** |
| Alpha vs SPY | 34.45% | 30.01% | **-4.44%** |
| Win Rate | 55.1% | 54.7% | **-0.4%** |
| Trades | 2,422 | 2,419 | -3 |

Backtest period: 2011-12-29 → 2026-04-02 (14.3 years, 534 symbols)

## Decision

**Decision rule**: Alpha >= +17% AND Sharpe >= 1.40 = clear win.

Both models pass absolute thresholds, but the alt-data model **degrades** performance on every metric. The insider features added noise without signal.

**VERDICT: DO NOT DEPLOY. Keep baseline 83-feature model.**

## Why Insider Features Failed

1. **S&P 500 insiders mostly sell** (net_buy mean = -0.94): Option exercises and diversification dominate; buying is rare and often defensive.
2. **CEO buys are extremely sparse** (0.23% non-zero): Too rare for tree-based models to learn from reliably.
3. **Market already prices Form 4 filings**: Institutional algo traders scrape EDGAR filings within minutes. By the next trading day (our point-in-time cutoff), the signal is fully absorbed.
4. **Cross-sectional prediction is harder**: We're predicting top-quintile relative performance, not absolute returns. Even if insider buying is mildly bullish in absolute terms, it doesn't reliably predict relative outperformance among S&P 500 peers.

## Files Created

- `ml_service/fmp_altdata_client.py` — FMP alt-data client (reusable)
- `ml_service/compute_insider_features.py` — insider feature computation
- `ml_service/train_altdata_model.py` — training script for 88-feature model
- `ml_service/backtest_altdata_compare.py` — A/B backtest comparison
- `ml_service/data/altdata_cache/` — 501 insider parquet cache files
- `ml_service/data/features_v5.parquet` — 88-feature dataset
- `ml_service/data/predictions_v5.parquet` — alt-data model predictions
- `ml_service/data/model_v5.lgb` — calibrated LightGBM (do not deploy)
- `ml_service/data/model_v5_rf.pkl` — calibrated Random Forest (do not deploy)
