# Phase 1B: Short Model v2 — Short-Specific Features

Generated: 2026-04-23 19:00

## Model Design

- **Target**: `target_short` = bottom 20% of S&P 500 by forward 10-day return
- **Architecture**: LGBM + XGBoost ensemble
- **Features**: Short-specific (accruals quality, debt deterioration, margin compression, insider selling, earnings misses, price breakdowns)
- **Walk-forward**: Train on years before Y, predict year Y
- **Key question**: Do the model's top-5 picks (highest `prob_short`) actually go DOWN?

## Model Quality (AUC per year)

| Year | LGBM AUC | XGB AUC |
|------|----------|---------|
| 2015 | 0.5908 | 0.5600 |
| 2016 | 0.6162 | 0.5495 |
| 2017 | 0.5999 | 0.5362 |
| 2018 | 0.5929 | 0.5493 |
| 2019 | 0.5959 | 0.5338 |
| 2020 | 0.5756 | 0.5087 |
| 2021 | 0.5802 | 0.5482 |
| 2022 | 0.5906 | 0.5411 |
| 2023 | 0.5987 | 0.5389 |
| 2024 | 0.5895 | 0.5386 |
| 2025 | 0.5941 | 0.5586 |

## Short Signal Validation

"Short candidates" = top-5 by `prob_short` (model thinks these will be bottom-20% performers)

| Year | Days | Short Cands Fwd | Non-Short Fwd | Random | Spread (NS-SC) | Spread p | SC<0 p |
|------|------|-----------------|---------------|--------|----------------|----------|--------|
| 2015 | 252 | -0.678% | -0.217% | +0.013% | +0.461% | 0.1900 | 0.0007 |
| 2016 | 252 | +0.407% | +0.558% | +0.658% | +0.151% | 0.5149 | 0.0384 |
| 2017 | 251 | +0.038% | +0.813% | +1.120% | +0.775%*** | 0.0002 | 0.8355 |
| 2018 | 251 | +0.773% | -0.228% | -0.091% | -1.001%*** | 0.0012 | 0.0005 |
| 2019 | 252 | +0.288% | +0.797% | +1.013% | +0.509%** | 0.0311 | 0.1008 |
| 2020 | 253 | +0.511% | +0.639% | +0.797% | +0.129% | 0.8103 | 0.3752 |
| 2021 | 252 | -0.470% | +0.619% | +0.948% | +1.090%*** | 0.0006 | 0.0741 |
| 2022 | 251 | -0.012% | +0.357% | -0.269% | +0.369% | 0.2973 | 0.9715 |
| 2023 | 250 | -0.690% | +0.324% | +0.488% | +1.014%*** | 0.0000 | 0.0018 |
| 2024 | 252 | -0.547% | +0.618% | +0.582% | +1.165%*** | 0.0000 | 0.0214 |
| 2025 | 250 | +0.913% | +0.727% | +0.527% | -0.187% | 0.6286 | 0.0154 |

## Aggregate Summary

- **Short candidates avg fwd return**: +0.048%
- **Non-short avg fwd return**: +0.455%
- **Random avg fwd return**: +0.526%
- **Spread (non-short minus short cands)**: +0.407%
- **Spread t-stat**: 4.16, p=0.000033
- **Short cands < 0**: t=0.54, p=0.586431
- **Positive spread years**: 9/11
- **Short cands negative years**: 5/11

## Quintile Analysis (by prob_short)

| Quintile | Avg Fwd 10d Ret | vs Random |
|----------|-----------------|-----------|

## Decision

**NO SHORT SIGNAL** (short candidates avg = +0.048% >= 0%)

Even a dedicated short model cannot identify stocks that go down. The feature set does not predict the downside tail.

### Comparison: Long model vs Short model

| Metric | Long Model (Phase 1) | Short Model (this) |
|--------|---------------------|--------------------|
| Top-5 picks fwd return | +1.44% (winners) | +0.048% (losers) |
| Bottom-5 picks fwd return | +0.33% | +0.455% |
| T-B spread | +1.11% | +0.407% |
| Spread p-value | <0.001 | 0.0000 |