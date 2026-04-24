# Phase 1C: Short Model v3 — Stocks Only, Bottom 10%

Generated: 2026-04-23 19:16

## Model Design

- **Target**: `target_short` = bottom **10%** of S&P 500 stocks by forward 10-day return
- **Architecture**: LGBM + XGBoost ensemble
- **Features**: Short-specific (accruals, debt, margin compression, insider selling, earnings misses, price breakdowns)
- **Critical fix**: ETFs excluded (VIXY/USO/PSKY were contaminating prior models)
- **Walk-forward**: Train on years before Y, predict year Y
- **Key question**: Do the model's top-5 picks (highest `prob_short`) actually go DOWN?

## Model Quality (AUC per year)

| Year | LGBM AUC | XGB AUC |
|------|----------|---------|
| 2015 | 0.6504 | 0.6234 |
| 2016 | 0.6702 | 0.6363 |
| 2017 | 0.6531 | 0.6283 |
| 2018 | 0.6335 | 0.6159 |
| 2019 | 0.6329 | 0.6045 |
| 2020 | 0.6170 | 0.5838 |
| 2021 | 0.6224 | 0.6134 |
| 2022 | 0.6350 | 0.6186 |
| 2023 | 0.6439 | 0.6179 |
| 2024 | 0.6349 | 0.6204 |
| 2025 | 0.6369 | 0.6228 |

## Short Signal Validation

"Short candidates" = top-5 by `prob_short` (model thinks these will be bottom-20% performers)

| Year | Days | Short Cands Fwd | Non-Short Fwd | Random | Spread (NS-SC) | Spread p | SC<0 p |
|------|------|-----------------|---------------|--------|----------------|----------|--------|
| 2015 | 252 | -0.700% | -0.161% | -0.019% | +0.539%** | 0.0163 | 0.0187 |
| 2016 | 252 | +1.867% | +1.148% | +0.693% | -0.720%** | 0.0107 | 0.0000 |
| 2017 | 251 | +1.335% | +0.971% | +0.960% | -0.364%* | 0.0950 | 0.0000 |
| 2018 | 251 | -0.147% | -0.339% | -0.152% | -0.192% | 0.5493 | 0.7201 |
| 2019 | 252 | +1.979% | +1.071% | +0.986% | -0.909%*** | 0.0032 | 0.0000 |
| 2020 | 253 | +1.577% | +0.388% | +0.887% | -1.190% | 0.1525 | 0.1305 |
| 2021 | 252 | +1.193% | +1.051% | +0.646% | -0.141% | 0.7037 | 0.0007 |
| 2022 | 251 | -0.090% | +0.695% | -0.241% | +0.785%* | 0.0535 | 0.8448 |
| 2023 | 250 | -0.144% | +0.523% | +0.485% | +0.667%** | 0.0254 | 0.6190 |
| 2024 | 252 | +0.083% | +0.457% | +0.652% | +0.374% | 0.3156 | 0.8178 |
| 2025 | 250 | +2.758% | +0.315% | +0.721% | -2.443%*** | 0.0000 | 0.0000 |

## Aggregate Summary

- **Short candidates avg fwd return**: +0.883%
- **Non-short avg fwd return**: +0.556%
- **Random avg fwd return**: +0.511%
- **Spread (non-short minus short cands)**: -0.327%
- **Spread t-stat**: -2.61, p=0.009231
- **Short cands < 0**: t=6.17, p=0.000000
- **Positive spread years**: 4/11
- **Short cands negative years**: 4/11

## Quintile Analysis (by prob_short)

| Quintile | Avg Fwd 10d Ret | vs Random |
|----------|-----------------|-----------|

## Decision

**NO SHORT SIGNAL** (short candidates avg = +0.883% >= 0%)

Even a dedicated short model cannot identify stocks that go down. The feature set does not predict the downside tail.

### Comparison: Long model vs Short model

| Metric | Long Model (Phase 1) | Short Model (this) |
|--------|---------------------|--------------------|
| Top-5 picks fwd return | +1.44% (winners) | +0.883% (losers) |
| Bottom-5 picks fwd return | +0.33% | +0.556% |
| T-B spread | +1.11% | -0.327% |
| Spread p-value | <0.001 | 0.0092 |