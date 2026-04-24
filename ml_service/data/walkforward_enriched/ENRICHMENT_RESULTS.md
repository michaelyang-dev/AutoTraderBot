# Feature Enrichment Results

Generated: 2026-04-23 21:23

## New Features Added

| Feature | Type | PIT Method | Intuition |
|---------|------|-----------|-----------|
| earnings_beat_streak | Earnings | count consecutive quarters where eps_actual > eps_estimated, using reportedDate | Companies on a streak tend to continue outperforming |
| revenue_surprise_last | Earnings | (revenue_actual - revenue_estimated) / |estimate|, available from reportedDate | Revenue beats are strong buy signals, underused vs EPS |
| earnings_surprise_3q_avg | Earnings | avg EPS surprise over last 3 reported quarters | More stable than single-quarter surprise |
| revenue_surprise_3q_avg | Earnings | avg revenue surprise over last 3 reported quarters | More stable than single-quarter |
| insider_buy_cluster | Insiders | binary: 3+ insider buys in trailing 90 calendar days | Clustered buying = high conviction signal |
| insider_net_shares_180d | Insiders | net shares bought minus sold, 180-day window | Longer window captures slower insider trends |

## Feature Importance (Top 15)

### Baseline Model

| Rank | Feature | Importance | % |
|------|---------|-----------|---|
| 1 | days_since_earnings | 936 | 13.2% |
| 2 | vol_rank_6m | 868 | 12.3% |
| 3 | current_ratio | 588 | 8.3% |
| 4 | tlt_ret_120d | 481 | 6.8% |
| 5 | vixy_level | 471 | 6.7% |
| 6 | dist_52w_high | 425 | 6.0% |
| 7 | hyg_lqd_ratio | 417 | 5.9% |
| 8 | revenue_growth_yoy | 415 | 5.9% |
| 9 | dxy_30d_change | 396 | 5.6% |
| 10 | month | 375 | 5.3% |
| 11 | ps_vs_universe_median | 358 | 5.1% |
| 12 | return_rank_12m | 350 | 4.9% |
| 13 | operating_margin | 337 | 4.8% |
| 14 | ps_ratio | 334 | 4.7% |
| 15 | debt_to_equity | 323 | 4.6% |

### Enriched Model

| Rank | Feature | Importance | % |
|------|---------|-----------|---|
| 1 | days_since_earnings | 1029 | 12.4% |
| 2 | vol_rank_6m | 955 | 11.5% |
| 3 | current_ratio | 652 | 7.8% |
| 4 | vixy_level | 599 | 7.2% |
| 5 | tlt_ret_120d | 583 | 7.0% |
| 6 | hyg_lqd_ratio | 512 | 6.2% |
| 7 | dist_52w_high | 486 | 5.8% |
| 8 | revenue_growth_yoy | 475 | 5.7% |
| 9 | dxy_30d_change | 449 | 5.4% |
| 10 | dxy_level | 444 | 5.3% |
| 11 | month | 441 | 5.3% |
| 12 | debt_to_equity | 438 | 5.3% |
| 13 | operating_margin | 434 | 5.2% |
| 14 | return_rank_12m | 424 | 5.1% |
| 15 | roe | 394 | 4.7% |

## Walk-Forward Comparison

| Year | Baseline Top-5 | Enriched Top-5 | Baseline Spread | Enriched Spread | Baseline AUC | Enriched AUC |
|------|---------------|---------------|----------------|----------------|-------------|-------------|
| 2015 | +0.346% | +0.421% | +0.452% | +0.676% | 0.6000 | 0.6003 |
| 2016 | +2.768% | +2.476% | +2.280% | +2.009% | 0.5955 | 0.5969 |
| 2017 | +1.002% | +0.471% | +0.153% | -0.084% | 0.6137 | 0.6137 |
| 2018 | -0.013% | -0.179% | +0.275% | +0.160% | 0.6228 | 0.6274 |
| 2019 | +1.932% | +1.806% | +1.456% | +1.195% | 0.6031 | 0.6037 |
| 2020 | +2.566% | +2.054% | +1.787% | +1.483% | 0.6031 | 0.6021 |
| 2021 | +1.792% | +2.208% | +1.031% | +1.652% | 0.6103 | 0.6110 |
| 2022 | +0.050% | +0.087% | +0.555% | +0.409% | 0.6073 | 0.6073 |
| 2023 | +0.975% | +0.675% | +0.597% | +0.483% | 0.5960 | 0.5945 |
| 2024 | +0.719% | +1.195% | +0.368% | +0.691% | 0.5982 | 0.5981 |
| 2025 | +2.260% | +1.960% | +1.657% | +1.259% | 0.6223 | 0.6206 |

### Baseline Summary
- Median top-5 forward return: +1.002%
- Mean top-5 forward return: +1.309%
- Median spread (top5 - bot5): +0.597%
- Mean AUC: 0.6066
- Positive top-5 years: 10/11
- Positive spread years: 11/11

### Enriched Summary
- Median top-5 forward return: +1.195%
- Mean top-5 forward return: +1.198%
- Median spread (top5 - bot5): +0.691%
- Mean AUC: 0.6069
- Positive top-5 years: 10/11
- Positive spread years: 10/11

## Recommendation

**KEEP BASELINE**: Enriched features only improve 4/11 years — not worth production complexity.
- AUC improved in 7/11 years
- Median top-5 change: +0.193pp
- Mean AUC change: +0.0003