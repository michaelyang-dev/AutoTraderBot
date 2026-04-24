# Feature Audit: FMP Data Usage

Generated: 2026-04-23

## Current Feature Set (83 columns in features.parquet)

### Technical (28 features) — all used
ret_5d, ret_10d, ret_20d, ret_60d, ret_120d, ret_126d, ret_252d,
vol_10d, vol_20d, vol_60d, vol_ratio_20d,
rsi_14, macd_line, macd_signal, bb_position,
dist_sma50, dist_sma200, sma200_slope,
new_high_20d, new_high_50d, obv_trend_20d,
dist_52w_high, dist_52w_low, max_dd_6m,
consec_up_months, consec_down_months, vol_126d (implied from vol_rank_6m)

### Fundamental (21 features) — from FMP
| Feature | Source File | Used? | Notes |
|---------|------------|-------|-------|
| revenue_growth_yoy | fundamentals_income | YES | |
| eps_growth_yoy | fundamentals_income | YES | |
| revenue_growth_qoq | fundamentals_income | YES | |
| gross_margin | fundamentals_income | YES | |
| operating_margin | fundamentals_income | YES | |
| net_margin | fundamentals_income | YES | |
| margin_trend_4q | fundamentals_income | YES | |
| pe_ratio | fundamentals_ratios | YES | |
| ps_ratio | fundamentals_ratios | YES | |
| pe_vs_universe_median | fundamentals_ratios | YES | cross-sectional |
| ps_vs_universe_median | fundamentals_ratios | YES | cross-sectional |
| debt_to_equity | fundamentals_ratios | YES | |
| current_ratio | fundamentals_ratios | YES | |
| roe | fundamentals_metrics | YES | |
| roa | fundamentals_metrics | YES | |
| days_since_earnings | fundamentals_earnings | YES | |
| eps_surprise_last | fundamentals_earnings | YES | single most recent quarter |
| eps_revision_30d | fundamentals_estimates | **NaN** | set to NaN (line 449) — only 4 annual estimates per symbol, can't compute |
| revenue_revision_30d | fundamentals_estimates | **NaN** | set to NaN (line 450) — same issue |
| insider_buy_ratio_90d | fundamentals_insiders | YES | |
| insider_net_shares_90d | fundamentals_insiders | YES | |

### Cross-Asset (13) — all used
spy_ret_5d/10d/20d/60d/120d, tlt_ret_5d/10d/20d/60d/120d, vixy_level, vixy_ret_5d, vixy_ret_20d

### Macro (8) — all used  
yield_curve_10y2y, yield_curve_30d_change, hy_spread, hy_spread_30d_change,
dxy_level, dxy_30d_change, copper_gold_ratio, copper_gold_30d_change, hyg_lqd_ratio, hyg_lqd_30d_change

### Cross-Sectional Ranks (9) — all used
return_rank_3m/6m/12m, vol_rank_3m/6m, + 4 others

### Other (4)
in_sp500, day_of_week, month, quarter, target, date, symbol

## FMP Endpoint Audit

| Endpoint | Fetched? | Saved? | Path | Features Derived | Untapped Data |
|----------|----------|--------|------|-----------------|---------------|
| /stable/earnings | YES | YES | fundamentals_earnings.parquet (64K rows, 503 syms) | eps_surprise_last, days_since_earnings | **earnings beat streak, revenue surprise, 3Q avg surprise** |
| /stable/analyst-estimates | YES | YES | fundamentals_estimates.parquet (2K rows, 503 syms) | eps_revision_30d (NaN), revenue_revision_30d (NaN) | Only ~4 annual estimates/symbol — **unusable for revisions** |
| /stable/insider-trading | YES | YES | fundamentals_insiders.parquet (50K rows, 503 syms) | insider_buy_ratio_90d, insider_net_shares_90d | **buy clustering, larger lookback windows** |
| /stable/key-metrics | YES | YES | fundamentals_metrics.parquet (29K rows) | roe, roa | All columns used |
| /stable/ratios | YES | YES | fundamentals_ratios.parquet (29K rows) | pe, ps, d/e, current_ratio | All columns used |

## Untapped Signal Summary

1. **Earnings beat streak** — data available (64K earnings rows with actual+estimate). Not computed.
2. **Revenue surprise** — revenue_actual vs revenue_estimated available for 47K rows. Not used at all.
3. **Multi-quarter earnings surprise average** — can average last 3Q surprises for stability.
4. **Insider buy clustering** — binary flag when 3+ insiders buy in 90 days. Not computed.
5. **Estimate revisions** — BLOCKED. Only 4 annual estimates per symbol, no daily snapshots. Cannot compute meaningful revision metrics.
