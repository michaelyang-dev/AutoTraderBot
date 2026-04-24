# Short Model v2 — Deep Analysis & Improvement Research

Generated: 2026-04-23 19:06

## 1. Top-N Concentration

Does concentrating on fewer, highest-conviction short candidates help?

| Top-N | Avg Fwd Ret | Median Fwd Ret | Neg Years | % Neg Days | t-stat | p-value |
|-------|-------------|----------------|-----------|------------|--------|---------|
| 1 | -1.340% | -1.600% | 7/11 | 62.0% | -5.03 | 0.0000 |
| 2 | -0.395% | -0.362% | 7/11 | 59.9% | -2.80 | 0.0052 |
| 3 | -0.159% | -0.313% | 6/11 | 56.0% | -1.55 | 0.1222 |
| 5 | +0.048% | +0.038% | 5/11 | 51.9% | 0.54 | 0.5864 |
| 10 | +0.279% | +0.156% | 3/11 | 47.1% | 3.01 | 0.0027 |

**Best**: Top-1 with avg fwd return = -1.340%

## 2. Regime Filtering

Only short when market conditions are favorable for shorts.

| Regime | Avg Fwd Ret | Neg Years | Active % | Days | t-stat | p-value |
|--------|-------------|-----------|----------|------|--------|---------|
| all | +0.048% | 5/11 | 100% | 2766 | 0.54 | 0.5864 |
| spy_below_sma200 | +0.308% | 4/8 | 18% | 491 | 0.44 | 0.6610 |
| spy_ret_20d_neg | -0.492% | 9/11 | 31% | 865 | -2.12 | 0.0346 |
| spy_ret_60d_neg | -0.064% | 5/8 | 16% | 444 | 0.13 | 0.8980 |
| spy_weak_or_flat | -0.354% | 7/11 | 42% | 1152 | -2.10 | 0.0359 |
| not_strong_bull | +0.314% | 6/11 | 58% | 1596 | 1.42 | 0.1569 |

## 3. Confidence Thresholds

Only short when model confidence exceeds a threshold.

| Threshold | Avg Fwd Ret | Neg Years | Days | Avg Pool | t-stat | p-value |
|-----------|-------------|-----------|------|----------|--------|---------|
| >0.00 | +0.048% | 5/11 | 2766 | 444 | 0.54 | 0.5864 |
| >0.25 | +0.038% | 4/11 | 2766 | 32 | 0.43 | 0.6642 |
| >0.30 | -0.247% | 8/11 | 2670 | 5 | -1.50 | 0.1334 |
| >0.35 | -3.087% | 8/11 | 1659 | 2 | -6.42 | 0.0000 |
| >0.40 | -4.124% | 9/11 | 1049 | 1 | -8.19 | 0.0000 |
| >0.45 | -4.222% | 8/9 | 494 | 1 | -6.66 | 0.0000 |
| >0.50 | -2.421% | 6/7 | 340 | 1 | -8.81 | 0.0000 |

## 4. Combined Filters (Best Combinations)

| Combination | Avg Fwd Ret | Neg Years | Active % | t-stat | p-value |
|-------------|-------------|-----------|----------|--------|---------|
| top3_spy_weak_high_conf | -1.831% | 9/11 | 30% | -5.42 | 0.0000 |
| top1_not_strong_bull | -1.794% | 9/11 | 58% | -4.04 | 0.0001 |
| top5_spy_weak_high_conf | -1.570% | 10/11 | 30% | -4.85 | 0.0000 |
| top3_spy_weak | -0.911% | 9/11 | 31% | -3.01 | 0.0027 |
| top2_not_strong_bull | -0.547% | 7/11 | 58% | -2.40 | 0.0167 |
| top3_high_conf | -0.498% | 8/11 | 97% | -3.31 | 0.0009 |
| top3_not_strong_bull | -0.167% | 7/11 | 58% | -1.10 | 0.2710 |
| top3_nofilter | -0.159% | 6/11 | 100% | -1.55 | 0.1222 |
| top3_not_strong_bull_conf25 | -0.145% | 7/11 | 58% | -0.93 | 0.3532 |
| top5_not_strong_bull | +0.314% | 6/11 | 58% | 1.42 | 0.1569 |

## 5. Multi-Signal Confirmation

Require N deterioration signals to fire before shorting.

| Min Signals | Avg Fwd Ret | Neg Years | Days | Avg Pool | t-stat | p-value |
|-------------|-------------|-----------|------|----------|--------|---------|
| >=0 | +0.048% | 5/11 | 2766 | 444 | 0.54 | 0.5864 |
| >=2 | +0.071% | 5/11 | 2766 | 253 | 0.66 | 0.5092 |
| >=3 | +0.211% | 6/11 | 2766 | 150 | 1.68 | 0.0940 |
| >=4 | +0.396% | 4/11 | 2766 | 68 | 2.70 | 0.0069 |
| >=5 | +0.508% | 3/11 | 2766 | 27 | 3.35 | 0.0008 |

## 6. Best & Worst Short Candidates (by symbol)

Symbols that appear ≥20 times as short candidates:

### Best Shorts (actually decline)
| Symbol | Avg Fwd Ret | Count | % Negative |
|--------|-------------|-------|------------|
| NCLH | -6.709% | 130 | 66% |
| EOG | -5.254% | 67 | 70% |
| GNRC | -4.948% | 25 | 68% |
| ANET | -3.249% | 30 | 70% |
| COIN | -3.041% | 21 | 52% |
| FSLR | -3.019% | 105 | 64% |
| SMCI | -2.865% | 69 | 45% |
| MPC | -2.716% | 44 | 70% |
| CI | -2.550% | 26 | 69% |
| GM | -2.378% | 36 | 56% |
| EXPD | -2.349% | 23 | 65% |
| REGN | -1.926% | 41 | 63% |
| AXON | -1.858% | 24 | 38% |
| TDG | -1.745% | 21 | 67% |
| BBY | -1.694% | 49 | 63% |

### Worst Shorts (go UP when shorted)
| Symbol | Avg Fwd Ret | Count | % Negative |
|--------|-------------|-------|------------|
| AMD | +3.732% | 64 | 44% |
| MOS | +3.735% | 97 | 34% |
| XLE | +3.836% | 181 | 35% |
| EQT | +4.105% | 46 | 33% |
| OXY | +4.121% | 124 | 40% |
| IRM | +4.278% | 23 | 22% |
| WDC | +5.959% | 123 | 37% |
| MGM | +6.246% | 22 | 27% |
| RL | +8.745% | 20 | 40% |
| PLTR | +9.604% | 21 | 29% |

## 7. Recommendation

### Best single filter: `spy_ret_20d_neg`
- Avg fwd return: -0.492%
- Negative years: 9/11
- Active: 31% of trading days

### Best combined filter: `top3_spy_weak_high_conf`
- Avg fwd return: -1.831%
- Negative years: 9/11
- Active: 30% of trading days

**SHORT SIGNAL FOUND** with combined filter `top3_spy_weak_high_conf`!
Short candidates avg = -1.831% (NEGATIVE)
→ Proceed to build B4 long/short with this filter configuration.