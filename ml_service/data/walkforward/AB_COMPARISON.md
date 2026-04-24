# Config A (Current Live) vs Config B (Proposed New) — Walk-Forward

Generated: 2026-04-23 22:15

## Methodology
Proper walk-forward: year Y uses model trained only on data <= Y-1.
Same universe, features, model. Only difference is strategy/portfolio construction.

## Configuration Details

| Setting | Config A (Current Live) | Config B (Proposed New) |
|---------|----------------------|----------------------|
| Slot config | ml:5, mom:3, max:8 | ml:5, mom:3, max:8 |
| Momentum regime filter | ON (SPY < 50-SMA blocks mom buys) | ON |
| CAUTIOUS ML filter | ON (top 2 only) | ON (top 2 only) |
| ML top_n | 5 | 5 |
| Regime-aware sizing | OFF | ON (8/5/2) |
| SPY parking | ON (30% reserve, 20% threshold) | ON |

## Regime Definition (Live Production)
- **BULLISH**: SPY > SMA50 AND SPY > SMA200
- **CAUTIOUS**: SPY > SMA200 (but below SMA50)
- **BEARISH**: SPY <= SMA200

## Regime Distribution by Year

| Year | BULLISH | CAUTIOUS | BEARISH | % BULLISH |
|------|---------|----------|---------|-----------|
| 2015 | 142d | 56d | 54d | 56% |
| 2016 | 155d | 49d | 48d | 62% |
| 2017 | 236d | 15d | 0d | 94% |
| 2018 | 145d | 65d | 41d | 58% |
| 2019 | 185d | 40d | 27d | 73% |
| 2020 | 173d | 21d | 59d | 68% |
| 2021 | 232d | 20d | 0d | 92% |
| 2022 | 29d | 18d | 204d | 12% |
| 2023 | 174d | 60d | 16d | 70% |
| 2024 | 221d | 31d | 0d | 88% |
| 2025 | 180d | 27d | 42d | 72% |
| **Total** | **1872d** | **402d** | **491d** | **68%** |

## Per-Year Comparison

| Year | Regime | A CAGR | B CAGR | Delta | A Sharpe | B Sharpe | A MaxDD | B MaxDD |
|------|--------|--------|--------|-------|----------|----------|---------|---------|
| 2015 | B56/C22/R21 | +1.7% | -4.8% | -6.5% | 0.24 | -0.46 | -11.6% | -13.3% |
| 2016 | B62/C19/R19 | +24.1% | +15.9% | -8.2% | 2.22 | 1.49 | -6.1% | -6.7% |
| 2017 | B94/C6/R0 | +15.9% | +12.6% | -3.3% | 1.83 | 1.48 | -4.4% | -4.4% |
| 2018 | B58/C26/R16 | +0.2% | -4.4% | -4.6% | 0.07 | -0.34 | -15.3% | -19.0% |
| 2019 | B73/C16/R11 | -0.1% | -2.6% | -2.5% | 0.03 | -0.24 | -12.9% | -10.2% |
| 2020 | B68/C8/R23 | +20.1% | +19.4% | -0.7% | 1.13 | 1.03 | -29.9% | -22.5% |
| 2021 | B92/C8/R0 | +30.1% | +22.1% | -8.0% | 1.98 | 1.51 | -8.7% | -8.7% |
| 2022 | B12/C7/R81 | -28.7% | -15.3% | +13.4% | -3.05 | -1.11 | -29.3% | -16.6% |
| 2023 | B70/C24/R6 | +46.4% | +11.9% | -34.5% | 2.50 | 0.98 | -12.4% | -15.3% |
| 2024 | B88/C12/R0 | +3.9% | -3.1% | -7.1% | 0.33 | -0.13 | -16.6% | -18.3% |
| 2025 | B72/C11/R17 | +29.3% | +24.3% | -5.1% | 1.82 | 1.46 | -11.3% | -11.4% |

## Summary Metrics

| Metric | Config A | Config B | Delta |
|--------|----------|----------|-------|
| Median CAGR | +15.9% | +11.9% | -4.1% |
| Mean CAGR | +13.0% | +6.9% | -6.1% |
| Median Sharpe | 1.13 | 0.98 | -0.15 |
| Mean Sharpe | 0.83 | 0.52 | -0.31 |
| Worst Max DD | -29.9% | -22.5% | +7.4% |
| Mean Max DD | -14.4% | -13.3% | +1.1% |
| Median Alpha | +3.0% | -1.8% | -4.7% |
| Mean Beta | 0.42 | 0.50 | +0.07 |
| Positive CAGR years | 9/11 | 6/11 | -3 |
| Alpha-positive years | 8/11 | 5/11 | -3 |
| Mean trades/year | 97 | 70 | -27 |
| Mean win rate | 51.5% | 46.7% | -4.7% |

## Bear Market Focus (2018, 2022)

### 2018 (BEAR=41d, CAUT=65d)
- Config A: CAGR=+0.2%, Sharpe=0.07, DD=-15.3%
- Config B: CAGR=-4.4%, Sharpe=-0.34, DD=-19.0%
- Delta CAGR: -4.6%, Delta Sharpe: -0.41

### 2022 (BEAR=204d, CAUT=18d)
- Config A: CAGR=-28.7%, Sharpe=-3.05, DD=-29.3%
- Config B: CAGR=-15.3%, Sharpe=-1.11, DD=-16.6%
- Delta CAGR: +13.4%, Delta Sharpe: +1.94

## Bull Year Regression Check
Did regime sizing hurt any strong bull years?

- 2016: A=+24.1% -> B=+15.9% (delta -8.2%) **REGRESSION**
- 2017: A=+15.9% -> B=+12.6% (delta -3.3%) **REGRESSION**
- 2019: A=-0.1% -> B=-2.6% (delta -2.5%) **REGRESSION**
- 2020: A=+20.1% -> B=+19.4% (delta -0.7%)
- 2021: A=+30.1% -> B=+22.1% (delta -8.0%) **REGRESSION**
- 2023: A=+46.4% -> B=+11.9% (delta -34.5%) **REGRESSION**
- 2024: A=+3.9% -> B=-3.1% (delta -7.1%) **REGRESSION**
- 2025: A=+29.3% -> B=+24.3% (delta -5.1%) **REGRESSION**

7 regressions (>2pp) in bull years.

## Regime Accuracy Check
- BULLISH: 1872d (68%)
- CAUTIOUS: 402d (15%)
- BEARISH: 491d (18%)

BEARISH triggered 17.8% of the time — significant exposure reduction in downturns.

## Verdict

- Median Sharpe improvement: -0.15 (threshold: >0.30)
- Median CAGR change: -4.1%
- Worst DD improved: -29.9% -> -22.5%
- Sharpe improved: 1/11 years
- CAGR improved: 1/11 years
- Regressions in bull years: 7

**DO NOT DEPLOY**: Config B does not clearly improve over Config A.