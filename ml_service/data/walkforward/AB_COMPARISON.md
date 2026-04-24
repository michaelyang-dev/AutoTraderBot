# Config A (Current Live) vs Config B (Proposed New) — Walk-Forward

Generated: 2026-04-23 22:04

## Methodology
Proper walk-forward: year Y uses model trained only on data <= Y-1.
Same universe, features, model. Only difference is strategy/portfolio construction.

## Configuration Details

| Setting | Config A (Current Live) | Config B (Proposed New) |
|---------|----------------------|----------------------|
| Slot config | ml:5, mom:3, max:8 | ml:5, mom:3, max:8 |
| Momentum regime filter | ON (SPY < 50-SMA blocks mom buys) | ON |
| Regime-aware sizing | OFF | ON (8/5/2) |
| SPY parking | OFF | OFF |
| top_n | 8 | 8 |

## Regime Definition (Config B)
- **BULLISH**: SPY > SMA50 AND SPY > SMA200 AND SMA50 > SMA200
- **BEARISH**: SPY < SMA200 AND SMA50 < SMA200
- **CAUTIOUS**: everything else

## Regime Distribution by Year

| Year | BULLISH | CAUTIOUS | BEARISH | % BULLISH |
|------|---------|----------|---------|-----------|
| 2015 | 111d | 105d | 36d | 44% |
| 2016 | 128d | 86d | 38d | 51% |
| 2017 | 236d | 15d | 0d | 94% |
| 2018 | 145d | 93d | 13d | 58% |
| 2019 | 153d | 74d | 25d | 61% |
| 2020 | 145d | 70d | 38d | 57% |
| 2021 | 232d | 20d | 0d | 92% |
| 2022 | 8d | 63d | 180d | 3% |
| 2023 | 166d | 76d | 8d | 66% |
| 2024 | 221d | 31d | 0d | 88% |
| 2025 | 148d | 84d | 17d | 59% |
| **Total** | **1693d** | **717d** | **355d** | **61%** |

## Per-Year Comparison

| Year | Regime | A CAGR | B CAGR | Delta | A Sharpe | B Sharpe | A MaxDD | B MaxDD |
|------|--------|--------|--------|-------|----------|----------|---------|---------|
| 2015 | B44/C42/R14 | +4.9% | -2.0% | -7.0% | 0.57 | -0.21 | -8.5% | -10.4% |
| 2016 | B51/C34/R15 | +25.3% | +14.5% | -10.8% | 2.40 | 1.52 | -7.0% | -4.8% |
| 2017 | B94/C6/R0 | +22.7% | +17.2% | -5.5% | 2.60 | 2.04 | -3.1% | -3.1% |
| 2018 | B58/C37/R5 | +4.1% | +6.5% | +2.4% | 0.47 | 0.72 | -13.5% | -11.9% |
| 2019 | B61/C29/R10 | +5.1% | -1.9% | -7.1% | 0.62 | -0.19 | -11.1% | -11.2% |
| 2020 | B57/C28/R15 | +25.1% | +9.8% | -15.3% | 1.56 | 0.73 | -25.2% | -22.1% |
| 2021 | B92/C8/R0 | +33.9% | +23.4% | -10.5% | 2.22 | 1.64 | -9.1% | -9.1% |
| 2022 | B3/C25/R72 | -24.6% | -13.4% | +11.2% | -3.06 | -1.80 | -24.6% | -15.0% |
| 2023 | B66/C30/R3 | +48.7% | +14.8% | -34.0% | 2.66 | 1.28 | -11.0% | -12.2% |
| 2024 | B88/C12/R0 | +3.7% | -0.8% | -4.5% | 0.32 | 0.02 | -16.2% | -16.3% |
| 2025 | B59/C34/R7 | +34.4% | +27.1% | -7.3% | 2.28 | 1.87 | -6.7% | -7.0% |

## Summary Metrics

| Metric | Config A | Config B | Delta |
|--------|----------|----------|-------|
| Median CAGR | +22.7% | +9.8% | -12.9% |
| Mean CAGR | +16.7% | +8.7% | -8.0% |
| Median Sharpe | 1.56 | 0.73 | -0.83 |
| Mean Sharpe | 1.15 | 0.69 | -0.46 |
| Worst Max DD | -25.2% | -22.1% | +3.1% |
| Mean Max DD | -12.4% | -11.2% | +1.2% |
| Median Alpha | +10.2% | +5.5% | -4.7% |
| Mean Beta | 0.33 | 0.31 | -0.03 |
| Positive CAGR years | 10/11 | 7/11 | -3 |
| Alpha-positive years | 8/11 | 7/11 | -1 |
| Mean trades/year | 101 | 72 | -29 |
| Mean win rate | 51.9% | 48.6% | -3.3% |

## Bear Market Focus (2018, 2022)

### 2018 (BEAR=13d, CAUT=93d)
- Config A: CAGR=+4.1%, Sharpe=0.47, DD=-13.5%
- Config B: CAGR=+6.5%, Sharpe=0.72, DD=-11.9%
- Delta CAGR: +2.4%, Delta Sharpe: +0.25

### 2022 (BEAR=180d, CAUT=63d)
- Config A: CAGR=-24.6%, Sharpe=-3.06, DD=-24.6%
- Config B: CAGR=-13.4%, Sharpe=-1.80, DD=-15.0%
- Delta CAGR: +11.2%, Delta Sharpe: +1.26

## Bull Year Regression Check
Did regime sizing hurt any strong bull years?

- 2016: A=+25.3% -> B=+14.5% (delta -10.8%) **REGRESSION**
- 2017: A=+22.7% -> B=+17.2% (delta -5.5%) **REGRESSION**
- 2019: A=+5.1% -> B=-1.9% (delta -7.1%) **REGRESSION**
- 2020: A=+25.1% -> B=+9.8% (delta -15.3%) **REGRESSION**
- 2021: A=+33.9% -> B=+23.4% (delta -10.5%) **REGRESSION**
- 2023: A=+48.7% -> B=+14.8% (delta -34.0%) **REGRESSION**
- 2024: A=+3.7% -> B=-0.8% (delta -4.5%) **REGRESSION**
- 2025: A=+34.4% -> B=+27.1% (delta -7.3%) **REGRESSION**

8 regressions (>2pp) in bull years.

## Regime Accuracy Check
- BULLISH: 1693d (61%)
- CAUTIOUS: 717d (26%)
- BEARISH: 355d (13%)

BEARISH triggered 12.8% — moderate impact.

## Verdict

- Median Sharpe improvement: -0.83 (threshold: >0.30)
- Median CAGR change: -12.9%
- Worst DD improved: -25.2% -> -22.1%
- Sharpe improved: 2/11 years
- CAGR improved: 2/11 years
- Regressions in bull years: 8

**DO NOT DEPLOY**: Config B does not clearly improve over Config A.