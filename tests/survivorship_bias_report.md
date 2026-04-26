# Survivorship Bias Diagnostic Report

Generated: 2026-04-26 01:09

## Executive Summary

**Survivorship bias status: MITIGATED**

The training pipeline correctly uses point-in-time S&P 500 membership and downloads data for historical members. Survivorship bias is adequately mitigated.

## Check 1: Massive Data Coverage for Delisted Tickers

| Ticker | Exit Year | Reason | Bars | First Date | Last Date | Expected End | Status |
|--------|-----------|--------|------|------------|-----------|--------------|--------|
| FRC | 2023 | Failed - FDIC takeover | 1763 | 2016-04-28 | 2023-04-28 | 2023-05-01 | FULL_COVERAGE |
| SIVB | 2023 | Failed - FDIC takeover | 1728 | 2016-04-28 | 2023-03-09 | 2023-03-10 | FULL_COVERAGE |
| SBNY | 2023 | Failed - FDIC takeover | 2172 | 2016-04-28 | 2024-12-31 | 2023-03-12 | PARTIAL_COVERAGE |
| CTXS | 2022 | Acquired by Vista Equity | 1618 | 2016-04-28 | 2022-09-29 | 2022-09-30 | FULL_COVERAGE |
| TWTR | 2022 | Acquired by Musk/X Corp | 1638 | 2016-04-28 | 2022-10-27 | 2022-10-27 | FULL_COVERAGE |
| XLNX | 2022 | Acquired by AMD | 1460 | 2016-04-28 | 2022-02-11 | 2022-02-14 | FULL_COVERAGE |
| ATVI | 2023 | Acquired by Microsoft | 1878 | 2016-04-28 | 2023-10-12 | 2023-10-13 | FULL_COVERAGE |
| DISCA | 2022 | Merged into WBD | 1499 | 2016-04-28 | 2022-04-08 | 2022-04-08 | FULL_COVERAGE |
| VIAC | 2022 | Renamed to PARA | 555 | 2019-12-05 | 2022-02-16 | 2022-02-15 | FULL_COVERAGE |
| TIF | 2021 | Acquired by LVMH | 1182 | 2016-04-28 | 2021-01-06 | 2021-01-07 | FULL_COVERAGE |
| XEC | 2021 | Acquired by Cabot Oil | 1367 | 2016-04-28 | 2021-09-30 | 2021-10-01 | FULL_COVERAGE |
| WLTW | 2022 | Merged into WTW | 1436 | 2016-04-28 | 2022-01-07 | 2022-09-22 | PARTIAL_COVERAGE |
| FB | 2022 | Renamed to META | 1540 | 2016-04-28 | 2022-06-08 | 2022-06-09 | FULL_COVERAGE |
| TWX | 2018 | Acquired by AT&T | 537 | 2016-04-28 | 2018-06-14 | 2018-06-14 | FULL_COVERAGE |
| SCG | 2019 | Acquired by Dominion | 674 | 2016-04-28 | 2018-12-31 | 2019-01-02 | FULL_COVERAGE |
| MNK | 2020 | Bankruptcy | 1330 | 2016-04-28 | 2023-08-25 | 2020-10-12 | PARTIAL_COVERAGE |
| SLM | 2014 | Spin-off Navient | 2184 | 2016-04-28 | 2024-12-31 | 2014-04-30 | PARTIAL_COVERAGE |
| KORS | 2018 | Renamed to CPRI | 674 | 2016-04-28 | 2018-12-31 | 2019-01-02 | FULL_COVERAGE |
| CELG | 2019 | Acquired by Bristol-Myers | 899 | 2016-04-28 | 2019-11-20 | 2019-11-22 | FULL_COVERAGE |
| RTN | 2020 | Merged into RTX | 990 | 2016-04-28 | 2020-04-02 | 2020-04-03 | FULL_COVERAGE |

**Summary:** 16/20 full coverage, 4/20 partial, 0/20 no coverage

## Check 2: Pipeline Audit

**Verdict: POINT-IN-TIME**

Training uses sp500_history.get_sp500_on_date() correctly and downloads historical members. Survivorship bias is mitigated.

| Check | Result |
|-------|--------|
| sp500_history.py exists | PASS |
| get_sp500_on_date() function defined | PASS |
| Uses reverse-application of changes (correct algorithm) | PASS |
| Starts from current list, removes future additions, adds back removals | PASS |
| data_pipeline.py imports get_sp500_on_date | PASS |
| data_pipeline.py computes per-date membership | PASS |
| data_pipeline.py sets in_sp500 column per date | PASS |
| train_production_model.py filters by in_sp500 | PASS |
| data_pipeline.py uses ALL_SYMBOLS for download | PASS |
| data_pipeline builds historical S&P 500 union | PASS |
| data_pipeline has ticker format normalization (BRK-B <-> BRK.B) | PASS |
| Current S&P 500 count: 503 | PASS |
| ALL_SYMBOLS count (downloaded): 747 | PASS |
| Delisted test tickers in download list: 14/20 | FAIL |
| Delisted tickers NOT in download list: ['DISCA', 'FB', 'KORS', 'SLM', 'VIAC', 'WLTW'] | FAIL |

## Check 3: Empirical Data Recovery

### Date: 2022-06-01

| Metric | Value |
|--------|-------|
| Point-in-time S&P 500 members | 504 |
| Members not in current S&P 500 | 67 |
| Full data recovered | 487 (96.6%) |
| Partial data | 9 |
| No data | 8 |
| **Recovery rate** | **96.6%** |

Tickers with no data: CPAY, DAY, EG, ELV, FBIN, MRSH, PSKY, RVTY

Of these, known historical/delisted: DAY, FBIN

### Date: 2018-06-01

| Metric | Value |
|--------|-------|
| Point-in-time S&P 500 members | 505 |
| Members not in current S&P 500 | 119 |
| Full data recovered | 471 (93.3%) |
| Partial data | 10 |
| No data | 24 |
| **Recovery rate** | **93.3%** |

Tickers with no data: BALL, BBWI, BKR, CPRI, CTRA, EG, ELV, FBIN, GL, HWM, J, LHX, LIN, LUMN, META, MRSH, PSKY, RTX, RVTY, SOLS, TFC, TT, VTRS, WBD

Of these, known historical/delisted: BBWI, CPRI, FBIN, LUMN, SOLS

## Recommendations

1. The current setup adequately addresses survivorship bias
2. Continue monitoring for edge cases (recent delistings, name changes)
3. Verify that Massive data coverage extends to all time periods used in training
