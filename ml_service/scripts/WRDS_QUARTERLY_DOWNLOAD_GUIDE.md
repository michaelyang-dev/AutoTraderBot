# WRDS Quarterly Download Guide

Do this every 3 months (Jan 1, Apr 1, Jul 1, Oct 1).
Takes ~15 minutes. 3 downloads, then 1 command to upload.

Login: https://wrds-www.wharton.upenn.edu/
Username: slymike

---

## Download 1: Compustat Short Interest

1. Go to: **Compustat North America → Supplemental Short Interest File**
2. Settings:
   - Date range: **2006-01 to present**
   - Tickers: **Leave blank** (all)
   - Variables: select **tic, datadate, shortint, shortintadj**
   - Output: **Parquet** (or CSV)
3. Submit query, download file
4. Rename to: `compustat_short_interest.parquet`

## Download 2: Compustat Fundamentals Quarterly

1. Go to: **Compustat North America → Fundamentals Quarterly**
2. Settings:
   - Date range: **2015-01 to present**
   - Tickers: **Leave blank** (all)
   - Variables: select these (search by name):
     - `tic` (ticker)
     - `datadate` (date)
     - `rdq` (report date — when earnings were announced)
     - `revtq` (revenue)
     - `niq` (net income)
     - `atq` (total assets)
     - `ltq` (total liabilities)
     - `ceqq` (common equity)
     - `oiadpq` (operating income)
     - `gpq` (gross profit)
     - `saleq` (sales/revenue)
     - `dlttq` (long-term debt)
     - `dlcq` (short-term debt)
     - `cheq` (cash)
     - `epspxq` (EPS excluding extraordinary)
     - `cshoq` (shares outstanding)
     - `oancfy` (operating cash flow)
   - Output: **Parquet**
3. Submit query, download file
4. Rename to: `compustat_fundamentals_quarterly.parquet`

## Download 3: IBES Earnings (Actuals + Summary)

### 3a: IBES Actuals
1. Go to: **IBES → Detail History → Actuals**
2. Settings:
   - Date range: **2015-01 to present**
   - Measure: **EPS**
   - Tickers: **Leave blank** (all US)
   - Variables: `TICKER, OFTIC, MEASURE, PESSION, ANNDATS, VALUE, CURR_ACT`
   - Output: **Parquet**
3. Rename to: `ibes_actuals_latest.parquet`

### 3b: IBES Summary (Consensus Estimates)
1. Go to: **IBES → Summary History**
2. Settings:
   - Date range: **2015-01 to present**
   - Measure: **EPS**
   - Forecast period: **1 (current quarter)**
   - Tickers: **Leave blank**
   - Variables: `TICKER, STATPERS, MEANEST, MEDEST, STDEV, NUMEST, ACTUAL, SURPMEAN, SURPSTDEV`
   - Output: **Parquet**
3. Rename to: `ibes_summary_latest.parquet`

---

## Upload to AWS

Put all 3 files in one folder, then run:

```bash
scp -i ~/Downloads/ExecTrade-key.pem \
  compustat_short_interest.parquet \
  compustat_fundamentals_quarterly.parquet \
  ibes_actuals_latest.parquet \
  ibes_summary_latest.parquet \
  ubuntu@54.158.238.15:/home/ubuntu/AutoTraderBot/ml_service/data/wrds/
```

Then SSH in and restart signal server:

```bash
ssh -i ~/Downloads/ExecTrade-key.pem ubuntu@54.158.238.15
pm2 restart signal-server
```

Done. Data is live.

---

## Schedule

| Month | Action |
|-------|--------|
| January 1 | Download all 3 + upload |
| April 1 | Download all 3 + upload |
| July 1 | Download all 3 + upload |
| October 1 | Download all 3 + upload |

Current data expires: ~July 2026 (uploaded Apr 15, 2026)
