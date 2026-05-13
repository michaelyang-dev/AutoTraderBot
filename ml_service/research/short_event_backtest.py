"""
Short Event Backtest: Test forced-selling event types from WRDS data.

For each event type, compute forward 30-day and 60-day returns after the event
to determine which events produce reliable post-event drops for shorting.

Events tested:
- Audit Analytics severity signals (going_concern, internal_controls, late_filing, etc.)
- Restatements (with fraud, errors, IC ineffectiveness)
- CRSP forced delistings (FING, INSC, DELQ, BKPY, LP)
- Dividend cuts/suspensions
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/michaelslyanggmail.com/Downloads/auto-trader 2/ml_service"

# ============================================================
# 1. Load CRSP price data (post-2000 only to keep memory sane)
# ============================================================
print("Loading CRSP daily prices (2000+)...")
prices = pd.read_parquet(
    f"{BASE}/data/wrds/crsp_daily_stock_full.parquet",
    columns=['PERMNO', 'Ticker', 'DlyCalDt', 'DlyPrc', 'DlyVol'],
    filters=[('DlyCalDt', '>=', '2000-01-01')]
)
prices['DlyCalDt'] = pd.to_datetime(prices['DlyCalDt'])
prices = prices.dropna(subset=['DlyPrc', 'Ticker'])
prices = prices[prices['DlyPrc'] > 0]
prices = prices.sort_values(['Ticker', 'DlyCalDt'])
print(f"  Loaded {len(prices):,} price rows, {prices['Ticker'].nunique():,} tickers")

# Build ticker->PERMNO mapping (use most recent PERMNO per ticker)
ticker_permno = prices.sort_values('DlyCalDt').drop_duplicates('Ticker', keep='last')[['Ticker', 'PERMNO']].set_index('Ticker')['PERMNO'].to_dict()
permno_ticker = {v: k for k, v in ticker_permno.items()}

# Build fast price index: ticker -> (sorted dates array, prices array)
print("Building price index...")
price_index = {}
for ticker, grp in prices.groupby('Ticker'):
    g = grp.sort_values('DlyCalDt')
    price_index[ticker] = (g['DlyCalDt'].values, g['DlyPrc'].values)
print(f"  Indexed {len(price_index):,} tickers")

# ============================================================
# Helper: compute forward returns for a set of (ticker, date) events
# ============================================================
def compute_forward_returns(events_df, ticker_col, date_col, min_price=5.0):
    """
    events_df: DataFrame with ticker_col and date_col
    Returns DataFrame with fwd_30d_ret, fwd_60d_ret columns
    """
    events = events_df[[ticker_col, date_col]].copy()
    events.columns = ['ticker', 'event_date']
    events['event_date'] = pd.to_datetime(events['event_date'])
    events = events.dropna()

    results = []
    # Group prices by ticker for fast lookup
    grouped = prices.groupby('Ticker')

    for _, row in events.iterrows():
        ticker = row['ticker']
        event_date = row['event_date']

        if ticker not in grouped.groups:
            continue

        tdf = grouped.get_group(ticker)
        # Find price on or after event date
        mask_after = tdf['DlyCalDt'] >= event_date
        if mask_after.sum() == 0:
            continue

        after = tdf[mask_after].sort_values('DlyCalDt')
        p0 = after.iloc[0]['DlyPrc']

        if p0 < min_price:
            continue

        # 30-day forward
        mask_30 = (tdf['DlyCalDt'] >= event_date + pd.Timedelta(days=25)) & \
                  (tdf['DlyCalDt'] <= event_date + pd.Timedelta(days=35))
        # 60-day forward
        mask_60 = (tdf['DlyCalDt'] >= event_date + pd.Timedelta(days=55)) & \
                  (tdf['DlyCalDt'] <= event_date + pd.Timedelta(days=65))

        p30 = tdf[mask_30]['DlyPrc'].iloc[0] if mask_30.sum() > 0 else np.nan
        p60 = tdf[mask_60]['DlyPrc'].iloc[0] if mask_60.sum() > 0 else np.nan

        ret_30 = (p30 / p0 - 1) if not np.isnan(p30) else np.nan
        ret_60 = (p60 / p0 - 1) if not np.isnan(p60) else np.nan

        results.append({
            'ticker': ticker,
            'event_date': event_date,
            'price_at_event': p0,
            'fwd_30d_ret': ret_30,
            'fwd_60d_ret': ret_60,
        })

    return pd.DataFrame(results)


def compute_forward_returns_fast(events_df, ticker_col, date_col, min_price=5.0):
    """
    Vectorized forward return computation using pre-built price index.
    """
    events = events_df[[ticker_col, date_col]].copy()
    events.columns = ['ticker', 'event_date']
    events['event_date'] = pd.to_datetime(events['event_date'], errors='coerce')
    events = events.dropna()
    events = events.drop_duplicates()

    empty = pd.DataFrame(columns=['ticker', 'event_date', 'price_at_event', 'fwd_30d_ret', 'fwd_60d_ret'])
    if len(events) == 0:
        return empty

    # Filter to tickers that exist in prices
    events = events[events['ticker'].isin(price_index)]
    if len(events) == 0:
        return empty

    results = []
    for ticker, grp in events.groupby('ticker'):
        dates_arr, prices_arr = price_index[ticker]
        n = len(dates_arr)

        for ed_ts in grp['event_date'].values:
            ed = ed_ts.astype('datetime64[ns]')

            # Find price at event (first trading day on or after, within 5 days)
            idx0 = np.searchsorted(dates_arr, ed, side='left')
            if idx0 >= n:
                continue
            if (dates_arr[idx0] - ed) > np.timedelta64(5, 'D'):
                continue
            p0 = prices_arr[idx0]
            if p0 < min_price:
                continue

            # Price at +30 days (nearest trading day)
            t30 = ed + np.timedelta64(30, 'D')
            idx30 = np.searchsorted(dates_arr, t30, side='left')
            if idx30 >= n:
                p30 = np.nan
            elif idx30 == 0:
                p30 = prices_arr[0] if abs((dates_arr[0] - t30)) <= np.timedelta64(5, 'D') else np.nan
            else:
                # Pick nearest of idx30 and idx30-1
                d1 = abs(dates_arr[idx30] - t30) if idx30 < n else np.timedelta64(999, 'D')
                d2 = abs(dates_arr[idx30-1] - t30)
                best_idx = idx30 if d1 <= d2 else idx30 - 1
                p30 = prices_arr[best_idx] if abs(dates_arr[best_idx] - t30) <= np.timedelta64(5, 'D') else np.nan

            # Price at +60 days
            t60 = ed + np.timedelta64(60, 'D')
            idx60 = np.searchsorted(dates_arr, t60, side='left')
            if idx60 >= n:
                p60 = np.nan
            elif idx60 == 0:
                p60 = prices_arr[0] if abs((dates_arr[0] - t60)) <= np.timedelta64(5, 'D') else np.nan
            else:
                d1 = abs(dates_arr[idx60] - t60) if idx60 < n else np.timedelta64(999, 'D')
                d2 = abs(dates_arr[idx60-1] - t60)
                best_idx = idx60 if d1 <= d2 else idx60 - 1
                p60 = prices_arr[best_idx] if abs(dates_arr[best_idx] - t60) <= np.timedelta64(5, 'D') else np.nan

            results.append({
                'ticker': ticker,
                'event_date': pd.Timestamp(ed),
                'price_at_event': p0,
                'fwd_30d_ret': (p30 / p0 - 1) if not np.isnan(p30) else np.nan,
                'fwd_60d_ret': (p60 / p0 - 1) if not np.isnan(p60) else np.nan,
            })

    return pd.DataFrame(results) if results else empty


def report_stats(name, results_df):
    """Print summary statistics for an event type."""
    if len(results_df) == 0:
        print(f"\n{'='*60}")
        print(f"  {name}: NO EVENTS WITH PRICE DATA")
        print(f"{'='*60}")
        return None

    r30 = results_df['fwd_30d_ret'].dropna()
    r60 = results_df['fwd_60d_ret'].dropna()

    n_events = len(results_df)
    avg_30 = r30.mean() * 100
    med_30 = r30.median() * 100
    avg_60 = r60.mean() * 100
    med_60 = r60.median() * 100
    pct_drop_5 = (r30 < -0.05).mean() * 100
    pct_drop_10 = (r30 < -0.10).mean() * 100
    win_rate_30 = (r30 < 0).mean() * 100  # shorting profitable if stock drops
    win_rate_60 = (r60 < 0).mean() * 100

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Events with price data:    {n_events:,}")
    print(f"  --- 30-Day Forward ---")
    print(f"  Avg return:                {avg_30:+.2f}%")
    print(f"  Median return:             {med_30:+.2f}%")
    print(f"  Short win rate:            {win_rate_30:.1f}%")
    print(f"  % dropped >5%:            {pct_drop_5:.1f}%")
    print(f"  % dropped >10%:           {pct_drop_10:.1f}%")
    print(f"  --- 60-Day Forward ---")
    print(f"  Avg return:                {avg_60:+.2f}%")
    print(f"  Median return:             {med_60:+.2f}%")
    print(f"  Short win rate:            {win_rate_60:.1f}%")

    return {
        'event_type': name,
        'n_events': n_events,
        'avg_30d_ret': avg_30,
        'med_30d_ret': med_30,
        'win_rate_30d': win_rate_30,
        'pct_drop_5pct_30d': pct_drop_5,
        'pct_drop_10pct_30d': pct_drop_10,
        'avg_60d_ret': avg_60,
        'med_60d_ret': med_60,
        'win_rate_60d': win_rate_60,
    }


# ============================================================
# 2. AUDIT ANALYTICS SEVERITY SIGNALS
# ============================================================
print("\n\nLoading Audit Analytics director/officer changes...")
severity_cols = [
    'late_filing_severity', 'ceo_change_severity', 'cfo_change_severity',
    'audit_fees_change_severity', 'audit_fees_outlier_severity',
    'non_audit_fees_severity', 'shareholder_action_lit_severity',
    'altman_score_severity', 'pledged_securities_severity',
    'shareholder_activism_severity', 'beneish_score_severity',
    'benfords_law_severity', 'going_concern_severity',
    'internal_controls_severity', 'auditor_change_severity',
    'material_impairment_severity', 'financial_restatement_severity',
    'regulatory_litigation_severity', 'illegal_activities_lit_severity',
    'disclosure_controls_severity',
]
audit_cols = ['best_edgar_ticker', 'opinion_file_date'] + severity_cols
audit = pd.read_parquet(f"{BASE}/data/wrds/audit_director_officer_changes.parquet", columns=audit_cols)
audit = audit[audit['best_edgar_ticker'].notna()].copy()
audit['opinion_file_date'] = pd.to_datetime(audit['opinion_file_date'], errors='coerce')
audit = audit[audit['opinion_file_date'] >= '2000-01-01']
print(f"  {len(audit):,} rows with tickers post-2000")

all_results = []

# Test each severity signal at HIGH severity (>=3) for strongest signal
# Also test at >=2 for broader coverage
for sev_col in severity_cols:
    nice_name = sev_col.replace('_severity', '').replace('_', ' ').title()

    for threshold, label in [(3, 'HIGH(>=3)'), (2, 'MED+(>=2)')]:
        mask = audit[sev_col] >= threshold
        events = audit[mask][['best_edgar_ticker', 'opinion_file_date']].copy()

        if len(events) < 20:
            continue

        # Sample if too many events (for speed)
        if len(events) > 5000:
            events = events.sample(5000, random_state=42)

        results = compute_forward_returns_fast(events, 'best_edgar_ticker', 'opinion_file_date')
        stats = report_stats(f"AUDIT: {nice_name} {label}", results)
        if stats:
            all_results.append(stats)

# ============================================================
# 3. RESTATEMENTS
# ============================================================
print("\n\nLoading Audit Analytics restatements...")
restate_cols = ['best_edgar_ticker', 'file_date', 'restatement', 'noteff_fin_fraud',
                'notefferrors', 'ic_is_effective', 'count_weak']
restate = pd.read_parquet(f"{BASE}/data/wrds/audit_restatements.parquet", columns=restate_cols)
restate = restate[restate['best_edgar_ticker'].notna()].copy()
restate['file_date'] = pd.to_datetime(restate['file_date'], errors='coerce')
restate = restate[restate['file_date'] >= '2000-01-01']
print(f"  {len(restate):,} rows with tickers post-2000")

# 3a. Any restatement
events = restate[restate['restatement'] == 1][['best_edgar_ticker', 'file_date']]
if len(events) > 5000:
    events = events.sample(5000, random_state=42)
results = compute_forward_returns_fast(events, 'best_edgar_ticker', 'file_date')
stats = report_stats("RESTATEMENT: Any Restatement", results)
if stats: all_results.append(stats)

# 3b. Restatement with fraud
events = restate[restate['noteff_fin_fraud'] == 3][['best_edgar_ticker', 'file_date']]
results = compute_forward_returns_fast(events, 'best_edgar_ticker', 'file_date')
stats = report_stats("RESTATEMENT: Financial Fraud", results)
if stats: all_results.append(stats)

# 3c. IC not effective
events = restate[restate['ic_is_effective'] == 'N'][['best_edgar_ticker', 'file_date']]
if len(events) > 5000:
    events = events.sample(5000, random_state=42)
results = compute_forward_returns_fast(events, 'best_edgar_ticker', 'file_date')
stats = report_stats("RESTATEMENT: IC Not Effective", results)
if stats: all_results.append(stats)

# 3d. High weakness count (>=3)
events = restate[restate['count_weak'] >= 3][['best_edgar_ticker', 'file_date']]
if len(events) > 5000:
    events = events.sample(5000, random_state=42)
results = compute_forward_returns_fast(events, 'best_edgar_ticker', 'file_date')
stats = report_stats("RESTATEMENT: High Weakness Count (>=3)", results)
if stats: all_results.append(stats)

# 3e. Restatement + IC not effective (double whammy)
events = restate[(restate['restatement'] == 1) & (restate['ic_is_effective'] == 'N')][['best_edgar_ticker', 'file_date']]
results = compute_forward_returns_fast(events, 'best_edgar_ticker', 'file_date')
stats = report_stats("RESTATEMENT: Restatement + IC Not Effective", results)
if stats: all_results.append(stats)

# ============================================================
# 4. CRSP FORCED DELISTINGS (pre-delisting returns)
# ============================================================
print("\n\nLoading CRSP delistings...")
delist = pd.read_parquet(f"{BASE}/data/wrds/crsp_delistings.parquet")
delist['DelistingDt'] = pd.to_datetime(delist['DelistingDt'], errors='coerce')
delist = delist[delist['DelistingDt'] >= '2000-01-01']

# Map PERMNO to ticker
delist['ticker'] = delist['PERMNO'].map(permno_ticker)
delist = delist[delist['ticker'].notna()]
print(f"  {len(delist):,} delistings with tickers post-2000")

forced_reasons = ['FING', 'INSC', 'DELQ', 'BKPY', 'LP']
for reason in forced_reasons:
    events = delist[delist['DelReasonType'] == reason][['ticker', 'DelistingDt']].copy()
    # For delistings, look at returns BEFORE the delisting (signal to short early)
    # Shift date back 60 days to simulate "catching the signal early"
    events['signal_date'] = events['DelistingDt'] - pd.Timedelta(days=60)
    if len(events) < 10:
        continue
    results = compute_forward_returns_fast(events, 'ticker', 'signal_date')
    stats = report_stats(f"DELIST: {reason} (60d before delist)", results)
    if stats: all_results.append(stats)

# Also test "any forced delisting"
events = delist[delist['DelReasonType'].isin(forced_reasons)][['ticker', 'DelistingDt']].copy()
events['signal_date'] = events['DelistingDt'] - pd.Timedelta(days=60)
results = compute_forward_returns_fast(events, 'ticker', 'signal_date')
stats = report_stats("DELIST: Any Forced (60d before)", results)
if stats: all_results.append(stats)

# ============================================================
# 5. DIVIDEND CUTS / SUSPENSIONS
# ============================================================
print("\n\nAnalyzing dividend cuts...")
div_cols = ['PERMNO', 'DisExDt', 'DisDivAmt', 'DisDetailType', 'DisFreqType']
divs = pd.read_parquet(f"{BASE}/data/wrds/crsp_dividends_distributions.parquet", columns=div_cols)
divs['DisExDt'] = pd.to_datetime(divs['DisExDt'], errors='coerce')

# Focus on ordinary cash dividends
divs = divs[divs['DisDetailType'] == 'CDIV']
divs = divs[divs['DisExDt'] >= '2000-01-01']
divs = divs.dropna(subset=['DisDivAmt'])
divs = divs.sort_values(['PERMNO', 'DisExDt'])

# Detect dividend cuts: compare each dividend to the previous one
divs['prev_div'] = divs.groupby('PERMNO')['DisDivAmt'].shift(1)
divs['prev_date'] = divs.groupby('PERMNO')['DisExDt'].shift(1)
divs = divs.dropna(subset=['prev_div'])

# Dividend cut >= 50%
divs['pct_change'] = (divs['DisDivAmt'] - divs['prev_div']) / divs['prev_div']
div_cuts_50 = divs[divs['pct_change'] <= -0.50].copy()
div_cuts_50['ticker'] = div_cuts_50['PERMNO'].map(permno_ticker)
div_cuts_50 = div_cuts_50[div_cuts_50['ticker'].notna()]

events = div_cuts_50[['ticker', 'DisExDt']]
if len(events) > 5000:
    events = events.sample(5000, random_state=42)
results = compute_forward_returns_fast(events, 'ticker', 'DisExDt')
stats = report_stats("DIVIDEND: Cut >= 50%", results)
if stats: all_results.append(stats)

# Dividend suspension (cut to 0)
div_suspend = divs[divs['DisDivAmt'] == 0].copy()
div_suspend['ticker'] = div_suspend['PERMNO'].map(permno_ticker)
div_suspend = div_suspend[div_suspend['ticker'].notna()]

events = div_suspend[['ticker', 'DisExDt']]
results = compute_forward_returns_fast(events, 'ticker', 'DisExDt')
stats = report_stats("DIVIDEND: Suspension (cut to $0)", results)
if stats: all_results.append(stats)


# ============================================================
# 6. SUMMARY TABLE
# ============================================================
print("\n\n" + "="*80)
print("  SUMMARY: ALL EVENT TYPES RANKED BY SHORT PROFITABILITY (30-day)")
print("="*80)

if all_results:
    summary = pd.DataFrame(all_results)
    summary = summary.sort_values('avg_30d_ret', ascending=True)  # Most negative = best for shorts

    print(f"\n{'Event Type':<50} {'N':>6} {'Avg30d':>8} {'Med30d':>8} {'Win%':>6} {'Drop5%':>7} {'Drop10%':>8} {'Avg60d':>8} {'Win60%':>7}")
    print("-" * 110)
    for _, row in summary.iterrows():
        print(f"{row['event_type']:<50} {row['n_events']:>6} {row['avg_30d_ret']:>+7.2f}% {row['med_30d_ret']:>+7.2f}% {row['win_rate_30d']:>5.1f}% {row['pct_drop_5pct_30d']:>6.1f}% {row['pct_drop_10pct_30d']:>7.1f}% {row['avg_60d_ret']:>+7.2f}% {row['win_rate_60d']:>6.1f}%")

    print("\n\nTOP CANDIDATES FOR SHORT STRATEGY (avg 30d return < -2% AND win rate > 55%):")
    print("-" * 80)
    top = summary[(summary['avg_30d_ret'] < -2.0) & (summary['win_rate_30d'] > 55)]
    if len(top) == 0:
        top = summary[(summary['avg_30d_ret'] < -1.0) & (summary['win_rate_30d'] > 52)]
        if len(top) == 0:
            top = summary.head(5)
            print("(Relaxed criteria - showing top 5 by avg 30d return)")

    for _, row in top.iterrows():
        print(f"  {row['event_type']}")
        print(f"    N={row['n_events']:,}  Avg30d={row['avg_30d_ret']:+.2f}%  Win={row['win_rate_30d']:.1f}%  Drop>5%={row['pct_drop_5pct_30d']:.1f}%  Avg60d={row['avg_60d_ret']:+.2f}%")

print("\nDone!")
