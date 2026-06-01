"""
Build Point-in-Time Enhanced Features for Honest Backtesting
==============================================================
Uses WRDS Compustat rdq (report date) and IBES announcement dates
to construct features that are available ONLY after they were released.

Builds:
1. fin_growth: {date: {ticker: {rev_growth, eps_growth}}} from Compustat YoY changes
2. revenue_surprise + beat_streak: {date: {ticker: value}} from IBES surprise
3. price_targets: {date: {ticker: {target}}} from IBES median consensus target

All keyed by date → only available after announcement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pickle
import time

def build_pit_fin_growth():
    """Build point-in-time revenue/EPS growth from Compustat quarterly.

    Uses rdq (report date) — the date the earnings were actually released.
    On each rdq date, compute YoY growth by comparing current quarter vs same quarter last year.
    """
    print("[1] Building point-in-time fin_growth from Compustat...")
    t0 = time.time()

    cq = pd.read_parquet("data/wrds/compustat_quarterly.parquet",
        columns=['tic', 'rdq', 'datadate', 'saleq', 'revtq', 'niq', 'seqq'])
    cq['rdq'] = pd.to_datetime(cq['rdq'])
    cq['datadate'] = pd.to_datetime(cq['datadate'])
    cq = cq.dropna(subset=['tic', 'rdq'])
    cq = cq.sort_values(['tic', 'datadate'])

    # For each ticker, compute YoY growth at each rdq date
    pit_fin_growth = {}  # {rdq_date: {ticker: {rev_growth, eps_growth}}}

    for tic, grp in cq.groupby('tic'):
        grp = grp.sort_values('datadate').drop_duplicates(subset=['datadate'], keep='last')
        if len(grp) < 5:
            continue

        for i in range(4, len(grp)):
            row = grp.iloc[i]
            prev = grp.iloc[i-4]  # same quarter last year
            rdq = row['rdq']

            if pd.isna(rdq):
                continue

            # Revenue growth YoY
            sale_now = row['saleq'] if pd.notna(row['saleq']) else row['revtq']
            sale_prev = prev['saleq'] if pd.notna(prev['saleq']) else prev['revtq']

            rev_growth = np.nan
            if sale_now and sale_prev and pd.notna(sale_now) and pd.notna(sale_prev) and abs(sale_prev) > 0:
                rev_growth = (sale_now - sale_prev) / abs(sale_prev)

            # EPS growth YoY (using net income / shares)
            eps_growth = np.nan
            ni_now = row['niq']
            ni_prev = prev['niq']
            if ni_now and ni_prev and pd.notna(ni_now) and pd.notna(ni_prev) and abs(ni_prev) > 0:
                eps_growth = (ni_now - ni_prev) / abs(ni_prev)

            if pd.isna(rev_growth) and pd.isna(eps_growth):
                continue

            rdq_key = pd.Timestamp(rdq).normalize()
            if rdq_key not in pit_fin_growth:
                pit_fin_growth[rdq_key] = {}

            pit_fin_growth[rdq_key][tic] = {
                'rev_growth': float(rev_growth) if pd.notna(rev_growth) else None,
                'eps_growth': float(eps_growth) if pd.notna(eps_growth) else None,
            }

    print(f"  Built {len(pit_fin_growth)} dates, {sum(len(v) for v in pit_fin_growth.values())} observations in {time.time()-t0:.1f}s")
    return pit_fin_growth


def build_pit_earnings():
    """Build point-in-time earnings surprise + beat streak from IBES.

    Uses anndats (announcement date) — when the surprise was revealed.
    """
    print("[2] Building point-in-time earnings surprise from IBES...")
    t0 = time.time()

    ib = pd.read_parquet("data/wrds/ibes_surprise.parquet",
        columns=['OFTIC', 'MEASURE', 'FISCALP', 'anndats', 'actual', 'surpmean', 'suescore'])
    ib['anndats'] = pd.to_datetime(ib['anndats'])

    # Filter to EPS quarterly surprises for US stocks
    eps = ib[(ib['MEASURE'] == 'EPS') & (ib['FISCALP'] == 'QTR')].copy()
    eps = eps.dropna(subset=['OFTIC', 'anndats', 'actual', 'surpmean'])
    eps = eps.sort_values(['OFTIC', 'anndats'])

    # Also get revenue surprises
    rev = ib[(ib['MEASURE'] == 'SAL') & (ib['FISCALP'] == 'QTR')].copy()
    rev = rev.dropna(subset=['OFTIC', 'anndats', 'actual', 'surpmean'])
    rev = rev.sort_values(['OFTIC', 'anndats'])

    # Build per-date surprise data
    pit_earnings = {}  # {date: {ticker: {eps_surprise, rev_surprise, sue_score}}}
    pit_beat_streak = {}  # {date: {ticker: streak_count}}
    pit_rev_surprise = {}  # {date: {ticker: surprise_pct}}

    # Track beat streaks per ticker
    ticker_streaks = {}  # ticker -> current consecutive beat count

    for _, row in eps.iterrows():
        tic = row['OFTIC']
        date = pd.Timestamp(row['anndats']).normalize()
        actual = row['actual']
        expected = row['surpmean']
        sue = row['suescore'] if pd.notna(row['suescore']) else 0

        if expected != 0:
            surprise_pct = (actual - expected) / abs(expected)
        else:
            surprise_pct = 0

        # Beat streak
        beat = actual > expected
        if tic not in ticker_streaks:
            ticker_streaks[tic] = 0
        if beat:
            ticker_streaks[tic] += 1
        else:
            ticker_streaks[tic] = 0

        if date not in pit_earnings:
            pit_earnings[date] = {}
        pit_earnings[date][tic] = {
            'eps_surprise': float(surprise_pct),
            'sue_score': float(sue),
            'beat_streak': int(ticker_streaks[tic]),
        }

        if date not in pit_beat_streak:
            pit_beat_streak[date] = {}
        pit_beat_streak[date][tic] = int(ticker_streaks[tic])

    # Revenue surprises
    for _, row in rev.iterrows():
        tic = row['OFTIC']
        date = pd.Timestamp(row['anndats']).normalize()
        actual = row['actual']
        expected = row['surpmean']

        if expected != 0:
            rev_surp = (actual - expected) / abs(expected)
        else:
            rev_surp = 0

        if date not in pit_rev_surprise:
            pit_rev_surprise[date] = {}
        pit_rev_surprise[date][tic] = float(rev_surp)

        # Also add to pit_earnings
        if date in pit_earnings and tic in pit_earnings[date]:
            pit_earnings[date][tic]['rev_surprise'] = float(rev_surp)

    print(f"  EPS surprises: {len(pit_earnings)} dates")
    print(f"  Rev surprises: {len(pit_rev_surprise)} dates")
    print(f"  Built in {time.time()-t0:.1f}s")

    return pit_earnings, pit_beat_streak, pit_rev_surprise


def build_pit_price_targets():
    """Build point-in-time consensus price targets from IBES.

    Uses ANNDATS — when the analyst issued the target.
    Computes median consensus target for each ticker, rolling 90-day window.
    """
    print("[3] Building point-in-time price targets from IBES...")
    t0 = time.time()

    pt = pd.read_parquet("data/wrds/ibes_price_targets.parquet",
        columns=['OFTIC', 'ANNDATS', 'VALUE', 'USFIRM'])
    pt['ANNDATS'] = pd.to_datetime(pt['ANNDATS'])
    pt = pt[(pt['USFIRM'] == 1) & (pt['VALUE'] > 0)].copy()
    pt = pt.dropna(subset=['OFTIC', 'ANNDATS', 'VALUE'])
    pt = pt.sort_values('ANNDATS')

    # For each month, compute median consensus from last 90 days of analyst targets
    pit_targets = {}  # {date: {ticker: {target: median_value}}}

    # Sample monthly to keep size manageable
    months = pd.date_range(pt['ANNDATS'].min(), pt['ANNDATS'].max(), freq='MS')

    for month_start in months:
        window_start = month_start - pd.Timedelta(days=90)
        window = pt[(pt['ANNDATS'] >= window_start) & (pt['ANNDATS'] < month_start)]

        if window.empty:
            continue

        # Median target per ticker
        medians = window.groupby('OFTIC')['VALUE'].median()

        date_key = pd.Timestamp(month_start).normalize()
        pit_targets[date_key] = {tic: {'target': float(val)} for tic, val in medians.items()}

    print(f"  Price targets: {len(pit_targets)} months, {sum(len(v) for v in pit_targets.values())} observations")
    print(f"  Built in {time.time()-t0:.1f}s")

    return pit_targets


def main():
    print("="*60)
    print("BUILDING POINT-IN-TIME ENHANCED FEATURES")
    print("="*60)

    pit_fin_growth = build_pit_fin_growth()
    pit_earnings, pit_beat_streak, pit_rev_surprise = build_pit_earnings()
    pit_targets = build_pit_price_targets()

    # Save
    output = {
        'pit_fin_growth': pit_fin_growth,
        'pit_earnings': pit_earnings,
        'pit_beat_streak': pit_beat_streak,
        'pit_rev_surprise': pit_rev_surprise,
        'pit_price_targets': pit_targets,
    }

    with open("data/wrds/pit_enhanced_features.pkl", "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved to data/wrds/pit_enhanced_features.pkl")
    print(f"Total size: {os.path.getsize('data/wrds/pit_enhanced_features.pkl') / 1e6:.1f} MB")

    # Quick summary
    print(f"\nSummary:")
    print(f"  fin_growth:     {len(pit_fin_growth)} dates")
    print(f"  earnings:       {len(pit_earnings)} dates")
    print(f"  beat_streak:    {len(pit_beat_streak)} dates")
    print(f"  rev_surprise:   {len(pit_rev_surprise)} dates")
    print(f"  price_targets:  {len(pit_targets)} months")


if __name__ == "__main__":
    main()
