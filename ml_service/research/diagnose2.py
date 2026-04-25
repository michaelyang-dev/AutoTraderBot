import pandas as pd
import numpy as np

df = pd.read_parquet('data/features.parquet')

# Compare old data vs latest data for a single symbol (AAPL)
aapl = df[df.symbol == 'AAPL'].sort_values('date')
print(f'AAPL total rows: {len(aapl)}')
print(f'First date: {aapl.date.min()}')
print(f'Last date: {aapl.date.max()}')

# Get a row from 2020 (backtest period) and compare to latest
old_row = aapl[aapl.date == '2020-06-15'].iloc[0] if len(aapl[aapl.date == '2020-06-15']) > 0 else aapl.iloc[len(aapl) // 2]
new_row = aapl.iloc[-1]

feature_cols = [c for c in df.columns if c not in ['symbol', 'date', 'target', 'target_10d', 'fwd_ret_10d', 'fwd_ret']]

print('\nFeature comparison (AAPL):')
print(f'{"Feature":<20} {"Old (2020)":<15} {"Latest":<15} {"Diff":<15}')
print('-' * 65)
for col in feature_cols:
    old_val = old_row[col]
    new_val = new_row[col]
    diff = new_val - old_val if not (pd.isna(old_val) or pd.isna(new_val)) else 'NaN'
    print(f'{col:<20} {str(old_val)[:13]:<15} {str(new_val)[:13]:<15} {str(diff)[:13]:<15}')

# Check for NaN or zero values in recent data
print('\n\nFeatures with NaN in latest 60 rows:')
latest_60 = df.sort_values('date').groupby('symbol').tail(1)
for col in feature_cols:
    nan_count = latest_60[col].isna().sum()
    zero_count = (latest_60[col] == 0).sum()
    if nan_count > 0 or zero_count > 30:
        print(f'  {col}: {nan_count} NaN, {zero_count} zeros')
