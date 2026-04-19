import lightgbm as lgb
import pandas as pd
import numpy as np

model = lgb.Booster(model_file='data/model.lgb')
df = pd.read_parquet('data/features.parquet')

# Sample 10 random dates from the last 6 months
feature_cols = [c for c in df.columns if c not in ['symbol', 'date', 'target', 'target_10d', 'fwd_ret_10d', 'fwd_ret']]

recent = df[df.date >= '2025-10-01'].copy()
recent['pred'] = model.predict(recent[feature_cols].fillna(0))

# Show max confidence per day
daily_max = recent.groupby('date').agg(
    max_conf=('pred', 'max'),
    n_above_55=('pred', lambda x: (x > 0.55).sum())
).tail(30)

print('Last 30 trading days — Max confidence per day:')
print(daily_max)
print()
print(f'Days with at least 1 signal above 0.55: {(daily_max.n_above_55 > 0).sum()}/30')
print(f'Average max confidence: {daily_max.max_conf.mean():.3f}')
