import lightgbm as lgb
import pandas as pd
import numpy as np

model = lgb.Booster(model_file='data/model.lgb')
print(f'Model trees: {model.num_trees()}')
print(f'Model features: {model.num_feature()}')

df = pd.read_parquet('data/features.parquet')
print(f'Features shape: {df.shape}')
print(f'Date range: {df.date.min()} to {df.date.max()}')

latest = df.sort_values('date').groupby('symbol').tail(1)
print(f'Latest data symbols: {len(latest)}')

feature_cols = [c for c in df.columns if c not in ['symbol', 'date', 'target', 'target_10d', 'fwd_ret_10d', 'fwd_ret']]
print(f'Feature columns: {len(feature_cols)}')

X = latest[feature_cols].fillna(0)
preds = model.predict(X)

print(f'Prediction range: {preds.min():.4f} to {preds.max():.4f}')
print(f'Prediction mean: {preds.mean():.4f}')
print(f'Prediction std: {preds.std():.4f}')

latest_with_preds = latest.copy()
latest_with_preds['pred'] = preds
top5 = latest_with_preds.nlargest(5, 'pred')[['symbol', 'pred']]
print('Top 5:')
print(top5)
