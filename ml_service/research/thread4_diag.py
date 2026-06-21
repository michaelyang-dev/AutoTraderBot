"""
Thread #4 DIAGNOSTIC — why is crash-AUC so high, and why does the overlay fail?
Hypothesis: the model is a concurrent VOLATILITY detector (vol clustering), not an
ahead-of-time predictor, so it fires DURING/AFTER drawdowns -> useless for de-risk.
Checks: (1) feature importance (does vol dominate?), (2) does aggregate crash-prob
LEAD or LAG losses (corr with FUTURE vs PAST market return)?, (3) overlapping-window
caveat via a purged split.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
df = pd.read_parquet("data/ml_training_enhanced.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["fwd_ret_15d"]).sort_values("date")
feat = [c for c in df.columns if c not in ("date", "symbol", "fwd_ret_15d") and df[c].dtype != "O"]
df["crash"] = (df["fwd_ret_15d"] < -0.10).astype(int)

# train one fold (through 2022), test 2023, capture importances
tr = df[df["date"] < "2023-01-01"]; te = df[(df["date"] >= "2023-01-01") & (df["date"] < "2024-01-01")]
m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                       n_jobs=1, verbose=-1, subsample=0.8, colsample_bytree=0.8)
m.fit(tr[feat], tr["crash"])
imp = pd.Series(m.feature_importances_, index=feat).sort_values(ascending=False)
print("=== top-12 feature importances (crash classifier) ===")
for k, v in imp.head(12).items():
    print(f"  {k:<24} {v}")

# PURGED test: drop test rows within 15d of train end (kill window overlap)
te_purged = te[te["date"] >= "2023-01-25"]
p_all = m.predict_proba(te[feat])[:, 1]
p_pur = m.predict_proba(te_purged[feat])[:, 1]
print(f"\n  AUC 2023 raw    {roc_auc_score(te['crash'], p_all):.3f}")
print(f"  AUC 2023 purged {roc_auc_score(te_purged['crash'], p_pur):.3f}  (overlap removed)")

# lead/lag: aggregate crash-prob vs market (mean stock) returns
te2 = te.copy(); te2["p"] = p_all
cp = te2.groupby("date")["p"].mean()
mkt = te2.groupby("date")["ret_5d"].mean()  # proxy market 5d trailing return
both = pd.concat([cp.rename("cp"), mkt.rename("mkt5")], axis=1).dropna()
# correlation of crash-prob with PAST 5d return (concurrent/lagging signal)
print(f"\n  corr(crash_prob, trailing ret_5d) = {both['cp'].corr(both['mkt5']):+.3f}")
print("  (strongly negative => prob HIGH right after losses => concurrent/lagging, not leading)")
print(f"\n[total {time.time()-t0:.0f}s]")
