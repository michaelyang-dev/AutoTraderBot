"""
DEEP ML return-ranking — the core 'can ML beat/augment momentum for selection'
question, done carefully: clean features (LEAK columns target/target_v5/target_rank
EXCLUDED), walk-forward, multiple feature sets + label variants + light tuning.
Measures OOS IC (cross-sectional rank corr of prediction vs forward return) and
crucially the ORTHOGONAL IC (ML residualized on momentum -> does ML add BEYOND
momentum?). Bar: momentum's own IC; ML must add orthogonal IC to be worth anything.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

LEAK = {"date", "symbol", "fwd_ret_15d", "target", "target_v5", "target_rank"}


def daily_ic(df, pcol, ycol="fwd_ret_15d"):
    ics = []
    for d, g in df.groupby("date"):
        if g[pcol].nunique() > 5 and len(g) > 20:
            ics.append(spearmanr(g[pcol], g[ycol]).statistic)
    return np.array(ics)


def resid_on_mom(df, pcol):
    """orthogonalize prediction vs momentum per date -> IC of the residual."""
    out = []
    for d, g in df.groupby("date"):
        if len(g) < 30 or g[pcol].nunique() < 5:
            continue
        x = g["ret_252d"].fillna(g["ret_252d"].median()).values
        p = g[pcol].values
        b = np.polyfit(x, p, 1)
        r = p - (b[0] * x + b[1])
        out.append(spearmanr(r, g["fwd_ret_15d"]).statistic)
    return np.array(out)


if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_parquet("data/ml_training_enhanced.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret_15d"]).sort_values("date")
    all_feats = [c for c in df.columns if c not in LEAK and df[c].dtype != "O"]
    price_feats = [c for c in all_feats if c.startswith(("ret_", "vol_")) or c in
                   ("rsi_14", "macd_line", "dist_sma50", "dist_sma200")]
    print(f"[{len(df)} rows, {len(all_feats)} clean feats (price subset {len(price_feats)})]")

    # momentum benchmark IC
    mom_ic = daily_ic(df.dropna(subset=["ret_252d"]).assign(p=lambda x: x["ret_252d"]), "p")
    print(f"\n=== benchmark: momentum (ret_252d) OOS IC ===")
    print(f"  mean {mom_ic.mean():+.4f}  t {mom_ic.mean()/(mom_ic.std()/np.sqrt(len(mom_ic))+1e-9):+.1f}")

    configs = [
        ("ALL feats, reg, lr.05 leaves31", all_feats, dict(num_leaves=31, learning_rate=0.05, n_estimators=300)),
        ("ALL feats, reg, lr.02 leaves15 deep", all_feats, dict(num_leaves=15, learning_rate=0.02, n_estimators=600, min_child_samples=100)),
        ("PRICE feats only", price_feats, dict(num_leaves=31, learning_rate=0.05, n_estimators=300)),
    ]
    for name, feats, hp in configs:
        preds = []
        for yr in range(2019, 2026):
            tr = df[df["date"] < f"{yr}-01-01"]; te = df[(df["date"] >= f"{yr}-01-01") & (df["date"] < f"{yr+1}-01-01")]
            if len(te) == 0 or len(tr) < 5000:
                continue
            m = lgb.LGBMRegressor(n_jobs=1, verbose=-1, subsample=0.8, colsample_bytree=0.8, **hp)
            m.fit(tr[feats], tr["fwd_ret_15d"])
            t = te[["date", "fwd_ret_15d", "ret_252d"]].copy(); t["p"] = m.predict(te[feats]); preds.append(t)
        allp = pd.concat(preds)
        ic = daily_ic(allp, "p"); oic = resid_on_mom(allp, "p")
        print(f"\n=== {name} ===")
        print(f"  raw OOS IC      mean {ic.mean():+.4f}  t {ic.mean()/(ic.std()/np.sqrt(len(ic))+1e-9):+.1f}")
        print(f"  ORTHOGONAL IC   mean {oic.mean():+.4f}  t {oic.mean()/(oic.std()/np.sqrt(len(oic))+1e-9):+.1f}  "
              f"(ML signal AFTER removing momentum -> this is what matters)")
    print(f"\n[total {time.time()-t0:.0f}s]")
