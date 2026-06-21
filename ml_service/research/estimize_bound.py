"""
Is Estimize worth downloading? Answer WITHOUT it, via perfect-foresight on IBES.
Estimize's value = predict the EPS surprise better than analyst consensus -> trade
the announcement. Upper bound = PERFECT surprise prediction (the realized suescore).
Two tests on IBES (ibes_surprise: OFTIC, anndats, actual, surpmean, suescore) +
cached prices, SP1500 2016-2024:
  (1) PEAD: realized SUE -> forward 20/60d MARKET-RELATIVE drift (post-announcement,
      tradeable WITHOUT Estimize since the surprise is public after the print).
  (2) PERFECT-FORESIGHT pre-announcement: realized SUE -> announcement-window return
      (t-1->t+2). = max capturable if you predicted the surprise PERFECTLY. If this
      is small / weakly related to SUE, Estimize (imperfect) cannot help.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

cl = pd.read_parquet("data/cached_close_prices.parquet"); cl.index = pd.to_datetime(cl.index)
dr = cl.pct_change()
ai = list(cl.index); pos = {d: i for i, d in enumerate(ai)}
universe = set(cl.columns.astype(str))

ib = pd.read_parquet("data/wrds/ibes_surprise.parquet",
                     columns=["OFTIC", "anndats", "actual", "surpmean", "suescore"])
ib["anndats"] = pd.to_datetime(ib["anndats"], errors="coerce")
ib = ib.dropna(subset=["OFTIC", "anndats", "suescore"])
ib = ib[(ib["anndats"] >= "2016-01-01") & (ib["anndats"] <= "2024-09-30")]
ib = ib[ib["OFTIC"].isin(universe)]
ib = ib[np.isfinite(ib["suescore"])]
print(f"[{len(ib)} earnings announcements, {ib['OFTIC'].nunique()} tickers, 2016-2024]")

rows = []
for r in ib.itertuples():
    sym = r.OFTIC; d = r.anndats
    # align to first trading day >= announcement
    fut = [x for x in ai if x >= d]
    if not fut:
        continue
    i = pos[fut[0]]
    if i - 2 < 0 or i + 61 >= len(ai) or sym not in dr.columns:
        continue
    def relret(a, b):  # market-relative cum return over trading indices [a,b]
        s = (1 + dr[sym].iloc[a:b]).prod() - 1
        m = (1 + dr["SPY"].iloc[a:b]).prod() - 1
        return s - m
    jump = relret(i, i + 2)          # t-... actually i (first day>=ann) .. i+2 (announcement window)
    pead20 = relret(i + 2, i + 22)   # post-announcement drift, 20d
    pead60 = relret(i + 2, i + 62)   # post-announcement drift, 60d
    rows.append({"sue": r.suescore, "jump": jump, "pead20": pead20, "pead60": pead60})
df = pd.DataFrame(rows).dropna()
# winsorize sue
df["sue"] = df["sue"].clip(df["sue"].quantile(.01), df["sue"].quantile(.99))
df["q"] = pd.qcut(df["sue"].rank(method="first"), 5, labels=["Q1neg","Q2","Q3","Q4","Q5pos"])
print(f"[{len(df)} usable]\n")

print("=== (2) PERFECT-FORESIGHT pre-announcement: announcement-window return by realized SUE ===")
print("    (= max capturable if you predicted the surprise PERFECTLY; Estimize <= this)")
for qn, g in df.groupby("q", observed=True):
    print(f"  SUE {str(qn):<6} announcement-window ret {g['jump'].mean()*100:+.2f}%  n={len(g)}")
spread = df[df.q=="Q5pos"]["jump"].mean() - df[df.q=="Q1neg"]["jump"].mean()
print(f"  Q5-Q1 announcement spread: {spread*100:+.2f}%  | corr(SUE, jump) {spearmanr(df['sue'],df['jump']).statistic:+.3f}")
print(f"  -> if you predicted SUE perfectly you'd capture ~{spread*100:.1f}% per name; Estimize captures a FRACTION")

print("\n=== (1) PEAD: post-announcement drift by realized SUE (tradeable WITHOUT Estimize) ===")
for h in ["pead20", "pead60"]:
    sp = df[df.q=="Q5pos"][h].mean() - df[df.q=="Q1neg"][h].mean()
    ic = spearmanr(df["sue"], df[h]).statistic
    print(f"  {h}: Q5pos {df[df.q=='Q5pos'][h].mean()*100:+.2f}%  Q1neg {df[df.q=='Q1neg'][h].mean()*100:+.2f}%  "
          f"spread {sp*100:+.2f}%  IC {ic:+.3f}")
print("\n  Verdict logic: if perfect-foresight announcement spread is small AND PEAD is")
print("  weak/sub-cost, Estimize cannot add tradeable alpha -> don't bother downloading.")
