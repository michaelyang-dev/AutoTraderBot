"""
Insurance / catastrophe event study: do P&C insurers show a tradeable pattern
around major US hurricane landfalls (panic pre-landfall -> relief recovery after,
or persistent loss)? Event study of the cat-exposed P&C basket's CUMULATIVE
MARKET-RELATIVE return around landfall (t-10..t+20). If a consistent post-landfall
recovery exists -> tradeable; if losses persist or it's noise -> dead.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

# major US-landfalling hurricanes 2016-2024 (approx landfall date)
HURR = [
 ("Matthew", "2016-10-08"), ("Harvey", "2017-08-25"), ("Irma", "2017-09-10"),
 ("Florence", "2018-09-14"), ("Michael", "2018-10-10"), ("Dorian", "2019-09-06"),
 ("Laura", "2020-08-27"), ("Sally", "2020-09-16"), ("Delta", "2020-10-09"),
 ("Ida", "2021-08-29"), ("Ian", "2022-09-28"), ("Idalia", "2023-08-30"),
 ("Helene", "2024-09-26"), ("Milton", "2024-10-09"),
]
# cat-exposed P&C insurers + reinsurers (avoid PGR/auto-heavy)
INS = ["ALL", "TRV", "CB", "CINF", "HIG", "WRB", "AIG", "ACGL", "L"]
PRE, POST = 10, 20

cl = pd.read_parquet("data/cached_close_prices.parquet"); cl.index = pd.to_datetime(cl.index)
dr = cl.pct_change()
have = [t for t in INS if t in dr.columns]
basket = dr[have].mean(axis=1)            # equal-weight daily insurer return
rel = basket - dr["SPY"]                  # market-relative
idx = list(cl.index); pos = {d: i for i, d in enumerate(idx)}

paths, pre_ret, post_ret = [], [], []
print(f"[{len(have)} insurers: {have}]\n")
print(f"{'storm':<10} {'landfall':<12} {'pre(-10..0)':>12} {'post(0..+20)':>13}")
for nm, d in HURR:
    d = pd.Timestamp(d)
    fut = [x for x in idx if x >= d]
    if not fut:
        continue
    i = pos[fut[0]]
    if i - PRE < 0 or i + POST >= len(idx):
        continue
    seg = rel.iloc[i-PRE:i+POST+1].values
    cum = np.cumsum(seg)
    cum = cum - cum[PRE]                   # zero at landfall (t=0)
    paths.append(cum)
    pre = rel.iloc[i-PRE:i+1].sum()        # -10..0
    post = rel.iloc[i+1:i+POST+1].sum()    # 0..+20
    pre_ret.append(pre); post_ret.append(post)
    print(f"{nm:<10} {str(d.date()):<12} {pre*100:>11.1f}% {post*100:>12.1f}%")

P = np.array(paths)
print(f"\n=== AVERAGE market-relative cumulative path (n={len(P)} storms, t=0 at landfall) ===")
for off in [-10, -5, -2, 0, 2, 5, 10, 20]:
    j = off + PRE
    print(f"  t{off:+3d}: {P[:, j].mean()*100:+.2f}%")
print(f"\n  PRE-landfall (-10..0):  mean {np.mean(pre_ret)*100:+.2f}%  "
      f"hit<0 {(np.array(pre_ret)<0).mean()*100:.0f}%")
print(f"  POST-landfall (0..+20): mean {np.mean(post_ret)*100:+.2f}%  "
      f"hit>0 {(np.array(post_ret)>0).mean()*100:.0f}%  (positive+consistent => tradeable relief rally)")
t = np.mean(post_ret)/(np.std(post_ret)/np.sqrt(len(post_ret))+1e-9)
print(f"  POST t-stat {t:+.2f}")
