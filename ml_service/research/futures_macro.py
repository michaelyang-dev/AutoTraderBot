"""Step 1 of the deeper dive: let the global futures panel reveal its own macro
structure. PCA the vol-normalized cross-asset return panel -> latent macro factors.
We expect ~PC1 global risk/beta, then duration (bonds vs stocks), commodity/inflation,
dollar. These become a principled REAL-TIME macro state to condition the equity book
(something a SP1500-only strategy is blind to). Saves PC scores for downstream use."""
import numpy as np, pandas as pd
import futures_lib as fl

# asset-class map (clear roots only; uncertain -> 'other' so it doesn't pollute)
ASSET = {}
for s in ["ES","NQ","YM","RTY","EMD","FDAX","FESX","FCE","FSMI","FTDX","HSI","NKD","NIY","SNK","SCN","KOS","SSG","SXF"]: ASSET[s]="EQ"
for s in ["ZT","ZF","ZN","TN","ZB","UB","ZQ","SR3","SO3","FGBS","FGBM","FGBL","FGBX","FOAT","FBTP","CGB","YIB","YIR","YXT","YYT","LLG","LFT","SJB"]: ASSET[s]="RATES"
for s in ["6A","6B","6C","6E","6J","6M","6N","6S","DX"]: ASSET[s]="FX"
for s in ["CL","BRN","NG","HO","RB","GAS","WBS","GC","SI","HG","PA","PL","ZC","ZS","ZW","KE","MWE","ZL","ZM","ZO","ZR","RS","SB","KC","CC","CT","OJ","LCC","LRC","LSU","LE","GF","HE","DC"]: ASSET[s]="COMMOD"
ASSET["VX"]="VOL"

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
vol = R.ewm(span=60, min_periods=20).std()
Z = (R / vol).clip(-6, 6)               # vol-normalized returns (each market ~unit risk)

Z = Z.loc["2000-01-01":]
cov = Z.notna().mean()
keep = [m for m in cov.index if cov[m] > 0.85]
Z = Z[keep].dropna(how="all").fillna(0.0)
Zs = (Z - Z.mean()) / Z.std()
print(f"[PCA on {Zs.shape[1]} markets x {Zs.shape[0]} days, 2000-2026]")

U, S, Vt = np.linalg.svd(Zs.values, full_matrices=False)
varexp = S**2 / (S**2).sum()
PCN = 6
load = pd.DataFrame(Vt[:PCN].T, index=keep, columns=[f"PC{i+1}" for i in range(PCN)])
scores = pd.DataFrame(U[:, :PCN] * S[:PCN], index=Zs.index, columns=[f"PC{i+1}" for i in range(PCN)])

print("\nvariance explained:", [f"PC{i+1} {varexp[i]:.0%}" for i in range(PCN)])
print("\n=== interpret each PC: net loading by asset class + top markets ===")
ac = pd.Series({m: ASSET.get(m, "other") for m in keep})
for pc in load.columns:
    nets = load[pc].groupby(ac).mean().sort_values()
    tilt = "  ".join(f"{k}{v:+.2f}" for k, v in nets.items())
    top = load[pc].sort_values()
    hi = ", ".join(f"{i}{top[i]:+.2f}" for i in top.tail(4).index[::-1])
    lo = ", ".join(f"{i}{top[i]:+.2f}" for i in top.head(4).index)
    print(f"\n{pc} ({varexp[load.columns.get_loc(pc)]:.0%}):  by class: {tilt}")
    print(f"    + {hi}")
    print(f"    - {lo}")

# sign-orient PC1 so that +PC1 = risk-ON (positive equity loading), for readability
if load.loc[[m for m in keep if ASSET.get(m)=='EQ'], 'PC1'].mean() < 0:
    scores['PC1'] *= -1; load['PC1'] *= -1
scores.to_parquet("_macro_pcs.parquet")
load.to_parquet("_macro_loadings.parquet")
print("\n[saved _macro_pcs.parquet, _macro_loadings.parquet]")
