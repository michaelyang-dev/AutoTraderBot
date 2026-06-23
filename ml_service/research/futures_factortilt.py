"""Novel angle: use the futures cross-asset MACRO STATE to time the equity factor mix
(momentum vs value), instead of fixed 50/35/15. Hypothesis: inflation/rising-rates/
strong-dollar regimes favor VALUE; disinflation favors MOMENTUM. Tested on 60yr of
Fama-French factors (monthly, non-overlapping, expanding z-scores = no lookahead).
Honest bar: factor timing usually fails OOS — must show real conditional spread AND a
conditioned tilt that beats fixed weighting."""
import numpy as np, pandas as pd
import futures_lib as fl

CCB, NON, mkts = fl.load()
ff = pd.read_parquet("../data/wrds/fama_french_5factors_momentum_daily.parquet")
ff["date"] = pd.to_datetime(ff["date"]); ff = ff.set_index("date")

COMMOD = ["CL","BRN","NG","HO","RB","GAS","HG","GC","SI","PA","PL","ZC","ZS","ZW","KE","SB","KC","CC","CT"]
BONDS  = ["ZN","ZB","ZF","TN","UB","FGBL","FGBM","FOAT","CGB"]
commod = [m for m in COMMOD if m in CCB.columns]; bonds = [m for m in BONDS if m in CCB.columns]

commod_tr = np.sign(CCB[commod].diff(120)).mean(axis=1)      # broad commodity trend (inflation)
rates_up  = -np.sign(CCB[bonds].diff(120)).mean(axis=1)       # bonds down => rates rising
dollar_tr = np.sign(CCB["DX"].diff(120))
sigs = {"commod_trend": commod_tr, "rates_rising": rates_up, "dollar_up": dollar_tr}

# monthly factor returns (non-overlapping)
fm = ff[["umd","hml","rmw","mktrf"]].resample("ME").sum()
nxt = fm.shift(-1)                                            # next-month factor return

def zexp(s):
    s = s.resample("ME").last()
    return (s - s.expanding(24).mean()) / s.expanding(24).std()

print("=== does futures macro state predict next-month UMD-HML (mom minus value)? ===")
print("    (60yr FF; high regime = inflation/rates/dollar UP -> expect VALUE to win, spread<0)")
spread = (nxt["umd"] - nxt["hml"])
for nm, s in sigs.items():
    sz = zexp(s).reindex(fm.index)
    d = pd.concat([sz.rename("z"), spread.rename("sp"), nxt["umd"].rename("u"), nxt["hml"].rename("h")], axis=1).dropna()
    ic = d["z"].corr(d["sp"], method="spearman")
    hi = d[d.z > 0.5]; lo = d[d.z < -0.5]
    print(f"  {nm:<13} IC {ic:+.3f} (n={len(d)}) | HIGH: umd {hi.u.mean():+.2%} hml {hi.h.mean():+.2%} (umd-hml {hi.sp.mean():+.2%}) "
          f"| LOW: umd {lo.u.mean():+.2%} hml {lo.h.mean():+.2%} ({lo.sp.mean():+.2%})")

# combined "value-favorable" regime score
vf = pd.concat([zexp(commod_tr), zexp(rates_up), zexp(dollar_tr)], axis=1).mean(axis=1).reindex(fm.index)
d = pd.concat([vf.rename("z"), spread.rename("sp")], axis=1).dropna()
print(f"\n  combined value-favorable score: IC(z, umd-hml) {d.z.corr(d.sp, method='spearman'):+.3f}")

# ---- does a CONDITIONED tilt beat fixed? momentum-heavy book vs regime-tilted ----
print("\n=== conditioned factor tilt vs fixed (next-month, OOS regime) ===")
al = pd.concat([vf.rename("vf"), nxt["umd"].rename("u"), nxt["hml"].rename("h"), nxt["rmw"].rename("q")], axis=1).dropna()
def stat(r, lbl):
    sh = r.mean()/r.std()*np.sqrt(12); cum=(1+r).prod()-1
    print(f"  {lbl:<28} ann.ret {r.mean()*12:+.2%}  Sharpe {sh:5.2f}  cum {cum:+.1%}")
stat(0.7*al.u + 0.3*al.h, "fixed 70/30 mom/val")
# tilt: when value-favorable (vf>0) shift toward value, else toward momentum
w_mom = np.clip(0.5 - 0.4*np.sign(al.vf), 0.1, 0.9)
stat(w_mom*al.u + (1-w_mom)*al.h, "regime-tilted mom/val")
stat(0.5*al.u+0.5*al.h, "fixed 50/50")
