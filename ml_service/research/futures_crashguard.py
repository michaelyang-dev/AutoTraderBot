"""Does a CROSS-ASSET stress gauge improve the equity de-risk INCREMENTALLY over
SPY<200d? Key idea: the 200d MA is slow and equity-only; cross-asset stress (dollar
surge, bonds-beating-stocks flight-to-quality, VIX) is FASTER and partly ORTHOGONAL
to equity price, so it may catch acute crashes (2020) the MA misses. Test on ES
1997-2026: 200d alone vs stress alone vs 200d-OR-stress. Honest bar: must cut
drawdowns (esp acute) without killing return."""
import numpy as np, pandas as pd
import futures_lib as fl

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
es = R["ES"].dropna()
c = CCB["ES"]
ma200 = c.rolling(200).mean()
on_200 = c > ma200

def z(s, win=252):
    return (s - s.rolling(win, min_periods=60).mean()) / s.rolling(win, min_periods=60).std()

rsum = R.rolling(20).sum()
dollar = z(R["DX"].rolling(60).sum())               # dollar up = stress
ftq = z(rsum["ZN"] - rsum["ES"])                    # bonds beating stocks = stress
vx = z(CCB["VX"]) if "VX" in CCB.columns else None  # VIX level elevated = stress
comp = [dollar, ftq] + ([vx] if vx is not None else [])
stress = pd.concat(comp, axis=1).mean(axis=1).reindex(c.index)

def overlay(mask_on, lo=0.4):
    expo = pd.Series(np.where(mask_on, 1.0, lo), index=c.index).shift(1)
    return (expo * es).dropna()

def report(r, label):
    r = r.dropna(); yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    def dd(a, b):
        s = (1 + r[(r.index >= a) & (r.index <= b)]).cumprod()
        return (s / s.cummax() - 1).min() if len(s) else np.nan
    print(f"  {label:<26} CAGR {cagr:+6.1%}  Sharpe {sh:5.2f}  MaxDD {mdd:6.1%} | "
          f"2008 {dd('2007-10-01','2009-03-31'):+.1%}  2020 {dd('2020-02-01','2020-04-30'):+.1%}  "
          f"2022 {dd('2022-01-01','2022-12-31'):+.1%}  2018Q4 {dd('2018-10-01','2018-12-31'):+.1%}")

print("=== Equity de-risk: cross-asset stress vs SPY<200d (ES 1997-2026) ===")
report(es, "buy & hold")
report(overlay(on_200), "200d only (live rule)")
for thr in [0.75, 1.0, 1.5]:
    report(overlay(stress < thr), f"stress only (thr {thr})")
for thr in [0.75, 1.0, 1.5]:
    report(overlay(on_200 & (stress < thr)), f"200d OR stress (thr {thr})")

# does stress LEAD? avg days from de-risk trigger to equity trough in each crash
print("\n  (does stress fire earlier than 200d in acute drops?)")
for name, a, b in [("2020 COVID","2020-02-01","2020-04-30"), ("2018Q4","2018-09-15","2018-12-31")]:
    seg = c[(c.index>=a)&(c.index<=b)]
    if len(seg)<5: continue
    trough = seg.idxmin()
    s200 = on_200[(on_200.index>=a)&(on_200.index<trough)]
    sstr = (stress<1.0)[((stress.index>=a)&(stress.index<trough))]
    d200 = s200[~s200].index.min() if (~s200).any() else None
    dstr = sstr[~sstr].index.min() if (~sstr).any() else None
    print(f"  {name}: trough {trough.date()} | 200d de-risked {d200.date() if d200 is not None else 'never'} | "
          f"stress de-risked {dstr.date() if dstr is not None else 'never'}")
