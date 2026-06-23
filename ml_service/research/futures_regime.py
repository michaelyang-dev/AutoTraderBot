"""Can cross-asset futures trends time the equity de-risk BETTER than the live
SPY<200d rule? Test an equity overlay on ES: buy-hold vs own-200d (mirrors live)
vs a multi-asset risk-on score (equity+copper trend up = risk-on; bonds+gold
rallying = risk-off). If the multi-asset version doesn't beat own-200d, the simple
rule is fine."""
import numpy as np, pandas as pd
import futures_lib as fl

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
es = R["ES"].dropna()
c = CCB["ES"]

def tsign(sym, L=120):
    return np.sign(CCB[sym].diff(L)) if sym in CCB.columns else 0

# baseline: own 200d trend -> 100% if up else 40% (mirrors SPY<200d -> 40%)
ma = c.rolling(200).mean()
expo_base = pd.Series(np.where(c > ma, 1.0, 0.4), index=c.index).shift(1)

# multi-asset risk-on score in [-4,+4]; higher = risk-on
score = tsign("ES") + tsign("HG") - tsign("ZN") - tsign("GC")
expo_multi = pd.Series(np.clip(0.4 + 0.10 * (score + 4), 0.4, 1.0), index=c.index).shift(1)

# combine: only de-risk when BOTH own-trend down AND multi-asset risk-off
expo_both = pd.Series(np.where((c > ma) | (score > 0), 1.0, 0.4), index=c.index).shift(1)

def line(r, label):
    r = r.dropna()
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    eqc = (1 + r).cumprod(); mdd = (eqc / eqc.cummax() - 1).min()
    def dd(a, b):
        s = (1 + r[(r.index >= a) & (r.index <= b)]).cumprod()
        return (s / s.cummax() - 1).min() if len(s) else np.nan
    print(f"  {label:<24} CAGR {cagr:+6.1%}  Sharpe {sh:5.2f}  MaxDD {mdd:6.1%}  "
          f"| 2008 {dd('2007-10-01','2009-03-31'):+.1%}  2020 {dd('2020-02-01','2020-04-30'):+.1%}  2022 {dd('2022-01-01','2022-12-31'):+.1%}")

print("=== Equity (ES) de-risk overlays, 1997-2026 ===")
line(es, "buy & hold")
line(expo_base * es, "own-200d (live rule)")
line(expo_multi * es, "multi-asset score")
line(expo_both * es, "own-200d OR risk-on")
print("\n  (if multi-asset doesn't beat own-200d on Sharpe & DD, the simple rule wins)")
