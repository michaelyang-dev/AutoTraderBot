"""THREAD V5 — closing checks: (i) the SP500-only arm really is ~500 names (not a degenerate
empty pool that would fake the A/B); (ii) date-clustered significance of the NON-SP500 vs SP500
per-pick return difference; (iii) VIR-profile picks in context."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402

R = os.path.dirname(os.path.abspath(__file__))
bt = FastBacktester(universe_path="data/wrds/complete_sp1500_universe.pkl")
ds = [d for d in bt.prices.index if pd.Timestamp("2018-01-01") <= d <= pd.Timestamp("2025-12-31")][::100]
print("SP500-only arm pool sizes (bt._original_get_sp500):",
      [len(bt._original_get_sp500(d)) for d in ds])
print("SP1500 arm pool sizes (bt._get_sp1500):          ",
      [len(bt._get_sp1500(d)) for d in ds])
del bt

df = pd.read_parquet(os.path.join(R, "_v2_mom_picks.parquet"))
d = df[df.period == "2018-25"]
print("\nDate-clustered test, 2018-25: per-date mean(NON-SP500 picks) - mean(SP500 picks)")
for h in ["fwd20", "fwd60"]:
    rows = []
    for dt, g in d.groupby("date"):
        a = g.loc[g.tier != "SP500", h].dropna()
        b = g.loc[g.tier == "SP500", h].dropna()
        if len(a) and len(b):
            rows.append(a.mean() - b.mean())
    x = np.array(rows)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print(f"  {h}: n_dates_with_both={len(x)}  mean diff {x.mean():+.2%}  "
          f"median {np.median(x):+.2%}  t={t:+.2f}  "
          f"(dates where non-SP500 wins: {(x>0).mean():.1%})")

print("\nVIR-profile (rev growth<0 AND net margin<0) picks, 2018-25, by tier:")
b = d[d.broken == 1]
print(f"  n={len(b)} of {len(d)} picks ({len(b)/len(d):.1%});  tier mix: "
      + ", ".join(f"{t}={int((b.tier==t).sum())}" for t in ["SP500", "SP400", "SP600"]))
print(f"  fwd20 mean {b.fwd20.mean():+.2%} median {b.fwd20.median():+.2%} hit {(b.fwd20>0).mean():.1%}")
print(f"  fwd60 mean {b.fwd60.mean():+.2%} median {b.fwd60.median():+.2%} hit {(b.fwd60>0).mean():.1%}")
print(f"  worst fwd60: {b.fwd60.min():+.1%}   best fwd60: {b.fwd60.max():+.1%}")
print("  most frequent broken-profile tickers:", b.sym.value_counts().head(10).to_dict())
print("\nAll picks 2018-25, most frequent tickers:", d.sym.value_counts().head(12).to_dict())
print("unique tickers ever picked, 2018-25:", d.sym.nunique(), "over", d.date.nunique(), "dates")
