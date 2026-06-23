"""Does adding the futures book to the live v12 equity strategy help? Futures are
capital-efficient (margin), so the realistic add is an OVERLAY: keep 100% equity,
add w% futures notional on top. Test 2018-2025 (incl 2018Q4, 2020 COVID, 2022)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from main_production_backtest import FastBacktester

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40}

bt = FastBacktester()
m = bt.run("2018-01-01", "2025-12-31", V12)
eqr = m["daily_values"].pct_change().dropna()
print(f"v12 equity (1x): CAGR {m['cagr']:+.1%}  Sharpe {m['sharpe']:.2f}  MaxDD {m['max_dd']:.1%}")

book = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "_futures_book.parquet"))["book"]
df = pd.concat([eqr.rename("eq"), book.rename("fut")], axis=1).dropna()
print(f"overlap: {df.index.min().date()}..{df.index.max().date()} ({len(df)} days)")
print(f"corr(equity, futures book): {df['eq'].corr(df['fut']):+.2f}")

def dd(r, a, b):
    s = (1 + r[(r.index >= a) & (r.index <= b)]).cumprod()
    return (s / s.cummax() - 1).min() if len(s) else np.nan

def line(r, label):
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    print(f"  {label:<22} CAGR {cagr:+6.1%}  Sharpe {sh:5.2f}  MaxDD {mdd:6.1%}  "
          f"| 2018Q4 {dd(r,'2018-10-01','2018-12-31'):+.1%}  2020 {dd(r,'2020-02-01','2020-04-30'):+.1%}  2022 {dd(r,'2022-01-01','2022-12-31'):+.1%}")

print("\n=== OVERLAY: 100% equity + w% futures book (futures on margin) ===")
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    line(df["eq"] + w * df["fut"], f"equity + {w:.0%} futures")

print("\n=== REALLOCATION: (1-w) equity + w futures (move capital) ===")
for w in [0.0, 0.15, 0.25, 0.35]:
    line((1 - w) * df["eq"] + w * df["fut"], f"{1-w:.0%}eq / {w:.0%}fut")
