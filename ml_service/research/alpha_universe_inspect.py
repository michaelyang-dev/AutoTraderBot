"""Inspect broader-universe pickles for structure / coverage before backtesting."""
import os, sys, pickle
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = "data/wrds"
for fn in ["r2000_universe.pkl", "expanded_r3000_universe.pkl"]:
    path = os.path.join(DATA, fn)
    if not os.path.exists(path):
        print(f"{fn}: MISSING"); continue
    print(f"\n===== {fn} ({os.path.getsize(path)/1e6:.0f} MB) =====")
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        print("keys:", list(d.keys()))
        px = d.get("prices_df")
        if isinstance(px, pd.DataFrame):
            print(f"prices_df: {px.shape}  dates {px.index.min()}..{px.index.max()}  "
                  f"#tickers {px.shape[1]}")
        # membership
        for mk in ["membership", "membership_by_date", "sp500_mem", "members_by_date"]:
            if mk in d:
                m = d[mk]
                if isinstance(m, dict) and m:
                    k0 = sorted(m.keys())[len(m)//2]
                    print(f"{mk}: {len(m)} dates; sample {k0} -> {len(m[k0])} members")
        fbd = d.get("features_by_date")
        if isinstance(fbd, dict) and fbd:
            k0 = sorted(fbd.keys())[len(fbd)//2]
            sample_sym = next(iter(fbd[k0]))
            print(f"features_by_date: {len(fbd)} dates; sample feats: "
                  f"{sorted(fbd[k0][sample_sym].keys())[:12]}")
    else:
        print("type:", type(d))
    del d
