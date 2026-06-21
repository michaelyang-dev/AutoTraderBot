"""
Improvement pass 1 — cost/turnover optimization. Cost-sensitivity is a binding constraint; the
spot leg (Coinbase/Kraken) is the dear part, charged on every ROTATION. The funding ranking is
sticky, so we cut turnover with: (a) less-frequent rebalance, (b) HYSTERESIS — keep a held coin
until its funding really fades, only add a new one if it clearly out-yields. Less churn → higher NET.

Reports turnover and NET CAGR at a realistic 60bp round-trip (spot-dominated) for each.
Run:  python crypto/backtest/carry_optimize.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from improved_carry import load, stats


def carry(F, P, top_k=8, rebal=7, weighting="funding", rt_cost=0.0060, keep_mult=2.0, min_keep=0.0):
    """hysteresis: hold a coin while its funding rank stays within top_k*keep_mult; only add new
       coins (highest funding) to fill empty slots. min_keep: drop if trailing funding < this."""
    dprem = P.diff()
    sig = F.rolling(14).mean().shift(1) * 365
    pvol = dprem.rolling(30).std().shift(1)
    held = []; w = {}; daily = []; cost = []; turn_tot = 0.0
    for i, dt in enumerate(F.index):
        tc = 0.0
        if i % rebal == 0:
            s = sig.loc[dt].dropna()
            ranked = s.sort_values(ascending=False)
            keep_set = set(ranked.head(int(top_k * keep_mult)).index)
            # keep held coins still in the wider band and above min_keep; drop the rest
            new_held = [c for c in held if c in keep_set and s.get(c, -1) > min_keep]
            # fill remaining slots with highest-funding non-held positive coins
            for c in ranked.index:
                if len(new_held) >= top_k:
                    break
                if c not in new_held and s.get(c, -1) > max(min_keep, 0):
                    new_held.append(c)
            held = new_held
            if held:
                if weighting == "funding":
                    raw = sig.loc[dt, held].clip(lower=0)
                elif weighting == "invvol":
                    raw = 1.0 / pvol.loc[dt, held].replace(0, np.nan)
                else:
                    raw = pd.Series(1.0, index=held)
                raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                nw = (raw / raw.sum()).to_dict() if raw.sum() > 0 else {}
            else:
                nw = {}
            tn = sum(abs(nw.get(c, 0) - w.get(c, 0)) for c in set(nw) | set(w))
            tc = tn * rt_cost; turn_tot += tn
            w = nw
        r = 0.0
        for c, wt in w.items():
            f = F.loc[dt, c] if pd.notna(F.loc[dt, c]) else 0.0
            dp = dprem.loc[dt, c] if pd.notna(dprem.loc[dt, c]) else 0.0
            r += wt * (f - dp)
        daily.append(r); cost.append(tc)
    return pd.Series(daily, index=F.index) - pd.Series(cost, index=F.index), turn_tot


if __name__ == "__main__":
    F, P = load()
    yrs = (F.index.max() - F.index.min()).days / 365.25
    print("=" * 90)
    print("CARRY COST/TURNOVER OPTIMIZATION (clean data, top8 funding, 60bp round-trip)")
    print("=" * 90)
    print(f"  {'config':<40}{'turnover/yr':>13}{'NET CAGR':>11}{'2024+':>8}{'worst-mo':>10}")

    def show(label, d, turn):
        cg, vol, sh, md, wm = stats(d); cg24, _, _, _, _ = stats(d, "2024")
        print(f"  {label:<40}{turn/yrs:>11.1f}x{cg*100:>10.0f}%{cg24*100:>7.0f}%{wm*100:>9.1f}%", flush=True)

    print("  -- baseline (weekly, no hysteresis) --")
    d, t = carry(F, P, rebal=7, keep_mult=1.0); show("rebal 7d, hard top8", d, t)
    print("  -- less-frequent rebalance --")
    for rb in [14, 21, 30]:
        d, t = carry(F, P, rebal=rb, keep_mult=1.0); show(f"rebal {rb}d, hard top8", d, t)
    print("  -- + HYSTERESIS (keep until rank falls out of top16) --")
    for rb in [7, 14, 30]:
        d, t = carry(F, P, rebal=rb, keep_mult=2.0, min_keep=0.02); show(f"rebal {rb}d, hysteresis", d, t)
    print("\n  Goal: cut turnover so the 60bp spot cost stops eating the carry. Compare NET CAGR to the")
    print("  weekly-hard baseline (which churned and bled fees). Best config carries into the final book.")
