"""
THREAD V4 — portfolio-level confirmation: momentum sleeve on SP1500 vs SP500-only,
run through the REAL FastBacktester (costs, 20d rebal, weights, stops) rather than my
synthetic compounding of pick-level forward returns.

Two configs per period, identical except universe:
  MOM-ONLY : 100% momentum sleeve, top-5, 20d rebal, cap 0.10, no RP, no vol-scale, no stop
             (bear_weights also 100% mom, so the regime blend cannot leak other sleeves in)
  V12      : the actual live blend 50/35/15, for reference
Also decomposes the realized momentum-sleeve return by index tier (weight x forward return
contribution) to answer 'are the mid/small picks the engine or the ballast'.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402

R = os.path.dirname(os.path.abspath(__file__))
CLEAN = "data/wrds/complete_sp1500_universe.pkl"


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


ALLMOM = {"mom": 1.0, "val": 0.0, "s5": 0.0, "s3": 0.0}
MOMONLY = dict(mom_w=1.0, val_w=0.0, lv_w=0.0, sec_w=0.0, top_n=5, rebal_days=20,
               cap=0.10, use_rp=False, bear_weights=ALLMOM, vol_scaling=False,
               trailing_stop=None)
V12 = dict(mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0, top_n=5, rebal_days=20,
           trailing_stop=0.40, cap=0.10, use_rp=False,
           bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
           vol_scaling=True, vol_target=0.15, vol_lookback=40)


def stats(m):
    v = m["daily_values"]
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1, dr.mean() / dr.std() * np.sqrt(252), \
        ((v - v.cummax()) / v.cummax()).min()


def main():
    for pname, path, start, end in [
        ("2018-25 (CLEAN pickle)", CLEAN, "2018-01-01", "2025-12-31"),
        ("2001-25 (26yr pickle, universe known inflated)",
         "data/wrds/sp1500_universe_2000.pkl", "2001-01-01", "2025-12-31"),
    ]:
        print("\n" + "=" * 92); print(pname); print("=" * 92, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time()-t0:.0f}s)", flush=True)
        print(f"{'config':<42}{'universe':<10}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}")
        print("-" * 79)
        for label, base in [("MOM-ONLY sleeve (100% s1, top5, r20)", MOMONLY),
                            ("V12 full blend 50/35/15", V12)]:
            for u in ["sp1500", "sp500"]:
                cfg = dict(base); cfg["universe"] = u
                m = bt.run(start, end, cfg)
                if m is None:
                    continue
                c, s, d = stats(m)
                print(f"{label:<42}{u:<10}{c:>+9.1%}{s:>9.2f}{d:>+9.1%}", flush=True)
        del bt

    # ---- tier decomposition of the momentum sleeve's realized return (2018-25, clean) ----
    print("\n" + "=" * 92)
    print("TIER DECOMPOSITION of momentum-sleeve realized return, 2018-25 (clean labels)")
    print("=" * 92)
    df = pd.read_parquet(os.path.join(R, "_v2_mom_picks.parquet"))
    d = df[df.period == "2018-25"].copy()
    for h in ["fwd20", "fwd60"]:
        d["contrib"] = d["weight"] * d[h]
        tot = d["contrib"].sum()
        print(f"\n  {h}: sum of weight x return across all {len(d)} picks = {tot:.3f} "
              f"(sleeve avg per period {tot/d['date'].nunique():+.2%})")
        for t in ["SP500", "SP400", "SP600"]:
            sub = d[d.tier == t]
            print(f"    {t:<7} weight share {sub['weight'].sum()/d['weight'].sum():>6.1%}  "
                  f"return contribution {sub['contrib'].sum()/tot:>7.1%}  "
                  f"(ratio {(sub['contrib'].sum()/tot)/(sub['weight'].sum()/d['weight'].sum()):>5.2f}x)")
        ns = d[d.tier != "SP500"]
        print(f"    NON-SP500 weight share {ns['weight'].sum()/d['weight'].sum():>6.1%}  "
              f"return contribution {ns['contrib'].sum()/tot:>7.1%}")


if __name__ == "__main__":
    main()
