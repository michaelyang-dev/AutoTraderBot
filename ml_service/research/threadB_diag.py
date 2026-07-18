"""
THREAD B diagnostic — WHY does credit de-risk smash DD on 2001-25 but not 2018-25?
Attribution: for the incumbent MaxDD window in each period + each named crisis, report
the policy's average gross vs incumbent (did it de-risk INTO the drawdown?). Plus per-start
distribution (means can hide a split) and a deep-tail p95 "cheap insurance" variant.
Reuses the prepped runs from threadB_signal_leverage.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML); sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np, pandas as pd  # noqa: E402
import threadB_signal_leverage as B  # noqa: E402

CRISES = {"2008 GFC": ("2008-06-01", "2009-03-31"),
          "2020 COVID": ("2020-02-15", "2020-04-15"),
          "2022 bear": ("2022-01-01", "2022-10-31"),
          "2011 EU": ("2011-07-01", "2011-10-31")}


def maxdd_window(r):
    c = (1 + r).cumprod(); dd = (c - c.cummax()) / c.cummax()
    trough = dd.idxmin(); peak = c.loc[:trough].idxmax()
    return peak, trough, dd.min()


def main():
    signals = B.load_signals()
    for pname, path, starts, end in B.PERIODS:
        print("\n" + "=" * 100, flush=True)
        print(f"PERIOD {pname}", flush=True)
        print("=" * 100, flush=True)
        bt = B.FastBacktester(universe_path=path); B.clear_deployed(bt)
        runs = B.prep_runs(bt, starts, end, signals)

        # credit p80 override and p95 deep-tail
        def pol_p80(ro, vs, prc): return B.build_gross(ro, vs, prc["hy_oas"], "override", 0.80, 0.50)
        def pol_p95(ro, vs, prc): return B.build_gross(ro, vs, prc["hy_oas"], "override", 0.95, 0.50)
        def pol_baa(ro, vs, prc): return B.build_gross(ro, vs, prc["baa_aaa"], "override", 0.80, 0.50)

        for polname, fn in [("credit p80 x0.5", pol_p80), ("credit p95 x0.5", pol_p95),
                            ("baa-aaa p80 x0.5", pol_baa)]:
            print(f"\n--- {polname} ---", flush=True)
            print(f"{'start':<12}{'inc CAGR':>9}{'pol CAGR':>9}{'inc DD':>8}{'pol DD':>8}{'DDgain':>8}{'avgGr':>7}", flush=True)
            for i, (ro, vs, idx, prc) in enumerate(runs):
                iser, _ = B.apply_gross(ro, B.base_gross(ro, vs))
                pser, gav = B.apply_gross(ro, fn(ro, vs, prc))
                ic, _, _, idd = B.stats(iser); pc, _, _, pdd = B.stats(pser)
                print(f"{str(idx[0].date()):<12}{ic:>+9.1%}{pc:>+9.1%}{idd:>+8.1%}{pdd:>+8.1%}"
                      f"{(pdd-idd)*100:>+7.1f}p{gav:>7.2f}", flush=True)
            # crisis attribution on first start
            ro, vs, idx, prc = runs[0]
            base = B.base_gross(ro, vs); pg = fn(ro, vs, prc)
            iser, _ = B.apply_gross(ro, base); pser, _ = B.apply_gross(ro, pg)
            pk, tr, mdd = maxdd_window(iser)
            print(f"  incumbent MaxDD window {str(pk.date())}->{str(tr.date())} ({mdd:+.1%})", flush=True)
            print(f"  crisis-window avg gross (incumbent -> policy):", flush=True)
            bs = pd.Series(base, index=idx); ps = pd.Series(pg, index=idx)
            for cname, (a, z) in CRISES.items():
                m = (idx >= a) & (idx <= z)
                if m.sum() < 5:
                    continue
                # DD contribution of the crisis window in each
                icw = (1 + iser[m]).prod() - 1; pcw = (1 + pser[m]).prod() - 1
                print(f"    {cname:<12} gross {bs[m].mean():.2f}->{ps[m].mean():.2f} | "
                      f"windowRet inc {icw:+.1%} pol {pcw:+.1%}", flush=True)
        del bt


if __name__ == "__main__":
    main()
