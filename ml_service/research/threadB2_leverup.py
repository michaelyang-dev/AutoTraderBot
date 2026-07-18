"""
THREAD B2 — can we LEVER UP in confirmed-good regimes (not just de-risk in bad)?

The book de-risks in crises (credit gate) but never up-risks in booms — asymmetric. Test
whether a BULL signal (credit-calm = HY-OAS in the BOTTOM tail, and/or SPY uptrend) justifies
levering ABOVE 1.49x. Kelly says at the peak this adds variance without CAGR — but credit-calm
may carry forward-return asymmetry. Verify honestly with the matched-gross control: if timed
lever-up only raises AVG gross, a CONSTANT book at that same avg gross matches it -> no edge.

base gross = 1.49*min(1,vs); DOWN: *0.5 if hy pctile>=0.95; UP: *up if hy pctile<=calm (and/or
SPY>SMA200). Both periods, multi-start, financing 6.3% borrowed. Extends threadB_signal_leverage.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML); sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np, pandas as pd  # noqa: E402
import threadB_signal_leverage as B  # noqa: E402


def spy_trend(bt, idx):
    try:
        spy = bt.prices["SPY"]
        sma = spy.rolling(200).mean()
        return (spy.reindex(idx).ffill() > sma.reindex(idx).ffill()).values
    except Exception:
        return np.ones(len(idx), dtype=bool)


def main():
    signals = B.load_signals()
    POLICIES = []  # name -> fn(base_gross_array, pr_hy, trend)->gross

    def build(down=True, up_mult=1.0, calm=0.20, need_trend=False):
        def fn(base, pr, trend):
            g = base.copy()
            if down:
                g = np.where(pr.values >= 0.95, g * 0.5, g)
            if up_mult > 1.0:
                calm_flag = pr.values <= calm
                if need_trend:
                    calm_flag = calm_flag & trend
                g = np.where(calm_flag & (pr.values < 0.95), g * up_mult, g)
            return g
        return fn

    POLICIES = [
        ("baseline 1.49x (down-gate only)", build(down=True, up_mult=1.0)),
        ("credit-calm UP x1.15 (~1.7x)", build(down=True, up_mult=1.15)),
        ("credit-calm UP x1.30 (~1.9x)", build(down=True, up_mult=1.30)),
        ("credit-calm+trend UP x1.30", build(down=True, up_mult=1.30, need_trend=True)),
        ("credit-calm UP x1.15 NO down-gate", build(down=False, up_mult=1.15)),
    ]

    for pname, path, starts, end in B.PERIODS:
        print("\n" + "=" * 100, flush=True)
        print(f"{pname} | {len(starts)}-start | lever-UP test | matched-gross control", flush=True)
        print("=" * 100, flush=True)
        t0 = time.time()
        bt = B.FastBacktester(universe_path=path); B.clear_deployed(bt)
        runs = B.prep_runs(bt, starts, end, signals)
        trends = [spy_trend(bt, idx) for _, _, idx, _ in runs]
        print(f"(ran {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'policy':<36}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'avgGross':>9}{'|match dCAGR':>13}"
        print(hdr); print("-" * len(hdr), flush=True)
        for name, fn in POLICIES:
            cs, vs, ss, ds, gs, mdc = [], [], [], [], [], []
            for (ro, vsr, idx, prc), tr in zip(runs, trends):
                base = B.base_gross(ro, vsr)
                g = fn(base, prc["hy_oas"], tr)
                ser, gav = B.apply_gross(ro, g)
                c, v, s, d = B.stats(ser)
                mser, _ = B.apply_gross(ro, np.full(len(ro), gav))  # matched-gross constant
                mc, _, _, _ = B.stats(mser)
                cs.append(c); vs.append(v); ss.append(s); ds.append(d); gs.append(gav); mdc.append(c - mc)
            print(f"{name:<36}{np.mean(cs):>+8.1%}{np.mean(vs):>7.1%}{np.mean(ss):>8.2f}"
                  f"{np.mean(ds):>+8.1%}{np.mean(gs):>9.2f}{np.mean(mdc)*100:>+11.1f}pp", flush=True)
        del bt
    print("\nmatch dCAGR = timed policy CAGR - constant-leverage-at-same-avg-gross CAGR.", flush=True)
    print(">0 means the lever-UP timing beats just running more constant leverage (real edge).", flush=True)


if __name__ == "__main__":
    main()
