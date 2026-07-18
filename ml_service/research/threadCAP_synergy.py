"""
CAPSTONE — cross-thread synergy: does layering the de-risk streams beat each alone, and is
the joint gain sub-additive (shared risk-off states) as expected?

Three complementary de-risk layers, each covering a DIFFERENT crisis type:
  - vol-scaling (incumbent, in the book)  -> equity-vol crises (2020 speed, 2022, Q4-18)
  - Thread B credit de-risk (leverage)    -> credit-led crises (2008)
  - Thread A micro futures (return stream) -> trend crises (2020/2022 convexity); no 2008 hedge

Joint config on the 26yr equity path (2001-25): base 1.49x vol-scaled, + credit p95 de-risk
(B), + micro realloc w=0.2 (A, micro=0 before its 2019 real existence -> conservative).
Reports full-period + per-crisis window so the coverage map is explicit. Sub-additivity
expected (B and A both cut exposure in overlapping risk-off states). Honest.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML); sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np, pandas as pd  # noqa: E402
import threadB_signal_leverage as B  # noqa: E402

MICRO = pd.read_parquet(os.path.join(ML, "research", "_threadA_micro50k.parquet"))["micro50k"]
MICRO.index = pd.to_datetime(MICRO.index)
CRISES = [("2008 GFC", "2008-06-01", "2009-03-31"), ("2020 COVID", "2020-02-15", "2020-04-15"),
          ("2022 bear", "2022-01-01", "2022-10-31"), ("2011 EU", "2011-07-01", "2011-10-31")]
W_MICRO = 0.20


def crisis_rets(r):
    out = {}
    for name, a, z in CRISES:
        m = (r.index >= a) & (r.index <= z)
        out[name] = ((1 + r[m]).prod() - 1) if m.sum() > 3 else None
    return out


def main():
    signals = B.load_signals()
    path = "data/wrds/sp1500_universe_2000.pkl"
    starts = ["2001-01-02", "2001-01-17"]; end = "2025-12-31"
    bt = B.FastBacktester(universe_path=path); B.clear_deployed(bt)
    runs = B.prep_runs(bt, starts, end, signals)

    def eval_variant(fn):
        cs, vs, ss, ds, cr = [], [], [], [], {c[0]: [] for c in CRISES}
        for ro, vs_, idx, prc in runs:
            r = fn(ro, vs_, idx, prc)
            c, v, s, d = B.stats(r)
            cs.append(c); vs.append(v); ss.append(s); ds.append(d)
            for k, val in crisis_rets(r).items():
                if val is not None:
                    cr[k].append(val)
        cw = {k: (np.mean(v) if v else None) for k, v in cr.items()}
        return np.mean(cs), np.mean(vs), np.mean(ss), np.mean(ds), cw

    def v_base(ro, vs, idx, prc):
        r, _ = B.apply_gross(ro, B.base_gross(ro, vs)); return r

    def v_credit(ro, vs, idx, prc):
        g = B.build_gross(ro, vs, prc["hy_oas"], "override", 0.95, 0.50)
        r, _ = B.apply_gross(ro, g); return r

    def add_micro(r, idx, w):
        mi = MICRO.reindex(idx).fillna(0.0)
        return (1 - w) * r + w * mi

    def v_micro(ro, vs, idx, prc):
        r, _ = B.apply_gross(ro, B.base_gross(ro, vs)); return add_micro(r, idx, W_MICRO)

    def v_joint(ro, vs, idx, prc):
        g = B.build_gross(ro, vs, prc["hy_oas"], "override", 0.95, 0.50)
        r, _ = B.apply_gross(ro, g); return add_micro(r, idx, W_MICRO)

    variants = [("BASE 1.49x vol-scaled", v_base),
                ("+B credit-p95 de-risk", v_credit),
                (f"+A micro realloc w{W_MICRO}", v_micro),
                ("+B+A JOINT", v_joint)]
    print("=" * 104, flush=True)
    print("CAPSTONE 26yr 2001-25 | de-risk layer stack | 2-start", flush=True)
    print("=" * 104, flush=True)
    hdr = f"{'variant':<26}{'CAGR':>7}{'Vol':>7}{'Shrp':>6}{'MaxDD':>8} | " + \
          "  ".join(c[0] for c in CRISES)
    print(hdr); print("-" * len(hdr), flush=True)
    base_dd = None
    for name, fn in variants:
        c, v, s, d, cw = eval_variant(fn)
        if base_dd is None:
            base_dd = d
        cwstr = "  ".join((f"{cw[k]*100:+5.0f}%" if cw[k] is not None else "  n/a") for k in [x[0] for x in CRISES])
        print(f"{name:<26}{c:>+7.1%}{v:>7.1%}{s:>6.2f}{d:>+8.1%} | {cwstr}", flush=True)
    print("\nRead: which layer protects which crisis. B cuts 2008; A cushions 2020/2022;", flush=True)
    print("neither helps 2011 much. Joint DD gain vs base is sub-additive (shared risk-off).", flush=True)


if __name__ == "__main__":
    main()
