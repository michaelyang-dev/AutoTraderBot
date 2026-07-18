"""
CLEAN COMPARISON vs baseline — p95 credit de-risk (Thread B) and micro-futures (Thread A),
SHORT (2018-25) and LONG (2001-25), both vs the live 1.49x vol-scaled baseline.

Baseline      = LIVE 1.49x vol-scaled (incumbent).
+B credit p95 = cut gross ~50% when HY-OAS in top 5% of expanding history (causal, next-day).
+A micro OVL  = baseline + w*micro book returns (margin overlay, CAGR-first).
+A micro RLC  = (1-w)*baseline + w*micro (reallocation, DD-first).
+B+A JOINT    = credit p95 gate AND micro realloc.
Micro book = the $50k capacity-realized micro-only futures book (research/_threadA_micro50k).
NOTE micros only exist since ~2019 -> in the LONG window the micro leg is counterfactual pre-2019
(reported, flagged); credit leg is real over the full window.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML); sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np, pandas as pd  # noqa: E402
import threadB_signal_leverage as B  # noqa: E402

MICRO = pd.read_parquet(os.path.join(ML, "research", "_threadA_micro50k.parquet"))["micro50k"]
MICRO.index = pd.to_datetime(MICRO.index)
W_OVL, W_RLC = 0.30, 0.20


def add_micro(r, w, mode):
    mi = MICRO.reindex(r.index).fillna(0.0)
    return r + w * mi if mode == "ovl" else (1 - w) * r + w * mi


def main():
    signals = B.load_signals()
    for pname, path, starts, end in B.PERIODS:
        print("\n" + "=" * 96, flush=True)
        tag = "SHORT" if pname.startswith("8yr") else "LONG"
        print(f"{tag}  {pname}  | {len(starts)}-start | 1.49x + fin 6.3% | vs LIVE baseline", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = B.FastBacktester(universe_path=path); B.clear_deployed(bt)
        runs = B.prep_runs(bt, starts, end, signals)
        print(f"(ran {time.time()-t0:.0f}s)\n", flush=True)

        def variant(fn):
            cs, vs, ss, ds = [], [], [], []
            for ro, vsr, idx, prc in runs:
                r = fn(ro, vsr, idx, prc)
                c, v, s, d = B.stats(r); cs.append(c); vs.append(v); ss.append(s); ds.append(d)
            return np.mean(cs), np.mean(vs), np.mean(ss), np.mean(ds)

        def base(ro, vsr, idx, prc):
            r, _ = B.apply_gross(ro, B.base_gross(ro, vsr)); return r

        def credit(ro, vsr, idx, prc):
            g = B.build_gross(ro, vsr, prc["hy_oas"], "override", 0.95, 0.50)
            r, _ = B.apply_gross(ro, g); return r

        def micro_ovl(ro, vsr, idx, prc):
            return add_micro(base(ro, vsr, idx, prc), W_OVL, "ovl")

        def micro_rlc(ro, vsr, idx, prc):
            return add_micro(base(ro, vsr, idx, prc), W_RLC, "rlc")

        def joint(ro, vsr, idx, prc):
            return add_micro(credit(ro, vsr, idx, prc), W_RLC, "rlc")

        rows = [("BASELINE 1.49x vol-scaled", base),
                ("+B credit p95 de-risk", credit),
                (f"+A micro OVERLAY w{W_OVL} (CAGR-first)", micro_ovl),
                (f"+A micro REALLOC w{W_RLC} (DD-first)", micro_rlc),
                ("+B+A JOINT (credit + micro rlc)", joint)]
        b = variant(base)
        hdr = f"{'variant':<38}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dMaxDD':>8}"
        print(hdr); print("-" * len(hdr), flush=True)
        for name, fn in rows:
            c, v, s, d = variant(fn)
            dc = "" if name.startswith("BASELINE") else f"{(c-b[0])*100:>+7.1f}p"
            dd = "" if name.startswith("BASELINE") else f"{(d-b[3])*100:>+7.1f}p"
            print(f"{name:<38}{c:>+8.1%}{v:>7.1%}{s:>8.2f}{d:>+8.1%}{dc:>8}{dd:>8}", flush=True)
        del bt
    print("\nMaxDD less negative = better. +B is near-free tail insurance; +A micro rlc trades", flush=True)
    print("CAGR for DD. (LONG micro leg counterfactual pre-2019; credit leg real full window.)", flush=True)


if __name__ == "__main__":
    main()
