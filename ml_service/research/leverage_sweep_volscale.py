"""
JOINT LEVERAGE x VOL-SCALE SWEEP — "is 1.49x optimal WITH vol-scaling ON?"

The alpha-sprint leverage test predates the vol-scaling deployment; this closes the
gap: LIVE policy (t0.15, lookback 40, floor 0.30, cap 1.0, rebalance-day piecewise)
run through the canonical fork, then the SAME financing overlay as the canonical
numbers (6.3% on borrowed only) evaluated across a leverage grid. No margin-call
modeling — read deep MaxDD rows with that in mind (Reg-T maintenance ~25/30.7%:
equity share 1-1/L must stay above it — L 2.0 leaves ~15pp cushion at entry, less
in drawdowns).
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np

from final_live_config_test import (FastBacktester, LIVE_FLAGS, PERIODS, RATE,
                                    clear_deployed, lever, run_policy, stats)

GRID = [1.00, 1.25, 1.49, 1.60, 1.75, 2.00]


def main():
    for pname, path, starts, end in PERIODS:
        print("=" * 92, flush=True)
        print(f"PERIOD {pname} | LIVE vol-scale policy ON | {len(starts)}-start avg | fin {RATE:.1%} on borrowed", flush=True)
        print("=" * 92, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path)
        clear_deployed(bt)
        live = run_policy(bt, starts, end, LIVE_FLAGS)
        print(f"(fork runs {time.time()-t0:.0f}s)", flush=True)
        hdr = f"{'leverage':<12}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'avgGross':>9}{'fin/yr':>7}"
        print(hdr); print("-" * len(hdr), flush=True)
        for L in GRID:
            cs, vls, ss, ds, gs, fs = [], [], [], [], [], []
            for r, vs in live:
                rl, g, f = lever(r, vs, L, RATE if L > 1 else 0.0)
                c, v, s, d = stats(rl)
                cs.append(c); vls.append(v); ss.append(s); ds.append(d); gs.append(g); fs.append(f)
            tag = " <= LIVE" if abs(L - 1.49) < 1e-9 else ""
            print(f"{L:<12.2f}{np.mean(cs):>+8.1%}{np.mean(vls):>7.1%}{np.mean(ss):>8.2f}"
                  f"{np.mean(ds):>+8.1%}{np.mean(gs):>9.2f}{np.mean(fs):>7.2%}{tag}", flush=True)


if __name__ == "__main__":
    main()
