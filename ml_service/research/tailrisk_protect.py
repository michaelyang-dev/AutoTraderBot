"""
Does the SP1500 book NEED extra single-name protection? Tests the obvious candidates against
the deployed baseline, both periods, multi-start, in the live-mirror at live parity.

Candidates:
  - tighter single-name cap  (0.10 book = 15% NAV deployed -> 0.08 = 12% NAV, 0.06 = 9% NAV)
  - tighter trailing stop    (40% deployed -> 30% / 25%)
  - inverse-vol sizing inside the mom+val sleeves (use_rp=True)
  - explicit per-name vol ceiling on the momentum sleeve (mom_vol_max: skip VIR-like names)
  - Barroso vol-managed momentum sleeve scaling

Usage: python3 research/tailrisk_protect.py 8yr|26yr
"""
import os, sys, time, pickle
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _tailrisk_fork import TailRiskBacktester  # noqa: E402
from tailrisk_run import BASE, GATE, PERIODS, clear_deployed, risk_stats  # noqa: E402


def main():
    period = sys.argv[1] if len(sys.argv) > 1 else "8yr"
    path, starts, end = PERIODS[period]
    t0 = time.time()
    bt = TailRiskBacktester(universe_path=path); clear_deployed(bt)
    print(f"[{period}] loaded {time.time()-t0:.0f}s", flush=True)

    VARIANTS = [
        ("BASELINE deployed (cap 15%NAV, stop 40%)", {}),
        ("cap 0.08  (12% NAV)", dict(cap=0.08)),
        ("cap 0.06  ( 9% NAV)", dict(cap=0.06)),
        ("stop 30%", dict(trailing_stop=0.30)),
        ("stop 25%", dict(trailing_stop=0.25)),
        ("inverse-vol sizing (use_rp=True)", dict(use_rp=True)),
        ("mom vol ceiling 60% ann.", dict(mom_vol_max=0.60)),
        ("mom vol ceiling 45% ann.", dict(mom_vol_max=0.45)),
        ("mom vol ceiling 35% ann.", dict(mom_vol_max=0.35)),
        ("vol-managed mom sleeve (tgt 30%)", dict(vol_managed_mom=0.30)),
        ("cap 0.08 + mom vol ceiling 60%", dict(cap=0.08, mom_vol_max=0.60)),
        ("CONTINUOUS 15% NAV cap (daily trim)", dict(daily_trim=0.15)),
        ("CONTINUOUS 20% NAV cap (daily trim)", dict(daily_trim=0.20)),
        ("CONTINUOUS 25% NAV cap (daily trim)", dict(daily_trim=0.25)),
        ("CONTINUOUS 20% cap + stop 40% (both)", dict(daily_trim=0.20, trailing_stop=0.40)),
    ]
    hdr = (f"{'variant':<44}{'CAGR':>8}{'Vol':>7}{'Sharpe':>7}{'Sortino':>8}{'MaxDD':>8}"
           f"{'worst1d':>8}{'worst20d':>9}{'ulcer':>7}{'maxPos':>8}{'trims':>7}{'endNAV':>10}")
    print(hdr); print("-" * len(hdr), flush=True)
    out = {}
    for name, extra in VARIANTS:
        ps, mxw, ntrim = [], [], []
        for st in starts:
            cfg = dict(BASE); cfg.update(GATE); cfg.update(extra)
            m = bt.run(st, end, cfg)
            ps.append(risk_stats(m["daily_values"]))
            p = pd.DataFrame(bt._pos_daily, columns=["date", "sym", "value", "nav"])
            mxw.append((p["value"] / p["nav"]).max())
            ntrim.append(len(bt._trim_log))
        r = {k: float(np.mean([p[k] for p in ps])) for k in ps[0]}
        r["maxw"] = float(np.mean(mxw)); r["ntrim"] = float(np.mean(ntrim))
        out[name] = r
        print(f"{name:<44}{r['CAGR']:>+8.1%}{r['Vol']:>7.1%}{r['Sharpe']:>7.2f}{r['Sortino']:>8.2f}"
              f"{r['MaxDD']:>+8.1%}{r['worst1d']:>+8.1%}{r['worst20d']:>+9.1%}{r['ulcer']:>7.1%}"
              f"{r['maxw']:>8.1%}{r['ntrim']:>7.0f}{r['finalNAV']:>10,.0f}", flush=True)
    b = out["BASELINE deployed (cap 15%NAV, stop 40%)"]
    print("\ndeltas vs BASELINE:")
    for name, r in out.items():
        if name.startswith("BASELINE"):
            continue
        print(f"  {name:<44} CAGR {100*(r['CAGR']-b['CAGR']):>+6.2f}pp  Sharpe {r['Sharpe']-b['Sharpe']:>+5.2f} "
              f" MaxDD {100*(r['MaxDD']-b['MaxDD']):>+6.2f}pp  worst20d {100*(r['worst20d']-b['worst20d']):>+6.2f}pp",
              flush=True)


if __name__ == "__main__":
    main()
