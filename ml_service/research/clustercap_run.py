"""
Correlation-cluster cap experiment (v12, honest A/B).
=====================================================
Hypothesis: v12 caps single names at 15% but not clusters of correlated names
(live book ~49% NAV in one memory/semis theme). Prior research: book-crowding
predicts RISK not return -> use as risk cap. Expect MaxDD down, ~no CAGR cost.

Design:
  1. PARITY CHECK: run the UNPATCHED main_production_backtest with V12 config,
     then the fork with cluster_cap=None. Must match to machine precision.
  2. Variants: baseline / cluster_cap=0.40 / cluster_cap=0.30,
     starts 2018-01-02 and 2018-01-17, end 2025-12-31. ONLY difference between
     variants is the cluster_cap key. Same costs, same universe, same flags.
  3. Diagnostics: bind rate (rebalances with >=1 capped cluster), example
     clusters 2024-25 (expect the semis/memory theme).

V12 config = research/v12_ground_truth.py dict + production vol-scaling flags
(vol_scaling=True, vol_target=0.20, vol_lookback=40 — the VOLSC block used as
the "deployed bar" in research/alpha_crowding_bt.py; LIVE_SYSTEM.md: ON).

Run:  cd ml_service && OMP_NUM_THREADS=1 ./venv/bin/python research/clustercap_run.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import gc
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40,
       "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
       "vol_scaling": True, "vol_target": 0.20, "vol_lookback": 40}

STARTS = ["2018-01-02", "2018-01-17"]
END = "2025-12-31"
CAPS = [None, 0.40, 0.30]


def cfg(cluster_cap=None):
    c = dict(V12)
    if cluster_cap is not None:
        c["cluster_cap"] = cluster_cap
    return c


def fmt(m):
    return (f"CAGR {m['cagr']*100:6.2f}%  Sharpe {m['sharpe']:5.2f}  "
            f"MaxDD {m['max_dd']*100:6.2f}%  Vol {m['vol']*100:5.2f}%  "
            f"final ${m['final']:,.0f}")


def main():
    t0 = time.time()
    results = {}   # (cap, start) -> metrics
    diags = {}     # (cap, start) -> diagnostics

    # ── 1. Unpatched production backtest (parity reference) ────────────────
    print("=" * 78)
    print("STEP 1 — UNPATCHED production backtest (parity reference)")
    print("=" * 78, flush=True)
    from main_production_backtest import FastBacktester as OrigBT
    bt0 = OrigBT()
    ref = {}
    for st in STARTS:
        m = bt0.run(st, END, cfg())
        ref[st] = {k: m[k] for k in ("cagr", "sharpe", "max_dd", "vol", "final")}
        print(f"  UNPATCHED  start {st}:  {fmt(m)}", flush=True)
    del bt0
    gc.collect()

    # ── 2. Fork: baseline + cluster caps ────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 2 — FORK (research/clustercap_fork_backtest.py)")
    print("=" * 78, flush=True)
    from clustercap_fork_backtest import FastBacktester as ForkBT
    bt = ForkBT()

    for cc in CAPS:
        label = "baseline (no cluster cap)" if cc is None else f"cluster_cap={cc:.2f}"
        for st in STARTS:
            t1 = time.time()
            m = bt.run(st, END, cfg(cc))
            results[(cc, st)] = m
            ev = list(getattr(bt, "_cluster_log", []))
            n_reb = getattr(bt, "_cluster_rebals", 0)
            bind_dates = sorted({e["date"] for e in ev})
            diags[(cc, st)] = {"events": ev, "n_rebals": n_reb,
                               "n_bind_dates": len(bind_dates)}
            extra = ""
            if cc is not None and n_reb:
                extra = (f"  | cap bound on {len(bind_dates)}/{n_reb} rebalances "
                         f"({100*len(bind_dates)/n_reb:.0f}%)")
            print(f"  {label:<28} start {st}:  {fmt(m)}{extra}  "
                  f"[{time.time()-t1:.0f}s]", flush=True)

    # ── 3. Parity check ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 3 — PARITY CHECK (fork baseline vs unpatched)")
    print("=" * 78)
    parity_ok = True
    for st in STARTS:
        m = results[(None, st)]
        r = ref[st]
        diffs = {k: abs(m[k] - r[k]) for k in r}
        ok = all(v < 1e-9 for v in diffs.values())
        parity_ok &= ok
        print(f"  start {st}: {'EXACT MATCH' if ok else 'MISMATCH! ' + str(diffs)}")

    # ── 4. Summary table (2-start mean) ─────────────────────────────────────
    print("\n" + "=" * 78)
    print("STEP 4 — RESULTS (v12 1x, 2018->2025, costs 8+5bps, identical config"
          " except cluster_cap)")
    print("=" * 78)
    hdr = (f"{'variant':<26} {'start':<12} {'CAGR':>8} {'Sharpe':>7} "
           f"{'MaxDD':>8} {'Vol':>7}")
    print(hdr)
    print("-" * len(hdr))
    for cc in CAPS:
        label = "baseline" if cc is None else f"cluster_cap {cc:.2f}"
        cs, ss, ds = [], [], []
        for st in STARTS:
            m = results[(cc, st)]
            cs.append(m["cagr"]); ss.append(m["sharpe"]); ds.append(m["max_dd"])
            print(f"{label:<26} {st:<12} {m['cagr']*100:7.2f}% {m['sharpe']:7.2f} "
                  f"{m['max_dd']*100:7.2f}% {m['vol']*100:6.2f}%")
        print(f"{label:<26} {'MEAN':<12} {np.mean(cs)*100:7.2f}% {np.mean(ss):7.2f} "
              f"{np.mean(ds)*100:7.2f}%")
        print("-" * len(hdr))

    # ── 5. Diagnostics: example clusters ─────────────────────────────────────
    print("\nDIAGNOSTICS — clusters the cap found (per variant, start 2018-01-02)")
    for cc in [0.40, 0.30]:
        d = diags.get((cc, STARTS[0]))
        if not d:
            continue
        ev = [e for e in d["events"] if e["pass"] == 0]
        print(f"\n  cluster_cap={cc}: {d['n_bind_dates']}/{d['n_rebals']} rebalances bound, "
              f"{len(ev)} capped-cluster events")
        tot_unplaced = sum(e['unplaced'] for e in ev)
        if tot_unplaced > 1e-6:
            print(f"    NOTE: total unplaced (gross lost) across events: {tot_unplaced:.4f}")
        recent = [e for e in ev if e["date"].year >= 2024]
        print(f"    2024-25 events ({len(recent)}):")
        for e in recent:
            print(f"      {e['date'].date()}  w {e['weight_before']:.2f}->"
                  f"{e['weight_after']:.2f}  members: {', '.join(e['members'])}")
        # biggest clusters overall
        big = sorted(ev, key=lambda e: -e["weight_before"])[:5]
        print("    5 largest clusters ever capped:")
        for e in big:
            print(f"      {e['date'].date()}  w={e['weight_before']:.2f}  "
                  f"members: {', '.join(e['members'])}")

    # yearly detail for baseline vs 0.30 (start 1)
    print("\nYEARLY (start 2018-01-02): baseline vs cluster_cap 0.30")
    yb = results[(None, STARTS[0])]["yearly"]
    y3 = results[(0.30, STARTS[0])]["yearly"]
    for yr in sorted(yb):
        b, c = yb[yr], y3.get(yr, {})
        print(f"  {yr}: base {b['cagr']*100:+6.1f}% (DD {b['max_dd']*100:6.1f}%)   "
              f"cap30 {c.get('cagr', 0)*100:+6.1f}% (DD {c.get('max_dd', 0)*100:6.1f}%)")

    print(f"\nParity: {'OK' if parity_ok else 'FAILED'} | total {time.time()-t0:.0f}s")

    # machine-readable dump
    out = {"parity_ok": bool(parity_ok), "ref_unpatched": {
        st: {k: float(v) for k, v in ref[st].items()} for st in STARTS}}
    for cc in CAPS:
        key = "baseline" if cc is None else f"cap{cc}"
        out[key] = {}
        for st in STARTS:
            m = results[(cc, st)]
            d = diags[(cc, st)]
            out[key][st] = {
                "cagr": float(m["cagr"]), "sharpe": float(m["sharpe"]),
                "max_dd": float(m["max_dd"]), "vol": float(m["vol"]),
                "final": float(m["final"]),
                "bind_dates": d["n_bind_dates"], "n_rebals": d["n_rebals"],
            }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_clustercap_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"JSON: {path}")


if __name__ == "__main__":
    main()
