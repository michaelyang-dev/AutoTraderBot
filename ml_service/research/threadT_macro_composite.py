"""
THREAD T — does a MACRO-STRESS COMPOSITE de-risk beat credit-alone?

Credit de-risk catches CREDIT crises (2008). But 2022 was a RATES/duration crisis (credit
only mild); an orthogonal rates-vol / curve signal might catch it. Test whether combining
credit + rates-vol (MOVE proxy) + curve-inversion into an OR / blended de-risk beats the
credit-only gate, both periods, with the matched-gross control (does the extra de-risk add
TIMING, or just lower avg leverage?).

Signals (_macro_stress.parquet, 1962-2026): hy_oas (high=risk), rates_vol=21d realized vol of
DGS10 daily changes (high=risk), curve=T10Y2Y (LOW=risk, inversion). All causal expanding
percentile, act next day. Overlay on the 1x vol-scaled book; financing 6.3% borrowed.
Extends threadB_signal_leverage.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML); sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np, pandas as pd  # noqa: E402
import threadB_signal_leverage as B  # noqa: E402


def main():
    ms = pd.read_parquet(os.path.join(ML, "research", "_macro_stress.parquet"))
    sig = {"hy_oas": ms["hy_oas"].dropna(), "rates_vol": ms["rates_vol"].dropna(),
           "curve": ms["curve"].dropna()}

    CRISES = [("2008", "2008-06-01", "2009-03-31"), ("2020", "2020-02-15", "2020-04-15"),
              ("2022", "2022-01-01", "2022-10-31")]

    for pname, path, starts, end in B.PERIODS:
        print("\n" + "=" * 104, flush=True)
        print(f"{pname} | {len(starts)}-start | macro-stress composite vs credit-alone | matched-gross control", flush=True)
        print("=" * 104, flush=True)
        t0 = time.time()
        bt = B.FastBacktester(universe_path=path); B.clear_deployed(bt)
        # base runs (r_off, vol-scale path) — no signal coupling
        runs = []
        for st in starts:
            r_off = bt.run(st, end, dict(B.V12))["daily_values"].pct_change().dropna()
            bt._vs_log = []
            cfg = dict(B.V12); cfg.update(B.VS)
            bt.run(st, end, cfg)
            vs = pd.Series({d: v for d, v in bt._vs_log}); vs.index = pd.to_datetime(vs.index)
            vs = vs.reindex(r_off.index.union(vs.index)).ffill().reindex(r_off.index).fillna(1.0)
            runs.append((r_off, vs, r_off.index, None))
        # precompute causal percentiles per run/signal
        prc = []
        for ro, vsr, idx, _ in runs:
            prc.append({"hy": B.risk_pct(sig["hy_oas"], idx, True),
                        "rv": B.risk_pct(sig["rates_vol"], idx, True),
                        "cv": B.risk_pct(sig["curve"], idx, False)})  # low curve = risk
        print(f"(ran {time.time()-t0:.0f}s)", flush=True)

        def derisk(pr, kind):
            if kind == "credit":
                off = pr["hy"].values >= 0.95
            elif kind == "ratesvol":
                off = pr["rv"].values >= 0.95
            elif kind == "curve":
                off = pr["cv"].values >= 0.95
            elif kind == "OR_cr_rv":
                off = (pr["hy"].values >= 0.95) | (pr["rv"].values >= 0.95)
            elif kind == "OR_all":
                off = (pr["hy"].values >= 0.95) | (pr["rv"].values >= 0.95) | (pr["cv"].values >= 0.95)
            elif kind == "blend_avg":
                avg = (pr["hy"].values + pr["rv"].values + pr["cv"].values) / 3
                off = avg >= 0.85
            return off

        POL = ["baseline", "credit", "ratesvol", "curve", "OR_cr_rv", "OR_all", "blend_avg"]
        hdr = f"{'policy':<14}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'avgGross':>9}{'match dDD':>10}{'match dCAGR':>12}"
        print(hdr); print("-" * len(hdr), flush=True)
        base_stats = None
        for kind in POL:
            cs, vs, ss, ds, gs, mdd_d, mc_d = [], [], [], [], [], [], []
            for (ro, vsr, idx, _), pr in zip(runs, prc):
                base = B.base_gross(ro, vsr)
                if kind == "baseline":
                    g = base
                else:
                    g = np.where(derisk(pr, kind), base * 0.5, base)
                ser, gav = B.apply_gross(ro, g)
                c, v, s, d = B.stats(ser)
                mser, _ = B.apply_gross(ro, np.full(len(ro), gav))
                mc, _, _, md = B.stats(mser)
                cs.append(c); vs.append(v); ss.append(s); ds.append(d); gs.append(gav)
                mdd_d.append(d - md); mc_d.append(c - mc)
            row = (np.mean(cs), np.mean(vs), np.mean(ss), np.mean(ds), np.mean(gs))
            if kind == "baseline":
                base_stats = row
            tag = ""
            print(f"{kind:<14}{row[0]:>+8.1%}{row[1]:>7.1%}{row[2]:>8.2f}{row[3]:>+8.1%}{row[4]:>9.2f}"
                  f"{np.mean(mdd_d)*100:>+9.1f}p{np.mean(mc_d)*100:>+11.1f}p{tag}", flush=True)

        # crisis attribution: which policy de-risks which crisis (first start)
        ro, vsr, idx, _ = runs[0]; pr = prc[0]; base = B.base_gross(ro, vsr)
        print("  crisis-window avg gross (baseline -> credit -> OR_cr_rv):", flush=True)
        for cname, a, z in CRISES:
            m = (idx >= a) & (idx <= z)
            if m.sum() < 5:
                continue
            gb = base
            gc = np.where(derisk(pr, "credit"), base * 0.5, base)
            go = np.where(derisk(pr, "OR_cr_rv"), base * 0.5, base)
            print(f"    {cname}: {gb[m].mean():.2f} -> {gc[m].mean():.2f} -> {go[m].mean():.2f}", flush=True)
        del bt
    print("\nmatch dDD>0 = policy DD shallower than constant-leverage-at-same-avg-gross (timed).", flush=True)
    print("If OR_cr_rv beats credit on DD in BOTH periods AT similar CAGR -> composite wins.", flush=True)


if __name__ == "__main__":
    main()
