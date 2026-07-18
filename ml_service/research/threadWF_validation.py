"""
WALK-FORWARD OOS VALIDATION — honest out-of-sample CAGR for the equity book, and does the
credit p95 gate survive OOS?

The 28% (8yr) / 20% (26yr) baseline is IN-SAMPLE: the live config was selected on this
history. Real walk-forward for a RULES strategy = re-SELECT the config each year using only
PAST data, score the held-out NEXT year, roll. This measures the honest OOS haircut.

Method (efficient): run a grid of candidate configs ONCE each (1x book returns). Then:
  - each test year Y (after a burn-in), rank configs by Sharpe over ALL data < Y (expanding,
    causal), pick the best, take THAT config's returns for year Y. Stitch -> OOS curve.
  - compare to: fixed LIVE config full-period (in-sample), and ex-post ORACLE best (upper bound).
Everything at 1.49x flat + 6.3% financing so it's comparable and isolates config-selection OOS.
Then apply the FROZEN credit p95 gate (causal expanding-pctile) to the walk-forward book ->
OOS test of the gate itself. Grid: top_n x rebal_days x trailing_stop (27), weights fixed live.
"""
import os, sys, time, itertools
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML); sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np, pandas as pd  # noqa: E402
import threadB_signal_leverage as B  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "cap": 0.10, "use_rp": False, "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
GRID_TOPN = [3, 5, 8]
GRID_REBAL = [15, 20, 30]
GRID_STOP = [0.30, 0.40, 0.55]
LIVE = (5, 20, 0.40)   # the deployed config point
PERIODS = [
    ("SHORT 2018-25", "data/wrds/complete_sp1500_universe.pkl", "2018-01-02", "2025-12-31", 2),
    ("LONG 2001-25", "data/wrds/sp1500_universe_2000.pkl", "2001-01-02", "2025-12-31", 3),
]
BASE_L, RATE = 1.49, 0.063


def stats(r):
    r = r.dropna(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    c = (1 + r).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min())


def lever_flat(r1x, L=BASE_L, rate=RATE):
    return L * r1x - max(0.0, L - 1.0) * (rate / 252)


def sharpe(r):
    r = r.dropna()
    return (r.mean() / r.std() * np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else -9


def main():
    signals = B.load_signals()
    configs = list(itertools.product(GRID_TOPN, GRID_REBAL, GRID_STOP))
    for pname, path, start, end, burnin in PERIODS:
        print("\n" + "=" * 98, flush=True)
        print(f"{pname} | walk-forward re-selection over {len(configs)} configs | {burnin}y burn-in | 1.49x flat", flush=True)
        print("=" * 98, flush=True)
        t0 = time.time()
        bt = B.FastBacktester(universe_path=path); B.clear_deployed(bt)
        # run each grid config once -> 1x daily returns
        rets = {}
        for (tn, rb, st) in configs:
            cfg = dict(BASE); cfg.update({"top_n": tn, "rebal_days": rb, "trailing_stop": st})
            rets[(tn, rb, st)] = bt.run(start, end, cfg)["daily_values"].pct_change().dropna()
        print(f"(ran {len(configs)} configs {time.time()-t0:.0f}s)", flush=True)

        idx = rets[LIVE].index
        years = sorted({d.year for d in idx})
        oos_years = years[burnin:]
        # walk-forward selection
        picks = {}
        oos_parts = []
        for Y in oos_years:
            train_mask = np.array([d.year < Y for d in idx])
            best, best_s = None, -1e9
            for k, r in rets.items():
                s = sharpe(r[train_mask])
                if s > best_s:
                    best_s, best = s, k
            picks[Y] = best
            r_test = rets[best][np.array([d.year == Y for d in idx])]
            oos_parts.append(r_test)
        wf_1x = pd.concat(oos_parts).sort_index()
        oos_idx = wf_1x.index

        # references, all restricted to the OOS window for apples-to-apples
        live_1x = rets[LIVE].reindex(oos_idx)
        # oracle: single best full-period config (ex-post) over OOS window
        oracle_k = max(rets, key=lambda k: sharpe(rets[k].reindex(oos_idx)))
        oracle_1x = rets[oracle_k].reindex(oos_idx)

        def lev(r):
            return stats(lever_flat(r))

        # credit p95 gate on the walk-forward book (frozen rule, causal expanding pctile)
        vs1 = pd.Series(1.0, index=oos_idx)
        pr = B.risk_pct(signals["hy_oas"], oos_idx, True)
        g = B.build_gross(wf_1x, vs1, pr, "override", 0.95, 0.50)  # base 1.49x * derisk
        wf_credit, _ = B.apply_gross(wf_1x, g)

        print(f"\nOOS window: {oos_idx[0].date()} -> {oos_idx[-1].date()} ({len(oos_years)} held-out years)", flush=True)
        hdr = f"{'variant':<44}{'CAGR':>8}{'Vol':>7}{'Sharpe':>8}{'MaxDD':>8}"
        print(hdr); print("-" * len(hdr), flush=True)
        c, v, s, d = lev(live_1x)
        print(f"{'LIVE config @1.49x (IN-SAMPLE ref, OOS window)':<44}{c:>+8.1%}{v:>7.1%}{s:>8.2f}{d:>+8.1%}", flush=True)
        c, v, s, d = lev(wf_1x)
        print(f"{'WALK-FORWARD selected @1.49x (TRUE OOS)':<44}{c:>+8.1%}{v:>7.1%}{s:>8.2f}{d:>+8.1%}", flush=True)
        c2, v2, s2, d2 = stats(wf_credit)
        print(f"{'WALK-FORWARD + credit p95 gate (OOS)':<44}{c2:>+8.1%}{v2:>7.1%}{s2:>8.2f}{d2:>+8.1%}", flush=True)
        c, v, s, d = lev(oracle_1x)
        print(f"{'ORACLE best config @1.49x (ex-post upper bnd)':<44}{c:>+8.1%}{v:>7.1%}{s:>8.2f}{d:>+8.1%}", flush=True)

        # which configs the walk-forward picked
        from collections import Counter
        pc = Counter(picks.values())
        print("\n  WF picks (config=(top_n,rebal,stop): #years):",
              {str(k): n for k, n in pc.most_common()}, flush=True)
        print(f"  LIVE config picked in {pc.get(LIVE,0)}/{len(oos_years)} OOS years; oracle={oracle_k}", flush=True)
        del bt
    print("\nHONEST READ: WF@1.49x CAGR = the OOS number to expect; gap vs LIVE-in-sample = the haircut.", flush=True)
    print("Credit-gate row shows whether the DD protection survives OOS on a re-selected book.", flush=True)


if __name__ == "__main__":
    main()
