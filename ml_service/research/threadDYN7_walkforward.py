"""thread DYN7 — WALK-FORWARD OOS. Choose the gate config from PAST data only.

DYN6 showed a broad parameter plateau (every gate_pct 0.90-0.97 and every derisk 0.30-0.70
positive on both horizons) and dominance over the deployed config in all three sub-periods.
But every one of those parameters was chosen with full knowledge of the answer. A plateau makes
in-sample selection LESS dangerous; it does not make it out-of-sample.

This is the honest version. Walking forward one year at a time:
    * at the start of each year Y, score every candidate config on data ending Y-1 only
    * pick the best by trailing Sharpe
    * apply THAT config through year Y, then re-select
The equity curve is stitched from the applied years, so no config is ever used on data that
chose it. Selection cost -- picking a config that then underperforms -- is fully paid.

Three arms:
    WF-selected    annual re-selection from the grid, out-of-sample
    BASE fixed     the single config DYN5/DYN6 liked (.95/.50/5d), in-sample-chosen
    DEPLOYED       rebalance-only credit gate, what runs today

The comparison that matters is WF vs DEPLOYED: both are decisions you could actually have made
without seeing the future. WF vs BASE measures how much of the headline was selection luck.

Implementation note: livemirror runs a whole span at once, so each (config, year) is run from a
fixed early start and the year's DAILY RETURNS are sliced out and chained. Costs, financing and
the vol/gate state therefore carry their real history into each year rather than being reset.

Run:  python3 research/threadDYN7_walkforward.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063, "live_sizing": True}

OFF = {"lev_recheck_every": 5, "lev_band": 0.05, "gate_offcadence": True}

# Candidate grid the walk-forward may choose from. Deliberately includes weak/■off options so
# selection has a real chance to pick badly.
GRID = {}
for pct in (0.90, 0.93, 0.95, 0.97):
    for dr in (0.30, 0.50, 0.70):
        GRID[f"OR hy+baa {pct}/{dr}"] = {"gate_cols": ["hy_oas", "baa_aaa"],
                                         "gate_pct": pct, "gate_derisk": dr, **OFF}
GRID["hy only .95/.50"] = {"gate_cols": ["hy_oas"], "gate_pct": 0.95,
                           "gate_derisk": 0.5, **OFF}
GRID["no gate (vol only)"] = {**OFF}

DEPLOYED = {"credit_pct": 0.95, "credit_derisk": 0.5}
BASE_FIXED = {"gate_cols": ["hy_oas", "baa_aaa"], "gate_pct": 0.95, "gate_derisk": 0.5, **OFF}

PERIODS = [
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl", "2001-01-02", 2004, 2025),
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl", "2018-01-02", 2020, 2025),
]


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}
    bt._si_change_ranks_by_month = {}
    bt._si_months = []
    bt.uni._short_interest_rank = {}
    bt.uni._si_change_rank = {}


def metrics(dr):
    """dr = daily return series of the stitched OOS path."""
    if len(dr) < 50:
        return (0.0, 0.0, 0.0)
    eq = (1 + dr).cumprod()
    yrs = max(len(dr) / 252.0, 1e-9)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return (float(cagr), float(sh), float(dd))


def main():
    for pname, path, start, y0, y1 in PERIODS:
        print("\n" + "=" * 100, flush=True)
        print(f"{pname} | WALK-FORWARD OOS | select on data < year, apply through year", flush=True)
        print("=" * 100, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        # one full run per candidate; slice years out of the daily-return path
        curves = {}
        for name, cfg in list(GRID.items()) + [("__DEPLOYED__", DEPLOYED),
                                               ("__BASE__", BASE_FIXED)]:
            c = dict(BASE); c.update(cfg)
            m = bt.run(start, f"{y1}-12-31", c)
            curves[name] = m["daily_values"].pct_change().dropna()
        print(f"  ran {len(curves)} candidate paths ({time.time()-t0:.0f}s)", flush=True)

        wf_parts, picks = [], []
        for y in range(y0, y1 + 1):
            tr_end = pd.Timestamp(f"{y-1}-12-31")
            best, best_sh = None, -9e9
            for name in GRID:
                tr = curves[name][curves[name].index <= tr_end]
                if len(tr) < 252:
                    continue
                sh = tr.mean() / tr.std() * np.sqrt(252) if tr.std() > 0 else -9e9
                if sh > best_sh:
                    best_sh, best = sh, name
            if best is None:
                continue
            yr = curves[best][(curves[best].index >= pd.Timestamp(f"{y}-01-01")) &
                              (curves[best].index <= pd.Timestamp(f"{y}-12-31"))]
            if len(yr):
                wf_parts.append(yr)
                picks.append((y, best))

        if not wf_parts:
            print("  no OOS years produced", flush=True)
            del bt
            continue
        wf = pd.concat(wf_parts).sort_index()
        lo, hi = wf.index.min(), wf.index.max()

        def slice_m(name):
            s = curves[name]
            return metrics(s[(s.index >= lo) & (s.index <= hi)])

        wc, ws, wd = metrics(wf)
        bc, bs, bd = slice_m("__BASE__")
        dc, ds, dd = slice_m("__DEPLOYED__")

        print(f"\n  OOS span {str(lo)[:10]} -> {str(hi)[:10]}  ({len(picks)} re-selections)",
              flush=True)
        h = f"  {'arm':<26}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
        print(h); print("  " + "-" * (len(h) - 2), flush=True)
        print(f"  {'WF-selected (OOS)':<26}{wc:>+9.2%}{ws:>8.2f}{wd:>+9.2%}", flush=True)
        print(f"  {'BASE fixed (in-sample)':<26}{bc:>+9.2%}{bs:>8.2f}{bd:>+9.2%}", flush=True)
        print(f"  {'DEPLOYED':<26}{dc:>+9.2%}{ds:>8.2f}{dd:>+9.2%}", flush=True)
        print(f"\n  WF vs DEPLOYED : CAGR {(wc-dc)*100:+.2f}pp | Sharpe {ws-ds:+.3f} | "
              f"MaxDD {(wd-dd)*100:+.2f}pp", flush=True)
        print(f"  WF vs BASE     : CAGR {(wc-bc)*100:+.2f}pp | Sharpe {ws-bs:+.3f} | "
              f"MaxDD {(wd-bd)*100:+.2f}pp   (how much was selection luck)", flush=True)

        from collections import Counter
        print(f"\n  picks: {Counter(p for _, p in picks).most_common()}", flush=True)
        print(f"  sequence: {[(y, p.replace('OR hy+baa ','')) for y, p in picks]}", flush=True)
        del bt

    print("\n" + "=" * 100, flush=True)
    print("WF vs DEPLOYED is the only honest comparison — both are decisions makeable in real", flush=True)
    print("time. If WF >= DEPLOYED on both horizons the effect is real; if WF collapses toward", flush=True)
    print("DEPLOYED, the headline was selection.", flush=True)


if __name__ == "__main__":
    main()
