"""thread GM2 — decompose the 8yr -1.50pp cost of the gross_margin bound (74d57cb).

The bound touches TWO independent mechanisms:

  (1) VALUE  — strategy_value filtered only on the LOWER side (g < 0.15), so a huge
      positive gm passed the filter AND scored enormously (gm carries weight 0.25).
      Effect = spurious SELECTION: names enter the book because their revenue
      denominator is broken. There is no mechanism by which that predicts returns.

  (2) LOWVOL — strategy5 feeds gm through zscore(). One absurd value drags the mean and
      inflates the std, collapsing every legitimate name toward z ~ 0. Effect = the
      quality factor is switched OFF. Removing the corruption switches it back ON, which
      genuinely CHANGES which names the sleeve likes — a real reshuffle, not a data error.

These have completely different interpretations, so attributing the -1.50pp to one or the
other is the whole question. 4 arms isolate them:

    A  neither bounded          (pre-fix / old canonical)
    B  value bounded only       (isolates mechanism 1)
    C  lowvol bounded only      (isolates mechanism 2)
    D  both bounded             (shipped)

Additivity check: (B-A) + (C-A) ~ (D-A) means the two act independently.

Dispatch note: _sane_gross_margin is called directly from inside the two sleeve functions,
so sys._getframe(1) identifies the caller reliably. Patching the module global works no
matter how the sleeves were imported (livemirror imports them by name), which is why this
is done here rather than by wrapping the sleeve functions.

Run:  python3 research/threadGM2_decompose.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import strategies.multi_strategy_engine as M  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

BASE = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
        "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
        "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
        "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
        "financing_rate": 0.063}

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]

_SHIPPED = M._sane_gross_margin
STATE = {"value": True, "lowvol": True}


def _dispatching_guard(gm_map):
    """Apply the bound only for the sleeve this arm has enabled."""
    caller = sys._getframe(1).f_code.co_name
    if caller == "strategy_value":
        return _SHIPPED(gm_map) if STATE["value"] else gm_map
    if caller == "strategy5_lowvol_quality":
        return _SHIPPED(gm_map) if STATE["lowvol"] else gm_map
    return _SHIPPED(gm_map)          # unknown caller -> safe default


M._sane_gross_margin = _dispatching_guard

ARMS = [
    ("A neither bounded  (pre-fix)", False, False),
    ("B value bounded only",         True,  False),
    ("C lowvol bounded only",        False, True),
    ("D both bounded     (shipped)", True,  True),
]


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}
    bt._si_change_ranks_by_month = {}
    bt._si_months = []
    bt.uni._short_interest_rank = {}
    bt.uni._si_change_rank = {}


def stat(v):
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return ((v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
            dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
            ((v - v.cummax()) / v.cummax()).min())


def main():
    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 100, flush=True)
        print(f"{pname} | gross_margin bound DECOMPOSITION | live-mirror 1.49x | "
              f"{len(starts)}-start", flush=True)
        print("=" * 100, flush=True)
        t0 = time.time()
        bt = LiveMirrorBacktester(universe_path=path)
        clear_deployed(bt)
        print(f"(universe loaded {time.time() - t0:.0f}s)", flush=True)

        hdr = (f"{'arm':<32}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
               f"{'dCAGR':>9}{'dSharpe':>9}{'dMaxDD':>9}")
        print(hdr)
        print("-" * len(hdr), flush=True)

        res = {}
        base = None
        for name, bv, bl in ARMS:
            STATE["value"], STATE["lowvol"] = bv, bl
            cs, ss, ds = [], [], []
            for st in starts:
                m = bt.run(st, end, dict(BASE))
                c, s, d = stat(m["daily_values"])
                cs.append(c)
                ss.append(s)
                ds.append(d)
            c, s, d = float(np.mean(cs)), float(np.mean(ss)), float(np.mean(ds))
            if base is None:
                base = (c, s, d)
                dc = dsh = dd = ""
            else:
                dc = f"{(c - base[0]) * 100:>+8.2f}p"
                dsh = f"{(s - base[1]):>+9.3f}"
                dd = f"{(d - base[2]) * 100:>+8.2f}p"
            res[name] = (c, s, d, cs)
            print(f"{name:<32}{c:>+9.2%}{s:>8.2f}{d:>+9.2%}{dc:>9}{dsh:>9}{dd:>9}", flush=True)

        print(f"\n  per-start CAGR:", flush=True)
        for name, _, _ in ARMS:
            print(f"    {name:<32} " + "  ".join(f"{x:+.2%}" for x in res[name][3]), flush=True)

        a = res[ARMS[0][0]][0]
        b, c_, d = (res[ARMS[i][0]][0] for i in (1, 2, 3))
        print(f"\n  ATTRIBUTION (vs arm A):", flush=True)
        print(f"    value-bound effect  (B-A): {(b - a) * 100:+.2f}pp", flush=True)
        print(f"    lowvol-bound effect (C-A): {(c_ - a) * 100:+.2f}pp", flush=True)
        print(f"    both                (D-A): {(d - a) * 100:+.2f}pp", flush=True)
        print(f"    additivity check    (B-A)+(C-A) = {((b - a) + (c_ - a)) * 100:+.2f}pp "
              f"vs (D-A) = {(d - a) * 100:+.2f}pp", flush=True)
        del bt

    STATE["value"] = STATE["lowvol"] = True
    print("\n" + "=" * 100, flush=True)
    print("READ: if the loss is concentrated in the VALUE arm, it is spurious SELECTION —", flush=True)
    print("      names entered the book because their revenue denominator was broken.", flush=True)
    print("      If concentrated in LOWVOL, it is a real factor reshuffle worth understanding.", flush=True)


if __name__ == "__main__":
    main()
