"""thread SI — settle whether the short-interest multiplier actually FIRES in canonical runs.

Open question from the 2026-08-10 audit: strategy1 contains

    ortex = uni._options.get(sym)
    if ortex and ortex.get("si_pct_float") is not None:   # LIVE branch  -> _options is {} ALWAYS
        ...
    elif hasattr(uni, "_short_interest_rank"):            # BACKTEST branch
        si_rank = uni._short_interest_rank.get(sym)
        if si_rank is not None:
            composite[sym] *= 0.80 if si_rank > 0.90 else (1.10 if si_rank < 0.10 else 1.0)

main_production_backtest loads data/wrds/compustat_short_interest.parquet (4.19M rows,
2006-2026) and sets _short_interest_rank; the live path never sets the attribute at all.
That looked like backtest-only alpha the live engine cannot reproduce.

BUT the research scripts call clear_deployed(), which zeroes _si_months and
_short_interest_rank. So canonical numbers are PROBABLY unaffected. "Probably" is not good
enough for a number we quote, and reading the code one layer is exactly how the earlier
mistakes happened — so COUNT the actual multiplier applications at runtime.

Arms:
  A  clear_deployed() called   (what every canonical research run does)
  B  clear_deployed() skipped  (raw FastBacktester/livemirror behavior)

If A applies 0 multipliers, the canonical numbers contain no SI contribution and the
divergence is documentation-only. If A applies > 0, every quoted number is contaminated.

Run:  python3 research/threadSI_runtime_counter.py
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

PATH = "data/wrds/complete_sp1500_universe.pkl"
START, END = "2018-01-02", "2021-12-31"      # 4yr slice is plenty to detect ANY firing

COUNT = {"calls": 0, "attr_present": 0, "map_nonempty": 0, "lookups": 0,
         "boost": 0, "penalty": 0, "options_nonempty": 0}

_orig = M.strategy1_momentum_reversal


def probe(date, uni, day_idx, **kw):
    COUNT["calls"] += 1
    if getattr(uni, "_options", None):
        COUNT["options_nonempty"] += 1
    if hasattr(uni, "_short_interest_rank"):
        COUNT["attr_present"] += 1
        sir = uni._short_interest_rank
        if sir:
            COUNT["map_nonempty"] += 1
            # replicate the exact bucket logic on the members the sleeve scores
            try:
                members = uni.get_sp500(date)
            except Exception:
                members = []
            for s in members:
                r = sir.get(s)
                if r is None:
                    continue
                COUNT["lookups"] += 1
                if r > 0.90:
                    COUNT["penalty"] += 1
                elif r < 0.10:
                    COUNT["boost"] += 1
    return _orig(date, uni, day_idx, **kw)


# CRITICAL: livemirror_backtest does `from strategies.multi_strategy_engine import
# strategy1_momentum_reversal` at module load, so it holds its OWN reference. Patching only
# M.strategy1_momentum_reversal leaves that reference untouched and the probe never fires --
# the first version of this script reported "0 calls" and would have been read as "SI never
# fires", when in fact nothing was measured. Patch EVERY namespace that holds the name.
# (Contrast _sane_gross_margin, which the sleeves resolve as a module global at call time and
# is therefore patchable in one place.)
import main_production_backtest as MPB  # noqa: E402
import livemirror_backtest as LM  # noqa: E402

_PATCHED = []
for _mod in (M, MPB, LM):
    if getattr(_mod, "strategy1_momentum_reversal", None) is _orig:
        _mod.strategy1_momentum_reversal = probe
        _PATCHED.append(_mod.__name__)
if not _PATCHED:
    raise SystemExit("FATAL: could not patch strategy1_momentum_reversal anywhere — "
                     "the probe would silently measure nothing.")
print(f"probe installed in: {_PATCHED}", flush=True)


def clear_deployed(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}
    bt._si_change_ranks_by_month = {}
    bt._si_months = []
    bt.uni._short_interest_rank = {}
    bt.uni._si_change_rank = {}


def run(label, do_clear):
    for k in COUNT:
        COUNT[k] = 0
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"    _si_months loaded by FastBacktester: {len(bt._si_months)}", flush=True)
    if do_clear:
        clear_deployed(bt)
        print(f"    after clear_deployed -> _si_months: {len(bt._si_months)}", flush=True)
    bt.run(START, END, dict(BASE))
    print(f"\n  === {label} ===")
    print(f"    strategy1 calls            : {COUNT['calls']}"
          f"{'   <-- ZERO = PROBE NEVER RAN, RESULT IS MEANINGLESS' if COUNT['calls'] == 0 else ''}")
    print(f"    _options non-empty (Ortex) : {COUNT['options_nonempty']}")
    print(f"    _short_interest_rank attr  : {COUNT['attr_present']} calls")
    print(f"    ...and map NON-EMPTY       : {COUNT['map_nonempty']} calls")
    print(f"    per-symbol SI lookups hit  : {COUNT['lookups']}")
    print(f"    x1.10 boosts applied       : {COUNT['boost']}")
    print(f"    x0.80 penalties applied    : {COUNT['penalty']}")
    total = COUNT["boost"] + COUNT["penalty"]
    print(f"    >>> TOTAL MULTIPLIERS APPLIED: {total}")
    del bt
    return total


def main():
    print("=" * 92, flush=True)
    print(f"SI runtime counter | {START} -> {END} | does the multiplier actually fire?", flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()
    a = run("ARM A — clear_deployed() called (canonical research path)", True)
    b = run("ARM B — clear_deployed() SKIPPED (raw backtester)", False)
    print("\n" + "=" * 92)
    if COUNT["calls"] == 0:
        print("  INVALID RUN: strategy1 was never intercepted — nothing was measured.")
        print("  Do NOT read 0 multipliers as 'SI never fires'.")
        return
    print(f"  canonical path applied {a} multipliers; raw path applied {b}")
    if a == 0 and b > 0:
        print("  VERDICT: canonical numbers contain NO short-interest contribution.")
        print("           The live/backtest gap is DOCUMENTATION-ONLY — nothing to fix in the")
        print("           numbers; the risk is a future run that forgets clear_deployed().")
    elif a > 0:
        print("  VERDICT: canonical numbers DO contain SI that live cannot reproduce —")
        print("           every quoted figure is contaminated and must be re-run.")
    else:
        print("  VERDICT: SI never fires on either path — the branch is dead everywhere.")
    print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
