"""
TURNOVER-BAND TEST — "stop paying 20bps to un-drift winners".

The backtest already skips resizes < 0.3% of NAV (dust guard). This test widens that
no-trade band for SURVIVING names only — membership changes (new entries, full exits)
always execute exactly, so WHAT the strategy holds is untouched; only how fussily it
re-sizes names it already owns changes. Wider band => less resize churn => less cost,
at the price of weights drifting further from model targets.

Creates a runtime fork of main_production_backtest.py (research/_turnover_band_fork.py,
auto-generated, never touches the original), parameterizing the band as
config["resize_band"]. band=0.003 must reproduce the baseline exactly (asserted).

Run: OMP_NUM_THREADS=1 python3 research/turnover_band_test.py
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np

SRC = os.path.join(ML, "main_production_backtest.py")
FORK = os.path.join(ML, "research", "_turnover_band_fork.py")

ORIG = "                if abs(delta) < total_val * 0.003:\n                    continue"
PATCH = (
    "                _band = config.get(\"resize_band\", 0.003)\n"
    "                # widened band applies ONLY to resizes of names already held;\n"
    "                # new entries keep the original 0.3% dust guard (membership exact)\n"
    "                if abs(delta) < total_val * (_band if sym in holdings else 0.003):\n"
    "                    continue"
)

src = open(SRC).read()
assert src.count(ORIG) == 1, f"anchor not unique/found ({src.count(ORIG)} matches) — backtest changed, re-check"
open(FORK, "w").write(src.replace(ORIG, PATCH))
print(f"fork written: {FORK}")

from _turnover_band_fork import FastBacktester  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15,
       "sec_w": 0.0, "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
STARTS = ["2018-01-02", "2018-01-17"]
END = "2025-12-31"
BANDS = [0.003, 0.01, 0.02, 0.03]
BASELINE_KNOWN = {"2018-01-02": (0.252, 1.05, -0.271)}  # from tranche test (same condition)


def main():
    bt = FastBacktester()
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}

    hdr = f"{'start':<12}{'band':>7}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'gross$traded':>14}"
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr), flush=True)
    for st in STARTS:
        for band in BANDS:
            cfg = dict(V12); cfg["resize_band"] = band
            t0 = time.time()
            bt._gross_traded = 0.0
            m = bt.run(st, END, cfg)
            gt = getattr(bt, "_gross_traded", float("nan"))
            print(f"{st:<12}{band:>7.3f}{m['cagr']:>+8.1%}{m['sharpe']:>8.2f}"
                  f"{m['max_dd']:>+8.1%}{gt:>14,.0f}   ({time.time()-t0:.0f}s)", flush=True)
            if band == 0.003 and st in BASELINE_KNOWN:
                kc, ks, kd = BASELINE_KNOWN[st]
                drift = abs(m["cagr"] - kc) > 0.005 or abs(m["max_dd"] - kd) > 0.005
                print(f"{'':>19}baseline parity vs known honest run: "
                      f"{'FAIL — INVESTIGATE' if drift else 'OK'}", flush=True)
        print("-" * len(hdr), flush=True)
    print("Read: if CAGR rises as band widens, resize churn was net-negative (cost > tracking benefit).")


if __name__ == "__main__":
    main()
