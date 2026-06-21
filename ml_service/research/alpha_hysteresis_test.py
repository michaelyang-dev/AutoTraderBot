"""
Hysteresis test — does a hold-band actually improve net CAGR/Sharpe, and how much
turnover does it cut? band=0 must reproduce the deployed baseline (behavior-neutral
check). Runs deployed condition (enhanced/SI off), 2016-2025 default universe + a
through-cycle check on the 25-yr universe for the winner.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
from main_production_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import COST_BPS

DEPLOYED = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
                top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
                bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
                trend_scale={"bear": 0.40, "caution": 0.75},
                vol_scaling=True, vol_target=0.20, vol_lookback=40)


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def run_one(bt, start, end, band):
    cfg = dict(DEPLOYED); cfg["hysteresis_band"] = band
    r = bt.run(start, end, cfg)
    nav = r["daily_values"]; yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    turn = bt._gross_traded / yrs / nav.mean()
    return r, turn


def show(label, r, turn):
    drag10 = turn * 10 / 10000
    print(f"  {label:<14} CAGR {r['cagr']*100:5.1f}%  Sharpe {r['sharpe']:.2f}  "
          f"MaxDD {r['max_dd']*100:6.1f}%  turnover {turn*100:4.0f}%  (drag@10bps {drag10*100:.2f}pp)")


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    clear(bt)
    print(f"[loaded {time.time()-t0:.0f}s]  cost model {COST_BPS+SLIPPAGE_BPS}bps/trade\n")

    print("=== 2016-2025 (default universe) — band=0 must equal deployed baseline ===")
    base = None
    for band in (0, 2, 3, 4, 5):
        r, turn = run_one(bt, "2016-01-01", "2025-12-31", band)
        if base is None:
            base = r["cagr"]
        tag = "baseline" if band == 0 else f"band={band}"
        show(tag, r, turn)
        if band > 0:
            print(f"  {'':14} vs baseline: CAGR {(r['cagr']-base)*100:+.2f}pp")
    print(f"\n[total {time.time()-t0:.0f}s]")
