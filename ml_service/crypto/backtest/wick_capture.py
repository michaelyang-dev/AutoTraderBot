"""
Liquidation-WICK capture — the realistic version. Model a market-maker placing resting BUY limits
x% below the prevailing price. During a liquidation cascade the price wicks to the intra-hour LOW,
fills your bid, then recovers — you earn (close − fill). This is the actual liquidity-provision
trade, only visible with intra-hour OHLC (close-to-close misses the wick).

Honest mechanics:
  bid = prev_close · (1 − x).  Fills this hour iff low ≤ bid (you'd have been hit).  Exit at close.
  pnl = close/bid − 1 − 2·fee.  (worst-case fill AT the limit; adverse selection is real — sometimes
  it wicks through and keeps falling.) Tested across depths x and exits, OOS, fees in.

Run (after binance_hourly_ohlc_fetcher):  python crypto/backtest/wick_capture.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
FEE = 0.0004


def load():
    o = pd.read_parquet(os.path.join(DATA, "binance_1h_o.parquet"))
    l = pd.read_parquet(os.path.join(DATA, "binance_1h_l.parquet"))
    c = pd.read_parquet(os.path.join(DATA, "binance_1h_c.parquet"))
    qv = pd.read_parquet(os.path.join(DATA, "binance_1h_qv.parquet"))
    return o.sort_index(), l.sort_index(), c.sort_index(), qv.sort_index()


def sharpe(x, periods=24 * 365):
    x = x.dropna()
    if len(x) < 100 or x.std() == 0:
        return 0.0, 0.0
    return ((1 + x).prod() ** (periods / len(x)) - 1, x.mean() / x.std() * np.sqrt(periods))


def book(prev_c, low, close, nextc, qv, x, exit_next=False):
    bid = prev_c * (1 - x)
    liquid = qv.rolling(24).mean() > 3e6
    fill = (low <= bid) & liquid & prev_c.notna()
    exitpx = nextc if exit_next else close
    pnl = (exitpx / bid - 1) - 2 * FEE
    pnl = pnl.where(fill)
    # equal-weight across the names filled each hour → hourly strategy return
    hourly = pnl.mean(axis=1).fillna(0.0)
    n = int(fill.sum().sum())
    return hourly, n, pnl


if __name__ == "__main__":
    o, l, c, qv = load()
    prev_c = c.shift(1)
    nextc = c.shift(-1)
    print("=" * 92)
    print("LIQUIDATION-WICK CAPTURE — resting dip-bids, fill on intra-hour low | %s→%s"
          % (c.index.min(), c.index.max()))
    print("=" * 92)

    print("\n[1] Expectancy per fill (exit at hour close): does a dip-bid that gets hit recover?")
    print(f"    {'bid depth':<12}{'fills':>9}{'mean pnl':>10}{'hit rate':>10}{'fills/yr':>10}")
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    for x in [0.02, 0.03, 0.05, 0.08]:
        _, n, pnl = book(prev_c, l, c, nextc, qv, x)
        flat = pnl.stack()
        print(f"    -{x*100:.0f}% bid     {n:>9}{flat.mean()*100:>9.2f}%{(flat>0).mean()*100:>9.0f}%{n/yrs:>10.0f}", flush=True)

    print("\n[2] TRADABLE BOOK — equal-weight all fills each hour, exit at close, fees in:")
    print(f"    {'config':<26}{'CAGR':>10}{'Sharpe':>9}{'2023+ Sh':>10}{'2024+ Sh':>10}")
    for x in [0.02, 0.03, 0.05]:
        h, n, _ = book(prev_c, l, c, nextc, qv, x)
        cg, sh = sharpe(h); _, sh23 = sharpe(h.loc["2023":]); _, sh24 = sharpe(h.loc["2024":])
        print(f"    -{x*100:.0f}% bid, exit close      {cg*100:>9.1f}%{sh:>9.2f}{sh23:>10.2f}{sh24:>10.2f}", flush=True)
    print("    -- exit NEXT hour close (hold the bounce longer) --")
    for x in [0.03, 0.05]:
        h, n, _ = book(prev_c, l, c, nextc, qv, x, exit_next=True)
        cg, sh = sharpe(h); _, sh23 = sharpe(h.loc["2023":]); _, sh24 = sharpe(h.loc["2024":])
        print(f"    -{x*100:.0f}% bid, exit +1h        {cg*100:>9.1f}%{sh:>9.2f}{sh23:>10.2f}{sh24:>10.2f}", flush=True)

    print("\n  Read: real liquidity-provision edge = positive expectancy per fill that SURVIVES OOS after")
    print("  fees. Adverse selection (wick-through) is the killer — watch the 2024+ Sharpe specifically.")
