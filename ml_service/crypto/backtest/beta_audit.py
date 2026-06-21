"""
DEEP, SKEPTICAL re-audit of the recommended BTC/ETH beta product. Independent re-implementation
(if it disagrees with beta_product.py, something is wrong). Stresses every inflation vector:

  1. SUB-PERIOD truth. Full-period Sharpe is inflated by the 2020-21 mania. Report each regime
     separately, especially 2023+ (out-of-sample) — product vs BTC side by side.
  2. FEE realism. 5bps is optimistic for retail spot. Sweep 5→100 bps.
  3. CASH-YIELD honesty. When de-risked to cash, does the result depend on earning 4.5% risk-free?
     Re-run with 0% cash yield.
  4. LOOK-AHEAD paranoia. Every signal is shift(lag). If the edge needs lag=1 (act instantly) and
     dies at lag=2/3, it's fragile/suspect. A real slow-regime edge should barely care.
  5. WHERE the edge comes from. % of time de-risked, and is the Sharpe edge just crash-avoidance?

Run:  python crypto/backtest/beta_audit.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
RF = 0.045


def load():
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    return close[["BTC", "ETH"]].dropna().sort_index()


def build(close, vt=0.30, rn=200, fee=0.0005, cash_yield=RF, lag=1, wbtc=0.6, rebal=5, lmax=2.0):
    """Independent re-implementation. EVERY signal uses .shift(lag) → no same-day info used."""
    ret = close.pct_change(fill_method=None)
    basket = wbtc * ret["BTC"] + (1 - wbtc) * ret["ETH"]
    rv = basket.rolling(30, min_periods=10).std().shift(lag) * np.sqrt(365)      # lagged realized vol
    expo = (vt / rv).clip(0, lmax).fillna(0.0)
    on = (close["BTC"] > close["BTC"].rolling(rn, min_periods=rn // 2).mean()).shift(lag).fillna(False)
    expo = expo * on.astype(float)
    step = pd.Series(np.arange(len(close)) % rebal == 0, index=close.index)
    expo_h = expo.where(step).ffill().fillna(0.0)                               # hold between rebalances
    wb, we = wbtc * expo_h, (1 - wbtc) * expo_h
    turn = (wb.diff().abs() + we.diff().abs()).fillna(0.0)
    port = wb * ret["BTC"] + we * ret["ETH"]
    cash = (1 - expo_h).clip(lower=0)
    fin = (expo_h - 1).clip(lower=0) * RF / 365
    return port + cash * cash_yield / 365 - turn * fee - fin, expo_h


def stats(d, lo=None):
    x = d.loc[lo:].dropna() if lo else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return (0, 0, 0, 0)
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


if __name__ == "__main__":
    close = load()
    prod, expo = build(close)
    btc = close["BTC"].pct_change(fill_method=None)

    print("=" * 94)
    print("DEEP AUDIT — recommended BTC/ETH beta product | data %s→%s" % (close.index.min().date(), close.index.max().date()))
    print("=" * 94)

    print("\n[1] SUB-PERIOD — product vs BTC buy&hold (is the edge real OUT-OF-SAMPLE, not just 2020-21 mania?)")
    print(f"    {'period':<16}{'PRODUCT cagr':>13}{'sh':>6}{'DD':>8}   |{'BTC cagr':>11}{'sh':>6}{'DD':>8}")
    for lab, lo in [("full 2020-26", None), ("2021+", "2021"), ("2022+", "2022"),
                    ("2023+ (OOS)", "2023"), ("2024+", "2024"), ("2025+", "2025")]:
        pc, pv, ps, pd_ = stats(prod, lo); bc, bv, bs, bd = stats(btc, lo)
        print(f"    {lab:<16}{pc*100:>12.0f}%{ps:>6.2f}{pd_*100:>7.0f}%   |{bc*100:>10.0f}%{bs:>6.2f}{bd*100:>7.0f}%", flush=True)

    print("\n[2] FEE realism (2023+ OOS) — 5bps is optimistic for retail spot; does it survive higher?")
    print(f"    {'fee/side':<12}{'2023+ CAGR':>11}{'Sharpe':>9}{'MaxDD':>8}")
    for fee in [0.0005, 0.0025, 0.0050, 0.0100]:
        d, _ = build(close, fee=fee); c, v, s, m = stats(d, "2023")
        print(f"    {fee*1e4:>4.0f} bps    {c*100:>10.1f}%{s:>9.2f}{m*100:>7.1f}%", flush=True)

    print("\n[3] CASH-YIELD honesty (2023+) — does it lean on earning 4.5%% risk-free while de-risked?")
    for cy, lab in [(RF, "cash @ 4.5%"), (0.0, "cash @ 0%")]:
        d, _ = build(close, cash_yield=cy); c, v, s, m = stats(d, "2023")
        print(f"    {lab:<14}2023+ CAGR {c*100:>6.1f}%  Sharpe {s:>5.2f}", flush=True)

    print("\n[4] LOOK-AHEAD paranoia (2023+) — extra lag should barely matter for a slow regime edge:")
    for lag in [1, 2, 3]:
        d, _ = build(close, lag=lag); c, v, s, m = stats(d, "2023")
        print(f"    signals lagged {lag}d:  2023+ CAGR {c*100:>6.1f}%  Sharpe {s:>5.2f}", flush=True)

    print("\n[5] WHERE the edge comes from:")
    pct_cash = (expo.loc["2023":] == 0).mean() * 100
    pct_cash_full = (expo == 0).mean() * 100
    print(f"    %% of days fully de-risked to cash:  full {pct_cash_full:.0f}%%  |  2023+ {pct_cash:.0f}%%")
    print(f"    avg exposure when in-market: {expo[expo>0].mean():.2f}x  (vol-target rarely wants >1x → spot-deployable)")
    # crash-avoidance: product vs BTC in the worst BTC months
    bm = btc.resample("M").apply(lambda x: (1+x).prod()-1); pm = prod.resample("M").apply(lambda x: (1+x).prod()-1)
    worst = bm.nsmallest(10).index
    print(f"    in BTC's 10 worst months: BTC avg {bm.loc[worst].mean()*100:+.1f}%  vs product {pm.loc[worst].mean()*100:+.1f}%")
    print("    → if the product's whole edge is just dodging crashes, that's REAL risk-mgmt, not alpha.")

    # cross-check vs the original implementation
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from beta_product import strategy as orig_strategy, load as orig_load
    od = orig_strategy(orig_load(), {"BTC": 60, "ETH": 40}, vol_target=0.30, regime=True)
    diff = (prod - od).abs().sum()
    print(f"\n[X] cross-check vs beta_product.py: sum|diff| = {diff:.4f}  (≈0 → two independent impls agree)")
