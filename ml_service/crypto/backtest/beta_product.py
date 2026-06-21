"""
The honest recommended product — risk-managed BTC/ETH beta.

Since no systematic alpha survives out-of-sample, the defensible retail strategy is disciplined
beta with the ONE robust finding applied: vol-targeting (cuts drawdown at ~equal Sharpe). We also
test a simple 200d regime filter (de-risk when BTC is below trend) to see if it further tames the
−63 to −77% crypto drawdown without giving up too much.

Configs compared (all vs buy & hold):
  - base BTC/ETH basket, no risk management
  - + vol-target to a fixed annual vol (leverage-capped)
  - + 200d regime de-risk (cut exposure when BTC < SMA200)
Honest costs: taker fee on rebalancing turnover. Spot assumption (no funding drag, no leverage
beyond the vol-target cap). 2023+ is the out-of-sample read.

Run:  python crypto/backtest/beta_product.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
RF = 0.045
TAKER = 0.0005
VOLWIN = 30
L_MAX = 2.0
REBAL = 5               # weekly-ish rebalance to limit turnover


def load(assets=("BTC", "ETH")):
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    return close[list(assets)].sort_index()


def stats(d, lo=None):
    x = d.loc[lo:] if lo else d
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


def strategy(close, base_w, vol_target=None, regime=False, regime_n=200):
    ret = close.pct_change(fill_method=None)
    bw = pd.Series(base_w, dtype=float); bw = bw / bw.sum()
    expo = pd.Series(1.0, index=close.index)                    # exposure multiplier on the basket
    base = (ret[list(bw.index)] * bw).sum(axis=1)
    if vol_target:
        rv = base.rolling(VOLWIN, min_periods=10).std().shift(1) * np.sqrt(365)
        expo = (vol_target / rv).clip(0, L_MAX).fillna(0.0)
    if regime:
        on = (close["BTC"] > close["BTC"].rolling(regime_n, min_periods=regime_n // 2).mean()).shift(1)
        expo = expo * on.astype(float).reindex(close.index).fillna(1.0)
    # rebalance only every REBAL days (hold exposure between)
    step = pd.Series(np.arange(len(close)) % REBAL == 0, index=close.index)
    expo_held = expo.where(step).ffill().fillna(0.0)
    w = pd.DataFrame({c: bw[c] * expo_held for c in bw.index})
    turn = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    port = (ret[list(bw.index)] * w).sum(axis=1)
    cash = (1 - w.sum(axis=1)).clip(lower=0)
    fin = (w.sum(axis=1) - 1).clip(lower=0) * RF / 365
    return port + cash * RF / 365 - turn * TAKER - fin


if __name__ == "__main__":
    close = load()
    print("=" * 92)
    print("RECOMMENDED PRODUCT — risk-managed BTC/ETH beta | %s→%s" % (close.index.min().date(), close.index.max().date()))
    print("=" * 92)
    print(f"  {'config':<42}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")

    def show(label, d):
        cg, vol, sh, md = stats(d); cg23, _, sh23, _ = stats(d, "2023-01-01")
        print(f"  {label:<42}{cg*100:>10.1f}%{cg23*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)

    show("BTC buy & hold (benchmark)", close["BTC"].pct_change(fill_method=None))
    show("60/40 BTC-ETH, no risk mgmt", strategy(close, {"BTC": 60, "ETH": 40}))
    for vt in [0.50, 0.40, 0.30]:
        show(f"60/40 vol-target {int(vt*100)}%", strategy(close, {"BTC": 60, "ETH": 40}, vol_target=vt))
    show("60/40 vt40 + 200d regime de-risk", strategy(close, {"BTC": 60, "ETH": 40}, vol_target=0.40, regime=True))
    show("60/40 vt30 + 200d regime de-risk", strategy(close, {"BTC": 60, "ETH": 40}, vol_target=0.30, regime=True))
    show("100% BTC vt40 + regime", strategy(close, {"BTC": 100}, vol_target=0.40, regime=True))

    print("\n  per-year — 60/40 vt40 + regime vs BTC B&H:")
    d = strategy(close, {"BTC": 60, "ETH": 40}, vol_target=0.40, regime=True); b = close["BTC"].pct_change(fill_method=None)
    yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    byr = b.groupby(b.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("  %-10s" % "year" + "".join("%8d" % y for y in yr.index))
    print("  %-10s" % "product" + "".join("%7.0f%%" % v for v in yr.values))
    print("  %-10s" % "BTC B&H" + "".join("%7.0f%%" % byr.get(y, 0) for y in yr.index))
    print("\n  Goal: materially lower drawdown than BTC's -77%% at comparable Sharpe — honest 'less risk' beta.")
