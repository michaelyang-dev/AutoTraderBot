"""
FAIR trend-following test — the legitimate version (the equal-weight basket was confounded).

Target bot: SLOW, daily-rebalanced, liquid majors only → exactly what daily close+funding data
can honestly backtest. Design choices that matter:
  - inverse-VOL weighting (risk parity) across active names, not equal-weight (which over-weighted
    the wild alts and blew up the drawdown).
  - portfolio VOL-TARGETING to a fixed annualized vol, leverage-capped, using LAGGED realized vol.
  - long/FLAT (cash @ risk-free when a coin is below trend) — the "less risk" thesis.
  - HONEST costs: taker fee per side on turnover, AND funding DRAG (long a perp pays funding when
    it's positive — the usual state — a real ~10%/yr headwind that spot-only would avoid).
Benchmarks: buy & hold BTC, 50/50 BTC-ETH, and a vol-targeted BTC (apples-to-apples on risk).

Run:  python crypto/backtest/trend_v2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
MAJORS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC",
          "DOT", "BCH", "TRX", "MATIC", "ATOM", "UNI", "ETC", "XLM", "FIL", "NEAR"]
RF = 0.045
TAKER = 0.0005          # 5 bps/side taker on a liquid major perp
VOLWIN = 30
L_MAX = 3.0
MIN_DVOL = 5e6          # only trade a coin on days it clears $5M/day volume (liquid)


def load():
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    qv = pd.read_parquet(os.path.join(DATA, "binance_qvol.parquet"))
    cols = [c for c in MAJORS if c in close.columns]
    close, qv = close[cols].sort_index(), qv[cols].sort_index()
    fund = pd.read_parquet(os.path.join(DATA, "binance_funding.parquet"))
    fund.index = pd.to_datetime(fund.index).normalize()
    fund = fund.reindex(close.index)[[c for c in cols if c in fund.columns]]
    return close, qv, fund


def stats(d, lo=None):
    x = d.loc[lo:] if lo else d
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


def sig_tsmom(close, n):
    return (close.pct_change(n, fill_method=None) > 0).astype(float)


def sig_sma(close, n):
    return (close > close.rolling(n, min_periods=n // 2).mean()).astype(float)


def sig_donchian(close, n):
    hi = close.rolling(n, min_periods=n // 2).max()
    return (close >= hi * 0.999).astype(float)            # at/near n-day closing high → long


def backtest(close, qv, fund, sigfn, n, vol_target=0.40, long_short=False, pay_funding=True):
    ret = close.pct_change(fill_method=None)
    liquid = qv.rolling(VOLWIN, min_periods=5).mean() > MIN_DVOL
    pos = sigfn(close, n)
    if long_short:
        pos = pos * 2 - 1
    pos = (pos * liquid.astype(float)).shift(1).fillna(0.0)     # only trade liquid names; trade on yesterday's signal
    cvol = ret.rolling(VOLWIN, min_periods=10).std()
    raw = pos / cvol.replace(0, np.nan)                        # inverse-vol (risk parity)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = raw.abs().sum(axis=1).replace(0, np.nan)
    w = raw.div(gross, axis=0).fillna(0.0)                    # gross-1 risk-parity weights
    base = (w * ret).sum(axis=1)
    rv = base.rolling(VOLWIN, min_periods=10).std().shift(1) * np.sqrt(365)
    scale = (vol_target / rv).clip(0, L_MAX).fillna(0.0)
    w = w.mul(scale, axis=0)
    turn = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    port = (w * ret).sum(axis=1)
    cash_wt = (1 - w.abs().sum(axis=1)).clip(lower=0)          # uninvested → risk-free
    fin = (w.abs().sum(axis=1) - 1).clip(lower=0) * RF / 365   # finance leverage beyond 1x
    out = port + cash_wt * RF / 365 - turn * TAKER - fin
    if pay_funding and fund is not None:
        fpnl = -(w * fund.reindex(close.index).fillna(0.0)).sum(axis=1)  # long perp pays +funding; short receives
        out = out + fpnl
    return out


def bh(close, weights=None, vol_target=None):
    ret = close.pct_change(fill_method=None)
    if weights is None:
        r = ret["BTC"]
    else:
        w = pd.Series(weights); r = (ret[list(w.index)] * w).sum(axis=1) / w.sum()
    if vol_target:
        rv = r.rolling(VOLWIN, min_periods=10).std().shift(1) * np.sqrt(365)
        s = (vol_target / rv).clip(0, L_MAX).fillna(0.0)
        r = s * r + (1 - s).clip(lower=0) * RF / 365 - (s - 1).clip(lower=0) * RF / 365
    return r


if __name__ == "__main__":
    close, qv, fund = load()
    print("=" * 96)
    print("FAIR TREND-FOLLOWING — liquid majors, risk-parity, vol-targeted | %d coins | %s→%s"
          % (close.shape[1], close.index.min().date(), close.index.max().date()))
    print("=" * 96)
    print(f"  {'strategy':<40}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")

    def show(label, d):
        cg, vol, sh, md = stats(d); cg23, _, sh23, _ = stats(d, "2023-01-01")
        print(f"  {label:<40}{cg*100:>10.1f}%{cg23*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)

    print("  -- benchmarks --")
    show("BTC buy & hold", bh(close))
    show("50/50 BTC-ETH buy & hold", bh(close, {"BTC": 1, "ETH": 1}))
    show("BTC vol-targeted 40%", bh(close, vol_target=0.40))
    print("  -- trend long/flat (cash below trend), funding drag IN --")
    for nm, fn, ns in [("TSMOM", sig_tsmom, [30, 60, 90]), ("SMA", sig_sma, [50, 100, 200]), ("Donchian", sig_donchian, [30, 60, 90])]:
        for n in ns:
            show(f"{nm}{n} long/flat vt40", backtest(close, qv, fund, fn, n))
    print("  -- best signal, long/short --")
    show("SMA100 long/short vt40", backtest(close, qv, fund, sig_sma, 100, long_short=True))
    show("TSMOM90 long/short vt40", backtest(close, qv, fund, sig_tsmom, 90, long_short=True))

    print("\n  per-year — SMA100 long/flat vs BTC B&H:")
    d = backtest(close, qv, fund, sig_sma, 100); b = bh(close)
    yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    byr = b.groupby(b.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("  %-10s" % "year" + "".join("%8d" % y for y in yr.index))
    print("  %-10s" % "trend" + "".join("%7.0f%%" % v for v in yr.values))
    print("  %-10s" % "BTC" + "".join("%7.0f%%" % byr.get(y, 0) for y in yr.index))
    print("\n  WIN = beat BTC B&H on Sharpe AND drawdown. If not, disciplined BTC/ETH beta is the answer.")
