"""
Shared futures-research library — ONE consistent methodology for every sleeve so
comparisons are honest. Built on the validated Norgate continuous panel:
  return per market = ccb.diff()/nonadj.shift(1)   (difference back-adjusted)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

PANEL = os.path.join(os.path.dirname(__file__), "..", "data", "norgate",
                     "norgate_continuous_futures.parquet")

# roll-variants + micros (dupe full-size, shorter history)
EXCL = {"FDAX9", "FESX9", "FOAT9", "LEU9", "LFT9", "YAP4", "YAP10",
        "MES", "MNQ", "MYM", "M2K", "MBT", "MET", "MHI"}

# equity-index futures (the rest = "diversifying": rates/FX/commodities/vol)
EQUITY = {"ES", "NQ", "YM", "RTY", "EMD", "FDAX", "FESX", "FCE", "FSMI", "FTDX",
          "HSI", "NKD", "NIY", "SNK", "SCN", "KOS", "SSG", "SXF", "NKD"}


def load():
    """Return (CCB, NON, markets): wide back-adjusted & non-adjusted close matrices."""
    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    ccb = df[df["symbol"].str.endswith("_CCB")].copy()
    ccb["mkt"] = ccb["symbol"].str.replace("_CCB$", "", regex=True).str.lstrip("&")
    non = df[~df["symbol"].str.endswith("_CCB")].copy()
    non["mkt"] = non["symbol"].str.lstrip("&")
    CCB = ccb.pivot_table(index="date", columns="mkt", values="Close")
    NON = non.pivot_table(index="date", columns="mkt", values="Close")
    mkts = [m for m in CCB.columns if m in NON.columns and m not in EXCL]
    return CCB[mkts].sort_index(), NON[mkts].sort_index(), mkts


def market_returns(CCB, NON):
    """Per-market daily return (gap-free), winsorized."""
    R = (CCB.diff() / NON.shift(1)).clip(-0.5, 0.5)
    return R.where(NON.shift(1) > 0)


def vol_target(signal, R, tgt_mkt=0.15, span=60, lev_cap=20):
    """Vol-targeted, 1-day-lagged positions from a signal in [-1,1]."""
    vol = R.ewm(span=span, min_periods=20).std().shift(1) * np.sqrt(252)
    pos = (signal * (tgt_mkt / vol).clip(upper=lev_cap)).shift(1)
    return pos.where(R.notna())


def backtest(pos, R, cost_bps=1.5, tgt_port=0.12, min_mkts=5):
    """Equal-risk-weight portfolio with rolling vol-scaling overlay (no lookahead)."""
    pnl = pos * R - pos.diff().abs() * (cost_bps / 1e4)
    act = pos.notna() & R.notna()
    port = pnl.where(act).mean(axis=1)[act.sum(axis=1) >= min_mkts]
    roll = port.rolling(252, min_periods=60).std().shift(1) * np.sqrt(252)
    sc = (tgt_port / roll).clip(upper=3).fillna(1.0)
    return (port * sc).dropna(), port.dropna()


def stats(r, label="", show=True):
    r = r.dropna()
    if len(r) < 60:
        return {}
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    d = dict(cagr=cagr, vol=vol, sharpe=sh, maxdd=dd, n=len(r))
    if show:
        print(f"  {label:<26} CAGR {cagr:+6.1%}  vol {vol:5.1%}  Sharpe {sh:5.2f}  "
              f"MaxDD {dd:6.1%}  ({r.index.min().date()}..{r.index.max().date()})")
    return d


CRISES = {
    "2000-02 dotcom": ("2000-03-01", "2002-10-31"),
    "2008 GFC":       ("2007-10-01", "2009-03-31"),
    "2020 COVID":     ("2020-02-19", "2020-03-23"),
    "2022 bear":      ("2022-01-01", "2022-12-31"),
}


def cum(r, a, b):
    s = r[(r.index >= a) & (r.index <= b)]
    return (1 + s).prod() - 1 if len(s) else np.nan


def crisis_table(r, eq_ret, label="sleeve"):
    print(f"\n  CRISIS  | {'equity':>8} | {label:>12}")
    for name, (a, b) in CRISES.items():
        e = cum(eq_ret, a, b)
        s = cum(r, a, b)
        es = f"{e:+7.1%}" if pd.notna(e) else "   n/a "
        print(f"  {name:<15} | {es:>8} | {s:+11.1%}")


def decade_table(r):
    for dec in [1980, 1990, 2000, 2010, 2020]:
        seg = r[(r.index.year >= dec) & (r.index.year < dec + 10)]
        if len(seg) > 60:
            stats(seg, f"  {dec}s")
