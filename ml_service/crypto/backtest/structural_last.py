"""
Two more structural, retail-accessible signals (not latency, not short-vol):

  A. DILUTION — coins whose circulating supply inflates faster should underperform (emission/unlock
     sell pressure). Cross-sectional short high-supply-growth / long low, on the 15 CM-covered coins.
  B. AGGREGATE FUNDING-STRESS TIMING — when market-wide perp funding is extremely high, the market
     is over-levered long and prone to a flush → de-risk BTC. When deeply negative (capitulation) →
     add. A risk-timing overlay on BTC (different from the cross-sectional carry/positioning, both dead).

Run:  python crypto/backtest/structural_last.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
import concurrent.futures as cf

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
RF = 0.045
FEE = 0.0006
COINS = ["btc", "eth", "xrp", "doge", "ada", "link", "ltc", "dot", "bch", "uni", "etc", "xlm", "aave", "algo", "icp"]


def cm_supply(a):
    rows, token = [], None
    while True:
        p = {"assets": a, "metrics": "SplyCur,PriceUSD", "frequency": "1d", "start_time": "2020-01-01", "page_size": 10000}
        if token:
            p["next_page_token"] = token
        j = requests.get("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics", params=p, timeout=40).json()
        rows += j.get("data", [])
        token = j.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    for m in ["SplyCur", "PriceUSD"]:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df.set_index("date")[["SplyCur", "PriceUSD"]]


def stats(d, lo=None):
    x = d.loc[lo:].dropna() if lo else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


if __name__ == "__main__":
    print("=" * 86)
    print("STRUCTURAL (last) — dilution + aggregate funding-stress timing")
    print("=" * 86)

    # ---- A. DILUTION ----
    print("\n[A] DILUTION — supply growth predicts underperformance? (15 CM coins, cross-sectional)")
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        sup = dict(zip(COINS, ex.map(cm_supply, COINS)))
    price = pd.DataFrame({a.upper(): sup[a]["PriceUSD"] for a in COINS}).sort_index()
    supply = pd.DataFrame({a.upper(): sup[a]["SplyCur"] for a in COINS}).sort_index()
    ret = price.pct_change(fill_method=None)
    sgrow = supply.pct_change(30)                              # 30d supply inflation
    print(f"    {'signal / book':<28}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'Sharpe':>8}{'2023+ Sh':>10}")
    for lb, lab in [(30, "supply-growth 30d")]:
        sig = (-sgrow).shift(1)                                # short high inflation
        rk = sig.rank(axis=1, pct=True)
        w = pd.DataFrame(0.0, index=price.index, columns=price.columns)
        w[rk >= 0.7] = 1.0; w[rk <= 0.3] = -1.0
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        rebal_mask = (np.arange(len(w)) % 7 == 0)
        w.iloc[~rebal_mask] = np.nan
        w = w.ffill().fillna(0.0)
        turn = (w - w.shift(1)).abs().sum(axis=1).fillna(0)
        d = (w * ret).sum(axis=1) - turn * FEE
        cg, sh, md = stats(d); cg23, sh23, _ = stats(d, "2023")
        print(f"    {lab+' L/S':<28}{cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>8.2f}{sh23:>10.2f}", flush=True)
        # IC
        fwd = price.pct_change(30).shift(-30)
        ic = pd.concat([sig.stack(), fwd.stack()], axis=1).dropna()
        print("    IC(supply-growth, fwd 30d ret) = %.3f  (negative = dilution hurts, as theorized)"
              % ic.iloc[:, 0].corr(ic.iloc[:, 1], method="spearman"))

    # ---- B. AGGREGATE FUNDING-STRESS TIMING ----
    print("\n[B] AGGREGATE FUNDING-STRESS — de-risk BTC when market over-levered long:")
    F = pd.read_parquet(os.path.join(DATA, "binance_funding.parquet"))
    F.index = pd.to_datetime(F.index).normalize()
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))["BTC"]
    agg = F.mean(axis=1)                                       # market-wide avg funding
    z = ((agg - agg.rolling(90).mean()) / agg.rolling(90).std()).shift(1)
    btc = close.pct_change(fill_method=None)
    print(f"    {'strategy':<30}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'Sharpe':>8}{'2023+ Sh':>10}{'MaxDD':>8}")
    for thr in [1.0, 1.5, 2.0]:
        expo = pd.Series(1.0, index=btc.index)
        expo[z > thr] = 0.0                                    # over-levered → flat
        expo[z < -thr] = 1.0                                   # capitulation → full (already 1)
        expo = expo.reindex(btc.index).ffill().fillna(1.0)
        d = expo * btc + (1 - expo) * RF / 365
        cg, sh, md = stats(d); cg23, sh23, _ = stats(d, "2023")
        print(f"    de-risk z>{thr:.1f}                  {cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>8.2f}{sh23:>10.2f}{md*100:>7.1f}%", flush=True)
    cg, sh, md = stats(btc); cg23, sh23, _ = stats(btc, "2023")
    print(f"    {'BTC buy & hold (bench)':<30}{cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>8.2f}{sh23:>10.2f}{md*100:>7.1f}%", flush=True)
    print("\n  Read: dilution real if IC clearly negative + L/S positive OOS. Funding-timing real if it")
    print("  beats BTC Sharpe OOS (lower DD without killing return).")
