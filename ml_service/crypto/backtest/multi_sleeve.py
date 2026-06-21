"""
Improvement pass 3 — three-sleeve portfolio. Add a THIRD uncorrelated return stream (unlock short)
to carry + beta, and re-optimize for max tail-aware Sharpe. Also adds a premium-spike filter to the
carry (drop the jumpiest-premium coins = the squeeze tail).

Sleeves (all market-aware, independent tails):
  CARRY   long spot / short HL perp, cost-optimized + premium-vol filter
  BETA    risk-managed BTC/ETH (drawdown reducer)
  UNLOCK  short alts with a large (>2% float) cliff unlock landing in a RISK-OFF regime, hold ~12d

Run:  python crypto/backtest/multi_sleeve.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from improved_carry import load as carry_load
from carry_optimize import carry as carry_opt
from beta_product import strategy as beta_strat, load as beta_load

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
CDN = "https://defillama-datasets.llama.fi/emissions"
MAP = {"arbitrum": "ARB", "optimism": "OP", "aptos": "APT", "sui": "SUI", "sei": "SEI",
       "celestia": "TIA", "injective": "INJ", "worldcoin": "WLD", "jito": "JTO", "starknet": "STRK",
       "ethena": "ENA", "wormhole": "W", "dydx": "DYDX", "ondo": "ONDO", "jupiter": "JUP",
       "pyth-network": "PYTH", "eigenlayer": "EIGEN"}


def unlock_sleeve(close):
    btc = close["BTC"]; btc_ma = btc.rolling(50).mean()
    ret = close.pct_change(fill_method=None)

    def fetch(slug):
        try:
            d = requests.get(f"{CDN}/{slug}", timeout=25, headers={"User-Agent": "m"}).json()
            ev = d.get("metadata", {}).get("events", [])
            series = d.get("documentedData", {}).get("data", [])
            cum = {}
            for s in series:
                for pt in s.get("data", []):
                    cum[pt["timestamp"]] = cum.get(pt["timestamp"], 0) + (pt.get("unlocked") or 0)
            cs = pd.Series(cum).sort_index() if cum else pd.Series(dtype=float)
            out = []
            for e in ev:
                if e.get("unlockType") != "cliff":
                    continue
                tok = sum(e.get("noOfTokens", []) or [0]); ts = e["timestamp"]
                fb = cs[cs.index <= ts].iloc[-1] if len(cs[cs.index <= ts]) else np.nan
                pct = tok / fb if fb and fb > 0 else np.nan
                if pct and pct > 0.02:
                    out.append((MAP[slug], pd.to_datetime(ts, unit="s").normalize()))
            return out
        except Exception:
            return []
    with ThreadPoolExecutor(max_workers=10) as ex:
        events = [e for sub in ex.map(fetch, MAP.keys()) for e in sub]
    # build a daily short book: short the coin from -2d to +10d around the unlock if risk-off
    pos = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for coin, dt in events:
        if coin not in close.columns:
            continue
        try:
            if btc.asof(dt) >= btc_ma.asof(dt):                  # only risk-off
                continue
        except Exception:
            continue
        win = close.index[(close.index >= dt - pd.Timedelta(days=2)) & (close.index <= dt + pd.Timedelta(days=10))]
        for d in win:
            if coin in pos.columns:
                pos.loc[d, coin] = -1.0
    w = pos.div(pos.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    turn = (w - w.shift(1)).abs().sum(axis=1).fillna(0)
    return (w.shift(1) * ret).sum(axis=1).fillna(0.0) - turn * 0.0008


def metrics(x, carry_w, beta_w, unlock_w, carry_tail=0.18):
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; vol = x.std() * np.sqrt(365)
    xt = x.copy(); xt.loc[nav.idxmax()] -= carry_tail * carry_w
    nav2 = (1 + xt).cumprod()
    return cagr, vol, xt.mean() / xt.std() * np.sqrt(365), ((nav2 - nav2.cummax()) / nav2.cummax()).min()


if __name__ == "__main__":
    F, P = carry_load()
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    carry, _ = carry_opt(F, P, top_k=8, rebal=30, weighting="funding", rt_cost=0.0060, keep_mult=2.0, min_keep=0.02)
    beta = beta_strat(beta_load(), {"BTC": 60, "ETH": 40}, vol_target=0.30, regime=True)
    unlock = unlock_sleeve(close)
    idx = carry.index.intersection(beta.index).intersection(unlock.index)
    c, b, u = carry.loc[idx], beta.loc[idx], unlock.loc[idx]
    print("=" * 84)
    print("MULTI-SLEEVE PORTFOLIO | corr(carry,beta)=%.2f corr(carry,unlock)=%.2f corr(beta,unlock)=%.2f"
          % (c.corr(b), c.corr(u), b.corr(u)))
    print("=" * 84)
    print("  unlock sleeve standalone: CAGR %.0f%% Sharpe %.2f" % (((1 + u).prod() ** (365 / len(u)) - 1) * 100, u.mean() / u.std() * np.sqrt(365) if u.std() else 0))
    print(f"\n  {'carry/beta/unlock':<22}{'CAGR':>8}{'vol':>7}{'Sharpe(tail)':>14}{'MaxDD(tail)':>13}")
    best = (-9, None)
    grid = [(0.75, 0.25, 0.0), (0.7, 0.2, 0.1), (0.6, 0.25, 0.15), (0.65, 0.2, 0.15),
            (0.7, 0.15, 0.15), (0.6, 0.3, 0.1), (0.8, 0.1, 0.1), (0.5, 0.3, 0.2)]
    for cw, bw, uw in grid:
        port = cw * c + bw * b + uw * u
        cagr, vol, sht, mdt = metrics(port, cw, bw, uw)
        if sht > best[0]:
            best = (sht, (cw, bw, uw))
        print(f"  {f'{cw:.0%}/{bw:.0%}/{uw:.0%}':<22}{cagr*100:>7.0f}%{vol*100:>6.1f}%{sht:>14.2f}{mdt*100:>12.0f}%", flush=True)
    print("\n  best tail-aware Sharpe: %.2f at weights %s" % (best[0], best[1]))
