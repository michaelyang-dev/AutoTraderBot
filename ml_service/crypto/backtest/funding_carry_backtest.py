"""
Cross-sectional funding-carry backtest (Hyperliquid).

Strategy: each rebalance, pick the coins with the highest TRAILING funding (no
look-ahead), go delta-neutral (long spot + short perp) to COLLECT funding. The
delta-neutral price legs cancel, so the realized return ≈ funding actually paid
on the held coins (incl. days funding flips negative before we rotate out) minus
trading fees on turnover. That captured-funding path is the honest return AND the
risk (the Sharpe denominator is real funding variability, not a guess).

Honest caveats baked into the read, not the math: (1) survivorship — only
currently-listed coins; (2) alt spot-leg slippage is modeled as a flat fee but is
worse for illiquid names; (3) liquidation tail not in a daily model.

Run:  python crypto/backtest/funding_carry_backtest.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
FUND = pd.read_parquet(os.path.join(DATA, "hl_funding.parquet"))   # daily funding (coin x date)

FEE = 0.0015            # one-way per position change (perp + spot ~15bps); round-trip 30bps
TRAIL = 7               # days of trailing funding for the signal
REBAL = 7               # rebalance every N days (limits turnover)
RISK_FREE = 0.045


def backtest(universe, top_k, min_ann=0.0, leverage=1.0, margin=0.08):
    f = FUND[universe].copy()
    sig = f.rolling(TRAIL).mean().shift(1) * 365     # trailing annualized funding, lagged (no look-ahead)
    held = pd.DataFrame(0.0, index=f.index, columns=f.columns)
    cur = {}
    daily, turn = [], []
    for i, dt in enumerate(f.index):
        if i % REBAL == 0:
            s = sig.loc[dt].dropna()
            picks = s[s > min_ann].sort_values(ascending=False).head(top_k)
            new = {c: 1.0 / len(picks) for c in picks.index} if len(picks) else {}
            t = sum(abs(new.get(c, 0) - cur.get(c, 0)) for c in set(new) | set(cur))
            turn.append(t)
            cur = new
        else:
            turn.append(0.0)
        # collect funding on held coins today (equal weight)
        if cur:
            r = sum(w * f.loc[dt].get(c, 0.0) for c, w in cur.items() if pd.notna(f.loc[dt].get(c, np.nan)))
        else:
            r = 0.0
        gross = leverage * r - (leverage - 1) * margin / 365     # margin on borrowed (levered)
        gross -= turn[-1] * FEE * leverage                       # turnover fee
        daily.append(gross)
    d = pd.Series(daily, index=f.index)
    nav = (1 + d).cumprod()
    yrs = (f.index[-1] - f.index[0]).days / 365.25
    cg = nav.iloc[-1] ** (1 / yrs) - 1
    vol = d.std() * np.sqrt(365)
    sh = d.mean() / d.std() * np.sqrt(365) if d.std() else 0
    md = ((nav - nav.cummax()) / nav.cummax()).min()
    return dict(cagr=cg, vol=vol, sharpe=sh, mdd=md, turn=np.mean(turn) * 52, nav=nav)


valid = [c for c in FUND.columns if FUND[c].notna().sum() > 300]
btc_eth = [c for c in ["BTC", "ETH"] if c in valid]
print("=" * 94)
print("HYPERLIQUID FUNDING-CARRY (%s → %s) | fees + funding flips in | benchmark cash %.1f%%"
      % (FUND.index.min().date(), FUND.index.max().date(), RISK_FREE * 100))
print("=" * 94)
print(f"  {'strategy':<34}{'CAGR':>8}{'excess':>9}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}{'turn/yr':>9}")


def show(label, **kw):
    r = backtest(**kw)
    print(f"  {label:<34}{r['cagr']*100:>7.1f}%{(r['cagr']-RISK_FREE)*100:>8.1f}%{r['vol']*100:>6.1f}%{r['sharpe']:>8.2f}{r['mdd']*100:>7.1f}%{r['turn']:>8.1f}x", flush=True)


show("BTC+ETH only  1x", universe=btc_eth, top_k=2, min_ann=0.0)
show("top-5 funding (all coins) 1x", universe=valid, top_k=5, min_ann=0.0)
show("top-10 funding 1x", universe=valid, top_k=10, min_ann=0.0)
show("top-10, only if >15%/yr  1x", universe=valid, top_k=10, min_ann=0.15)
print("  -- leverage frontier (top-10, >15%) --")
for L in [2, 3]:
    show(f"top-10 >15%  {L}x", universe=valid, top_k=10, min_ann=0.15, leverage=L)
print("\n  Real if: excess over cash is solid AND Sharpe survives the funding flips + fees.")
print("  Watch turnover (alt funding rotates fast) and remember survivorship flatters the top names.")
