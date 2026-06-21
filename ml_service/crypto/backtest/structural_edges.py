"""
Option 3 — genuinely DIFFERENT edges with a structural reason to pay, not price-momentum.

  A. SHORT-TERM REVERSAL (1-3d). Overreaction / liquidity-provision premium: buy yesterday's
     losers, sell winners. Structurally different from momentum (which we killed). High turnover,
     so fees are the enemy → tiered slippage IN, on the liquid volume-ranked universe.
  B. FUNDING-AS-POSITIONING. High funding = crowded longs paying up = positioning that tends to
     UNWIND → predicts negative forward return. This is the OPPOSITE of the carry trade (collect
     funding) we already found decayed. Long low/negative-funding, short high-funding.
  C. REVERSAL × FUNDING combined.

Honest costs (tiered slippage), point-in-time liquid universe, funding tailwind on shorts.
Daily and 2-3d rebalance (reversal is short-horizon). 2023+ is the out-of-sample read.

Run:  python crypto/backtest/structural_edges.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from momentum_binance import load, stats, slip_bps

VOLWIN = 30


def xs(close, qv, fund, sig, rebal=1, top_uni=40, top_frac=0.25, long_short=True,
       use_funding=False, min_dvol=5e6):
    """Generic cross-sectional book. sig: higher = more long. Point-in-time top-vol universe."""
    ret = close.pct_change(fill_method=None)
    advol = qv.rolling(VOLWIN, min_periods=10).mean()
    fcols = set(fund.columns) if fund is not None else set()
    w, gross, costd, fundd = {}, [], [], []
    for i, dt in enumerate(close.index):
        turn_cost = 0.0
        if i % rebal == 0:
            v = advol.loc[dt].dropna()
            v = v[v > min_dvol].sort_values(ascending=False).head(top_uni)
            rankmap = {c: r for r, c in enumerate(v.index)}
            s = sig.loc[dt, v.index].dropna()
            if len(s) >= 8:
                k = max(int(len(s) * top_frac), 2)
                longs = s.sort_values(ascending=False).head(k).index
                nw = {c: 1.0 / len(longs) for c in longs}
                if long_short:
                    for c in s.sort_values().head(k).index:
                        nw[c] = nw.get(c, 0) - 1.0 / k
                for c in set(nw) | set(w):
                    turn_cost += abs(nw.get(c, 0) - w.get(c, 0)) * slip_bps(rankmap.get(c, 999))
                w = nw
        r = sum(wt * ret.loc[dt].get(c, 0.0) for c, wt in w.items() if pd.notna(ret.loc[dt].get(c, np.nan)))
        fr = 0.0
        if use_funding and fund is not None and dt in fund.index:
            row = fund.loc[dt]
            for c, wt in w.items():
                if wt < 0 and c in fcols and pd.notna(row.get(c, np.nan)):
                    fr += (-wt) * row[c]
        gross.append(r); fundd.append(fr); costd.append(turn_cost)
    return pd.Series(gross, index=close.index) + pd.Series(fundd, index=close.index) - pd.Series(costd, index=close.index)


if __name__ == "__main__":
    close, qv, fund = load()
    ret = close.pct_change(fill_method=None)
    fund_full = fund.reindex(close.index)
    print("=" * 92)
    print("STRUCTURAL EDGES — reversal + funding-positioning | liquid Binance perps | %s→%s"
          % (close.index.min().date(), close.index.max().date()))
    print("=" * 92)
    print(f"  {'edge':<44}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'Sharpe':>9}{'MaxDD':>8}")

    def show(label, d):
        cg, vol, sh, md = stats(d); cg23, _, sh23, _ = stats(d, "2023-01-01")
        print(f"  {label:<44}{cg*100:>10.1f}%{cg23*100:>11.1f}%{sh:>9.2f}{md*100:>7.1f}%", flush=True)

    print("  -- A. SHORT-TERM REVERSAL (long losers / short winners) --")
    for k in [1, 2, 3]:
        sig = (-ret.rolling(k).sum()).shift(1)
        for rb in ([1] if k == 1 else [k]):
            show(f"reversal {k}d, rebal{rb}, top40", xs(close, qv, fund_full, sig, rebal=rb, top_uni=40))
    show("reversal 1d, top20 (most liquid)", xs(close, qv, fund_full, (-ret).shift(1), rebal=1, top_uni=20))
    show("reversal 2d, top40, long-only", xs(close, qv, fund_full, (-ret.rolling(2).sum()).shift(1), rebal=2, top_uni=40, long_short=False))

    print("  -- B. FUNDING-AS-POSITIONING (long low-funding / short crowded high-funding) --")
    fcols = [c for c in fund.columns if c in close.columns]
    for k in [1, 3, 7]:
        fsig = (-fund_full[fcols].rolling(k, min_periods=1).mean()).shift(1).reindex(columns=close.columns)
        show(f"funding-reversal {k}d avg, rebal3", xs(close, qv, fund_full, fsig, rebal=3, top_uni=30, use_funding=False))

    print("  -- C. REVERSAL x FUNDING (avoid crowded longs among the losers) --")
    rev = (-ret.rolling(2).sum()).shift(1)
    fz = (-fund_full.rolling(3, min_periods=1).mean()).shift(1).reindex(columns=close.columns).fillna(0)
    combo = rev.rank(axis=1) + fz.rank(axis=1)                 # rank-sum of reversal + low-funding
    show("reversal2d + funding rank-combo, rebal2", xs(close, qv, fund_full, combo, rebal=2, top_uni=30))

    print("\n  Read: reversal is real if 2023+ Sharpe survives the high turnover/fees. Funding-positioning")
    print("  is real if low-funding beats high-funding out-of-sample. Next: validate on INTRADAY data,")
    print("  because at this turnover, execution (not signal) decides whether it's actually tradable.")
