"""
AUDIT the HL carry backtests to the same standard as the momentum work: no inflation, no
look-ahead, survivorship honest. This is the gate before trusting any carry number.

  A. LOOK-AHEAD — re-run with signals lagged an EXTRA day (lag 1→2→3). A real slow carry edge
     should barely move; a big drop means we were leaning on same-day info.
  B. SURVIVORSHIP — the HL universe is pulled as 'current top-N by OI', which is survivorship-
     biased (coins that FADED out of the top, or delisted, are missing — exactly the tail losses).
     Quantify: which previously-tracked coins dropped out, and verify the panel has NaN-before-listing
     (no back-fill) and NaN-after (no forward-fill).
  C. COSTS — Coinbase retail spot fees are far above 20bp; sweep realistic costs.
  D. TAIL — restate that the Sharpe/MaxDD are daily-vol illusions; the real tail is structural.

Run (after the expanded HL pull):  python crypto/backtest/carry_audit.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from improved_carry import load, carry, stats

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")


def carry_lag(F, P, top_k, lag):
    """carry() with an extra signal lag, to probe look-ahead sensitivity."""
    dprem = P.diff()
    sig = F.rolling(14).mean().shift(lag) * 365
    w = {}; daily = []
    for i, dt in enumerate(F.index):
        if i % 7 == 0:
            s = sig.loc[dt].dropna(); s = s[s > 0]
            picks = s.sort_values(ascending=False).head(top_k)
            w = {c: 1.0 / len(picks) for c in picks.index} if len(picks) else {}
        r = sum(wt * ((F.loc[dt, c] if pd.notna(F.loc[dt, c]) else 0) - (dprem.loc[dt, c] if pd.notna(dprem.loc[dt, c]) else 0)) for c, wt in w.items())
        daily.append(r)
    return pd.Series(daily, index=F.index)


if __name__ == "__main__":
    F, P = load()
    print("=" * 90)
    print("HL CARRY AUDIT | %d coins | %s→%s" % (F.shape[1], F.index.min().date(), F.index.max().date()))
    print("=" * 90)

    print("\n[A] LOOK-AHEAD — extra signal lag should barely move a slow carry edge:")
    print(f"    {'signal lag':<14}{'CAGR':>9}{'2024+ CAGR':>12}")
    for lag in [1, 2, 3]:
        d = carry_lag(F, P, 15, lag)
        cg, _, _, _, _ = stats(d); cg24, _, _, _, _ = stats(d, "2024")
        print(f"    {lag}d            {cg*100:>8.1f}%{cg24*100:>11.1f}%", flush=True)
    print("    (carry is collected while HELD on past-signal positions → low look-ahead sensitivity expected)")

    print("\n[B] SURVIVORSHIP — the panel is 'current top-OI', biased to survivors:")
    # coins we tracked earlier (top-30) that dropped out of current top-100 = faded (survivorship examples)
    # check NaN structure: no back-fill (NaN before first obs) and no forward-fill (NaN after last)
    first = F.apply(lambda s: s.first_valid_index())
    last = F.apply(lambda s: s.last_valid_index())
    end = F.index.max()
    late = first[first > F.index.min() + pd.Timedelta(days=90)]
    nan_before_ok = sum(1 for c in late.index if F.loc[:first[c] - pd.Timedelta(days=1), c].notna().sum() == 0)
    terminated = [c for c in F.columns if last[c] is not None and last[c] < end - pd.Timedelta(days=14)]
    print("    coins listing >90d after start: %d | with all-NaN before listing (no back-fill): %d"
          % (len(late), nan_before_ok))
    print("    coins terminating >14d before end (delisted/faded from HL): %d %s" % (len(terminated), terminated[:8]))
    print("    >> KNOWN BIAS: universe = current top-100 by OI. Coins that faded/delisted before today")
    print("       are absent → the funding carry here is biased UP (missing the squeeze/death tail losses).")
    print("       Honest haircut: treat the backtest CAGR as an UPPER bound; live capture will be lower.")

    print("\n[C] COSTS — Coinbase retail spot fees ≫ 20bp; sweep realistic round-trips (top_k=15, invvol):")
    print(f"    {'round-trip':<14}{'CAGR':>9}{'2024+ CAGR':>12}{'worst-mo':>10}")
    for rt in [0.0020, 0.0050, 0.0100, 0.0200]:
        d = carry(F, P, top_k=15, weighting="invvol", rt_cost=rt)
        cg, _, _, _, wm = stats(d); cg24, _, _, _, _ = stats(d, "2024")
        print(f"    {rt*1e4:.0f}bp          {cg*100:>8.1f}%{cg24*100:>11.1f}%{wm*100:>9.1f}%", flush=True)
    print("    (use Coinbase Advanced / Kraken Pro ~5-25bp, hold positions to amortize; high turnover kills it)")

    print("\n[D] TAIL — the Sharpe (~10) and MaxDD (~-1%) ARE DAILY-VOL ILLUSIONS. Real tail is structural:")
    print("    HL insolvency, intraday cross-venue liquidation, funding-flip cascade — none in daily P&L.")
    print("    Size by the tail (1-2x max), not the fake Sharpe. Backtest CAGR is an UPPER bound (survivorship).")
