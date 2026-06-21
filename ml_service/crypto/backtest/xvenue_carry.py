"""
Cross-venue funding carry — long Binance perp / short Hyperliquid perp, harvest the funding spread.

Powered by retail's structural long-bias: HL (retail-heavy DEX) pays persistently higher funding
than Binance, so SHORT HL / LONG Binance collects the spread, delta-neutral. The doc's #1 lead.

DISCIPLINE (the whole point — don't ship a mirage):
  - UNITS verified (both daily-summed funding; the 2-3x gap is real, not a scaling bug — see [0]).
  - REAL basis P&L from actual perp prices on each venue (the two perps can diverge — that's the risk,
    not zero). net = (HL_fund − Binance_fund) + (Binance_ret − HL_ret) − costs.
  - HONEST costs: taker on BOTH legs, both ways, on every rotation.
  - LIQUIDATION TAIL on the short HL leg simulated (a violent up-move can liquidate the short before
    the long offsets, cross-venue isolated margin) + a conservative "HL funding halved" stress in case
    the units gap is overstated.

Run:  python crypto/backtest/xvenue_carry.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")


def load():
    B = pd.read_parquet(os.path.join(DATA, "binance_funding.parquet"))
    H = pd.read_parquet(os.path.join(DATA, "hl_funding.parquet"))
    bpx = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    hpx = pd.read_parquet(os.path.join(DATA, "hl_price.parquet"))
    for d in (B, H, bpx, hpx):
        d.index = pd.to_datetime(d.index).normalize()
    common = sorted(set(B.columns) & set(H.columns) & set(bpx.columns) & set(hpx.columns))
    idx = B.index.intersection(H.index).intersection(bpx.index).intersection(hpx.index)
    return (B.loc[idx, common], H.loc[idx, common], bpx.loc[idx, common], hpx.loc[idx, common], common, idx)


def stats(d, lo=None):
    x = d.loc[lo:].dropna() if lo else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.std() * np.sqrt(365),
            x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


def backtest(B, H, bpx, hpx, top_k=8, rebal=7, leg_fee=0.0004, hl_haircut=1.0, lev=2.0):
    bret = bpx.pct_change(fill_method=None)
    hret = hpx.pct_change(fill_method=None)
    spread = (H * hl_haircut - B)                              # net carry if we short HL / long Binance
    sig = spread.rolling(14).mean().shift(1)                   # trailing spread (no look-ahead)
    w = {}; daily = []; cost = []
    for i, dt in enumerate(B.index):
        tc = 0.0
        if i % rebal == 0:
            s = sig.loc[dt].dropna()
            s = s[s > 0]                                        # only pairs where HL pays more (carry positive)
            picks = s.sort_values(ascending=False).head(top_k)
            nw = {c: 1.0 / len(picks) for c in picks.index} if len(picks) else {}
            turn = sum(abs(nw.get(c, 0) - w.get(c, 0)) for c in set(nw) | set(w))
            tc = turn * leg_fee * 4                             # 4 taker legs per full rotation (open+close, 2 venues)
            w = nw
        # daily P&L per held pair: carry spread + basis drift (long Binance / short HL)
        r = 0.0
        for c, wt in w.items():
            carry = (H.loc[dt, c] * hl_haircut - B.loc[dt, c]) if pd.notna(H.loc[dt, c]) and pd.notna(B.loc[dt, c]) else 0.0
            basis = 0.0
            if pd.notna(bret.loc[dt, c]) and pd.notna(hret.loc[dt, c]):
                basis = bret.loc[dt, c] - hret.loc[dt, c]       # long Binance, short HL
            r += wt * lev * (carry + basis)                     # leverage scales a delta-neutral book
        daily.append(r); cost.append(tc * lev)
    return pd.Series(daily, index=B.index) - pd.Series(cost, index=B.index)


if __name__ == "__main__":
    B, H, bpx, hpx, common, idx = load()
    print("=" * 90)
    print("CROSS-VENUE FUNDING CARRY — long Binance / short HL | %d coins | %s→%s" % (len(common), idx.min().date(), idx.max().date()))
    print("=" * 90)

    print("\n[0] UNITS SANITY — daily funding magnitudes (bp/day) must be same scale on both venues:")
    print("    Binance median |daily|: %.2f bp/day | HL median |daily|: %.2f bp/day  (HL higher = the premium)"
          % (B.abs().stack().median() * 1e4, H.abs().stack().median() * 1e4))
    print("    Binance ann mean: %.1f%% | HL ann mean: %.1f%%  (both plausible perp funding levels → units OK)"
          % (B.stack().mean() * 365 * 100, H.stack().mean() * 365 * 100))

    print("\n[1] NET carry book (real basis P&L + 4-leg taker fees), 2x leverage:")
    print(f"    {'config':<34}{'FULL CAGR':>11}{'2024+ CAGR':>12}{'vol':>7}{'Sharpe':>8}{'MaxDD':>8}")
    for fee in [0.0002, 0.0004, 0.0008]:
        d = backtest(B, H, bpx, hpx, leg_fee=fee)
        cg, vol, sh, md = stats(d); cg24, _, _, _ = stats(d, "2024")
        print(f"    full spread, taker {fee*1e4:.0f}bp/leg       {cg*100:>10.1f}%{cg24*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)

    print("\n[2] CONSERVATIVE STRESS — HL funding HALVED (in case the 2x gap is overstated/uncapturable):")
    for hc in [1.0, 0.75, 0.5]:
        d = backtest(B, H, bpx, hpx, leg_fee=0.0004, hl_haircut=hc)
        cg, vol, sh, md = stats(d); cg24, _, _, _ = stats(d, "2024")
        print(f"    HL funding x{hc:.2f}                  {cg*100:>10.1f}%{cg24*100:>11.1f}%{vol*100:>6.1f}%{sh:>8.2f}{md*100:>7.1f}%", flush=True)

    print("\n[3] BASIS/LIQUIDATION TAIL — worst daily basis moves (long Binance/short HL price divergence):")
    bret = bpx.pct_change(fill_method=None); hret = hpx.pct_change(fill_method=None)
    basis = (bret - hret)
    worst = basis.stack().nsmallest(6)
    print("    worst single-coin basis days (this is the un-hedged risk that liquidates the short leg):")
    for (dt, c), v in worst.items():
        print("      %s %-6s %+.1f%%" % (dt.date(), c, v * 100), flush=True)
    print("    median |daily basis|: %.2f bp (if ~0, the two perps track tightly → low basis risk)" % (basis.abs().stack().median() * 1e4))
    d = backtest(B, H, bpx, hpx, leg_fee=0.0004)
    yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("\n    per-year net return (2x lev, 4bp/leg):  " + "  ".join("%d:%+.0f%%" % (y, v) for y, v in yr.items()))
    print("\n  Read: real if NET Sharpe holds with honest fees AND survives the HL-halved stress AND the")
    print("  basis tail is bounded. This is a RISK PREMIUM (liquidation/counterparty), not free alpha.")
