"""
EXTENSIVE backtest of the 'armored' carry vs naive — with an actual LIQUIDATION model, not just
funding accrual. On the 6yr survivorship-complete Binance universe (funding + price incl. delisted).

The funding-only carry hides the squeeze/liquidation tail. Here we model it from the REAL price data:
each holding period, if a held coin SQUEEZES (daily up-move > a margin-buster threshold), the short
perp liquidates → realize a loss = liq_loss_frac × per-coin-notional (margin burned + reversal +
slippage), scaled by leverage. This is what actually happens when a short gets run over.

Configs compared:
  NAIVE   : top-6 funding, no filter           (concentrated, squeeze-prone)
  ARMORED : top-20 funding, QUALITY FILTER      (diversified; excludes coins with a recent >40% day
            (exclude recent big-movers)          — the squeeze-prone names — point-in-time, no look-ahead)
Each at leverage 1x / 1.5x / 2x, over the full 6yr, through 2022, and split-half. Liq threshold swept.
Run:  python crypto/backtest/armored_backtest.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
STABLES = {"USDC", "USDT", "USDE", "DAI", "FDUSD", "TUSD", "BUSD", "USDP"}
LIQ_LOSS_FRAC = 0.5        # on liquidation, lose ~half the per-coin notional (margin + reversal + slippage)


def load():
    F = pd.read_parquet(os.path.join(DATA, "binance_funding_full.parquet"))
    C = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    common = [c for c in F.columns if c in C.columns and c not in STABLES]
    idx = F.index.intersection(C.index)
    return F.loc[idx, common], C.loc[idx, common]


def carry(F, C, top_k=8, rebal=30, lev=1.0, quality=False, liq_pct=0.50, keep_mult=2.0, min_keep=0.02, rt_cost=0.0060):
    ret = C.pct_change(fill_method=None)
    sig = F.rolling(14).mean().shift(1) * 365
    recent_max = ret.rolling(60).max().shift(1)      # biggest daily move in last 60d (squeeze-prone flag), lagged
    held = []; w = {}; daily = []; nliq = 0
    for i, dt in enumerate(F.index):
        tc = 0.0
        if i % rebal == 0:
            s = sig.loc[dt].dropna()
            if quality:                              # exclude coins with a recent >40% day (point-in-time)
                rm = recent_max.loc[dt]
                s = s[[c for c in s.index if not (pd.notna(rm.get(c)) and rm.get(c) > 0.40)]]
            ranked = s.sort_values(ascending=False)
            keep_set = set(ranked.head(int(top_k * keep_mult)).index)
            new_held = [c for c in held if c in keep_set and s.get(c, -1) > min_keep]
            for c in ranked.index:
                if len(new_held) >= top_k:
                    break
                if c not in new_held and s.get(c, -1) > 0:
                    new_held.append(c)
            held = new_held
            nw = {c: 1.0 / len(held) for c in held} if held else {}
            tc = sum(abs(nw.get(c, 0) - w.get(c, 0)) for c in set(nw) | set(w)) * rt_cost
            w = nw
        # funding (delta-neutral) collected on the levered notional
        r = lev * sum(wt * (F.loc[dt, c] if pd.notna(F.loc[dt, c]) else 0.0) for c, wt in w.items())
        # LIQUIDATION: any held coin squeezing up > liq_pct this day -> short run over, lose margin
        for c in list(w.keys()):
            mv = ret.loc[dt, c] if pd.notna(ret.loc[dt, c]) else 0.0
            if mv > liq_pct:
                r -= LIQ_LOSS_FRAC * w[c] * lev      # liquidation loss on that position
                del w[c]; nliq += 1                  # position is gone (liquidated)
        daily.append(r - tc)
    return pd.Series(daily, index=F.index), nliq


def stats(d, lo=None, hi=None):
    x = d.loc[lo:hi].dropna() if (lo or hi) else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, ((nav - nav.cummax()) / nav.cummax()).min(), x.min())


if __name__ == "__main__":
    F, C = load()
    mid = F.index[len(F) // 2].strftime("%Y-%m-%d")
    print("=" * 100)
    print("ARMORED vs NAIVE CARRY — with liquidation model | %s→%s (%d coins, survivorship-complete)"
          % (F.index.min().date(), F.index.max().date(), F.shape[1]))
    print("=" * 100)
    print(f"  {'config':<34}{'lev':>5}{'CAGR':>8}{'MaxDD':>8}{'worstDay':>9}{'#liq':>6}{'2022':>8}{'1stH':>7}{'2ndH':>7}")

    def show(lab, **kw):
        d, nl = carry(F, C, **kw)
        cg, md, wd = stats(d); cg22, _, _ = stats(d, "2022-01-01", "2022-12-31")
        c1, _, _ = stats(d, None, mid); c2, _, _ = stats(d, mid, None)
        print(f"  {lab:<34}{kw.get('lev',1):>5.1f}{cg*100:>7.0f}%{md*100:>7.0f}%{wd*100:>8.0f}%{nl:>6}{cg22*100:>7.0f}%{c1*100:>6.0f}%{c2*100:>6.0f}%", flush=True)

    print("  -- NAIVE (top-6, no quality filter) --")
    for L in [1.0, 1.5, 2.0]:
        show("naive top6", top_k=6, lev=L, quality=False)
    print("  -- ARMORED (top-20, quality-filtered) --")
    for L in [1.0, 1.5, 2.0]:
        show("armored top20 +quality", top_k=20, lev=L, quality=True)
    print("\n  -- liquidation-threshold sensitivity (armored 1.5x): how robust to the squeeze assumption --")
    for lq in [0.30, 0.40, 0.50, 0.70]:
        d, nl = carry(F, C, top_k=20, lev=1.5, quality=True, liq_pct=lq)
        cg, md, wd = stats(d)
        print("    liq if +%2.0f%% day: CAGR %3.0f%%  MaxDD %3.0f%%  #liq %d" % (lq * 100, cg * 100, md * 100, nl), flush=True)
    print("\n  Read: does ARMORED at 1.5x beat/equal NAIVE at 1x on tail (MaxDD, #liq, worstDay)? If the")
    print("  diversification+quality filter shrinks the realized liquidation tail enough, leverage is earned.")
