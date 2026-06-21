"""
The carry on the SURVIVORSHIP-COMPLETE, 6-YEAR Binance funding universe — the honest long-data test.

Directly measures the two things we couldn't before:
  1. SURVIVORSHIP BIAS, quantified — run the funding carry on the FULL universe (incl. delisted coins,
     captured point-in-time) vs SURVIVORS-ONLY (coins still trading today). The gap = the bias.
  2. CRASH behavior over 6 years — per-year incl. 2022 (LUNA + FTX), real max drawdown.

Funding-only P&L (audit showed premium ≈ 0). Monthly rebalance + hysteresis, realistic 60bp fees.
Run (after binance_funding_full_fetcher):  python crypto/backtest/binance_carry_full.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
STABLES = {"USDC", "BUSD", "TUSD", "USDT", "DAI", "FDUSD", "USDP"}


def load():
    F = pd.read_parquet(os.path.join(DATA, "binance_funding_full.parquet"))
    F = F[[c for c in F.columns if c not in STABLES]]
    return F.sort_index()


def carry(F, top_k=8, rebal=30, rt_cost=0.0060, keep_mult=2.0, min_keep=0.02, survivors_only=False):
    if survivors_only:
        end = F.index.max()
        alive = [c for c in F.columns if F[c].last_valid_index() is not None and F[c].last_valid_index() >= end - pd.Timedelta(days=14)]
        F = F[alive]
    sig = F.rolling(14).mean().shift(1) * 365
    held = []; w = {}; daily = []; cost = []
    for i, dt in enumerate(F.index):
        tc = 0.0
        if i % rebal == 0:
            s = sig.loc[dt].dropna(); ranked = s.sort_values(ascending=False)
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
        r = sum(wt * (F.loc[dt, c] if pd.notna(F.loc[dt, c]) else 0.0) for c, wt in w.items())
        daily.append(r); cost.append(tc)
    return pd.Series(daily, index=F.index) - pd.Series(cost, index=F.index)


def stats(d, lo=None):
    x = d.loc[lo:].dropna() if lo else d.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.mean() / x.std() * np.sqrt(365), ((nav - nav.cummax()) / nav.cummax()).min())


if __name__ == "__main__":
    F = load()
    end = F.index.max()
    n_alive = sum(1 for c in F.columns if F[c].last_valid_index() >= end - pd.Timedelta(days=14))
    print("=" * 92)
    print("BINANCE FUNDING CARRY — survivorship-COMPLETE, 6yr | %s→%s | %d coins (%d alive, %d delisted)"
          % (F.index.min().date(), end.date(), F.shape[1], n_alive, F.shape[1] - n_alive))
    print("=" * 92)

    full = carry(F, survivors_only=False)
    surv = carry(F, survivors_only=True)
    print("\n[1] SURVIVORSHIP BIAS — quantified (full incl. delisted vs survivors-only):")
    print(f"    {'universe':<26}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'MaxDD':>9}")
    for lab, d in [("FULL (incl. delisted) ✓honest", full), ("survivors-only (biased)", surv)]:
        cg, sh, md = stats(d); cg23, _, _ = stats(d, "2023")
        print(f"    {lab:<26}{cg*100:>10.1f}%{cg23*100:>11.1f}%{md*100:>8.1f}%", flush=True)
    cgf, _, _ = stats(full); cgs, _, _ = stats(surv)
    print("    >> survivorship bias = %.1f pts/yr (survivors-only overstates by this much)" % ((cgs - cgf) * 100))

    print("\n[2] 6-YEAR per-year (FULL universe) — incl. the 2022 crashes:")
    yr = full.groupby(full.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("    " + "  ".join("%d:%+.0f%%" % (y, v) for y, v in yr.items()))
    for lab, lo, hi in [("LUNA May-2022", "2022-05-01", "2022-05-31"), ("FTX Nov-2022", "2022-11-01", "2022-11-30")]:
        x = full.loc[lo:hi]
        print("    %-16s %+.1f%%" % (lab, (1 + x).prod() * 100 - 100), flush=True)
    cg, sh, md = stats(full)
    print("\n    FULL 6yr: CAGR %.1f%%  MaxDD %.1f%%  | this is survivorship-complete + crash-tested." % (cg * 100, md * 100))
