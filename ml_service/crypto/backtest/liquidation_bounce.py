"""
NEW edge category #1 — liquidation/dislocation bounce (intraday, CONDITIONAL on extreme moves).

Different mechanism from the reversal I already killed: not "buy losers every hour" (competed, fails),
but "provide liquidity ONLY when forced selling craters a coin in a single hour" — a capacity-limited
dislocation that's hard to arb because you must be there in the minutes after. Structural: liquidation
cascades overshoot, then snap back as the forced flow clears. This is a liquidity-provision premium.

Test: flag an hour where coin return < −k·(trailing hourly vol). Measure the forward bounce over the
next H hours vs the unconditional move. Then a tradable book: long the dislocated names, hold H hours,
fees per round trip. Symmetric short-side for blow-off pumps too. OOS + fees decide it.

Run:  python crypto/backtest/liquidation_bounce.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
FEE = 0.0004
VOLWIN = 168            # 1 week of hours for the vol baseline


def load():
    close = pd.read_parquet(os.path.join(DATA, "binance_close_1h.parquet"))
    qv = pd.read_parquet(os.path.join(DATA, "binance_qvol_1h.parquet"))
    return close.sort_index(), qv.sort_index()


def sharpe(x, periods=24 * 365):
    x = x.dropna()
    if len(x) < 100 or x.std() == 0:
        return 0.0, 0.0
    return ((1 + x).prod() ** (periods / len(x)) - 1, x.mean() / x.std() * np.sqrt(periods))


if __name__ == "__main__":
    close, qv = load()
    ret = close.pct_change()
    vol = ret.rolling(VOLWIN, min_periods=48).std()
    z = ret / vol                                              # how many sigma was this hour's move
    liquid = qv.rolling(24).mean() > 3e6

    print("=" * 90)
    print("LIQUIDATION-BOUNCE — conditional on extreme hourly dislocations | hourly majors")
    print("=" * 90)

    print("\n[1] Forward bounce AFTER a sharp drop (return < -k·vol) — is there a systematic snap-back?")
    print(f"    {'drop threshold':<16}{'count':>8}{'fwd 1h':>9}{'fwd 3h':>9}{'fwd 6h':>9}{'fwd 12h':>9}  (mean fwd ret, bps)")
    for k in [2.0, 2.5, 3.0, 4.0]:
        sig = (z < -k) & liquid
        row = []
        for H in [1, 3, 6, 12]:
            fwd = close.shift(-H) / close - 1
            row.append(fwd[sig].stack().mean() * 1e4)
        n = int(sig.sum().sum())
        print(f"    drop < -{k:.1f} vol     {n:>8}{row[0]:>9.1f}{row[1]:>9.1f}{row[2]:>9.1f}{row[3]:>9.1f}", flush=True)
    # baseline: unconditional forward return for comparison
    print("    %-16s%8s%9.1f%9.1f%9.1f%9.1f  (unconditional baseline)" % (
        "ANY hour", int(liquid.sum().sum()),
        *[((close.shift(-H) / close - 1)[liquid].stack().mean() * 1e4) for H in [1, 3, 6, 12]]))

    print("\n[2] Symmetric — forward move AFTER a sharp PUMP (return > +k·vol), for the short side:")
    print(f"    {'pump threshold':<16}{'count':>8}{'fwd 1h':>9}{'fwd 3h':>9}{'fwd 6h':>9}{'fwd 12h':>9}")
    for k in [2.5, 3.0, 4.0]:
        sig = (z > k) & liquid
        row = [( (close.shift(-H) / close - 1)[sig].stack().mean() * 1e4) for H in [1, 3, 6, 12]]
        print(f"    pump > +{k:.1f} vol     {int(sig.sum().sum()):>8}{row[0]:>9.1f}{row[1]:>9.1f}{row[2]:>9.1f}{row[3]:>9.1f}", flush=True)

    print("\n[3] TRADABLE BOOK — long dislocated names, hold H hours, equal-weight, fees in:")
    print(f"    {'config':<30}{'CAGR':>10}{'Sharpe':>9}{'2023+ Sh':>10}{'trades/yr':>11}")
    for k in [2.5, 3.0]:
        for H in [3, 6, 12]:
            entry = ((z < -k) & liquid).astype(float)
            # hold H hours after entry: position = rolling sum of entries over last H hours (overlapping)
            pos = entry.rolling(H).sum().shift(1).clip(0, 1)     # in a long if dislocated in last H hrs
            w = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            turn = (w - w.shift(1)).abs().sum(axis=1).fillna(0)
            pnl = (w * ret).sum(axis=1) - turn * FEE
            cg, sh = sharpe(pnl); _, sh23 = sharpe(pnl.loc["2023":])
            tpy = entry.sum().sum() / ((close.index[-1] - close.index[0]).days / 365.25)
            print(f"    long drops<-{k:.1f}vol, hold {H:>2}h     {cg*100:>9.1f}%{sh:>9.2f}{sh23:>10.2f}{tpy:>11.0f}", flush=True)

    print("\n  Read: a real liquidity-provision edge = positive forward bounce that GROWS with the drop")
    print("  severity, and a book Sharpe that survives fees OOS. If the bounce ≈ baseline, no edge.")
