"""
VRP hardened — does the BTC short-vol edge survive REALISTIC friction and the vega tail?

The idealized variance-carry (vrp_pilot.py) hides the two things that actually kill short-vol:
  1. VEGA MARK-TO-MARKET. When IV (DVOL) spikes during a selloff, a short-vega book is marked
     down IMMEDIATELY — before realized vol shows up. We model this: daily P&L of a short 1-month
     straddle ≈ vega·(IV_entry − IV_now) path + gamma/theta (IV²/365 − r²). This adds the
     mark-to-market spike the proxy ignored, and is what triggers margin stress.
  2. FRICTION. Selling vol isn't free at the mid: option bid/ask + daily delta-hedge slippage.
     Modeled as a per-roll cost and a daily hedge cost proportional to |move|.

We roll a 30-day short straddle (delta-hedged), book daily vega-MTM + gamma-theta − costs, and
crucially apply a MARGIN/STOP: if a drawdown breaches a limit the book is cut (forced de-risk),
so the realistic tail and Sharpe show. Stressed sub-periods incl. the LUNA/FTX vol spikes.

Run:  python crypto/backtest/vrp_hardened.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")


def load():
    iv = pd.read_parquet(os.path.join(DATA, "deribit_dvol.parquet"))["BTC"]
    px = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))["BTC"]
    idx = iv.index.intersection(px.index)
    return iv.loc[idx], px.loc[idx]


def stats(x, lo=None):
    x = x.loc[lo:].dropna() if lo else x.dropna()
    if len(x) < 30 or x.std() == 0:
        return 0, 0, 0, 0
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, x.mean() / x.std() * np.sqrt(365),
            ((nav - nav.cummax()) / nav.cummax()).min(), x.min())


def run(iv, px, tenor=30, vega_frac=1.0, spread=0.02, hedge_bps=0.0010,
        target_vol=0.10, margin_stop=None):
    """Short rolling straddle, delta-hedged. Daily P&L = gamma/theta + vega-MTM − costs."""
    r = px.pct_change(fill_method=None).fillna(0.0)
    ivv = iv.values
    n = len(iv)
    entry_iv = np.full(n, np.nan)
    age = np.zeros(n)
    daily = np.zeros(n)
    cost = np.zeros(n)
    cur_entry = ivv[0]
    a = 0
    for i in range(1, n):
        # roll every `tenor` days: close (pay spread) + reopen (collect, pay spread)
        if a >= tenor:
            cost[i] += spread                       # round-trip option spread on the roll
            cur_entry = ivv[i - 1]
            a = 0
        entry_iv[i] = cur_entry
        # gamma/theta: collect implied daily variance, pay realized
        gt = (cur_entry ** 2) / 365 - r.iloc[i] ** 2
        # vega mark-to-market: short vega loses when IV rises since entry (path risk / margin trigger)
        # approx vega P&L per unit straddle ≈ -(IV_now - IV_entry) scaled by remaining-tenor weight
        rem = max(tenor - a, 1) / tenor
        vega_mtm = -vega_frac * (ivv[i] - cur_entry) * rem * 0.5
        # daily delta-hedge slippage proportional to move size
        cost[i] += hedge_bps * abs(r.iloc[i]) / 0.02
        daily[i] = gt + vega_mtm
        a += 1
    gross = pd.Series(daily, index=iv.index) - pd.Series(cost, index=iv.index)
    # vol-target the book; optional margin stop that cuts exposure after a drawdown breach
    sc = (target_vol / (gross.rolling(30).std() * np.sqrt(365)).shift(1)).clip(0, 5).fillna(0)
    book = sc * gross
    if margin_stop:
        nav = (1 + book).cumprod()
        dd = nav / nav.cummax() - 1
        # if drawdown breaches stop, force flat for 10 days (margin liquidation + cooldown)
        flat = pd.Series(False, index=book.index)
        cool = 0
        navc = 1.0; peak = 1.0
        out = book.copy()
        for i in range(len(book)):
            if cool > 0:
                out.iloc[i] = 0.0; cool -= 1; continue
            navc *= (1 + book.iloc[i]); peak = max(peak, navc)
            if navc / peak - 1 < -margin_stop:
                cool = 10; navc = peak * (1 - margin_stop)      # realize the stop loss
        book = out
    return book


if __name__ == "__main__":
    iv, px = load()
    print("=" * 94)
    print("VRP HARDENED — BTC short-vol with vega-MTM tail + friction | %s→%s" % (iv.index.min().date(), iv.index.max().date()))
    print("=" * 94)
    print(f"  {'model':<46}{'FULL CAGR':>11}{'2023+ CAGR':>12}{'2023+ Sh':>10}{'MaxDD':>8}{'worstDay':>9}")

    def show(label, **kw):
        b = run(iv, px, **kw)
        cg, sh, md, wd = stats(b); cg23, sh23, md23, _ = stats(b, "2023")
        print(f"  {label:<46}{cg*100:>10.1f}%{cg23*100:>11.1f}%{sh23:>10.2f}{md*100:>7.1f}%{wd*100:>8.1f}%", flush=True)

    show("1. idealized (no vega-MTM, no costs)", vega_frac=0.0, spread=0.0, hedge_bps=0.0)
    show("2. + vega mark-to-market spike", vega_frac=1.0, spread=0.0, hedge_bps=0.0)
    show("3. + option spread + hedge slippage", vega_frac=1.0, spread=0.02, hedge_bps=0.0010)
    show("4. + margin stop (-20% → forced flat)", vega_frac=1.0, spread=0.02, hedge_bps=0.0010, margin_stop=0.20)
    show("5. realistic, half vega exposure", vega_frac=0.5, spread=0.02, hedge_bps=0.0010, margin_stop=0.20)

    print("\n  per-year — realistic model (#4):")
    b = run(iv, px, vega_frac=1.0, spread=0.02, hedge_bps=0.0010, margin_stop=0.20)
    yr = b.groupby(b.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    print("  " + "".join("%8d" % y for y in yr.index))
    print("  " + "".join("%7.0f%%" % v for v in yr.values))
    print("\n  Verdict: if 2023+ Sharpe stays > ~1 after vega-MTM + friction + margin stop, VRP is a REAL")
    print("  edge worth real option data. If the vega tail/margin guts it, it was another idealized mirage.")
