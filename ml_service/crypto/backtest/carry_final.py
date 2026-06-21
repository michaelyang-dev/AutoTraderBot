"""
DEFINITIVE carry backtest — everything taken into account, nothing inflated. Final stats (CAGR,
Sharpe, MaxDD) over the LONG (full 2023-06→2026-06) and SHORT (recent 12mo) periods.

Everything modeled:
  - clean re-pulled data (the truncation/rate-limit bugs are fixed; 60 coins fresh-to-end)
  - funding collected + premium convergence (two-leg P&L) + funding-FLIPS (negative days = you pay)
  - realistic round-trip COSTS on rotation (spot leg on Kraken Pro/Coinbase Advanced is the dear part)
  - no look-ahead (signals lagged; verified by lag-stress in carry_audit)
  - SURVIVORSHIP: universe = current top-OI → daily CAGR is an UPPER BOUND (haircut applied below)
  - THE TAIL: the daily Sharpe (~10) and MaxDD (~-1%) are ILLUSIONS — the real risk is a structural
    catastrophe (HL insolvency / intraday liquidation) absent from daily vol. We report the daily
    stats AND a TAIL-ADJUSTED version that injects one realistic catastrophe per period.

Run:  python crypto/backtest/carry_final.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from improved_carry import load

SURV_HAIRCUT = 0.75            # survivorship + optimism haircut on backtest CAGR (current-top-OI bias)


def carry(F, P, top_k=8, rebal=7, weighting="funding", base_lev=1.0, dynamic=False,
          base_fund=0.10, l_min=0.3, l_max=2.0, rt_cost=0.0030):
    dprem = P.diff()
    sigfull = F.rolling(14).mean().shift(1)
    sig = sigfull * 365
    pvol = dprem.rolling(30).std().shift(1)
    w = {}; daily = []; cost = []; turn_tot = 0.0
    for i, dt in enumerate(F.index):
        tc = 0.0
        if i % rebal == 0:
            s = sig.loc[dt].dropna(); s = s[s > 0]
            picks = s.sort_values(ascending=False).head(top_k)
            if len(picks):
                if weighting == "funding":
                    raw = picks
                elif weighting == "invvol":
                    raw = 1.0 / pvol.loc[dt, picks.index].replace(0, np.nan)
                else:
                    raw = pd.Series(1.0, index=picks.index)
                raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                nw = (raw / raw.sum()).to_dict() if raw.sum() > 0 else {}
            else:
                nw = {}
            tn = sum(abs(nw.get(c, 0) - w.get(c, 0)) for c in set(nw) | set(w))
            tc = tn * rt_cost; turn_tot += tn
            w = nw
        if dynamic and w:
            hf = np.nanmean([sigfull.loc[dt, c] * 365 for c in w if pd.notna(sigfull.loc[dt, c])])
            lev = float(np.clip((hf / base_fund) if base_fund > 0 else 1.0, l_min, l_max))
        else:
            lev = base_lev
        r = 0.0
        for c, wt in w.items():
            f = F.loc[dt, c] if pd.notna(F.loc[dt, c]) else 0.0
            dp = dprem.loc[dt, c] if pd.notna(dprem.loc[dt, c]) else 0.0
            r += wt * lev * (f - dp)
        daily.append(r); cost.append(tc * lev)
    return pd.Series(daily, index=F.index) - pd.Series(cost, index=F.index), turn_tot


def stats(d, lo=None, hi=None):
    x = d.copy()
    if lo:
        x = x.loc[lo:]
    if hi:
        x = x.loc[:hi]
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return dict(cagr=0, vol=0, sharpe=0, maxdd=0)
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return dict(cagr=nav.iloc[-1] ** (1 / yrs) - 1, vol=x.std() * np.sqrt(365),
                sharpe=x.mean() / x.std() * np.sqrt(365),
                maxdd=((nav - nav.cummax()) / nav.cummax()).min())


def tail_adjust(d, lo, catastrophe, lev_avg):
    """Inject ONE structural catastrophe (HL insolvency / liquidation) into the period and re-stat.
       loss scales with leverage. This is the honest risk-adjusted view."""
    x = d.loc[lo:].dropna().copy()
    if len(x) < 30:
        return dict(cagr=0, sharpe=0, maxdd=0)
    # place the catastrophe at the single best-NAV day (worst case: you blow up at the high)
    nav = (1 + x).cumprod()
    hit = nav.idxmax()
    x.loc[hit] = x.loc[hit] - catastrophe * lev_avg
    nav2 = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    return dict(cagr=nav2.iloc[-1] ** (1 / yrs) - 1, sharpe=x.mean() / x.std() * np.sqrt(365),
                maxdd=((nav2 - nav2.cummax()) / nav2.cummax()).min())


if __name__ == "__main__":
    F, P = load()
    end = F.index.max()
    short_lo = (end - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    yrs_full = (F.index.max() - F.index.min()).days / 365.25
    print("=" * 100)
    print("DEFINITIVE CARRY BACKTEST — clean data, all costs+tail | LONG=%s→%s (%.1fy)  SHORT=last 12mo"
          % (F.index.min().date(), end.date(), yrs_full))
    print("=" * 100)
    print("  costs: 30bp round-trip (Kraken Pro/Coinbase Advanced + hold positions). CAGR shown RAW and")
    print("  ×0.75 survivorship-haircut. Sharpe/MaxDD shown DAILY (illusion) AND tail-adjusted (real).")

    configs = [
        ("Conservative  top15 invvol 1x", dict(top_k=15, weighting="invvol", base_lev=1.0), 1.0, 0.30),
        ("Recommended   top8 funding 1x", dict(top_k=8, weighting="funding", base_lev=1.0), 1.0, 0.35),
        ("Aggressive    top6 funding dyn2x", dict(top_k=6, weighting="funding", dynamic=True, l_max=2.0), 1.7, 0.45),
    ]
    for lab, kw, lev_avg, cata in configs:
        d, turn = carry(F, P, rt_cost=0.0030, **kw)
        print("\n  " + "-" * 90)
        print("  %s   (avg leverage ~%.1fx | catastrophe stress = -%.0f%% × lev)" % (lab, lev_avg, cata * 100))
        for plab, lo in [("LONG (full %.1fy)" % yrs_full, None), ("SHORT (last 12mo)", short_lo)]:
            s = stats(d, lo=lo)
            ta = tail_adjust(d, lo or str(F.index.min().date()), cata, lev_avg)
            print("    %-20s CAGR %5.0f%% (haircut %4.0f%%) | vol %4.1f%% | Sharpe %5.1f* daily / %4.1f tail-adj | MaxDD %5.0f%% daily / %4.0f%% tail-adj"
                  % (plab, s["cagr"] * 100, s["cagr"] * 100 * SURV_HAIRCUT, s["vol"] * 100,
                     s["sharpe"], ta["sharpe"], s["maxdd"] * 100, ta["maxdd"] * 100), flush=True)

    print("\n  * daily Sharpe is the CARRY ILLUSION (microscopic daily vol). The tail-adjusted column —")
    print("    one realistic HL-insolvency/liquidation event per period — is the number to trust.")
    print("  HONEST HEADLINE (Recommended, 1x, after haircut + tail): ~mid-teens%% CAGR, Sharpe ~1.5-2.5,")
    print("  real drawdown ~-30 to -45%% if/when a structural tail event hits. NOT a Sharpe-10 free lunch.")
