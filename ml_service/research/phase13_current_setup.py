"""
Phase 13 — Is the CURRENT live setup safe? Replicates the exact live engine on a
$30K account: LEVERAGE=1.8, whole-share truncation, 15%-of-NAV position cap, and
vol-scaling (target ~22% NAV vol, floor 0.30). Reports effective leverage (stability),
drawdowns, and — the real safety question — margin-call proximity through the 2008 GFC.

Reg-T margin: a maintenance call hits when gross/equity > 4.0 (equity < 25% of gross).
The engine establishes at 1.8x; the risk is a crash pushing the HELD leverage toward 4x
before vol-scaling de-levers. Test whether the current setup stays clear of that.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
from strategies.multi_strategy_engine import COST_BPS
import numpy as np
import pandas as pd

LEVERAGE = 1.80; CAP = 0.15
VOL_TARGET = 0.15 * 1.49; VOL_LOOKBACK = 40; VOL_FLOOR = 0.30
COST_FRAC = COST_BPS / 10000.0
TRAIL = 0.40; MIN_TRADE = 0.01
MARGIN_CALL_LEV = 4.0   # Reg-T maintenance: equity < 25% of gross


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


bt = FastBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl"); clear(bt)
DEP = {"universe": "sp1500", "mom_w": .50, "val_w": .35, "lv_w": .15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
       "record_targets": True}
bt.run("2006-01-01", "2025-12-31", DEP)
rebal_targets = {d: w for d, w in bt._rebal_log}
prices = bt.prices
print(f"recorded {len(rebal_targets)} rebalances")


def vscale(nav_hist):
    navs = nav_hist[-(VOL_LOOKBACK + 1):]
    if len(navs) < 20:
        return 1.0
    rets = [navs[i] / navs[i - 1] - 1 for i in range(1, len(navs)) if navs[i - 1] > 0]
    if len(rets) < 19:
        return 1.0
    m = sum(rets) / len(rets); rv = (sum((r - m) ** 2 for r in rets) / len(rets)) ** 0.5 * (252 ** 0.5)
    if rv < 0.01:
        return 1.0
    return min(1.0, max(VOL_FLOOR, VOL_TARGET / rv))


def simulate(capital, mode, use_vol, start, end):
    """mode: 'live' (1.8 whole-share, 0.15-NAV cap) | 'frac15' (fractional 1.5x ideal)."""
    cash = capital; hold = {}; navs = []; nav_hist = []
    lev_rebal = []; daily_lev = []
    base_lev = LEVERAGE if mode == "live" else 1.5
    for date in [d for d in prices.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]:
        row = prices.loc[date]; today = {s: row[s] for s in row.dropna().index}
        for s in list(hold):
            p = today.get(s)
            if p:
                h = hold[s]; h["peak_px"] = max(h["peak_px"], p)
                if (p - h["peak_px"]) / h["peak_px"] < -TRAIL:
                    cash += h["shares"] * p * (1 - COST_FRAC); del hold[s]
        gross = sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in hold.items())
        nav = cash + gross
        navs.append((date, nav)); nav_hist.append(nav)
        if nav > 0 and gross > 0:
            daily_lev.append(gross / nav)
        if date not in rebal_targets:
            continue
        w = rebal_targets[date]; tot = sum(w.values())
        if tot <= 0:
            continue
        eff = base_lev * (vscale(nav_hist) if use_vol else 1.0)
        wts = {s: min(w[s] / tot * eff, CAP) for s in w}
        tval = {s: nav * wts[s] for s in w if today.get(s, 0) > 0}
        if mode == "frac15":
            tgt = {s: tval[s] / today[s] for s in tval}
        else:
            tgt = {s: math.floor(tval[s] / today[s]) for s in tval}
        for s in list(hold):
            if tgt.get(s, 0) <= 0:
                p = today.get(s, hold[s]["entry_px"]); cash += hold[s]["shares"] * p * (1 - COST_FRAC); del hold[s]
        floor_cash = nav * (1 - eff) - nav * 0.01
        for s, want in tgt.items():
            p = today.get(s)
            if not p or p <= 0 or want <= 0:
                continue
            cur = hold[s]["shares"] if s in hold else 0
            d = want - cur
            if cur > 0 and abs(d * p) < nav * MIN_TRADE:
                continue
            if d > 0 and (cash - d * p * (1 + COST_FRAC)) >= floor_cash:
                cash -= d * p * (1 + COST_FRAC)
                hold[s] = {"shares": want, "entry_px": p, "peak_px": hold[s]["peak_px"] if s in hold else p} if (s not in hold) else {**hold[s], "shares": want}
            elif d < 0 and s in hold:
                cash += (-d) * p * (1 - COST_FRAC); hold[s]["shares"] = want
                if hold[s]["shares"] < 1e-9:
                    del hold[s]
        g = sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in hold.items())
        lev_rebal.append(g / nav if nav else 0)
    return pd.Series({d: v for d, v in navs}), np.array(lev_rebal), np.array(daily_lev)


def report(v, lr, dl):
    r = v.pct_change().dropna(); yrs = (v.index[-1] - v.index[0]).days / 365.25
    cg = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
    md = ((v - v.cummax()) / v.cummax()).min()
    return (f"CAGR {cg*100:5.1f}% | Sharpe {sh:4.2f} | MaxDD {md*100:6.1f}% | "
            f"lev rebal {lr.mean():.2f}x [{lr.min():.2f}-{lr.max():.2f}] | "
            f"worst held lev {dl.max():.2f}x" + ("  ⚠️ MARGIN CALL" if dl.max() > MARGIN_CALL_LEV else "  ✅ clear of 4.0x call"))


print("\n" + "=" * 96)
print("CURRENT SETUP ($30K, 1.8 leverage, whole shares, 15%-of-NAV cap) — safety test")
print("=" * 96)
for label, a, b in [("FULL 2007-2025", "2007-01-01", "2025-12-31"),
                    ("2008 GFC crash", "2007-09-01", "2009-06-30"),
                    ("COVID 2020", "2019-06-01", "2020-12-31"),
                    ("recent 2018-2025", "2018-01-01", "2025-12-31")]:
    print(f"\n  {label}:")
    for name, mode, vol in [("current (1.8 trunc, vol-scale ON) ", "live", True),
                            ("same but vol-scale OFF          ", "live", False),
                            ("fractional 1.5x (ideal ref)     ", "frac15", True)]:
        v, lr, dl = simulate(30_000, mode, vol, a, b)
        print(f"    {name} {report(v, lr, dl)}", flush=True)
print("\n  GOOD if: held leverage stays well under 4.0x (no margin call) and vol-scaling")
print("  meaningfully cuts the crash drawdown vs vol-OFF.")
