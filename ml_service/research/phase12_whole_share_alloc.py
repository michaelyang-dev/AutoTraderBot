"""
Phase 12 — Whole-share allocation on the $30K live account: does a smarter allocator
beat plain truncation, which leftover-priority rule wins, and do we lower leverage?

Method (faithful, not hand-wavy):
  1. Run the REAL backtester (deployed v12 config) and record the strategy's target
     WEIGHTS at every rebalance (weights are signal-driven, capital-independent).
  2. Replay those exact targets through a daily simulator that reproduces the
     backtester's own trailing stops / costs / no-trade band.
  3. VALIDATE: fractional @1x @$100k must reproduce the backtester's daily_values
     (proves the replay is faithful) — only then trust the whole-share variants.
  4. Compare on the real $30K account: truncation@1.8 (current) vs fractional@1.5
     (ideal) vs largest-remainder rules A/B/C @1.5.

Rules for distributing the leftover (round-down) cash as whole shares:
  A  fill-underweight   — share goes to whoever it most reduces weight tracking error
  B  rank/conviction    — share goes to highest target-weight name (up to 15% cap)
  C  hybrid             — among underweight names, fund highest-conviction first
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
from main_production_backtest import FastBacktester
from strategies.multi_strategy_engine import COST_BPS
import numpy as np
import pandas as pd

try:
    from main_production_backtest import SLIPPAGE_BPS
except Exception:
    SLIPPAGE_BPS = 0
COST_FRAC = (COST_BPS + SLIPPAGE_BPS) / 10000.0
CAP_W = 0.15
TRAIL = 0.40
NO_TRADE = 0.003   # backtester skips |delta$| < nav*0.003


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
res = bt.run("2017-06-01", "2025-12-31", DEP)
bt_daily = res["daily_values"]
rebal_targets = {d: w for d, w in bt._rebal_log}
prices = bt.prices
print(f"recorded {len(rebal_targets)} rebalances | {len(prices.index)} price days")


def alloc(weights, today, rule, budget):
    """{sym: target whole (or fractional) share count} for a gross dollar budget."""
    syms = [s for s in weights if today.get(s, 0) and today[s] > 0]
    px = {s: today[s] for s in syms}
    tdol = {s: weights[s] * budget for s in syms}
    cap_dol = CAP_W * budget
    if rule == "frac":
        return {s: tdol[s] / px[s] for s in syms}
    sh = {s: math.floor(tdol[s] / px[s]) for s in syms}
    if rule == "trunc":
        return sh
    leftover = budget - sum(sh[s] * px[s] for s in syms)
    while True:
        best, best_key = None, None
        for s in syms:
            if px[s] > leftover + 1e-6:
                continue
            if (sh[s] + 1) * px[s] > cap_dol + 1e-6:
                continue
            cur = sh[s] * px[s]; gap = tdol[s] - cur
            if rule == "A":                                   # max tracking-err reduction
                key = abs(gap) - abs(tdol[s] - (cur + px[s]))
                if key <= 1e-9:
                    continue
            elif rule == "B":                                 # conviction first (allow overshoot to cap)
                key = weights[s]
            elif rule == "C":                                 # underweight, conviction-first
                if gap <= 1e-9:
                    continue
                key = weights[s]
            if best_key is None or key > best_key:
                best_key, best = key, s
        if best is None:
            break
        sh[best] += 1; leftover -= px[best]
    return sh


def simulate(capital, leverage, rule, start, end):
    cash = capital
    hold = {}                       # sym -> {shares, entry_px, peak_px}
    navs = []
    gross_at_rebal = []
    dates = [d for d in prices.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    for i, date in enumerate(dates):
        row = prices.loc[date]
        today = {s: row[s] for s in row.dropna().index}
        # trailing stops (sell whole position at -40% from peak)
        for s in list(hold):
            p = today.get(s)
            if p:
                h = hold[s]
                if p > h["peak_px"]:
                    h["peak_px"] = p
                if (p - h["peak_px"]) / h["peak_px"] < -TRAIL:
                    cash += h["shares"] * p * (1 - COST_FRAC)
                    del hold[s]
        nav = cash + sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in hold.items())
        navs.append((date, nav))
        if date not in rebal_targets:
            continue
        weights = rebal_targets[date]
        budget = nav * leverage
        tgt_sh = alloc(weights, today, rule, budget)
        # sell positions not targeted
        for s in list(hold):
            if s not in tgt_sh or tgt_sh.get(s, 0) <= 0:
                p = today.get(s, hold[s]["entry_px"])
                cash += hold[s]["shares"] * p * (1 - COST_FRAC)
                del hold[s]
        # trade to targets
        for s, want in tgt_sh.items():
            p = today.get(s)
            if not p or p <= 0 or want <= 0:
                continue
            cur = hold[s]["shares"] if s in hold else 0.0
            d_sh = want - cur
            if abs(d_sh * p) < nav * NO_TRADE:
                continue
            # allow buying on margin down to the leverage limit (cash >= nav*(1-leverage))
            margin_floor = nav * (1 - leverage) - nav * 0.01
            if d_sh > 0 and (cash - d_sh * p * (1 + COST_FRAC)) >= margin_floor:
                cash -= d_sh * p * (1 + COST_FRAC)
                if s in hold:
                    hold[s]["shares"] = want
                else:
                    hold[s] = {"shares": want, "entry_px": p, "peak_px": p}
            elif d_sh < 0 and s in hold:
                cash += (-d_sh) * p * (1 - COST_FRAC)
                hold[s]["shares"] = want
                if hold[s]["shares"] < 1e-9:
                    del hold[s]
        gross = sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in hold.items())
        gross_at_rebal.append(gross / nav if nav else 0)
    s = pd.Series({d: v for d, v in navs})
    return s, float(np.mean(gross_at_rebal)) if gross_at_rebal else 0.0


def stats(v):
    r = v.pct_change().dropna(); yrs = (v.index[-1] - v.index[0]).days / 365.25
    cg = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() else 0
    md = ((v - v.cummax()) / v.cummax()).min()
    return cg, sh, md


# ── 1) VALIDATION: fractional @1x @$100k must reproduce the backtester ──
val, _ = simulate(100_000, 1.0, "frac", "2017-06-01", "2025-12-31")
common = bt_daily.index.intersection(val.index)
corr = bt_daily.reindex(common).pct_change().corr(val.reindex(common).pct_change())
err = abs(val.reindex(common).iloc[-1] / bt_daily.reindex(common).iloc[-1] - 1)
print(f"\nVALIDATION (frac@1x vs backtester): daily-return corr {corr:.4f} | "
      f"final-value diff {err*100:.2f}%  {'✅ faithful' if corr > 0.99 and err < 0.03 else '⚠️ check'}")

# ── 2) the real comparison: $30K account ──
for label, (a, b) in [("2018-2025 (full)", ("2018-01-01", "2025-12-31")),
                      ("2022-2025 (recent, high prices)", ("2022-01-01", "2025-12-31"))]:
    print(f"\n{'='*68}\n$30K WHOLE-SHARE — {label}\n{'='*68}")
    print(f"  {'variant':<26}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'eff.lev':>9}")
    ref = None
    for name, lev, rule in [("truncate @1.8 (current)", 1.8, "trunc"),
                            ("fractional @1.5 (ideal)", 1.5, "frac"),
                            ("A fill-underweight @1.5", 1.5, "A"),
                            ("B rank/conviction @1.5", 1.5, "B"),
                            ("C hybrid @1.5", 1.5, "C")]:
        v, lev_eff = simulate(30_000, lev, rule, a, b)
        cg, shp, md = stats(v)
        if rule == "frac" and lev == 1.5:
            ref = v
        print(f"  {name:<26}{cg*100:>7.1f}%{shp:>8.2f}{md*100:>8.1f}%{lev_eff:>8.2f}x", flush=True)
print("\n  Read: does A/B/C @1.5 beat truncate@1.8, and which rule tops the rest?")
print("  eff.lev confirms the leverage answer (smart alloc @1.5 should land ~1.5x).")
