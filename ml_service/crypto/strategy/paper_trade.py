"""
PAPER-TRADE the HL funding carry forward — live, zero capital, real-time out-of-sample validation.

Each run: pulls the ACTUAL realized HL funding since the last run for the held coins, accrues the
paper P&L (you'd collect funding as the short), rebalances monthly (top-funding + hysteresis), and
logs a growing track record. After a few weeks, compare the realized paper CAGR to the backtest's
~15-21% — that's the honest live signal validation, before any account or dollar is committed.

Config mirrors the validated book: top-8 funding-weighted, monthly rebalance + hysteresis, 60bp
round-trip costs, $10k notional (long spot / short HL perp, delta-neutral → P&L ≈ funding collected).

State persists in data/crypto/paper_state.json; log in paper_log.csv. Run daily:
  python crypto/strategy/paper_trade.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from crypto.data.funding_fetcher import post, funding_history, universe_by_oi

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
STATE = os.path.join(DATA, "paper_state.json")
LOG = os.path.join(DATA, "paper_log.csv")
STABLES = {"USDC", "USDT", "USDE", "DAI"}

NOTIONAL = 10000.0        # paper notional per the carry (long spot / short perp, delta-neutral)
TOP_K = 8
REBAL_DAYS = 30
KEEP_MULT = 2.0           # hysteresis: hold while still in top-(K*mult) by funding
MIN_KEEP = 0.02           # drop if trailing ann funding < 2%
RT_COST = 0.0060          # round-trip cost on rotation (spot leg dominated)


def now_ms():
    return int(time.time() * 1000)


def trailing_funding(coins, days=14):
    """annualized trailing funding per coin (for ranking) — sums realized hourly funding."""
    start = now_ms() - days * 86400 * 1000
    out = {}
    for c in coins:
        fh = funding_history(c, start, now_ms())
        if fh:
            rates = [float(x["fundingRate"]) for x in fh]
            out[c] = sum(rates) / days * 365      # ann
    return pd.Series(out)


def realized_funding(coin, start_ms, end_ms):
    """actual funding collected (per $1 short) from start to end — sum of realized hourly rates."""
    fh = funding_history(coin, start_ms, end_ms)
    return sum(float(x["fundingRate"]) for x in fh) if fh else 0.0


def select(rank_series, held):
    keep_set = set(rank_series.sort_values(ascending=False).head(int(TOP_K * KEEP_MULT)).index)
    new = [c for c in held if c in keep_set and rank_series.get(c, -1) > MIN_KEEP]
    for c in rank_series.sort_values(ascending=False).index:
        if len(new) >= TOP_K:
            break
        if c not in new and rank_series.get(c, -1) > 0:
            new.append(c)
    return new


def run():
    first = not os.path.exists(STATE)
    s = {} if first else json.load(open(STATE))
    t = now_ms()
    uni = [c for c in universe_by_oi(40) if c not in STABLES]

    if first:
        rank = trailing_funding(uni)
        held = select(rank, [])
        s = {"inception": datetime.now(timezone.utc).isoformat(), "last_update_ms": t,
             "last_rebal_ms": t, "positions": held, "cum_pnl": 0.0, "n_days": 0}
        print("PAPER-TRADE INITIALIZED %s" % s["inception"][:10])
        print("  initial positions (top-%d funding):" % TOP_K, held)
        print("  current ann funding:", {c: round(rank.get(c, 0) * 100, 0) for c in held})
    else:
        last = s["last_update_ms"]
        held = s["positions"]
        wt = 1.0 / len(held) if held else 0
        # accrue realized funding since last run (you SHORT the perp → collect positive funding)
        period_fund = sum(wt * realized_funding(c, last, t) for c in held)
        period_pnl = period_fund * NOTIONAL
        s["cum_pnl"] += period_pnl
        s["n_days"] += (t - last) / 86400 / 1000
        cost = 0.0
        # monthly rebalance + hysteresis
        if (t - s["last_rebal_ms"]) / 86400 / 1000 >= REBAL_DAYS:
            rank = trailing_funding(uni)
            new = select(rank, held)
            turn = len(set(new) ^ set(held)) / max(len(new), 1)
            cost = turn * RT_COST * NOTIONAL
            s["cum_pnl"] -= cost
            s["positions"] = new
            s["last_rebal_ms"] = t
            print("  REBALANCED:", held, "->", new)
            held = new
        s["last_update_ms"] = t
        days = (t - last) / 86400 / 1000
        print("  +%.1f days: funding collected $%.2f%s | cum P&L $%.2f" %
              (days, period_pnl, (" - cost $%.2f" % cost if cost else ""), s["cum_pnl"]))

    json.dump(s, open(STATE, "w"), indent=2)
    # report running track record
    n = max(s["n_days"], 1e-9)
    ret = s["cum_pnl"] / NOTIONAL
    ann = (1 + ret) ** (365 / n) - 1 if n >= 1 else 0
    line = pd.DataFrame([{"ts": datetime.now(timezone.utc).isoformat(), "n_days": round(s["n_days"], 1),
                          "cum_pnl": round(s["cum_pnl"], 2), "cum_ret_pct": round(ret * 100, 3),
                          "ann_pct": round(ann * 100, 1), "positions": "|".join(s["positions"])}])
    line.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    print("=" * 70)
    print("  LIVE PAPER TRACK: %.1f days | cum %.3f%% | annualized %.1f%% | positions: %s"
          % (s["n_days"], ret * 100, ann * 100, ", ".join(s["positions"])))
    print("  (compare annualized to backtest ~15-21%% once a few weeks accrue)")


if __name__ == "__main__":
    print("=" * 70)
    print("HL CARRY — LIVE PAPER TRADE  (%s UTC)" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    print("=" * 70)
    run()
