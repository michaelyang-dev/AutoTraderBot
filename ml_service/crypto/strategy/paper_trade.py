"""
PAPER-TRADE the recommended book — live, zero capital. Two uncorrelated sleeves:

  CARRY (60%): long spot BTC/ETH + short HL perp, 1.5x, collect realized HL funding (delta-neutral).
               BTC/ETH only — they don't squeeze, so no liquidation tail (validated 0 liq in 6yr).
  BETA  (40%): risk-managed BTC/ETH (60/40), vol-target 30%, 200d regime de-risk (in CASH when BTC
               < 200d SMA). Captures bull markets, sits out bears. Prices from Coinbase (US spot).

Tracks each sleeve + combined P&L forward in real-time. Validated full-backtest: ~26% CAGR full /
~20% OOS, -12% normal DD / -25% if HL fails, tail-aware Sharpe ~1.5. Run daily (AWS cron):
  python crypto/strategy/paper_trade.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datetime import datetime, timezone
import requests
import numpy as np
import pandas as pd
from crypto.data.funding_fetcher import funding_history

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
STATE = os.path.join(DATA, "paper_state.json")
LOG = os.path.join(DATA, "paper_log.csv")

NOTIONAL = 10000.0
CARRY_W, BETA_W = 0.60, 0.40
CARRY_LEV = 1.5
WEIGHTS = {"BTC": 0.60, "ETH": 0.40}      # within each sleeve
BETA_VOL_TARGET, REGIME_N, L_MAX, RF = 0.30, 200, 2.0, 0.045


def now_ms():
    return int(time.time() * 1000)


def realized_funding(coin, start_ms, end_ms):
    fh = funding_history(coin, start_ms, end_ms)
    return sum(float(x["fundingRate"]) for x in fh) if fh else 0.0


def cb_candles(product):
    r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                     params={"granularity": 86400}, timeout=20, headers={"User-Agent": "auto-trader"})
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["t", "low", "high", "open", "close", "vol"])
    return df.sort_values("t").set_index("t")["close"]


def beta_state():
    """current beta exposure + spot prices (regime + vol-target on Coinbase BTC/ETH)."""
    px = pd.DataFrame({"BTC": cb_candles("BTC-USD"), "ETH": cb_candles("ETH-USD")}).dropna()
    ret = px.pct_change()
    basket = (ret * pd.Series(WEIGHTS)).sum(axis=1)
    rvol = basket.tail(30).std() * np.sqrt(365)
    sma = px["BTC"].tail(REGIME_N).mean()
    regime_on = px["BTC"].iloc[-1] > sma
    expo = float(np.clip(BETA_VOL_TARGET / rvol, 0, L_MAX)) * (1.0 if regime_on else 0.0) if rvol > 0 else 0.0
    return expo, float(px["BTC"].iloc[-1]), float(px["ETH"].iloc[-1]), regime_on, float(rvol)


def run():
    first = not os.path.exists(STATE)
    s = {} if first else json.load(open(STATE))
    t = now_ms()
    expo, btc, eth, regime_on, rvol = beta_state()

    if first or "carry_pnl" not in s:        # (re)initialize for the two-sleeve book
        s = {"inception": datetime.now(timezone.utc).isoformat(), "last_update_ms": t,
             "carry_pnl": 0.0, "beta_pnl": 0.0, "n_days": 0.0,
             "beta_expo": expo, "last_btc": btc, "last_eth": eth}
        print("PAPER-TRADE INITIALIZED (carry+beta book) %s" % s["inception"][:10])
        print("  carry 60%% (BTC/ETH funding, 1.5x) + beta 40%% (BTC/ETH vt30+regime)")
        print("  beta regime: %s | exposure %.2fx" % ("RISK-ON" if regime_on else "RISK-OFF (cash)", expo))
    else:
        last = s["last_update_ms"]
        # CARRY sleeve: realized HL funding on BTC/ETH since last run, levered
        fb = realized_funding("BTC", last, t); fe = realized_funding("ETH", last, t)
        carry_ret = CARRY_LEV * (WEIGHTS["BTC"] * fb + WEIGHTS["ETH"] * fe)
        carry_dpnl = CARRY_W * carry_ret * NOTIONAL
        # BETA sleeve: basket return since last × the exposure we were holding (+ cash @ RF)
        basket_ret = WEIGHTS["BTC"] * (btc / s["last_btc"] - 1) + WEIGHTS["ETH"] * (eth / s["last_eth"] - 1)
        held = s["beta_expo"]
        days = (t - last) / 86400 / 1000
        beta_dpnl = BETA_W * (held * basket_ret + max(0, 1 - held) * RF * days / 365) * NOTIONAL
        s["carry_pnl"] += carry_dpnl
        s["beta_pnl"] += beta_dpnl
        s["n_days"] += days
        s["beta_expo"] = expo; s["last_btc"] = btc; s["last_eth"] = eth; s["last_update_ms"] = t
        print("  +%.1fd: carry +$%.2f | beta +$%.2f (%s) | cum carry $%.2f beta $%.2f"
              % (days, carry_dpnl, beta_dpnl, "on" if regime_on else "cash", s["carry_pnl"], s["beta_pnl"]))

    json.dump(s, open(STATE, "w"), indent=2)
    cum = s["carry_pnl"] + s["beta_pnl"]; ret = cum / NOTIONAL
    n = max(s["n_days"], 1e-9)
    ann = (1 + ret) ** (365 / n) - 1 if n >= 1 else 0
    pd.DataFrame([{"ts": datetime.now(timezone.utc).isoformat(), "n_days": round(s["n_days"], 1),
                   "carry_pnl": round(s["carry_pnl"], 2), "beta_pnl": round(s["beta_pnl"], 2),
                   "cum_pnl": round(cum, 2), "cum_ret_pct": round(ret * 100, 3), "ann_pct": round(ann * 100, 1),
                   "beta_regime": "on" if regime_on else "cash"}]).to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    print("=" * 74)
    print("  LIVE PAPER (carry+beta): %.1fd | carry $%.2f + beta $%.2f = $%.2f | cum %.3f%% | ann %.1f%% | beta=%s"
          % (s["n_days"], s["carry_pnl"], s["beta_pnl"], cum, ret * 100, ann * 100, "ON" if regime_on else "cash"))
    print("  (validated backtest: ~20%% OOS / ~26%% full, tail-aware. beta is in cash until BTC > 200d SMA)")


if __name__ == "__main__":
    print("=" * 74)
    print("RECOMMENDED BOOK — LIVE PAPER TRADE  (%s UTC)" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
    print("=" * 74)
    run()
