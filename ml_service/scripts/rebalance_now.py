#!/usr/bin/env python3
"""Manual rebalance: resize ALL positions to exact v12 target weights."""
import json, os, sys, time, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
SIGNAL_URL = "http://localhost:5001/signals"
LEVERAGE = 1.50
CAP = 0.15

def get(path):
    return requests.get(f"{BASE}/v2/{path}", headers=HEADERS, timeout=10).json()

def post_order(symbol, notional=None, qty=None, side="buy"):
    data = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day"}
    if notional: data["notional"] = round(notional, 2)
    if qty: data["qty"] = str(qty)
    r = requests.post(f"{BASE}/v2/orders", headers=HEADERS, json=data, timeout=10)
    return r.json()

# Get state
account = get("account")
equity = float(account["equity"])
positions = get("positions")
signals = requests.get(SIGNAL_URL, timeout=10).json()

buys = sorted([s for s in signals["signals"] if s["signal"] == "BUY"], key=lambda x: -x["probability"])
total_prob = sum(s["probability"] for s in buys)

# Target weights (capped)
targets = {}
for s in buys:
    w = min(s["probability"] / total_prob, CAP)
    targets[s["symbol"]] = w
total_w = sum(targets.values())
targets = {s: w / total_w for s, w in targets.items()}

target_dollars = {s: equity * LEVERAGE * w for s, w in targets.items()}
current_dollars = {p["symbol"]: abs(float(p["market_value"])) for p in positions}

print(f"Equity: ${equity:,.0f}")
print(f"Target deployment: ${equity * LEVERAGE:,.0f} (1.5x)")
print(f"Target stocks: {len(targets)}")
print(f"Current stocks: {len(positions)}")
print()

# Phase 1: SELL positions not in target
for p in positions:
    sym = p["symbol"]
    if sym not in targets:
        qty = p["qty"]
        print(f"SELL ALL {sym} ({qty} shares)")
        r = post_order(sym, qty=abs(float(qty)), side="sell")
        print(f"  → {r.get('status', r.get('message', r))}")
        time.sleep(0.5)

# Phase 2: TRIM oversized positions
for p in positions:
    sym = p["symbol"]
    if sym not in targets:
        continue
    cur = abs(float(p["market_value"]))
    tgt = target_dollars.get(sym, 0)
    if cur > tgt * 1.05:  # >5% over target
        trim = cur - tgt
        print(f"TRIM {sym}: ${cur:,.0f} → ${tgt:,.0f} (sell ${trim:,.0f})")
        r = post_order(sym, notional=trim, side="sell")
        print(f"  → {r.get('status', r.get('message', r))}")
        time.sleep(0.5)

time.sleep(3)

# Refresh positions after sells
positions = get("positions")
current_dollars = {p["symbol"]: abs(float(p["market_value"])) for p in positions}
account = get("account")
equity = float(account["equity"])

# Phase 3: BUY missing positions and top up undersized ones
for sym in sorted(targets, key=lambda s: -target_dollars[s]):
    tgt = target_dollars[sym]
    cur = current_dollars.get(sym, 0)
    diff = tgt - cur
    if diff > equity * 0.005:  # >0.5% of equity
        print(f"BUY {sym}: ${cur:,.0f} → ${tgt:,.0f} (buy ${diff:,.0f})")
        r = post_order(sym, notional=diff, side="buy")
        print(f"  → {r.get('status', r.get('message', r))}")
        time.sleep(0.5)

time.sleep(5)

# Final state
positions = get("positions")
total_val = sum(abs(float(p["market_value"])) for p in positions)
print(f"\n{'='*50}")
print(f"FINAL: {len(positions)} positions, ${total_val:,.0f} deployed")
for p in sorted(positions, key=lambda x: -abs(float(x["market_value"]))):
    mv = abs(float(p["market_value"]))
    wt = mv / total_val * 100 if total_val > 0 else 0
    tgt_w = targets.get(p["symbol"], 0) * 100
    flag = " ⚠️" if abs(wt - tgt_w) > 2 else " ✓"
    print(f"  {p['symbol']:<8} {wt:>5.1f}% (target {tgt_w:>5.1f}%){flag}")
